//! `HandBwd`: a fast fused forward with a hand-written backward.
//!
//! candle's fused kernels (`rms_norm`, `rope_i`, `softmax_last_dim`, ...) are registered without a gradient, so using
//! one in training silently yields zero gradients. The pattern here wraps such a kernel as a custom op whose *forward*
//! result was computed up front from detached inputs and is merely handed over (no recompute, no copy on Metal: the
//! buffer `Arc` is shared), and whose *backward* is written with ordinary tensor ops. Nothing from the forward besides
//! the op inputs and the output is retained.
//!
//! The same module holds gradient checkpointing for layers and for attention tiles, built on the same pattern.

use std::sync::{Arc, Mutex};

use super::Kernel;
use candle_core::backprop::GradStore;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, D, Layout, MetalStorage, Result, Shape, Storage, Tensor, Var, bail,
};

type Bwd1 = dyn Fn(&Tensor, &Tensor, &Tensor) -> Result<Option<Tensor>> + Send + Sync;
type Bwd2 = dyn Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> + Send + Sync;
type Grad3 = (Option<Tensor>, Option<Tensor>, Option<Tensor>);
type Bwd3 = dyn Fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) -> Result<Grad3> + Send + Sync;

/// Take ownership of the precomputed forward result (it may be consumed exactly once).
fn take_pre(slot: &Mutex<Option<Tensor>>, name: &str) -> Result<Tensor> {
    let t = slot
        .lock()
        .map_err(|_| candle_core::Error::Msg(format!("{name}: poisoned")))?
        .take()
        .ok_or_else(|| candle_core::Error::Msg(format!("{name}: forward already consumed")))?;
    let (_, l) = t.storage_and_layout();
    if !l.is_contiguous() || l.start_offset() != 0 {
        bail!("{name}: precomputed output must be contiguous with offset 0");
    }
    Ok(t)
}

fn cpu_storage_of(t: &Tensor) -> Result<(CpuStorage, Shape)> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        Storage::Cpu(c) => Ok((c.clone(), l.shape().clone())),
        _ => bail!("HandBwd: precomputed output is not on the cpu"),
    }
}

fn metal_storage_of(t: &Tensor) -> Result<(MetalStorage, Shape)> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        Storage::Metal(m) => Ok((m.clone(), l.shape().clone())),
        _ => bail!("HandBwd: precomputed output is not on metal"),
    }
}

/// One-input hand-backward op.
pub struct HandBwd1 {
    name: &'static str,
    pre: Mutex<Option<Tensor>>,
    bwd: Box<Bwd1>,
}

impl CustomOp1 for HandBwd1 {
    fn name(&self) -> &'static str {
        self.name
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        cpu_storage_of(&take_pre(&self.pre, self.name)?)
    }
    fn metal_fwd(&self, _s: &MetalStorage, _l: &Layout) -> Result<(MetalStorage, Shape)> {
        metal_storage_of(&take_pre(&self.pre, self.name)?)
    }
    fn bwd(&self, arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        (self.bwd)(&arg.detach(), &res.detach(), grad_res)
    }
}

/// Two-input hand-backward op.
pub struct HandBwd2 {
    name: &'static str,
    pre: Mutex<Option<Tensor>>,
    bwd: Box<Bwd2>,
}

impl CustomOp2 for HandBwd2 {
    fn name(&self) -> &'static str {
        self.name
    }
    fn cpu_fwd(&self, _s1: &CpuStorage, _l1: &Layout, _s2: &CpuStorage, _l2: &Layout) -> Result<(CpuStorage, Shape)> {
        cpu_storage_of(&take_pre(&self.pre, self.name)?)
    }
    fn metal_fwd(
        &self,
        _s1: &MetalStorage,
        _l1: &Layout,
        _s2: &MetalStorage,
        _l2: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        metal_storage_of(&take_pre(&self.pre, self.name)?)
    }
    fn bwd(
        &self,
        a1: &Tensor,
        a2: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        (self.bwd)(&a1.detach(), &a2.detach(), &res.detach(), grad_res)
    }
}

/// Three-input hand-backward op.
pub struct HandBwd3 {
    name: &'static str,
    pre: Mutex<Option<Tensor>>,
    bwd: Box<Bwd3>,
}

impl CustomOp3 for HandBwd3 {
    fn name(&self) -> &'static str {
        self.name
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cpu_storage_of(&take_pre(&self.pre, self.name)?)
    }
    fn metal_fwd(
        &self,
        _s1: &MetalStorage,
        _l1: &Layout,
        _s2: &MetalStorage,
        _l2: &Layout,
        _s3: &MetalStorage,
        _l3: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        metal_storage_of(&take_pre(&self.pre, self.name)?)
    }
    fn bwd(&self, a1: &Tensor, a2: &Tensor, a3: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Grad3> {
        (self.bwd)(&a1.detach(), &a2.detach(), &a3.detach(), &res.detach(), grad_res)
    }
}

/// Attach `bwd(arg, res, grad_res) -> d_arg` to the already computed `out = f(arg)`.
pub fn hand_bwd1(
    name: &'static str,
    arg: &Tensor,
    out: Tensor,
    bwd: impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Option<Tensor>> + Send + Sync + 'static,
) -> Result<Tensor> {
    arg.apply_op1(HandBwd1 { name, pre: Mutex::new(Some(out)), bwd: Box::new(bwd) })
}

/// Attach `bwd(a1, a2, res, grad_res) -> (d_a1, d_a2)` to the already computed `out = f(a1, a2)`.
pub fn hand_bwd2(
    name: &'static str,
    a1: &Tensor,
    a2: &Tensor,
    out: Tensor,
    bwd: impl Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> + Send + Sync + 'static,
) -> Result<Tensor> {
    a1.apply_op2(a2, HandBwd2 { name, pre: Mutex::new(Some(out)), bwd: Box::new(bwd) })
}

/// Attach `bwd(a1, a2, a3, res, grad_res) -> (d_a1, d_a2, d_a3)` to the already computed `out`.
pub fn hand_bwd3(
    name: &'static str,
    a1: &Tensor,
    a2: &Tensor,
    a3: &Tensor,
    out: Tensor,
    bwd: impl Fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) -> Result<Grad3> + Send + Sync + 'static,
) -> Result<Tensor> {
    a1.apply_op3(a2, a3, HandBwd3 { name, pre: Mutex::new(Some(out)), bwd: Box::new(bwd) })
}

/// RMSNorm with learned scale: fused forward, backward by hand.
///
/// With `r = (mean(x^2) + eps)^(-1/2)` and `gw = g * w`: `dx = r*gw - x * r^3 * mean(x*gw)`, `dw = sum_rows(g * x * r)`.
pub fn rms_norm_hand(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let out = candle_nn::ops::rms_norm(&x.detach().contiguous()?, &w.detach().contiguous()?, eps)?;
    hand_bwd2("rms_norm_hand", x, w, out, move |x, w, _y, g| {
        let d = x.dim(D::Minus1)?;
        let g = g.contiguous()?;
        let r = (x.sqr()?.mean_keepdim(D::Minus1)? + eps as f64)?.sqrt()?.recip()?;
        let gw = g.broadcast_mul(w)?;
        let dot = x.mul(&gw)?.mean_keepdim(D::Minus1)?;
        let r3 = r.mul(&r)?.mul(&r)?;
        let dx = (gw.broadcast_mul(&r)? - x.broadcast_mul(&r3.mul(&dot)?)?)?;
        let dw = g.mul(&x.broadcast_mul(&r)?)?.reshape(((), d))?.sum(0)?;
        Ok((Some(dx), Some(dw)))
    })
}

/// Interleaved RoPE: fused forward, backward is the same rotation with `-sin` (the rotation is orthogonal, so its
/// transpose is its inverse). `x`: `[B, H, T, Dh]`, `cos`/`sin`: `[T, Dh/2]` contiguous.
pub fn rope_i_hand(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let out = candle_nn::rotary_emb::rope_i(&x.detach().contiguous()?, cos, sin)?;
    let (cos, nsin) = (cos.detach(), sin.detach().neg()?.contiguous()?);
    hand_bwd1("rope_i_hand", x, out, move |_x, _y, g| {
        Ok(Some(candle_nn::rotary_emb::rope_i(&g.contiguous()?, &cos, &nsin)?))
    })
}

/// Softmax over the last dimension: fused forward, backward `dx = y * (g - sum(g * y))`.
pub fn softmax_last_hand(x: &Tensor) -> Result<Tensor> {
    let out = candle_nn::ops::softmax_last_dim(&x.detach().contiguous()?)?;
    hand_bwd1("softmax_last_hand", x, out, move |_x, y, g| {
        let g = g.contiguous()?;
        let dot = g.mul(y)?.sum_keepdim(D::Minus1)?;
        Ok(Some(y.mul(&g.broadcast_sub(&dot)?)?))
    })
}

/// Side channel for the *parameter* gradients computed inside checkpointed layers.
///
/// A candle custom op can only return gradients for its (at most three) tensor arguments, so a layer-level checkpoint
/// cannot hand back the gradients of the layer's weights through autograd. Instead the recompute-and-backward step
/// stores them here, and [`CkptCtx::merge_into`] adds them to the `GradStore` returned by the outer `backward()`
/// (summing with any gradient the weights received on the regular path, e.g. the tied embedding).
#[derive(Default)]
pub struct CkptCtx {
    grads: Mutex<Vec<(Var, Tensor)>>,
}

impl CkptCtx {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn add(&self, v: &Var, g: Tensor) -> Result<()> {
        let mut all = self.grads.lock().map_err(|_| candle_core::Error::Msg("CkptCtx poisoned".into()))?;
        if let Some((_, prev)) = all.iter_mut().find(|(pv, _)| pv.id() == v.id()) {
            *prev = (&*prev + &g)?.detach();
        } else {
            all.push((v.clone(), g));
        }
        Ok(())
    }

    /// Move the collected parameter gradients into `store` (adding to existing entries).
    pub fn merge_into(&self, store: &mut GradStore) -> Result<()> {
        let all =
            std::mem::take(&mut *self.grads.lock().map_err(|_| candle_core::Error::Msg("CkptCtx poisoned".into()))?);
        for (v, g) in all {
            let merged = match store.remove(&v) {
                Some(prev) => (prev + g)?.detach(),
                None => g,
            };
            store.insert(&v, merged);
        }
        Ok(())
    }
}

type CkptFn1 = dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync;
type CkptFn2 = dyn Fn(&Tensor, &Tensor) -> Result<Tensor> + Send + Sync;

fn collect_param_grads(ctx: &CkptCtx, params: &[Var], local: &GradStore) -> Result<()> {
    for p in params {
        if let Some(g) = local.get(p) {
            ctx.add(p, g.detach())?;
        }
    }
    Ok(())
}

/// Checkpoint a one-input layer `y = f(x; params)`: the forward keeps only `y` (the intermediate graph is dropped
/// immediately); the backward recomputes `f` on a fresh leaf, back-propagates `grad_res`, returns `dx` through autograd
/// and parks the `params` gradients in `ctx`.
pub fn checkpoint1(
    ctx: &Arc<CkptCtx>,
    x: &Tensor,
    params: &[Var],
    f: impl Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
) -> Result<Tensor> {
    let f: Arc<CkptFn1> = Arc::new(f);
    let out = f(&x.detach())?.detach();
    let (ctx, params) = (ctx.clone(), params.to_vec());
    hand_bwd1("ckpt_layer", x, out, move |x, _y, g| {
        let xv = Var::from_tensor(&x.contiguous()?)?;
        let y = f(xv.as_tensor())?;
        let local = y.mul(&g.contiguous()?.detach())?.sum_all()?.backward()?;
        collect_param_grads(&ctx, &params, &local)?;
        Ok(local.get(&xv).cloned())
    })
}

/// Two-input variant of [`checkpoint1`] (e.g. the recurrent state and the prelude output).
pub fn checkpoint2(
    ctx: &Arc<CkptCtx>,
    x1: &Tensor,
    x2: &Tensor,
    params: &[Var],
    f: impl Fn(&Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
) -> Result<Tensor> {
    let f: Arc<CkptFn2> = Arc::new(f);
    let out = f(&x1.detach(), &x2.detach())?.detach();
    let (ctx, params) = (ctx.clone(), params.to_vec());
    hand_bwd2("ckpt_layer2", x1, x2, out, move |a, b, _y, g| {
        let (av, bv) = (Var::from_tensor(&a.contiguous()?)?, Var::from_tensor(&b.contiguous()?)?);
        let y = f(av.as_tensor(), bv.as_tensor())?;
        let local = y.mul(&g.contiguous()?.detach())?.sum_all()?.backward()?;
        collect_param_grads(&ctx, &params, &local)?;
        Ok((local.get(&av).cloned(), local.get(&bv).cloned()))
    })
}

/// Zero-pad `g` (`[B,H,n,D]`) along dim 2 so it sits at `at..at+n` of a length-`total` tensor.
fn pad_rows(g: &Tensor, at: usize, total: usize) -> Result<Tensor> {
    let (b, h, n, d) = g.dims4()?;
    let mut parts = Vec::new();
    if at > 0 {
        parts.push(Tensor::zeros((b, h, at, d), g.dtype(), g.device())?);
    }
    parts.push(g.clone());
    if at + n < total {
        parts.push(Tensor::zeros((b, h, total - at - n, d), g.dtype(), g.device())?);
    }
    Tensor::cat(&parts, 2)
}

/// One tile of causal attention, forward only: `len` queries at absolute positions `start..start+len` against the key
/// and value prefix `[0, start+len)`. `mask` is the additive `[block, block]` causal mask in absolute positions.
pub fn attn_tile(
    qb: &Tensor,
    kb: &Tensor,
    vb: &Tensor,
    mask: &Tensor,
    start: usize,
    len: usize,
    kern: Kernel,
) -> Result<Tensor> {
    let kv_len = start + len;
    let scale = 1.0 / (qb.dim(D::Minus1)? as f64).sqrt();
    let mb = mask.narrow(0, start, len)?.narrow(1, 0, kv_len)?;
    let att = (qb.matmul(&kb.transpose(2, 3)?)? * scale)?.broadcast_add(&mb)?;
    super::softmax(&att, kern)?.matmul(vb)
}

/// Gradient-checkpointed attention tile. The forward runs on detached inputs and keeps *nothing* but the
/// `[B,H,len,Dh]` output (the `[B,H,len,kv]` score and softmax tensors are freed at once); the backward recomputes the
/// tile with fresh `Var`s, back-propagates `grad_res` through it and returns zero-padded full-size gradients for `q`,
/// `k` and `v`. Peak activation memory is then `O(B*H*len*T)` for one tile at a time instead of `O(B*H*T*T)`.
///
/// `q` holds the local queries (`[B,H,Tq,Dh]`); this tile is `q[.., local..local+len]` and sits at absolute position
/// `abs_start = offset + local` (`offset` is the number of cached positions before the first query).
#[allow(clippy::too_many_arguments)]
pub fn attention_tile_ckpt(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    offset: usize,
    local: usize,
    len: usize,
    kern: Kernel,
) -> Result<Tensor> {
    let abs = offset + local;
    let kv_len = abs + len;
    let out = {
        let (qd, kd, vd) = (q.detach(), k.detach(), v.detach());
        let o = attn_tile(
            &qd.narrow(2, local, len)?,
            &kd.narrow(2, 0, kv_len)?,
            &vd.narrow(2, 0, kv_len)?,
            mask,
            abs,
            len,
            kern,
        )?;
        let o = o.detach(); // drop the op graph so the big intermediates are freed right now
        if kern.sync_tiles {
            o.device().synchronize()?;
        }
        o
    };
    let mask = mask.detach();
    hand_bwd3("attn_tile_ckpt", q, k, v, out, move |q, k, v, _res, g| {
        let (tq, tk) = (q.dim(2)?, k.dim(2)?);
        let qv = Var::from_tensor(&q.narrow(2, local, len)?.contiguous()?)?;
        let kv = Var::from_tensor(&k.narrow(2, 0, kv_len)?.contiguous()?)?;
        let vv = Var::from_tensor(&v.narrow(2, 0, kv_len)?.contiguous()?)?;
        let o = attn_tile(qv.as_tensor(), kv.as_tensor(), vv.as_tensor(), &mask, abs, len, kern)?;
        let grads = o.mul(&g.contiguous()?.detach())?.sum_all()?.backward()?;
        if kern.sync_tiles {
            qv.device().synchronize()?;
        }
        let need = |x: &Var, what: &str| {
            grads
                .get(x)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("attention tile: no gradient for {what}")))
        };
        let dq = pad_rows(&need(&qv, "q")?, local, tq)?;
        let dk = pad_rows(&need(&kv, "k")?, 0, tk)?;
        let dv = pad_rows(&need(&vv, "v")?, 0, tk)?;
        Ok((Some(dq), Some(dk), Some(dv)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{self, testutil::randn};
    use candle_core::Device;

    fn rel(a: &Tensor, b: &Tensor) -> f64 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let num = a.iter().zip(&b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max);
        let den = b.iter().map(|y| y.abs() as f64).fold(1e-12, f64::max);
        num / den
    }

    fn loss(y: &Tensor, seed: u64) -> Tensor {
        let r = randn(y.dims(), 1.0, seed, y.device());
        y.mul(&r).unwrap().sum_all().unwrap()
    }

    #[test]
    fn rms_norm_hand_matches_composed_value_and_grads() {
        let d = Device::Cpu;
        for shape in [vec![37usize, 64], vec![2, 9, 64]] {
            let x = Var::from_tensor(&randn(&shape, 1.0, 1, &d)).unwrap();
            let w = Var::from_tensor(&randn(&[64], 0.5, 2, &d)).unwrap();
            let run = |hand: bool| {
                let y = if hand {
                    rms_norm_hand(x.as_tensor(), w.as_tensor(), 1e-6).unwrap()
                } else {
                    ops::rms_norm_composed(x.as_tensor(), w.as_tensor(), 1e-6).unwrap()
                };
                let g = loss(&y, 3).backward().unwrap();
                (y, g.get(&x).unwrap().clone(), g.get(&w).unwrap().clone())
            };
            let (yh, dxh, dwh) = run(true);
            let (ys, dxs, dws) = run(false);
            assert!(rel(&yh, &ys) < 1e-5);
            assert!(rel(&dxh, &dxs) < 1e-5, "dx {}", rel(&dxh, &dxs));
            assert!(rel(&dwh, &dws) < 1e-5, "dw {}", rel(&dwh, &dws));
        }
    }

    #[test]
    fn rope_hand_matches_composed_value_and_grads() {
        let d = Device::Cpu;
        let (cos, sin) = ops::rope_tables(24, 16, 10000.0, &d).unwrap();
        let x = Var::from_tensor(&randn(&[2, 3, 24, 16], 1.0, 4, &d)).unwrap();
        let run = |hand: bool| {
            let y = if hand {
                rope_i_hand(x.as_tensor(), &cos, &sin).unwrap()
            } else {
                ops::rope_i_composed(x.as_tensor(), &cos, &sin).unwrap()
            };
            let g = loss(&y, 5).backward().unwrap();
            (y, g.get(&x).unwrap().clone())
        };
        let (yh, dxh) = run(true);
        let (ys, dxs) = run(false);
        assert!(rel(&yh, &ys) < 1e-6);
        assert!(rel(&dxh, &dxs) < 1e-5);
    }

    #[test]
    fn softmax_hand_matches_composed_value_and_grads() {
        let d = Device::Cpu;
        let x = Var::from_tensor(&randn(&[2, 3, 10, 10], 2.0, 6, &d)).unwrap();
        let run = |hand: bool| {
            let y = if hand {
                softmax_last_hand(x.as_tensor()).unwrap()
            } else {
                ops::softmax_composed(x.as_tensor()).unwrap()
            };
            let g = loss(&y, 7).backward().unwrap();
            (y, g.get(&x).unwrap().clone())
        };
        let (yh, dxh) = run(true);
        let (ys, dxs) = run(false);
        assert!(rel(&yh, &ys) < 1e-6);
        assert!(rel(&dxh, &dxs) < 1e-5, "dx {}", rel(&dxh, &dxs));
    }

    #[test]
    fn hand_op_in_a_graph_with_two_consumers_accumulates() {
        // y = rms(x); loss = sum(y*R1) + sum(y^2*R2): the custom node receives the *sum* of both gradients.
        let d = Device::Cpu;
        let x = Var::from_tensor(&randn(&[8, 32], 1.0, 6, &d)).unwrap();
        let w = Var::from_tensor(&randn(&[32], 0.5, 7, &d)).unwrap();
        let run = |hand: bool| {
            let y = if hand {
                rms_norm_hand(x.as_tensor(), w.as_tensor(), 1e-6).unwrap()
            } else {
                ops::rms_norm_composed(x.as_tensor(), w.as_tensor(), 1e-6).unwrap()
            };
            let l = (loss(&y, 8) + loss(&y.sqr().unwrap(), 9)).unwrap();
            let g = l.backward().unwrap();
            (g.get(&x).unwrap().clone(), g.get(&w).unwrap().clone())
        };
        let (a, b) = run(true);
        let (c, e) = run(false);
        assert!(rel(&a, &c) < 1e-5 && rel(&b, &e) < 1e-5);
    }

    #[test]
    fn precomputed_forward_is_consumed_exactly_once() {
        let x = Tensor::ones((2, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let slot = Mutex::new(Some((&x * 2.0).unwrap()));
        assert!(take_pre(&slot, "t").is_ok());
        assert!(take_pre(&slot, "t").is_err());
    }

    #[test]
    fn checkpointed_tiled_attention_matches_full_attention_grads() {
        let d = Device::Cpu;
        let shape = [1usize, 2, 40, 8];
        let q = Var::from_tensor(&randn(&shape, 1.0, 11, &d)).unwrap();
        let k = Var::from_tensor(&randn(&shape, 1.0, 12, &d)).unwrap();
        let v = Var::from_tensor(&randn(&shape, 1.0, 13, &d)).unwrap();
        let mask = ops::causal_mask(40, &d).unwrap();
        let run = |tile: Option<usize>| {
            let y = match tile {
                Some(b) => {
                    ops::attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), &mask, 0, ops::Kernel::tiled(b, true))
                        .unwrap()
                }
                None => {
                    ops::attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), &mask, 0, ops::Kernel::plain()).unwrap()
                }
            };
            let g = loss(&y, 14).backward().unwrap();
            (y, g.get(&q).unwrap().clone(), g.get(&k).unwrap().clone(), g.get(&v).unwrap().clone())
        };
        let full = run(None);
        for b in [8usize, 13, 40] {
            let c = run(Some(b));
            assert!(rel(&c.0, &full.0) < 1e-5, "value, block {b}");
            assert!(rel(&c.1, &full.1) < 1e-4, "dq, block {b}: {}", rel(&c.1, &full.1));
            assert!(rel(&c.2, &full.2) < 1e-4, "dk, block {b}: {}", rel(&c.2, &full.2));
            assert!(rel(&c.3, &full.3) < 1e-4, "dv, block {b}: {}", rel(&c.3, &full.3));
        }
    }

    #[test]
    fn checkpointed_tiles_respect_a_cache_offset() {
        // 5 cached positions then 11 new queries: tiles must see the cached prefix and mask by absolute position.
        let d = Device::Cpu;
        let (p, t) = (5usize, 11usize);
        let q = Var::from_tensor(&randn(&[1, 2, t, 8], 1.0, 21, &d)).unwrap();
        let k = Var::from_tensor(&randn(&[1, 2, p + t, 8], 1.0, 22, &d)).unwrap();
        let v = Var::from_tensor(&randn(&[1, 2, p + t, 8], 1.0, 23, &d)).unwrap();
        let mask = ops::causal_mask(32, &d).unwrap();
        let run = |kern: ops::Kernel| {
            let y = ops::attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), &mask, p, kern).unwrap();
            let g = loss(&y, 24).backward().unwrap();
            (y, g.get(&q).unwrap().clone(), g.get(&k).unwrap().clone())
        };
        let full = run(ops::Kernel::plain());
        let tiled = run(ops::Kernel::tiled(4, true));
        assert!(rel(&tiled.0, &full.0) < 1e-5);
        assert!(rel(&tiled.1, &full.1) < 1e-4);
        assert!(rel(&tiled.2, &full.2) < 1e-4);
    }
}
