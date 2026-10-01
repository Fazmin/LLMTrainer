//! Tensor primitives: RMSNorm, rotary positions, causal attention, SwiGLU, top-k and the cross-entropy.
//!
//! candle's fused kernels (`rms_norm`, `rope_i`, `softmax_last_dim`) are registered *without* a gradient, so training
//! must not call them directly. Each op here exists in two forms:
//!
//! - **composed** from differentiable primitives (always correct, more kernel launches), and
//! - **hand** (`hand.rs`): the fused forward kernel plus a hand-written backward, which the spike measured at 16-21%
//!   faster per step. [`Kernel::hand`] selects it.
//!
//! Rules every op follows (found by the spikes): tensors stay at rank 4 or below, ids are contiguous `u32`, no
//! `broadcast_as` feeds a matmul, and `scatter_add` is never used in a differentiable path (its backward is wrong).
//!
//! Weights are kept in PyTorch's `[out, in]` layout so a checkpoint maps one-to-one; [`linear`] multiplies by the
//! transposed view.

pub mod hand;

use candle_core::{D, DType, Device, Result, Tensor};

/// How the heavy ops are executed. Cheap to copy, passed down from the model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernel {
    /// Use the fused forward + hand-written backward versions (RMSNorm, RoPE, softmax).
    pub hand: bool,
    /// Query-tile size for attention; `0` computes the whole score matrix at once.
    pub q_block: usize,
    /// Recompute each attention tile in the backward pass instead of keeping its scores (needs `q_block > 0`).
    pub ckpt: bool,
    /// Synchronise the device after each tile so Metal returns its buffers to the pool early.
    pub sync_tiles: bool,
}

impl Kernel {
    /// Composed ops, no tiling: the reference behaviour.
    pub const fn plain() -> Self {
        Self { hand: false, q_block: 0, ckpt: false, sync_tiles: false }
    }

    /// Composed ops with tiled attention.
    pub const fn tiled(q_block: usize, ckpt: bool) -> Self {
        Self { hand: false, q_block, ckpt, sync_tiles: false }
    }

    /// Fused forward + hand backward, no tiling.
    pub const fn hand() -> Self {
        Self { hand: true, q_block: 0, ckpt: false, sync_tiles: false }
    }

    pub const fn with_tiles(mut self, q_block: usize, ckpt: bool) -> Self {
        self.q_block = q_block;
        self.ckpt = ckpt;
        self
    }
}

/// `x @ w^T` for `x: [N, in]` and a PyTorch-layout weight `w: [out, in]`.
pub fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)
}

/// RoPE cos/sin tables of shape `[t, head_dim/2]` (f32): frequency `theta^(-2i/head_dim)` per pair.
pub fn rope_tables(t: usize, head_dim: usize, theta: f32, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)).collect();
    let mut cos = Vec::with_capacity(t * half);
    let mut sin = Vec::with_capacity(t * half);
    for pos in 0..t {
        for f in &inv {
            let a = pos as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    Ok((Tensor::from_vec(cos, (t, half), dev)?, Tensor::from_vec(sin, (t, half), dev)?))
}

/// Interleaved ("GPT-J style", pairs `(x[2i], x[2i+1])`) RoPE composed from differentiable ops at rank <= 4.
/// `x`: `[B, H, T, Dh]`, `cos`/`sin`: `[T, Dh/2]`.
pub fn rope_i_composed(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, t, dh) = x.dims4()?;
    let cos = cos.narrow(0, 0, t)?.unsqueeze(2)?; // [T, Dh/2, 1]
    let sin = sin.narrow(0, 0, t)?.unsqueeze(2)?;
    let x = x.reshape((b * h, t, dh / 2, 2))?;
    let x0 = x.narrow(3, 0, 1)?;
    let x1 = x.narrow(3, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    Tensor::cat(&[y0, y1], 3)?.reshape((b, h, t, dh))
}

/// Rotary positions for `x: [B, H, T, Dh]` with tables already sliced to the right absolute positions.
pub fn rope_i(x: &Tensor, cos: &Tensor, sin: &Tensor, k: Kernel) -> Result<Tensor> {
    if k.hand { hand::rope_i_hand(x, &cos.contiguous()?, &sin.contiguous()?) } else { rope_i_composed(x, cos, sin) }
}

/// RMSNorm with learned scale, composed (`x * rsqrt(mean(x^2) + eps) * w`).
pub fn rms_norm_composed(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm_slow(x, w, eps)
}

pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32, k: Kernel) -> Result<Tensor> {
    if k.hand { hand::rms_norm_hand(x, w, eps) } else { rms_norm_composed(x, w, eps) }
}

/// Softmax over the last dim with a *detached* max (softmax is shift invariant, so detaching is exact and avoids
/// routing gradient through `max`).
pub fn softmax_composed(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?.detach();
    let e = x.broadcast_sub(&m)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    e.broadcast_div(&s)
}

/// Alias of [`softmax_composed`] for callers that never want the fused kernel.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    softmax_composed(x)
}

pub fn softmax(x: &Tensor, k: Kernel) -> Result<Tensor> {
    if k.hand { hand::softmax_last_hand(x) } else { softmax_composed(x) }
}

/// Additive causal mask `[n, n]` in absolute positions: 0 on and below the diagonal, `-1e9` above.
pub fn causal_mask(n: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; n * n];
    for i in 0..n {
        for j in i + 1..n {
            m[i * n + j] = -1e9;
        }
    }
    Tensor::from_vec(m, (n, n), dev)
}

/// Causal attention for `q: [B,H,T,Dh]` against `k, v: [B,H,P+T,Dh]` where the first `offset = P` key positions are
/// cached ones. Query `i` sits at absolute position `P + i` and sees keys `0..=P+i`. `mask` is the additive
/// `[block, block]` causal mask in absolute positions.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, offset: usize, kern: Kernel) -> Result<Tensor> {
    let t = q.dim(2)?;
    if kern.q_block == 0 || t <= kern.q_block {
        return hand::attn_tile(q, k, v, mask, offset, t, kern);
    }
    let mut outs = Vec::with_capacity(t.div_ceil(kern.q_block));
    let mut local = 0;
    while local < t {
        let len = kern.q_block.min(t - local);
        outs.push(if kern.ckpt {
            hand::attention_tile_ckpt(q, k, v, mask, offset, local, len, kern)?
        } else {
            let kv_len = offset + local + len;
            hand::attn_tile(
                &q.narrow(2, local, len)?,
                &k.narrow(2, 0, kv_len)?,
                &v.narrow(2, 0, kv_len)?,
                mask,
                offset + local,
                len,
                kern,
            )?
        });
        local += len;
    }
    Tensor::cat(&outs, 2)
}

/// Top-k along the last dim of a 2-D tensor via descending argsort.
/// Returns `(values [N,k], indices [N,k] u32 contiguous)`; values are differentiable. Ties go to the lower index.
pub fn topk_last(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let idx = x.arg_sort_last_dim(false)?.narrow(D::Minus1, 0, k)?.contiguous()?;
    let vals = x.gather(&idx, D::Minus1)?;
    Ok((vals, idx))
}

/// Per-row negative log likelihood: `logits [N, V]`, `targets [N]` (u32) -> `[N]` (no reduction).
pub fn cross_entropy_rows(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let m = logits.max_keepdim(D::Minus1)?.detach();
    let z = logits.broadcast_sub(&m)?;
    let lse = z.exp()?.sum_keepdim(D::Minus1)?.log()?;
    let logp = z.broadcast_sub(&lse)?;
    logp.gather(&targets.unsqueeze(1)?.contiguous()?, 1)?.squeeze(1)?.neg()
}

/// `silu(x @ w1^T) * (x @ w3^T)` then `@ w2^T`, with PyTorch-layout weights (`w1, w3: [ff, d]`, `w2: [d, ff]`).
pub fn swiglu(x: &Tensor, w1: &Tensor, w3: &Tensor, w2: &Tensor) -> Result<Tensor> {
    let a = candle_nn::ops::silu(&linear(x, w1)?)?;
    let b = linear(x, w3)?;
    linear(&(a * b)?, w2)
}

/// Numerically stable `logsumexp` over the last dim, keeping the reduced dim.
pub fn logsumexp_keepdim(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?.detach();
    let s = x.broadcast_sub(&m)?.exp()?.sum_keepdim(D::Minus1)?.log()?;
    s + m
}

/// Read a scalar tensor as f32.
pub fn scalar(t: &Tensor) -> Result<f32> {
    t.to_dtype(DType::F32)?.flatten_all()?.get(0)?.to_scalar::<f32>()
}

/// Copy a tensor to the host as a flat `Vec<f32>`.
pub fn to_host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

#[cfg(test)]
pub(crate) mod testutil {
    use candle_core::{Device, Tensor};

    /// A normal tensor from a seeded host generator, identical on every backend.
    pub fn randn(shape: &[usize], std: f32, seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let v = crate::rng::HostRng::new(seed).normal_vec(n, std);
        Tensor::from_vec(v, shape.to_vec(), dev).unwrap()
    }

    pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{max_abs_diff, randn};
    use super::*;

    #[test]
    fn rope_composed_matches_candle_slow_and_fused() {
        let d = Device::Cpu;
        let x = randn(&[2, 3, 16, 8], 1.0, 1, &d);
        let (c, s) = rope_tables(16, 8, 10000.0, &d).unwrap();
        let a = rope_i_composed(&x, &c, &s).unwrap();
        let b = candle_nn::rotary_emb::rope_i_slow(&x, &c, &s).unwrap();
        assert!(max_abs_diff(&a, &b) < 1e-6);
        let f = candle_nn::rotary_emb::rope_i(&x.contiguous().unwrap(), &c, &s).unwrap();
        assert!(max_abs_diff(&a, &f) < 1e-5);
    }

    #[test]
    fn rope_is_relative_so_a_shifted_window_keeps_its_offsets() {
        // The dot product of a rotated q and k depends only on their position difference.
        let d = Device::Cpu;
        let (c, s) = rope_tables(64, 8, 10000.0, &d).unwrap();
        let q = randn(&[1, 1, 1, 8], 1.0, 2, &d);
        let k = randn(&[1, 1, 1, 8], 1.0, 3, &d);
        let dot = |pq: usize, pk: usize| {
            let rq = rope_i_composed(&q, &c.narrow(0, pq, 1).unwrap(), &s.narrow(0, pq, 1).unwrap()).unwrap();
            let rk = rope_i_composed(&k, &c.narrow(0, pk, 1).unwrap(), &s.narrow(0, pk, 1).unwrap()).unwrap();
            scalar(&rq.mul(&rk).unwrap().sum_all().unwrap()).unwrap()
        };
        assert!((dot(10, 4) - dot(40, 34)).abs() < 1e-4);
        assert!((dot(10, 4) - dot(10, 5)).abs() > 1e-3);
    }

    #[test]
    fn tiled_attention_equals_full_and_is_causal() {
        let d = Device::Cpu;
        let (q, k, v) =
            (randn(&[1, 2, 40, 8], 1.0, 4, &d), randn(&[1, 2, 40, 8], 1.0, 5, &d), randn(&[1, 2, 40, 8], 1.0, 6, &d));
        let m = causal_mask(64, &d).unwrap();
        let full = attention(&q, &k, &v, &m, 0, Kernel::plain()).unwrap();
        for qb in [7, 16, 40, 64] {
            let tiled = attention(&q, &k, &v, &m, 0, Kernel::tiled(qb, false)).unwrap();
            assert!(max_abs_diff(&full, &tiled) < 1e-5, "q_block {qb}");
        }
        let hand = attention(&q, &k, &v, &m, 0, Kernel::hand()).unwrap();
        assert!(max_abs_diff(&full, &hand) < 1e-5);
        // causality: changing keys/values after position j must not change outputs <= j
        let j = 20;
        let k2 =
            Tensor::cat(&[&k.narrow(2, 0, j + 1).unwrap(), &randn(&[1, 2, 40 - j - 1, 8], 1.0, 7, &d)], 2).unwrap();
        let v2 =
            Tensor::cat(&[&v.narrow(2, 0, j + 1).unwrap(), &randn(&[1, 2, 40 - j - 1, 8], 1.0, 8, &d)], 2).unwrap();
        let out2 = attention(&q, &k2, &v2, &m, 0, Kernel::plain()).unwrap();
        assert!(max_abs_diff(&full.narrow(2, 0, j + 1).unwrap(), &out2.narrow(2, 0, j + 1).unwrap()) < 1e-6);
        assert!(max_abs_diff(&full.narrow(2, j + 1, 5).unwrap(), &out2.narrow(2, j + 1, 5).unwrap()) > 1e-3);
    }

    #[test]
    fn cached_attention_matches_one_pass() {
        // Attending the last T queries against P cached + T new keys equals the tail of a full-length pass.
        let d = Device::Cpu;
        let (n, p) = (24usize, 15usize);
        let (q, k, v) =
            (randn(&[1, 2, n, 8], 1.0, 9, &d), randn(&[1, 2, n, 8], 1.0, 10, &d), randn(&[1, 2, n, 8], 1.0, 11, &d));
        let m = causal_mask(32, &d).unwrap();
        let full = attention(&q, &k, &v, &m, 0, Kernel::plain()).unwrap();
        let tail = attention(&q.narrow(2, p, n - p).unwrap(), &k, &v, &m, p, Kernel::plain()).unwrap();
        assert!(max_abs_diff(&full.narrow(2, p, n - p).unwrap(), &tail) < 1e-5);
    }

    #[test]
    fn softmax_topk_cross_entropy_basics() {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 3.0, 2.0, 0.5, -1.0, 4.0, 0.0, 2.5], (2, 4), &d).unwrap();
        let s = softmax_last(&x).unwrap().sum(1).unwrap().to_vec1::<f32>().unwrap();
        assert!(s.iter().all(|v| (v - 1.0).abs() < 1e-6));
        let (vals, idx) = topk_last(&x, 2).unwrap();
        assert_eq!(idx.to_vec2::<u32>().unwrap(), vec![vec![1, 2], vec![1, 3]]);
        assert_eq!(vals.to_vec2::<f32>().unwrap(), vec![vec![3.0, 2.0], vec![4.0, 2.5]]);
        // uniform logits give ln(V) for every row
        let u = Tensor::zeros((3, 5), DType::F32, &d).unwrap();
        let t = Tensor::from_vec(vec![0u32, 2, 4], 3, &d).unwrap();
        let rows = cross_entropy_rows(&u, &t).unwrap().to_vec1::<f32>().unwrap();
        assert!(rows.iter().all(|r| (r - 5f32.ln()).abs() < 1e-6));
    }

    #[test]
    fn logsumexp_matches_naive() {
        let d = Device::Cpu;
        let x = randn(&[4, 7], 3.0, 12, &d);
        let a = logsumexp_keepdim(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rows = x.to_vec2::<f32>().unwrap();
        for (i, row) in rows.iter().enumerate() {
            let naive = row.iter().map(|v| v.exp()).sum::<f32>().ln();
            assert!((a[i] - naive).abs() < 1e-5);
        }
    }

    #[test]
    fn linear_uses_the_pytorch_weight_layout() {
        let d = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0], (1, 2), &d).unwrap();
        let w = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &d).unwrap(); // [out=3, in=2]
        assert_eq!(linear(&x, &w).unwrap().to_vec2::<f32>().unwrap(), vec![vec![1.0, 2.0, 3.0]]);
    }
}
