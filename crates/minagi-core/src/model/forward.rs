//! The forward pass: dense prelude, then the weight-shared recurrent block applied for several "rows", with PonderNet
//! halting per character.
//!
//! **Rows.** Row `n` mixes the running state with the embedded input (`adapter([h, x])`), applies the one recurrent
//! block (attention, then the expert pool) and reads out logits and a halting probability `lam`. A character's chance of
//! stopping at row `n` is `p_n = cum * lam` where `cum` is the chance it has not stopped yet. Training minimises the
//! expected loss under that distribution, `sum_n p_n * CE_n`, plus a KL pull toward a geometric prior so it does not
//! simply always run long.
//!
//! **Freeze and carry.** A character that has halted is finished: its state stops changing, its experts stop running
//! (it asks for none and takes no expert's capacity), and the characters after it read its final state at every deeper
//! row through the keys and values the recurrent block makes of it. Training runs exactly this as well, so what the
//! model learns from is what it writes with.
//!
//! **Caches.** Each row has its own key/value cache (`n_prelude + max_steps` in all) because the same block at
//! different rows sees different states. With a cache the queries sit at absolute positions `P..P+T` against `P+T` keys.

use candle_core::{DType, Tensor};

use super::Model;
use super::params::AttnView;
use crate::error::{EngineError, Result};
use crate::moe::{RouteCfg, RouteStats};
use crate::ops::{self, Kernel};

/// Per-row keys and values: `[1, heads, positions, head_dim]` each.
#[derive(Default)]
pub struct Caches {
    slots: Vec<Option<(Tensor, Tensor)>>,
}

impl Caches {
    pub fn new(n_slots: usize) -> Self {
        Self { slots: (0..n_slots).map(|_| None).collect() }
    }

    pub fn clear(&mut self) {
        self.slots.iter_mut().for_each(|s| *s = None);
    }

    /// Positions held (every slot holds the same number).
    pub fn positions(&self) -> usize {
        self.slots.iter().flatten().next().map(|(k, _)| k.dims()[2]).unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

pub struct FwdIn<'a> {
    pub ids: &'a [u32],
    /// Next-character targets. With targets the pass computes the PonderNet loss; without, it picks each character's
    /// logits from the row where it halted (what writing uses).
    pub targets: Option<&'a [u32]>,
    pub caches: Option<&'a mut Caches>,
    /// Absolute position of `ids[0]` (the number of cached positions).
    pub pos_offset: usize,
    /// Rows to run (the ceiling at inference; sampled per step while training).
    pub n_steps: usize,
    /// Record a graph for the backward pass, explore when choosing experts, truncate backprop.
    pub train: bool,
    /// Also return the halting-weighted mixture of every row's logits (with targets) - for tests and accuracy.
    pub want_logits: bool,
}

#[derive(Default)]
pub struct FwdOut {
    /// The PonderNet loss (scalar), when targets were given.
    pub loss: Option<Tensor>,
    /// The pool's auxiliary load-balancing loss from the last row that routed.
    pub aux: Option<Tensor>,
    /// `[T, vocab]`: the logits to read (halted row's logits when writing; the mixture when `want_logits`).
    pub logits: Option<Tensor>,
    /// Writing: the row at which each character halted (1-based).
    pub steps: Vec<f32>,
    /// With targets: mean halting probability of each row, i.e. the distribution of rows used.
    pub row_mass: Vec<f32>,
    /// Rows that did any work.
    pub rows_run: usize,
    pub route: RouteStats,
    /// Experts brought onto the card by this forward.
    pub loads: u64,
}

impl FwdOut {
    /// Expected number of rows per character, from `row_mass`.
    pub fn expected_rows(&self) -> f32 {
        self.row_mass.iter().enumerate().map(|(i, m)| (i as f32 + 1.0) * m).sum()
    }
}

pub(super) struct Ctx<'a> {
    pub(super) heads: usize,
    pub(super) head_dim: usize,
    pub(super) eps: f32,
    pub(super) kernel: Kernel,
    pub(super) mask: &'a Tensor,
    pub(super) cos: &'a Tensor,
    pub(super) sin: &'a Tensor,
}

fn split_heads(qkv: &Tensor, i: usize, d: usize, t: usize, heads: usize, dh: usize) -> Result<Tensor> {
    Ok(qkv.narrow(1, i * d, d)?.reshape((1, t, heads, dh))?.transpose(1, 2)?.contiguous()?)
}

/// Attention half of a block: `x + proj(attn(rope(qkv(norm(x)))))`, reading and extending this slot's cache.
pub(super) fn attn_block(
    c: &Ctx,
    w: &AttnView,
    x: &Tensor,
    cache: Option<&mut Option<(Tensor, Tensor)>>,
) -> Result<Tensor> {
    let (t, d) = x.dims2()?;
    let xn = ops::rms_norm(x, &w.ln1, c.eps, c.kernel)?;
    let qkv = ops::linear(&xn, &w.qkv)?;
    let q = ops::rope_i(&split_heads(&qkv, 0, d, t, c.heads, c.head_dim)?, c.cos, c.sin, c.kernel)?;
    let k = ops::rope_i(&split_heads(&qkv, 1, d, t, c.heads, c.head_dim)?, c.cos, c.sin, c.kernel)?;
    let v = split_heads(&qkv, 2, d, t, c.heads, c.head_dim)?;
    let (k, v) = match cache {
        Some(slot) => {
            let (k2, v2) = match slot.take() {
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
                None => (k, v),
            };
            *slot = Some((k2.detach(), v2.detach()));
            (k2, v2)
        }
        None => (k, v),
    };
    let offset = k.dim(2)? - t;
    let y = ops::attention(&q, &k, &v, c.mask, offset, c.kernel)?;
    let y = y.transpose(1, 2)?.contiguous()?.reshape((t, d))?;
    Ok((x + ops::linear(&y, &w.proj)?)?)
}

/// The keys and values a position would add to a cache, with nothing else computed.
fn kv_only(c: &Ctx, w: &AttnView, x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (t, d) = x.dims2()?;
    let xn = ops::rms_norm(x, &w.ln1, c.eps, c.kernel)?;
    let kv = ops::linear(&xn, &w.qkv.narrow(0, d, 2 * d)?)?;
    let k = ops::rope_i(&split_heads(&kv, 0, d, t, c.heads, c.head_dim)?, c.cos, c.sin, c.kernel)?;
    let v = split_heads(&kv, 1, d, t, c.heads, c.head_dim)?;
    Ok((k.detach(), v.detach()))
}

fn extend_cache(slot: &mut Option<(Tensor, Tensor)>, k: &Tensor, v: &Tensor) -> Result<()> {
    *slot = Some(match slot.take() {
        Some((pk, pv)) => (Tensor::cat(&[&pk, k], 2)?, Tensor::cat(&[&pv, v], 2)?),
        None => (k.clone(), v.clone()),
    });
    Ok(())
}

/// Float 0/1 column from a host mask.
pub(super) fn column(mask: &[bool], dev: &candle_core::Device) -> Result<Tensor> {
    let v: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    Ok(Tensor::from_vec(v, (mask.len(), 1), dev)?)
}

/// `mask ? a : b` row-wise, differentiable in both (a multiply-add rather than `where`).
pub(super) fn select_rows(mask: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let inv = mask.affine(-1.0, 1.0)?;
    Ok((a.broadcast_mul(mask)? + b.broadcast_mul(&inv)?)?)
}

impl Model {
    /// Run the network over `inp.ids`. See the module docs for what happens; the returned tensors belong to the
    /// autograd graph when `inp.train` is set.
    pub fn forward(&mut self, inp: FwdIn) -> Result<FwdOut> {
        let FwdIn { ids, targets, mut caches, pos_offset, n_steps, train, want_logits } = inp;
        let t = ids.len();
        let cfg = self.cfg.clone();
        let (d, block) = (cfg.d_model as usize, cfg.block as usize);
        if t == 0 {
            return Err(EngineError::other("nothing to read"));
        }
        if pos_offset + t > block {
            return Err(EngineError::Config(format!(
                "reading at position {} but the position tables were built to {}. The window may not exceed the longest \
                 context the model was built with.",
                pos_offset + t,
                block
            )));
        }
        if let Some(tg) = targets
            && tg.len() != t
        {
            return Err(EngineError::other("targets and inputs differ in length"));
        }
        let n_steps = n_steps.clamp(1, cfg.max_steps as usize);
        let dev = self.dev.clone();
        let loads_before = self.pool.book.loads;

        // THE TEXT CHOOSES: nothing admitted yet; this forward's first pass adds its requests to its text's vote. A
        // forward from position 0 begins a new text; a forward that trains also explores.
        if caches.is_none() || pos_offset == 0 {
            self.pool.book.begin_text(train);
        } else {
            self.pool.book.begin_forward(train);
        }

        let view = self.trunk.view(train);
        let cos = self.cos.narrow(0, pos_offset, t)?.contiguous()?;
        let sin = self.sin.narrow(0, pos_offset, t)?.contiguous()?;
        let ctx = Ctx {
            heads: cfg.n_head as usize,
            head_dim: d / cfg.n_head as usize,
            eps: 1e-6,
            kernel: self.opts.kernel,
            mask: &self.mask,
            cos: &cos,
            sin: &sin,
        };
        let route_cfg = RouteCfg {
            top_k: cfg.pool_top_k as usize,
            capacity_factor: self.opts.capacity_factor,
            mode: self.opts.moe_mode,
            z_weight: self.opts.z_weight,
            track: train,
            kernel: self.opts.kernel,
        };
        let n_pre = cfg.n_prelude as usize;

        let ids_t = Tensor::from_slice(ids, t, &dev)?;
        let targets_t = match targets {
            Some(tg) => Some(Tensor::from_slice(tg, t, &dev)?),
            None => None,
        };

        // ---- prelude -------------------------------------------------------------------------------------------------
        let mut x = view.emb.index_select(&ids_t, 0)?;
        for (i, (a, m)) in view.prelude.iter().enumerate() {
            x = attn_block(&ctx, a, &x, caches.as_deref_mut().map(|c| &mut c.slots[i]))?;
            let xn = ops::rms_norm(&x, &a.ln2, ctx.eps, ctx.kernel)?;
            x = (&x + ops::swiglu(&xn, &m.w1, &m.w3, &m.w2)?)?;
        }

        // ---- rows ---------------------------------------------------------------------------------------------------
        let mut h = Tensor::zeros((t, d), DType::F32, &dev)?;
        let mut cum = Tensor::ones((t, 1), DType::F32, &dev)?;
        let mut halted = vec![false; t];
        let (mut loss_terms, mut p_terms, mut masses) = (Vec::new(), Vec::new(), Vec::new());
        let mut mixture: Option<Tensor> = None;
        let mut halted_logits: Option<Tensor> = None;
        let mut steps_used = vec![1f32; t];
        let mut carried: Option<(Tensor, Tensor)> = None;
        let mut out = FwdOut::default();
        let freeze = cfg.halt_freeze;
        let thresh = cfg.halt_thresh as f32;
        let bptt = cfg.bptt_window as usize;

        for n in 0..n_steps {
            // truncated backprop: only the last few rows carry gradient from the one before
            if targets.is_some() && bptt > 0 && n + bptt < n_steps {
                h = h.detach();
            }
            let active: Option<Vec<bool>> = (freeze && n > 0).then(|| halted.iter().map(|&b| !b).collect());
            let any_active = active.as_ref().is_none_or(|a| a.iter().any(|&b| b));
            let slot = n_pre + n;

            if !any_active {
                // Every character has halted. What is left is what the characters after them will read at this row.
                if let Some(c) = caches.as_deref_mut() {
                    if carried.is_none() {
                        let u = ops::linear(&Tensor::cat(&[&h, &x], 1)?, &view.adapter)?;
                        carried = Some(kv_only(&ctx, &view.recur, &u)?);
                    }
                    if let Some((k, v)) = &carried {
                        extend_cache(&mut c.slots[slot], k, v)?;
                    }
                }
                if targets.is_none() {
                    continue;
                }
            } else {
                let u = ops::linear(&Tensor::cat(&[&h, &x], 1)?, &view.adapter)?;
                let xa = attn_block(&ctx, &view.recur, &u, caches.as_deref_mut().map(|c| &mut c.slots[slot]))?;
                let xn = ops::rms_norm(&xa, &view.recur.ln2, ctx.eps, ctx.kernel)?;
                let y = match &active {
                    Some(a) if !a.iter().all(|&b| b) => {
                        // a halted character routes nowhere: it asks for no experts, takes no capacity, and counts in
                        // none of the statistics or the balancing loss
                        let idx: Vec<u32> = a.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u32).collect();
                        let it = Tensor::from_slice(&idx, idx.len(), &dev)?;
                        let routed =
                            self.pool.route(&xn.index_select(&it, 0)?, &view.router, &view.depth_emb, &route_cfg)?;
                        let mut inv = vec![u32::MAX; t];
                        for (p, &i) in idx.iter().enumerate() {
                            inv[i as usize] = p as u32;
                        }
                        let inv = Tensor::from_vec(inv, t, &dev)?;
                        out.aux = Some(routed.aux);
                        add_stats(&mut out.route, routed.stats);
                        routed.y.index_select(&inv, 0)?
                    }
                    _ => {
                        let routed = self.pool.route(&xn, &view.router, &view.depth_emb, &route_cfg)?;
                        out.aux = Some(routed.aux);
                        add_stats(&mut out.route, routed.stats);
                        routed.y
                    }
                };
                let hn = (&xa + y)?;
                h = match &active {
                    Some(a) if !a.iter().all(|&b| b) => select_rows(&column(a, &dev)?, &hn, &h)?,
                    _ => hn,
                };
                out.rows_run += 1;
            }

            let yf = ops::rms_norm(&h, &view.ln_f, ctx.eps, ctx.kernel)?;
            let logits_n = ops::linear(&yf, &view.emb)?;
            let mut lam = candle_nn::ops::sigmoid(&ops::linear(&yf, &view.halt_w)?.broadcast_add(&view.halt_b)?)?;
            if n + 1 == n_steps {
                lam = lam.ones_like()?; // must stop
            } else if (n as u32) + 1 < cfg.min_steps {
                lam = lam.zeros_like()?; // must continue
            }
            let p_n = cum.mul(&lam)?;
            cum = cum.mul(&lam.affine(-1.0, 1.0)?)?;
            let last = n + 1 == n_steps;

            if let Some(tt) = &targets_t {
                let ce = ops::cross_entropy_rows(&logits_n, tt)?;
                let p_flat = p_n.squeeze(1)?;
                loss_terms.push(p_flat.mul(&ce)?);
                masses.push(p_flat.detach().mean_all()?);
                p_terms.push(p_flat);
                if want_logits {
                    let term = logits_n.broadcast_mul(&p_n)?;
                    mixture = Some(match mixture.take() {
                        Some(m) => (m + term)?,
                        None => term,
                    });
                }
                if freeze && !last {
                    // the same rule writing halts by; from the next row a halted character's loss is its final prediction's
                    let c: Vec<f32> = cum.detach().flatten_all()?.to_vec1()?;
                    for (hl, c) in halted.iter_mut().zip(c) {
                        *hl |= 1.0 - c >= thresh;
                    }
                }
            } else {
                // Each character halts on its own schedule: the first row whose cumulative halting mass crosses the
                // threshold is the one whose logits it keeps. The forced stop at the last row guarantees every
                // character halts somewhere.
                let c: Vec<f32> = cum.detach().flatten_all()?.to_vec1()?;
                let first = halted_logits.is_none();
                if first {
                    halted_logits = Some(logits_n.clone());
                }
                let newly: Vec<bool> = halted.iter().zip(&c).map(|(&hl, &c)| !hl && 1.0 - c >= thresh).collect();
                if newly.iter().any(|&b| b) {
                    let m = column(&newly, &dev)?;
                    halted_logits = Some(match halted_logits.take() {
                        Some(prev) => select_rows(&m, &logits_n, &prev)?,
                        None => logits_n.clone(),
                    });
                    for (i, &nw) in newly.iter().enumerate() {
                        if nw {
                            steps_used[i] = (n + 1) as f32;
                        }
                    }
                }
                for (hl, nw) in halted.iter_mut().zip(newly) {
                    *hl |= nw;
                }
            }
        }

        out.loads = self.pool.book.loads - loads_before;
        if targets.is_none() {
            out.logits = halted_logits;
            out.steps = steps_used;
            return Ok(out);
        }

        // PonderNet: expected loss under the halting distribution, plus a KL pull toward a geometric prior.
        let rows = p_terms.len();
        let p = Tensor::stack(&p_terms, 0)?; // [N, T]
        let l = Tensor::stack(&loss_terms, 0)?;
        let mut loss = l.sum(0)?.mean_all()?;
        let prior: Vec<f32> =
            (0..rows).map(|n| (cfg.halt_prior * (1.0 - cfg.halt_prior).powi(n as i32)) as f32).collect();
        let total: f32 = prior.iter().sum();
        let ln_prior: Vec<f32> = prior.iter().map(|p| (p / total).ln()).collect();
        let pc = p.clamp(1e-8f32, 1e30f32)?;
        let ln_prior = Tensor::from_vec(ln_prior, (rows, 1), &dev)?;
        let kl = pc.mul(&pc.log()?.broadcast_sub(&ln_prior)?)?.sum(0)?.mean_all()?;
        loss = (loss + kl.affine(cfg.ponder_beta, 0.0)?)?;
        out.row_mass = Tensor::stack(&masses, 0)?.to_vec1()?;
        out.loss = Some(loss);
        out.logits = mixture;
        Ok(out)
    }
}

fn add_stats(total: &mut RouteStats, s: RouteStats) {
    total.tokens += s.tokens;
    total.routed += s.routed;
    total.dropped += s.dropped;
}

/// Row count to run for one training step: `Poisson(mean) + 1`, clamped to `[min_steps, max_steps]` (the ceiling when
/// `mean <= 0`). Writing and evaluation always use the ceiling.
pub fn sample_depth(cfg: &minagi_types::ModelConfig, train: bool, rng: &mut crate::rng::HostRng) -> usize {
    let (lo, hi) = (cfg.min_steps as usize, cfg.max_steps as usize);
    if !train || cfg.train_steps_mean <= 0.0 {
        return hi;
    }
    ((rng.poisson(cfg.train_steps_mean) + 1) as usize).clamp(lo, hi)
}
