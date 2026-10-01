//! Row checkpointing: a training step whose memory does not grow with the number of rows.
//!
//! The ordinary training forward keeps every row's activations alive until the backward pass (a Full-size row at 2,048
//! characters is gigabytes, and a step runs a dozen). Here the forward runs **without a graph** and keeps only the small
//! state that links rows (the latent state `h` and the not-yet-halted share `cum`, each one row's worth of vectors) plus
//! the decisions that must be repeated exactly: which characters were still active and which experts were on the card.
//!
//! The backward then walks the rows from the last to the first. Row `n`'s loss decomposes into terms that depend only on
//! that row's computation, so it is recomputed from its saved inputs *with* a graph, back-propagated on its own, and its
//! graph is dropped before the next row is touched:
//!
//! ```text
//! objective_n = local_loss_n + <h_out, g_h> + <cum_out, g_cum>      (g_* = what later rows said about this row's outputs)
//! backward(objective_n)  ->  weight gradients (summed over rows), and new g_h, g_cum for row n-1, and a share of g_x
//! ```
//!
//! `x` (the prelude's output) feeds every row, so its gradient is summed over the rows and carried back through the
//! prelude and the embedding at the end. The result equals the ordinary step's gradient to rounding; the cost is one more
//! forward through each row.

use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};

use super::Model;
use super::forward::{Ctx, attn_block, column, select_rows};
use super::params::TrunkView;
use crate::error::{EngineError, Result};
use crate::moe::{PagedPool, RouteCfg, RouteStats};
use crate::ops;

/// What a checkpointed step reports (the gradients come back separately).
#[derive(Debug, Clone, Default)]
pub struct CkptOut {
    /// The PonderNet loss in nats per character (without the balancing term).
    pub loss: f64,
    /// Expected rows per character.
    pub rows: f32,
    /// Mean halting probability of each row.
    pub row_mass: Vec<f32>,
    pub route: RouteStats,
}

/// What the forward must remember about a row to repeat it exactly.
struct Rec {
    /// The latent state going in, before any backprop truncation.
    h_in: Tensor,
    cum_in: Tensor,
    /// Characters still being computed; `None` on the first row (everything is).
    active: Option<Vec<bool>>,
    /// Backprop from this row stops at its input state (`bptt_window`).
    cut: bool,
    /// Slots that were in reach when the row routed; `None` if the row skipped routing (nothing was active).
    admitted: Option<Vec<bool>>,
}

struct RowArgs<'a> {
    c: &'a Ctx<'a>,
    view: &'a TrunkView,
    route: RouteCfg,
    x: &'a Tensor,
    targets: &'a Tensor,
    n: usize,
    n_steps: usize,
    min_steps: u32,
    /// `ln` of this row's share of the geometric prior over the rows that run.
    ln_prior: f32,
}

struct RowResult {
    h: Tensor,
    cum: Tensor,
    /// `mean_t(p_n * ce_n)`.
    ce_term: Tensor,
    /// `mean_t(P ln(P / prior_n))` with `P = max(p_n, 1e-8)`.
    kl_term: Tensor,
    /// `mean_t(p_n)`, detached.
    mass: Tensor,
    aux: Option<Tensor>,
    stats: RouteStats,
    admitted: Option<Vec<bool>>,
}

/// One recurrent row of the training forward (no caches): mix state with input, attend, route through the experts,
/// freeze the characters that halted, read out, and score. Used untracked for the forward and tracked for the recompute.
#[allow(clippy::too_many_arguments)]
fn row_forward(
    pool: &mut PagedPool,
    a: &RowArgs,
    h_in: &Tensor,
    cum_in: &Tensor,
    active: Option<&[bool]>,
    cut: bool,
    replay: Option<&[bool]>,
) -> Result<RowResult> {
    let dev = h_in.device().clone();
    let t = h_in.dim(0)?;
    let h_in = if cut { h_in.detach() } else { h_in.clone() };
    let any_active = active.is_none_or(|m| m.iter().any(|&b| b));
    let (mut aux, mut stats, mut admitted) = (None, RouteStats::default(), None);
    let h = if any_active {
        let u = ops::linear(&Tensor::cat(&[&h_in, a.x], 1)?, &a.view.adapter)?;
        let xa = attn_block(a.c, &a.view.recur, &u, None)?;
        let xn = ops::rms_norm(&xa, &a.view.recur.ln2, a.c.eps, a.c.kernel)?;
        let partial = active.filter(|m| !m.iter().all(|&b| b));
        let (xr, inv) = match partial {
            Some(m) => {
                let idx: Vec<u32> = m.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u32).collect();
                let it = Tensor::from_slice(&idx, idx.len(), &dev)?;
                let mut inv = vec![u32::MAX; t];
                for (p, &i) in idx.iter().enumerate() {
                    inv[i as usize] = p as u32;
                }
                (xn.index_select(&it, 0)?, Some(Tensor::from_vec(inv, t, &dev)?))
            }
            None => (xn, None),
        };
        let routed = match replay {
            Some(mask) => pool.route_replay(&xr, &a.view.router, &a.view.depth_emb, &a.route, mask)?,
            None => pool.route(&xr, &a.view.router, &a.view.depth_emb, &a.route)?,
        };
        admitted = Some(match replay {
            Some(m) => m.to_vec(),
            None => pool.book.admitted_mask(),
        });
        stats = routed.stats;
        aux = Some(routed.aux);
        let y = match &inv {
            Some(inv) => routed.y.index_select(inv, 0)?,
            None => routed.y,
        };
        let hn = (&xa + y)?;
        match partial {
            Some(m) => select_rows(&column(m, &dev)?, &hn, &h_in)?,
            None => hn,
        }
    } else {
        h_in
    };

    let yf = ops::rms_norm(&h, &a.view.ln_f, a.c.eps, a.c.kernel)?;
    let logits = ops::linear(&yf, &a.view.emb)?;
    let mut lam = candle_nn::ops::sigmoid(&ops::linear(&yf, &a.view.halt_w)?.broadcast_add(&a.view.halt_b)?)?;
    if a.n + 1 == a.n_steps {
        lam = lam.ones_like()?;
    } else if (a.n as u32) + 1 < a.min_steps {
        lam = lam.zeros_like()?;
    }
    let p = cum_in.mul(&lam)?;
    let cum = cum_in.mul(&lam.affine(-1.0, 1.0)?)?;
    let p_flat = p.squeeze(1)?;
    let ce = ops::cross_entropy_rows(&logits, a.targets)?;
    let ce_term = p_flat.mul(&ce)?.mean_all()?;
    let pc = p_flat.clamp(1e-8f32, 1e30f32)?;
    let kl_term = pc.mul(&pc.log()?.affine(1.0, -f64::from(a.ln_prior))?)?.mean_all()?;
    Ok(RowResult { h, cum, ce_term, kl_term, mass: p_flat.detach().mean_all()?, aux, stats, admitted })
}

impl Model {
    /// One training step's forward **and** backward with row checkpointing. Returns the loss report and the gradients
    /// of every parameter (the three slot tensors included), ready for the optimiser. `aux_weight` scales the pool's
    /// balancing loss (taken, as always, from the last row that routed).
    pub fn forward_backward(
        &mut self,
        ids: &[u32],
        targets: &[u32],
        n_steps: usize,
        aux_weight: f64,
    ) -> Result<(CkptOut, GradStore)> {
        let t = ids.len();
        let cfg = self.cfg.clone();
        let (d, block) = (cfg.d_model as usize, cfg.block as usize);
        if t == 0 || targets.len() != t {
            return Err(EngineError::other("nothing to read, or targets and inputs differ in length"));
        }
        if t > block {
            return Err(EngineError::Config(format!(
                "reading {t} characters but the position tables were built to {block}. The window may not exceed the \
                 longest context the model was built with."
            )));
        }
        let n_steps = n_steps.clamp(1, cfg.max_steps as usize);
        let dev = self.dev.clone();
        self.pool.book.begin_text(true);

        let cos = self.cos.narrow(0, 0, t)?.contiguous()?;
        let sin = self.sin.narrow(0, 0, t)?.contiguous()?;
        let ctx = Ctx {
            heads: cfg.n_head as usize,
            head_dim: d / cfg.n_head as usize,
            eps: 1e-6,
            kernel: self.opts.kernel,
            mask: &self.mask,
            cos: &cos,
            sin: &sin,
        };
        let route_cfg = |track: bool| RouteCfg {
            top_k: cfg.pool_top_k as usize,
            capacity_factor: self.opts.capacity_factor,
            mode: self.opts.moe_mode,
            z_weight: self.opts.z_weight,
            track,
            kernel: self.opts.kernel,
        };
        let ids_t = Tensor::from_slice(ids, t, &dev)?;
        let targets_t = Tensor::from_slice(targets, t, &dev)?;
        let prior: Vec<f32> =
            (0..n_steps).map(|n| (cfg.halt_prior * (1.0 - cfg.halt_prior).powi(n as i32)) as f32).collect();
        let total: f32 = prior.iter().sum();
        let ln_prior: Vec<f32> = prior.iter().map(|p| (p / total).ln()).collect();
        let (freeze, thresh, bptt) = (cfg.halt_freeze, cfg.halt_thresh as f32, cfg.bptt_window as usize);

        // ---- forward, no graph ---------------------------------------------------------------------------------------
        let view_f = self.trunk.view(false);
        let prelude = |view: &TrunkView| -> Result<Tensor> {
            let mut x = view.emb.index_select(&ids_t, 0)?;
            for (a, m) in view.prelude.iter() {
                x = attn_block(&ctx, a, &x, None)?;
                let xn = ops::rms_norm(&x, &a.ln2, ctx.eps, ctx.kernel)?;
                x = (&x + ops::swiglu(&xn, &m.w1, &m.w3, &m.w2)?)?;
            }
            Ok(x)
        };
        let x_p = prelude(&view_f)?;
        let mut h = Tensor::zeros((t, d), DType::F32, &dev)?;
        let mut cum = Tensor::ones((t, 1), DType::F32, &dev)?;
        let mut halted = vec![false; t];
        let mut records: Vec<Rec> = Vec::with_capacity(n_steps);
        let (mut ce_terms, mut kl_terms, mut masses) = (Vec::new(), Vec::new(), Vec::new());
        let mut last_routed = None;
        let mut route = RouteStats::default();
        #[allow(clippy::needless_range_loop)]
        for n in 0..n_steps {
            let active: Option<Vec<bool>> = (freeze && n > 0).then(|| halted.iter().map(|&b| !b).collect());
            let cut = bptt > 0 && n + bptt < n_steps;
            let args = RowArgs {
                c: &ctx,
                view: &view_f,
                route: route_cfg(false),
                x: &x_p,
                targets: &targets_t,
                n,
                n_steps,
                min_steps: cfg.min_steps,
                ln_prior: ln_prior[n],
            };
            let res = row_forward(&mut self.pool, &args, &h, &cum, active.as_deref(), cut, None)?;
            records.push(Rec { h_in: h.clone(), cum_in: cum.clone(), active, cut, admitted: res.admitted.clone() });
            if res.admitted.is_some() {
                last_routed = Some(n);
            }
            route.tokens += res.stats.tokens;
            route.routed += res.stats.routed;
            route.dropped += res.stats.dropped;
            ce_terms.push(res.ce_term);
            kl_terms.push(res.kl_term);
            masses.push(res.mass);
            if freeze && n + 1 < n_steps {
                let c: Vec<f32> = res.cum.flatten_all()?.to_vec1()?;
                for (hl, c) in halted.iter_mut().zip(c) {
                    *hl |= 1.0 - c >= thresh;
                }
            }
            h = res.h;
            cum = res.cum;
            if self.opts.kernel.sync_tiles {
                dev.synchronize()?;
            }
        }
        let ce_sum: Vec<f32> = Tensor::stack(&ce_terms, 0)?.to_vec1()?;
        let kl_sum: Vec<f32> = Tensor::stack(&kl_terms, 0)?.to_vec1()?;
        let row_mass: Vec<f32> = Tensor::stack(&masses, 0)?.to_vec1()?;
        let loss = f64::from(ce_sum.iter().sum::<f32>()) + cfg.ponder_beta * f64::from(kl_sum.iter().sum::<f32>());
        let rows = row_mass.iter().enumerate().map(|(i, m)| (i as f32 + 1.0) * m).sum();
        drop((h, cum, ce_terms, kl_terms, masses, view_f));

        // ---- backward, one row at a time ------------------------------------------------------------------------------
        let view_t = self.trunk.view(true);
        let vars = self.all_vars();
        let x_leaf = Var::from_tensor(&x_p)?;
        let mut acc: Option<GradStore> = None;
        let merge = |acc: &mut Option<GradStore>, store: GradStore| -> Result<()> {
            match acc {
                None => *acc = Some(store),
                Some(total) => {
                    for v in &vars {
                        if let Some(g) = store.get(v) {
                            let sum = match total.remove(v) {
                                Some(prev) => (prev + g)?.detach(),
                                None => g.detach(),
                            };
                            total.insert(v, sum);
                        }
                    }
                }
            }
            Ok(())
        };
        let mut upstream: Option<(Tensor, Tensor)> = None; // (g_h, g_cum) from the rows after this one
        let mut g_x: Option<Tensor> = None;
        for n in (0..n_steps).rev() {
            let rec = &records[n];
            let (h_leaf, cum_leaf) = (Var::from_tensor(&rec.h_in)?, Var::from_tensor(&rec.cum_in)?);
            let args = RowArgs {
                c: &ctx,
                view: &view_t,
                route: route_cfg(true),
                x: x_leaf.as_tensor(),
                targets: &targets_t,
                n,
                n_steps,
                min_steps: cfg.min_steps,
                ln_prior: ln_prior[n],
            };
            let res = row_forward(
                &mut self.pool,
                &args,
                h_leaf.as_tensor(),
                cum_leaf.as_tensor(),
                rec.active.as_deref(),
                rec.cut,
                rec.admitted.as_deref(),
            )?;
            let mut obj = (&res.ce_term + res.kl_term.affine(cfg.ponder_beta, 0.0)?)?;
            if Some(n) == last_routed
                && aux_weight != 0.0
                && let Some(aux) = &res.aux
            {
                obj = (obj + aux.affine(aux_weight, 0.0)?)?;
            }
            if let Some((gh, gc)) = &upstream {
                obj = (obj + res.h.mul(gh)?.sum_all()?)?;
                obj = (obj + res.cum.mul(gc)?.sum_all()?)?;
            }
            let store = obj.backward()?;
            let gh = match store.get(&h_leaf) {
                Some(g) => g.detach(),
                None => Tensor::zeros((t, d), DType::F32, &dev)?,
            };
            let gc = match store.get(&cum_leaf) {
                Some(g) => g.detach(),
                None => Tensor::zeros((t, 1), DType::F32, &dev)?,
            };
            if let Some(gx) = store.get(&x_leaf) {
                g_x = Some(match g_x.take() {
                    Some(prev) => (prev + gx)?.detach(),
                    None => gx.detach(),
                });
            }
            upstream = Some((gh, gc));
            merge(&mut acc, store)?;
            // let the device hand this row's buffers back before the next row asks for its own
            if self.opts.kernel.sync_tiles {
                dev.synchronize()?;
            }
        }

        // the prelude and the embedding: everything the rows said about x
        if let Some(gx) = g_x {
            let x_out = prelude(&view_t)?;
            let store = x_out.mul(&gx)?.sum_all()?.backward()?;
            merge(&mut acc, store)?;
        }
        let grads = acc.ok_or_else(|| EngineError::other("a step with no rows"))?;
        let out = CkptOut { loss, rows, row_mass, route };
        Ok((out, grads))
    }
}
