//! Routing and dispatch: how one recurrent pass sends each active character through its experts.
//!
//! For `x` (`[N, D]`, already normalised) the router scores *every* expert in the pool while the forward may still
//! admit experts (see [`PoolBook`]); once the card is full it scores only the experts on it. A character picks its top-k
//! among the admitted slots by the router's probability (plus the exploration bonus while training), the picks are
//! renormalised, scaled by each expert's gate, and the experts' outputs are summed.
//!
//! Two ways to compute the experts, which give the same result (including which assignments are dropped):
//!
//! - **Dense-masked** (`MoeMode::DenseMasked`, used for Tiny): one wide SwiGLU over all slots, each character's output
//!   scaled by a per-slot gain that is zero for slots it did not pick (or whose assignment the capacity limit dropped).
//! - **Sparse dispatch** (`SparseDispatch`): a Switch-style padded `[slots, capacity, D]` buffer built with
//!   `index_select`, three batched matmuls, and an inverse `index_select`. An expert takes at most
//!   `capacity_factor` times its fair share of the assignments; the rest are dropped (a dropped assignment costs a
//!   character one of its top-k experts, not the character). Which assignments are dropped is decided in token order
//!   (the reference's order is unspecified).
//!
//! The auxiliary load-balancing loss is `sum(frac_top1 * mean_prob) * slots + z * mean(logsumexp(logits)^2)`.

use candle_core::{D, DType, Tensor};
use minagi_types::MoeMode;

use super::pool::PagedPool;
use crate::error::Result;
use crate::ops::{self, Kernel};

/// A very negative score for slots the forward did not admit (finite, so no `inf - inf` can arise).
const NOT_ADMITTED: f32 = -1e30;

#[derive(Debug, Clone, Copy)]
pub struct RouteCfg {
    pub top_k: usize,
    pub capacity_factor: f64,
    pub mode: MoeMode,
    /// Weight of the router z-loss inside the auxiliary loss.
    pub z_weight: f32,
    /// Whether to record a graph for the backward pass (false for evaluation and generation).
    pub track: bool,
    pub kernel: Kernel,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RouteStats {
    /// Characters routed.
    pub tokens: usize,
    /// Expert assignments requested (`tokens * k`).
    pub routed: usize,
    /// Assignments dropped by the capacity limit.
    pub dropped: usize,
}

pub struct RouteOut {
    /// `[N, D]`: the experts' combined output.
    pub y: Tensor,
    /// Scalar auxiliary loss (zero when not tracking).
    pub aux: Tensor,
    pub stats: RouteStats,
}

/// Requests the router would make, summed per expert: every row's top-k probabilities added to its experts.
fn requested(scores: &Tensor, k: usize) -> Result<Vec<f32>> {
    let e = scores.dim(D::Minus1)?;
    let q = ops::softmax_composed(scores)?;
    let (vals, idx) = ops::topk_last(&q, k.min(e))?;
    let (vals, idx) = (vals.to_vec2::<f32>()?, idx.to_vec2::<u32>()?);
    let mut mass = vec![0f32; e];
    for (vr, ir) in vals.iter().zip(&idx) {
        for (&v, &i) in vr.iter().zip(ir) {
            mass[i as usize] += v;
        }
    }
    Ok(mass)
}

/// Host-side dispatch plan: assignments bucketed by expert in token order, each expert keeping at most `cap`.
struct Plan {
    cap: usize,
    /// `[slots * cap]`: token for each padded row, or `u32::MAX` for an empty one.
    gather: Vec<u32>,
    /// `[N * k]`: padded row holding assignment `(token, j)`, or `u32::MAX` if it was dropped.
    inverse: Vec<u32>,
    dropped: usize,
}

fn plan(idx: &[u32], n: usize, k: usize, slots: usize, capacity_factor: f64) -> Plan {
    let mut counts = vec![0usize; slots];
    for &e in idx {
        counts[e as usize] += 1;
    }
    let mut cap = counts.iter().copied().max().unwrap_or(0);
    if capacity_factor > 0.0 && slots > 0 {
        let limit = ((capacity_factor * idx.len() as f64 / slots as f64).ceil() as usize).max(1);
        cap = cap.min(limit);
    }
    let mut fill = vec![0usize; slots];
    let mut gather = vec![u32::MAX; slots * cap];
    let mut inverse = vec![u32::MAX; n * k];
    let mut dropped = 0;
    for (a, &e) in idx.iter().enumerate() {
        let e = e as usize;
        if fill[e] < cap {
            let row = e * cap + fill[e];
            fill[e] += 1;
            gather[row] = (a / k) as u32;
            inverse[a] = row as u32;
        } else {
            dropped += 1;
        }
    }
    Plan { cap, gather, inverse, dropped }
}

impl PagedPool {
    /// Route `flat` (`[N, D]`, `N >= 1`) through the pool. `router` is `[n_experts, D]` and `depth_emb` is `[D]`.
    ///
    /// While the forward may still admit experts this ranks the whole pool and admits (and pages in) the most
    /// requested ones first, which moves data and needs the host, so it is not a pure function of its inputs.
    pub fn route(&mut self, flat: &Tensor, router: &Tensor, depth_emb: &Tensor, cfg: &RouteCfg) -> Result<RouteOut> {
        self.route_inner(flat, router, depth_emb, cfg, None)
    }

    /// Route again exactly as an earlier call did, for recomputing a row in the backward pass: nothing is admitted,
    /// nothing is counted, and the slots admitted at that time (`admitted`, one flag per slot) are the only ones in reach.
    pub fn route_replay(
        &mut self,
        flat: &Tensor,
        router: &Tensor,
        depth_emb: &Tensor,
        cfg: &RouteCfg,
        admitted: &[bool],
    ) -> Result<RouteOut> {
        self.route_inner(flat, router, depth_emb, cfg, Some(admitted))
    }

    fn route_inner(
        &mut self,
        flat: &Tensor,
        router: &Tensor,
        depth_emb: &Tensor,
        cfg: &RouteCfg,
        replay: Option<&[bool]>,
    ) -> Result<RouteOut> {
        let (n_tok, _) = flat.dims2()?;
        let xin = flat.broadcast_add(depth_emb)?;

        let bias_t = match self.book.selection_bias() {
            Some(b) => Some(Tensor::from_slice(b, b.len(), flat.device())?),
            None => None,
        };

        if replay.is_none() && self.book.admitting() {
            // THE SELECTION RULE, while the forward has room on the card: every character ranks the whole pool and asks
            // for its top-k. No gradient: admission decides what is reachable, the routing below decides weights.
            let z = ops::linear(&xin.detach(), &router.detach())?;
            let (mass, merit) = match &bias_t {
                None => (requested(&z, cfg.top_k)?, None),
                // the bonus decides what is admitted; what the router alone would have asked for feeds the prune clock
                Some(b) => (requested(&z.broadcast_add(b)?, cfg.top_k)?, Some(requested(&z, cfg.top_k)?)),
            };
            self.admit(&mass, merit.as_deref())?;
        }

        let r = self.book.resident;
        let rows = self.book.resident_rows();
        let rows_t = Tensor::from_vec(rows.clone(), r, flat.device())?;
        // only the rows belonging to the experts on the card, in slot order: column j of the logits is slot j
        let w = router.index_select(&rows_t, 0)?;
        let admitted = match replay {
            Some(m) => m.to_vec(),
            None => self.book.admitted_mask(),
        };
        let mask_add: Vec<f32> = admitted.iter().map(|&a| if a { 0.0 } else { NOT_ADMITTED }).collect();
        let logits = ops::linear(&xin, &w)?.broadcast_add(&Tensor::from_vec(mask_add, r, flat.device())?)?;
        let probs = ops::softmax(&logits, cfg.kernel)?;
        let k = cfg.top_k.min(r);
        let (wv, idx) = match &bias_t {
            None => {
                let (w, i) = ops::topk_last(&probs, k)?;
                (w, i)
            }
            Some(b) => {
                // chosen by score plus bonus; weighted by the router's own probabilities, so the gradient reaching a
                // chosen expert's row is the same as if it had been chosen on its score alone
                let pick = b.index_select(&rows_t, 0)?;
                let (_, i) = ops::topk_last(&logits.detach().broadcast_add(&pick)?, k)?;
                (probs.gather(&i, 1)?, i)
            }
        };
        let wv = wv.broadcast_div(&wv.sum_keepdim(1)?)?;
        let gate = if cfg.track { self.gate.as_tensor().clone() } else { self.gate.as_tensor().detach() };
        let gsel = gate.index_select(&rows_t, 0)?;
        let idx_flat = idx.flatten_all()?;
        let wv = wv.broadcast_mul(&gsel.index_select(&idx_flat, 0)?.reshape((n_tok, k))?)?;

        // statistics and the host's copy of the choices
        let idx_host: Vec<u32> = idx_flat.to_vec1()?;
        let mut hits = vec![0f32; r];
        let mut top1 = vec![0f32; r];
        for (a, &e) in idx_host.iter().enumerate() {
            hits[e as usize] += 1.0;
            if a % k == 0 {
                top1[e as usize] += 1.0;
            }
        }
        if replay.is_none() {
            self.book.note_use(&hits);
            self.book.last_pick =
                idx_host[(n_tok - 1) * k..].iter().map(|&slot| self.book.uid[rows[slot as usize] as usize]).collect();
        }

        let aux = if cfg.track {
            let frac = Tensor::from_vec(top1.iter().map(|c| c / n_tok as f32).collect::<Vec<_>>(), r, flat.device())?;
            let balance = (probs.mean(0)?.mul(&frac)?).sum_all()?.affine(r as f64, 0.0)?;
            let z = ops::logsumexp_keepdim(&logits)?.sqr()?.mean_all()?.affine(cfg.z_weight as f64, 0.0)?;
            (balance + z)?
        } else {
            Tensor::zeros((), DType::F32, flat.device())?
        };

        let (w1, w3, w2) = if cfg.track {
            (self.w1.as_tensor().clone(), self.w3.as_tensor().clone(), self.w2.as_tensor().clone())
        } else {
            (self.w1.as_tensor().detach(), self.w3.as_tensor().detach(), self.w2.as_tensor().detach())
        };
        let (d_model, d_ff) = (self.d_model, self.d_ff);
        let mut stats = RouteStats { tokens: n_tok, routed: n_tok * k, dropped: 0 };

        let y = match cfg.mode {
            MoeMode::DenseMasked => {
                // the same capacity rule as sparse dispatch, applied as a keep mask on the gains, so both modes drop
                // exactly the same assignments and compute the same function
                let p = plan(&idx_host, n_tok, k, r, cfg.capacity_factor);
                stats.dropped = p.dropped;
                let wv = if p.dropped > 0 {
                    let keep: Vec<f32> = p.inverse.iter().map(|&row| if row == u32::MAX { 0.0 } else { 1.0 }).collect();
                    wv.mul(&Tensor::from_vec(keep, (n_tok, k), flat.device())?)?
                } else {
                    wv
                };
                let gains = Tensor::zeros((n_tok, r), DType::F32, flat.device())?.scatter(&idx, &wv, 1)?;
                let a = ops::linear(flat, &w1.reshape((r * d_ff, d_model))?)?;
                let b = ops::linear(flat, &w3.reshape((r * d_ff, d_model))?)?;
                let h = (candle_nn::ops::silu(&a)? * b)?;
                let h = h.reshape((n_tok, r, d_ff))?.broadcast_mul(&gains.unsqueeze(2)?)?.reshape((n_tok, r * d_ff))?;
                let w2t = w2.transpose(1, 2)?.contiguous()?.reshape((r * d_ff, d_model))?;
                h.matmul(&w2t)?
            }
            MoeMode::SparseDispatch => {
                let p = plan(&idx_host, n_tok, k, r, cfg.capacity_factor);
                stats.dropped = p.dropped;
                if p.cap == 0 {
                    Tensor::zeros((n_tok, d_model), DType::F32, flat.device())?
                } else {
                    let gather = Tensor::from_vec(p.gather, r * p.cap, flat.device())?;
                    let inverse = Tensor::from_vec(p.inverse, n_tok * k, flat.device())?;
                    let buf = flat.index_select(&gather, 0)?.reshape((r, p.cap, d_model))?;
                    let a = buf.matmul(&w1.transpose(1, 2)?)?;
                    let b = buf.matmul(&w3.transpose(1, 2)?)?;
                    let h = (candle_nn::ops::silu(&a)? * b)?;
                    let y = h.matmul(&w2.transpose(1, 2)?)?.reshape((r * p.cap, d_model))?;
                    let yt = y.index_select(&inverse, 0)?.reshape((n_tok, k, d_model))?;
                    yt.broadcast_mul(&wv.unsqueeze(2)?)?.sum(1)?
                }
            }
        };
        Ok(RouteOut { y, aux, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::store::{ExpertEntry, ExpertStore, MemStore};
    use crate::ops::testutil::{max_abs_diff, randn};
    use candle_core::{Device, Var};

    const DM: usize = 16;
    const DF: usize = 12;

    fn fresh_pool(n: usize, resident: usize, seed: u64) -> PagedPool {
        let mut store = MemStore::new();
        for i in 0..n {
            let mut r = crate::rng::HostRng::new(seed + i as u64);
            store
                .put(
                    i as u64,
                    ExpertEntry {
                        w1: r.normal_vec(DF * DM, 0.3),
                        w3: r.normal_vec(DF * DM, 0.3),
                        w2: r.normal_vec(DM * DF, 0.3),
                        moments: None,
                    },
                    false,
                )
                .unwrap();
        }
        PagedPool::new(&Device::Cpu, Box::new(store), DM, DF, n, resident).unwrap()
    }

    fn cfg(mode: MoeMode, cap: f64) -> RouteCfg {
        RouteCfg { top_k: 2, capacity_factor: cap, mode, z_weight: 1e-3, track: true, kernel: Kernel::plain() }
    }

    struct Run {
        y: Tensor,
        aux: f32,
        grads: Vec<Tensor>,
        stats: RouteStats,
    }

    fn run(mode: MoeMode, cap: f64, n_tok: usize, explore: bool) -> Run {
        let d = Device::Cpu;
        let mut p = fresh_pool(8, 4, 100);
        if explore {
            p.book.explore_bias = 0.65;
        }
        p.book.begin_text(explore);
        let x = Var::from_tensor(&randn(&[n_tok, DM], 1.0, 1, &d)).unwrap();
        let router = Var::from_tensor(&randn(&[8, DM], 0.4, 2, &d)).unwrap();
        let depth = Var::from_tensor(&randn(&[DM], 0.1, 3, &d)).unwrap();
        let out = p.route(x.as_tensor(), router.as_tensor(), depth.as_tensor(), &cfg(mode, cap)).unwrap();
        let rr = randn(&[n_tok, DM], 1.0, 4, &d);
        let loss = (out.y.mul(&rr).unwrap().sum_all().unwrap() + out.aux.affine(5.0, 0.0).unwrap()).unwrap();
        let g = loss.backward().unwrap();
        let mut grads = vec![
            g.get(&x).unwrap().clone(),
            g.get(&router).unwrap().clone(),
            g.get(&depth).unwrap().clone(),
            g.get(&p.gate).unwrap().clone(),
        ];
        for v in p.slot_vars() {
            grads.push(g.get(v).unwrap().clone());
        }
        Run { y: out.y.clone(), aux: crate::ops::scalar(&out.aux).unwrap(), grads, stats: out.stats }
    }

    #[test]
    fn dense_masked_and_sparse_dispatch_agree_without_drops() {
        for explore in [false, true] {
            let a = run(MoeMode::DenseMasked, 0.0, 40, explore);
            let b = run(MoeMode::SparseDispatch, 0.0, 40, explore);
            assert_eq!(b.stats.dropped, 0);
            assert!(max_abs_diff(&a.y, &b.y) < 1e-4, "outputs differ (explore {explore})");
            assert!((a.aux - b.aux).abs() < 1e-5);
            for (i, (x, y)) in a.grads.iter().zip(&b.grads).enumerate() {
                let scale = y.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().max(1e-6);
                assert!(max_abs_diff(x, y) / scale < 1e-3, "gradient {i} differs (explore {explore})");
            }
        }
    }

    #[test]
    fn every_parameter_receives_a_gradient_and_nothing_is_nan() {
        for mode in [MoeMode::DenseMasked, MoeMode::SparseDispatch] {
            let r = run(mode, 1.5, 40, true);
            for (i, g) in r.grads.iter().enumerate() {
                let v = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(v.iter().all(|x| x.is_finite()), "gradient {i} has non-finite values");
                assert!(v.iter().any(|x| *x != 0.0), "gradient {i} is all zero ({mode:?})");
            }
        }
    }

    #[test]
    fn capacity_drops_are_counted_and_cut_the_output() {
        let free = run(MoeMode::SparseDispatch, 0.0, 64, false);
        let tight = run(MoeMode::SparseDispatch, 0.6, 64, false);
        assert_eq!(free.stats.routed, 64 * 2);
        assert!(tight.stats.dropped > 0, "a limit of 0.6x the fair share must drop something");
        assert!(max_abs_diff(&free.y, &tight.y) > 1e-4);
        // each expert keeps at most ceil(0.6 * 128 / 4) = 20 assignments, so at least 128 - 4*20 are dropped
        assert!(tight.stats.dropped >= 128 - 4 * 20);
    }

    #[test]
    fn only_admitted_slots_are_chosen_and_the_card_holds_what_was_requested() {
        let d = Device::Cpu;
        let mut p = fresh_pool(8, 4, 5);
        p.book.begin_text(false);
        let x = randn(&[30, DM], 1.0, 7, &d);
        let router = randn(&[8, DM], 0.4, 8, &d);
        let out = p
            .route(&x, &router, &Tensor::zeros(DM, DType::F32, &d).unwrap(), &cfg(MoeMode::SparseDispatch, 0.0))
            .unwrap();
        assert_eq!(out.stats.tokens, 30);
        let on_card = p.book.experts_in_slots();
        assert_eq!(on_card.len(), 4);
        // the experts on the card are the four the text asked for most
        let z = x.matmul(&router.t().unwrap()).unwrap();
        let mass = requested(&z, 2).unwrap();
        let mut order: Vec<usize> = (0..8).collect();
        order.sort_by(|&a, &b| mass[b].partial_cmp(&mass[a]).unwrap());
        let mut want = order[..4].to_vec();
        want.sort();
        let mut got = on_card.clone();
        got.sort();
        assert_eq!(got, want);
        // the usage counters landed on experts, not slots
        assert!((p.book.use_.iter().sum::<f64>() - 60.0).abs() < 1e-9);
        assert!(p.book.use_.iter().enumerate().all(|(e, &u)| u == 0.0 || on_card.contains(&e)));
    }

    #[test]
    fn a_gated_off_expert_contributes_nothing() {
        let d = Device::Cpu;
        let mut p = fresh_pool(4, 4, 9);
        p.gate = Var::from_tensor(&Tensor::zeros(4, DType::F32, &d).unwrap()).unwrap();
        p.book.begin_text(false);
        let x = randn(&[10, DM], 1.0, 1, &d);
        let router = randn(&[4, DM], 0.4, 2, &d);
        let out =
            p.route(&x, &router, &Tensor::zeros(DM, DType::F32, &d).unwrap(), &cfg(MoeMode::DenseMasked, 0.0)).unwrap();
        assert!(out.y.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() == 0.0);
    }

    #[test]
    fn untracked_routing_builds_no_graph() {
        let d = Device::Cpu;
        let mut p = fresh_pool(4, 4, 9);
        p.book.begin_text(false);
        let x = randn(&[10, DM], 1.0, 1, &d);
        let router = randn(&[4, DM], 0.4, 2, &d);
        let mut c = cfg(MoeMode::SparseDispatch, 0.0);
        c.track = false;
        let out = p.route(&x, &router, &Tensor::zeros(DM, DType::F32, &d).unwrap(), &c).unwrap();
        let reach = out.y.sum_all().unwrap().backward().unwrap();
        assert!(p.slot_vars().iter().all(|v| reach.get(v).is_none()) && reach.get(&p.gate).is_none());
    }

    #[test]
    fn plan_buckets_in_token_order_and_respects_capacity() {
        // 6 tokens, k = 1, 2 slots: tokens 0..4 want slot 0, tokens 4..6 want slot 1; capacity limit ceil(0.5*6/2) = 2
        let p = plan(&[0, 0, 0, 0, 1, 1], 6, 1, 2, 0.5);
        assert_eq!(p.cap, 2);
        assert_eq!(p.gather, vec![0, 1, 4, 5], "the first two tokens of slot 0 are kept, in order");
        assert_eq!(p.inverse, vec![0, 1, u32::MAX, u32::MAX, 2, 3]);
        assert_eq!(p.dropped, 2);
        let free = plan(&[0, 0, 0, 0, 1, 1], 6, 1, 2, 0.0);
        assert_eq!((free.cap, free.dropped), (4, 0));
    }
}
