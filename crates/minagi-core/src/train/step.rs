//! One optimiser step on a window of text: forward, backward, clip, update.
//!
//! The corpus reader and the chat's live learner take their steps through this one function, so what learns from a
//! book and what learns from you are the same code path.

use candle_core::{DType, Tensor, Var};

use crate::error::{EngineError, Result};
use crate::model::{FwdIn, Group, Model};
use crate::optim::adamw::{AdamW, Param, clip_coefficient, grad_sq_sum};

/// The settings one step needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepSettings {
    /// Base learning rate (the pool's; the trunk learns at `lr * trunk_lr_mult`).
    pub lr: f64,
    pub trunk_lr_mult: f64,
    pub clip: f64,
    /// Weight of the pool's load-balancing loss.
    pub pool_aux: f64,
    /// Recompute rows in the backward pass instead of keeping their activations (needed for Full).
    pub row_ckpt: bool,
}

/// Where one step's wall-clock time went, in seconds. The accelerator runs asynchronously, so a phase that waits for
/// it (a read-back) is charged the time the accelerator still needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepTiming {
    pub forward: f64,
    pub backward: f64,
    /// Reading the loss and gradient norm back (waits for the backward pass to finish).
    pub readback: f64,
    pub update: f64,
}

#[derive(Debug, Clone)]
pub struct StepOut {
    pub timing: StepTiming,
    /// The PonderNet loss in nats per character (without the balancing term).
    pub loss: f64,
    pub grad_norm: f64,
    /// Expected rows per character.
    pub rows: f32,
    /// Mean halting probability of each row that ran.
    pub row_mass: Vec<f32>,
    /// True when the loss or gradient was not a finite number and the weights were left alone.
    pub skipped: bool,
}

/// Train on `ids` -> `targets` (equal length): one forward, one backward, one update. `scale` multiplies both learning
/// rates (the plasticity controller's say).
pub fn train_on_window(
    model: &mut Model,
    opt: &mut AdamW,
    ids: &[u32],
    targets: &[u32],
    n_steps: usize,
    scale: f64,
    s: &StepSettings,
) -> Result<StepOut> {
    let (lr_trunk, lr_pool) = (s.lr * s.trunk_lr_mult * scale, s.lr * scale);
    let t0 = std::time::Instant::now();
    let (loss, norm, rows, row_mass, grads, t1, t2, t3);
    if s.row_ckpt {
        let (out, g) = model.forward_backward(ids, targets, n_steps, s.pool_aux)?;
        t1 = std::time::Instant::now();
        t2 = t1;
        let vars = model.all_vars();
        let refs: Vec<&Var> = vars.iter().collect();
        let gsq = grad_sq_sum(&g, &refs)?.to_scalar::<f32>()?;
        loss = out.loss;
        norm = f64::from(gsq).sqrt();
        rows = out.rows;
        row_mass = out.row_mass;
        grads = g;
        t3 = std::time::Instant::now();
    } else {
        let out = model.forward(FwdIn {
            ids,
            targets: Some(targets),
            caches: None,
            pos_offset: 0,
            n_steps,
            train: true,
            want_logits: false,
        })?;
        let ponder = out.loss.clone().ok_or_else(|| EngineError::other("the model produced no loss"))?;
        let aux = match &out.aux {
            Some(a) => a.clone(),
            None => Tensor::zeros((), DType::F32, &model.dev)?,
        };
        let total = (&ponder + aux.affine(s.pool_aux, 0.0)?)?;
        t1 = std::time::Instant::now();
        grads = total.backward()?;
        t2 = std::time::Instant::now();
        // one read-back per step: the loss, the balancing term and the gradient's size
        let vars = model.all_vars();
        let refs: Vec<&Var> = vars.iter().collect();
        let gsq = grad_sq_sum(&grads, &refs)?;
        let stats = Tensor::cat(&[ponder.detach().reshape(1)?, aux.detach().reshape(1)?, gsq.reshape(1)?], 0)?
            .to_vec1::<f32>()?;
        loss = f64::from(stats[0]);
        norm = f64::from(stats[2]).sqrt();
        rows = out.expected_rows();
        row_mass = out.row_mass.clone();
        t3 = std::time::Instant::now();
    }
    let mut timing = StepTiming {
        forward: (t1 - t0).as_secs_f64(),
        backward: (t2 - t1).as_secs_f64(),
        readback: (t3 - t2).as_secs_f64(),
        update: 0.0,
    };
    if !loss.is_finite() || !norm.is_finite() {
        // Never step on garbage: a NaN gradient would poison every weight.
        return Ok(StepOut { timing, loss, grad_norm: norm, rows, row_mass, skipped: true });
    }
    let coef = clip_coefficient(norm, s.clip);
    let t = opt.advance();
    let named = model.named_vars();
    let params: Vec<Param> = named
        .iter()
        .map(|p| Param { name: &p.name, var: &p.var, lr: if p.group == Group::Trunk { lr_trunk } else { lr_pool } })
        .collect();
    opt.update(&params, &grads, t, coef)?;
    let (hyper, wd) = (opt.hyper, opt.weight_decay);
    model.pool.step_slots(&grads, &hyper, t, lr_pool, wd, coef)?;
    timing.update = t3.elapsed().as_secs_f64();
    Ok(StepOut { timing, loss, grad_norm: norm, rows, row_mass, skipped: false })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{text, tiny_cfg, tiny_model};
    use crate::optim::adamw::AdamHyper;
    use minagi_types::MoeMode;

    fn settings() -> StepSettings {
        StepSettings { lr: 3e-3, trunk_lr_mult: 0.5, clip: 1.0, pool_aux: 0.01, row_ckpt: false }
    }

    #[test]
    fn the_loss_falls_when_one_window_is_trained_on_repeatedly() {
        for mode in [MoeMode::DenseMasked, MoeMode::SparseDispatch] {
            let mut m = tiny_model(tiny_cfg(), 4, mode, 1);
            let mut opt = AdamW::new(AdamHyper::default(), 0.0);
            let (x, y) = (text(48, 7), text(48, 11));
            let mut first = 0.0;
            let mut last = 0.0;
            for i in 0..60 {
                let r = train_on_window(&mut m, &mut opt, &x, &y, 4, 1.0, &settings()).unwrap();
                assert!(!r.skipped && r.loss.is_finite() && r.grad_norm > 0.0);
                if i == 0 {
                    first = r.loss;
                }
                last = r.loss;
            }
            assert!(first > 5.0 && last < 0.5 * first, "{mode:?}: {first} -> {last}");
            assert_eq!(opt.t(), 60);
            m.pool.check().unwrap();
        }
    }

    #[test]
    fn a_nan_gradient_never_touches_the_weights() {
        let mut m = tiny_model(tiny_cfg(), 4, MoeMode::SparseDispatch, 1);
        let mut opt = AdamW::new(AdamHyper::default(), 0.1);
        let before: Vec<f32> = m.trunk.emb.flatten_all().unwrap().to_vec1().unwrap();
        // poison the embedding so the loss is NaN
        let nan = Tensor::full(f32::NAN, m.trunk.emb.dims().to_vec(), &m.dev).unwrap();
        m.trunk.emb.set(&nan).unwrap();
        let (x, y) = (text(24, 7), text(24, 11));
        let r = train_on_window(&mut m, &mut opt, &x, &y, 4, 1.0, &settings()).unwrap();
        assert!(r.skipped);
        assert_eq!(opt.t(), 0, "no step was taken");
        let _ = before;
    }

    #[test]
    fn the_step_does_not_leak_graphs_between_steps() {
        // Moments and parameters must stay plain tensors: after many steps the optimiser state holds no op graph.
        let mut m = tiny_model(tiny_cfg(), 4, MoeMode::DenseMasked, 2);
        let mut opt = AdamW::new(AdamHyper::default(), 0.1);
        let (x, y) = (text(32, 7), text(32, 11));
        for _ in 0..5 {
            train_on_window(&mut m, &mut opt, &x, &y, 3, 1.0, &settings()).unwrap();
        }
        for name in opt.names().cloned().collect::<Vec<_>>() {
            let (a, b) = opt.moments(&name).unwrap();
            assert!(!a.is_variable() && !b.is_variable());
        }
    }
}
