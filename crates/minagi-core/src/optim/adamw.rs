//! AdamW with decoupled weight decay, built from plain tensor ops, plus global gradient clipping.
//!
//! Matches `torch.optim.AdamW` (`betas` as configured, `eps = 1e-8`, no amsgrad): with step count `t`,
//! `p <- p * (1 - lr * wd) - lr / (1 - b1^t) * m / (sqrt(v / (1 - b2^t)) + eps)`. One global step count is shared by
//! every parameter, as in the reference (all parameters take a gradient on every step).
//!
//! Two things matter on candle and are easy to get wrong:
//!
//! - The carried moments **must be detached** each step. candle has no `no_grad`, so a moment tensor built from the
//!   previous one and the gradient keeps the whole op graph alive; the spike reached ~15 GB after 30 steps.
//! - Parameters are updated in place (`Var::set`) so every tensor that shares their storage sees the new values.
//!
//! The expert pool's three slot tensors are updated with the same [`adam_update`], but their moments belong to the
//! *experts* (they travel with them between slots), so the pool owns them; see `moe::pool`.

use std::collections::BTreeMap;

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamHyper {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Default for AdamHyper {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.95, eps: 1e-8 }
    }
}

/// One AdamW update of `p` from gradient `g` and moments `(m, v)`; returns the new detached moments.
///
/// `gscale` multiplies the gradient first (the clipping coefficient), `lr` is the already-scaled learning rate.
#[allow(clippy::too_many_arguments)]
pub fn adam_update(
    p: &Var,
    m: &Tensor,
    v: &Tensor,
    g: &Tensor,
    h: &AdamHyper,
    t: u64,
    lr: f64,
    wd: f64,
    gscale: f64,
) -> Result<(Tensor, Tensor)> {
    let g = g.detach();
    let (c1, c2) = (1.0 - h.beta1.powi(t as i32), 1.0 - h.beta2.powi(t as i32));
    let m = (m.affine(h.beta1, 0.0)? + g.affine((1.0 - h.beta1) * gscale, 0.0)?)?;
    let v = (v.affine(h.beta2, 0.0)? + g.sqr()?.affine((1.0 - h.beta2) * gscale * gscale, 0.0)?)?;
    let denom = (v.affine(1.0 / c2, 0.0)?.sqrt()? + h.eps)?;
    let upd = (m.affine(lr / c1, 0.0)? / denom)?;
    let new = (p.as_tensor().affine(1.0 - lr * wd, 0.0)? - upd)?;
    p.set(&new)?;
    Ok((m.detach(), v.detach()))
}

/// A parameter to update this step.
pub struct Param<'a> {
    pub name: &'a str,
    pub var: &'a Var,
    /// Learning rate for this parameter (base rate of its group times the plasticity scale).
    pub lr: f64,
}

#[derive(Debug)]
pub struct AdamW {
    pub hyper: AdamHyper,
    pub weight_decay: f64,
    t: u64,
    state: BTreeMap<String, (Tensor, Tensor)>,
}

impl AdamW {
    pub fn new(hyper: AdamHyper, weight_decay: f64) -> Self {
        Self { hyper, weight_decay, t: 0, state: BTreeMap::new() }
    }

    /// Optimiser steps taken so far (the Adam bias-correction counter).
    pub fn t(&self) -> u64 {
        self.t
    }

    pub fn set_t(&mut self, t: u64) {
        self.t = t;
    }

    /// Begin a step: returns the new step count, to be used for every update in this step (including the pool's).
    pub fn advance(&mut self) -> u64 {
        self.t += 1;
        self.t
    }

    /// Update `params` that have a gradient (the others are skipped, as torch does). `t` comes from [`advance`].
    pub fn update(&mut self, params: &[Param], grads: &GradStore, t: u64, gscale: f64) -> Result<()> {
        for p in params {
            let Some(g) = grads.get(p.var) else { continue };
            let (m, v) = match self.state.remove(p.name) {
                Some(s) => s,
                None => (p.var.zeros_like()?, p.var.zeros_like()?),
            };
            let (m, v) = adam_update(p.var, &m, &v, g, &self.hyper, t, p.lr, self.weight_decay, gscale)?;
            self.state.insert(p.name.to_string(), (m, v));
        }
        Ok(())
    }

    pub fn moments(&self, name: &str) -> Option<&(Tensor, Tensor)> {
        self.state.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.state.keys()
    }

    /// Install moments (restoring a checkpoint).
    pub fn put_moments(&mut self, name: &str, m: Tensor, v: Tensor) {
        self.state.insert(name.to_string(), (m.detach(), v.detach()));
    }

    pub fn drop_moments(&mut self, name: &str) {
        self.state.remove(name);
    }

    /// The parameter gained `extra` rows (growth): its moments gain zero rows to match.
    pub fn grow_rows(&mut self, name: &str, extra: usize) -> Result<()> {
        if let Some((m, v)) = self.state.remove(name) {
            let pad = |t: &Tensor| -> Result<Tensor> {
                let mut dims = t.dims().to_vec();
                dims[0] = extra;
                Tensor::cat(&[t, &Tensor::zeros(dims, t.dtype(), t.device())?], 0)?.contiguous()
            };
            self.state.insert(name.to_string(), (pad(&m)?, pad(&v)?));
        }
        Ok(())
    }

    /// The parameter kept only the rows at `keep` (pruning): its moments follow.
    pub fn select_rows(&mut self, name: &str, keep: &[usize]) -> Result<()> {
        if let Some((m, v)) = self.state.remove(name) {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx = Tensor::from_vec(idx, keep.len(), m.device())?;
            self.state.insert(
                name.to_string(),
                (m.index_select(&idx, 0)?.contiguous()?, v.index_select(&idx, 0)?.contiguous()?),
            );
        }
        Ok(())
    }
}

/// Sum of squares of the gradients of `vars` as one device scalar (zero if none has a gradient).
pub fn grad_sq_sum(grads: &GradStore, vars: &[&Var]) -> Result<Tensor> {
    let mut parts = Vec::with_capacity(vars.len());
    let mut dev = None;
    for v in vars {
        dev = Some(v.device().clone());
        if let Some(g) = grads.get(v) {
            parts.push(g.sqr()?.sum_all()?.reshape(1)?);
        }
    }
    match (parts.is_empty(), dev) {
        (false, _) => Tensor::cat(&parts, 0)?.sum_all(),
        (true, Some(d)) => Tensor::zeros((), candle_core::DType::F32, &d),
        (true, None) => Tensor::zeros((), candle_core::DType::F32, &candle_core::Device::Cpu),
    }
}

/// The multiplier torch's `clip_grad_norm_` applies: `max_norm / (norm + 1e-6)`, at most 1.
pub fn clip_coefficient(norm: f64, max_norm: f64) -> f64 {
    if !norm.is_finite() || max_norm <= 0.0 {
        return 1.0;
    }
    (max_norm / (norm + 1e-6)).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::randn;
    use candle_core::Device;

    /// A reference AdamW on host vectors, written straight from the torch documentation.
    #[allow(clippy::too_many_arguments)]
    fn reference(p: &mut [f64], m: &mut [f64], v: &mut [f64], g: &[f64], h: &AdamHyper, t: u64, lr: f64, wd: f64) {
        for i in 0..p.len() {
            p[i] *= 1.0 - lr * wd;
            m[i] = h.beta1 * m[i] + (1.0 - h.beta1) * g[i];
            v[i] = h.beta2 * v[i] + (1.0 - h.beta2) * g[i] * g[i];
            let c1 = 1.0 - h.beta1.powi(t as i32);
            let c2 = 1.0 - h.beta2.powi(t as i32);
            p[i] -= lr / c1 * m[i] / ((v[i] / c2).sqrt() + h.eps);
        }
    }

    #[test]
    fn matches_the_torch_formula_over_several_steps() {
        let d = Device::Cpu;
        let var = Var::from_tensor(&randn(&[3, 5], 1.0, 1, &d)).unwrap();
        let mut p: Vec<f64> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|&x| x as f64).collect();
        let (mut m, mut v) = (vec![0.0; 15], vec![0.0; 15]);
        let mut opt = AdamW::new(AdamHyper::default(), 0.1);
        for step in 1..=6u64 {
            // gradient of 0.5*|p - target|^2 with a moving target, built as a real autograd graph
            let target = randn(&[3, 5], 1.0, 100 + step, &d);
            let loss = (var.as_tensor() - &target).unwrap().sqr().unwrap().sum_all().unwrap().affine(0.5, 0.0).unwrap();
            let grads = loss.backward().unwrap();
            let g: Vec<f64> = grads
                .get(&var)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|&x| x as f64)
                .collect();
            let lr = 0.01 * step as f64;
            let t = opt.advance();
            assert_eq!(t, step);
            opt.update(&[Param { name: "w", var: &var, lr }], &grads, t, 1.0).unwrap();
            reference(&mut p, &mut m, &mut v, &g, &opt.hyper, t, lr, 0.1);
            let got: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
            for (a, b) in got.iter().zip(&p) {
                assert!((*a as f64 - b).abs() < 1e-5, "step {step}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn gradient_scale_is_equivalent_to_scaling_the_gradient() {
        let d = Device::Cpu;
        let mk = || Var::from_tensor(&randn(&[4], 1.0, 5, &d)).unwrap();
        let (a, b) = (mk(), mk());
        let grads_of = |v: &Var, k: f64| {
            let l = v.as_tensor().sqr().unwrap().sum_all().unwrap().affine(k, 0.0).unwrap();
            l.backward().unwrap()
        };
        let mut o1 = AdamW::new(AdamHyper::default(), 0.0);
        let mut o2 = AdamW::new(AdamHyper::default(), 0.0);
        let t = o1.advance();
        o2.advance();
        o1.update(&[Param { name: "a", var: &a, lr: 0.1 }], &grads_of(&a, 1.0), t, 0.25).unwrap();
        o2.update(&[Param { name: "b", var: &b, lr: 0.1 }], &grads_of(&b, 0.25), t, 1.0).unwrap();
        let (x, y): (Vec<f32>, Vec<f32>) = (a.to_vec1().unwrap(), b.to_vec1().unwrap());
        for (p, q) in x.iter().zip(&y) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn parameters_without_a_gradient_are_skipped() {
        let d = Device::Cpu;
        let used = Var::from_tensor(&randn(&[3], 1.0, 1, &d)).unwrap();
        let unused = Var::from_tensor(&randn(&[3], 1.0, 2, &d)).unwrap();
        let before: Vec<f32> = unused.to_vec1().unwrap();
        let grads = used.as_tensor().sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let mut opt = AdamW::new(AdamHyper::default(), 0.1);
        let t = opt.advance();
        opt.update(
            &[Param { name: "u", var: &used, lr: 0.1 }, Param { name: "x", var: &unused, lr: 0.1 }],
            &grads,
            t,
            1.0,
        )
        .unwrap();
        assert_eq!(unused.to_vec1::<f32>().unwrap(), before, "no gradient: not even weight decay");
        assert!(opt.moments("x").is_none() && opt.moments("u").is_some());
    }

    #[test]
    fn moments_do_not_hold_the_graph() {
        // After a step the stored moments must be plain tensors: no op, not a variable.
        let d = Device::Cpu;
        let var = Var::from_tensor(&randn(&[3], 1.0, 1, &d)).unwrap();
        let grads = var.as_tensor().sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let mut opt = AdamW::new(AdamHyper::default(), 0.0);
        let t = opt.advance();
        opt.update(&[Param { name: "w", var: &var, lr: 0.1 }], &grads, t, 1.0).unwrap();
        let (m, v) = opt.moments("w").unwrap();
        assert!(!m.is_variable() && !v.is_variable());
        // a loss built from the moments must not reach back to the parameter through a retained graph
        let reach = (m + v).unwrap().sum_all().unwrap().backward().unwrap();
        assert!(reach.get(&var).is_none(), "the moments still hold the op graph of the step that made them");
    }

    #[test]
    fn rows_follow_growth_and_pruning() {
        let d = Device::Cpu;
        let mut opt = AdamW::new(AdamHyper::default(), 0.0);
        let m = Tensor::from_vec((0..6).map(|i| i as f32).collect::<Vec<_>>(), (3, 2), &d).unwrap();
        opt.put_moments("r", m.clone(), m.affine(2.0, 0.0).unwrap());
        opt.grow_rows("r", 2).unwrap();
        let (gm, _) = opt.moments("r").unwrap();
        assert_eq!(gm.dims(), &[5, 2]);
        assert_eq!(gm.to_vec2::<f32>().unwrap()[4], vec![0.0, 0.0]);
        opt.select_rows("r", &[0, 2, 4]).unwrap();
        let (sm, sv) = opt.moments("r").unwrap();
        assert_eq!(sm.to_vec2::<f32>().unwrap(), vec![vec![0.0, 1.0], vec![4.0, 5.0], vec![0.0, 0.0]]);
        assert_eq!(sv.to_vec2::<f32>().unwrap()[1], vec![8.0, 10.0]);
    }

    #[test]
    fn clipping_matches_torch() {
        assert!((clip_coefficient(10.0, 1.0) - 1.0 / (10.0 + 1e-6)).abs() < 1e-12);
        assert_eq!(clip_coefficient(0.5, 1.0), 1.0, "never scales up");
        assert_eq!(clip_coefficient(f64::NAN, 1.0), 1.0);
        let d = Device::Cpu;
        let a = Var::from_tensor(&Tensor::from_vec(vec![3f32, 4.0], 2, &d).unwrap()).unwrap();
        let b = Var::from_tensor(&Tensor::from_vec(vec![12f32], 1, &d).unwrap()).unwrap();
        let loss = (a.as_tensor().sqr().unwrap().sum_all().unwrap().affine(0.5, 0.0).unwrap()
            + b.as_tensor().sqr().unwrap().sum_all().unwrap().affine(0.5, 0.0).unwrap())
        .unwrap();
        let grads = loss.backward().unwrap(); // gradients equal the values: norm = sqrt(9+16+144) = 13
        let n = grad_sq_sum(&grads, &[&a, &b]).unwrap().to_scalar::<f32>().unwrap().sqrt();
        assert!((n - 13.0).abs() < 1e-4);
    }
}
