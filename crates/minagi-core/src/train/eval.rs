//! Held-out evaluation: how well the model predicts text it has never trained on.
//!
//! Each domain's files are read in order, a chunk at a time through the attention caches (a cached chunk sees exactly
//! the attention context a whole-window pass would, for a fraction of the work). Every chunk's score is the PonderNet
//! loss in nats per character. The standard error is estimated over *windows* of chunks, because chunks inside one
//! window are correlated: the sample that matters is the window, not the chunk.

use minagi_text::EvalSet;
use minagi_types::{Count, DomainScore};

use crate::backend;
use crate::error::{EngineError, Result};
use crate::model::{FwdIn, Model};
use crate::ops;

pub struct Evaluator {
    set: EvalSet,
    chunk: usize,
    /// The longest context a cache may hold before it is reset (the model's block).
    context: usize,
}

#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub domains: Vec<DomainScore>,
    /// Mean of the per-domain means: every domain counts equally.
    pub overall_nats: f64,
    pub overall_se: f64,
}

/// An evaluator looked at through a particular context window (see [`Evaluator::with_context`]).
pub struct EvalView<'a> {
    ev: &'a Evaluator,
    context: usize,
}

impl EvalView<'_> {
    pub fn run(&self, model: &mut Model, eval_chars: usize, stop: &dyn Fn() -> bool) -> Result<EvalOutcome> {
        self.ev.run_at(model, eval_chars, self.context, stop)
    }
}

/// Standard error of the mean of `a`, estimated over windows of `per` chunks (numpy `std`, ddof 1 over windows; or
/// the plain std over chunks when there are fewer than two windows).
pub fn window_se(a: &[f64], per: usize) -> f64 {
    let per = per.max(1);
    let usable = a.len() / per * per;
    if usable >= 2 * per {
        let wm: Vec<f64> = a[..usable].chunks(per).map(|w| w.iter().sum::<f64>() / per as f64).collect();
        let n = wm.len() as f64;
        let mean = wm.iter().sum::<f64>() / n;
        let var = wm.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt() / n.sqrt()
    } else if a.is_empty() {
        0.0
    } else {
        let n = a.len() as f64;
        let mean = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt() / n.sqrt()
    }
}

impl Evaluator {
    pub fn open(set: EvalSet, chunk: usize, context: usize) -> Self {
        Self { set, chunk, context }
    }

    /// Score at a different context window than this evaluator was opened with: what the model is being trained at
    /// right now is the window its score should describe (a window longer than it has ever read at would measure
    /// positions it was never taught).
    pub fn with_context(&self, context: usize) -> EvalView<'_> {
        EvalView { ev: self, context: context.max(self.chunk) }
    }

    pub fn domains(&self) -> Vec<String> {
        self.set.domains()
    }

    /// Score every domain on about `eval_chars` characters at the context this evaluator was opened with. `stop` is
    /// polled between chunks; if it returns true the evaluation is abandoned with [`EngineError::Cancelled`].
    pub fn run(&self, model: &mut Model, eval_chars: usize, stop: &dyn Fn() -> bool) -> Result<EvalOutcome> {
        self.run_at(model, eval_chars, self.context, stop)
    }

    fn run_at(
        &self,
        model: &mut Model,
        eval_chars: usize,
        context: usize,
        stop: &dyn Fn() -> bool,
    ) -> Result<EvalOutcome> {
        let per = (context / self.chunk).max(1);
        let n_steps = model.cfg.max_steps as usize;
        let mut domains = Vec::new();
        let mut every: Vec<f64> = Vec::new();
        for (di, name) in self.set.domains().into_iter().enumerate() {
            let chunks = self.set.chunks(di, self.chunk, eval_chars)?;
            let mut losses = Vec::with_capacity(chunks.len());
            let mut caches = model.new_caches();
            let mut seen = 0usize;
            let mut fresh = true;
            let mut tokens = 0usize;
            for c in &chunks {
                if stop() {
                    return Err(EngineError::Cancelled);
                }
                let n = c.x.len();
                if c.new_text || seen + n > context {
                    caches = model.new_caches();
                    seen = 0;
                    fresh = true;
                }
                let (x, y): (Vec<u32>, Vec<u32>) =
                    (c.x.iter().map(|&t| t as u32).collect(), c.y.iter().map(|&t| t as u32).collect());
                let loss = backend::pool(|| -> Result<f64> {
                    let out = model.forward(FwdIn {
                        ids: &x,
                        targets: Some(&y),
                        caches: Some(&mut caches),
                        pos_offset: seen,
                        n_steps,
                        train: false,
                        want_logits: false,
                    })?;
                    Ok(ops::scalar(out.loss.as_ref().ok_or_else(|| EngineError::other("no loss"))?)? as f64)
                })?;
                let _ = fresh;
                fresh = false;
                seen += n;
                tokens += n;
                if loss.is_finite() {
                    losses.push(loss);
                } else {
                    losses.push(f64::NAN);
                }
            }
            if losses.is_empty() {
                continue;
            }
            let finite: Vec<f64> = losses.iter().copied().filter(|l| l.is_finite()).collect();
            let nats =
                if finite.len() == losses.len() { finite.iter().sum::<f64>() / finite.len() as f64 } else { f64::NAN };
            domains.push(DomainScore {
                domain: name,
                nats,
                se: window_se(&finite, per),
                n_chars: Count(tokens as u64),
            });
            every.extend(finite);
        }
        if domains.is_empty() {
            return Err(EngineError::Text(minagi_text::TextError::NoText("the held-out text".into())));
        }
        let overall = domains.iter().map(|d| d.nats).sum::<f64>() / domains.len() as f64;
        Ok(EvalOutcome { domains, overall_nats: overall, overall_se: window_se(&every, per) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_standard_error_follows_the_reference() {
        // 8 chunks in windows of 2: window means 1.5, 3.5, 5.5, 7.5; sd (ddof 1) = 2.582; se = sd / 2
        let a: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let se = window_se(&a, 2);
        assert!((se - (2.58198889747161 / 2.0)).abs() < 1e-9, "{se}");
        // too few windows: plain std over chunks / sqrt(n)
        let b = [1.0, 3.0];
        assert!((window_se(&b, 4) - (1.0 / 2f64.sqrt())).abs() < 1e-9);
        assert_eq!(window_se(&[], 4), 0.0);
        // a trailing partial window is ignored
        assert_eq!(window_se(&[1.0, 1.0, 2.0, 2.0, 99.0], 2), window_se(&[1.0, 1.0, 2.0, 2.0], 2));
    }
}
