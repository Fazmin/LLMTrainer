//! Learning from a conversation while it happens.
//!
//! **One stream.** The model reads everything it is given and everything it writes as a single continuous stream of
//! characters, and takes an optimiser step every `chunk` of them: the same mechanism the corpus reader uses, with the
//! interleaving between subjects removed because a conversation is one subject. What the model sees at every step is
//! the last `context` characters, and the gradient reaches all of them.
//!
//! This is learning partly from the model's own output, and reading a few hundred characters of chat under many
//! gradient steps is a far denser diet than a corpus: it can cost the model skill on other text. The chat screen says
//! so, and only ever applies this to a *copy* of the model.

use crate::error::Result;
use crate::model::{Model, sample_depth};
use crate::optim::adamw::AdamW;
use crate::rng::HostRng;
use crate::train::step::{StepOut, StepSettings, train_on_window};

pub struct LiveLearner {
    chunk: usize,
    context: usize,
    /// The last `context` token ids and nothing older.
    buf: Vec<u32>,
    /// Characters read since the last step.
    pending: usize,
    settings: StepSettings,
    rng: HostRng,
    /// Optimiser steps taken in this conversation.
    pub steps: u64,
}

impl LiveLearner {
    pub fn new(chunk: usize, context: usize, settings: StepSettings, seed: u64) -> Self {
        Self {
            chunk: chunk.max(1),
            context: context.max(8),
            buf: Vec::new(),
            pending: 0,
            settings,
            rng: HostRng::fork(seed, 7),
            steps: 0,
        }
    }

    /// Add `ids` to the stream and take a step for every `chunk` characters; returns one record per step taken (none
    /// for a short message, several for a long one).
    pub fn feed(&mut self, model: &mut Model, opt: &mut AdamW, ids: &[u32]) -> Result<Vec<StepOut>> {
        let mut recs = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            let take = (self.chunk - self.pending).min(ids.len() - i);
            self.buf.extend_from_slice(&ids[i..i + take]);
            self.pending += take;
            i += take;
            if self.pending >= self.chunk {
                // trim to the window BEFORE the step: what the model sees is the last `context` characters
                self.trim();
                if let Some(r) = self.step(model, opt)? {
                    recs.push(r);
                }
                self.pending = 0;
            }
        }
        self.trim();
        Ok(recs)
    }

    fn trim(&mut self) {
        if self.buf.len() > self.context {
            let cut = self.buf.len() - self.context;
            self.buf.drain(..cut);
        }
    }

    fn step(&mut self, model: &mut Model, opt: &mut AdamW) -> Result<Option<StepOut>> {
        if self.buf.len() < 8 {
            return Ok(None);
        }
        let n_steps = sample_depth(&model.cfg, true, &mut self.rng);
        let (x, y) = (&self.buf[..self.buf.len() - 1], &self.buf[1..]);
        let out = crate::backend::pool(|| train_on_window(model, opt, x, y, n_steps, 1.0, &self.settings))?;
        if !out.skipped {
            self.steps += 1;
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{text, tiny_cfg, tiny_model};
    use crate::optim::adamw::AdamHyper;
    use minagi_types::MoeMode;

    #[test]
    fn a_step_is_taken_every_chunk_and_the_window_slides() {
        let mut m = tiny_model(tiny_cfg(), 4, MoeMode::DenseMasked, 1);
        let mut opt = AdamW::new(AdamHyper::default(), 0.1);
        let s = StepSettings { lr: 1e-3, trunk_lr_mult: 0.2, clip: 1.0, pool_aux: 0.0, row_ckpt: false };
        let mut live = LiveLearner::new(16, 40, s, 1);
        assert!(live.feed(&mut m, &mut opt, &text(10, 3)).unwrap().is_empty(), "a short message takes no step");
        let recs = live.feed(&mut m, &mut opt, &text(30, 5)).unwrap();
        assert_eq!(recs.len(), 2, "10 + 30 = 40 characters: steps at 16 and 32");
        assert_eq!(live.steps, 2);
        assert_eq!(live.buf.len(), 40);
        live.feed(&mut m, &mut opt, &text(30, 7)).unwrap();
        assert_eq!(live.buf.len(), 40, "text older than the window leaves it");
        assert!(recs.iter().all(|r| r.loss.is_finite() && !r.skipped));
    }
}
