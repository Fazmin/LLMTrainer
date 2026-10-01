//! Writing one character at a time.
//!
//! The prompt is one forward at position 0, which admits the experts its characters ask for most; then every character
//! of the reply is a forward of its own and routes among what the prompt and the reply so far have voted for. When the
//! attention window fills, the context is re-read from position 0 (a new text, whose vote is the re-read context's).

use super::decode::{Decode, pick_next};
use crate::backend;
use crate::error::Result;
use crate::model::{Caches, FwdIn, Model};

/// One written character and what the model did to produce it.
#[derive(Debug, Clone, PartialEq)]
pub struct Written {
    pub token: u32,
    /// The row at which the character halted (how long the model "thought").
    pub rows: u8,
    /// Uids of the experts that character's last row picked.
    pub experts: Vec<u64>,
}

pub struct Writer {
    caches: Caches,
    offset: usize,
    history: Vec<u32>,
    cur: Vec<u32>,
    block: usize,
}

impl Writer {
    /// Start writing after `prompt` (only the last `block` tokens are kept). Nothing is computed until [`next`].
    pub fn new(model: &Model, prompt: &[u32]) -> Self {
        let block = model.cfg.block as usize;
        let start = prompt.len().saturating_sub(block);
        let history = prompt[start..].to_vec();
        Self { caches: model.new_caches(), offset: 0, cur: history.clone(), history, block }
    }

    /// Everything written so far, prompt included.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Run the model on what it has not read yet and choose the next token.
    pub fn next(&mut self, model: &mut Model, decode: &Decode) -> Result<Written> {
        if self.offset + self.cur.len() > self.block {
            self.caches = model.new_caches();
            let keep = self.block / 2;
            self.cur = self.history[self.history.len().saturating_sub(keep)..].to_vec();
            self.offset = 0;
        }
        let n_steps = model.cfg.max_steps as usize;
        let (logits, rows) = backend::pool(|| -> Result<(Vec<f32>, f32)> {
            let out = model.forward(FwdIn {
                ids: &self.cur,
                targets: None,
                caches: Some(&mut self.caches),
                pos_offset: self.offset,
                n_steps,
                train: false,
                want_logits: false,
            })?;
            let all = out.logits.ok_or_else(|| crate::EngineError::other("the model produced no logits"))?;
            let t = all.dim(0)?;
            let last: Vec<f32> = all.narrow(0, t - 1, 1)?.flatten_all()?.to_vec1()?;
            Ok((last, out.steps.last().copied().unwrap_or(1.0)))
        })?;
        self.offset += self.cur.len();
        let token = pick_next(&logits, &self.history, decode);
        self.history.push(token);
        self.cur = vec![token];
        Ok(Written { token, rows: rows.round().clamp(1.0, 255.0) as u8, experts: model.pool.book.last_pick.clone() })
    }

    /// Write `n` tokens.
    pub fn write(&mut self, model: &mut Model, n: usize, decode: &Decode) -> Result<Vec<u32>> {
        (0..n).map(|_| self.next(model, decode).map(|w| w.token)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{text, tiny_cfg, tiny_model};
    use minagi_types::MoeMode;

    fn model() -> Model {
        let mut cfg = tiny_cfg();
        cfg.pool_top_k = cfg.pool_experts; // every expert on the card for every character: no admission effects
        let m = tiny_model(cfg, 8, MoeMode::SparseDispatch, 21);
        m.trunk.halt_b.set(&candle_core::Tensor::from_vec(vec![0.0f32], 1, &m.dev).unwrap()).unwrap();
        m
    }

    #[test]
    fn cached_writing_equals_rereading_the_whole_history_each_time() {
        let mut m = model();
        let prompt: Vec<u32> = text(12, 7);
        let mut w = Writer::new(&m, &prompt);
        let cached = w.write(&mut m, 10, &Decode::RAW).unwrap();
        // the slow way: a fresh forward over everything so far, take the last logits
        let mut hist = prompt.clone();
        for &want in &cached {
            let out = m
                .forward(FwdIn {
                    ids: &hist,
                    targets: None,
                    caches: None,
                    pos_offset: 0,
                    n_steps: 4,
                    train: false,
                    want_logits: false,
                })
                .unwrap();
            let all = out.logits.unwrap();
            let t = all.dim(0).unwrap();
            let last: Vec<f32> = all.narrow(0, t - 1, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let tok = pick_next(&last, &hist, &Decode::RAW);
            assert_eq!(tok, want);
            hist.push(tok);
        }
    }

    #[test]
    fn writing_is_deterministic_and_reports_rows_and_experts() {
        let mut m = model();
        let prompt: Vec<u32> = text(8, 5);
        let a = Writer::new(&m, &prompt).write(&mut m, 6, &Decode::RAW).unwrap();
        let b = Writer::new(&m, &prompt).write(&mut m, 6, &Decode::RAW).unwrap();
        assert_eq!(a, b);
        let mut w = Writer::new(&m, &prompt);
        let step = w.next(&mut m, &Decode::RAW).unwrap();
        assert!((1..=4).contains(&step.rows));
        assert!(!step.experts.is_empty() && step.experts.iter().all(|&u| u < 8));
        assert_eq!(w.history().len(), prompt.len() + 1);
    }

    #[test]
    fn a_full_window_is_reread_from_the_start() {
        let mut m = model(); // block 64
        let prompt: Vec<u32> = text(60, 7);
        let mut w = Writer::new(&m, &prompt);
        let out = w.write(&mut m, 12, &Decode::RAW).unwrap(); // crosses the 64-token limit
        assert_eq!(out.len(), 12);
        assert!(w.history().len() == 72);
    }

    #[test]
    fn the_adaptation_trace_changes_what_a_looping_model_writes() {
        let mut m = model();
        let prompt: Vec<u32> = text(8, 5);
        let raw = Writer::new(&m, &prompt).write(&mut m, 30, &Decode::RAW).unwrap();
        let adapted = Writer::new(&m, &prompt)
            .write(&mut m, 30, &Decode { adapt_strength: 2.5, adapt_decay: 0.88, rep_penalty: 1.0 })
            .unwrap();
        let distinct = |v: &[u32]| v.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct(&adapted) >= distinct(&raw), "adaptation should not make writing more repetitive");
    }
}
