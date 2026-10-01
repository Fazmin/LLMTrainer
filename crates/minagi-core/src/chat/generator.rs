//! A trained model that writes text on request, and can learn from the conversation on its own copy.
//!
//! The generator loads a checkpoint into a private working directory (hard links to the checkpoint's expert files, so
//! it costs no space), so writing never touches the original and learning only ever changes this copy. `save_adapted`
//! writes the copy out as a new checkpoint.

use std::path::{Path, PathBuf};
use std::time::Instant;

use minagi_text::{ByteTokenizer, SPECIALS, Utf8Stream};
use minagi_types::{
    AppError, AppResult, BackendKind, Chars, CheckpointKind, CheckpointMeta, Count, DecodingConfig, GenChar,
    GenRequest, GenStats, Generator, GeneratorInfo, LearnReport, Preset, Step,
};

use super::decode::Decode;
use super::live::LiveLearner;
use super::writer::Writer;
use crate::backend;
use crate::error::{EngineError, Result};
use crate::model::{FwdIn, Model, ModelOpts};
use crate::ops::{self, Kernel};
use crate::optim::adamw::AdamW;
use crate::train::session::{RunPaths, Saved, load_model, preset_of, read_checkpoint, write_checkpoint};
use crate::train::step::StepSettings;

/// The token that ends a document; writing stops when the model produces it.
fn end_of_text() -> u32 {
    u32::from(minagi_text::MARKER_BASE) + SPECIALS.iter().position(|m| *m == "<|endoftext|>").unwrap_or(8) as u32
}

pub struct ModelGenerator {
    model: Model,
    opt: AdamW,
    learner: LiveLearner,
    decoding: DecodingConfig,
    info: GeneratorInfo,
    saved: Saved,
    preset: Preset,
    work: PathBuf,
}

impl Drop for ModelGenerator {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.work);
    }
}

fn unique_dir(base: &Path) -> PathBuf {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    base.join(format!(".chat-{}-{n}", std::process::id()))
}

impl ModelGenerator {
    pub fn open(checkpoint: &Path, backend_kind: BackendKind) -> Result<Self> {
        let ck = read_checkpoint(checkpoint)?;
        let parsed = ck
            .data
            .manifest
            .parsed_cfg()
            .map_err(|e| EngineError::Checkpoint(format!("the checkpoint's settings are unreadable: {e}")))?
            .ok_or_else(|| EngineError::Checkpoint("the checkpoint does not say what shape the model has".into()))?;
        let preset = preset_of(&parsed.model);
        let mut train = preset.train();
        if let Some(r) = parsed.resident() {
            train.resident = r.max(1);
        }
        train.ram_cache = train.ram_cache.min(32);
        // the working copy lives next to the checkpoint's run (same filesystem, so experts are hard-linked)
        let base = checkpoint.parent().and_then(|p| p.parent()).unwrap_or(checkpoint);
        let work = unique_dir(base);
        let dev = backend::device(backend_kind)?;
        let kernel = Kernel::hand();
        let opts = ModelOpts::from_train(&train, kernel);
        let (mut model, opt, saved) = match load_model(&dev, &ck, &train, opts, &RunPaths::new(&work)) {
            Ok(x) => x,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&work);
                return Err(e);
            }
        };
        model.pool.book.explore_bias = 0.0; // never explore while writing; learning turns nothing on either
        let chunk = (train.chunk as usize).min(model.cfg.block as usize);
        let settings = StepSettings { lr: 3e-4, trunk_lr_mult: 0.2, clip: train.clip, pool_aux: 0.0, row_ckpt: false };
        let learner = LiveLearner::new(chunk, model.cfg.block as usize, settings, 1);
        let params = model.total_params() as f64;
        let resident_params = (model.trunk.n_params() + model.pool.vram_params()) as f64;
        let info = GeneratorInfo {
            label: format!("{} model, step {}", preset.display_name(), saved.step),
            backend: backend_kind,
            supports_learn: true,
            // weights alone to write; weights, gradients and Adam's two moments to learn
            needs_gb: (resident_params * 4.0 * 4.0 + (params - resident_params).max(0.0) * 0.0) / 1e9,
        };
        Ok(Self { model, opt, learner, decoding: train.decoding.clone(), info, saved, preset, work })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        ByteTokenizer::new().encode_str(text).into_iter().map(u32::from).collect()
    }

    /// Mean nats per character of `ids` under the model as it is now (no learning).
    fn score(&mut self, ids: &[u32]) -> Result<f64> {
        let block = self.model.cfg.block as usize;
        let ids = &ids[ids.len().saturating_sub(block)..];
        if ids.len() < 2 {
            return Ok(f64::NAN);
        }
        let n_steps = self.model.cfg.max_steps as usize;
        let model = &mut self.model;
        let loss = backend::pool(|| -> Result<f32> {
            let out = model.forward(FwdIn {
                ids: &ids[..ids.len() - 1],
                targets: Some(&ids[1..]),
                caches: None,
                pos_offset: 0,
                n_steps,
                train: false,
                want_logits: false,
            })?;
            Ok(ops::scalar(out.loss.as_ref().ok_or_else(|| EngineError::other("no loss"))?)?)
        })?;
        Ok(f64::from(loss))
    }
}

fn app(e: EngineError) -> AppError {
    e.to_app(None)
}

impl Generator for ModelGenerator {
    fn info(&self) -> GeneratorInfo {
        self.info.clone()
    }

    fn generate(&mut self, req: &GenRequest, out: &mut dyn FnMut(GenChar) -> bool) -> AppResult<GenStats> {
        let started = Instant::now();
        let prompt = self.encode(&req.prompt);
        if prompt.is_empty() {
            return Err(AppError::Invalid("There is nothing to continue: the prompt is empty.".into()));
        }
        let decode = if req.adapt { Decode::adapted(&self.decoding) } else { Decode::RAW };
        let mut writer = Writer::new(&self.model, &prompt);
        let mut stream = Utf8Stream::new();
        let mut text = String::new();
        let mut written = 0u64;
        let eot = end_of_text();
        for _ in 0..req.max_new {
            let w = writer.next(&mut self.model, &decode).map_err(app)?;
            written += 1;
            let piece = stream.push(w.token as u16);
            text.push_str(&piece);
            let keep_going = out(GenChar {
                text: piece,
                rows: w.rows,
                experts: w.experts.iter().take(8).map(|&u| u.min(u64::from(u16::MAX)) as u16).collect(),
            });
            if !keep_going || w.token == eot {
                break;
            }
            if req.stop_at.as_deref().is_some_and(|s| !s.is_empty() && text.ends_with(s)) {
                break;
            }
        }
        let tail = stream.finish();
        if !tail.is_empty() {
            let _ = out(GenChar { text: tail, rows: 1, experts: Vec::new() });
        }
        let seconds = started.elapsed().as_secs_f64();
        Ok(GenStats { chars: Count(written), seconds, chars_per_sec: written as f64 / seconds.max(1e-9) })
    }

    fn learn(&mut self, text: &str) -> AppResult<LearnReport> {
        let ids = self.encode(text);
        let before = self.score(&ids).map_err(app)?;
        let ModelGenerator { model, opt, learner, .. } = self;
        learner.feed(model, opt, &ids).map_err(app)?;
        let after = self.score(&ids).map_err(app)?;
        Ok(LearnReport { nats_before: before, nats_after: after })
    }

    fn save_adapted(&mut self, dest: &Path) -> AppResult<CheckpointMeta> {
        let mut saved = self.saved.clone();
        saved.step += self.learner.steps;
        saved.val = None;
        let report = write_checkpoint(&mut self.model, &self.opt, &saved, Some(self.preset), dest).map_err(app)?;
        Ok(CheckpointMeta {
            step: Step(saved.step),
            chars: Chars(saved.chars),
            kind: CheckpointKind::Manual,
            path: dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            bytes: Count(report.total_bytes),
            heldout_nats: None,
            n_experts: report.n_experts as u32,
            engine_format: crate::store::manifest::RS_FORMAT,
        })
    }
}
