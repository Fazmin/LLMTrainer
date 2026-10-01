//! The seam between the app and the engines.
//!
//! The real engine (`minagi-core`) and the simulated one (`minagi-mock`) implement the same traits, so the whole
//! app can be built and tested without compiling candle, and a contract test suite keeps them honest.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::api::ImportPreview;
use crate::config::{ModelConfig, TrainConfig};
use crate::error::{AppError, AppResult};
use crate::events::{CheckpointMeta, Ctl, EngineEvent, Outcome};
use crate::hardware::{BackendInfo, BackendKind, DataStats, Estimate, HardwareInfo};
use crate::units::Count;

/// Receives engine events. Implementations must never block the engine.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: EngineEvent);
}

impl EventSink for flume::Sender<EngineEvent> {
    fn emit(&self, event: EngineEvent) {
        // The channel is unbounded; a closed receiver just means nobody is listening any more.
        let _ = self.send(event);
    }
}

/// Everything an engine needs to start (or resume) a run.
#[derive(Debug, Clone)]
pub struct RunSpec {
    /// Directory the run owns: live weights, checkpoints, logs.
    pub run_dir: PathBuf,
    /// Contains `train/<lane>/**` and `val/<domain>/**`.
    pub data_root: PathBuf,
    /// Lanes to read, by name. `None` reads every lane under `train/`.
    pub lanes: Option<Vec<String>>,
    pub model: ModelConfig,
    pub train: TrainConfig,
    pub seed: u64,
    /// Checkpoint directory to continue from, if any.
    pub resume_from: Option<PathBuf>,
    pub backend: BackendKind,
}

/// A training engine. `run` blocks the calling (dedicated) thread until the run ends.
pub trait Engine: Send {
    fn run(&mut self, spec: RunSpec, ctl: flume::Receiver<Ctl>, sink: &dyn EventSink) -> Outcome;
}

/// Creates engines and answers capability questions.
pub trait EngineFactory: Send + Sync {
    fn name(&self) -> &'static str;
    /// True for the simulated engine; the UI shows an unmistakable badge.
    fn is_mock(&self) -> bool;
    fn version(&self) -> String;
    fn probe_backends(&self) -> Vec<BackendInfo>;
    fn estimate(
        &self,
        model: &ModelConfig,
        train: &TrainConfig,
        hw: &HardwareInfo,
        data: Option<&DataStats>,
    ) -> Estimate;
    fn new_engine(&self) -> Box<dyn Engine>;
    fn open_generator(&self, checkpoint: &Path, backend: BackendKind) -> AppResult<Box<dyn Generator>>;

    /// Look at a weights folder written by the original Python program and say whether (and how) it can be imported.
    fn inspect_python(&self, _src: &Path) -> AppResult<ImportPreview> {
        Err(AppError::Invalid("This version of the app cannot open models made by the original program.".into()))
    }

    /// Write the checkpoint at `checkpoint` as `model.safetensors` inside `dest_dir`; returns its size in bytes.
    fn export_safetensors(&self, _checkpoint: &Path, _dest_dir: &Path) -> AppResult<u64> {
        Err(AppError::Invalid("This version of the app cannot export models for other tools.".into()))
    }

    /// Convert a Python weights folder into this engine's checkpoint at `dst` (which must not exist).
    fn import_python(&self, _src: &Path, _dst: &Path) -> AppResult<ImportPreview> {
        Err(AppError::Invalid("This version of the app cannot open models made by the original program.".into()))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct GenRequest {
    pub prompt: String,
    pub max_new: u32,
    /// Repetition trace on ("adapted" decoding).
    pub adapt: bool,
    /// Stop when the output ends with this text (e.g. `</bot>`).
    pub stop_at: Option<String>,
}

/// One generated piece of text plus what the model did to produce it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct GenChar {
    /// UTF-8 text completed by this byte (may be empty mid-character).
    pub text: String,
    /// Recurrent rows used for this character.
    pub rows: u8,
    /// The most-used expert uids for this character.
    pub experts: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct GenStats {
    pub chars: Count,
    pub seconds: f64,
    pub chars_per_sec: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct LearnReport {
    pub nats_before: f64,
    pub nats_after: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct GeneratorInfo {
    pub label: String,
    pub backend: BackendKind,
    pub supports_learn: bool,
    pub needs_gb: f64,
}

/// Streams text from a trained model, optionally learning from the conversation on its own copy.
pub trait Generator: Send {
    fn info(&self) -> GeneratorInfo;
    /// Calls `out` for each generated byte; returning `false` from `out` cancels.
    fn generate(&mut self, req: &GenRequest, out: &mut dyn FnMut(GenChar) -> bool) -> AppResult<GenStats>;
    fn learn(&mut self, text: &str) -> AppResult<LearnReport>;
    fn save_adapted(&mut self, dest: &Path) -> AppResult<CheckpointMeta>;
}
