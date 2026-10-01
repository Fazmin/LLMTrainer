//! Events the engine emits while it runs, and the control messages it accepts.
//!
//! Metrics are in nats per character (natural-log cross-entropy); the UI converts to bits per character.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::error::AppError;
use crate::units::{Chars, Count, Step, UnixMs};

/// Coarse phase of a run, shown in the "What's happening now" banner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Idle,
    PreparingData,
    CreatingModel,
    LoadingCheckpoint,
    /// The main loop: read a chunk, guess each next character, learn from the mistakes.
    Reading,
    Evaluating,
    Sampling,
    GrowPrune,
    Checkpointing,
    Paused,
    Stopping,
    Finished,
    Failed,
}

impl Stage {
    /// Stages after which the run can no longer be resumed live (it is over or must be restarted).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Stage::Finished | Stage::Failed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub done: f64,
    pub total: Option<f64>,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct StageInfo {
    pub stage: Stage,
    pub detail: Option<String>,
    pub progress: Option<Progress>,
}

/// When the next scheduled things will happen, so the UI can say "next check-up in 3:12".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Schedule {
    pub next_eval_chars: Option<Chars>,
    pub next_sample_ms: Option<Count>,
    pub next_checkpoint_ms: Option<Count>,
    pub next_growth_chars: Option<Chars>,
}

/// One optimizer step. Cheap; emitted every step and coalesced by the recorder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct StepMetrics {
    pub step: Step,
    pub chars_read: Chars,
    pub train_nats: f64,
    pub grad_norm: f64,
    pub clip: f64,
    /// Plasticity multiplier applied to the base learning rate.
    pub lr_scale: f64,
    pub lr_trunk: f64,
    pub lr_experts: f64,
    pub context_now: u32,
    pub read_cps: f64,
    pub write_cps: Option<f64>,
    /// Average number of recurrent rows used per character ("thinking depth").
    pub avg_rows: f32,
    /// Histogram of rows used (index = rows - 1). Not sent every step.
    pub halt_hist: Option<Vec<f32>>,
    pub rep8_pct: Option<f32>,
    pub ctx_gain: Option<f64>,
    pub n_experts: u32,
    pub step_ms: f32,
    pub schedule: Schedule,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct DomainScore {
    pub domain: String,
    pub nats: f64,
    pub se: f64,
    pub n_chars: Count,
}

/// A held-out evaluation round: the model is scored on text it has never trained on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct EvalResult {
    pub step: Step,
    pub chars: Chars,
    pub overall_nats: f64,
    pub overall_se: f64,
    pub train_nats_ema: f64,
    pub domains: Vec<DomainScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SampleItem {
    pub domain: String,
    pub prompt: String,
    /// Plain greedy continuation.
    pub raw: String,
    /// Greedy continuation with the repetition trace switched on.
    pub adapted: String,
    pub raw_rep8: Option<f32>,
    pub adapted_rep8: Option<f32>,
    pub note: Option<String>,
}

/// What the model writes for each prompt at a point in training ("watch it learn to write").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SampleRound {
    pub step: Step,
    pub chars: Chars,
    pub items: Vec<SampleItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ExpertInfo {
    pub uid: u32,
    /// GPU slot when resident.
    pub slot: Option<u8>,
    pub gate: f32,
    /// Share of routing this expert received recently (0..1).
    pub use_share: f32,
    pub admits: u32,
    pub age_chars: Chars,
    pub on_trial: bool,
    /// Fraction of the survival window it has gone unused (0..1).
    pub staleness: f32,
    pub dying: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Brake {
    pub ok: bool,
    pub why: String,
}

/// The five conditions that must hold before the pool may grow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct BrakeReport {
    pub room: Brake,
    pub used: Brake,
    pub earning: Brake,
    pub fits: Brake,
    pub honest: Brake,
}

impl BrakeReport {
    pub fn all_ok(&self) -> bool {
        self.room.ok && self.used.ok && self.earning.ok && self.fits.ok && self.honest.ok
    }
}

/// State of the expert pool at a moment in training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct PoolSnapshot {
    pub step: Step,
    pub chars: Chars,
    pub n_experts: u32,
    pub resident: Vec<ExpertInfo>,
    /// Routing share of every expert in the pool, busiest first, quantised to 0..=255.
    pub usage: Vec<u8>,
    pub brakes: BrakeReport,
    /// Histogram of rows used per character.
    pub halting_hist: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GrowthEvent {
    Born { uid: u32, parents: Vec<u32> },
    Pruned { uid: u32, reason: String },
    Blocked { brakes: BrakeReport },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct PlasticityEvent {
    pub rate_scale: f64,
    pub evidence_t: f64,
    pub reason: String,
    pub jumped: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    Auto,
    Manual,
    Best,
    Final,
    Stop,
    Imported,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointMeta {
    pub step: Step,
    pub chars: Chars,
    pub kind: CheckpointKind,
    /// Directory name relative to the run directory.
    pub path: String,
    pub bytes: Count,
    pub heldout_nats: Option<f64>,
    pub n_experts: u32,
    pub engine_format: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct MemoryReport {
    pub resident_gb: f64,
    pub ram_gb: f64,
    pub disk_gb: f64,
    pub gpu_budget_gb: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Outcome {
    /// The training goal or the data ran out.
    Completed,
    /// The user stopped it.
    Stopped,
    Failed {
        error: AppError,
    },
}

/// Everything the engine can tell the app. The engine never blocks on delivering these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineEvent {
    Stage(StageInfo),
    Step(StepMetrics),
    Eval(EvalResult),
    Samples(SampleRound),
    Pool(PoolSnapshot),
    Growth(GrowthEvent),
    Plasticity(PlasticityEvent),
    ContextGrow { from: u32, to: u32 },
    CheckpointSaved(CheckpointMeta),
    Memory(MemoryReport),
    Warn(Diagnostic),
    Finished(Outcome),
}

/// Control messages sent to a running engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Ctl {
    Pause,
    Resume,
    /// Stop; with `save` the engine writes a checkpoint first.
    Stop {
        save: bool,
    },
    CheckpointNow,
    SampleNow,
    EvalNow,
    SetSampleEveryMin {
        minutes: f64,
    },
    SetCheckpointEveryMin {
        minutes: f64,
    },
}

/// Timestamp helper used when events are persisted.
pub fn now_ms() -> UnixMs {
    UnixMs::now()
}
