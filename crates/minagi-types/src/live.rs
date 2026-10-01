//! Types shared by the store, the app and the UI for runs, live telemetry and chart series.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::config::Preset;
use crate::error::AppError;
use crate::events::{
    CheckpointMeta, EvalResult, GrowthEvent, PlasticityEvent, PoolSnapshot, SampleRound, Schedule, StageInfo,
};
use crate::insight::{Eta, Milestone, Verdict, VerdictReport};
use crate::units::{Chars, Count, Step, UnixMs};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Created,
    Preparing,
    Running,
    Paused,
    Stopping,
    Completed,
    Stopped,
    Failed,
    /// The app was closed or crashed while the run was active.
    Interrupted,
    Imported,
}

impl RunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunState::Created => "created",
            RunState::Preparing => "preparing",
            RunState::Running => "running",
            RunState::Paused => "paused",
            RunState::Stopping => "stopping",
            RunState::Completed => "completed",
            RunState::Stopped => "stopped",
            RunState::Failed => "failed",
            RunState::Interrupted => "interrupted",
            RunState::Imported => "imported",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "created" => RunState::Created,
            "preparing" => RunState::Preparing,
            "running" => RunState::Running,
            "paused" => RunState::Paused,
            "stopping" => RunState::Stopping,
            "completed" => RunState::Completed,
            "stopped" => RunState::Stopped,
            "failed" => RunState::Failed,
            "interrupted" => RunState::Interrupted,
            "imported" => RunState::Imported,
            _ => return None,
        })
    }

    /// True while an engine thread is (or should be) working on the run.
    pub fn is_active(&self) -> bool {
        matches!(self, RunState::Preparing | RunState::Running | RunState::Paused | RunState::Stopping)
    }
}

/// When a run should stop by itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Goal {
    Minutes {
        value: f64,
    },
    Chars {
        value: Chars,
    },
    /// Stop when the held-out score reaches this many bits per character.
    TargetBits {
        value: f64,
    },
    UntilStopped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    pub id: i64,
    pub uid: String,
    pub name: String,
    pub notes: String,
    pub status: RunState,
    pub preset: Preset,
    pub dataset_id: Option<i64>,
    pub dataset_name: Option<String>,
    pub step: Step,
    pub chars_read: Chars,
    pub chars_total: Option<Chars>,
    pub active_ms: Count,
    pub best_heldout_nats: Option<f64>,
    pub last_heldout_nats: Option<f64>,
    pub last_train_nats: Option<f64>,
    pub n_experts: Option<u32>,
    pub verdict: Option<Verdict>,
    pub stage: Option<String>,
    pub backend: Option<String>,
    pub goal: Option<Goal>,
    pub error: Option<String>,
    pub created_at: UnixMs,
    pub started_at: Option<UnixMs>,
    pub ended_at: Option<UnixMs>,
}

/// A once-per-second sample of the training curves, persisted and charted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct TickPoint {
    pub step: Step,
    pub chars: Chars,
    /// Active training time in ms (pauses excluded).
    pub t_ms: Count,
    pub train_nats: f64,
    pub grad_norm: f64,
    pub lr_scale: f64,
    pub lr_effective: f64,
    pub read_cps: f64,
    pub write_cps: Option<f64>,
    pub context_now: u32,
    pub n_experts: u32,
    pub avg_rows: f32,
    pub rep8_pct: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Insight {
    pub report: VerdictReport,
    pub eta: Option<Eta>,
    pub milestone: Milestone,
    /// Latest held-out score in bits per character.
    pub bits_per_char: Option<f64>,
}

/// A point of interest on the timeline (checkpoint, pause, growth, warning...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct RunEvent {
    pub id: i64,
    pub step: Step,
    pub chars: Chars,
    pub at: UnixMs,
    pub kind: String,
    pub expert_uid: Option<u32>,
    pub brake: Option<String>,
    pub payload: Option<String>,
}

/// One ordered stream carries everything for the active run to the UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum LiveMsg {
    Stage {
        run_id: i64,
        info: StageInfo,
        at: UnixMs,
    },
    State {
        run_id: i64,
        state: RunState,
        reason: Option<String>,
    },
    /// At most about 4 per second; tiny.
    Pulse {
        run_id: i64,
        step: Step,
        chars: Chars,
        chars_total: Option<Chars>,
        train_nats_ema: f64,
        read_cps: f64,
        active_ms: Count,
        schedule: Schedule,
    },
    /// At most about 1 per second, averaged.
    Ticks {
        run_id: i64,
        points: Vec<TickPoint>,
    },
    Eval {
        run_id: i64,
        result: EvalResult,
    },
    Samples {
        run_id: i64,
        round: SampleRound,
    },
    Pool {
        run_id: i64,
        snapshot: PoolSnapshot,
    },
    Growth {
        run_id: i64,
        event: GrowthEvent,
    },
    Plasticity {
        run_id: i64,
        event: PlasticityEvent,
    },
    Insight {
        run_id: i64,
        insight: Insight,
    },
    Checkpoint {
        run_id: i64,
        checkpoint: CheckpointMeta,
    },
    Warning {
        run_id: i64,
        code: String,
        message: String,
        hint: Option<String>,
    },
    Error {
        run_id: i64,
        error: AppError,
        fatal: bool,
    },
}

/// Current state of the live feed, returned on (re)subscribe so a reloaded UI recovers immediately.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct LiveSnapshot {
    pub run: Option<RunSummary>,
    pub stage: Option<StageInfo>,
    pub last_eval: Option<EvalResult>,
    pub last_pool: Option<PoolSnapshot>,
    pub insight: Option<Insight>,
    pub schedule: Option<Schedule>,
}

/// What a chart axis measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum XAxis {
    Chars,
    Step,
    Time,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SeriesRequest {
    pub run_id: i64,
    /// Metric names such as `train.nats`, `lr.scale`.
    pub keys: Vec<String>,
    pub x: XAxis,
    pub from: Option<f64>,
    pub to: Option<f64>,
    pub max_points: u32,
}

/// Columnar chart data: parallel arrays, with a min/max envelope per bucket so spikes survive downsampling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SeriesData {
    pub key: String,
    pub x: Vec<f64>,
    pub y: Vec<Option<f64>>,
    pub y_lo: Vec<Option<f64>>,
    pub y_hi: Vec<Option<f64>>,
    /// Steps per bucket (1 = raw).
    pub bucket: u32,
}
