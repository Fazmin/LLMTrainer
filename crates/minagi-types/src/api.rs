//! Request and response types for the Tauri commands that are not part of the engine contract.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::config::{ModelConfig, Preset, TrainConfig};
use crate::events::CheckpointKind;
use crate::live::Goal;
use crate::units::{Chars, Count, Step, UnixMs};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct AppInfo {
    pub version: String,
    pub engine_name: String,
    pub engine_version: String,
    /// True when the simulated engine is active (the UI shows an unmistakable badge).
    pub is_mock: bool,
    pub data_dir: String,
    pub schema_version: u32,
}

/// A preset's full configuration, for the Setup screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct PresetConfig {
    pub preset: Preset,
    pub model: ModelConfig,
    pub train: TrainConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct CreateRunRequest {
    pub name: Option<String>,
    pub preset: Preset,
    /// Overrides for the preset's configuration (advanced settings). `None` uses the preset as is.
    pub model: Option<ModelConfig>,
    pub train: Option<TrainConfig>,
    pub dataset_id: Option<i64>,
    pub goal: Option<Goal>,
}

/// Continue a run's saved model as a new run, optionally with different training settings or different text.
/// The model's shape is fixed by the save; everything about *how it trains* may change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ForkRunRequest {
    pub run_id: i64,
    /// A specific save; the best or latest one when omitted.
    pub checkpoint_id: Option<i64>,
    pub name: Option<String>,
    /// The text to read; the original run's text when omitted.
    pub dataset_id: Option<i64>,
    /// New training settings; the original run's when omitted.
    pub train: Option<TrainConfig>,
    pub goal: Option<Goal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointInfo {
    pub id: i64,
    pub step: Step,
    pub chars: Chars,
    pub kind: CheckpointKind,
    pub path: String,
    pub bytes: Count,
    pub heldout_nats: Option<f64>,
    pub n_experts: Option<u32>,
    pub is_best: bool,
    pub pinned: bool,
    pub created_at: UnixMs,
}

/// A point on the training timeline (a step and the characters read by then).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct StepAt {
    pub step: Step,
    pub chars: Chars,
}

// ───────── chat ─────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ChatMode {
    /// The model continues whatever you type, like a story opening.
    Continue,
    /// Your text is wrapped as a chat turn and the model answers it.
    Conversation,
}

impl ChatMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatMode::Continue => "continue",
            ChatMode::Conversation => "conversation",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "continue" => Some(ChatMode::Continue),
            "conversation" => Some(ChatMode::Conversation),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ChatOpenRequest {
    pub run_id: i64,
    /// A specific save to chat with; the latest save of the run when omitted.
    pub checkpoint_id: Option<i64>,
    pub mode: ChatMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ChatSessionInfo {
    pub id: i64,
    pub title: String,
    pub run_id: Option<i64>,
    /// Frozen at creation, e.g. "Tiny run 1, after 5.4M characters".
    pub model_label: String,
    pub mode: ChatMode,
    pub learn_enabled: bool,
    pub supports_learn: bool,
    /// Memory this chat needs, in GB (the app warns when training and chat will not both fit).
    pub needs_gb: f64,
    pub has_adapted_copy: bool,
    pub created_at: UnixMs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    User,
    Model,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub id: i64,
    pub role: ChatRole,
    pub content: String,
    pub learned: bool,
    pub learn_nats_before: Option<f64>,
    pub learn_nats_after: Option<f64>,
    /// Recurrent rows used for each generated character (model messages only).
    pub rows: Vec<u8>,
    pub chars_per_sec: Option<f64>,
    pub created_at: UnixMs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ChatParams {
    pub max_new: u32,
    /// Repetition guard on.
    pub adapt: bool,
}

/// What streams back while the model writes a reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum ChatEvent {
    /// The model is being loaded from its save.
    Loading,
    /// A piece of the reply, with what the model did to produce it.
    Chars {
        text: String,
        rows: Vec<u8>,
        experts: Vec<u16>,
    },
    /// The model learned from this exchange (only when learning is on).
    Learned {
        nats_before: f64,
        nats_after: f64,
    },
    Done {
        message_id: i64,
        chars: Count,
        chars_per_sec: f64,
    },
    Error {
        error: crate::error::AppError,
    },
}

// ───────── background jobs ─────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Paused,
    Done,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Paused => "paused",
            JobState::Done => "done",
            JobState::Failed => "failed",
            JobState::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "queued" => JobState::Queued,
            "running" => JobState::Running,
            "paused" => JobState::Paused,
            "done" => JobState::Done,
            "failed" => JobState::Failed,
            "cancelled" => JobState::Cancelled,
            _ => return None,
        })
    }

    pub fn is_finished(&self) -> bool {
        matches!(self, JobState::Done | JobState::Failed | JobState::Cancelled)
    }
}

/// A long-running task (scanning a folder, downloading, generating text) the UI can follow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct JobInfo {
    pub id: String,
    /// `scan`, `download`, `generate`, `prepare`...
    pub kind: String,
    pub state: JobState,
    /// What it is about, e.g. `dataset:12`.
    pub subject: Option<String>,
    /// 0.0..=1.0 when known.
    pub progress: Option<f64>,
    pub message: Option<String>,
    pub error: Option<String>,
    pub created_at: UnixMs,
    pub updated_at: UnixMs,
}

/// Messages on a job's live channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum JobEvent {
    Progress { job_id: String, progress: crate::dataset::JobProgress },
    State { job_id: String, state: JobState, error: Option<String> },
}

// ───────── storage ─────────

/// Where the app's disk space goes, in bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct StorageUsage {
    pub data_dir: String,
    pub datasets_bytes: Count,
    pub runs_bytes: Count,
    pub chats_bytes: Count,
    pub database_bytes: Count,
    pub free_bytes: Count,
    /// Disk used by each run (largest first), so people can see what to delete.
    pub per_run: Vec<RunStorage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct RunStorage {
    pub run_id: i64,
    pub name: String,
    pub bytes: Count,
}

// ───────── export and import ─────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ExportRequest {
    pub run_id: i64,
    /// A specific save; the best or latest one when omitted.
    pub checkpoint_id: Option<i64>,
    /// Folder the export is created inside.
    pub dest_dir: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    /// The folder that was created.
    pub path: String,
    pub bytes: Count,
    pub files: Count,
}

/// Where a model folder offered for import comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ImportKind {
    /// A folder made by this app's Export.
    Portable,
    /// The weights folder of the original Python program (mini-AGI).
    Python,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ImportSeverity {
    /// The model cannot be imported as it is.
    Blocker,
    /// It can be imported, but something changes or is left behind.
    Warning,
    Note,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ImportIssue {
    pub severity: ImportSeverity,
    /// Plain English, ready to show.
    pub message: String,
}

/// What importing a folder would do, shown before anything is copied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ImportPreview {
    pub path: String,
    pub kind: ImportKind,
    pub importable: bool,
    /// One plain-English sentence about the model.
    pub summary: String,
    pub issues: Vec<ImportIssue>,
    /// A suggested name for the new run.
    pub name: String,
    pub preset: Preset,
    pub model: Option<ModelConfig>,
    pub step: Step,
    pub chars: Chars,
    pub heldout_nats: Option<f64>,
    pub n_experts: u32,
    pub bytes: Count,
}

/// What an exported folder says about itself (the `llm-trainer-export.json` file), checked before importing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ExportManifest {
    pub format: u32,
    pub app_version: String,
    /// Which engine wrote the model files: only the same engine can read them back.
    pub engine_name: String,
    pub engine_version: String,
    pub engine_format: u32,
    pub name: String,
    pub preset: Preset,
    pub model: ModelConfig,
    pub train: TrainConfig,
    pub step: Step,
    pub chars: Chars,
    pub heldout_nats: Option<f64>,
    pub n_experts: u32,
    pub created_at: UnixMs,
}
