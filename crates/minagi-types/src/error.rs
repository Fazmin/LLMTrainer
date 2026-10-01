//! The one error type that crosses the Rust/TypeScript boundary.
//!
//! Each variant maps to a plain-English card in the UI (`ui/src/content/errors.ts`), so variants carry
//! structured detail rather than prose.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::config::Preset;
use crate::units::Count;

#[derive(Debug, Clone, PartialEq, thiserror::Error, Serialize, Deserialize, Type)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum AppError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("a training run is already active")]
    RunActive,
    #[error("no training run is active")]
    NoActiveRun,
    #[error("dataset is not ready")]
    DatasetNotReady,
    #[error("not enough disk space")]
    #[serde(rename_all = "camelCase")]
    DiskFull { need_bytes: Count, free_bytes: Count },
    #[error("out of memory")]
    #[serde(rename_all = "camelCase")]
    OutOfMemory { suggested_preset: Option<Preset> },
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("checkpoint is incompatible or corrupt: {0}")]
    Checkpoint(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("database error: {0}")]
    Db(String),
    #[error("engine error: {0}")]
    Engine(String),
    #[error("cancelled")]
    Cancelled,
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e.to_string())
    }
}

pub type AppResult<T> = Result<T, AppError>;
