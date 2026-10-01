//! The engine's error type, and how it is shown to the app.

use minagi_types::{AppError, Preset};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("{0}")]
    Candle(#[from] candle_core::Error),
    #[error("{0}")]
    Text(#[from] minagi_text::TextError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    /// A checkpoint or expert file could not be read, written or understood.
    #[error("{0}")]
    Checkpoint(String),
    #[error("{0}")]
    Store(#[from] crate::store::StoreError),
    /// A configuration the engine cannot run.
    #[error("{0}")]
    Config(String),
    /// A model that cannot be loaded because it uses features this engine does not support.
    #[error("{0}")]
    Unsupported(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

impl EngineError {
    pub fn other(msg: impl Into<String>) -> Self {
        EngineError::Other(msg.into())
    }

    /// True when the failure looks like the accelerator or system ran out of memory.
    pub fn is_out_of_memory(&self) -> bool {
        let text = self.to_string().to_lowercase();
        text.contains("out of memory") || text.contains("outofmemory") || text.contains("failed to allocate")
    }

    /// The error as the app shows it to the user. `suggest` is the preset to recommend when memory ran out.
    pub fn to_app(&self, suggest: Option<Preset>) -> AppError {
        if self.is_out_of_memory() {
            return AppError::OutOfMemory { suggested_preset: suggest };
        }
        match self {
            EngineError::Checkpoint(m) => AppError::Checkpoint(m.clone()),
            EngineError::Store(e) => match e {
                crate::store::StoreError::IoAt { source, .. } | crate::store::StoreError::Io(source)
                    if source.raw_os_error() == Some(28) =>
                {
                    AppError::Io("The disk is full.".into())
                }
                crate::store::StoreError::IoAt { .. } | crate::store::StoreError::Io(_) => AppError::Io(e.to_string()),
                other => AppError::Checkpoint(other.to_string()),
            },
            EngineError::Unsupported(m) => AppError::Checkpoint(m.clone()),
            EngineError::Io(e) if e.raw_os_error() == Some(28) => AppError::Io("The disk is full.".into()),
            EngineError::Io(e) => AppError::Io(e.to_string()),
            EngineError::Text(minagi_text::TextError::NoText(_)) => AppError::DatasetNotReady,
            EngineError::Config(m) => AppError::Invalid(m.clone()),
            EngineError::Cancelled => AppError::Cancelled,
            other => AppError::Engine(other.to_string()),
        }
    }
}

impl From<EngineError> for AppError {
    fn from(e: EngineError) -> Self {
        e.to_app(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_memory_is_recognised_whatever_the_wording() {
        for m in ["Metal: out of memory", "kIOGPUCommandBufferCallbackErrorOutOfMemory", "Failed to allocate buffer"] {
            let e = EngineError::other(m);
            assert!(matches!(
                e.to_app(Some(Preset::Tiny)),
                AppError::OutOfMemory { suggested_preset: Some(Preset::Tiny) }
            ));
        }
        assert!(matches!(EngineError::other("boom").to_app(None), AppError::Engine(_)));
        assert!(matches!(EngineError::Cancelled.to_app(None), AppError::Cancelled));
    }
}
