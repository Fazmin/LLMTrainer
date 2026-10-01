//! The crate's error type and its mapping onto the app-wide [`AppError`].

use std::io;
use std::path::{Path, PathBuf};

use minagi_types::{AppError, Count};

/// Everything that can go wrong in the dataset manager.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// A filesystem operation failed. `path` is the file or folder involved when known.
    #[error("{}", io_message(path, source))]
    Io { path: Option<PathBuf>, source: io::Error },
    /// The network misbehaved: connection trouble, an unexpected HTTP status, a server that ignored a `Range` header,
    /// or downloaded bytes that failed verification.
    #[error("network error: {0}")]
    Network(String),
    /// Downloaded bytes do not match the published checksum.
    #[error("checksum mismatch for {file}: expected {expected}, got {actual}")]
    Checksum { file: String, expected: String, actual: String },
    /// There is not enough free space to start (or continue) the job.
    #[error("not enough disk space: need {need_bytes} bytes, {free_bytes} free")]
    DiskFull { need_bytes: u64, free_bytes: u64 },
    /// The caller cancelled the job (the job's partial state is kept where it can be resumed).
    #[error("cancelled")]
    Cancelled,
    /// The input makes no sense (bad parameters, a layout the operation cannot handle).
    #[error("invalid input: {0}")]
    Invalid(String),
    /// Something that was asked for does not exist.
    #[error("not found: {0}")]
    NotFound(String),
}

fn io_message(path: &Option<PathBuf>, source: &io::Error) -> String {
    match path {
        Some(p) => format!("{}: {source}", p.display()),
        None => format!("io error: {source}"),
    }
}

/// Result alias used across the crate.
pub type DataResult<T> = Result<T, DataError>;

impl DataError {
    /// An I/O error that happened on `path`.
    pub fn io(path: impl AsRef<Path>, source: io::Error) -> Self {
        DataError::Io { path: Some(path.as_ref().to_path_buf()), source }
    }

    /// Shorthand for [`DataError::Invalid`].
    pub fn invalid(msg: impl Into<String>) -> Self {
        DataError::Invalid(msg.into())
    }

    /// True for [`DataError::Cancelled`].
    pub fn is_cancelled(&self) -> bool {
        matches!(self, DataError::Cancelled)
    }
}

impl From<io::Error> for DataError {
    fn from(source: io::Error) -> Self {
        DataError::Io { path: None, source }
    }
}

impl From<reqwest::Error> for DataError {
    fn from(e: reqwest::Error) -> Self {
        // `reqwest::Error` can embed the full URL (which may carry a signature); keep only the kind of failure.
        DataError::Network(e.without_url().to_string())
    }
}

impl From<DataError> for AppError {
    fn from(e: DataError) -> Self {
        match e {
            DataError::Io { .. } => AppError::Io(e.to_string()),
            DataError::Network(_) | DataError::Checksum { .. } => AppError::Network(e.to_string()),
            DataError::DiskFull { need_bytes, free_bytes } => {
                AppError::DiskFull { need_bytes: Count(need_bytes), free_bytes: Count(free_bytes) }
            }
            DataError::Cancelled => AppError::Cancelled,
            DataError::Invalid(msg) => AppError::Invalid(msg),
            DataError::NotFound(msg) => AppError::NotFound(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_variant_to_the_app_error() {
        let io = DataError::io("/x/y", io::Error::other("boom"));
        assert!(matches!(AppError::from(io), AppError::Io(m) if m.contains("/x/y") && m.contains("boom")));
        assert!(matches!(AppError::from(DataError::Network("down".into())), AppError::Network(_)));
        let bad = DataError::Checksum { file: "f".into(), expected: "a".into(), actual: "b".into() };
        assert!(matches!(AppError::from(bad), AppError::Network(m) if m.contains("checksum")));
        assert_eq!(
            AppError::from(DataError::DiskFull { need_bytes: 10, free_bytes: 3 }),
            AppError::DiskFull { need_bytes: Count(10), free_bytes: Count(3) }
        );
        assert_eq!(AppError::from(DataError::Cancelled), AppError::Cancelled);
        assert_eq!(AppError::from(DataError::invalid("no")), AppError::Invalid("no".into()));
        assert_eq!(AppError::from(DataError::NotFound("lane".into())), AppError::NotFound("lane".into()));
    }

    #[test]
    fn plain_io_errors_convert() {
        let e: DataError = io::Error::from(io::ErrorKind::PermissionDenied).into();
        assert!(matches!(e, DataError::Io { path: None, .. }));
    }
}
