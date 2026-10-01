//! On-disk tensor interchange: NumPy `.npy` / `.npz` readers and writers plus bf16 packing.
//!
//! Checkpoints are exchanged with the Python reference implementation as `.npz` archives, so the
//! formats here are byte-compatible with what `numpy.savez` writes and `numpy.load` reads:
//!
//! * [`npy`]: single arrays in `.npy` v1.0 (little-endian only; `f4 f8 i2 i4 i8 u1`), plus
//!   reading v2.0/v3.0 headers.
//! * [`npz`]: ZIP archives of `<key>.npy` entries. Writing is *stored* (uncompressed) like
//!   `numpy.savez`; reading also accepts deflate (`numpy.savez_compressed`).
//! * [`bf16`]: bfloat16 packing (round-to-nearest-even) used to store weights as `<i2` bit
//!   patterns, which is how the reference implementation keeps bf16 tensors in an npz.
//!
//! Keys are free-form strings (for example `recur.0.mlp.router.weight` or names containing `|`);
//! they are stored verbatim as the archive entry name plus the `.npy` suffix, exactly as numpy
//! does.
//!
//! On top of the file formats sit the pieces that make up a saved model:
//!
//! * [`tiers`]: the expert pool on disk (one `experts/eNNNNN.npz` per expert, named by its stable
//!   uid) with a RAM cache in front of it and write-back of changed experts.
//! * [`manifest`]: `manifest.json`, the index of a checkpoint directory, read and written exactly
//!   like the Python reference does, tolerant of keys it does not know.
//! * [`checkpoint`]: the whole directory (`core.npz`, `routers.npz`, `optim.npz`, `experts/`,
//!   `manifest.json`, `COMPLETE`), written atomically, plus a cheap way to inspect one.
//! * [`python`]: importing a directory written by the Python reference (and exporting back).
//!
//! Nothing in here depends on candle: tensors are plain `Vec<f32>` plus a shape, so these modules
//! build and test quickly and can be used by tools that never touch a GPU.

pub mod bf16;
pub mod checkpoint;
pub mod fsio;
pub mod manifest;
pub mod npy;
pub mod npz;
pub mod python;
pub mod safetensors;
pub mod tiers;

pub use npy::{NpyArray, NpyData};
pub use npz::{NpzReader, NpzWriter};

use std::path::PathBuf;

/// Errors from the store layer.
///
/// The messages are written for a person reading a log or a dialog: they say which file or expert
/// is the problem and, where there is an obvious next step, what it is.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// An i/o error that knows which path it happened on.
    #[error("{}: {source}", path.display())]
    IoAt { path: PathBuf, source: std::io::Error },
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    /// The bytes are not a valid `.npy`/`.npz`.
    #[error("invalid npy data: {0}")]
    Format(String),
    /// Valid numpy data this implementation deliberately does not handle.
    #[error("unsupported npy feature: {0}")]
    Unsupported(String),
    #[error("key not found in npz: {0:?}")]
    MissingKey(String),
    /// `manifest.json` could not be parsed or written.
    #[error("manifest.json problem in {}: {message}", path.display())]
    Manifest { path: PathBuf, message: String },
    /// A pool expert's file is not where the pool says it is.
    #[error(
        "expert e{uid:05} is missing: its file {} does not exist (the experts folder may have been moved, \
         partly deleted, or copied without its files)",
        path.display()
    )]
    ExpertMissing { uid: u64, path: PathBuf },
    /// A pool expert's file exists but cannot be used.
    #[error("expert e{uid:05} is unreadable: {} {reason}", path.display())]
    ExpertCorrupt { uid: u64, path: PathBuf, reason: String },
    /// The store was opened read-only and was asked to change something.
    #[error("this store is read-only; refusing to {0}")]
    ReadOnly(String),
    /// A checkpoint directory (or part of one) is missing, inconsistent or malformed.
    #[error("invalid checkpoint: {0}")]
    Invalid(String),
    /// Something is already there.
    #[error("already exists: {}", .0.display())]
    AlreadyExists(PathBuf),
    /// A checkpoint written by the Python reference that this engine cannot run. The text is plain
    /// English and can be shown to the user as it is.
    #[error("{0}")]
    Incompatible(String),
}

impl StoreError {
    /// Attach a path to an i/o error.
    pub fn io_at(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        StoreError::IoAt { path: path.into(), source }
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;
