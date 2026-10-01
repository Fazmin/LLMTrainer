//! SQLite persistence for LLM Trainer: runs, telemetry, evaluations, samples, checkpoints and (later) datasets,
//! chat and jobs. See `migrations/0001_init.sql` for the schema.

mod chat;
mod datasets;
mod db;
mod pool;
mod runs;
mod telemetry;

pub use chat::{ChatSessionRow, NewChat};
pub use datasets::{DatasetFinal, DatasetRoot, NewDataset};
pub use db::{SCHEMA_VERSION, Store, StoreError, StoreResult};
pub use runs::{CONFIG_SCHEMA_VERSION, NewRun, RunOrigin, RunProgress, slugify};
pub use telemetry::{MetricKey, NewEvent, snap_bucket};
