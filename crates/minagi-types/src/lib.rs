//! Shared contract for LLM Trainer.
//!
//! This crate deliberately has no ML dependencies, so the app, the store and the simulated engine compile in
//! seconds. Everything here derives `serde` and `specta::Type`, and the TypeScript bindings are generated from it.

pub mod api;
pub mod config;
pub mod contract;
pub mod dataset;
pub mod error;
pub mod events;
pub mod hardware;
pub mod insight;
pub mod live;
pub mod traits;
pub mod units;

pub use api::*;
pub use config::*;
pub use dataset::*;
pub use error::*;
pub use events::*;
pub use hardware::*;
pub use insight::*;
pub use live::*;
pub use traits::*;
pub use units::*;
