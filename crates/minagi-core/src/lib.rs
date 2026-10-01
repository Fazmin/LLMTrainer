//! The mini-AGI engine: a recurrent, adaptive-depth transformer with a paged mixture-of-experts pool, trained on
//! candle (Metal on Apple Silicon, CUDA or CPU elsewhere).
//!
//! Module map and ownership while the engine is being built:
//!
//! - [`backend`]: device selection, the per-step autorelease pool, backend probing.
//! - [`ops`]: tensor primitives (RMSNorm, RoPE, attention, SwiGLU, top-k) with hand-written backward where it pays.
//! - [`model`]: the network and the halting loss.
//! - [`moe`]: router, the text's vote and slot admission, dispatch, growth and pruning of the expert pool.
//! - [`optim`]: AdamW with per-expert moments, gradient clipping, the plasticity controller, growth brakes.
//! - [`train`]: the training loop, context ramp, evaluation, sampling, checkpointing, events.
//! - [`chat`]: generation, prompts, learning from a conversation.
//! - [`store`]: `.npz` files, expert tiers (RAM and disk), the checkpoint layout, Python checkpoint import.
//!
//! The candle build is re-exported so every crate in the workspace uses the same pinned version and features.

pub use candle_core;
pub use candle_nn;

pub mod backend;
pub mod chat;
pub mod engine;
pub mod error;
pub mod model;
pub mod moe;
pub mod ops;
pub mod optim;
pub mod rng;
pub mod store;
pub mod train;

pub use engine::{RealEngine, RealFactory};
pub use error::{EngineError, Result};
