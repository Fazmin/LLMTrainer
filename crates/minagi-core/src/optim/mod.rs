//! Optimization: AdamW with per-expert moments, gradient clipping, the plasticity controller and the growth brakes.

pub mod adamw;
pub mod brakes;
pub mod plasticity;
