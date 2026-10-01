//! Simulated engine for UI development and contract tests.
//!
//! `MockFactory` implements the same `EngineFactory` / `Engine` / `Generator` traits as the real engine, produces
//! deterministic, plausible learning curves, and can be told to misbehave (plateau, diverge, overfit, run out of memory
//! or disk) so every screen and error card can be exercised without a GPU.

mod engine;
mod generator;
mod sim;

use std::path::Path;

use minagi_types::{
    AppResult, BackendInfo, BackendKind, DataStats, Engine, EngineFactory, Estimate, Generator, HardwareInfo,
    ModelConfig, TrainConfig, estimate_formula,
};

pub use engine::MockEngine;
pub use generator::MockGenerator;
pub use sim::{Curve, Rng, Scenario};

pub struct MockFactory {
    /// How many times faster than real time the simulated clock runs.
    pub speed: f64,
    pub scenario: Scenario,
}

impl MockFactory {
    pub fn new(speed: f64, scenario: Scenario) -> Self {
        Self { speed: speed.max(0.1), scenario }
    }

    /// `MINAGI_MOCK_SPEED` (default 20) and `MINAGI_MOCK_SCENARIO` (default `normal`).
    pub fn from_env() -> Self {
        let speed = std::env::var("MINAGI_MOCK_SPEED").ok().and_then(|s| s.parse().ok()).unwrap_or(20.0);
        let scenario =
            std::env::var("MINAGI_MOCK_SCENARIO").ok().and_then(|s| Scenario::parse(&s)).unwrap_or(Scenario::Normal);
        Self::new(speed, scenario)
    }
}

impl EngineFactory for MockFactory {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn is_mock(&self) -> bool {
        true
    }

    fn version(&self) -> String {
        format!("mock-{}", env!("CARGO_PKG_VERSION"))
    }

    fn probe_backends(&self) -> Vec<BackendInfo> {
        let mut v = Vec::new();
        if self.scenario != Scenario::NoGpu {
            v.push(BackendInfo {
                kind: BackendKind::Metal,
                name: "Simulated GPU".into(),
                available: true,
                reason: None,
                mem_budget_gb: 14.0,
                bf16_ok: true,
            });
        }
        v.push(BackendInfo {
            kind: BackendKind::Cpu,
            name: "CPU".into(),
            available: true,
            reason: None,
            mem_budget_gb: 12.0,
            bf16_ok: false,
        });
        v
    }

    fn estimate(
        &self,
        model: &ModelConfig,
        train: &TrainConfig,
        hw: &HardwareInfo,
        _data: Option<&DataStats>,
    ) -> Estimate {
        estimate_formula(model, train, hw)
    }

    fn new_engine(&self) -> Box<dyn Engine> {
        Box::new(MockEngine { speed: self.speed, scenario: self.scenario })
    }

    fn open_generator(&self, checkpoint: &Path, backend: BackendKind) -> AppResult<Box<dyn Generator>> {
        Ok(Box::new(MockGenerator::open(checkpoint, backend, self.speed)))
    }
}

#[cfg(test)]
mod tests;
