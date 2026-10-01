//! The real engine, plugged into the app through the same traits the simulated engine implements.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use minagi_types::{
    AppError, AppResult, BackendInfo, BackendKind, Chars, Count, Ctl, DataStats, Engine, EngineEvent, EngineFactory,
    Estimate, EventSink, Generator, HardwareInfo, ImportIssue, ImportKind, ImportPreview, ImportSeverity, ModelConfig,
    Outcome, RunSpec, Stage, StageInfo, Step, TrainConfig, estimate_formula,
};

use crate::backend;
use crate::chat::ModelGenerator;
use crate::train::trainer;

/// Creates the real (candle) engine. `ram_gb` sizes the memory budgets reported for each backend.
pub struct RealFactory {
    ram_gb: f64,
}

impl RealFactory {
    pub fn new(ram_gb: f64) -> Self {
        Self { ram_gb }
    }
}

impl EngineFactory for RealFactory {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn is_mock(&self) -> bool {
        false
    }

    fn version(&self) -> String {
        format!("minagi-core {}", env!("CARGO_PKG_VERSION"))
    }

    fn probe_backends(&self) -> Vec<BackendInfo> {
        backend::probe(self.ram_gb)
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
        Box::new(RealEngine { ram_gb: self.ram_gb })
    }

    fn open_generator(&self, checkpoint: &Path, backend: BackendKind) -> AppResult<Box<dyn Generator>> {
        ModelGenerator::open(checkpoint, backend).map(|g| Box::new(g) as Box<dyn Generator>).map_err(AppError::from)
    }

    fn export_safetensors(&self, checkpoint: &Path, dest_dir: &Path) -> AppResult<u64> {
        crate::store::safetensors::export_safetensors(checkpoint, dest_dir)
            .map(|r| r.bytes)
            .map_err(|e| AppError::Checkpoint(e.to_string()))
    }

    fn inspect_python(&self, src: &Path) -> AppResult<ImportPreview> {
        let report =
            crate::store::python::check_python_checkpoint(src).map_err(|e| AppError::Checkpoint(e.to_string()))?;
        Ok(python_preview(src, &report))
    }

    fn import_python(&self, src: &Path, dst: &Path) -> AppResult<ImportPreview> {
        use crate::store::python::{ImportOptions, import_python};
        // A directory fresh from the original's `create` says halt_freeze is off because the trainer turns it on at load.
        let opts = ImportOptions { override_halt_freeze: true, overwrite: false };
        let report = import_python(src, dst, &opts).map_err(|e| AppError::Checkpoint(e.to_string()))?;
        Ok(python_preview(src, &report))
    }
}

fn python_preview(src: &Path, report: &crate::store::python::CompatReport) -> ImportPreview {
    use crate::store::python::Severity;
    let manifest = crate::store::manifest::Manifest::read(&src.join(crate::store::manifest::MANIFEST_FILE)).ok();
    let name = src
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .map_or_else(|| "Original model".to_string(), |n| format!("Original model ({n})"));
    ImportPreview {
        path: src.to_string_lossy().into_owned(),
        kind: ImportKind::Python,
        importable: report.is_importable(),
        summary: report.summary(),
        issues: report
            .issues
            .iter()
            .map(|i| ImportIssue {
                severity: match i.severity {
                    Severity::Blocker => ImportSeverity::Blocker,
                    Severity::Warning => ImportSeverity::Warning,
                    Severity::Note => ImportSeverity::Note,
                },
                message: i.message.clone(),
            })
            .collect(),
        name,
        preset: report.suggested_preset,
        model: report.config.clone(),
        step: Step(report.step.unwrap_or(0).max(0) as u64),
        chars: Chars(manifest.as_ref().and_then(|m| m.read_chars).unwrap_or(0)),
        heldout_nats: report.val.filter(|v| v.is_finite()),
        n_experts: report.n_experts as u32,
        bytes: Count(report.total_bytes),
    }
}

pub struct RealEngine {
    ram_gb: f64,
}

impl Engine for RealEngine {
    fn run(&mut self, spec: RunSpec, ctl: flume::Receiver<Ctl>, sink: &dyn EventSink) -> Outcome {
        let budget = backend::probe(self.ram_gb)
            .into_iter()
            .find(|b| b.kind == spec.backend)
            .map(|b| b.mem_budget_gb)
            .unwrap_or(0.0);
        // A panic inside the engine must end the run with a message, not take the app down with it.
        match catch_unwind(AssertUnwindSafe(|| trainer::run(&spec, &ctl, sink, budget))) {
            Ok(outcome) => outcome,
            Err(payload) => {
                let what = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown error".into());
                sink.emit(EngineEvent::Stage(StageInfo {
                    stage: Stage::Failed,
                    detail: Some("The training engine hit an unexpected problem".into()),
                    progress: None,
                }));
                Outcome::Failed { error: AppError::Engine(format!("The training engine crashed: {what}")) }
            }
        }
    }
}
