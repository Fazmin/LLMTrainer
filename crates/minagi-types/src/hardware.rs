//! Hardware description, run estimates and preset recommendation.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::config::{ModelConfig, MoeMode, Preset, TrainConfig};
use crate::units::Count;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Metal,
    Cuda,
    Cpu,
}

/// One compute backend and whether it can be used on this machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct BackendInfo {
    pub kind: BackendKind,
    pub name: String,
    pub available: bool,
    /// Plain-English reason when unavailable ("built without CUDA", "needs macOS 15 or newer", ...).
    pub reason: Option<String>,
    /// Memory the engine may use on this backend, in GB.
    pub mem_budget_gb: f64,
    pub bf16_ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub os: String,
    pub cpu: String,
    pub cores: u32,
    pub ram_gb: f64,
    pub ram_free_gb: f64,
    pub disk_free_gb: f64,
    pub backends: Vec<BackendInfo>,
    /// Backend the engine will use (best available unless the user chose one).
    pub selected: BackendKind,
}

impl HardwareInfo {
    pub fn selected_backend(&self) -> Option<&BackendInfo> {
        self.backends.iter().find(|b| b.kind == self.selected)
    }

    /// True when training would run on the CPU only (shown as a persistent warning in the UI).
    pub fn is_cpu_only(&self) -> bool {
        self.selected == BackendKind::Cpu
    }
}

/// How well a configuration fits the selected backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Fit {
    Comfortable,
    Tight,
    WontFit,
}

/// Predicted cost of a configuration on this machine. Numbers marked `measured` came from a benchmark here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Estimate {
    pub params_start: Count,
    pub params_full_pool: Count,
    /// Memory resident on the accelerator while training, in GB.
    pub gpu_gb: f64,
    pub ram_gb: f64,
    /// Disk used by the expert pool and checkpoints at the growth ceiling, in GB.
    pub disk_gb: f64,
    pub chars_per_sec: f64,
    /// True when `chars_per_sec` came from a benchmark on this machine instead of a formula.
    pub measured: bool,
    pub fit: Fit,
    pub notes: Vec<String>,
}

/// Summary of a prepared dataset the estimator needs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct DataStats {
    pub train_bytes: Count,
    pub val_bytes: Count,
    pub lanes: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct PresetFit {
    pub preset: Preset,
    pub estimate: Estimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct Recommendation {
    pub preset: Preset,
    pub reasons: Vec<String>,
    pub fits: Vec<PresetFit>,
}

const GB: f64 = 1_073_741_824.0;

/// Formula-based estimate (the real engine refines `chars_per_sec` from benchmarks). Pure and cheap, so the Setup
/// screen can call it on every edit.
///
/// Memory = what stays resident (weights, gradients and Adam state for the trunk and the resident experts) plus the
/// activations of the rows kept alive for the backward pass. Speed = FLOPs per step over an effective throughput, plus
/// a fixed per-step cost: small models are launch-bound on a GPU, not compute-bound.
pub fn estimate_formula(model: &ModelConfig, train: &TrainConfig, hw: &HardwareInfo) -> Estimate {
    let params_start = model.total_params(model.pool_experts);
    let params_full = model.total_params(model.pool_max);
    let (d, heads, t) = (model.d_model as f64, model.n_head as f64, train.context_end as f64);
    let expert = model.expert_params() as f64;
    let trunk = model.trunk_params() as f64;
    let slots = (train.resident as f64).min(model.pool_experts as f64);
    let k = model.pool_top_k as f64;
    let dff_e = model.pool_d_ff as f64;

    // Resident: weights + gradients + Adam m and v, 4 bytes each.
    let persistent = (trunk + slots * expert) * 4.0 * 4.0;
    // Activations per row: attention scores, expert buffers, and the dense tensors.
    let attn = 4.0 * heads * t * t * 4.0;
    let moe = match train.moe_mode {
        MoeMode::DenseMasked => t * slots * dff_e * 3.0 * 4.0,
        MoeMode::SparseDispatch => train.capacity_factor * t * k * (d + 3.0 * dff_e) * 4.0,
    };
    let dense = t * d * 4.0 * 20.0;
    let rows_live = if train.row_checkpointing { 1.0 } else { model.train_steps_mean.max(1.0) };
    // Calibrated against the real engine on an Apple M4 Pro: the buffers a training step really holds (backward-pass
    // copies, the allocator's pool) come to about twice what the tensors themselves add up to, plus a fixed overhead.
    let gpu_gb = 2.2 * (persistent + rows_live * (attn + moe + dense)) / GB + 0.5;

    let expert_with_moments = expert * 4.0 * 3.0; // weights + Adam m + v, as stored per expert
    let ram_gb = train.ram_cache as f64 * expert_with_moments / GB + 0.5;
    let disk_gb = (model.pool_max as f64 * expert_with_moments / GB).min(train.growth.max_disk_gb.max(0.1)) + 0.1;

    // FLOPs per token for forward + backward: dense prelude once, then the recurrent row (attention projections, adapter,
    // and the active experts) for each row, plus attention over the whole window.
    let ff = model.d_ff as f64;
    let prelude = model.n_prelude as f64 * (4.0 * d * d + 3.0 * d * ff);
    let active_experts = match train.moe_mode {
        MoeMode::DenseMasked => slots,
        MoeMode::SparseDispatch => k,
    };
    let rows = model.train_steps_mean.max(1.0);
    let per_row = 6.0 * d * d + active_experts * expert;
    let flops_per_token = 6.0 * (prelude + rows * per_row) + 12.0 * t * d * (model.n_prelude as f64 + rows);
    let recompute = if train.row_checkpointing { 1.33 } else { 1.0 };
    let (tflops, overhead_s, label) = match hw.selected {
        BackendKind::Metal => (2.0, 0.035, "Metal"),
        BackendKind::Cuda => (8.0, 0.012, "CUDA"),
        BackendKind::Cpu => (0.25, 0.005, "CPU"),
    };
    let step_s = t * flops_per_token * recompute / (tflops * 1e12) + overhead_s;
    let chars_per_sec = (train.chunk as f64 / step_s).clamp(1.0, 50_000.0);

    let budget = hw.selected_backend().map(|b| b.mem_budget_gb).unwrap_or(hw.ram_gb * 0.5);
    let mut notes = Vec::new();
    let fit = if hw.selected == BackendKind::Cpu && chars_per_sec < 200.0 {
        notes.push("This size is far too slow on a CPU. Try Tiny.".to_string());
        Fit::WontFit
    } else if gpu_gb > budget {
        notes.push(format!("Needs about {gpu_gb:.1} GB but this computer can offer about {budget:.1} GB."));
        Fit::WontFit
    } else if gpu_gb > budget * 0.8 {
        notes.push("Close to the memory limit; other apps may slow it down.".to_string());
        Fit::Tight
    } else {
        Fit::Comfortable
    };
    if ram_gb > hw.ram_gb * 0.7 {
        notes.push("The expert cache is large for this computer's RAM.".to_string());
    }
    if hw.selected == BackendKind::Cpu {
        notes.push(format!("Running on the {label}: expect much slower training than with a GPU."));
    }
    Estimate {
        params_start: Count(params_start),
        params_full_pool: Count(params_full),
        gpu_gb,
        ram_gb,
        disk_gb,
        chars_per_sec,
        measured: false,
        fit,
        notes,
    }
}

/// Pick the largest preset that fits comfortably and suits the amount of data.
pub fn recommend_preset(hw: &HardwareInfo, data: Option<&DataStats>) -> Recommendation {
    let fits: Vec<PresetFit> = [Preset::Tiny, Preset::Small, Preset::Full]
        .into_iter()
        .map(|p| PresetFit { preset: p, estimate: estimate_formula(&p.model(), &p.train(), hw) })
        .collect();
    let mut reasons = Vec::new();
    let mut preset = Preset::Tiny;
    let budget = hw.selected_backend().map(|b| b.mem_budget_gb).unwrap_or(0.0);
    if hw.is_cpu_only() {
        reasons.push("No GPU was found, so the smallest model keeps training fast enough to be useful.".to_string());
    } else if fits.iter().any(|f| f.preset == Preset::Small && f.estimate.fit == Fit::Comfortable) && budget >= 8.0 {
        preset = Preset::Small;
        reasons.push("Your GPU has enough memory for a Small model.".to_string());
    }
    if let Some(d) = data {
        let mb = d.train_bytes.0 as f64 / 1e6;
        if mb < 5.0 && preset != Preset::Tiny {
            preset = Preset::Tiny;
            reasons.push("Your text is small (under 5 MB), so a bigger model would just memorise it.".to_string());
        } else if mb < 50.0 && preset == Preset::Full {
            preset = Preset::Small;
        }
    }
    if preset == Preset::Tiny && reasons.is_empty() {
        reasons.push("Tiny gives your first readable results in about ten minutes.".to_string());
    }
    Recommendation { preset, reasons, fits }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(kind: BackendKind, budget: f64) -> HardwareInfo {
        HardwareInfo {
            os: "test".into(),
            cpu: "test".into(),
            cores: 8,
            ram_gb: 24.0,
            ram_free_gb: 16.0,
            disk_free_gb: 200.0,
            backends: vec![BackendInfo {
                kind,
                name: "test".into(),
                available: true,
                reason: None,
                mem_budget_gb: budget,
                bf16_ok: false,
            }],
            selected: kind,
        }
    }

    #[test]
    fn cpu_recommends_tiny_and_blocks_full() {
        let h = hw(BackendKind::Cpu, 12.0);
        let r = recommend_preset(&h, None);
        assert_eq!(r.preset, Preset::Tiny);
        let full = r.fits.iter().find(|f| f.preset == Preset::Full).unwrap();
        assert_eq!(full.estimate.fit, Fit::WontFit);
    }

    #[test]
    fn small_data_forces_tiny() {
        let h = hw(BackendKind::Metal, 14.0);
        let d = DataStats { train_bytes: Count(2_000_000), ..Default::default() };
        assert_eq!(recommend_preset(&h, Some(&d)).preset, Preset::Tiny);
    }

    #[test]
    fn estimates_are_ordered_and_in_a_believable_range() {
        let h = hw(BackendKind::Metal, 14.0);
        let e = |p: Preset| estimate_formula(&p.model(), &p.train(), &h);
        let (t, s, f) = (e(Preset::Tiny), e(Preset::Small), e(Preset::Full));
        assert!(t.gpu_gb < s.gpu_gb && s.gpu_gb < f.gpu_gb, "memory {} < {} < {}", t.gpu_gb, s.gpu_gb, f.gpu_gb);
        assert!(t.chars_per_sec > s.chars_per_sec && s.chars_per_sec > f.chars_per_sec);
        // From the engine analysis: Tiny is launch-bound at roughly 8-12k chars/s on Metal; Full needs several GB.
        assert!((5_000.0..15_000.0).contains(&t.chars_per_sec), "tiny speed {}", t.chars_per_sec);
        assert!((8.0..16.0).contains(&f.gpu_gb), "full memory {}", f.gpu_gb);
        assert!(t.gpu_gb < 2.0, "tiny memory {}", t.gpu_gb);
    }

    #[test]
    fn row_checkpointing_trades_speed_for_memory() {
        let h = hw(BackendKind::Metal, 24.0);
        let mut with = Preset::Small.train();
        with.row_checkpointing = true;
        let without = Preset::Small.train();
        let (a, b) = (
            estimate_formula(&Preset::Small.model(), &with, &h),
            estimate_formula(&Preset::Small.model(), &without, &h),
        );
        assert!(a.gpu_gb < b.gpu_gb);
        assert!(a.chars_per_sec < b.chars_per_sec);
    }

    #[test]
    fn tiny_is_cheaper_than_full() {
        let h = hw(BackendKind::Metal, 14.0);
        let t = estimate_formula(&Preset::Tiny.model(), &Preset::Tiny.train(), &h);
        let f = estimate_formula(&Preset::Full.model(), &Preset::Full.train(), &h);
        assert!(t.gpu_gb < f.gpu_gb);
        assert!(t.chars_per_sec > f.chars_per_sec);
        assert_eq!(t.fit, Fit::Comfortable);
    }
}
