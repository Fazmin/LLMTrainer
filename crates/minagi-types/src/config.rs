//! Model and training configuration, plus the Tiny / Small / Full presets.
//!
//! Field names follow the original mini-AGI `config.yaml` / `RecurConfig` so a checkpoint written by the Python
//! reference maps one-to-one. `ModelConfig` is fixed when a model is created; `TrainConfig` can change between runs.

use serde::{Deserialize, Serialize};
use specta::Type;

/// Model size presets. `Custom` means the user edited advanced settings; `Imported` is a checkpoint from elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    Tiny,
    Small,
    Full,
    Custom,
    Imported,
}

impl Preset {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "tiny" => Preset::Tiny,
            "small" => Preset::Small,
            "full" => Preset::Full,
            "custom" => Preset::Custom,
            "imported" => Preset::Imported,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Preset::Tiny => "tiny",
            Preset::Small => "small",
            Preset::Full => "full",
            Preset::Custom => "custom",
            Preset::Imported => "imported",
        }
    }
}

/// How the mixture-of-experts layer is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MoeMode {
    /// One wide feed-forward over all resident experts with per-token gains. Simple, no dropped tokens. Used for Tiny.
    DenseMasked,
    /// Switch-style capacity dispatch: only the chosen experts run. Used for Small and Full.
    SparseDispatch,
}

/// Numeric precision of matrix multiplications. Master weights are always f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F16,
    Bf16,
}

/// Shape of the network. Fixed once the model exists (the UI locks these fields on resume).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ModelConfig {
    pub d_model: u32,
    pub n_head: u32,
    /// Hidden width of the dense (trunk) SwiGLU blocks.
    pub d_ff: u32,
    pub n_prelude: u32,
    pub n_recur: u32,
    pub n_coda: u32,
    /// Maximum number of recurrent "rows" (thinking steps) per character.
    pub max_steps: u32,
    pub min_steps: u32,
    pub halt_prior: f64,
    pub halt_thresh: f64,
    pub halt_freeze: bool,
    pub ponder_beta: f64,
    pub bptt_window: u32,
    /// Mean number of rows sampled per training step (Poisson).
    pub train_steps_mean: f64,
    /// Longest context the position tables are built for.
    pub block: u32,
    pub vocab_size: u32,
    /// Experts the pool starts with.
    pub pool_experts: u32,
    /// Hard ceiling on the number of experts the pool may grow to.
    pub pool_max: u32,
    /// Hidden width of one expert.
    pub pool_d_ff: u32,
    pub pool_depth: u32,
    pub pool_top_k: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Preset::Full.model()
    }
}

/// Pool growth settings (new experts are born by recombining existing ones).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct GrowthConfig {
    pub every_chars: u32,
    pub k: u32,
    pub max_gap: f64,
    pub dying_frac_max: f64,
    pub max_in_flight: u32,
    pub keep_ratio_min: f64,
    pub recent_mult: f64,
    pub max_disk_gb: f64,
    pub mem_frac: f64,
    pub birth_gate: f64,
}

impl Default for GrowthConfig {
    fn default() -> Self {
        Self {
            every_chars: 2_000_000,
            k: 1,
            max_gap: 0.4,
            dying_frac_max: 0.35,
            max_in_flight: 50,
            keep_ratio_min: 0.35,
            recent_mult: 4.0,
            max_disk_gb: 10.0,
            mem_frac: 0.95,
            birth_gate: 0.001,
        }
    }
}

/// Pruning settings (experts nothing has addressed for a long time are removed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct PruneConfig {
    pub survival_chars: u32,
    pub dying_at: f64,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self { survival_chars: 100_000_000, dying_at: 0.65 }
    }
}

/// Greedy decoding with a repetition trace ("adapted" reading in the original).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct DecodingConfig {
    pub adapt_strength: f64,
    pub adapt_decay: f64,
    pub rep_penalty: f64,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self { adapt_strength: 2.5, adapt_decay: 0.88, rep_penalty: 1.0 }
    }
}

/// Everything about how training proceeds. Can differ between runs of the same model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct TrainConfig {
    pub lr: f64,
    /// Trunk (embeddings, attention, norms, halting) learns at `lr * trunk_lr_mult`.
    pub trunk_lr_mult: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub clip: f64,
    /// Characters read per optimizer step.
    pub chunk: u32,
    /// Characters per passage visit in a lane.
    pub passage: u32,
    pub shuffle_seed: u32,
    pub pool_aux: f64,
    /// Experts on the accelerator at once.
    pub resident: u32,
    /// Experts cached in RAM.
    pub ram_cache: u32,
    pub capacity_factor: f64,
    pub explore_bias: f64,
    pub explore_steps: u32,
    pub context_start: u32,
    pub context_end: u32,
    pub context_step: u32,
    pub context_grow_every_chars: u32,
    pub context_gain_min: f64,
    pub context_every_chars: u32,
    pub growth: GrowthConfig,
    pub prune: PruneConfig,
    pub decoding: DecodingConfig,
    /// Minutes between held-out evaluation + sample rounds.
    pub sample_every_min: f64,
    /// Minutes between automatic checkpoints.
    pub save_every_min: f64,
    /// Held-out characters scored per domain in a full evaluation.
    pub eval_chars: u32,
    pub precision: Precision,
    pub moe_mode: MoeMode,
    /// Recompute rows in the backward pass instead of storing activations (needed for Full).
    pub row_checkpointing: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Preset::Full.train()
    }
}

impl Preset {
    /// Network shape for this preset. `Custom` and `Imported` fall back to the Small shape as a starting point.
    pub fn model(&self) -> ModelConfig {
        match self {
            Preset::Tiny => ModelConfig {
                d_model: 256,
                n_head: 4,
                d_ff: 704,
                n_prelude: 1,
                n_recur: 1,
                n_coda: 0,
                max_steps: 6,
                min_steps: 1,
                halt_prior: 0.29,
                halt_thresh: 0.9,
                halt_freeze: true,
                ponder_beta: 0.01,
                bptt_window: 6,
                train_steps_mean: 3.2,
                block: 1024,
                vocab_size: 265,
                pool_experts: 16,
                pool_max: 64,
                pool_d_ff: 256,
                pool_depth: 1,
                pool_top_k: 2,
            },
            Preset::Small | Preset::Custom | Preset::Imported => ModelConfig {
                d_model: 384,
                n_head: 6,
                d_ff: 1024,
                n_prelude: 2,
                n_recur: 1,
                n_coda: 0,
                max_steps: 12,
                min_steps: 1,
                halt_prior: 0.2,
                halt_thresh: 0.9,
                halt_freeze: true,
                ponder_beta: 0.01,
                bptt_window: 12,
                train_steps_mean: 4.0,
                block: 2048,
                vocab_size: 265,
                pool_experts: 32,
                pool_max: 256,
                pool_d_ff: 768,
                pool_depth: 1,
                pool_top_k: 4,
            },
            Preset::Full => ModelConfig {
                d_model: 512,
                n_head: 8,
                d_ff: 1408,
                n_prelude: 2,
                n_recur: 1,
                n_coda: 0,
                max_steps: 24,
                min_steps: 1,
                halt_prior: 0.072,
                halt_thresh: 0.9,
                halt_freeze: true,
                ponder_beta: 0.01,
                bptt_window: 24,
                train_steps_mean: 12.8,
                block: 4096,
                vocab_size: 265,
                pool_experts: 64,
                pool_max: 1024,
                pool_d_ff: 2048,
                pool_depth: 1,
                pool_top_k: 8,
            },
        }
    }

    /// Training settings for this preset. The demo presets use a higher trunk learning rate than the original's 0.1x,
    /// which exists to prevent forgetting in continual learning and would slow a short from-scratch run.
    pub fn train(&self) -> TrainConfig {
        let base = TrainConfig {
            lr: 3e-4,
            trunk_lr_mult: 0.1,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            clip: 1.0,
            chunk: 2048,
            passage: 32768,
            shuffle_seed: 0,
            pool_aux: 0.01,
            resident: 32,
            ram_cache: 96,
            capacity_factor: 1.5,
            explore_bias: 0.65,
            explore_steps: 1000,
            context_start: 2048,
            context_end: 4096,
            context_step: 1,
            context_grow_every_chars: 100_000,
            context_gain_min: 0.015,
            context_every_chars: 65_536,
            growth: GrowthConfig::default(),
            prune: PruneConfig::default(),
            decoding: DecodingConfig::default(),
            sample_every_min: 10.0,
            save_every_min: 5.0,
            eval_chars: 122_880,
            precision: Precision::F32,
            moe_mode: MoeMode::SparseDispatch,
            row_checkpointing: true,
        };
        match self {
            Preset::Tiny => TrainConfig {
                lr: 2e-3,
                trunk_lr_mult: 0.5,
                chunk: 512,
                passage: 8192,
                resident: 8,
                ram_cache: 16,
                context_start: 512,
                context_end: 1024,
                context_grow_every_chars: 50_000,
                context_every_chars: 16_384,
                growth: GrowthConfig { every_chars: 1_000_000, max_disk_gb: 1.0, ..GrowthConfig::default() },
                prune: PruneConfig { survival_chars: 8_000_000, ..PruneConfig::default() },
                sample_every_min: 1.0,
                eval_chars: 30_720,
                moe_mode: MoeMode::DenseMasked,
                row_checkpointing: false,
                ..base
            },
            Preset::Small | Preset::Custom | Preset::Imported => TrainConfig {
                lr: 1e-3,
                trunk_lr_mult: 0.25,
                chunk: 512,
                passage: 16384,
                resident: 16,
                ram_cache: 32,
                // The window is re-read every step, so a window longer than the chunk costs a multiple of the reading:
                // starting at the chunk size keeps a step as cheap as it can be, and the window grows if it helps.
                context_start: 512,
                context_end: 2048,
                growth: GrowthConfig { every_chars: 4_000_000, max_disk_gb: 4.0, ..GrowthConfig::default() },
                prune: PruneConfig { survival_chars: 40_000_000, ..PruneConfig::default() },
                sample_every_min: 5.0,
                eval_chars: 61_440,
                row_checkpointing: false,
                ..base
            },
            Preset::Full => base,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Preset::Tiny => "Tiny",
            Preset::Small => "Small",
            Preset::Full => "Full",
            Preset::Custom => "Custom",
            Preset::Imported => "Imported",
        }
    }
}

/// Recursively merge `over` into `base` (objects merge key by key; anything else replaces).
fn merge_json(base: &mut serde_json::Value, over: &serde_json::Value) {
    match (base, over) {
        (serde_json::Value::Object(b), serde_json::Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(slot) => merge_json(slot, v),
                    None => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (slot, v) => *slot = v.clone(),
    }
}

impl Preset {
    /// Load a stored model configuration, filling any field added since it was written from this preset's defaults.
    pub fn load_model(&self, stored: &serde_json::Value) -> Result<ModelConfig, serde_json::Error> {
        let mut base = serde_json::to_value(self.model())?;
        merge_json(&mut base, stored);
        serde_json::from_value(base)
    }

    /// Load a stored training configuration, filling any field added since it was written from this preset's defaults.
    pub fn load_train(&self, stored: &serde_json::Value) -> Result<TrainConfig, serde_json::Error> {
        let mut base = serde_json::to_value(self.train())?;
        merge_json(&mut base, stored);
        serde_json::from_value(base)
    }
}

impl ModelConfig {
    /// Number of per-row KV-cache slots: the dense prelude plus one per recurrent row.
    pub fn kv_slots(&self) -> u32 {
        self.n_prelude + self.max_steps * (self.n_recur + self.n_coda)
    }

    /// Parameters in the trunk (everything except the expert pool).
    pub fn trunk_params(&self) -> u64 {
        let d = self.d_model as u64;
        let ff = self.d_ff as u64;
        let attn = 3 * d * d + d * d; // fused qkv + output projection
        let swiglu = 3 * d * ff;
        let norms = 2 * d;
        let prelude = self.n_prelude as u64 * (attn + swiglu + norms);
        let recur = (self.n_recur + self.n_coda) as u64 * (attn + norms) + 2 * d * d; // adapter Linear(2d -> d)
        let embed = self.vocab_size as u64 * d; // tied with the output head
        prelude + recur + embed + d + d + 1 // final norm, halting head (weight + bias)
    }

    /// Parameters in one expert.
    pub fn expert_params(&self) -> u64 {
        3 * self.d_model as u64 * self.pool_d_ff as u64 * self.pool_depth.max(1) as u64
    }

    /// Total parameters when the pool holds `experts` experts.
    pub fn total_params(&self, experts: u32) -> u64 {
        self.trunk_params() + experts as u64 * self.expert_params()
    }

    /// Check internal consistency; returns a plain-English reason when something cannot work.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_head == 0 || !self.d_model.is_multiple_of(self.n_head) {
            return Err("The model width must be divisible by the number of attention heads.".into());
        }
        if !(self.d_model / self.n_head).is_multiple_of(2) {
            return Err("Each attention head needs an even width (rotary positions work on pairs).".into());
        }
        if self.n_recur != 1 || self.n_coda != 0 {
            return Err("Only one shared recurrent block and no coda blocks are supported.".into());
        }
        if !self.halt_freeze {
            return Err("Halting must freeze finished characters (halt_freeze) in this version.".into());
        }
        if self.min_steps == 0 || self.min_steps > self.max_steps {
            return Err("Minimum rows must be between 1 and the maximum rows.".into());
        }
        if self.pool_top_k == 0 || self.pool_top_k > self.pool_experts {
            return Err("Experts per character must be between 1 and the number of experts.".into());
        }
        if self.pool_max < self.pool_experts {
            return Err("The expert ceiling must be at least the starting number of experts.".into());
        }
        Ok(())
    }
}

impl TrainConfig {
    pub fn validate(&self, model: &ModelConfig) -> Result<(), String> {
        if self.chunk == 0 || self.chunk > model.block {
            return Err("Characters per step must be between 1 and the model's longest context.".into());
        }
        if self.context_start > self.context_end || self.context_end > model.block {
            return Err(
                "The context window must grow from a start value up to at most the model's longest context.".into()
            );
        }
        if self.resident == 0 || self.resident < model.pool_top_k {
            return Err("At least as many resident experts as experts-per-character are needed.".into());
        }
        if !(self.lr > 0.0 && self.lr.is_finite()) {
            return Err("The learning rate must be a positive number.".into());
        }
        if self.sample_every_min <= 0.0 || self.save_every_min <= 0.0 {
            return Err("Evaluation and save intervals must be positive.".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate() {
        for p in [Preset::Tiny, Preset::Small, Preset::Full] {
            let m = p.model();
            m.validate().unwrap_or_else(|e| panic!("{p:?} model: {e}"));
            p.train().validate(&m).unwrap_or_else(|e| panic!("{p:?} train: {e}"));
        }
    }

    #[test]
    fn full_matches_original_parameter_counts() {
        let m = Preset::Full.model();
        // Python reference: trunk ~8.1M, 3.1M per expert, 174 experts -> ~559M total.
        assert_eq!(m.expert_params(), 3_145_728);
        let trunk = m.trunk_params();
        assert!((8_000_000..8_300_000).contains(&trunk), "trunk params {trunk}");
        let total = m.total_params(174);
        assert!((550_000_000..565_000_000).contains(&total), "total params {total}");
        assert_eq!(m.kv_slots(), 26);
    }

    #[test]
    fn preset_sizes_are_ordered() {
        let t = Preset::Tiny.model().total_params(16);
        let s = Preset::Small.model().total_params(32);
        let f = Preset::Full.model().total_params(64);
        assert!(t < s && s < f);
        assert!((3_000_000..7_000_000).contains(&t), "tiny {t}");
    }

    #[test]
    fn config_round_trips_through_json() {
        let c = Preset::Small.train();
        let s = serde_json::to_string(&c).unwrap();
        let back: TrainConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
        // Stored configs written before a field existed still load, taking that field from the preset.
        let stored = serde_json::json!({ "lr": 0.01, "growth": { "k": 3 } });
        let loaded = Preset::Small.load_train(&stored).unwrap();
        assert_eq!(loaded.lr, 0.01);
        assert_eq!(loaded.growth.k, 3);
        assert_eq!(loaded.growth.every_chars, Preset::Small.train().growth.every_chars);
        assert_eq!(loaded.chunk, Preset::Small.train().chunk);
        // ...but a stray field is still a hard error rather than silently ignored.
        assert!(serde_json::from_str::<TrainConfig>(r#"{"lr":0.01}"#).is_err());
    }

    #[test]
    fn invalid_configs_are_rejected_with_plain_reasons() {
        let mut m = Preset::Tiny.model();
        m.n_head = 3;
        assert!(m.validate().unwrap_err().contains("divisible"));
        let m = Preset::Tiny.model();
        let mut t = Preset::Tiny.train();
        t.chunk = 100_000;
        assert!(t.validate(&m).is_err());
    }
}
