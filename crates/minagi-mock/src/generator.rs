//! Simulated text generation for the chat screen.

use std::path::Path;
use std::time::{Duration, Instant};

use minagi_types::{
    AppResult, BackendKind, Chars, CheckpointKind, CheckpointMeta, Count, GenChar, GenRequest, GenStats, Generator,
    GeneratorInfo, LearnReport, Step,
};

use crate::sim::Rng;

const REPLIES: &[&str] = &[
    "The little fox looked up at the moon and smiled. \"Tomorrow,\" he said, \"I will find my way home.\"",
    "I am a small language model. I learn by reading text one character at a time, and I am still learning.",
    "Once upon a time, a kind old woman planted a tiny seed. Every day she watered it, and slowly it grew.",
];

pub struct MockGenerator {
    label: String,
    backend: BackendKind,
    speed: f64,
    rng: Rng,
    learned: u32,
    quality: f64,
}

impl MockGenerator {
    pub fn open(checkpoint: &Path, backend: BackendKind, speed: f64) -> Self {
        let label =
            checkpoint.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "checkpoint".into());
        let chars = std::fs::read_to_string(checkpoint.join("mock-state.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v["chars"].as_f64())
            .unwrap_or(0.0);
        // More training → fewer typos in the simulated writing.
        let quality = (chars / 3.0e6).clamp(0.0, 1.0);
        Self { label, backend, speed: speed.max(0.1), rng: Rng::new(chars as u64 ^ 0xABCD), learned: 0, quality }
    }
}

impl Generator for MockGenerator {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo { label: self.label.clone(), backend: self.backend, supports_learn: true, needs_gb: 0.5 }
    }

    fn generate(&mut self, req: &GenRequest, out: &mut dyn FnMut(GenChar) -> bool) -> AppResult<GenStats> {
        let reply = REPLIES[self.rng.range(0, REPLIES.len())];
        let started = Instant::now();
        let mut n = 0u64;
        let mut produced = String::new();
        for ch in reply.chars().take(req.max_new as usize) {
            let ch = if self.rng.f64() > 0.55 + 0.45 * self.quality { 'e' } else { ch };
            produced.push(ch);
            n += 1;
            let experts =
                vec![(self.rng.range(1, 30)) as u16, self.rng.range(1, 30) as u16, self.rng.range(1, 30) as u16];
            let rows = (1 + self.rng.range(0, 6)) as u8;
            if !out(GenChar { text: ch.to_string(), rows, experts }) {
                break;
            }
            if let Some(stop) = &req.stop_at
                && produced.ends_with(stop.as_str())
            {
                break;
            }
            std::thread::sleep(Duration::from_secs_f64(0.012 / self.speed.min(10.0)));
        }
        let seconds = started.elapsed().as_secs_f64().max(1e-6);
        Ok(GenStats { chars: Count(n), seconds, chars_per_sec: n as f64 / seconds })
    }

    fn learn(&mut self, _text: &str) -> AppResult<LearnReport> {
        self.learned += 1;
        let before = 2.4 - 0.1 * self.learned.saturating_sub(1) as f64;
        Ok(LearnReport { nats_before: before, nats_after: before - 0.1 })
    }

    fn save_adapted(&mut self, dest: &Path) -> AppResult<CheckpointMeta> {
        std::fs::create_dir_all(dest)?;
        std::fs::write(dest.join("mock-state.json"), br#"{"step":0,"chars":0}"#)?;
        std::fs::write(dest.join("COMPLETE"), b"ok")?;
        Ok(CheckpointMeta {
            step: Step(0),
            chars: Chars(0),
            kind: CheckpointKind::Manual,
            path: dest.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            bytes: Count(24),
            heldout_nats: None,
            n_experts: 16,
            engine_format: 0,
        })
    }
}
