//! Headless trainer: the same engine the app runs, from the command line.
//!
//!   minagi_train --data DIR            DIR holds train/<lane>/... and val/<domain>/...
//!   minagi_train --synthetic           make a small made-up dataset in a temporary folder
//!   options: --preset tiny|small|full  --backend cpu|metal|cuda  --minutes N  --steps N
//!            --run-dir DIR  --resume CHECKPOINT  --eval-every-min X  --lr X  --chunk N

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use minagi_core::RealFactory;
use minagi_types::contract::run_with_driver;
use minagi_types::{BackendKind, Ctl, EngineEvent, EngineFactory, Preset, RunSpec};

fn args() -> HashMap<String, String> {
    let rest: Vec<String> = std::env::args().skip(1).collect();
    let mut kv = HashMap::new();
    let mut i = 0;
    while i < rest.len() {
        if let Some(k) = rest[i].strip_prefix("--") {
            if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                kv.insert(k.to_string(), rest[i + 1].clone());
                i += 2;
            } else {
                kv.insert(k.to_string(), "true".into());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    kv
}

fn prose(seed: usize, bytes: usize) -> String {
    const NOUNS: [&str; 16] = [
        "river", "garden", "bridge", "lantern", "harbor", "forest", "machine", "window", "orchard", "letter",
        "mountain", "market", "castle", "island", "valley", "teacher",
    ];
    const ADJ: [&str; 12] = [
        "quiet", "bright", "ancient", "narrow", "golden", "hidden", "gentle", "patient", "curious", "silver", "warm",
        "tiny",
    ];
    const VERBS: [&str; 10] =
        ["carries", "watches", "follows", "builds", "remembers", "crosses", "guards", "paints", "teaches", "gathers"];
    let mut s = String::new();
    let mut i = seed;
    while s.len() < bytes {
        s.push_str(&format!(
            "The {} {} {} the {} {}. ",
            ADJ[i % 12],
            NOUNS[(i * 5 + 3) % 16],
            VERBS[(i * 7 + 1) % 10],
            ADJ[(i * 11 + 4) % 12],
            NOUNS[(i * 3 + 7) % 16]
        ));
        if i % 4 == 3 {
            s.push('\n');
        }
        i += 1;
    }
    s
}

fn synthetic(root: &Path) -> Result<()> {
    for (side, files) in [("train", 6), ("val", 2)] {
        let lane = root.join(side).join("stories");
        std::fs::create_dir_all(&lane)?;
        for f in 0..files {
            std::fs::write(
                lane.join(format!("{f}.txt")),
                prose(f * 97 + if side == "val" { 1000 } else { 0 }, 200_000),
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let a = args();
    let preset = Preset::parse(a.get("preset").map(String::as_str).unwrap_or("tiny"))
        .ok_or_else(|| anyhow::anyhow!("unknown preset"))?;
    let backend = match a.get("backend").map(String::as_str).unwrap_or("cpu") {
        "metal" => BackendKind::Metal,
        "cuda" => BackendKind::Cuda,
        _ => BackendKind::Cpu,
    };
    let tmp = tempfile::tempdir_in(std::env::temp_dir())?;
    let data_root: PathBuf = match (a.get("data"), a.contains_key("synthetic")) {
        (Some(d), _) => d.into(),
        (None, true) => {
            let root = tmp.path().join("data");
            synthetic(&root)?;
            root
        }
        _ => bail!("give --data DIR or --synthetic"),
    };
    let run_dir: PathBuf = a.get("run-dir").map(PathBuf::from).unwrap_or_else(|| tmp.path().join("run"));
    let mut train = preset.train();
    if let Some(v) = a.get("eval-every-min").and_then(|v| v.parse().ok()) {
        train.sample_every_min = v;
    }
    if let Some(v) = a.get("lr").and_then(|v| v.parse().ok()) {
        train.lr = v;
    }
    if let Some(v) = a.get("chunk").and_then(|v| v.parse::<u32>().ok()) {
        train.chunk = v;
    }
    match a.get("moe").map(String::as_str) {
        Some("dense") => train.moe_mode = minagi_types::MoeMode::DenseMasked,
        Some("sparse") => train.moe_mode = minagi_types::MoeMode::SparseDispatch,
        _ => {}
    }
    if let Some(v) = a.get("resident").and_then(|v| v.parse::<u32>().ok()) {
        train.resident = v;
    }
    let spec = RunSpec {
        run_dir,
        data_root,
        lanes: None,
        model: preset.model(),
        train,
        seed: a.get("seed").and_then(|v| v.parse().ok()).unwrap_or(1),
        resume_from: a.get("resume").map(PathBuf::from),
        backend,
    };
    let max_steps: u64 = a.get("steps").and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    let max_secs: f64 = a.get("minutes").and_then(|v| v.parse::<f64>().ok()).map(|m| m * 60.0).unwrap_or(f64::INFINITY);
    let factory = RealFactory::new(16.0);
    println!(
        "engine {} on {:?}, preset {}, run dir {}",
        factory.version(),
        backend,
        preset.display_name(),
        spec.run_dir.display()
    );
    let started = Instant::now();
    let (mut last_print, mut chars_at_print, mut window_t) = (0u64, 0u64, Instant::now());
    let rec = run_with_driver(&factory, spec, |ev, ctl| match ev {
        EngineEvent::Stage(s) => println!(
            "[{:>7.1}s] stage {:?}: {}",
            started.elapsed().as_secs_f64(),
            s.stage,
            s.detail.clone().unwrap_or_default()
        ),
        EngineEvent::Step(s) => {
            if s.step.0 >= last_print + 20 || s.step.0 == 1 {
                let dt = window_t.elapsed().as_secs_f64().max(1e-9);
                let cps = (s.chars_read.0 - chars_at_print) as f64 / dt;
                println!(
                    "[{:>7.1}s] step {:>6} loss {:.3} grad {:.2} lr x{:.2} rows {:.1} ctx {} experts {} | {:.0} chars/s",
                    started.elapsed().as_secs_f64(),
                    s.step.0,
                    s.train_nats,
                    s.grad_norm,
                    s.lr_scale,
                    s.avg_rows,
                    s.context_now,
                    s.n_experts,
                    cps
                );
                (last_print, chars_at_print, window_t) = (s.step.0, s.chars_read.0, Instant::now());
            }
            if s.step.0 >= max_steps || started.elapsed().as_secs_f64() >= max_secs {
                let _ = ctl.send(Ctl::Stop { save: false });
            }
        }
        EngineEvent::Eval(e) => println!(
            "[{:>7.1}s] HELD-OUT {:.4} nats ({:.3} bits/char) +/- {:.4}   train {:.4}",
            started.elapsed().as_secs_f64(),
            e.overall_nats,
            e.overall_nats / std::f64::consts::LN_2,
            e.overall_se,
            e.train_nats_ema
        ),
        EngineEvent::Samples(r) => {
            for i in r.items.iter().take(2) {
                println!(
                    "           [{}] {:?} -> {:?}",
                    i.domain,
                    i.prompt,
                    i.raw.chars().take(80).collect::<String>()
                );
            }
        }
        EngineEvent::Growth(g) => println!("[{:>7.1}s] pool: {g:?}", started.elapsed().as_secs_f64()),
        EngineEvent::Memory(m) => println!(
            "[{:>7.1}s] memory: process {:.2} GB, expert cache {:.2} GB, experts on disk {:.2} GB",
            started.elapsed().as_secs_f64(),
            m.resident_gb,
            m.ram_gb,
            m.disk_gb
        ),
        EngineEvent::Warn(w) => println!("WARNING {}: {}", w.code, w.message),
        EngineEvent::CheckpointSaved(c) => {
            println!("[{:>7.1}s] saved {} ({} experts)", started.elapsed().as_secs_f64(), c.path, c.n_experts)
        }
        _ => {}
    });
    println!("finished: {:?}", rec.outcome);
    Ok(())
}
