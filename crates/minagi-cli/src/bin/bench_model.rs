//! Times whole training steps of a preset's model at a fixed number of rows, to separate per-row cost from fixed cost.
//!   bench_model [--preset tiny|small] [--device cpu|metal] [--t 1024] [--moe sparse|dense] [--resident N]

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use minagi_core::candle_core::Device;
use minagi_core::model::{Model, ModelOpts};
use minagi_core::moe::MemStore;
use minagi_core::ops::Kernel;
use minagi_core::optim::adamw::{AdamHyper, AdamW};
use minagi_core::train::step::{StepSettings, train_on_window};
use minagi_types::{MoeMode, Preset};

fn main() -> Result<()> {
    let mut kv: HashMap<String, String> = HashMap::new();
    let rest: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < rest.len() {
        if let Some(k) = rest[i].strip_prefix("--") {
            kv.insert(k.to_string(), rest[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    let preset = Preset::parse(kv.get("preset").map(String::as_str).unwrap_or("small")).unwrap();
    let dev = match kv.get("device").map(String::as_str) {
        Some("cpu") => Device::Cpu,
        _ => Device::new_metal(0)?,
    };
    let t: usize = kv.get("t").and_then(|v| v.parse().ok()).unwrap_or(1024);
    let cfg = preset.model();
    let mut train = preset.train();
    if let Some(r) = kv.get("resident").and_then(|v| v.parse().ok()) {
        train.resident = r;
    }
    let mode = match kv.get("moe").map(String::as_str) {
        Some("dense") => MoeMode::DenseMasked,
        Some("sparse") => MoeMode::SparseDispatch,
        _ => train.moe_mode,
    };
    let mut kernel = if kv.contains_key("composed") { Kernel::plain() } else { Kernel::hand() };
    kernel.sync_tiles = kv.contains_key("sync");
    if let Some(q) = kv.get("qblock").and_then(|v| v.parse().ok()) {
        kernel.q_block = q;
        kernel.ckpt = true;
    }
    let opts = ModelOpts { kernel, capacity_factor: train.capacity_factor, moe_mode: mode, z_weight: 1e-3 };
    println!("before the model: {:.1} GB", minagi_core::backend::process_memory_gb().0);
    let mut model = Model::create(&dev, cfg.clone(), opts, train.resident as usize, Box::new(MemStore::new()), 1)?;
    model.pool.book.explore_bias = train.explore_bias;
    println!("model created: {:.1} GB", minagi_core::backend::process_memory_gb().0);
    let mut opt = AdamW::new(AdamHyper::default(), 0.1);
    let s =
        StepSettings { lr: 1e-4, trunk_lr_mult: 0.25, clip: 1.0, pool_aux: 0.01, row_ckpt: kv.contains_key("ckpt") };
    let ids: Vec<u32> = (0..t as u32).map(|i| (i * 7 + 3) % 200 + 40).collect();
    let tg: Vec<u32> = (0..t as u32).map(|i| (i * 11 + 5) % 200 + 40).collect();
    println!(
        "{} T={t} moe {mode:?} resident {} kernel hand={}",
        preset.display_name(),
        train.resident,
        !kv.contains_key("composed")
    );
    for rows in [1usize, 2, 4, 6, 12] {
        if rows > cfg.max_steps as usize {
            continue;
        }
        for _ in 0..2 {
            minagi_core::backend::pool(|| train_on_window(&mut model, &mut opt, &ids, &tg, rows, 1.0, &s))?;
        }
        let (mut tot, mut f, mut b, mut r, mut u) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let n = 6;
        for _ in 0..n {
            let t0 = Instant::now();
            let o = minagi_core::backend::pool(|| train_on_window(&mut model, &mut opt, &ids, &tg, rows, 1.0, &s))?;
            tot += t0.elapsed().as_secs_f64();
            (f, b, r, u) = (f + o.timing.forward, b + o.timing.backward, r + o.timing.readback, u + o.timing.update);
        }
        let n = n as f64;
        println!(
            "rows {rows}: step {:6.1} ms  (forward {:5.1}, backward {:5.1}, readback {:5.1}, update {:5.1})",
            tot / n * 1e3,
            f / n * 1e3,
            b / n * 1e3,
            r / n * 1e3,
            u / n * 1e3
        );
        let (now, peak) = minagi_core::backend::process_memory_gb();
        println!("         memory now {now:.1} GB, peak so far {peak:.1} GB");
    }
    Ok(())
}
