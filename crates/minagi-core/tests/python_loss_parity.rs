//! A model trained and saved by the Rust engine must score text the same way in the original Python implementation.
//! This is the whole-model check that complements the per-operation golden fixtures: it exercises weight naming and
//! layout, expert files, routing, halting and the loss together, on weights that training has really changed.
//! Skipped (with a note) when the Python virtual environment is not there.

use std::path::PathBuf;
use std::process::Command;

use minagi_core::candle_core::Device;
use minagi_core::model::{FwdIn, ModelOpts};
use minagi_core::ops::{self, Kernel};
use minagi_core::optim::adamw::{AdamHyper, AdamW};
use minagi_core::train::session::{RunPaths, Saved, create_model, write_checkpoint};
use minagi_core::train::step::{StepSettings, train_on_window};
use minagi_types::{MoeMode, Preset};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(n: usize, mul: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * mul + 3) % 60 + 97) as u32).collect()
}

#[test]
fn a_trained_and_saved_model_scores_the_same_in_the_python_reference() {
    let py = repo().join(".venv/bin/python");
    let script = repo().join("tools/golden/ref_loss.py");
    if !py.exists() || !repo().join("tools/golden/ref/train.py").exists() {
        eprintln!("skipping: the Python reference environment is not set up");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let paths = RunPaths::new(dir.path());
    let mut cfg = Preset::Tiny.model();
    cfg.d_model = 32;
    cfg.n_head = 2;
    cfg.d_ff = 64;
    cfg.n_prelude = 1;
    cfg.max_steps = 4;
    cfg.bptt_window = 4;
    cfg.train_steps_mean = 0.0;
    cfg.halt_prior = 0.4;
    cfg.block = 64;
    cfg.pool_experts = 8;
    cfg.pool_max = 8;
    cfg.pool_d_ff = 16;
    cfg.pool_top_k = 2;
    let mut train = Preset::Tiny.train();
    train.resident = 4;
    train.ram_cache = 4;
    train.capacity_factor = 1.5; // what the Python reference reads from its config.yaml
    let opts =
        ModelOpts { kernel: Kernel::hand(), capacity_factor: 1.5, moe_mode: MoeMode::SparseDispatch, z_weight: 1e-3 };
    let mut model = create_model(&Device::Cpu, &cfg, &train, opts, &paths, 11).unwrap();
    model.pool.book.explore_bias = 0.65;
    let mut opt = AdamW::new(AdamHyper::default(), 0.1);
    let s = StepSettings { lr: 4e-3, trunk_lr_mult: 0.5, clip: 1.0, pool_aux: 0.01, row_ckpt: false };
    // train on a few different windows so the router, the gates and the experts all move
    for k in 0..40 {
        let (x, y) = (text(48, 3 + k % 5), text(48, 7 + k % 3));
        train_on_window(&mut model, &mut opt, &x, &y, 4, 1.0, &s).unwrap();
    }
    model.pool.book.explore_bias = 0.0;
    let ckpt = paths.checkpoints().join("c");
    std::fs::create_dir_all(paths.checkpoints()).unwrap();
    write_checkpoint(&mut model, &opt, &Saved { step: 40, ..Default::default() }, None, &ckpt).unwrap();

    // the text to score, and what each engine says about it
    let (x, y) = (text(40, 5), text(40, 9));
    let ids_path = dir.path().join("ids.json");
    std::fs::write(&ids_path, serde_json::json!({ "x": x, "y": y }).to_string()).unwrap();
    let python_loss = |capacity: Option<f64>| -> f64 {
        let mut cmd = Command::new(&py);
        cmd.arg(&script).arg(&ckpt).arg(&ids_path);
        if let Some(c) = capacity {
            cmd.arg(c.to_string());
        }
        let res = cmd.output().expect("run python");
        assert!(res.status.success(), "python failed: {}", String::from_utf8_lossy(&res.stderr));
        let stdout = String::from_utf8_lossy(&res.stdout);
        let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{')).expect("a json line from python");
        serde_json::from_str::<serde_json::Value>(line).unwrap()["loss"].as_f64().unwrap()
    };
    for capacity in [0.0, 1.5] {
        model.opts.capacity_factor = capacity;
        let out = model
            .forward(FwdIn {
                ids: &x,
                targets: Some(&y),
                caches: None,
                pos_offset: 0,
                n_steps: 4,
                train: false,
                want_logits: false,
            })
            .unwrap();
        let rust_loss = f64::from(ops::scalar(&out.loss.unwrap()).unwrap());
        assert!(rust_loss.is_finite() && rust_loss > 0.0);
        let py_loss = python_loss(Some(capacity));
        eprintln!("capacity {capacity}: Rust engine {rust_loss:.6}, Python reference {py_loss:.6}");
        // With no capacity bound nothing depends on the order of tokens, so the two agree to float rounding; with a bound
        // the reference's token order is unspecified (its sort is unstable), so a dropped assignment may differ.
        let tol = if capacity == 0.0 { 2e-5 } else { 5e-3 };
        assert!(
            (py_loss - rust_loss).abs() < tol * rust_loss.max(1.0),
            "capacity {capacity}: the Python reference scores this text {py_loss}, the Rust engine {rust_loss}"
        );
    }
}
