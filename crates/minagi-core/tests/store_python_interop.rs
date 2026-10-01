//! Byte-level compatibility in the direction Python -> Rust is covered by `store_golden.rs` and
//! `store_python_import.rs` (they read files the real Python wrote). This file covers the other
//! direction and the full circle: **the real Python code loads what the Rust store wrote**.
//!
//! Each test writes a directory with the Rust store, then runs `tools/golden/store_golden.py verify`
//! with the repository's venv (`.venv/bin/python`, which has CPU torch and numpy). That script calls
//! the reference's own `_load_dir` -> `train.build_paged`, `store._load_optim` and the paged pool's
//! `Tiers`, then compares every tensor with a reference file the test wrote from the values it started
//! with (moments as the original float32 values, so Python's bf16 rounding is compared with ours).
//!
//! The tests are skipped, with a message on stderr, when that interpreter or its torch is not there
//! (set `MINAGI_PYTHON` to use a different one, `MINAGI_REPO` if the crate is built outside the repo).

mod store_support;

use minagi_core::store::bf16;
use minagi_core::store::checkpoint::{Checkpoint, CheckpointData, ExpertSource, ReadOptions, WriteOptions};
use minagi_core::store::npz::{NpzReader, NpzWriter};
use minagi_core::store::python::{ImportOptions, import_python};
use minagi_core::store::tiers::{ExpertEntry, MOMENT_KEYS, expert_file_name, read_expert_file};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};
use store_support::*;

fn repo_root() -> PathBuf {
    if let Ok(p) = std::env::var("MINAGI_REPO") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The interpreter to use, if there is one with torch and numpy.
fn python() -> Option<&'static PathBuf> {
    static PY: OnceLock<Option<PathBuf>> = OnceLock::new();
    PY.get_or_init(|| {
        let root = repo_root();
        let candidate =
            std::env::var("MINAGI_PYTHON").map(PathBuf::from).unwrap_or_else(|_| root.join(".venv/bin/python"));
        if !candidate.exists() || !root.join("tools/golden/store_golden.py").exists() {
            eprintln!("SKIPPED: no Python at {} (the Rust -> Python tests need the repo's .venv)", candidate.display());
            return None;
        }
        let ok = Command::new(&candidate)
            .args(["-c", "import torch, numpy, yaml"])
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("SKIPPED: {} cannot import torch, numpy and yaml", candidate.display());
            return None;
        }
        Some(candidate)
    })
    .as_ref()
}

fn run_script(py: &Path, args: &[&Path], cmd: &str) -> (bool, String) {
    let script = repo_root().join("tools/golden/store_golden.py");
    let out = Command::new(py)
        .arg(&script)
        .arg(cmd)
        .args(args)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run python");
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// The reference file `verify` compares against (see the script's docs for the key scheme).
fn write_reference(
    path: &Path,
    data: &CheckpointData,
    uids: &[u64],
    expert_f32: &BTreeMap<u64, Vec<(String, Vec<f32>)>>,
    fp32_moments: bool,
) {
    let file = std::fs::File::create(path).unwrap();
    let mut z = NpzWriter::new(std::io::BufWriter::new(file));
    for (k, t) in data.core.iter() {
        z.add_slice(&format!("core/{k}"), &t.shape, &t.data).unwrap();
    }
    for (k, t) in data.routers.iter() {
        z.add_slice(&format!("routers/{k}"), &t.shape, &t.data).unwrap();
    }
    for p in &data.optim.params {
        if let (Some(m), Some(v)) = (&p.m, &p.v) {
            z.add_slice(&format!("optim/{}|m", p.name), &m.shape, &m.data).unwrap();
            z.add_slice(&format!("optim/{}|v", p.name), &v.shape, &v.data).unwrap();
        }
        if let Some(t) = p.step {
            z.add_slice(&format!("optim/{}|t", p.name), &[], &[t]).unwrap();
        }
    }
    for (uid, arrays) in expert_f32 {
        for (k, v) in arrays {
            z.add_slice(&format!("expert/{uid}/{k}"), &[v.len()], v).unwrap();
        }
    }
    let tel = data.manifest.telemetry.as_ref().unwrap();
    z.add_slice("meta/uids", &[uids.len()], &uids.iter().map(|&u| u as i64).collect::<Vec<_>>()).unwrap();
    z.add_slice("meta/next_uid", &[], &[tel.next_uid.unwrap() as i64]).unwrap();
    z.add_slice("meta/segments", &[], &[tel.segments.unwrap() as i64]).unwrap();
    if fp32_moments {
        z.add_slice("meta/expert_moments_fp32", &[], &[1i64]).unwrap();
    }
    z.finish().unwrap();
}

/// Weights and (optionally raw) moment values of every expert in a Python-style directory, flat.
fn expert_arrays_from_files(dir: &Path, uids: &[u64]) -> BTreeMap<u64, Vec<(String, Vec<f32>)>> {
    let mut out = BTreeMap::new();
    for &uid in uids {
        let mut z = NpzReader::open(dir.join("experts").join(expert_file_name(uid))).unwrap();
        let mut arrays = Vec::new();
        for k in ["w1", "w3", "w2"].into_iter().chain(MOMENT_KEYS) {
            if !z.contains(k) {
                continue;
            }
            let a = z.get(k).unwrap();
            let v = match a.data() {
                minagi_core::store::NpyData::F32(v) => v.clone(),
                minagi_core::store::NpyData::I16(v) => bf16::unpack(v),
                other => panic!("{k}: {}", other.descr()),
            };
            arrays.push((k.to_string(), v));
        }
        out.insert(uid, arrays);
    }
    out
}

fn assert_python_accepts(py: &Path, dir: &Path, reference: &Path) {
    let (ok, text) = run_script(py, &[dir, reference], "verify");
    assert!(ok && text.contains("\"ok\": true"), "Python rejected {}:\n{text}", dir.display());
}

#[test]
fn python_loads_a_checkpoint_the_rust_store_wrote() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-rust-written");
    let model = tiny_model();
    // uids with holes and out of order relative to their value: files are named by uid
    let uids = [3u64, 5, 6, 10, 11];
    let data = synthetic_data(&model, &uids, 4);
    let (experts, originals) = synthetic_experts(&model, &uids);
    let dst = tmp.join("rust-written");
    Checkpoint::write(&dst, &data, ExpertSource::Entries(&experts)).unwrap();

    // reference: weights as written, moments as the ORIGINAL f32 values (before our bf16 rounding)
    let mut expert_f32 = BTreeMap::new();
    for (uid, e) in &experts {
        let mut arrays =
            vec![("w1".to_string(), e.w1.clone()), ("w3".to_string(), e.w3.clone()), ("w2".to_string(), e.w2.clone())];
        if let Some(orig) = originals.get(uid) {
            for (k, v) in MOMENT_KEYS.iter().zip(orig) {
                arrays.push((k.to_string(), v.clone()));
            }
        }
        expert_f32.insert(*uid, arrays);
    }
    let reference = tmp.join("reference.npz");
    write_reference(&reference, &data, &uids, &expert_f32, false);
    assert_python_accepts(py, &dst, &reference);
}

/// Python's loader cannot take a `null` inside the per-expert arrays (it feeds them to `torch.tensor`),
/// so a non-finite value must be written the way Python writes it: as the bare token `NaN`.
#[test]
fn python_loads_non_finite_telemetry_written_as_python_writes_it() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-nan");
    let model = tiny_model();
    let uids = [3u64, 5, 6];
    let mut data = synthetic_data(&model, &uids, 3);
    {
        let t = data.manifest.telemetry.as_mut().unwrap();
        t.usage[1] = f64::NAN;
        t.last_seen[0] = f64::INFINITY;
        t.recent[2] = f64::NEG_INFINITY;
    }
    let (experts, originals) = synthetic_experts(&model, &uids);
    let dst = tmp.join("ck");
    Checkpoint::write(&dst, &data, ExpertSource::Entries(&experts)).unwrap();
    let text = std::fs::read_to_string(dst.join("manifest.json")).unwrap();
    assert!(text.contains("NaN") && text.contains("Infinity") && !text.contains("null,\n  "), "bare tokens expected");
    let mut expert_f32 = BTreeMap::new();
    for (uid, e) in &experts {
        let mut arrays =
            vec![("w1".to_string(), e.w1.clone()), ("w3".to_string(), e.w3.clone()), ("w2".to_string(), e.w2.clone())];
        if let Some(orig) = originals.get(uid) {
            for (k, v) in MOMENT_KEYS.iter().zip(orig) {
                arrays.push((k.to_string(), v.clone()));
            }
        }
        expert_f32.insert(*uid, arrays);
    }
    let reference = tmp.join("reference.npz");
    write_reference(&reference, &data, &uids, &expert_f32, false);
    assert_python_accepts(py, &dst, &reference);
    // and our own reader gets the same values back
    let back = Checkpoint::read(&dst).unwrap();
    let t = back.data.manifest.telemetry.unwrap();
    assert!(t.usage[1].is_nan());
    assert_eq!((t.last_seen[0], t.recent[2]), (f64::INFINITY, f64::NEG_INFINITY));
}

/// Negative controls: the Python check must fail when a weight is off by one value, or when a moment
/// was rounded the wrong way (truncated instead of round-to-nearest-even). Without these a check
/// that compares nothing would pass.
#[test]
fn the_python_check_is_not_vacuous() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-negative");
    let model = tiny_model();
    let uids = [3u64, 5, 6, 10, 11];
    let data = synthetic_data(&model, &uids, 4);
    let (experts, originals) = synthetic_experts(&model, &uids);
    let mut expert_f32 = BTreeMap::new();
    for (uid, e) in &experts {
        let mut arrays =
            vec![("w1".to_string(), e.w1.clone()), ("w3".to_string(), e.w3.clone()), ("w2".to_string(), e.w2.clone())];
        if let Some(orig) = originals.get(uid) {
            for (k, v) in MOMENT_KEYS.iter().zip(orig) {
                arrays.push((k.to_string(), v.clone()));
            }
        }
        expert_f32.insert(*uid, arrays);
    }
    let reference = tmp.join("reference.npz");
    write_reference(&reference, &data, &uids, &expert_f32, false);

    // (a) one weight of expert 5 is different on disk
    let a = tmp.join("a");
    Checkpoint::write(&a, &data, ExpertSource::Entries(&experts)).unwrap();
    let path = a.join("experts").join(expert_file_name(5));
    let mut e = read_expert_file(&path, 5, 32, 16).unwrap();
    e.w1[7] += 0.5;
    minagi_core::store::tiers::write_expert_file(&path, &e, 32, 16, false).unwrap();
    let (ok, text) = run_script(py, &[&a, &reference], "verify");
    assert!(!ok && text.contains("expert 5 w1 differs"), "a changed weight must be caught:\n{text}");

    // (b) expert 6's moments were truncated to bf16 instead of rounded to nearest even
    let b = tmp.join("b");
    Checkpoint::write(&b, &data, ExpertSource::Entries(&experts)).unwrap();
    let path = b.join("experts").join(expert_file_name(6));
    let mut e = read_expert_file(&path, 6, 32, 16).unwrap();
    let truncate = |v: &[f32]| -> Vec<i16> { v.iter().map(|x| (x.to_bits() >> 16) as u16 as i16).collect() };
    let m = e.moments.as_mut().unwrap();
    m.w1_m = truncate(&originals[&6][0]);
    assert_ne!(m.w1_m, bf16::pack(&originals[&6][0]), "the test data must make rounding matter");
    minagi_core::store::tiers::write_expert_file(&path, &e, 32, 16, false).unwrap();
    let (ok, text) = run_script(py, &[&b, &reference], "verify");
    assert!(!ok && text.contains("expert 6 moment w1_m differs"), "a mis-rounded moment must be caught:\n{text}");
    assert!(!text.contains("expert 6 moment w1_v differs"), "only the truncated array differs:\n{text}");
}

#[test]
fn python_loads_the_rust_rewrite_of_a_python_checkpoint() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-rewrite");
    let src = fixture("py_weights");
    let ck = Checkpoint::read_with(&src, &ReadOptions { slot_moments: true, skip_optim: false }).unwrap();
    let uids = ck.uids();
    let dst = tmp.join("rewritten");
    Checkpoint::write_with(
        &dst,
        &ck.data,
        ExpertSource::Dir(&ck.experts_dir()),
        &WriteOptions { overwrite: false, durable: false },
    )
    .unwrap();
    let reference = tmp.join("reference.npz");
    write_reference(&reference, &ck.data, &uids, &expert_arrays_from_files(&src, &uids), false);
    assert_python_accepts(py, &dst, &reference);
}

#[test]
fn python_loads_an_imported_python_checkpoint() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-import");
    let src = fixture("py_weights");
    let dst = tmp.join("imported");
    import_python(&src, &dst, &ImportOptions::default()).unwrap();
    let ck = Checkpoint::read_with(&dst, &ReadOptions { slot_moments: true, skip_optim: false }).unwrap();
    let uids = ck.uids();
    let reference = tmp.join("reference.npz");
    write_reference(&reference, &ck.data, &uids, &expert_arrays_from_files(&dst, &uids), false);
    assert_python_accepts(py, &dst, &reference);
}

#[test]
fn python_loads_an_imported_non_paged_checkpoint_with_fp32_expert_moments() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-import-nonpaged");
    let dst = tmp.join("imported");
    import_python(&fixture("non_paged_weights"), &dst, &ImportOptions::default()).unwrap();
    let ck = Checkpoint::read(&dst).unwrap();
    let uids = ck.uids();
    assert_eq!(uids, (0..8).collect::<Vec<u64>>());
    let reference = tmp.join("reference.npz");
    // Python passes fp32 moments through unrounded, so the reference holds the raw values
    write_reference(&reference, &ck.data, &uids, &expert_arrays_from_files(&dst, &uids), true);
    assert_python_accepts(py, &dst, &reference);
}

#[test]
fn rust_to_python_load_and_save_and_back_to_rust_changes_nothing() {
    let Some(py) = python() else { return };
    let tmp = Scratch::new("interop-circle");
    let model = tiny_model();
    let uids = [2u64, 3, 7, 8, 40];
    let data = synthetic_data(&model, &uids, 4);
    let (experts, _) = synthetic_experts(&model, &uids);
    let first = tmp.join("rust");
    Checkpoint::write(&first, &data, ExpertSource::Entries(&experts)).unwrap();
    let second = tmp.join("python-resaved");
    let (ok, text) = run_script(py, &[&first, &second], "resave");
    assert!(ok, "Python could not load and re-save the Rust checkpoint:\n{text}");

    let a = Checkpoint::read(&first).unwrap();
    let b = Checkpoint::read(&second).unwrap();
    assert!(b.data.manifest.written_by_python(), "store.save rebuilds the manifest, so our namespace is gone");
    assert!(!b.has_complete_marker);
    assert_eq!(b.uids(), uids);
    // manifest: what the Rust side recorded and Python carries
    let (ma, mb) = (&a.data.manifest, &b.data.manifest);
    assert_eq!((ma.step, ma.val), (mb.step, mb.val));
    assert_eq!((ma.read_chars, ma.read_nats, ma.context_now), (mb.read_chars, mb.read_nats, mb.context_now));
    assert_eq!(ma.plasticity, mb.plasticity);
    assert_eq!((ma.n_experts, ma.d_model, ma.d_ff), (mb.n_experts, mb.d_model, mb.d_ff));
    let (ta, tb) = (ma.telemetry.as_ref().unwrap(), mb.telemetry.as_ref().unwrap());
    assert_eq!((ta.uid.clone(), ta.next_uid, ta.segments), (tb.uid.clone(), tb.next_uid, tb.segments));
    assert_eq!(ta.usage, tb.usage);
    assert_eq!(ta.born, tb.born);
    assert_eq!(ta.ever, tb.ever);
    assert_eq!(mb.parsed_cfg().unwrap().unwrap().model.d_model, 32);
    // tensors: exactly as written
    assert_eq!(a.data.core, b.data.core);
    assert_eq!(a.data.routers, b.data.routers);
    // optimiser state: Python wrote what it loaded, i.e. our bf16 rounding of the original values
    for p in &a.data.optim.params {
        if p.m.is_none() {
            continue; // slot-indexed entries were not written by us
        }
        let q = b.data.optim.get(&p.name).unwrap_or_else(|| panic!("{} lost", p.name));
        assert_eq!(p.m, q.m, "{} m", p.name);
        assert_eq!(p.v, q.v, "{} v", p.name);
        assert_eq!(p.step, q.step, "{} step", p.name);
    }
    // experts: not rewritten by Python (they are already files), so byte-identical
    for uid in uids {
        let f = expert_file_name(uid);
        assert_eq!(
            std::fs::read(first.join("experts").join(&f)).unwrap(),
            std::fs::read(second.join("experts").join(&f)).unwrap()
        );
        let e: ExpertEntry = read_expert_file(&second.join("experts").join(&f), uid, 32, 16).unwrap();
        let orig: &Arc<ExpertEntry> = &experts.iter().find(|(u, _)| *u == uid).unwrap().1;
        assert_eq!(&e, orig.as_ref());
    }
}
