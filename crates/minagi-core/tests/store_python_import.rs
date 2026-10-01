//! Importing directories written by the Python reference: the preview (`CompatReport`), the
//! conversion, and the plain-English refusals. The sources are the golden fixtures made by
//! `tools/golden/store_golden.py` (a paged directory with grown and pruned experts, and the older
//! non-paged layout), copied into scratch directories and damaged in specific ways.

mod store_support;

use minagi_core::store::checkpoint::{Checkpoint, GATE_KEY};
use minagi_core::store::npz::NpzReader;
use minagi_core::store::python::{
    ImportOptions, PythonLayout, Severity, check_python_checkpoint, check_python_checkpoint_with, export_to_python,
    import_python,
};
use minagi_core::store::tiers::{expert_file_name, read_expert_file};
use minagi_core::store::{StoreError, bf16, npy};
use minagi_types::config::Preset;
use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use store_support::*;

/// Copy a fixture directory and rewrite its manifest.
fn patched(fixture_dir: &str, scratch: &Scratch, name: &str, f: impl FnOnce(&mut Value)) -> std::path::PathBuf {
    let dir = scratch.join(name);
    copy_dir(&fixture(fixture_dir), &dir);
    let path = dir.join("manifest.json");
    let mut m: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    f(&mut m);
    fs::write(&path, serde_json::to_string_pretty(&m).unwrap()).unwrap();
    dir
}

fn codes(r: &minagi_core::store::python::CompatReport, sev: Severity) -> Vec<&'static str> {
    r.issues.iter().filter(|i| i.severity == sev).map(|i| i.code).collect()
}

fn same_bytes(a: &Path, b: &Path) {
    assert_eq!(fs::read(a).unwrap(), fs::read(b).unwrap(), "{} differs from {}", a.display(), b.display());
}

// ---- the preview ------------------------------------------------------------------------------

#[test]
fn previews_the_paged_python_checkpoint() {
    let r = check_python_checkpoint(&fixture("py_weights")).unwrap();
    assert_eq!(r.layout, PythonLayout::Paged);
    assert_eq!((r.step, r.val, r.n_experts, r.d_model, r.d_ff), (Some(9), Some(1.234), 10, Some(32), Some(16)));
    assert!(r.is_importable(), "{}", r.blocker_message());
    let c = r.config.as_ref().unwrap();
    assert_eq!(
        (c.d_model, c.n_head, c.d_ff, c.n_prelude, c.n_recur, c.n_coda, c.pool_d_ff, c.pool_top_k),
        (32, 2, 64, 1, 1, 0, 16, 2)
    );
    assert!(c.halt_freeze);
    assert_eq!(c.pool_experts, 10, "the pool was created with 8 but holds 10; the number present is what counts");
    assert!(c.pool_max >= 10);
    assert_eq!(r.suggested_preset, Preset::Imported);
    assert_eq!(r.suggested_preset_name(), "imported");
    assert_eq!(
        (r.train_hints.resident, r.train_hints.capacity_factor, r.train_hints.pool_aux),
        (Some(4), Some(1.5), Some(0.01))
    );
    // 17 trunk + 3 router/gate + 22 optimiser parameters; nothing it does not understand
    let by_file = |f: &str| r.mapped.iter().filter(|m| m.file == f).count();
    assert_eq!((by_file("core.npz"), by_file("routers.npz"), by_file("optim.npz")), (17, 3, 22));
    assert!(r.unmapped.is_empty(), "{:?}", r.unmapped);
    let gate = r.mapped.iter().find(|m| m.name == GATE_KEY && m.file == "routers.npz").unwrap();
    assert_eq!((gate.shape.clone(), gate.dtype.as_str()), (vec![10], "<f4"));
    assert_eq!((r.experts.expected, r.experts.ok, r.experts.with_moments, r.experts.fp32_moments), (10, 10, 5, 0));
    assert!(r.experts.missing.is_empty() && r.experts.damaged.is_empty() && r.experts.orphans.is_empty());
    assert_eq!((r.optim.present, r.optim.params, r.optim.slot_entries), (true, 22, 3));
    assert!(codes(&r, Severity::Note).contains(&"slot_moments") && codes(&r, Severity::Note).contains(&"pool_experts"));
    assert!(codes(&r, Severity::Blocker).is_empty());
    assert!(r.total_bytes > 300_000);
    let s = r.summary();
    assert!(s.contains("step 9") && s.contains("10 experts") && s.contains("32-wide"), "{s}");
    // issues are sorted: blockers, then warnings, then notes
    assert!(r.issues.windows(2).all(|w| w[0].severity <= w[1].severity));
    // the report can be sent to the UI as JSON
    let j = serde_json::to_value(&r).unwrap();
    assert_eq!(j["layout"], "paged");
    assert_eq!(j["suggested_preset"], "imported");
}

#[test]
fn a_manifest_shaped_like_a_preset_suggests_that_preset() {
    let tmp = Scratch::new("preset");
    let dir = patched("py_weights", &tmp, "p", |m| {
        let full = Preset::Full.model();
        let cfg = m["cfg"].as_object_mut().unwrap();
        for (k, v) in [
            ("d_model", json!(full.d_model)),
            ("n_head", json!(full.n_head)),
            ("d_ff", json!(full.d_ff)),
            ("n_prelude", json!(full.n_prelude)),
            ("max_steps", json!(full.max_steps)),
            ("pool_d_ff", json!(full.pool_d_ff)),
            ("pool_top_k", json!(full.pool_top_k)),
        ] {
            cfg.insert(k.into(), v);
        }
    });
    let r = check_python_checkpoint(&dir).unwrap();
    assert_eq!(r.suggested_preset, Preset::Full);
    assert_eq!(r.suggested_preset_name(), "full");
    // the tensors are tiny, so it is not importable, and the report says what is wrong in plain words
    assert!(!r.is_importable());
    assert!(r.issues.iter().any(|i| i.code == "tensor_mismatch" && i.message.contains("tok_emb.weight")));
}

// ---- converting -------------------------------------------------------------------------------

#[test]
fn imports_the_paged_checkpoint_byte_for_byte() {
    let tmp = Scratch::new("import-paged");
    let src = fixture("py_weights");
    let dst = tmp.join("imported");
    let r = import_python(&src, &dst, &ImportOptions::default()).unwrap();
    assert!(r.is_importable());
    assert!(dst.join("COMPLETE").is_file(), "the app lists a checkpoint only if it has the marker");
    for f in ["core.npz", "routers.npz", "optim.npz"] {
        same_bytes(&src.join(f), &dst.join(f));
    }
    for uid in [1, 2, 4, 6, 7, 8, 9, 10, 11, 12] {
        let f = expert_file_name(uid);
        same_bytes(&src.join("experts").join(&f), &dst.join("experts").join(&f));
    }
    assert_eq!(fs::read_dir(dst.join("experts")).unwrap().count(), 10);
    let ck = Checkpoint::read(&dst).unwrap();
    assert!(ck.has_complete_marker);
    let m = &ck.data.manifest;
    let rs = m.minagi_rs.as_ref().unwrap();
    assert_eq!((rs.origin.as_deref(), rs.preset.as_deref(), rs.format), (Some("python-import"), Some("imported"), 1));
    assert_eq!(rs.imported_from.as_deref(), Some(src.display().to_string().as_str()));
    assert!(rs.notes.iter().any(|n| n.contains("pool was created with 8")), "{:?}", rs.notes);
    // what Python recorded survives
    let orig = Checkpoint::read(&src).unwrap();
    let o = &orig.data.manifest;
    assert_eq!(
        (m.step, m.val, m.read_chars, m.read_nats, m.context_now),
        (o.step, o.val, o.read_chars, o.read_nats, o.context_now)
    );
    assert_eq!(m.plasticity, o.plasticity);
    assert_eq!(m.telemetry.as_ref().unwrap().uid, o.telemetry.as_ref().unwrap().uid);
    assert_eq!(m.telemetry.as_ref().unwrap().born, o.telemetry.as_ref().unwrap().born);
    assert_eq!(m.telemetry.as_ref().unwrap().next_uid, Some(13));
    assert_eq!(m.n_experts, 10);
    assert_eq!(m.experts.iter().filter(|e| e.moments).count(), 5, "the moments flags are now the truth");
    let cfg = m.cfg.as_ref().unwrap();
    assert_eq!(cfg["pool_experts"], 10);
    assert_eq!(cfg["halt_freeze"], true);
    assert_eq!(cfg["pool_resident"], 4);
    assert!(ck.verify().is_empty(), "{:?}", ck.verify());
    let info = Checkpoint::inspect(&dst).unwrap();
    assert!(info.has_complete_marker && !info.written_by_python);
    assert_eq!(info.origin, "python-import");
    assert_eq!(info.n_experts, 10);
    // the source is untouched, and nothing temporary is left beside the result
    assert!(!src.join("COMPLETE").exists());
    let mut siblings: Vec<String> =
        fs::read_dir(tmp.path()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into()).collect();
    siblings.sort();
    assert_eq!(siblings, ["imported"]);
    // refuses to overwrite, unless told to
    let err = import_python(&src, &dst, &ImportOptions::default()).unwrap_err();
    assert!(matches!(err, StoreError::AlreadyExists(_)), "{err:?}");
    import_python(&src, &dst, &ImportOptions { overwrite: true, ..Default::default() }).unwrap();
    assert!(dst.join("COMPLETE").is_file());
}

#[test]
fn imports_the_non_paged_layout_moving_the_gate_and_synthesising_telemetry() {
    let tmp = Scratch::new("import-nonpaged");
    let src = fixture("non_paged_weights");
    let r = check_python_checkpoint(&src).unwrap();
    assert_eq!(r.layout, PythonLayout::NonPaged);
    assert!(r.is_importable(), "{}", r.blocker_message());
    assert_eq!((r.experts.ok, r.experts.with_moments, r.experts.fp32_moments), (8, 8, 8));
    let notes = codes(&r, Severity::Note);
    assert!(notes.contains(&"layout_non_paged") && notes.contains(&"fp32_moments"), "{notes:?}");
    assert!(r.mapped.iter().any(|m| m.file == "core.npz" && m.name == GATE_KEY), "the gate is found in core.npz");
    assert!(r.unmapped.is_empty(), "{:?}", r.unmapped);

    let dst = tmp.join("imported");
    import_python(&src, &dst, &ImportOptions::default()).unwrap();
    let ck = Checkpoint::read(&dst).unwrap();
    assert!(ck.warnings.is_empty(), "the gate was already moved: {:?}", ck.warnings);
    let z = NpzReader::open(dst.join("core.npz")).unwrap();
    assert!(!z.contains(GATE_KEY));
    let mut z = NpzReader::open(dst.join("routers.npz")).unwrap();
    assert!(z.contains(GATE_KEY));
    let gate = z.get(GATE_KEY).unwrap();
    let m = &ck.data.manifest;
    assert!(m.paged && m.removed_expert_files.is_none());
    let t = m.telemetry.as_ref().expect("telemetry is synthesised");
    assert_eq!(t.uid, (0..8).collect::<Vec<u64>>());
    assert_eq!(t.next_uid, Some(8));
    // (serde_json's default float parser can be one ulp off for 17-digit values, hence a tolerance)
    for (a, b) in t.gate.iter().zip(gate.as_f32().unwrap()) {
        assert!((a - f64::from(*b)).abs() < 1e-12, "{a} vs {b}");
    }
    assert_eq!(t.gate.len(), 8);
    assert!(m.experts.iter().all(|e| e.moments && e.file.ends_with(".npz")));
    // the experts are the same files, with fp32 moments that read as bf16
    for uid in 0..8u64 {
        let f = expert_file_name(uid);
        same_bytes(&src.join("experts").join(&f), &dst.join("experts").join(&f));
    }
    let e = read_expert_file(&dst.join("experts/e00002.npz"), 2, 32, 16).unwrap();
    let mut zz = NpzReader::open(dst.join("experts/e00002.npz")).unwrap();
    let raw = zz.get("w2_v").unwrap();
    assert_eq!(e.moments.unwrap().w2_v, bf16::pack(raw.as_f32().unwrap()));
    assert!(ck.verify().is_empty(), "{:?}", ck.verify());
    assert_eq!(ck.data.optim.get(GATE_KEY).map(|p| p.step), Some(Some(2.0)));
}

#[test]
fn converts_the_oldest_flat_npy_expert_layout() {
    let tmp = Scratch::new("import-flat");
    let dir = patched("non_paged_weights", &tmp, "flat", |m| {
        for e in m["experts"].as_array_mut().unwrap() {
            let f = e["file"].as_str().unwrap().replace(".npz", ".npy");
            e["file"] = json!(f);
        }
    });
    // rewrite every expert as one flat float32 vector: w1 | w3 | w2
    let mut originals = Vec::new();
    for uid in 0..8u64 {
        let p = dir.join("experts").join(expert_file_name(uid));
        let e = read_expert_file(&p, uid, 32, 16).unwrap();
        let flat: Vec<f32> = e.w1.iter().chain(&e.w3).chain(&e.w2).copied().collect();
        let mut bytes = Vec::new();
        let n = flat.len();
        npy::write_npy(&mut bytes, &npy::NpyArray::f32(vec![n], flat).unwrap()).unwrap();
        fs::remove_file(&p).unwrap();
        fs::write(dir.join("experts").join(format!("e{uid:05}.npy")), bytes).unwrap();
        originals.push(e);
    }
    let r = check_python_checkpoint(&dir).unwrap();
    assert_eq!(r.layout, PythonLayout::FlatExperts);
    assert!(r.is_importable(), "{}", r.blocker_message());
    assert!(codes(&r, Severity::Note).contains(&"layout_flat_experts"));
    let dst = tmp.join("out");
    import_python(&dir, &dst, &ImportOptions::default()).unwrap();
    for (uid, orig) in originals.iter().enumerate() {
        let got =
            read_expert_file(&dst.join("experts").join(expert_file_name(uid as u64)), uid as u64, 32, 16).unwrap();
        assert_eq!((&got.w1, &got.w3, &got.w2), (&orig.w1, &orig.w3, &orig.w2), "expert {uid}");
        assert!(got.moments.is_none(), "flat files carry no moments");
    }
    assert!(Checkpoint::read(&dst).unwrap().verify().is_empty());
    // a flat file of the wrong length is reported as damaged, naming the expert
    fs::write(dir.join("experts/e00003.npy"), {
        let mut b = Vec::new();
        npy::write_npy(&mut b, &npy::NpyArray::f32(vec![5], vec![0.0; 5]).unwrap()).unwrap();
        b
    })
    .unwrap();
    let bad = check_python_checkpoint(&dir).unwrap();
    assert!(bad.issues.iter().any(|i| i.code == "expert_damaged" && i.message.contains("e00003")));
}

// ---- refusing ---------------------------------------------------------------------------------

/// Patch the cfg, check the preview blocks with `code` and a message containing `needle`, and that
/// `import_python` refuses with that same text and leaves nothing behind.
fn assert_refused(tag: &str, patch: impl FnOnce(&mut serde_json::Map<String, Value>), code: &str, needle: &str) {
    let tmp = Scratch::new(tag);
    let dir = patched("py_weights", &tmp, "src", |m| patch(m["cfg"].as_object_mut().unwrap()));
    let r = check_python_checkpoint(&dir).unwrap();
    assert!(!r.is_importable(), "{tag}: should not be importable");
    let hit = r
        .blockers()
        .into_iter()
        .find(|i| i.code == code)
        .unwrap_or_else(|| panic!("{tag}: no blocker {code}: {:?}", r.issues));
    assert!(hit.message.contains(needle), "{tag}: {:?} lacks {needle:?}", hit.message);
    let dst = tmp.join("dst");
    let err = import_python(&dir, &dst, &ImportOptions::default()).unwrap_err();
    assert!(matches!(&err, StoreError::Incompatible(m) if m.contains(needle)), "{tag}: {err:?}");
    assert!(!dst.exists(), "{tag}: nothing may be left at the destination");
    assert!(!tmp.join(".tmp-dst").exists(), "{tag}: nor a temporary directory");
}

#[test]
fn refuses_architectures_the_engine_does_not_run_in_plain_english() {
    assert_refused(
        "n_recur",
        |c| {
            c.insert("n_recur".into(), json!(3));
        },
        "recurrent_blocks",
        "3 recurrent blocks per row",
    );
    assert_refused(
        "n_coda",
        |c| {
            c.insert("n_coda".into(), json!(1));
        },
        "coda_blocks",
        "1 coda",
    );
    assert_refused(
        "use_pool",
        |c| {
            c.insert("use_pool".into(), json!(false));
        },
        "unsupported_setting",
        "no expert pool",
    );
    assert_refused(
        "tie",
        |c| {
            c.insert("tie_embeddings".into(), json!(false));
        },
        "unsupported_setting",
        "separate input and output",
    );
    assert_refused(
        "rope",
        |c| {
            c.insert("rope_theta".into(), json!(500000.0));
        },
        "unsupported_setting",
        "rotary base",
    );
    assert_refused(
        "depth",
        |c| {
            c.insert("pool_depth".into(), json!(2));
        },
        "expert_depth",
        "stacks 2 blocks",
    );
    assert_refused(
        "vocab",
        |c| {
            c.insert("vocab_size".into(), json!(8192));
        },
        "vocabulary",
        "8192 tokens",
    );
    assert_refused(
        "heads",
        |c| {
            c.insert("n_head".into(), json!(5));
        },
        "invalid_config",
        "divisible",
    );
}

#[test]
fn halt_freeze_off_is_refused_unless_the_user_chooses_to_override_it() {
    let tmp = Scratch::new("halt");
    let dir = patched("py_weights", &tmp, "src", |m| {
        m["cfg"]["halt_freeze"] = json!(false);
    });
    let r = check_python_checkpoint(&dir).unwrap();
    let b = r.blockers();
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].code, "halt_freeze");
    assert!(b[0].message.contains("halt_freeze off") && b[0].message.contains("import it anyway"), "{}", b[0].message);
    assert!(r.blocker_message().starts_with("This checkpoint cannot be imported: "));
    assert!(import_python(&dir, &tmp.join("a"), &ImportOptions::default()).is_err());
    // with the override: a warning, and the imported manifest records the freeze
    let opts = ImportOptions { override_halt_freeze: true, overwrite: false };
    let r2 = check_python_checkpoint_with(&dir, &opts).unwrap();
    assert!(r2.is_importable());
    assert!(codes(&r2, Severity::Warning).contains(&"halt_freeze"));
    assert!(r2.config.as_ref().unwrap().halt_freeze);
    let dst = tmp.join("b");
    import_python(&dir, &dst, &opts).unwrap();
    assert_eq!(Checkpoint::read(&dst).unwrap().data.manifest.cfg.as_ref().unwrap()["halt_freeze"], true);
}

#[test]
fn several_problems_are_listed_together() {
    let tmp = Scratch::new("several");
    let dir = patched("py_weights", &tmp, "src", |m| {
        m["cfg"]["n_recur"] = json!(2);
        m["cfg"]["n_coda"] = json!(1);
    });
    let r = check_python_checkpoint(&dir).unwrap();
    let msg = r.blocker_message();
    assert!(
        msg.starts_with("This checkpoint cannot be imported (")
            && msg.contains("recurrent blocks")
            && msg.contains("coda"),
        "{msg}"
    );
    assert!(msg.lines().count() >= 3);
}

#[test]
fn reports_missing_damaged_and_mismatched_files() {
    let tmp = Scratch::new("damage");
    let dir = patched("py_weights", &tmp, "src", |_| {});
    fs::remove_file(dir.join("experts/e00004.npz")).unwrap();
    let bytes = fs::read(dir.join("experts/e00006.npz")).unwrap();
    fs::write(dir.join("experts/e00006.npz"), &bytes[..bytes.len() / 3]).unwrap();
    // a core.npz without the adapter, and a gate of the wrong length
    let mut core = minagi_core::store::checkpoint::read_bundle(&dir.join("core.npz")).unwrap();
    core.remove("adapter.weight");
    minagi_core::store::checkpoint::write_bundle(&dir.join("core.npz"), &core).unwrap();
    let mut routers = minagi_core::store::checkpoint::read_bundle(&dir.join("routers.npz")).unwrap();
    routers.insert(GATE_KEY, minagi_core::store::checkpoint::Tensor::zeros(vec![9]));
    minagi_core::store::checkpoint::write_bundle(&dir.join("routers.npz"), &routers).unwrap();

    let r = check_python_checkpoint(&dir).unwrap();
    assert_eq!(r.experts.missing, [4]);
    assert_eq!(r.experts.damaged.len(), 1);
    assert_eq!(r.experts.damaged[0].0, 6);
    let msgs: Vec<&str> = r.blockers().iter().map(|i| i.message.as_str()).collect();
    assert!(
        msgs.iter().any(|m| m.contains("1 of the 10 experts have no file") && m.contains("e00004.npz")),
        "{msgs:?}"
    );
    assert!(msgs.iter().any(|m| m.contains("e00006.npz") && m.contains("cannot be used")), "{msgs:?}");
    assert!(msgs.iter().any(|m| m.contains("adapter.weight")), "{msgs:?}");
    assert!(msgs.iter().any(|m| m.contains("pool.gate") && m.contains("[9]") && m.contains("[10]")), "{msgs:?}");
    let err = import_python(&dir, &tmp.join("dst"), &ImportOptions::default()).unwrap_err().to_string();
    assert!(err.contains("e00004.npz"), "{err}");
}

#[test]
fn not_a_weights_folder_is_an_error_with_a_clear_message() {
    let tmp = Scratch::new("notweights");
    let e = check_python_checkpoint(tmp.path()).unwrap_err().to_string();
    assert!(e.contains("no manifest.json") && e.contains("Python mini-AGI"), "{e}");
    let e = check_python_checkpoint(&tmp.join("nope")).unwrap_err().to_string();
    assert!(e.contains("not a folder"), "{e}");
    fs::write(tmp.join("manifest.json"), "{ not json").unwrap();
    let e = check_python_checkpoint(tmp.path()).unwrap_err().to_string();
    assert!(e.contains("manifest.json problem"), "{e}");
    // a manifest without a model configuration is a blocker, not a crash
    fs::write(tmp.join("manifest.json"), r#"{"step": 1, "cfg": null, "n_experts": 0}"#).unwrap();
    let r = check_python_checkpoint(tmp.path()).unwrap();
    let c: Vec<&str> = r.blockers().iter().map(|i| i.code).collect();
    assert!(c.contains(&"no_config") && c.contains(&"no_experts") && c.contains(&"bundle_missing"), "{c:?}");
}

#[test]
fn unknown_tensors_and_orphan_experts_are_reported_and_copied_along() {
    let tmp = Scratch::new("unknown");
    let dir = patched("py_weights", &tmp, "src", |_| {});
    // an older architecture's leftover in routers.npz, and an expert file the pool does not list
    let mut routers = minagi_core::store::checkpoint::read_bundle(&dir.join("routers.npz")).unwrap();
    routers.insert("pool.segment_router.weight", minagi_core::store::checkpoint::Tensor::zeros(vec![3, 32]));
    minagi_core::store::checkpoint::write_bundle(&dir.join("routers.npz"), &routers).unwrap();
    fs::copy(dir.join("experts/e00001.npz"), dir.join("experts/e00099.npz")).unwrap();
    let r = check_python_checkpoint(&dir).unwrap();
    assert!(r.is_importable(), "{}", r.blocker_message());
    assert_eq!(r.unmapped.len(), 1);
    assert_eq!(r.unmapped[0].name, "pool.segment_router.weight");
    assert!(r.unmapped[0].reason.contains("older architecture"));
    assert_eq!(r.experts.orphans, [99]);
    assert!(codes(&r, Severity::Note).contains(&"orphan_experts"));
    let dst = tmp.join("dst");
    import_python(&dir, &dst, &ImportOptions::default()).unwrap();
    same_bytes(&dir.join("routers.npz"), &dst.join("routers.npz"));
    assert!(!dst.join("experts/e00099.npz").exists(), "an expert the pool does not list is not part of the model");
    assert!(Checkpoint::read(&dst).unwrap().data.routers.contains("pool.segment_router.weight"));
}

// ---- exporting --------------------------------------------------------------------------------

#[test]
fn export_to_python_is_a_copy_without_the_marker() {
    let tmp = Scratch::new("export");
    let imported = tmp.join("ck");
    import_python(&fixture("py_weights"), &imported, &ImportOptions::default()).unwrap();
    let out = tmp.join("for-python");
    export_to_python(&imported, &out).unwrap();
    assert!(!out.join("COMPLETE").exists());
    for f in ["manifest.json", "core.npz", "routers.npz", "optim.npz"] {
        same_bytes(&imported.join(f), &out.join(f));
    }
    assert_eq!(fs::read_dir(out.join("experts")).unwrap().count(), 10);
    assert!(matches!(export_to_python(&imported, &out), Err(StoreError::AlreadyExists(_))));
    // the exported folder is a valid checkpoint for our own reader (which does not need the marker)
    let back = Checkpoint::read(&out).unwrap();
    assert!(!back.has_complete_marker && back.verify().is_empty());
    // a broken checkpoint is not exported
    fs::remove_file(imported.join("experts/e00004.npz")).unwrap();
    let err = export_to_python(&imported, &tmp.join("again")).unwrap_err().to_string();
    assert!(err.contains("cannot be exported") && err.contains("no file"), "{err}");
    assert!(!tmp.join("again").exists() && !tmp.join(".tmp-again").exists());
}
