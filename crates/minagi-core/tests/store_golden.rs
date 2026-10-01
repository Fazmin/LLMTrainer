//! The Rust store against files written by the REAL Python reference (`tools/golden/store_golden.py
//! generate` made everything under `tests/fixtures/store/` with `minagi.create`, `minagi.store`,
//! `minagi.paged` and a few real AdamW steps).
//!
//! * `bf16_*`: our float32 -> bf16 rounding equals torch's, on 36,020 bit patterns.
//! * `reads_*`: every array of every file is decoded to exactly the values Python holds (crc32 of
//!   the raw bytes and of the float32-widened values, computed by Python).
//! * `rewrite_*`: reading Python's files and writing them again gives the same `.npy` bytes, entry
//!   by entry, including the bf16 moments.

mod store_support;

use minagi_core::store::bf16;
use minagi_core::store::checkpoint::{Checkpoint, ExpertSource, ReadOptions, WriteOptions, read_bundle, write_bundle};
use minagi_core::store::npy::NpyData;
use minagi_core::store::npz::NpzReader;
use minagi_core::store::tiers::{
    MOMENT_KEYS, TierConfig, Tiers, describe_expert_file, expert_file_name, read_expert_file, write_expert_file,
};
use serde_json::Value;
use std::path::Path;
use store_support::*;

fn expected() -> Value {
    let text = std::fs::read_to_string(fixture("expected.json"))
        .expect("expected.json (run tools/golden/store_golden.py generate)");
    serde_json::from_str(&text).unwrap()
}

// ---- bf16 ---------------------------------------------------------------------------------------

#[test]
fn bf16_rounding_equals_torch_on_every_golden_value() {
    let mut z = NpzReader::open(fixture("bf16_golden.npz")).unwrap();
    let bits = z.get("bits").unwrap();
    let want = z.get("bf16").unwrap();
    let (NpyData::I32(bits), NpyData::I16(want)) = (bits.into_data(), want.into_data()) else {
        panic!("unexpected dtypes")
    };
    assert_eq!(bits.len(), want.len());
    assert!(bits.len() > 30_000);
    let mut ties = 0;
    for (i, (&b, &w)) in bits.iter().zip(&want).enumerate() {
        let x = f32::from_bits(b as u32);
        let got = bf16::f32_to_bf16_bits(x);
        assert_eq!(
            got, w as u16,
            "value #{i}: f32 bits {:#010x} -> ours {got:#06x}, torch {:#06x}",
            b as u32, w as u16
        );
        ties += usize::from((b as u32) & 0xFFFF == 0x8000);
    }
    assert!(ties > 7000, "the fixture must contain thousands of exact ties, has {ties}");
    // and the packing helpers agree with the scalar function
    let f: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b as u32)).collect();
    assert_eq!(bf16::pack(&f), want);
}

#[test]
fn bf16_unpack_equals_torch_for_every_bit_pattern() {
    // torch's view(bfloat16).to(float32) is a shift; the python side proved identical values for all
    // 65,536 patterns when it computed crc_f32 below, so here only the shift has to be right
    for bits in 0..=u16::MAX {
        assert_eq!(bf16::bf16_bits_to_f32(bits).to_bits(), u32::from(bits) << 16);
    }
}

// ---- reading ------------------------------------------------------------------------------------

/// Compare every array in `file` with the dtype / shape / crc recorded by Python.
fn check_npz(path: &Path, want: &Value, label: &str) -> usize {
    let mut z = NpzReader::open(path).unwrap_or_else(|e| panic!("{label}: {e}"));
    let want = want.as_object().unwrap_or_else(|| panic!("{label}: no expectation"));
    let mut keys: Vec<&String> = z.keys().iter().collect();
    keys.sort();
    let mut want_keys: Vec<&String> = want.keys().collect();
    want_keys.sort();
    assert_eq!(keys, want_keys, "{label}: the set of arrays differs");
    let mut checked = 0;
    for key in want.keys() {
        let e = &want[key];
        let a = z.get(key).unwrap_or_else(|er| panic!("{label}/{key}: {er}"));
        let dtype = a.data().descr();
        assert_eq!(dtype, e["dtype"].as_str().unwrap(), "{label}/{key} dtype");
        let shape: Vec<usize> = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(a.shape(), shape.as_slice(), "{label}/{key} shape");
        assert_eq!(u64::from(crc32(&raw_bytes(&a))), e["crc_raw"].as_u64().unwrap(), "{label}/{key} raw bytes");
        let widened: Vec<f32> = match a.data() {
            NpyData::I16(v) => bf16::unpack(v),
            NpyData::F32(v) => v.clone(),
            NpyData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            other => panic!("{label}/{key}: unexpected dtype {}", other.descr()),
        };
        assert_eq!(
            u64::from(crc32(&f32_bytes(&widened))),
            e["crc_f32"].as_u64().unwrap(),
            "{label}/{key} float32 values"
        );
        if let Some(v) = e.get("value") {
            assert_eq!(a.as_f64().unwrap()[0], v.as_f64().unwrap(), "{label}/{key} scalar");
        }
        checked += 1;
    }
    checked
}

#[test]
fn reads_every_array_of_both_python_directories_exactly() {
    let ex = expected();
    let mut total = 0;
    for dir in ["py_weights", "non_paged_weights"] {
        let files = ex[dir]["files"].as_object().unwrap();
        assert!(files.len() >= 10, "{dir}: {} files", files.len());
        for (file, want) in files {
            total += check_npz(&fixture(dir).join(file), want, &format!("{dir}/{file}"));
        }
    }
    assert!(total > 250, "{total} arrays compared");
}

#[test]
fn reads_the_paged_manifest_and_trunk_the_way_python_wrote_them() {
    let ex = expected();
    let man = &ex["py_weights"]["manifest"];
    let ck = Checkpoint::read(&fixture("py_weights")).unwrap();
    assert!(!ck.has_complete_marker, "a Python-written directory has no COMPLETE marker and must still be readable");
    assert!(ck.warnings.is_empty(), "{:?}", ck.warnings);
    let m = &ck.data.manifest;
    assert!(m.written_by_python());
    assert_eq!((m.step, m.val), (Some(9), Some(1.234)));
    assert_eq!((m.n_experts, m.d_model, m.d_ff, m.paged), (10, Some(32), Some(16), true));
    assert_eq!(
        m.uids(),
        [1, 2, 4, 6, 7, 8, 9, 10, 11, 12],
        "files are named by uid; pruned uids 0, 3 and 5 left holes"
    );
    let t = m.telemetry.as_ref().unwrap();
    assert_eq!((t.next_uid, t.segments), (Some(13), Some(10)));
    assert_eq!(t.uid.len(), 10);
    assert_eq!(t.ever.iter().filter(|&&b| b).count(), 5);
    assert_eq!((m.read_chars, m.read_nats, m.context_now), (Some(4321), Some(1234.5), Some(23)));
    assert_eq!(m.plasticity, Some(serde_json::json!({"lr_mult": 0.75, "history": [1, 2]})));
    assert_eq!(m.resident(), Some(4));
    // every top-level key Python wrote is accounted for (typed or kept in `extra`), nothing is invented
    let py_keys: Vec<&String> = man.as_object().unwrap().keys().collect();
    let back: Value = serde_json::from_str(&m.to_json().unwrap()).unwrap();
    for k in py_keys {
        assert!(back.get(k).is_some(), "key {k} lost");
    }
    // cfg -> ModelConfig
    let p = m.parsed_cfg().unwrap().unwrap();
    assert_eq!(p.model, tiny_model_from_python());
    assert!(p.defaulted.is_empty());
    // trunk: 17 tensors in core (the tied embedding is stored twice), routers + gate in routers
    assert_eq!(ck.data.core.len(), 17);
    assert_eq!(ck.data.core.get("tok_emb.weight").unwrap().data, ck.data.core.get("head.weight").unwrap().data);
    let routers: Vec<&str> = ck.data.routers.names().collect();
    assert_eq!(routers, ["recur.0.mlp.depth_emb", "recur.0.mlp.router.weight", "pool.gate"]);
    assert_eq!(ck.data.routers.get("recur.0.mlp.router.weight").unwrap().shape, [10, 32]);
    assert_eq!(ck.data.routers.get("pool.gate").unwrap().shape, [10]);
    // optim: 22 parameters; the slot-indexed pool.w1/w3/w2 keep only their step count by default
    assert_eq!(ck.data.optim.params.len(), 22);
    for slot in ["pool.w1", "pool.w3", "pool.w2"] {
        let s = ck.data.optim.get(slot).unwrap();
        assert!(s.m.is_none() && s.v.is_none());
        assert_eq!(s.step, Some(10.0));
    }
    let emb = ck.data.optim.get("tok_emb.weight").unwrap();
    assert_eq!(emb.step, Some(10.0));
    let opt_expect = ex["py_weights"]["files"]["optim.npz"]["tok_emb.weight|m"]["crc_f32"].as_u64().unwrap();
    assert_eq!(
        u64::from(crc32(&f32_bytes(&emb.m.as_ref().unwrap().data))),
        opt_expect,
        "bf16 moments widen to torch's float32"
    );
    assert!(ck.data.optim.get("head.weight").is_none(), "the tied head has no optimiser entry of its own");
    let full =
        Checkpoint::read_with(&fixture("py_weights"), &ReadOptions { slot_moments: true, skip_optim: false }).unwrap();
    let w1 = full.data.optim.get("pool.w1").unwrap();
    assert_eq!(w1.m.as_ref().unwrap().shape, [4, 16, 32]);
    // the checkpoint is healthy by our own deep check
    assert!(ck.verify().is_empty(), "{:?}", ck.verify());
}

fn tiny_model_from_python() -> minagi_types::config::ModelConfig {
    let mut m = tiny_model();
    // the Python run's own numbers: the pool was created with 8 experts and grew to 12 (the ceiling
    // is recorded as the largest it ever was, 10 at the time of the save), halting froze characters
    m.pool_experts = 8;
    m.pool_max = 10;
    m.pool_top_k = 2;
    m
}

#[test]
fn reads_every_expert_file_through_the_tiers_exactly() {
    let ex = expected();
    let ck = Checkpoint::read(&fixture("py_weights")).unwrap();
    let mut tiers = ck.open_tiers(3, true).unwrap();
    let mut with_moments = 0;
    for uid in ck.uids() {
        let want = &ex["py_weights"]["files"][format!("experts/{}", expert_file_name(uid))];
        let e = tiers.fetch(uid).unwrap();
        let crc_of = |k: &str| want[k]["crc_f32"].as_u64().unwrap();
        assert_eq!(u64::from(crc32(&f32_bytes(&e.w1))), crc_of("w1"), "e{uid} w1");
        assert_eq!(u64::from(crc32(&f32_bytes(&e.w3))), crc_of("w3"), "e{uid} w3");
        assert_eq!(u64::from(crc32(&f32_bytes(&e.w2))), crc_of("w2"), "e{uid} w2");
        match &e.moments {
            Some(m) => {
                with_moments += 1;
                for (k, bits) in m.named() {
                    assert_eq!(u64::from(crc32(&f32_bytes(&bf16::unpack(bits)))), crc_of(k), "e{uid} {k}");
                }
            }
            None => assert!(MOMENT_KEYS.iter().all(|k| want.get(*k).is_none()), "e{uid} has moments on disk"),
        }
    }
    assert_eq!(with_moments, 5, "the trained experts have moments, the newly grown ones do not");
    // Python's manifest claims moments for all ten; the files say otherwise (it writes `true` always)
    assert!(ck.data.manifest.experts.iter().all(|e| e.moments));
    let truth = ck
        .uids()
        .iter()
        .filter(|&&u| describe_expert_file(&ck.experts_dir().join(expert_file_name(u))).unwrap().has_moments)
        .count();
    assert_eq!(truth, 5);
    let r = tiers.report();
    assert_eq!((r.misses, r.hits, r.evictions), (10, 0, 7), "ram capacity 3 over ten experts");
}

#[test]
fn reads_the_non_paged_directory_with_fp32_moments() {
    let ex = expected();
    let ck = Checkpoint::read(&fixture("non_paged_weights")).unwrap();
    let m = &ck.data.manifest;
    assert!(!m.paged && m.telemetry.is_none());
    assert_eq!(m.uids(), (0..8).collect::<Vec<u64>>());
    assert_eq!(m.removed_expert_files, Some(0));
    // pool.gate is in core.npz in this layout and is moved to routers
    assert!(ck.data.routers.contains("pool.gate") && !ck.data.core.contains("pool.gate"));
    assert!(ck.warnings.iter().any(|w| w.contains("non-paged")), "{:?}", ck.warnings);
    // expert files hold fp32 moments: read as bf16, rounded like torch would
    let mut tiers = ck.open_tiers(8, true).unwrap();
    let e = tiers.fetch(3).unwrap();
    let want = &ex["non_paged_weights"]["files"]["experts/e00003.npz"];
    assert_eq!(want["w1_m"]["dtype"], "<f4");
    let mut z = NpzReader::open(ck.experts_dir().join("e00003.npz")).unwrap();
    let NpyData::F32(raw_m) = z.get("w1_m").unwrap().into_data() else { panic!("not fp32") };
    assert_eq!(e.moments.as_ref().unwrap().w1_m, bf16::pack(&raw_m));
    assert!(ck.verify().is_empty(), "{:?}", ck.verify());
}

// ---- rewriting ----------------------------------------------------------------------------------

#[test]
fn rewriting_python_bundles_is_byte_identical_entry_by_entry() {
    let tmp = Scratch::new("rewrite-bundles");
    for dir in ["py_weights", "non_paged_weights"] {
        for f in ["core.npz", "routers.npz"] {
            let src = fixture(dir).join(f);
            let out = tmp.join(&format!("{dir}-{f}"));
            write_bundle(&out, &read_bundle(&src).unwrap()).unwrap();
            let (a, b) = (npz_entries(&src), npz_entries(&out));
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "{dir}/{f}: keys");
            for (k, bytes) in &a {
                assert_eq!(bytes, &b[k], "{dir}/{f}: entry {k} differs from numpy's");
            }
        }
    }
}

#[test]
fn rewriting_python_experts_is_byte_identical_including_bf16_moments() {
    let tmp = Scratch::new("rewrite-experts");
    let ck = Checkpoint::read(&fixture("py_weights")).unwrap();
    let (mut n, mut with_moments) = (0, 0);
    for uid in ck.uids() {
        let src = ck.experts_dir().join(expert_file_name(uid));
        let entry = read_expert_file(&src, uid, 32, 16).unwrap();
        with_moments += usize::from(entry.moments.is_some());
        let out = tmp.join(&expert_file_name(uid));
        write_expert_file(&out, &entry, 32, 16, false).unwrap();
        let (a, b) = (npz_entries(&src), npz_entries(&out));
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "e{uid}: keys");
        for (k, bytes) in &a {
            assert_eq!(bytes, &b[k], "e{uid}: entry {k} differs from numpy's");
        }
        n += 1;
    }
    assert_eq!((n, with_moments), (10, 5));
}

#[test]
fn rewriting_the_whole_python_checkpoint_reproduces_its_optimiser_file_bit_for_bit() {
    // read everything (slot moments included), write a native checkpoint around Python's experts,
    // and compare optim.npz entry by entry: this is the bf16 pack(unpack(x)) == x identity on real data
    let tmp = Scratch::new("rewrite-all");
    let src = fixture("py_weights");
    let ck = Checkpoint::read_with(&src, &ReadOptions { slot_moments: true, skip_optim: false }).unwrap();
    let dst = tmp.join("native");
    let rep = Checkpoint::write_with(
        &dst,
        &ck.data,
        ExpertSource::Dir(&ck.experts_dir()),
        &WriteOptions { overwrite: false, durable: false },
    )
    .unwrap();
    assert_eq!(rep.n_experts, 10);
    for f in ["core.npz", "routers.npz", "optim.npz"] {
        let (a, b) = (npz_entries(&src.join(f)), npz_entries(&dst.join(f)));
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "{f}: keys");
        for (k, bytes) in &a {
            assert_eq!(bytes, &b[k], "{f}: entry {k} differs from what Python wrote");
        }
    }
    // the manifest keeps what Python recorded and adds our namespace
    let out = Checkpoint::read(&dst).unwrap();
    let (a, b) = (&ck.data.manifest, &out.data.manifest);
    assert_eq!(
        (a.step, a.val, a.read_chars, a.read_nats, a.context_now),
        (b.step, b.val, b.read_chars, b.read_nats, b.context_now)
    );
    assert_eq!(a.plasticity, b.plasticity);
    assert_eq!(a.telemetry.as_ref().unwrap().uid, b.telemetry.as_ref().unwrap().uid);
    assert_eq!(a.telemetry.as_ref().unwrap().usage, b.telemetry.as_ref().unwrap().usage);
    assert!(!b.written_by_python() && out.has_complete_marker);
    // moments flags are now the truth rather than Python's blanket `true`
    assert_eq!(b.experts.iter().filter(|e| e.moments).count(), 5);
    // the expert files are the very same bytes
    for uid in out.uids() {
        let f = expert_file_name(uid);
        assert_eq!(
            std::fs::read(src.join("experts").join(&f)).unwrap(),
            std::fs::read(dst.join("experts").join(&f)).unwrap()
        );
    }
    let mut t = Tiers::open(TierConfig::new(dst.join("experts"), 32, 16).read_only(true)).unwrap();
    assert_eq!(t.uids(), out.uids());
    t.fetch(4).unwrap();
}
