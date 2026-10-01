//! Helpers shared by the `store_*` integration tests: fixtures, scratch directories, crc32 and
//! a way to look at the raw `.npy` bytes inside an `.npz`.
#![allow(dead_code)]

use minagi_core::store::checkpoint::{CheckpointData, NamedTensors, OptimState, ParamState, Tensor};
use minagi_core::store::checkpoint::{expected_core_tensors, expected_router_tensors};
use minagi_core::store::manifest::{Manifest, ParsedCfg, Telemetry};
use minagi_core::store::npy::{NpyArray, NpyData};
use minagi_core::store::tiers::{ExpertEntry, ExpertMoments};
use minagi_types::config::ModelConfig;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `tests/fixtures/store/<rel>`.
pub fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store").join(rel)
}

/// A temporary directory removed on drop.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!(
            "minagi-it-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("create scratch dir");
        Scratch(d)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Recursively copy a directory (plain copies, no links).
pub fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let to = dst.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to);
        } else {
            std::fs::copy(e.path(), to).unwrap();
        }
    }
}

/// CRC-32 (IEEE), the checksum Python's `zlib.crc32` computes.
pub fn crc32(bytes: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    });
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c = t[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Raw little-endian bytes of an array's payload (C order), like numpy's `tobytes()`.
pub fn raw_bytes(a: &NpyArray) -> Vec<u8> {
    let mut out = Vec::new();
    match a.data() {
        NpyData::F32(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        NpyData::F64(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        NpyData::I16(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        NpyData::I32(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        NpyData::I64(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        NpyData::U8(v) => out.extend_from_slice(v),
    }
    out
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The raw `.npy` bytes of every entry of an `.npz`, by key (read with the `zip` crate directly, so
/// it does not depend on our own reader).
pub fn npz_entries(path: &Path) -> BTreeMap<String, Vec<u8>> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut z = zip::ZipArchive::new(std::io::BufReader::new(f)).unwrap();
    let mut out = BTreeMap::new();
    for i in 0..z.len() {
        let mut e = z.by_index(i).unwrap();
        let name = e.name().strip_suffix(".npy").unwrap_or(e.name()).to_string();
        let mut buf = Vec::new();
        e.read_to_end(&mut buf).unwrap();
        out.insert(name, buf);
    }
    out
}

// ---- synthetic checkpoint data -------------------------------------------------------------

/// The model of the golden fixtures: d_model 32, 2 heads, 1 prelude + 1 recurrent block, experts of 16 units.
pub fn tiny_model() -> ModelConfig {
    ModelConfig {
        d_model: 32,
        n_head: 2,
        d_ff: 64,
        n_prelude: 1,
        n_recur: 1,
        n_coda: 0,
        max_steps: 4,
        min_steps: 1,
        halt_prior: 0.4,
        halt_thresh: 0.9,
        halt_freeze: true,
        ponder_beta: 0.01,
        bptt_window: 4,
        train_steps_mean: 0.0,
        block: 64,
        vocab_size: 265,
        pool_experts: 5,
        pool_max: 16,
        pool_d_ff: 16,
        pool_depth: 1,
        pool_top_k: 2,
    }
}

/// Deterministic pseudo-random f32 values with a wide dynamic range (so bf16 rounding matters).
pub fn vals(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5677);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mant = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let exp = ((s >> 8) % 24) as i32 - 14;
            mant * 2f32.powi(exp)
        })
        .collect()
}

fn named(want: &[(String, Vec<usize>)], seed: u64) -> NamedTensors {
    want.iter()
        .enumerate()
        .map(|(i, (k, shape))| {
            let n = shape.iter().product();
            (k.clone(), Tensor { shape: shape.clone(), data: vals(seed + i as u64, n) })
        })
        .collect()
}

/// Experts by uid, as `ExpertSource::Entries` wants them.
pub type Experts = Vec<(u64, Arc<ExpertEntry>)>;
/// The original (pre-rounding) f32 moment values of the experts that have moments, in `MOMENT_KEYS` order.
pub type MomentOriginals = BTreeMap<u64, Vec<Vec<f32>>>;

/// A full set of checkpoint data for `model` and the given expert uids (pool order), with
/// arbitrary f32 optimiser moments (not bf16-representable) so rounding is exercised.
pub fn synthetic_data(model: &ModelConfig, uids: &[u64], resident: u32) -> CheckpointData {
    let n = uids.len();
    let parsed = ParsedCfg::from_model(model.clone(), resident);
    let mut manifest = Manifest::new(21, Some(0.875), &parsed);
    let mut tel = Telemetry::fresh(uids);
    tel.usage = (0..n).map(|i| 5.0 + i as f64).collect();
    tel.admits = (0..n).map(|i| (i % 3) as f64).collect();
    tel.born = (0..n).map(|i| (i * 2) as f64).collect();
    tel.last_seen = (0..n).map(|i| (7 + i) as f64).collect();
    tel.ever = (0..n).map(|i| i % 2 == 0).collect();
    tel.segments = Some(33);
    tel.next_uid = Some(uids.iter().max().map_or(0, |m| m + 1) + 2);
    manifest.telemetry = Some(tel);
    manifest.read_chars = Some(987_654);
    manifest.read_nats = Some(1234.5);
    manifest.plasticity = Some(serde_json::json!({"lr_mult": 0.8, "calm": 3}));
    manifest.context_now = Some(40);

    let mut core = named(&expected_core_tensors(model), 1);
    // the embedding is tied: Python stores it twice, under both names, with the same values
    let emb = core.get("tok_emb.weight").unwrap().clone();
    core.insert("head.weight", emb);
    let routers = named(&expected_router_tensors(model, n), 400);

    let mut params = Vec::new();
    for (i, (name, t)) in core.iter().filter(|(k, _)| *k != "head.weight").chain(routers.iter()).enumerate() {
        let m = Tensor { shape: t.shape.clone(), data: vals(900 + i as u64, t.numel()) };
        let v =
            Tensor { shape: t.shape.clone(), data: vals(950 + i as u64, t.numel()).iter().map(|x| x.abs()).collect() };
        params.push(ParamState { name: name.to_string(), m: Some(m), v: Some(v), step: Some(30.0 + i as f64) });
    }
    for pool_slot in ["pool.w1", "pool.w3", "pool.w2"] {
        params.push(ParamState { name: pool_slot.into(), m: None, v: None, step: Some(30.0) });
    }
    CheckpointData { manifest, core, routers, optim: OptimState { params } }
}

/// Experts for `uids`: odd positions without moments, even ones with moments built from arbitrary f32 values.
/// Returns the entries and, for experts with moments, the original f32 moment values (in `MOMENT_KEYS` order).
pub fn synthetic_experts(model: &ModelConfig, uids: &[u64]) -> (Experts, MomentOriginals) {
    let n = (model.d_model * model.pool_d_ff) as usize;
    let mut originals = BTreeMap::new();
    let mut out = Vec::new();
    for (pos, &uid) in uids.iter().enumerate() {
        let moments = (pos % 2 == 0).then(|| {
            let v: Vec<Vec<f32>> = (0..6).map(|k| vals(uid * 1000 + 100 + k, n)).collect();
            let m = ExpertMoments::pack([&v[0], &v[1], &v[2], &v[3], &v[4], &v[5]].map(|x| x.as_slice()));
            originals.insert(uid, v);
            m
        });
        out.push((
            uid,
            Arc::new(ExpertEntry {
                w1: vals(uid * 1000 + 1, n),
                w3: vals(uid * 1000 + 2, n),
                w2: vals(uid * 1000 + 3, n),
                moments,
            }),
        ));
    }
    (out, originals)
}
