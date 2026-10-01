//! `manifest.json`: the index of a checkpoint directory.
//!
//! The format is the one the Python reference writes in `minagi/store.py::_save_paged` and reads
//! in `store.load` / `train.build_paged`; this module reads and writes exactly that, so a manifest
//! written here is loadable by the Python code and a Python manifest is loadable here.
//!
//! ```json
//! {
//!  "step": 2, "val": 1.5,
//!  "cfg": { "d_model": 32, "n_head": 2, ..., "pool_resident": 4, "pool_max": 8, "pool_ever": [...] },
//!  "telemetry": { "gate": [...], "gate_seen": [...], "use": [...], "admits": [...], "born": [...],
//!                 "last_seen": [...], "recent": [...], "ever": [...], "uid": [...],
//!                 "next_uid": 8, "segments": 3 },
//!  "n_experts": 8, "d_model": 32, "d_ff": 16, "paged": true,
//!  "core_tensors": [...], "router_tensors": [...],
//!  "experts": [ {"id": 0, "file": "e00000.npz", "bytes": 14458, "params": 1536, "moments": true, "gate": 1.01} ],
//!  "total_bytes": 377160,
//!  "read_chars": 72, "read_nats": 100.5, "plasticity": {...}, "context_now": 23
//! }
//! ```
//!
//! # Unknown keys survive
//!
//! A manifest written by an older or newer Python (or by hand) may carry keys this code has never
//! heard of, at the top level, inside `telemetry`, and inside each `experts` record. They are kept
//! in an `extra` map and written back unchanged, so a Rust read-modify-write never silently drops
//! them. (Key *order* is not preserved: maps are sorted by key. Order is meaningless in JSON.)
//!
//! Python's `json.dump` also writes the non-standard tokens `NaN`, `Infinity` and `-Infinity` for
//! non-finite floats, and Python's loader cannot take the `null` a strict writer would use there (it
//! feeds the per-expert arrays to `torch.tensor`). So they are accepted on reading and written back
//! the way Python writes them, inside the per-expert arrays and an expert's gate. Where a number is
//! optional (`val`, `read_nats`) a non-finite value reads as "none" and is not written.
//!
//! # `cfg` and the app's `ModelConfig`
//!
//! `cfg` is the Python `RecurConfig` as a dictionary with **snake_case** keys (`d_model`, `n_head`,
//! `pool_experts`, `pool_d_ff`, `pool_top_k`, `halt_prior`, ...) plus a few fields the reference adds
//! (`pool_resident`, `pool_max`, `pool_ever`). It stays a free-form JSON map in [`Manifest::cfg`].
//!
//! The app's [`ModelConfig`] (in `minagi-types`) has exactly the 21 shape fields of `RecurConfig`
//! that matter, but serialises with **camelCase** keys. The two relate like this:
//!
//! * `cfg` is the single source of truth for the model's shape. [`ParsedCfg::from_cfg`] turns it into
//!   a `ModelConfig`; fields a very old manifest lacks are filled from `RecurConfig`'s defaults and
//!   listed in [`ParsedCfg::defaulted`].
//! * `ParsedCfg::from_cfg` also accepts a `cfg` that already uses our camelCase keys.
//! * The Python-only fields (`n_layer`, `rope_theta`, `tie_embeddings`, `use_pool`,
//!   `pool_capacity_factor`, `pool_aux`, `pool_resident`, `pool_ever`, and anything unknown) are
//!   carried in [`ParsedCfg::python_only`] and written back verbatim, which is what lets Python load
//!   what we write.
//! * Our engine always *writes* the snake_case form, so there is never a second copy of the model
//!   shape to disagree with the first.
//!
//! # The `minagi_rs` namespace
//!
//! Everything this engine records that the Python reference has no place for goes under one
//! top-level key, `"minagi_rs"` ([`MinagiRs`]): a format number, which program wrote the file, where
//! it came from (`native` or `python-import`), the preset it was created from, notes. The Python
//! code never reads that key. A directory without it was written by Python
//! ([`Manifest::written_by_python`]).

use super::{Result, StoreError, fsio};
use minagi_types::config::{ModelConfig, Preset};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use std::borrow::Cow;
use std::path::Path;

/// File name of the manifest inside a checkpoint directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Top-level key under which this engine keeps its own additions.
pub const RS_KEY: &str = "minagi_rs";
/// Version of the `minagi_rs` block.
pub const RS_FORMAT: u32 = 1;
/// [`MinagiRs::origin`] of a checkpoint this engine wrote itself.
pub const ORIGIN_NATIVE: &str = "native";
/// [`MinagiRs::origin`] of a checkpoint converted from a Python-written directory.
pub const ORIGIN_PYTHON_IMPORT: &str = "python-import";

/// A free-form JSON object (sorted by key).
pub type JsonMap = Map<String, Value>;

// ---------------------------------------------------------------------------------------------
// lenient number handling for fields Python writes loosely
// ---------------------------------------------------------------------------------------------

/// Stand-in strings for non-finite floats (see [`strip_nonstandard_numbers`] and `ser_f64`).
const NAN_MARK: &str = "\u{1}minagi:nan";
const INF_MARK: &str = "\u{1}minagi:inf";
const NEG_INF_MARK: &str = "\u{1}minagi:-inf";

fn f64_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        // the stand-ins `strip_nonstandard_numbers` puts where Python wrote NaN / Infinity
        Value::String(s) => match s.as_str() {
            NAN_MARK => Some(f64::NAN),
            INF_MARK => Some(f64::INFINITY),
            NEG_INF_MARK => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

fn de_f64<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<f64, D::Error> {
    Ok(Option::<Value>::deserialize(d)?.as_ref().and_then(f64_of).unwrap_or(f64::NAN))
}

fn de_opt_f64<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<f64>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?.as_ref().and_then(f64_of).filter(|x| x.is_finite()))
}

fn de_opt_u64<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<u64>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?.as_ref().and_then(|v| {
        v.as_u64().or_else(|| {
            v.as_f64().filter(|x| x.is_finite() && *x >= 0.0 && *x < u64::MAX as f64).map(|x| x.round() as u64)
        })
    }))
}

fn de_opt_i64<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<i64>, D::Error> {
    Ok(Option::<Value>::deserialize(d)?.as_ref().and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_f64().filter(|x| x.is_finite() && x.abs() < i64::MAX as f64).map(|x| x.round() as i64))
    }))
}

fn de_f64_vec<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<f64>, D::Error> {
    let v = Option::<Vec<Option<Value>>>::deserialize(d)?.unwrap_or_default();
    Ok(v.iter().map(|x| x.as_ref().and_then(f64_of).unwrap_or(f64::NAN)).collect())
}

fn one() -> f64 {
    1.0
}

// Python's `json.dump` writes `NaN` / `Infinity` / `-Infinity` for non-finite floats, and its loader
// (`torch.tensor(v)` over the per-expert arrays) cannot take the `null` a strict writer would use. So
// non-finite floats are written as Python writes them: serialised as marker strings first, which
// `Manifest::to_json` then replaces with the bare tokens.

fn ser_f64<S: serde::Serializer>(x: &f64, s: S) -> std::result::Result<S::Ok, S::Error> {
    if x.is_nan() {
        s.serialize_str(NAN_MARK)
    } else if *x == f64::INFINITY {
        s.serialize_str(INF_MARK)
    } else if *x == f64::NEG_INFINITY {
        s.serialize_str(NEG_INF_MARK)
    } else {
        s.serialize_f64(*x)
    }
}

fn ser_f64_vec<S: serde::Serializer>(v: &[f64], s: S) -> std::result::Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for x in v {
        if x.is_finite() {
            seq.serialize_element(x)?;
        } else if x.is_nan() {
            seq.serialize_element(NAN_MARK)?;
        } else if *x > 0.0 {
            seq.serialize_element(INF_MARK)?;
        } else {
            seq.serialize_element(NEG_INF_MARK)?;
        }
    }
    seq.end()
}

// ---------------------------------------------------------------------------------------------
// telemetry
// ---------------------------------------------------------------------------------------------

/// Per-expert history (`PagedPool.telemetry()`): "none of it is reconstructable afterwards".
///
/// All vectors are indexed by **position** in the pool (the order of the router rows), not by uid;
/// [`uid`](Telemetry::uid) maps position to uid. Missing keys read as empty vectors, like Python's
/// `load_telemetry`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Telemetry {
    #[serde(default, deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub gate: Vec<f64>,
    #[serde(default, deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub gate_seen: Vec<f64>,
    /// How often each expert was routed to (the JSON key is `use`).
    #[serde(default, rename = "use", deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub usage: Vec<f64>,
    /// How many times each expert was brought onto the accelerator.
    #[serde(default, deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub admits: Vec<f64>,
    /// The training step at which each expert was born.
    #[serde(default, deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub born: Vec<f64>,
    /// The prune clock: the text count at which each expert was last admitted.
    #[serde(default, deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub last_seen: Vec<f64>,
    /// Each expert's share of recent training forwards (the exploration bonus reads it).
    #[serde(default, deserialize_with = "de_f64_vec", serialize_with = "ser_f64_vec")]
    pub recent: Vec<f64>,
    /// Whether each expert has ever been on the card.
    #[serde(default)]
    pub ever: Vec<bool>,
    /// Position -> uid. An expert's file is `e<uid>.npz`; pruning never renumbers uids.
    #[serde(default)]
    pub uid: Vec<u64>,
    #[serde(default)]
    pub next_uid: Option<u64>,
    /// Texts begun so far (the unit `last_seen` is counted in).
    #[serde(default)]
    pub segments: Option<u64>,
    /// Anything else a different version put here.
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl Telemetry {
    /// A telemetry block for a pool of `uids.len()` freshly created experts.
    pub fn fresh(uids: &[u64]) -> Self {
        let n = uids.len();
        Self {
            gate: vec![1.0; n],
            gate_seen: vec![0.0; n],
            usage: vec![0.0; n],
            admits: vec![0.0; n],
            born: vec![0.0; n],
            last_seen: vec![0.0; n],
            recent: vec![0.0; n],
            ever: vec![false; n],
            uid: uids.to_vec(),
            next_uid: Some(uids.iter().max().map_or(0, |m| m + 1)),
            segments: Some(0),
            extra: JsonMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the expert list
// ---------------------------------------------------------------------------------------------

/// One entry of the manifest's `experts` list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRecord {
    /// The expert's uid.
    pub id: u64,
    /// File name inside `experts/`, `e<uid>.npz` (an old layout used `.npy`).
    pub file: String,
    #[serde(default)]
    pub bytes: u64,
    /// Number of weights (3 x d_ff x d_model).
    #[serde(default)]
    pub params: u64,
    /// Whether the file holds Adam moments. (Python's paged save writes `true` for every expert,
    /// whether or not its file has them; ask [`super::tiers::describe_expert_file`] for the truth.)
    #[serde(default)]
    pub moments: bool,
    /// The expert's gate when the manifest was written.
    #[serde(default = "one", deserialize_with = "de_f64", serialize_with = "ser_f64")]
    pub gate: f64,
    #[serde(flatten)]
    pub extra: JsonMap,
}

// ---------------------------------------------------------------------------------------------
// this engine's additions
// ---------------------------------------------------------------------------------------------

/// What this engine records under the `minagi_rs` key (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MinagiRs {
    /// Version of this block (currently [`RS_FORMAT`]).
    #[serde(default)]
    pub format: u32,
    /// The program that wrote the checkpoint, for example `minagi-core 0.1.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writer: Option<String>,
    /// [`ORIGIN_NATIVE`] or [`ORIGIN_PYTHON_IMPORT`] (other values are kept as they are).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// The preset the model was created from (`tiny`, `small`, `full`, `custom`, `imported`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    /// For an import: the Python directory it was converted from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from: Option<String>,
    /// Plain-English remarks about the conversion (defaulted fields, layout changes, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl MinagiRs {
    /// The block for a checkpoint this engine writes.
    pub fn native(preset: Option<Preset>) -> Self {
        Self {
            format: RS_FORMAT,
            writer: Some(format!("minagi-core {}", env!("CARGO_PKG_VERSION"))),
            origin: Some(ORIGIN_NATIVE.into()),
            preset: preset.map(|p| p.as_str().to_string()),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the manifest
// ---------------------------------------------------------------------------------------------

/// `manifest.json`. See the module docs.
///
/// When *writing* a checkpoint through [`super::checkpoint::Checkpoint::write`] the caller fills in
/// `step`, `val`, `cfg`, `telemetry`, `read_chars`, `read_nats`, `plasticity`, `context_now` and
/// `minagi_rs`; the derived fields (`n_experts`, `d_model`, `d_ff`, `paged`, `core_tensors`,
/// `router_tensors`, `experts`, `total_bytes`) are computed from what is actually written.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Training step of the state on disk; Python writes `-1` for a model created but not trained.
    #[serde(default, deserialize_with = "de_opt_i64")]
    pub step: Option<i64>,
    /// Validation loss (nats per character) of this state, if one was measured.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub val: Option<f64>,
    /// The `RecurConfig` as a free-form map (see the module docs).
    #[serde(default)]
    pub cfg: Option<JsonMap>,
    /// Per-expert history. `None` for a directory written before paging existed.
    #[serde(default)]
    pub telemetry: Option<Telemetry>,
    #[serde(default)]
    pub n_experts: usize,
    #[serde(default)]
    pub d_model: Option<usize>,
    /// Hidden width of one expert.
    #[serde(default)]
    pub d_ff: Option<usize>,
    /// Written by a paged pool's save. A pre-paging directory has no such key (read as `false`).
    #[serde(default)]
    pub paged: bool,
    #[serde(default)]
    pub core_tensors: Vec<String>,
    #[serde(default)]
    pub router_tensors: Vec<String>,
    #[serde(default)]
    pub experts: Vec<ExpertRecord>,
    #[serde(default)]
    pub total_bytes: u64,
    /// Only the non-paged save writes this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removed_expert_files: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "de_opt_u64")]
    pub read_chars: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "de_opt_f64")]
    pub read_nats: Option<f64>,
    /// The plasticity controller's state. Another module owns its meaning; it is kept as JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plasticity: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "de_opt_u64")]
    pub context_now: Option<u64>,
    /// This engine's additions; absent in a directory written by Python.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minagi_rs: Option<MinagiRs>,
    /// Every other key, untouched.
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Replace the non-standard `NaN`, `Infinity` and `-Infinity` tokens (outside of strings) with the
/// stand-in strings the typed float fields understand, so Python's `json.dump` output parses.
fn strip_nonstandard_numbers(text: &str) -> Cow<'_, str> {
    if !(text.contains("NaN") || text.contains("Infinity")) {
        return Cow::Borrowed(text);
    }
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut i, mut in_str, mut last) = (0, false, 0);
    while i < b.len() {
        let c = b[i];
        if in_str {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_str = true;
            i += 1;
            continue;
        }
        let rest = &text[i..];
        let tok =
            [("-Infinity", "-inf"), ("Infinity", "inf"), ("NaN", "nan")].into_iter().find(|(t, _)| rest.starts_with(t));
        if let Some((t, mark)) = tok {
            out.push_str(&text[last..i]);
            out.push_str("\"\\u0001minagi:");
            out.push_str(mark);
            out.push('"');
            i += t.len();
            last = i;
            continue;
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    Cow::Owned(out)
}

impl Manifest {
    /// A manifest for `cfg`, ready for [`super::checkpoint::Checkpoint::write`] to complete.
    pub fn new(step: i64, val: Option<f64>, cfg: &ParsedCfg) -> Self {
        Self { step: Some(step), val, cfg: Some(cfg.to_cfg()), paged: true, ..Self::default() }
    }

    /// Parse the text of a `manifest.json`.
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut m: Manifest = match serde_json::from_str(text) {
            Ok(m) => m,
            Err(first) => {
                let fixed = strip_nonstandard_numbers(text);
                if matches!(fixed, Cow::Borrowed(_)) {
                    return Err(first.to_string());
                }
                serde_json::from_str(&fixed).map_err(|e| e.to_string())?
            }
        };
        if m.n_experts == 0 {
            m.n_experts = m.experts.len();
        }
        m.scrub_free_form();
        Ok(m)
    }

    /// Inside free-form JSON (`cfg`, `plasticity`, the `extra` maps) a non-finite number cannot be a
    /// `serde_json::Value`, so it reads as `null`; the typed float fields keep the exact value.
    fn scrub_free_form(&mut self) {
        fn scrub(v: &mut Value) {
            match v {
                Value::String(s) if matches!(s.as_str(), NAN_MARK | INF_MARK | NEG_INF_MARK) => *v = Value::Null,
                Value::Array(a) => a.iter_mut().for_each(scrub),
                Value::Object(o) => o.values_mut().for_each(scrub),
                _ => {}
            }
        }
        let scrub_map = |m: &mut JsonMap| m.values_mut().for_each(scrub);
        if let Some(c) = &mut self.cfg {
            scrub_map(c);
        }
        if let Some(p) = &mut self.plasticity {
            scrub(p);
        }
        scrub_map(&mut self.extra);
        if let Some(t) = &mut self.telemetry {
            scrub_map(&mut t.extra);
        }
        for e in &mut self.experts {
            scrub_map(&mut e.extra);
        }
        if let Some(r) = &mut self.minagi_rs {
            scrub_map(&mut r.extra);
        }
    }

    /// Read and parse `<dir>/manifest.json`'s file at `path`.
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| StoreError::io_at(path, e))?;
        Self::parse(&text).map_err(|message| StoreError::Manifest { path: path.to_path_buf(), message })
    }

    /// Serialise like Python's `json.dump(..., indent=1)`.
    pub fn to_json(&self) -> std::result::Result<String, String> {
        let mut buf = Vec::new();
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
        self.serialize(&mut ser).map_err(|e| e.to_string())?;
        let text = String::from_utf8(buf).map_err(|e| e.to_string())?;
        // the marker strings become the bare tokens Python writes (see ser_f64)
        Ok(text
            .replace("\"\\u0001minagi:nan\"", "NaN")
            .replace("\"\\u0001minagi:inf\"", "Infinity")
            .replace("\"\\u0001minagi:-inf\"", "-Infinity"))
    }

    /// Write `path` atomically (temporary file, then rename).
    pub fn write(&self, path: &Path, durable: bool) -> Result<()> {
        let text = self.to_json().map_err(|message| StoreError::Manifest { path: path.to_path_buf(), message })?;
        fsio::write_atomic(path, durable, |w| {
            use std::io::Write;
            w.write_all(text.as_bytes())?;
            Ok(())
        })?;
        Ok(())
    }

    /// Whether this manifest was written by the Python reference (it has no `minagi_rs` block).
    pub fn written_by_python(&self) -> bool {
        self.minagi_rs.is_none()
    }

    /// Position -> uid of every expert, `n_experts` long.
    ///
    /// Taken from `telemetry.uid`; positions it does not cover keep their position as uid, which is
    /// what Python does ("a directory written before ids existed has files named by position").
    pub fn uids(&self) -> Vec<u64> {
        let n = self.n_experts.max(self.experts.len());
        let mut uids: Vec<u64> = (0..n as u64).collect();
        if let Some(t) = &self.telemetry {
            for (slot, &u) in uids.iter_mut().zip(&t.uid) {
                *slot = u;
            }
        }
        uids
    }

    /// The uids named in the `experts` list, in list order.
    pub fn listed_uids(&self) -> Vec<u64> {
        self.experts.iter().map(|e| e.id).collect()
    }

    /// `cfg` as the app's [`ModelConfig`], or `None` if the manifest has no `cfg` (Python writes
    /// `null` when saved without one).
    pub fn parsed_cfg(&self) -> std::result::Result<Option<ParsedCfg>, String> {
        self.cfg.as_ref().map(ParsedCfg::from_cfg).transpose()
    }

    /// The `pool_resident` recorded in `cfg`, if any.
    pub fn resident(&self) -> Option<u32> {
        let v = self.cfg.as_ref()?.get("pool_resident")?;
        v.as_u64().and_then(|n| u32::try_from(n).ok())
    }
}

// ---------------------------------------------------------------------------------------------
// cfg <-> ModelConfig
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Count,
    Real,
    Flag,
}

/// `(python name, ModelConfig's camelCase name, kind)`, in `RecurConfig` declaration order.
const FIELDS: [(&str, &str, Kind); 21] = [
    ("vocab_size", "vocabSize", Kind::Count),
    ("n_head", "nHead", Kind::Count),
    ("d_model", "dModel", Kind::Count),
    ("block", "block", Kind::Count),
    ("d_ff", "dFf", Kind::Count),
    ("pool_experts", "poolExperts", Kind::Count),
    ("pool_d_ff", "poolDFf", Kind::Count),
    ("pool_depth", "poolDepth", Kind::Count),
    ("pool_top_k", "poolTopK", Kind::Count),
    ("pool_max", "poolMax", Kind::Count),
    ("n_prelude", "nPrelude", Kind::Count),
    ("n_recur", "nRecur", Kind::Count),
    ("n_coda", "nCoda", Kind::Count),
    ("max_steps", "maxSteps", Kind::Count),
    ("min_steps", "minSteps", Kind::Count),
    ("train_steps_mean", "trainStepsMean", Kind::Real),
    ("bptt_window", "bpttWindow", Kind::Count),
    ("ponder_beta", "ponderBeta", Kind::Real),
    ("halt_prior", "haltPrior", Kind::Real),
    ("halt_thresh", "haltThresh", Kind::Real),
    ("halt_freeze", "haltFreeze", Kind::Flag),
];

/// The default of a `RecurConfig` field (`minagi/recur.py`, `minagi/model.py`), used for a field a
/// very old manifest does not record.
fn python_default(name: &str) -> Value {
    match name {
        "vocab_size" => 8192.into(),
        "n_head" => 8.into(),
        "d_model" => 512.into(),
        "block" => 512.into(),
        "d_ff" => 1408.into(),
        "pool_experts" => 64.into(),
        "pool_d_ff" => 192.into(),
        "pool_depth" => 1.into(),
        "pool_top_k" => 4.into(),
        "pool_max" => 1024.into(),
        "n_prelude" => 1.into(),
        "n_recur" => 2.into(),
        "n_coda" => 1.into(),
        "max_steps" => 4.into(),
        "min_steps" => 1.into(),
        "train_steps_mean" => 0.0.into(),
        "bptt_window" => 4.into(),
        "ponder_beta" => 0.01.into(),
        "halt_prior" => 0.4.into(),
        "halt_thresh" => 0.9.into(),
        "halt_freeze" => false.into(),
        _ => Value::Null,
    }
}

/// `RecurConfig` fields that are not part of [`ModelConfig`] and their defaults, written for
/// Python's benefit when we create a manifest.
fn python_only_defaults() -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("n_layer".into(), 8.into());
    m.insert("rope_theta".into(), 10000.0.into());
    m.insert("tie_embeddings".into(), true.into());
    m.insert("use_pool".into(), true.into());
    m.insert("pool_capacity_factor".into(), 1.5.into());
    m.insert("pool_aux".into(), 0.01.into());
    m
}

fn coerce(name: &str, kind: Kind, v: &Value) -> std::result::Result<Value, String> {
    let bad = || {
        format!(
            "cfg.{name} should be {}, found {v}",
            match kind {
                Kind::Count => "a whole number",
                Kind::Real => "a number",
                Kind::Flag => "true or false",
            }
        )
    };
    match kind {
        Kind::Count => {
            if let Some(n) = v.as_u64().filter(|&n| n <= u32::MAX as u64) {
                return Ok(n.into());
            }
            match v.as_f64() {
                Some(x) if x.is_finite() && x.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&x) => {
                    Ok((x as u64).into())
                }
                _ => Err(bad()),
            }
        }
        Kind::Real => v.as_f64().filter(|x| x.is_finite()).map(Value::from).ok_or_else(bad),
        Kind::Flag => match v {
            Value::Bool(_) => Ok(v.clone()),
            Value::Number(n) if n.as_u64().is_some_and(|n| n <= 1) => Ok(Value::Bool(n.as_u64() == Some(1))),
            _ => Err(bad()),
        },
    }
}

/// A manifest's `cfg` read as the app's [`ModelConfig`], plus everything in it that `ModelConfig`
/// has no place for.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCfg {
    pub model: ModelConfig,
    /// Python-only and unknown `cfg` keys (`n_layer`, `rope_theta`, `tie_embeddings`, `use_pool`,
    /// `pool_capacity_factor`, `pool_aux`, `pool_resident`, `pool_ever`, ...), written back as they
    /// were.
    pub python_only: JsonMap,
    /// Fields the manifest did not record and that were filled from `RecurConfig`'s defaults.
    pub defaulted: Vec<String>,
}

impl ParsedCfg {
    /// Read a `cfg` map (snake_case Python keys, or camelCase `ModelConfig` keys).
    pub fn from_cfg(cfg: &JsonMap) -> std::result::Result<Self, String> {
        let camel = cfg.contains_key("dModel") && !cfg.contains_key("d_model");
        let mut model_obj = JsonMap::new();
        let mut defaulted = Vec::new();
        for (snake, camel_name, kind) in FIELDS {
            let key = if camel { camel_name } else { snake };
            let value = match cfg.get(key) {
                Some(v) if !v.is_null() => coerce(key, kind, v)?,
                _ => {
                    defaulted.push(snake.to_string());
                    python_default(snake)
                }
            };
            model_obj.insert(camel_name.to_string(), value);
        }
        let model: ModelConfig = serde_json::from_value(Value::Object(model_obj)).map_err(|e| e.to_string())?;
        let known = |k: &str| FIELDS.iter().any(|(s, c, _)| *s == k || *c == k);
        let python_only: JsonMap = cfg.iter().filter(|(k, _)| !known(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
        Ok(Self { model, python_only, defaulted })
    }

    /// Wrap a [`ModelConfig`] for writing, with the Python-only fields at their defaults and
    /// `pool_resident` set.
    pub fn from_model(model: ModelConfig, resident: u32) -> Self {
        let mut python_only = python_only_defaults();
        python_only.insert("pool_resident".into(), resident.into());
        Self { model, python_only, defaulted: Vec::new() }
    }

    /// The `cfg` map to put in a manifest: snake_case keys Python can load, with the Python-only
    /// fields alongside. The expert ceiling `pool_max` is never below what the pool holds
    /// (callers raise it via [`ParsedCfg::model`] before writing).
    pub fn to_cfg(&self) -> JsonMap {
        let mut out = JsonMap::new();
        let as_json = serde_json::to_value(&self.model).unwrap_or(Value::Null);
        let obj = as_json.as_object();
        // Python's dataclass order for the fields it has, then the extras; maps sort by key anyway,
        // but keep the intent visible.
        for (snake, camel, _) in FIELDS {
            if let Some(v) = obj.and_then(|o| o.get(camel)) {
                out.insert(snake.to_string(), v.clone());
            }
        }
        for (k, v) in &self.python_only {
            out.insert(k.clone(), v.clone());
        }
        out
    }

    /// `pool_resident` (experts on the card) if recorded.
    pub fn resident(&self) -> Option<u32> {
        self.python_only.get("pool_resident")?.as_u64().and_then(|n| u32::try_from(n).ok())
    }

    /// Python-only fields the engine does not support at a non-default value, as plain-English
    /// problems (an empty list means all is as the engine expects).
    pub fn python_only_problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        let get = |k: &str| self.python_only.get(k);
        if get("use_pool").and_then(Value::as_bool) == Some(false) {
            out.push(
                "This model has no expert pool (use_pool is false); this engine only runs the pooled architecture."
                    .to_string(),
            );
        }
        if get("tie_embeddings").and_then(Value::as_bool) == Some(false) {
            out.push(
                "This model has separate input and output embeddings (tie_embeddings is false); this engine ties them."
                    .to_string(),
            );
        }
        if let Some(theta) = get("rope_theta").and_then(Value::as_f64)
            && (theta - 10000.0).abs() > 1e-6
        {
            out.push(format!(
                "This model uses rotary base {theta} but this engine uses 10000; the positions would not line up."
            ));
        }
        out
    }
}

/// The preset whose shape matches `m` (width, heads, depth, expert size and count of experts per
/// character, vocabulary), else [`Preset::Imported`].
pub fn suggest_preset(m: &ModelConfig) -> Preset {
    let shape = |c: &ModelConfig| {
        (
            c.d_model,
            c.n_head,
            c.d_ff,
            c.n_prelude,
            c.n_recur,
            c.n_coda,
            c.max_steps,
            c.pool_d_ff,
            c.pool_depth,
            c.pool_top_k,
            c.vocab_size,
        )
    };
    [Preset::Tiny, Preset::Small, Preset::Full]
        .into_iter()
        .find(|p| shape(&p.model()) == shape(m))
        .unwrap_or(Preset::Imported)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manifest exactly as the Python reference writes it for a tiny paged model (trimmed).
    const PY_MANIFEST: &str = r#"{
 "step": 2,
 "val": 1.5,
 "cfg": {
  "vocab_size": 265, "n_layer": 8, "n_head": 2, "d_model": 32, "block": 64, "d_ff": 64,
  "rope_theta": 10000.0, "tie_embeddings": true, "use_pool": true,
  "pool_experts": 8, "pool_d_ff": 16, "pool_depth": 1, "pool_top_k": 2,
  "pool_capacity_factor": 1.5, "pool_max": 8, "pool_aux": 0.01,
  "n_prelude": 1, "n_recur": 1, "n_coda": 0, "max_steps": 4, "min_steps": 1,
  "train_steps_mean": 0.0, "bptt_window": 4, "ponder_beta": 0.01, "halt_prior": 0.4,
  "halt_thresh": 0.9, "halt_freeze": true, "pool_resident": 4,
  "pool_ever": [true, true, false]
 },
 "telemetry": {
  "gate": [1.01, 0.99, 0.5], "gate_seen": [0.0, 0.0, 0.0], "use": [63.0, 39.0, 0.0],
  "admits": [1.0, 1.0, 0.0], "born": [0.0, 0.0, 5.0], "last_seen": [2.0, 3.0, 0.0],
  "recent": [0.0, 0.0, 0.0], "ever": [true, true, false], "uid": [0, 2, 7],
  "next_uid": 8, "segments": 3
 },
 "n_experts": 3, "d_model": 32, "d_ff": 16, "paged": true,
 "core_tensors": ["adapter.weight", "tok_emb.weight"],
 "router_tensors": ["pool.gate", "recur.0.mlp.depth_emb", "recur.0.mlp.router.weight"],
 "experts": [
  {"id": 0, "file": "e00000.npz", "bytes": 14458, "params": 1536, "moments": true, "gate": 1.0100976},
  {"id": 2, "file": "e00002.npz", "bytes": 14458, "params": 1536, "moments": true, "gate": 0.99},
  {"id": 7, "file": "e00007.npz", "bytes": 6144, "params": 1536, "moments": true, "gate": 0.5}
 ],
 "total_bytes": 377160,
 "read_chars": 72, "read_nats": 100.5, "plasticity": {"lr_mult": 1.0}, "context_now": 23
}"#;

    #[test]
    fn reads_a_python_manifest() {
        let m = Manifest::parse(PY_MANIFEST).unwrap();
        assert_eq!((m.step, m.val), (Some(2), Some(1.5)));
        assert_eq!((m.n_experts, m.d_model, m.d_ff, m.paged), (3, Some(32), Some(16), true));
        assert_eq!(m.uids(), [0, 2, 7]);
        assert_eq!(m.listed_uids(), [0, 2, 7]);
        assert_eq!(m.experts[2].file, "e00007.npz");
        assert_eq!((m.read_chars, m.read_nats, m.context_now), (Some(72), Some(100.5), Some(23)));
        assert_eq!(m.plasticity, Some(serde_json::json!({"lr_mult": 1.0})));
        assert!(m.written_by_python());
        assert_eq!(m.resident(), Some(4));
        let t = m.telemetry.as_ref().unwrap();
        assert_eq!((t.next_uid, t.segments), (Some(8), Some(3)));
        assert_eq!(t.usage, [63.0, 39.0, 0.0]);
        assert_eq!(t.ever, [true, true, false]);
        assert_eq!(t.born, [0.0, 0.0, 5.0]);
        let p = m.parsed_cfg().unwrap().unwrap();
        assert_eq!((p.model.d_model, p.model.pool_d_ff, p.model.n_head, p.model.pool_top_k), (32, 16, 2, 2));
        assert!(p.model.halt_freeze && p.defaulted.is_empty());
        assert_eq!(p.resident(), Some(4));
        assert!(p.python_only.contains_key("pool_ever") && p.python_only.contains_key("rope_theta"));
        assert!(p.python_only_problems().is_empty());
    }

    #[test]
    fn unknown_keys_survive_a_read_modify_write_everywhere() {
        let text = PY_MANIFEST
            .replacen("{\n \"step\": 2,", "{\n \"future_top\": {\"a\": [1, 2, {\"b\": null}]},\n \"step\": 2,", 1)
            .replacen("\"segments\": 3", "\"segments\": 3, \"future_tel\": [0.5]", 1)
            .replacen("\"id\": 2,", "\"id\": 2, \"lineage\": [0, 7],", 1);
        let mut m = Manifest::parse(&text).unwrap();
        assert!(m.extra.contains_key("future_top"));
        assert!(m.telemetry.as_ref().unwrap().extra.contains_key("future_tel"));
        assert!(m.experts[1].extra.contains_key("lineage"));
        // modify something and write back
        m.step = Some(99);
        m.minagi_rs = Some(MinagiRs::native(Some(Preset::Tiny)));
        let out = m.to_json().unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["future_top"], serde_json::json!({"a": [1, 2, {"b": null}]}));
        assert_eq!(v["telemetry"]["future_tel"], serde_json::json!([0.5]));
        assert_eq!(v["experts"][1]["lineage"], serde_json::json!([0, 7]));
        assert_eq!(v["step"], 99);
        assert_eq!(v["minagi_rs"]["origin"], ORIGIN_NATIVE);
        assert_eq!(v["minagi_rs"]["preset"], "tiny");
        // and a second read gives the same structure back
        let again = Manifest::parse(&out).unwrap();
        assert_eq!(again, m);
        assert!(!again.written_by_python());
    }

    #[test]
    fn python_nan_and_infinity_tokens_are_accepted() {
        let text = r#"{"step": 3, "val": NaN, "cfg": null, "n_experts": 2,
            "telemetry": {"gate": [NaN, 1.0], "use": [Infinity, -Infinity], "uid": [4, 5], "note": "NaN stays in strings"},
            "experts": [{"id": 4, "file": "e00004.npz", "gate": NaN}], "read_nats": Infinity,
            "plasticity": {"snr": NaN, "list": [1.0, Infinity]}}"#;
        let m = Manifest::parse(text).unwrap();
        assert_eq!(m.val, None);
        assert_eq!(m.cfg, None);
        let t = m.telemetry.as_ref().unwrap();
        assert!(t.gate[0].is_nan() && t.gate[1] == 1.0);
        assert_eq!((t.usage[0], t.usage[1]), (f64::INFINITY, f64::NEG_INFINITY));
        assert_eq!(t.extra["note"], "NaN stays in strings");
        // free-form JSON cannot hold a NaN: it reads as null
        assert_eq!(m.plasticity, Some(serde_json::json!({"snr": null, "list": [1.0, null]})));
        assert!(m.experts[0].gate.is_nan());
        assert_eq!(m.read_nats, None);
        // written back the way Python writes them (bare tokens), and read again as the same values
        let out = m.to_json().unwrap();
        assert!(out.contains("[NaN, 1.0]") || out.contains("NaN"), "{out}");
        assert!(out.contains("Infinity") && out.contains("-Infinity"), "{out}");
        assert!(!out.contains("minagi:"), "markers must not leak: {out}");
        let again = Manifest::parse(&out).unwrap();
        let t2 = again.telemetry.as_ref().unwrap();
        assert!(t2.gate[0].is_nan() && t2.gate[1] == 1.0);
        assert_eq!((t2.usage[0], t2.usage[1]), (f64::INFINITY, f64::NEG_INFINITY));
        assert!(again.experts[0].gate.is_nan());
        assert_eq!(t2.extra["note"], "NaN stays in strings");
        // a genuinely broken file is still an error with a position
        let err = Manifest::parse("{\"step\": ").unwrap_err();
        assert!(err.contains("line 1"), "{err}");
        assert!(Manifest::parse("[1,2,3]").is_err());
    }

    #[test]
    fn missing_optional_keys_and_older_layouts_are_tolerated() {
        // a pre-paging manifest: no paged, telemetry, read_*; experts without moments/params
        let text = r#"{"step": -1, "val": null, "cfg": {"d_model": 32}, "d_model": 32, "d_ff": 16,
            "experts": [{"id": 0, "file": "e00000.npy"}, {"id": 1, "file": "e00001.npy"}], "total_bytes": 10}"#;
        let m = Manifest::parse(text).unwrap();
        assert_eq!(m.step, Some(-1));
        assert!(!m.paged && m.telemetry.is_none());
        assert_eq!(m.n_experts, 2, "n_experts falls back to the list length");
        assert_eq!(m.uids(), [0, 1], "uid = position when there is no telemetry");
        assert_eq!(m.experts[0].gate, 1.0);
        assert!(!m.experts[0].moments);
        let p = m.parsed_cfg().unwrap().unwrap();
        assert_eq!(p.model.d_model, 32);
        assert!(p.defaulted.contains(&"n_head".to_string()) && !p.defaulted.contains(&"d_model".to_string()));
        assert_eq!(p.model.n_head, 8, "an unrecorded field takes RecurConfig's default");
        // telemetry uid shorter than the pool: the rest keep their position
        let m2 = Manifest::parse(r#"{"n_experts": 4, "telemetry": {"uid": [10, 11]}}"#).unwrap();
        assert_eq!(m2.uids(), [10, 11, 2, 3]);
        // an empty object is a (useless but valid) manifest
        let m3 = Manifest::parse("{}").unwrap();
        assert_eq!((m3.n_experts, m3.uids().len()), (0, 0));
    }

    #[test]
    fn loosely_typed_numbers_are_tolerated() {
        let m = Manifest::parse(r#"{"step": 7.0, "read_chars": 1.5e9, "context_now": 23.0, "read_nats": 3}"#).unwrap();
        assert_eq!(
            (m.step, m.read_chars, m.context_now, m.read_nats),
            (Some(7), Some(1_500_000_000), Some(23), Some(3.0))
        );
        let cfg: JsonMap = serde_json::from_str(r#"{"d_model": 64.0, "halt_freeze": 1, "n_head": 4}"#).unwrap();
        let p = ParsedCfg::from_cfg(&cfg).unwrap();
        assert_eq!((p.model.d_model, p.model.n_head, p.model.halt_freeze), (64, 4, true));
        let bad: JsonMap = serde_json::from_str(r#"{"d_model": "wide"}"#).unwrap();
        let err = ParsedCfg::from_cfg(&bad).unwrap_err();
        assert!(err.contains("cfg.d_model") && err.contains("whole number"), "{err}");
        let frac: JsonMap = serde_json::from_str(r#"{"d_model": 64.5}"#).unwrap();
        assert!(ParsedCfg::from_cfg(&frac).is_err());
    }

    #[test]
    fn field_table_matches_model_config_serialisation() {
        let v = serde_json::to_value(Preset::Full.model()).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut table: Vec<&str> = FIELDS.iter().map(|f| f.1).collect();
        table.sort_unstable();
        assert_eq!(keys, table, "every ModelConfig field must be in FIELDS (and vice versa)");
        for (snake, _, kind) in FIELDS {
            let d = python_default(snake);
            assert!(!d.is_null(), "{snake}");
            assert!(coerce(snake, kind, &d).is_ok(), "{snake}");
        }
    }

    #[test]
    fn model_config_round_trips_through_snake_case_and_camel_case_cfg() {
        for preset in [Preset::Tiny, Preset::Small, Preset::Full] {
            let model = preset.model();
            let p = ParsedCfg::from_model(model.clone(), 12);
            let cfg = p.to_cfg();
            for k in [
                "d_model",
                "n_head",
                "pool_d_ff",
                "pool_top_k",
                "halt_freeze",
                "pool_resident",
                "rope_theta",
                "tie_embeddings",
            ] {
                assert!(cfg.contains_key(k), "{k}");
            }
            assert!(!cfg.contains_key("dModel"));
            let back = ParsedCfg::from_cfg(&cfg).unwrap();
            assert_eq!(back.model, model, "{preset:?} via snake_case");
            assert!(back.defaulted.is_empty());
            assert_eq!(back.resident(), Some(12));
            // our own camelCase serialisation as `cfg` is accepted as well
            let camel = serde_json::to_value(&model).unwrap().as_object().unwrap().clone();
            let from_camel = ParsedCfg::from_cfg(&camel).unwrap();
            assert_eq!(from_camel.model, model, "{preset:?} via camelCase");
            assert_eq!(suggest_preset(&model), preset);
        }
        // a custom shape is "imported", and a different expert count alone does not change the preset
        let mut custom = Preset::Small.model();
        custom.pool_experts = 99;
        assert_eq!(suggest_preset(&custom), Preset::Small);
        custom.d_model = 400;
        assert_eq!(suggest_preset(&custom), Preset::Imported);
    }

    #[test]
    fn unsupported_python_only_settings_are_reported_in_plain_english() {
        let mut p = ParsedCfg::from_model(Preset::Tiny.model(), 8);
        assert!(p.python_only_problems().is_empty());
        p.python_only.insert("use_pool".into(), false.into());
        p.python_only.insert("tie_embeddings".into(), false.into());
        p.python_only.insert("rope_theta".into(), 500000.0.into());
        let probs = p.python_only_problems();
        assert_eq!(probs.len(), 3);
        assert!(
            probs[0].contains("no expert pool")
                && probs[1].contains("separate input and output")
                && probs[2].contains("500000")
        );
    }

    #[test]
    fn output_is_indented_like_python_and_roundtrips_through_a_file() {
        let m = Manifest::parse(PY_MANIFEST).unwrap();
        let out = m.to_json().unwrap();
        assert!(out.starts_with("{\n \"step\": 2,"), "{}", &out[..40.min(out.len())]);
        assert!(!out.ends_with('\n'), "json.dump writes no trailing newline");
        let d = fsio::scratch_dir("manifest");
        let path = d.join(MANIFEST_FILE);
        m.write(&path, false).unwrap();
        assert_eq!(Manifest::read(&path).unwrap(), m);
        assert!(!fsio::tmp_path(&path).exists());
        // a missing file and a broken file name the path
        let e = Manifest::read(&d.join("nope.json")).unwrap_err().to_string();
        assert!(e.contains("nope.json"), "{e}");
        std::fs::write(d.join("bad.json"), "{nope").unwrap();
        let e = Manifest::read(&d.join("bad.json")).unwrap_err().to_string();
        assert!(e.contains("bad.json") && e.contains("manifest.json problem"), "{e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn fresh_telemetry_matches_a_new_pool() {
        let t = Telemetry::fresh(&[0, 1, 5]);
        assert_eq!((t.gate.len(), t.ever.len(), t.uid.len()), (3, 3, 3));
        assert_eq!((t.next_uid, t.segments), (Some(6), Some(0)));
        assert!(t.gate.iter().all(|&g| g == 1.0));
        let json = serde_json::to_value(&t).unwrap();
        for k in
            ["gate", "gate_seen", "use", "admits", "born", "last_seen", "recent", "ever", "uid", "next_uid", "segments"]
        {
            assert!(json.get(k).is_some(), "telemetry key {k} is written");
        }
    }
}
