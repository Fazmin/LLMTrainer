//! The checkpoint directory: the whole model on disk, in a layout the Python reference reads too.
//!
//! ```text
//! <checkpoint>/
//!   manifest.json     the index (see [`super::manifest`])
//!   core.npz          trunk tensors, fp32, under the Python `state_dict()` names
//!   routers.npz       recur.0.mlp.router.weight [n, d_model], recur.0.mlp.depth_emb [d_model], pool.gate [n]
//!   optim.npz         Adam state of everything that is not an expert
//!   experts/          one file per expert, e00000.npz ... (see [`super::tiers`])
//!   COMPLETE          written last; the app lists a checkpoint only if this exists
//! ```
//!
//! # Names, shapes and where things live
//!
//! * `core.npz` has `tok_emb.weight [vocab, d_model]` **and** `head.weight` (the same values: the
//!   Python code stores a tied embedding twice), `prelude.<i>.{ln1,ln2}.weight [d_model]`,
//!   `prelude.<i>.attn.qkv.weight [3 d_model, d_model]`, `...attn.proj.weight [d_model, d_model]`,
//!   `prelude.<i>.mlp.{w1,w3}.weight [d_ff, d_model]`, `...mlp.w2.weight [d_model, d_ff]`, the same
//!   attention/norm tensors for `recur.0`, `adapter.weight [d_model, 2 d_model]`, `ln_f.weight`,
//!   `halt.weight [1, d_model]` and `halt.bias [1]`. [`expected_core_tensors`] lists them for a config.
//! * `routers.npz` has one router per recurrent call site (rows belong to *experts*, in pool order)
//!   and the gates, `pool.gate [n]`. A directory written by the non-paged Python save keeps
//!   `pool.gate` in `core.npz` instead; [`Checkpoint::read`] moves it so callers always find it in
//!   `routers`.
//! * `optim.npz` has, per parameter `<name>`: `<name>|m` and `<name>|v` (bf16 bit patterns in `<i2`)
//!   and `<name>|t` (the step count, a 0-d `<f8`). The tied embedding appears once, as
//!   `tok_emb.weight`. The three slot tensors of the paged pool appear as `pool.w1`, `pool.w3` and
//!   `pool.w2` with the shape `[resident, ...]`: those moments belong to *whichever experts happened
//!   to sit on the card* when they were saved, the experts' own moments live in their files (the
//!   Python reference overwrites the slots from the files before its first step), so only their
//!   `|t` step count matters. They are large (at Full size 800 MB as f32), so [`Checkpoint::read`]
//!   skips the arrays and keeps only the step count unless [`ReadOptions::slot_moments`] is set.
//!
//! # Atomic writes and hard-linked experts
//!
//! [`Checkpoint::write`] builds the directory under a sibling name (`.tmp-<name>`), writes the
//! `COMPLETE` marker as the very last file, and renames the whole directory into place, so a
//! checkpoint either exists completely or not at all. Experts are the bulk of the bytes
//! (1.6 GB at Full size with 64 experts), so from live [`Tiers`] they are **hard-linked**, not
//! copied (falling back to a copy where links are impossible). That is safe because the tiers never
//! rewrite a file in place: a later write-back replaces the *name* and leaves the checkpoint's file
//! as it was.
//!
//! The reader does **not** require `COMPLETE`: a directory written by the Python reference has
//! none. [`Checkpoint::inspect`] reports whether it is there.

use super::manifest::{
    JsonMap, MANIFEST_FILE, Manifest, MinagiRs, ORIGIN_NATIVE, ParsedCfg, RS_FORMAT, Telemetry, suggest_preset,
};
use super::npz::{NpzReader, NpzWriter};
use super::tiers::{
    ExpertEntry, TierConfig, Tiers, describe_expert_file, expert_file_name, parse_expert_file_name, write_expert_file,
};
use super::{NpyData, Result, StoreError, bf16, fsio};
use minagi_types::config::{ModelConfig, Preset};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const COMPLETE_FILE: &str = "COMPLETE";
pub const CORE_FILE: &str = "core.npz";
pub const ROUTERS_FILE: &str = "routers.npz";
pub const OPTIM_FILE: &str = "optim.npz";
pub const EXPERTS_DIR: &str = "experts";
/// The gate vector's name in `routers.npz` (and in `optim.npz`).
pub const GATE_KEY: &str = "pool.gate";

// ---------------------------------------------------------------------------------------------
// plain tensors
// ---------------------------------------------------------------------------------------------

/// A dense row-major `f32` tensor: shape and flat data. Candle-free on purpose.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Build a tensor, checking that `data` has `shape.iter().product()` elements.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
        if n != Some(data.len()) {
            return Err(StoreError::Invalid(format!("tensor shape {shape:?} does not match {} values", data.len())));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { shape, data: vec![0.0; n] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Tensors by name, in insertion order (which is the order they have in the file).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NamedTensors {
    items: Vec<(String, Tensor)>,
}

impl NamedTensors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor, replacing any existing one of the same name (keeping its position).
    pub fn insert(&mut self, name: impl Into<String>, t: Tensor) {
        let name = name.into();
        match self.items.iter_mut().find(|(k, _)| *k == name) {
            Some(slot) => slot.1 = t,
            None => self.items.push((name, t)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.items.iter().find(|(k, _)| k == name).map(|(_, t)| t)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.items.iter_mut().find(|(k, _)| k == name).map(|(_, t)| t)
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        let i = self.items.iter().position(|(k, _)| k == name)?;
        Some(self.items.remove(i).1)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.items.iter().map(|(k, _)| k.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.items.iter().map(|(k, t)| (k.as_str(), t))
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Name -> shape of every tensor.
    pub fn shapes(&self) -> BTreeMap<String, Vec<usize>> {
        self.items.iter().map(|(k, t)| (k.clone(), t.shape.clone())).collect()
    }
}

impl FromIterator<(String, Tensor)> for NamedTensors {
    fn from_iter<I: IntoIterator<Item = (String, Tensor)>>(iter: I) -> Self {
        let mut s = Self::new();
        for (k, t) in iter {
            s.insert(k, t);
        }
        s
    }
}

/// Adam state of one parameter: first and second moment (same shape as the parameter) and the
/// optimiser's step count for it. In the file the moments are bf16; here they are widened to `f32`
/// (exactly: every bf16 value is an `f32`), and rounded back, ties to even, when written.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamState {
    pub name: String,
    pub m: Option<Tensor>,
    pub v: Option<Tensor>,
    pub step: Option<f64>,
}

/// Whether `name` is one of the paged pool's three slot tensors (`pool.w1`, `pool.w3`, `pool.w2`),
/// whose moments are indexed by card slot rather than by expert (see the module docs).
pub fn is_slot_indexed(name: &str) -> bool {
    matches!(name, "pool.w1" | "pool.w3" | "pool.w2")
}

/// The contents of `optim.npz`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OptimState {
    pub params: Vec<ParamState>,
}

impl OptimState {
    pub fn get(&self, name: &str) -> Option<&ParamState> {
        self.params.iter().find(|p| p.name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------
// what a checkpoint should contain
// ---------------------------------------------------------------------------------------------

fn expected_block(out: &mut Vec<(String, Vec<usize>)>, prefix: &str, d: usize, dense_mlp_ff: Option<usize>) {
    out.push((format!("{prefix}.ln1.weight"), vec![d]));
    out.push((format!("{prefix}.attn.qkv.weight"), vec![3 * d, d]));
    out.push((format!("{prefix}.attn.proj.weight"), vec![d, d]));
    out.push((format!("{prefix}.ln2.weight"), vec![d]));
    if let Some(ff) = dense_mlp_ff {
        out.push((format!("{prefix}.mlp.w1.weight"), vec![ff, d]));
        out.push((format!("{prefix}.mlp.w3.weight"), vec![ff, d]));
        out.push((format!("{prefix}.mlp.w2.weight"), vec![d, ff]));
    }
}

/// The tensors `core.npz` must hold for `m`, with their shapes, in the Python `state_dict()` order.
pub fn expected_core_tensors(m: &ModelConfig) -> Vec<(String, Vec<usize>)> {
    let d = m.d_model as usize;
    let vocab = m.vocab_size as usize;
    let mut out = vec![("tok_emb.weight".to_string(), vec![vocab, d])];
    for i in 0..m.n_prelude {
        expected_block(&mut out, &format!("prelude.{i}"), d, Some(m.d_ff as usize));
    }
    // recurrent and coda blocks route into the pool, so they have no dense MLP of their own
    for i in 0..m.n_recur {
        expected_block(&mut out, &format!("recur.{i}"), d, None);
    }
    for i in 0..m.n_coda {
        expected_block(&mut out, &format!("coda.{i}"), d, None);
    }
    out.push(("adapter.weight".into(), vec![d, 2 * d]));
    out.push(("ln_f.weight".into(), vec![d]));
    out.push(("head.weight".into(), vec![vocab, d]));
    out.push(("halt.weight".into(), vec![1, d]));
    out.push(("halt.bias".into(), vec![1]));
    out
}

/// The tensors `routers.npz` must hold for `m` and a pool of `n_experts`.
pub fn expected_router_tensors(m: &ModelConfig, n_experts: usize) -> Vec<(String, Vec<usize>)> {
    let d = m.d_model as usize;
    let mut out = Vec::new();
    let sites = (0..m.n_recur).map(|i| format!("recur.{i}")).chain((0..m.n_coda).map(|i| format!("coda.{i}")));
    for site in sites {
        out.push((format!("{site}.mlp.depth_emb"), vec![d]));
        out.push((format!("{site}.mlp.router.weight"), vec![n_experts, d]));
    }
    out.push((GATE_KEY.into(), vec![n_experts]));
    out
}

/// Compare shapes against an expected list; returns one plain-English line per problem (missing
/// tensor, wrong shape). Extra tensors are not a problem.
pub fn check_shapes(file: &str, have: &BTreeMap<String, Vec<usize>>, want: &[(String, Vec<usize>)]) -> Vec<String> {
    let mut out = Vec::new();
    for (name, shape) in want {
        match have.get(name) {
            None => out.push(format!("{file} has no tensor named {name:?} (expected shape {shape:?})")),
            Some(got) if got != shape => {
                out.push(format!("{file}: tensor {name:?} has shape {got:?} but the model config needs {shape:?}"));
            }
            Some(_) => {}
        }
    }
    out
}

/// [`check_shapes`] for tensors in memory.
pub fn check_tensors(file: &str, have: &NamedTensors, want: &[(String, Vec<usize>)]) -> Vec<String> {
    check_shapes(file, &have.shapes(), want)
}

/// Everything in a checkpoint except the experts' own files.
///
/// For writing, the caller fills `manifest` (step, val, cfg, telemetry, read counters, plasticity,
/// `minagi_rs`) and the tensors; see [`Manifest`] for which manifest fields are derived.
#[derive(Debug, Clone, Default)]
pub struct CheckpointData {
    pub manifest: Manifest,
    pub core: NamedTensors,
    pub routers: NamedTensors,
    pub optim: OptimState,
}

/// Where [`Checkpoint::write`] gets the experts from.
pub enum ExpertSource<'a> {
    /// Live tiers: dirty experts are flushed, then each expert's file is hard-linked into the
    /// checkpoint (copied if linking is impossible). Nothing is read into memory.
    Tiers(&'a mut Tiers),
    /// Experts held in memory, written as new files.
    Entries(&'a [(u64, Arc<ExpertEntry>)]),
    /// Another `experts/` directory (for example a previous checkpoint's); files are linked/copied.
    Dir(&'a Path),
}

/// Options for [`Checkpoint::write_with`].
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Replace the destination if it exists (the old one is moved aside first and removed only
    /// after the new one is in place). Default: refuse.
    pub overwrite: bool,
    /// Force the bundles, the manifest and the marker to disk before the final rename.
    pub durable: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self { overwrite: false, durable: true }
    }
}

/// What [`Checkpoint::write`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteReport {
    pub dir: PathBuf,
    pub n_experts: usize,
    /// Bytes in all files of the checkpoint (hard-linked experts included, counted once per link).
    pub total_bytes: u64,
    pub experts_linked: usize,
    pub experts_copied: usize,
    pub experts_written: usize,
}

/// Options for [`Checkpoint::read_with`].
#[derive(Debug, Clone, Default)]
pub struct ReadOptions {
    /// Also load the moments of `pool.w1/w3/w2` (slot-indexed, huge, normally useless).
    pub slot_moments: bool,
    /// Skip `optim.npz` entirely.
    pub skip_optim: bool,
}

/// A checkpoint read from disk: the trunk in memory, the experts left in their files.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub dir: PathBuf,
    pub data: CheckpointData,
    pub has_complete_marker: bool,
    /// Things the reader noticed and tolerated (unknown entries in `optim.npz`, `pool.gate` found
    /// in `core.npz`, ...).
    pub warnings: Vec<String>,
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> StoreError + '_ {
    move |e| StoreError::io_at(path, e)
}

fn tensor_from_npy(file: &str, key: &str, arr: super::NpyArray) -> Result<Tensor> {
    let shape = arr.shape().to_vec();
    match arr.into_data() {
        NpyData::F32(data) => Ok(Tensor { shape, data }),
        other => Err(StoreError::Invalid(format!(
            "{file}: tensor {key:?} has dtype {} but trunk tensors are stored as <f4",
            other.descr()
        ))),
    }
}

/// Read every array of a bundle (`core.npz`, `routers.npz`) as `f32` tensors, in file order.
pub fn read_bundle(path: &Path) -> Result<NamedTensors> {
    let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut z = NpzReader::open(path).map_err(|e| match e {
        StoreError::Io(io_err) => StoreError::io_at(path, io_err),
        other => StoreError::Invalid(format!("{} is not a readable .npz archive ({other})", path.display())),
    })?;
    let mut out = NamedTensors::new();
    for (k, a) in z.read_all()? {
        let t = tensor_from_npy(&file, &k, a)?;
        out.insert(k, t);
    }
    Ok(out)
}

/// Write tensors as a bundle (stored npz entries, atomically). Returns the file size.
pub fn write_bundle(path: &Path, items: &NamedTensors) -> Result<u64> {
    write_named(path, items, false)
}

fn write_named(path: &Path, items: &NamedTensors, durable: bool) -> Result<u64> {
    fsio::write_atomic(path, durable, |w| {
        let mut z = NpzWriter::new(w);
        for (k, t) in items.iter() {
            z.add_slice(k, &t.shape, &t.data)?;
        }
        z.finish()?;
        Ok(())
    })
}

fn write_optim(path: &Path, optim: &OptimState, durable: bool) -> Result<u64> {
    fsio::write_atomic(path, durable, |w| {
        let mut z = NpzWriter::new(w);
        for p in &optim.params {
            if let (Some(m), Some(v)) = (&p.m, &p.v) {
                if m.shape != v.shape {
                    return Err(StoreError::Invalid(format!(
                        "optimiser state of {}: m has shape {:?} but v has {:?}",
                        p.name, m.shape, v.shape
                    )));
                }
                z.add_slice(&format!("{}|m", p.name), &m.shape, &bf16::pack(&m.data))?;
                z.add_slice(&format!("{}|v", p.name), &v.shape, &bf16::pack(&v.data))?;
            } else if p.m.is_some() || p.v.is_some() {
                return Err(StoreError::Invalid(format!("optimiser state of {} has only one of m and v", p.name)));
            }
            if let Some(t) = p.step {
                z.add_slice(&format!("{}|t", p.name), &[], &[t])?;
            }
        }
        z.finish()?;
        Ok(())
    })
}

fn read_optim(path: &Path, opts: &ReadOptions, warnings: &mut Vec<String>) -> Result<OptimState> {
    let mut z = NpzReader::open(path).map_err(|e| match e {
        StoreError::Io(io_err) => StoreError::io_at(path, io_err),
        other => StoreError::Invalid(format!("{} is not a readable .npz archive ({other})", path.display())),
    })?;
    let mut order: Vec<String> = Vec::new();
    let mut slots: BTreeMap<String, (bool, bool, bool)> = BTreeMap::new();
    for key in z.keys().to_vec() {
        let Some((name, kind)) = key.rsplit_once('|') else {
            warnings.push(format!("optim.npz: ignored the entry {key:?} (not of the form <param>|m, |v or |t)"));
            continue;
        };
        if !matches!(kind, "m" | "v" | "t") {
            warnings.push(format!("optim.npz: ignored the entry {key:?} (not of the form <param>|m, |v or |t)"));
            continue;
        }
        let e = slots.entry(name.to_string()).or_insert_with(|| {
            order.push(name.to_string());
            (false, false, false)
        });
        match kind {
            "m" => e.0 = true,
            "v" => e.1 = true,
            _ => e.2 = true,
        }
    }
    let mut params = Vec::new();
    for name in order {
        let (has_m, has_v, has_t) = slots[&name];
        if has_m != has_v {
            return Err(StoreError::Invalid(format!(
                "optim.npz has {name}|{} but not {name}|{}",
                if has_m { "m" } else { "v" },
                if has_m { "v" } else { "m" }
            )));
        }
        let mut state = ParamState { name: name.clone(), m: None, v: None, step: None };
        if has_m && (opts.slot_moments || !is_slot_indexed(&name)) {
            let mut moment = |suffix: &str| -> Result<Tensor> {
                let key = format!("{name}|{suffix}");
                let a = z.get(&key)?;
                let shape = a.shape().to_vec();
                let data = match a.into_data() {
                    NpyData::I16(bits) => bf16::unpack(&bits),
                    NpyData::F32(v) => v, // an older file kept the moments in fp32
                    other => {
                        return Err(StoreError::Invalid(format!(
                            "optim.npz: {key:?} has dtype {} but moments are stored as <i2 (bf16 bits) or <f4",
                            other.descr()
                        )));
                    }
                };
                Tensor::new(shape, data)
            };
            let m = moment("m")?;
            let v = moment("v")?;
            if m.shape != v.shape {
                return Err(StoreError::Invalid(format!(
                    "optim.npz: {name}|m has shape {:?} but {name}|v has {:?}",
                    m.shape, v.shape
                )));
            }
            state.m = Some(m);
            state.v = Some(v);
        }
        if has_t {
            let key = format!("{name}|t");
            let a = z.get(&key)?;
            let one = match a.data() {
                NpyData::F64(v) if v.len() == 1 => Some(v[0]),
                NpyData::F32(v) if v.len() == 1 => Some(f64::from(v[0])),
                NpyData::I64(v) if v.len() == 1 => Some(v[0] as f64),
                NpyData::I32(v) if v.len() == 1 => Some(f64::from(v[0])),
                _ => None,
            };
            state.step = Some(one.ok_or_else(|| {
                StoreError::Invalid(format!(
                    "optim.npz: {key:?} should be a single number, found shape {:?}",
                    a.shape()
                ))
            })?);
        }
        params.push(state);
    }
    Ok(OptimState { params })
}

/// The pool's uids in position order: `telemetry.uid` if the manifest has it, else `0..n` where `n`
/// is the length of `pool.gate`.
fn resolve_uids(data: &CheckpointData) -> Result<Vec<u64>> {
    if let Some(t) = &data.manifest.telemetry
        && !t.uid.is_empty()
    {
        return Ok(t.uid.clone());
    }
    let gate = data.routers.get(GATE_KEY).ok_or_else(|| {
        StoreError::Invalid(format!("routers has no {GATE_KEY:?}, so the number of experts is unknown"))
    })?;
    Ok((0..gate.numel() as u64).collect())
}

impl Checkpoint {
    /// Write a complete checkpoint directory at `dir` (which must not exist). See the module docs.
    pub fn write(dir: &Path, data: &CheckpointData, experts: ExpertSource<'_>) -> Result<WriteReport> {
        Self::write_with(dir, data, experts, &WriteOptions::default())
    }

    /// [`write`](Checkpoint::write) with options.
    pub fn write_with(
        dir: &Path,
        data: &CheckpointData,
        experts: ExpertSource<'_>,
        opts: &WriteOptions,
    ) -> Result<WriteReport> {
        // -- check what we were given before touching the disk --------------------------------
        let parsed = data
            .manifest
            .parsed_cfg()
            .map_err(|e| StoreError::Invalid(format!("the manifest's cfg is not usable: {e}")))?
            .ok_or_else(|| StoreError::Invalid("the manifest has no cfg, so the model's shape is unknown".into()))?;
        let model = &parsed.model;
        let (d_model, d_ff) = (model.d_model as usize, model.pool_d_ff as usize);
        let uids = resolve_uids(data)?;
        let n = uids.len();
        if BTreeSet::from_iter(uids.iter().copied()).len() != n {
            return Err(StoreError::Invalid("two experts have the same uid".into()));
        }
        let mut problems = check_tensors(CORE_FILE, &data.core, &expected_core_tensors(model));
        problems.extend(check_tensors(ROUTERS_FILE, &data.routers, &expected_router_tensors(model, n)));
        if let Some(t) = &data.manifest.telemetry {
            let lens = [
                ("gate", t.gate.len()),
                ("gate_seen", t.gate_seen.len()),
                ("use", t.usage.len()),
                ("admits", t.admits.len()),
                ("born", t.born.len()),
                ("last_seen", t.last_seen.len()),
                ("recent", t.recent.len()),
                ("ever", t.ever.len()),
            ];
            for (name, len) in lens {
                if len != 0 && len != n {
                    problems.push(format!("telemetry.{name} has {len} entries for {n} experts"));
                }
            }
        }
        if let Some(first) = problems.first() {
            let more =
                if problems.len() > 1 { format!(" (and {} more problems)", problems.len() - 1) } else { String::new() };
            return Err(StoreError::Invalid(format!("refusing to write an inconsistent checkpoint: {first}{more}")));
        }
        if let ExpertSource::Tiers(t) = &experts
            && (t.d_model(), t.d_ff()) != (d_model, d_ff)
        {
            return Err(StoreError::Invalid(format!(
                "the expert store holds experts of {} x {} but the model config says {d_model} x {d_ff}",
                t.d_ff(),
                t.d_model()
            )));
        }

        // -- the destination and a temporary sibling --------------------------------------------
        let name = dir
            .file_name()
            .ok_or_else(|| StoreError::Invalid(format!("{} is not a usable checkpoint path", dir.display())))?
            .to_os_string();
        let parent = match dir.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        fs::create_dir_all(&parent).map_err(io(&parent))?;
        if dir.exists() && !opts.overwrite {
            return Err(StoreError::AlreadyExists(dir.to_path_buf()));
        }
        let mut tmp_name = std::ffi::OsString::from(".tmp-");
        tmp_name.push(&name);
        let tmp = parent.join(tmp_name);
        if tmp.exists() {
            fs::remove_dir_all(&tmp).map_err(io(&tmp))?;
        }
        fs::create_dir(&tmp).map_err(io(&tmp))?;

        let built = Self::build(&tmp, data, &parsed, &uids, experts, opts);
        let report = match built {
            Ok(r) => r,
            Err(e) => {
                let _ = fs::remove_dir_all(&tmp);
                return Err(e);
            }
        };

        // -- put it in place --------------------------------------------------------------------
        let mut aside = None;
        if dir.exists() {
            let mut a = std::ffi::OsString::from(".old-");
            a.push(&name);
            let a = parent.join(a);
            let _ = fs::remove_dir_all(&a);
            fs::rename(dir, &a).map_err(|e| {
                let _ = fs::remove_dir_all(&tmp);
                StoreError::io_at(dir, e)
            })?;
            aside = Some(a);
        }
        if let Err(e) = fsio::rename_replace(&tmp, dir) {
            // put the old one back so nothing is lost
            if let Some(a) = &aside {
                let _ = fs::rename(a, dir);
            }
            let _ = fs::remove_dir_all(&tmp);
            return Err(StoreError::io_at(dir, e));
        }
        if opts.durable {
            fsio::sync_dir(&parent);
        }
        if let Some(a) = aside {
            let _ = fs::remove_dir_all(a);
        }
        Ok(WriteReport { dir: dir.to_path_buf(), ..report })
    }

    /// Fill `tmp` with the complete checkpoint, `COMPLETE` last.
    fn build(
        tmp: &Path,
        data: &CheckpointData,
        parsed: &ParsedCfg,
        uids: &[u64],
        experts: ExpertSource<'_>,
        opts: &WriteOptions,
    ) -> Result<WriteReport> {
        let model = &parsed.model;
        let (d_model, d_ff) = (model.d_model as usize, model.pool_d_ff as usize);
        let n = uids.len();
        let exp_dir = tmp.join(EXPERTS_DIR);
        fs::create_dir(&exp_dir).map_err(io(&exp_dir))?;

        // experts first: they are the bulk, and a missing one should fail before anything else is written
        let (mut linked, mut copied, mut written) = (0, 0, 0);
        let mut experts = experts;
        if let ExpertSource::Tiers(t) = &mut experts {
            t.flush()?;
        }
        let mut have_moments: Vec<bool> = Vec::with_capacity(n);
        let by_uid: HashMap<u64, &Arc<ExpertEntry>> = match &experts {
            ExpertSource::Entries(list) => list.iter().map(|(u, e)| (*u, e)).collect(),
            _ => HashMap::new(),
        };
        for &uid in uids {
            let file = expert_file_name(uid);
            let dst = exp_dir.join(&file);
            match &experts {
                ExpertSource::Tiers(t) => {
                    let src = t.path_of(uid);
                    if !src.exists() {
                        return Err(StoreError::ExpertMissing { uid, path: src });
                    }
                    match fsio::link_or_copy(&src, &dst)? {
                        fsio::LinkKind::Linked => linked += 1,
                        fsio::LinkKind::Copied => copied += 1,
                    }
                    have_moments.push(describe_expert_file(&dst)?.has_moments);
                }
                ExpertSource::Dir(d) => {
                    let src = d.join(&file);
                    if !src.exists() {
                        return Err(StoreError::ExpertMissing { uid, path: src });
                    }
                    match fsio::link_or_copy(&src, &dst)? {
                        fsio::LinkKind::Linked => linked += 1,
                        fsio::LinkKind::Copied => copied += 1,
                    }
                    have_moments.push(describe_expert_file(&dst)?.has_moments);
                }
                ExpertSource::Entries(_) => {
                    let Some(e) = by_uid.get(&uid) else {
                        return Err(StoreError::ExpertMissing { uid, path: dst });
                    };
                    write_expert_file(&dst, e, d_model, d_ff, false)?;
                    written += 1;
                    have_moments.push(e.moments.is_some());
                }
            }
        }

        // bundles
        write_named(&tmp.join(CORE_FILE), &data.core, opts.durable)?;
        write_named(&tmp.join(ROUTERS_FILE), &data.routers, opts.durable)?;
        if !data.optim.is_empty() {
            write_optim(&tmp.join(OPTIM_FILE), &data.optim, opts.durable)?;
        }

        // the manifest, completed from what is on disk
        let gate: Vec<f32> = data.routers.get(GATE_KEY).map(|t| t.data.clone()).unwrap_or_default();
        let mut records = Vec::with_capacity(n);
        for (i, &uid) in uids.iter().enumerate() {
            let file = expert_file_name(uid);
            let bytes = fs::metadata(exp_dir.join(&file)).map_err(io(&exp_dir.join(&file)))?.len();
            records.push(super::manifest::ExpertRecord {
                id: uid,
                file,
                bytes,
                params: (3 * d_model * d_ff) as u64,
                moments: have_moments[i],
                gate: f64::from(gate.get(i).copied().unwrap_or(1.0)),
                extra: JsonMap::new(),
            });
        }
        let mut m = data.manifest.clone();
        let mut model_for_cfg = parsed.clone();
        // the ceiling is never below what the pool holds (Python ratchets it the same way)
        model_for_cfg.model.pool_max = model_for_cfg.model.pool_max.max(n as u32);
        let mut cfg = model_for_cfg.to_cfg();
        if let Some(t) = &m.telemetry
            && !t.ever.is_empty()
        {
            cfg.insert("pool_ever".into(), serde_json::json!(t.ever));
        }
        m.cfg = Some(cfg);
        let tel = m.telemetry.get_or_insert_with(|| Telemetry::fresh(uids));
        tel.uid = uids.to_vec();
        let floor = uids.iter().max().map_or(0, |u| u + 1);
        tel.next_uid = Some(tel.next_uid.unwrap_or(0).max(floor));
        m.n_experts = n;
        m.d_model = Some(d_model);
        m.d_ff = Some(d_ff);
        m.paged = true;
        let mut core_names: Vec<String> = data.core.names().map(str::to_string).collect();
        core_names.sort();
        let mut router_names: Vec<String> = data.routers.names().map(str::to_string).collect();
        router_names.sort();
        m.core_tensors = core_names;
        m.router_tensors = router_names;
        m.experts = records;
        m.removed_expert_files = None;
        m.total_bytes = fsio::dir_size(tmp);
        let rs = m.minagi_rs.get_or_insert_with(|| MinagiRs::native(Some(suggest_preset(&model_for_cfg.model))));
        rs.format = RS_FORMAT;
        rs.writer.get_or_insert_with(|| format!("minagi-core {}", env!("CARGO_PKG_VERSION")));
        rs.origin.get_or_insert_with(|| ORIGIN_NATIVE.to_string());
        m.write(&tmp.join(MANIFEST_FILE), opts.durable)?;
        let total_bytes = m.total_bytes;

        // the marker is the last thing written
        let marker = tmp.join(COMPLETE_FILE);
        fsio::write_atomic(&marker, opts.durable, |w| {
            use std::io::Write;
            w.write_all(b"ok")?;
            Ok(())
        })?;
        if opts.durable {
            fsio::sync_dir(&exp_dir);
            fsio::sync_dir(tmp);
        }
        Ok(WriteReport {
            dir: tmp.to_path_buf(),
            n_experts: n,
            total_bytes,
            experts_linked: linked,
            experts_copied: copied,
            experts_written: written,
        })
    }

    /// Read a checkpoint directory (trunk, routers and optimiser state into memory; the experts stay
    /// in their files). `COMPLETE` is not required.
    pub fn read(dir: &Path) -> Result<Checkpoint> {
        Self::read_with(dir, &ReadOptions::default())
    }

    /// [`read`](Checkpoint::read) with options.
    pub fn read_with(dir: &Path, opts: &ReadOptions) -> Result<Checkpoint> {
        let manifest_path = dir.join(MANIFEST_FILE);
        if !dir.is_dir() {
            return Err(StoreError::Invalid(format!("{} is not a folder", dir.display())));
        }
        if !manifest_path.is_file() {
            return Err(StoreError::Invalid(format!(
                "{} has no {MANIFEST_FILE}, so it is not a checkpoint folder",
                dir.display()
            )));
        }
        let manifest = Manifest::read(&manifest_path)?;
        let mut warnings = Vec::new();
        for (file, what) in [(CORE_FILE, "the trunk weights"), (ROUTERS_FILE, "the routers and gates")] {
            if !dir.join(file).is_file() {
                return Err(StoreError::Invalid(format!("{} has no {file} ({what})", dir.display())));
            }
        }
        let mut core = read_bundle(&dir.join(CORE_FILE))?;
        let mut routers = read_bundle(&dir.join(ROUTERS_FILE))?;
        if !routers.contains(GATE_KEY) {
            // the non-paged Python save keeps the gate in core.npz
            if let Some(g) = core.remove(GATE_KEY) {
                routers.insert(GATE_KEY, g);
                warnings.push("pool.gate was in core.npz (a non-paged Python save); moved it to routers".into());
            }
        }
        let optim_path = dir.join(OPTIM_FILE);
        let optim = if optim_path.is_file() && !opts.skip_optim {
            read_optim(&optim_path, opts, &mut warnings)?
        } else {
            OptimState::default()
        };
        Ok(Checkpoint {
            dir: dir.to_path_buf(),
            data: CheckpointData { manifest, core, routers, optim },
            has_complete_marker: dir.join(COMPLETE_FILE).is_file(),
            warnings,
        })
    }

    /// The `experts/` directory of this checkpoint.
    pub fn experts_dir(&self) -> PathBuf {
        self.dir.join(EXPERTS_DIR)
    }

    /// Pool order -> uid.
    pub fn uids(&self) -> Vec<u64> {
        self.data.manifest.uids()
    }

    fn pool_dims(&self) -> Result<(usize, usize)> {
        let m = &self.data.manifest;
        if let (Some(dm), Some(df)) = (m.d_model, m.d_ff) {
            return Ok((dm, df));
        }
        match m.parsed_cfg() {
            Ok(Some(p)) => Ok((p.model.d_model as usize, p.model.pool_d_ff as usize)),
            _ => Err(StoreError::Invalid("the manifest does not say how large the experts are".into())),
        }
    }

    /// Open this checkpoint's experts as [`Tiers`]. Use `read_only = true` to look without
    /// touching; to *train* from a checkpoint, [`materialize_experts`](Checkpoint::materialize_experts)
    /// them into the run's own directory first.
    pub fn open_tiers(&self, ram_capacity: usize, read_only: bool) -> Result<Tiers> {
        let (d_model, d_ff) = self.pool_dims()?;
        Tiers::open(TierConfig::new(self.experts_dir(), d_model, d_ff).ram_capacity(ram_capacity).read_only(read_only))
    }

    /// Uids the manifest lists whose file is not in `experts/`.
    pub fn missing_experts(&self) -> Vec<u64> {
        let dir = self.experts_dir();
        self.uids().into_iter().filter(|&u| !dir.join(expert_file_name(u)).is_file()).collect()
    }

    /// Put this checkpoint's expert files into `dest` (created if needed) by hard link, or by copy
    /// where linking is impossible, so a run can start from the checkpoint and write its own
    /// changes without touching it. Returns `(linked, copied)`.
    pub fn materialize_experts(&self, dest: &Path) -> Result<(usize, usize)> {
        fs::create_dir_all(dest).map_err(io(dest))?;
        let (mut linked, mut copied) = (0, 0);
        for uid in self.uids() {
            let file = expert_file_name(uid);
            let src = self.experts_dir().join(&file);
            if !src.is_file() {
                return Err(StoreError::ExpertMissing { uid, path: src });
            }
            let dst = dest.join(&file);
            if dst.exists() {
                fs::remove_file(&dst).map_err(io(&dst))?;
            }
            match fsio::link_or_copy(&src, &dst)? {
                fsio::LinkKind::Linked => linked += 1,
                fsio::LinkKind::Copied => copied += 1,
            }
        }
        Ok((linked, copied))
    }

    /// Deep check: every listed expert file exists, is a readable archive and has the right shape;
    /// the trunk matches the config's expected tensors. Reads headers only, so it is fast even at
    /// Full size. Returns the list of problems (empty = healthy).
    pub fn verify(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let n = self.uids().len();
        match self.data.manifest.parsed_cfg() {
            Ok(Some(p)) => {
                problems.extend(check_tensors(CORE_FILE, &self.data.core, &expected_core_tensors(&p.model)));
                problems.extend(check_tensors(ROUTERS_FILE, &self.data.routers, &expected_router_tensors(&p.model, n)));
            }
            Ok(None) => problems.push("the manifest has no cfg".into()),
            Err(e) => problems.push(format!("the manifest's cfg is not usable: {e}")),
        }
        let dims = self.pool_dims().ok();
        for uid in self.uids() {
            let path = self.experts_dir().join(expert_file_name(uid));
            if !path.is_file() {
                problems.push(format!("expert e{uid:05} is missing ({})", path.display()));
                continue;
            }
            match describe_expert_file(&path) {
                Err(e) => problems.push(format!("expert e{uid:05} is unreadable: {e}")),
                Ok(info) => {
                    if !info.has_w1_w3_w2 {
                        problems.push(format!("expert e{uid:05} lacks one of w1, w3, w2"));
                    } else if let (Some((dm, df)), Some(shape)) = (dims, &info.w1_shape)
                        && shape != &[df, dm]
                    {
                        problems.push(format!("expert e{uid:05} has w1 of shape {shape:?}, expected [{df}, {dm}]"));
                    }
                }
            }
        }
        problems
    }

    /// A cheap look at a directory, for the "import" preview and the checkpoint list: reads only
    /// `manifest.json` and file metadata.
    pub fn inspect(dir: &Path) -> Result<CheckpointInfo> {
        if !dir.is_dir() {
            return Err(StoreError::Invalid(format!("{} is not a folder", dir.display())));
        }
        let manifest_path = dir.join(MANIFEST_FILE);
        if !manifest_path.is_file() {
            return Err(StoreError::Invalid(format!(
                "{} has no {MANIFEST_FILE}, so it is not a checkpoint folder",
                dir.display()
            )));
        }
        let m = Manifest::read(&manifest_path)?;
        let mut problems = Vec::new();
        let (config, config_defaulted) = match m.parsed_cfg() {
            Ok(Some(p)) => (Some(p.model), p.defaulted),
            Ok(None) => {
                problems.push("the manifest records no model configuration".to_string());
                (None, Vec::new())
            }
            Err(e) => {
                problems.push(format!("the manifest's model configuration is unusable: {e}"));
                (None, Vec::new())
            }
        };
        let (has_core, has_routers, has_optim) =
            (dir.join(CORE_FILE).is_file(), dir.join(ROUTERS_FILE).is_file(), dir.join(OPTIM_FILE).is_file());
        if !has_core {
            problems.push("core.npz is missing".into());
        }
        if !has_routers {
            problems.push("routers.npz is missing".into());
        }
        let exp_dir = dir.join(EXPERTS_DIR);
        let on_disk: BTreeSet<u64> = fs::read_dir(&exp_dir)
            .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().and_then(parse_expert_file_name)).collect())
            .unwrap_or_default();
        let legacy_flat = m.experts.iter().any(|e| e.file.ends_with(".npy"));
        let listed = m.uids();
        let experts_missing: Vec<u64> = if legacy_flat {
            m.experts.iter().filter(|e| !exp_dir.join(&e.file).is_file()).map(|e| e.id).collect()
        } else {
            listed.iter().copied().filter(|u| !on_disk.contains(u)).collect()
        };
        if !experts_missing.is_empty() {
            problems.push(format!(
                "{} of {} experts have no file (first: e{:05})",
                experts_missing.len(),
                listed.len(),
                experts_missing[0]
            ));
        }
        if let (Some(c), Some(df)) = (&config, m.d_ff)
            && c.pool_d_ff as usize != df
        {
            problems.push(format!("the config says experts are {} wide but the manifest says {df}", c.pool_d_ff));
        }
        let preset = config.as_ref().map_or(Preset::Imported, suggest_preset);
        let summary = config.as_ref().map(|c| {
            format!(
                "{}-wide, {} heads, {}+{} blocks, up to {} rows; {} experts of {} units, {} per character",
                c.d_model,
                c.n_head,
                c.n_prelude,
                c.n_recur + c.n_coda,
                c.max_steps,
                m.n_experts,
                c.pool_d_ff,
                c.pool_top_k
            )
        });
        let origin = match &m.minagi_rs {
            None => "python".to_string(),
            Some(rs) => rs.origin.clone().unwrap_or_else(|| ORIGIN_NATIVE.to_string()),
        };
        Ok(CheckpointInfo {
            dir: dir.to_path_buf(),
            step: m.step,
            val: m.val,
            n_experts: m.n_experts,
            d_model: m.d_model,
            d_ff: m.d_ff,
            paged: m.paged,
            config,
            config_defaulted,
            preset,
            summary,
            resident: m.resident(),
            total_bytes: fsio::dir_size(dir),
            expert_files: on_disk.len(),
            experts_missing,
            has_complete_marker: dir.join(COMPLETE_FILE).is_file(),
            written_by_python: m.written_by_python(),
            origin,
            writer: m.minagi_rs.as_ref().and_then(|r| r.writer.clone()),
            legacy_flat_experts: legacy_flat,
            has_core,
            has_routers,
            has_optim,
            read_chars: m.read_chars,
            read_nats: m.read_nats,
            problems,
        })
    }
}

/// A cheap look at a checkpoint directory; see [`Checkpoint::inspect`].
pub fn inspect(dir: &Path) -> Result<CheckpointInfo> {
    Checkpoint::inspect(dir)
}

/// What [`recover_interrupted`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Recovery {
    /// Half-built checkpoints (`.tmp-<name>`) that were deleted.
    pub removed_incomplete: usize,
    /// Checkpoints that an interrupted overwrite had moved aside (`.old-<name>`) and that were put
    /// back because their replacement never arrived.
    pub restored: usize,
    /// Moved-aside checkpoints that were deleted because their replacement is in place.
    pub removed_old: usize,
}

/// Tidy up after a crash while checkpoints were being written into `parent` (for example a run's
/// `checkpoints/` folder): delete half-built `.tmp-*` directories, restore a `.old-<name>`
/// directory whose replacement never took its place, and delete the ones whose replacement did.
/// Call it once when a run starts; it never touches a directory that is not named like that.
pub fn recover_interrupted(parent: &Path) -> Result<Recovery> {
    let mut r = Recovery::default();
    let rd = match fs::read_dir(parent) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(r),
        Err(e) => return Err(StoreError::io_at(parent, e)),
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if name.starts_with(".tmp-") {
            fs::remove_dir_all(&path).map_err(io(&path))?;
            r.removed_incomplete += 1;
        } else if let Some(original) = name.strip_prefix(".old-") {
            let target = parent.join(original);
            if target.exists() {
                fs::remove_dir_all(&path).map_err(io(&path))?;
                r.removed_old += 1;
            } else {
                fs::rename(&path, &target).map_err(io(&target))?;
                r.restored += 1;
            }
        }
    }
    Ok(r)
}

/// What [`Checkpoint::inspect`] learned without reading any tensor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CheckpointInfo {
    pub dir: PathBuf,
    pub step: Option<i64>,
    /// Validation loss in nats per character, if recorded.
    pub val: Option<f64>,
    pub n_experts: usize,
    pub d_model: Option<usize>,
    /// Hidden width of one expert.
    pub d_ff: Option<usize>,
    pub paged: bool,
    /// The model's shape, if the manifest records it.
    pub config: Option<ModelConfig>,
    /// Config fields the manifest did not record (filled from the Python defaults).
    pub config_defaulted: Vec<String>,
    /// Which preset the shape matches; [`Preset::Imported`] if none.
    pub preset: Preset,
    /// One line describing the shape, for a preview.
    pub summary: Option<String>,
    /// Experts on the card, if recorded.
    pub resident: Option<u32>,
    /// Size of every file under the directory.
    pub total_bytes: u64,
    /// Number of `e*.npz` files in `experts/`.
    pub expert_files: usize,
    /// Uids the manifest lists that have no file.
    pub experts_missing: Vec<u64>,
    /// Whether `COMPLETE` exists. Directories written by Python never have it.
    pub has_complete_marker: bool,
    pub written_by_python: bool,
    /// `python`, `native` or `python-import`.
    pub origin: String,
    pub writer: Option<String>,
    /// Experts are flat `.npy` files (the layout from before moments lived in the expert file).
    pub legacy_flat_experts: bool,
    pub has_core: bool,
    pub has_routers: bool,
    pub has_optim: bool,
    pub read_chars: Option<u64>,
    pub read_nats: Option<f64>,
    /// Cheap structural problems, in plain English (empty = nothing wrong found).
    pub problems: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tiers::ExpertMoments;
    use fsio::scratch_dir;

    /// A small but real config: d_model 8, d_ff (trunk) 16, experts 4 wide.
    fn model() -> ModelConfig {
        ModelConfig {
            d_model: 8,
            n_head: 2,
            d_ff: 16,
            n_prelude: 1,
            n_recur: 1,
            n_coda: 0,
            max_steps: 3,
            min_steps: 1,
            halt_prior: 0.4,
            halt_thresh: 0.9,
            halt_freeze: true,
            ponder_beta: 0.01,
            bptt_window: 3,
            train_steps_mean: 0.0,
            block: 32,
            vocab_size: 265,
            pool_experts: 5,
            pool_max: 16,
            pool_d_ff: 4,
            pool_depth: 1,
            pool_top_k: 2,
        }
    }

    /// A tiny deterministic generator of "random" f32 values with a wide dynamic range.
    fn vals(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let mant = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let exp = ((s >> 8) % 20) as i32 - 12;
                mant * 2f32.powi(exp)
            })
            .collect()
    }

    fn tensors(want: &[(String, Vec<usize>)], seed: u64) -> NamedTensors {
        want.iter()
            .enumerate()
            .map(|(i, (k, shape))| {
                let n = shape.iter().product();
                (k.clone(), Tensor { shape: shape.clone(), data: vals(seed + i as u64, n) })
            })
            .collect()
    }

    fn moments(seed: u64, n: usize) -> ExpertMoments {
        let v: Vec<Vec<f32>> = (0..6).map(|i| vals(seed * 10 + i, n)).collect();
        ExpertMoments::pack([&v[0], &v[1], &v[2], &v[3], &v[4], &v[5]].map(|x| x.as_slice()))
    }

    fn entry(uid: u64, with_moments: bool) -> Arc<ExpertEntry> {
        let n = 4 * 8;
        Arc::new(ExpertEntry {
            w1: vals(uid * 100 + 1, n),
            w3: vals(uid * 100 + 2, n),
            w2: vals(uid * 100 + 3, n),
            moments: with_moments.then(|| moments(uid, n)),
        })
    }

    fn data(uids: &[u64]) -> CheckpointData {
        let m = model();
        let n = uids.len();
        let parsed = ParsedCfg::from_model(m.clone(), 3);
        let mut manifest = Manifest::new(7, Some(1.25), &parsed);
        let mut tel = Telemetry::fresh(uids);
        tel.usage = (0..n).map(|i| i as f64 * 10.0).collect();
        tel.segments = Some(9);
        manifest.telemetry = Some(tel);
        manifest.read_chars = Some(1234);
        manifest.read_nats = Some(99.5);
        manifest.plasticity = Some(serde_json::json!({"lr_mult": 0.5}));
        manifest.context_now = Some(20);
        let core = tensors(&expected_core_tensors(&m), 1);
        let routers = tensors(&expected_router_tensors(&m, n), 500);
        // moments are bf16 in the file, so build the optimiser state from values bf16 can hold
        let bf = |t: &Tensor| Tensor { shape: t.shape.clone(), data: bf16::unpack(&bf16::pack(&t.data)) };
        let mut params = Vec::new();
        for (i, (name, t)) in core.iter().filter(|(k, _)| *k != "head.weight").chain(routers.iter()).enumerate() {
            params.push(ParamState {
                name: name.to_string(),
                m: Some(bf(&Tensor { shape: t.shape.clone(), data: vals(900 + i as u64, t.numel()) })),
                v: Some(bf(&Tensor {
                    shape: t.shape.clone(),
                    data: vals(950 + i as u64, t.numel()).iter().map(|x| x.abs()).collect(),
                })),
                step: Some(40.0 + i as f64),
            });
        }
        params.push(ParamState { name: "pool.w1".into(), m: None, v: None, step: Some(40.0) });
        CheckpointData { manifest, core, routers, optim: OptimState { params } }
    }

    fn entries(uids: &[u64]) -> Vec<(u64, Arc<ExpertEntry>)> {
        uids.iter().map(|&u| (u, entry(u, u % 2 == 0))).collect()
    }

    #[test]
    fn write_then_read_round_trips_everything_exactly() {
        let d = scratch_dir("ckpt-rt");
        let uids = [0u64, 2, 7, 9];
        let dat = data(&uids);
        let ex = entries(&uids);
        let rep = Checkpoint::write(&d.join("step-7"), &dat, ExpertSource::Entries(&ex)).unwrap();
        assert_eq!((rep.n_experts, rep.experts_written, rep.experts_linked), (4, 4, 0));
        let dir = d.join("step-7");
        for f in ["manifest.json", "core.npz", "routers.npz", "optim.npz", "COMPLETE"] {
            assert!(dir.join(f).is_file(), "{f}");
        }
        assert_eq!(fs::read(dir.join("COMPLETE")).unwrap(), b"ok");
        let names: Vec<String> = fs::read_dir(dir.join("experts"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into())
            .collect();
        let mut names = names;
        names.sort();
        assert_eq!(names, ["e00000.npz", "e00002.npz", "e00007.npz", "e00009.npz"]);
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1, "no temporary directory is left beside it");

        let ck = Checkpoint::read(&dir).unwrap();
        assert!(ck.has_complete_marker && ck.warnings.is_empty(), "{:?}", ck.warnings);
        assert_eq!(ck.data.core, dat.core);
        assert_eq!(ck.data.routers, dat.routers);
        // the optimiser state, bit for bit (moments were bf16-representable), 0-d step counts included
        let slot = |p: &ParamState| is_slot_indexed(&p.name);
        let want: Vec<&ParamState> = dat.optim.params.iter().collect();
        assert_eq!(ck.data.optim.params.len(), want.len());
        for (a, b) in ck.data.optim.params.iter().zip(want) {
            assert_eq!(a, b, "{}", b.name);
            assert_eq!(slot(a), slot(b));
        }
        // the manifest was completed from what was written
        let m = &ck.data.manifest;
        assert_eq!(
            (m.step, m.val, m.n_experts, m.d_model, m.d_ff, m.paged),
            (Some(7), Some(1.25), 4, Some(8), Some(4), true)
        );
        assert_eq!(m.uids(), uids);
        assert_eq!(m.listed_uids(), uids);
        assert_eq!(m.telemetry.as_ref().unwrap().next_uid, Some(10));
        assert_eq!(m.telemetry.as_ref().unwrap().segments, Some(9));
        assert_eq!((m.read_chars, m.read_nats, m.context_now), (Some(1234), Some(99.5), Some(20)));
        assert_eq!(m.plasticity, Some(serde_json::json!({"lr_mult": 0.5})));
        assert!(
            m.core_tensors.contains(&"head.weight".to_string()) && m.router_tensors.contains(&GATE_KEY.to_string())
        );
        assert_eq!(m.experts.iter().map(|e| e.moments).collect::<Vec<_>>(), [true, true, false, false]);
        assert!(m.experts.iter().all(|e| e.params == 96 && e.bytes > 0));
        assert!((m.experts[1].gate - f64::from(dat.routers.get(GATE_KEY).unwrap().data[1])).abs() == 0.0);
        let rs = m.minagi_rs.as_ref().unwrap();
        assert_eq!((rs.format, rs.origin.as_deref(), rs.preset.as_deref()), (1, Some("native"), Some("imported")));
        assert!(rs.writer.as_ref().unwrap().starts_with("minagi-core "));
        let cfg = m.cfg.as_ref().unwrap();
        assert_eq!(cfg["pool_resident"], 3);
        assert_eq!(cfg["pool_max"], 16);
        assert_eq!(cfg["d_model"], 8, "Python-readable snake_case cfg");
        assert_eq!(cfg["pool_ever"].as_array().unwrap().len(), 4);
        assert_eq!(cfg["tie_embeddings"], true);
        assert!(!m.written_by_python());
        // every expert comes back exactly
        let mut tiers = ck.open_tiers(2, true).unwrap();
        for (u, e) in &ex {
            assert_eq!(**e, *tiers.fetch(*u).unwrap(), "expert {u}");
        }
        assert!(ck.verify().is_empty(), "{:?}", ck.verify());
        assert!(ck.missing_experts().is_empty());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn slot_moments_are_skipped_by_default_but_their_step_count_is_kept() {
        let d = scratch_dir("ckpt-slots");
        let uids = [0u64, 1];
        let mut dat = data(&uids);
        dat.optim.params.retain(|p| p.name != "pool.w1");
        for name in ["pool.w1", "pool.w3", "pool.w2"] {
            let shape = if name == "pool.w2" { vec![3, 8, 4] } else { vec![3, 4, 8] };
            let n: usize = shape.iter().product();
            let t = |s: u64| Tensor { shape: shape.clone(), data: bf16::unpack(&bf16::pack(&vals(s, n))) };
            dat.optim.params.push(ParamState { name: name.into(), m: Some(t(1)), v: Some(t(2)), step: Some(77.0) });
        }
        let ex = entries(&uids);
        Checkpoint::write(&d.join("c"), &dat, ExpertSource::Entries(&ex)).unwrap();
        let slim = Checkpoint::read(&d.join("c")).unwrap();
        for name in ["pool.w1", "pool.w3", "pool.w2"] {
            let p = slim.data.optim.get(name).unwrap();
            assert!(p.m.is_none() && p.v.is_none());
            assert_eq!(p.step, Some(77.0));
        }
        assert!(slim.data.optim.get("tok_emb.weight").unwrap().m.is_some());
        let full = Checkpoint::read_with(&d.join("c"), &ReadOptions { slot_moments: true, skip_optim: false }).unwrap();
        assert_eq!(full.data.optim.get("pool.w2").unwrap(), dat.optim.get("pool.w2").unwrap());
        let none = Checkpoint::read_with(&d.join("c"), &ReadOptions { slot_moments: false, skip_optim: true }).unwrap();
        assert!(none.data.optim.is_empty());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn live_experts_are_hard_linked_and_later_writes_leave_the_checkpoint_alone() {
        let d = scratch_dir("ckpt-link");
        let uids = [3u64, 4, 5];
        let mut tiers = Tiers::open(TierConfig::new(d.join("live"), 8, 4).ram_capacity(2)).unwrap();
        for (u, e) in entries(&uids) {
            tiers.create(u, e).unwrap();
        }
        // an unflushed change must be in the checkpoint
        let changed = entry(99, true);
        tiers.put(4, changed.clone(), true).unwrap();
        let dat = data(&uids);
        let rep = Checkpoint::write(&d.join("ck"), &dat, ExpertSource::Tiers(&mut tiers)).unwrap();
        assert_eq!(rep.experts_linked + rep.experts_copied, 3);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let a = fs::metadata(d.join("live/e00003.npz")).unwrap();
            let b = fs::metadata(d.join("ck/experts/e00003.npz")).unwrap();
            assert_eq!(a.ino(), b.ino(), "same inode: linked, not copied");
            assert_eq!(rep.experts_linked, 3);
        }
        let ck = Checkpoint::read(&d.join("ck")).unwrap();
        let mut ro = ck.open_tiers(4, true).unwrap();
        assert_eq!(*ro.fetch(4).unwrap(), *changed, "the flush happened before linking");
        // training continues in the live directory: new content replaces the name, not the bytes
        tiers.put(3, entry(77, true), true).unwrap();
        tiers.flush().unwrap();
        let mut ro2 = ck.open_tiers(4, true).unwrap();
        assert_eq!(*ro2.fetch(3).unwrap(), *entry(3, false), "the checkpoint still holds the old expert");
        // starting a run from the checkpoint
        let (linked, copied) = ck.materialize_experts(&d.join("run2")).unwrap();
        assert_eq!(linked + copied, 3);
        let mut run2 = Tiers::open(TierConfig::new(d.join("run2"), 8, 4)).unwrap();
        run2.put(5, entry(55, true), true).unwrap();
        run2.flush().unwrap();
        assert_eq!(*ck.open_tiers(4, true).unwrap().fetch(5).unwrap(), *entry(5, false));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_failed_write_leaves_no_checkpoint_and_no_temporary_directory() {
        let d = scratch_dir("ckpt-fail");
        let uids = [0u64, 1, 2];
        let dat = data(&uids);
        let mut ex = entries(&uids);
        ex.pop(); // expert 2 is missing
        let err = Checkpoint::write(&d.join("ck"), &dat, ExpertSource::Entries(&ex)).unwrap_err();
        assert!(matches!(err, StoreError::ExpertMissing { uid: 2, .. }), "{err:?}");
        assert!(err.to_string().contains("e00002"));
        assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "nothing at all was left behind");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn existing_destination_is_refused_unless_overwrite_and_then_replaced_whole() {
        let d = scratch_dir("ckpt-over");
        let dat = data(&[0, 1]);
        let ex = entries(&[0, 1]);
        let dir = d.join("ck");
        Checkpoint::write(&dir, &dat, ExpertSource::Entries(&ex)).unwrap();
        let err = Checkpoint::write(&dir, &dat, ExpertSource::Entries(&ex)).unwrap_err();
        assert!(matches!(err, StoreError::AlreadyExists(_)));
        let mut dat2 = data(&[0, 1]);
        dat2.manifest.step = Some(8);
        let opts = WriteOptions { overwrite: true, durable: false };
        Checkpoint::write_with(&dir, &dat2, ExpertSource::Entries(&ex), &opts).unwrap();
        assert_eq!(Checkpoint::read(&dir).unwrap().data.manifest.step, Some(8));
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1, "the old copy was cleaned up");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn inconsistent_data_is_refused_before_anything_is_written() {
        let d = scratch_dir("ckpt-bad");
        let ex = entries(&[0, 1, 2]);
        let write = |dat: &CheckpointData| {
            Checkpoint::write(&d.join("ck"), dat, ExpertSource::Entries(&ex)).unwrap_err().to_string()
        };
        // gate of the wrong length
        let mut a = data(&[0, 1, 2]);
        a.routers.insert(GATE_KEY, Tensor::zeros(vec![5]));
        assert!(write(&a).contains("pool.gate") && write(&a).contains("[3]"), "{}", write(&a));
        // missing trunk tensor
        let mut b = data(&[0, 1, 2]);
        b.core.remove("adapter.weight");
        assert!(write(&b).contains("adapter.weight"), "{}", write(&b));
        // wrongly shaped trunk tensor names the tensor and both shapes
        let mut c = data(&[0, 1, 2]);
        c.core.insert("ln_f.weight", Tensor::zeros(vec![9]));
        let msg = write(&c);
        assert!(msg.contains("ln_f.weight") && msg.contains("[9]") && msg.contains("[8]"), "{msg}");
        // duplicate uids
        let mut dup = data(&[0, 1, 2]);
        dup.manifest.telemetry.as_mut().unwrap().uid = vec![0, 1, 1];
        assert!(write(&dup).contains("same uid"));
        // telemetry length
        let mut t = data(&[0, 1, 2]);
        t.manifest.telemetry.as_mut().unwrap().gate.pop();
        assert!(write(&t).contains("telemetry.gate has 2 entries for 3 experts"));
        // no cfg
        let mut nc = data(&[0, 1, 2]);
        nc.manifest.cfg = None;
        assert!(write(&nc).contains("no cfg"));
        assert!(!d.join("ck").exists());
        assert_eq!(fs::read_dir(&d).unwrap().count(), 0);
        // an optimiser entry with only one moment
        let mut half = data(&[0, 1, 2]);
        half.optim.params[0].v = None;
        assert!(write(&half).contains("only one of m and v"));
        assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "even a failure half way cleans up");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn inspect_is_cheap_and_reports_the_marker_and_origin() {
        let d = scratch_dir("ckpt-inspect");
        let uids = [0u64, 2];
        let ex = entries(&uids);
        Checkpoint::write(&d.join("ck"), &data(&uids), ExpertSource::Entries(&ex)).unwrap();
        let info = Checkpoint::inspect(&d.join("ck")).unwrap();
        assert!(info.has_complete_marker && !info.written_by_python);
        assert_eq!(info.origin, "native");
        assert_eq!(
            (info.step, info.val, info.n_experts, info.d_model, info.d_ff),
            (Some(7), Some(1.25), 2, Some(8), Some(4))
        );
        assert_eq!((info.expert_files, info.experts_missing.len()), (2, 0));
        assert!(info.has_core && info.has_routers && info.has_optim);
        assert_eq!(info.preset, Preset::Imported);
        assert!(info.summary.as_ref().unwrap().contains("8-wide, 2 heads"));
        assert!(info.problems.is_empty(), "{:?}", info.problems);
        assert_eq!(info.total_bytes, fsio::dir_size(&d.join("ck")));
        assert_eq!(info.resident, Some(3));
        // a directory without the marker (like every Python-written one) still reads fine
        fs::remove_file(d.join("ck/COMPLETE")).unwrap();
        let info2 = Checkpoint::inspect(&d.join("ck")).unwrap();
        assert!(!info2.has_complete_marker);
        let ck = Checkpoint::read(&d.join("ck")).unwrap();
        assert!(!ck.has_complete_marker);
        assert_eq!(ck.data.core.len(), data(&uids).core.len());
        // deleting an expert shows up as a problem
        fs::remove_file(d.join("ck/experts/e00002.npz")).unwrap();
        let info3 = Checkpoint::inspect(&d.join("ck")).unwrap();
        assert_eq!(info3.experts_missing, [2]);
        assert!(info3.problems.iter().any(|p| p.contains("1 of 2 experts have no file")), "{:?}", info3.problems);
        assert_eq!(Checkpoint::read(&d.join("ck")).unwrap().missing_experts(), [2]);
        assert!(Checkpoint::read(&d.join("ck")).unwrap().verify().iter().any(|p| p.contains("e00002 is missing")));
        // not a checkpoint at all
        assert!(Checkpoint::inspect(&d).unwrap_err().to_string().contains("no manifest.json"));
        assert!(Checkpoint::inspect(&d.join("nope")).is_err());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_non_paged_python_layout_has_its_gate_moved_into_routers() {
        let d = scratch_dir("ckpt-nonpaged");
        let uids = [0u64, 1];
        let ex = entries(&uids);
        let dir = d.join("ck");
        Checkpoint::write(&dir, &data(&uids), ExpertSource::Entries(&ex)).unwrap();
        // rewrite it the way the non-paged Python save does: gate in core.npz, not in routers.npz
        let mut ck = Checkpoint::read(&dir).unwrap();
        let gate = ck.data.routers.remove(GATE_KEY).unwrap();
        ck.data.core.insert(GATE_KEY, gate.clone());
        write_named(&dir.join(CORE_FILE), &ck.data.core, false).unwrap();
        write_named(&dir.join(ROUTERS_FILE), &ck.data.routers, false).unwrap();
        let again = Checkpoint::read(&dir).unwrap();
        assert_eq!(again.data.routers.get(GATE_KEY), Some(&gate));
        assert!(!again.data.core.contains(GATE_KEY));
        assert!(again.warnings.iter().any(|w| w.contains("non-paged")));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn reading_a_bad_directory_gives_plain_messages() {
        let d = scratch_dir("ckpt-readbad");
        let e = Checkpoint::read(&d).unwrap_err().to_string();
        assert!(e.contains("no manifest.json"), "{e}");
        fs::write(d.join("manifest.json"), "{}").unwrap();
        let e = Checkpoint::read(&d).unwrap_err().to_string();
        assert!(e.contains("no core.npz"), "{e}");
        fs::write(d.join("core.npz"), b"junk").unwrap();
        fs::write(d.join("routers.npz"), b"junk").unwrap();
        let e = Checkpoint::read(&d).unwrap_err().to_string();
        assert!(e.contains("core.npz") && e.contains("not a readable .npz"), "{e}");
        assert!(Checkpoint::read(&d.join("missing")).unwrap_err().to_string().contains("not a folder"));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn recover_interrupted_cleans_up_after_a_crash_without_losing_the_only_copy() {
        let d = scratch_dir("ckpt-recover");
        for n in [".tmp-step-5", ".old-step-3", ".old-step-4", "step-4", "step-2"] {
            fs::create_dir(d.join(n)).unwrap();
            fs::write(d.join(n).join("marker"), n).unwrap();
        }
        fs::write(d.join(".tmp-not-a-dir"), b"x").unwrap();
        let r = recover_interrupted(&d).unwrap();
        assert_eq!(r, Recovery { removed_incomplete: 1, restored: 1, removed_old: 1 });
        assert!(!d.join(".tmp-step-5").exists());
        // step-3 had only its moved-aside copy: it is back under its own name, with its content
        assert_eq!(fs::read(d.join("step-3/marker")).unwrap(), b".old-step-3");
        // step-4 was replaced: the old copy is gone, the new one stays
        assert_eq!(fs::read(d.join("step-4/marker")).unwrap(), b"step-4");
        assert!(!d.join(".old-step-4").exists());
        assert!(d.join("step-2").exists() && d.join(".tmp-not-a-dir").exists());
        assert_eq!(recover_interrupted(&d.join("missing")).unwrap(), Recovery::default());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn expected_tensor_lists_match_the_python_state_dict() {
        let m = model();
        let core: Vec<String> = expected_core_tensors(&m).into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            core,
            [
                "tok_emb.weight",
                "prelude.0.ln1.weight",
                "prelude.0.attn.qkv.weight",
                "prelude.0.attn.proj.weight",
                "prelude.0.ln2.weight",
                "prelude.0.mlp.w1.weight",
                "prelude.0.mlp.w3.weight",
                "prelude.0.mlp.w2.weight",
                "recur.0.ln1.weight",
                "recur.0.attn.qkv.weight",
                "recur.0.attn.proj.weight",
                "recur.0.ln2.weight",
                "adapter.weight",
                "ln_f.weight",
                "head.weight",
                "halt.weight",
                "halt.bias"
            ]
        );
        let routers = expected_router_tensors(&m, 5);
        assert_eq!(routers[0], ("recur.0.mlp.depth_emb".to_string(), vec![8]));
        assert_eq!(routers[1], ("recur.0.mlp.router.weight".to_string(), vec![5, 8]));
        assert_eq!(routers[2], ("pool.gate".to_string(), vec![5]));
        // slot names
        assert!(
            is_slot_indexed("pool.w2") && !is_slot_indexed("pool.gate") && !is_slot_indexed("prelude.0.mlp.w2.weight")
        );
    }
}
