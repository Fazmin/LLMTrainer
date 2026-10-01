//! Importing a weights directory written by the Python reference (`github.com/volotat/mini-AGI`),
//! and exporting one back.
//!
//! The Rust checkpoint layout *is* the Python one (see [`super::checkpoint`]), so importing is
//! mostly checking and copying, not converting:
//!
//! 1. [`check_python_checkpoint`] looks at the directory without changing anything and returns a
//!    [`CompatReport`]: the detected model configuration and the preset it matches, every tensor it
//!    mapped, everything it could not map, and a list of [`Issue`]s. Issues that make the model
//!    unrunnable here are [`Severity::Blocker`]s with a plain-English message (for example "this
//!    model has 3 recurrent blocks per row; this engine supports exactly one"). This is what the
//!    app's import preview shows.
//! 2. [`import_python`] runs the same check and, if there is no blocker, produces our layout at
//!    `dst`: the bundles and expert files are copied byte for byte (so Adam moments, bf16 bits
//!    included, are exactly what Python wrote), `COMPLETE` is added, and `manifest.json` gets a
//!    `minagi_rs` block saying where it came from. The result is built beside `dst` and renamed into
//!    place, like every checkpoint.
//!
//! # Layouts understood
//!
//! * **Paged** (what `train.py` writes now): `manifest.paged` is true, `telemetry` carries the uid of
//!   every position, `pool.gate` lives in `routers.npz`, expert files hold bf16 moments.
//! * **Non-paged** (the older `store.save` branch, also what `minagi.create` writes for a fresh
//!   model): no `paged`, no `telemetry`; `pool.gate` is in `core.npz`; expert files may hold fp32
//!   moments or none. Converted: the gate moves to `routers.npz`, telemetry is synthesised
//!   (uid = position), the expert files are copied as they are (fp32 moments are read and rounded to
//!   bf16 when an expert is next written, exactly like the Python loader does).
//! * **Flat experts** (before moments moved into the expert file): `manifest.experts[].file` ends in
//!   `.npy` and holds `w1 | w3 | w2` concatenated. Converted to `eNNNNN.npz`.
//!
//! # What is rejected, and what is merely noted
//!
//! Blockers: no `cfg`; `use_pool` false; `n_recur != 1` or `n_coda != 0`; `halt_freeze` false (unless
//! [`ImportOptions::override_halt_freeze`]: the Python trainer turns it on from `config.yaml` when it
//! loads, so a freshly created directory has it off); stacked blocks inside an expert
//! (`pool_depth != 1`); untied embeddings; a vocabulary other than the 265 bytes+markers; a rotary
//! base other than 10000; trunk or router tensors that are missing or of the wrong shape; expert
//! files that are missing or the wrong size; and whatever [`ModelConfig::validate`] says.
//!
//! Notes: Python-only config the engine has no field for (`pool_capacity_factor`, `pool_aux`,
//! `pool_resident` are reported as hints for the training settings), `pool_experts` replaced by the
//! number of experts actually present, unknown tensors (copied along untouched), slot-indexed
//! optimiser moments (kept in the file, not used), expert files not listed in the manifest (not
//! copied).

use super::checkpoint::{
    COMPLETE_FILE, CORE_FILE, Checkpoint, EXPERTS_DIR, GATE_KEY, NamedTensors, OPTIM_FILE, ROUTERS_FILE, check_shapes,
    expected_core_tensors, expected_router_tensors, is_slot_indexed, read_bundle, write_bundle,
};
use super::manifest::{
    ExpertRecord, MANIFEST_FILE, Manifest, MinagiRs, ORIGIN_PYTHON_IMPORT, ParsedCfg, RS_FORMAT, Telemetry,
    suggest_preset,
};
use super::npz::NpzReader;
use super::tiers::{MOMENT_KEYS, expert_file_name, parse_expert_file_name, write_expert_file};
use super::{NpyData, Result, StoreError, fsio, npy};
use minagi_types::config::{ModelConfig, Preset};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// How seriously to take an [`Issue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// This checkpoint cannot be imported (as it stands).
    Blocker,
    /// It can be imported but something changes or is lost.
    Warning,
    /// For information.
    Note,
}

/// One finding of the compatibility check, in plain English.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Issue {
    pub severity: Severity,
    /// A short stable identifier (`recurrent_blocks`, `halt_freeze`, `expert_missing`, ...) for the
    /// UI to hang an explanation or a button on.
    pub code: &'static str,
    pub message: String,
}

/// Which on-disk layout the Python directory uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonLayout {
    Paged,
    NonPaged,
    FlatExperts,
}

/// A tensor that maps onto this engine's model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MappedTensor {
    /// `core.npz`, `routers.npz` or `optim.npz`.
    pub file: String,
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Something in the directory this engine has no use for.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Unmapped {
    pub file: String,
    pub name: String,
    pub reason: String,
}

/// What the expert files look like.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ExpertScan {
    /// Experts the pool holds (manifest `n_experts`).
    pub expected: usize,
    /// Of those, how many have a readable file of the right shape.
    pub ok: usize,
    /// Experts whose file holds Adam moments (all six arrays).
    pub with_moments: usize,
    /// Of those, how many still hold fp32 moments rather than bf16 bits.
    pub fp32_moments: usize,
    /// Experts with some but not all moment arrays (the rest start at zero).
    pub partial_moments: usize,
    /// Uids with no file.
    pub missing: Vec<u64>,
    /// Uids whose file exists but is unusable, with the reason.
    pub damaged: Vec<(u64, String)>,
    /// Expert files in `experts/` that the pool does not list (not copied).
    pub orphans: Vec<u64>,
    pub bytes: u64,
}

/// What `optim.npz` holds.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct OptimScan {
    pub present: bool,
    /// Parameters with Adam state.
    pub params: usize,
    /// The paged pool's slot-indexed entries (`pool.w1/w3/w2`): kept in the file, not used.
    pub slot_entries: usize,
}

/// Settings of the Python run that have no field in the model configuration but matter when
/// choosing training settings for the imported model.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct TrainHints {
    /// Experts the card held at once (`pool_resident`).
    pub resident: Option<u32>,
    pub capacity_factor: Option<f64>,
    pub pool_aux: Option<f64>,
}

/// The result of looking at a Python-written directory. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompatReport {
    pub source: PathBuf,
    pub layout: PythonLayout,
    pub step: Option<i64>,
    pub val: Option<f64>,
    pub n_experts: usize,
    pub d_model: Option<usize>,
    pub d_ff: Option<usize>,
    /// The model configuration the import would use (with the adjustments listed in `issues`), if
    /// the manifest had a usable `cfg`.
    pub config: Option<ModelConfig>,
    /// Config fields the manifest did not record, filled from the Python defaults.
    pub config_defaulted: Vec<String>,
    /// The preset whose shape matches, else `Imported`.
    pub suggested_preset: Preset,
    pub train_hints: TrainHints,
    pub mapped: Vec<MappedTensor>,
    pub unmapped: Vec<Unmapped>,
    pub experts: ExpertScan,
    pub optim: OptimScan,
    pub issues: Vec<Issue>,
    /// Size of every file in the source.
    pub total_bytes: u64,
}

impl CompatReport {
    /// The findings that stop the import.
    pub fn blockers(&self) -> Vec<&Issue> {
        self.issues.iter().filter(|i| i.severity == Severity::Blocker).collect()
    }

    /// Whether the import can go ahead.
    pub fn is_importable(&self) -> bool {
        self.blockers().is_empty()
    }

    /// The suggested preset's lower-case name (`tiny`, `small`, `full`, `imported`).
    pub fn suggested_preset_name(&self) -> &'static str {
        self.suggested_preset.as_str()
    }

    /// The blockers as one message that can be shown to the user as it is.
    pub fn blocker_message(&self) -> String {
        let b = self.blockers();
        match b.as_slice() {
            [] => "This checkpoint can be imported.".to_string(),
            [one] => format!("This checkpoint cannot be imported: {}", one.message),
            many => {
                let mut s = format!("This checkpoint cannot be imported ({} problems):", many.len());
                for i in many {
                    s.push_str("\n- ");
                    s.push_str(&i.message);
                }
                s
            }
        }
    }

    /// A short plain-English description for a preview.
    pub fn summary(&self) -> String {
        let shape = match &self.config {
            Some(c) => format!(
                "{}-wide, {} heads, {} dense + {} recurrent block(s), up to {} rows, experts of {} units with {} per character",
                c.d_model,
                c.n_head,
                c.n_prelude,
                c.n_recur + c.n_coda,
                c.max_steps,
                c.pool_d_ff,
                c.pool_top_k
            ),
            None => "model shape unknown".to_string(),
        };
        let step = self.step.map_or("an unknown step".to_string(), |s| format!("step {s}"));
        let val = self.val.map_or(String::new(), |v| format!(", validation loss {v:.4}"));
        format!(
            "Python checkpoint at {step}{val}: {} experts, {shape}. Closest preset: {}.",
            self.n_experts,
            self.suggested_preset.display_name()
        )
    }
}

/// Options for [`import_python`].
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Import a checkpoint saved with `halt_freeze` off anyway, recording it as on. The Python
    /// trainer overrides the saved value from `config.yaml` whenever it loads, so a directory fresh
    /// from `minagi.create` always says off.
    pub override_halt_freeze: bool,
    /// Replace `dst` if it exists.
    pub overwrite: bool,
}

fn issue(severity: Severity, code: &'static str, message: impl Into<String>) -> Issue {
    Issue { severity, code, message: message.into() }
}

/// Header information of every array in an npz: `(key, dtype, shape)`.
fn npz_headers(path: &Path) -> Result<Vec<(String, String, Vec<usize>)>> {
    let mut z = NpzReader::open(path).map_err(|e| match e {
        StoreError::Io(io) => StoreError::io_at(path, io),
        other => StoreError::Invalid(format!("{} is not a readable .npz archive ({other})", path.display())),
    })?;
    let mut out = Vec::new();
    for k in z.keys().to_vec() {
        let h = z.header(&k)?;
        out.push((k, h.descr, h.shape));
    }
    Ok(out)
}

/// What one expert file looks like, from headers alone.
struct ExpertLook {
    w1_shape: Vec<usize>,
    w3_shape: Vec<usize>,
    w2_shape: Vec<usize>,
    moments: usize,
    fp32_moments: usize,
    bad_moment_shape: Option<String>,
    size: u64,
}

fn look_at_expert(path: &Path, d_model: usize, d_ff: usize) -> std::result::Result<ExpertLook, String> {
    let size = fs::metadata(path).map_err(|e| e.to_string())?.len();
    let mut z = NpzReader::open(path).map_err(|e| format!("it is not a readable .npz archive ({e})"))?;
    let mut shape_of = |k: &str| -> std::result::Result<Vec<usize>, String> {
        z.header(k).map(|h| h.shape).map_err(|_| format!("it has no array named {k:?}"))
    };
    let (w1_shape, w3_shape, w2_shape) = (shape_of("w1")?, shape_of("w3")?, shape_of("w2")?);
    let (mut moments, mut fp32, mut bad) = (0, 0, None);
    for k in MOMENT_KEYS {
        if let Ok(h) = z.header(k) {
            moments += 1;
            if h.descr == "<f4" {
                fp32 += 1;
            }
            let want = if k.starts_with("w2") { [d_model, d_ff] } else { [d_ff, d_model] };
            if h.shape != want && bad.is_none() {
                bad = Some(format!("its moment array {k:?} has shape {:?}, expected {want:?}", h.shape));
            }
        }
    }
    Ok(ExpertLook { w1_shape, w3_shape, w2_shape, moments, fp32_moments: fp32, bad_moment_shape: bad, size })
}

/// The `.npy` payload of a flat expert, if it parses.
fn read_flat_expert(
    path: &Path,
    d_model: usize,
    d_ff: usize,
) -> std::result::Result<super::tiers::ExpertEntry, String> {
    let f = fs::File::open(path).map_err(|e| e.to_string())?;
    let arr = npy::read_npy(&mut std::io::BufReader::new(f)).map_err(|e| e.to_string())?;
    let n = d_model * d_ff;
    let NpyData::F32(flat) = arr.into_data() else {
        return Err("a flat expert file should hold float32 values".into());
    };
    if flat.len() != 3 * n {
        return Err(format!(
            "a flat expert file should hold 3 x {d_ff} x {d_model} = {} values, found {}",
            3 * n,
            flat.len()
        ));
    }
    Ok(super::tiers::ExpertEntry {
        w1: flat[..n].to_vec(),
        w3: flat[n..2 * n].to_vec(),
        w2: flat[2 * n..].to_vec(),
        moments: None,
    })
}

/// Everything [`check_python_checkpoint`] works out that the importer also needs.
struct Analysis {
    report: CompatReport,
    manifest: Manifest,
    parsed: Option<ParsedCfg>,
    uids: Vec<u64>,
    /// `manifest.experts[].file` by uid (for the flat layout).
    flat_files: Vec<(u64, String)>,
}

fn analyse(src: &Path, opts: &ImportOptions) -> Result<Analysis> {
    if !src.is_dir() {
        return Err(StoreError::Invalid(format!("{} is not a folder", src.display())));
    }
    let manifest_path = src.join(MANIFEST_FILE);
    if !manifest_path.is_file() {
        return Err(StoreError::Invalid(format!(
            "{} has no {MANIFEST_FILE}; it is not a weights folder written by the Python mini-AGI",
            src.display()
        )));
    }
    let man = Manifest::read(&manifest_path)?;
    let mut issues: Vec<Issue> = Vec::new();
    let mut notes_extra: Vec<Issue> = Vec::new();

    let flat = man.experts.iter().any(|e| e.file.ends_with(".npy"));
    let layout = if flat {
        PythonLayout::FlatExperts
    } else if man.paged {
        PythonLayout::Paged
    } else {
        PythonLayout::NonPaged
    };
    match layout {
        PythonLayout::Paged => {}
        PythonLayout::NonPaged => issues.push(issue(
            Severity::Note,
            "layout_non_paged",
            "This folder was saved by the older, non-paged Python save: the expert gates are moved into routers.npz and \
             the per-expert history starts fresh.",
        )),
        PythonLayout::FlatExperts => issues.push(issue(
            Severity::Note,
            "layout_flat_experts",
            "This folder uses the oldest layout (one flat .npy per expert, no optimiser moments in the expert files); \
             the experts are converted to .npz files and start with no Adam history.",
        )),
    }

    // -- configuration ------------------------------------------------------------------------
    let n_experts = man.n_experts.max(man.experts.len());
    let uids = man.uids();
    let mut parsed = None;
    let mut config = None;
    let mut config_defaulted = Vec::new();
    let mut hints = TrainHints::default();
    match man.parsed_cfg() {
        Ok(None) => issues.push(issue(
            Severity::Blocker,
            "no_config",
            "The manifest records no model configuration (cfg), so the shape of the model is unknown.",
        )),
        Err(e) => issues.push(issue(
            Severity::Blocker,
            "bad_config",
            format!("The model configuration in the manifest cannot be read: {e}."),
        )),
        Ok(Some(mut p)) => {
            config_defaulted = p.defaulted.clone();
            if !p.defaulted.is_empty() {
                issues.push(issue(
                    Severity::Warning,
                    "config_defaulted",
                    format!("The manifest does not record {}; Python's defaults were assumed.", p.defaulted.join(", ")),
                ));
            }
            hints.resident = p.resident();
            hints.capacity_factor = p.python_only.get("pool_capacity_factor").and_then(|v| v.as_f64());
            hints.pool_aux = p.python_only.get("pool_aux").and_then(|v| v.as_f64());
            for msg in p.python_only_problems() {
                issues.push(issue(Severity::Blocker, "unsupported_setting", msg));
            }
            let m = &mut p.model;
            if m.n_recur != 1 {
                issues.push(issue(
                    Severity::Blocker,
                    "recurrent_blocks",
                    format!(
                        "This model has {} recurrent blocks per row (n_recur); this engine supports exactly one.",
                        m.n_recur
                    ),
                ));
            }
            if m.n_coda != 0 {
                issues.push(issue(
                    Severity::Blocker,
                    "coda_blocks",
                    format!("This model has {} coda (readout) blocks (n_coda); this engine supports none.", m.n_coda),
                ));
            }
            if !m.halt_freeze {
                if opts.override_halt_freeze {
                    m.halt_freeze = true;
                    issues.push(issue(
                        Severity::Warning,
                        "halt_freeze",
                        "This checkpoint was saved with halt_freeze off; it is recorded as on, because this engine always \
                         stops computing a character once it has halted (the Python trainer does the same from config.yaml).",
                    ));
                } else {
                    issues.push(issue(
                        Severity::Blocker,
                        "halt_freeze",
                        "This checkpoint was saved with halt_freeze off (halted characters keep being computed), but this \
                         engine always freezes them. The Python trainer turns it on from config.yaml when it loads, so this is \
                         normal for a freshly created folder: import it anyway to record it as on.",
                    ));
                }
            }
            if m.pool_depth != 1 {
                issues.push(issue(
                    Severity::Blocker,
                    "expert_depth",
                    format!(
                        "Each expert in this model stacks {} blocks (pool_depth); this engine supports experts of one block.",
                        m.pool_depth
                    ),
                ));
            }
            if m.vocab_size != 265 {
                issues.push(issue(
                    Severity::Blocker,
                    "vocabulary",
                    format!(
                        "This model has a vocabulary of {} tokens; this engine reads text as bytes (256 values plus 9 markers = 265).",
                        m.vocab_size
                    ),
                ));
            }
            // the pool grows and shrinks, but `pool_experts` stays at the size it was created at
            let present = n_experts as u32;
            if present > 0 && m.pool_experts != present {
                issues.push(issue(
                    Severity::Note,
                    "pool_experts",
                    format!(
                        "The manifest says the pool was created with {} experts; it holds {} now, and {} is what is used.",
                        m.pool_experts, present, present
                    ),
                ));
                m.pool_experts = present;
            }
            if m.pool_max < present {
                m.pool_max = present;
            }
            if issues.iter().all(|i| i.severity != Severity::Blocker)
                && let Err(why) = m.validate()
            {
                issues.push(issue(Severity::Blocker, "invalid_config", why));
            }
            if let Some(df) = man.d_ff
                && df != m.pool_d_ff as usize
            {
                issues.push(issue(
                    Severity::Blocker,
                    "expert_width",
                    format!(
                        "The manifest says experts are {df} wide but the model configuration says {}.",
                        m.pool_d_ff
                    ),
                ));
            }
            if let Some(dm) = man.d_model
                && dm != m.d_model as usize
            {
                issues.push(issue(
                    Severity::Blocker,
                    "model_width",
                    format!("The manifest says the model is {dm} wide but the model configuration says {}.", m.d_model),
                ));
            }
            config = Some(p.model.clone());
            parsed = Some(p);
        }
    }
    if n_experts == 0 {
        issues.push(issue(Severity::Blocker, "no_experts", "The manifest lists no experts."));
    }
    if let Some(t) = &man.telemetry {
        for (name, len) in
            [("gate", t.gate.len()), ("use", t.usage.len()), ("uid", t.uid.len()), ("ever", t.ever.len())]
        {
            if len != 0 && len != n_experts {
                issues.push(issue(
                    Severity::Warning,
                    "telemetry_length",
                    format!(
                        "The recorded per-expert history ({name}) covers {len} experts but the pool has {n_experts}."
                    ),
                ));
            }
        }
    }

    // -- tensors -------------------------------------------------------------------------------
    let mut mapped = Vec::new();
    let mut unmapped = Vec::new();
    let d_model = parsed.as_ref().map(|p| p.model.d_model as usize).or(man.d_model);
    let d_ff = parsed.as_ref().map(|p| p.model.pool_d_ff as usize).or(man.d_ff);
    let mut core_have: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut router_have: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut known: BTreeSet<String> = BTreeSet::new();
    let want_core = parsed.as_ref().map(|p| expected_core_tensors(&p.model)).unwrap_or_default();
    let want_routers = parsed.as_ref().map(|p| expected_router_tensors(&p.model, n_experts)).unwrap_or_default();
    known.extend(want_core.iter().map(|w| w.0.clone()));
    known.extend(want_routers.iter().map(|w| w.0.clone()));
    for (file, have) in [(CORE_FILE, &mut core_have), (ROUTERS_FILE, &mut router_have)] {
        let path = src.join(file);
        if !path.is_file() {
            issues.push(issue(
                Severity::Blocker,
                "bundle_missing",
                format!(
                    "{file} is missing, so the {} are not there.",
                    if file == CORE_FILE { "trunk weights" } else { "routers and gates" }
                ),
            ));
            continue;
        }
        match npz_headers(&path) {
            Err(e) => issues.push(issue(Severity::Blocker, "bundle_unreadable", format!("{file} cannot be read: {e}"))),
            Ok(hs) => {
                for (k, descr, shape) in hs {
                    // shapes only: the data is not read here
                    have.insert(k.clone(), shape.clone());
                    if descr != "<f4" {
                        issues.push(issue(
                            Severity::Blocker,
                            "tensor_dtype",
                            format!("{file}: tensor {k:?} has dtype {descr} but trunk tensors must be float32 (<f4)."),
                        ));
                    }
                    if known.contains(&k) || (k == GATE_KEY && file == CORE_FILE) {
                        mapped.push(MappedTensor { file: file.to_string(), name: k, shape, dtype: descr });
                    } else {
                        let reason = if k.starts_with("pool.segment_router.") {
                            "left over from an older architecture (the Python loader skips it too); copied along, not used"
                        } else {
                            "not a tensor of this model; copied along unchanged, not used"
                        };
                        unmapped.push(Unmapped { file: file.to_string(), name: k, reason: reason.into() });
                    }
                }
            }
        }
    }
    // shapes of what the engine expects (the gate may still be in core.npz in a non-paged save)
    if parsed.is_some() {
        let mut routers_plus_gate = router_have.clone();
        if !routers_plus_gate.contains_key(GATE_KEY)
            && let Some(g) = core_have.get(GATE_KEY)
        {
            routers_plus_gate.insert(GATE_KEY.to_string(), g.clone());
        }
        for msg in check_shapes(CORE_FILE, &core_have, &want_core) {
            issues.push(issue(Severity::Blocker, "tensor_mismatch", msg));
        }
        for msg in check_shapes(ROUTERS_FILE, &routers_plus_gate, &want_routers) {
            issues.push(issue(Severity::Blocker, "tensor_mismatch", msg));
        }
    }

    // -- optimiser state -----------------------------------------------------------------------
    let mut optim = OptimScan::default();
    let optim_path = src.join(OPTIM_FILE);
    if optim_path.is_file() {
        optim.present = true;
        match npz_headers(&optim_path) {
            Err(e) => issues.push(issue(
                Severity::Warning,
                "optim_unreadable",
                format!("optim.npz cannot be read ({e}); Adam starts fresh."),
            )),
            Ok(hs) => {
                let mut names = BTreeSet::new();
                for (k, descr, shape) in hs {
                    let Some((name, kind)) = k.rsplit_once('|').filter(|(_, kd)| matches!(*kd, "m" | "v" | "t")) else {
                        unmapped.push(Unmapped {
                            file: OPTIM_FILE.into(),
                            name: k,
                            reason: "not of the form <param>|m, |v or |t; copied along, not used".into(),
                        });
                        continue;
                    };
                    let is_known = known.contains(name) || is_slot_indexed(name) || name == "head.weight";
                    if !is_known {
                        unmapped.push(Unmapped {
                            file: OPTIM_FILE.into(),
                            name: k.clone(),
                            reason: "optimiser state of a parameter this model does not have; copied along, not used"
                                .into(),
                        });
                        continue;
                    }
                    if names.insert(name.to_string()) {
                        optim.params += 1;
                        if is_slot_indexed(name) {
                            optim.slot_entries += 1;
                        }
                    }
                    if kind == "m" {
                        mapped.push(MappedTensor {
                            file: OPTIM_FILE.into(),
                            name: format!("{name}|m, |v, |t"),
                            shape,
                            dtype: descr,
                        });
                    }
                }
                if optim.slot_entries > 0 {
                    notes_extra.push(issue(
                        Severity::Note,
                        "slot_moments",
                        "optim.npz holds moments indexed by card slot (pool.w1, pool.w3, pool.w2). They belong to whichever experts \
                         happened to be on the card, so they are kept in the file but not used: each expert's own moments are in its file.",
                    ));
                }
            }
        }
    } else {
        notes_extra.push(issue(
            Severity::Note,
            "no_optim",
            "There is no optim.npz: Adam starts with no history for the trunk.",
        ));
    }

    // -- experts -------------------------------------------------------------------------------
    let mut scan = ExpertScan { expected: n_experts, ..ExpertScan::default() };
    let exp_dir = src.join(EXPERTS_DIR);
    let on_disk: BTreeSet<u64> = fs::read_dir(&exp_dir)
        .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().and_then(parse_expert_file_name)).collect())
        .unwrap_or_default();
    let mut flat_files = Vec::new();
    if let (Some(dm), Some(df)) = (d_model, d_ff) {
        for &uid in &uids {
            let (file, is_flat) = match man.experts.iter().find(|e| e.id == uid) {
                Some(r) if r.file.ends_with(".npy") => (r.file.clone(), true),
                _ => (expert_file_name(uid), false),
            };
            let path = exp_dir.join(&file);
            if !path.is_file() {
                scan.missing.push(uid);
                continue;
            }
            if is_flat {
                flat_files.push((uid, file));
                match read_flat_expert(&path, dm, df) {
                    Ok(_) => {
                        scan.ok += 1;
                        scan.bytes += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    }
                    Err(why) => scan.damaged.push((uid, why)),
                }
                continue;
            }
            match look_at_expert(&path, dm, df) {
                Err(why) => scan.damaged.push((uid, why)),
                Ok(l) => {
                    let ok = l.w1_shape == [df, dm] && l.w3_shape == [df, dm] && l.w2_shape == [dm, df];
                    if !ok {
                        scan.damaged.push((
                            uid,
                            format!(
                                "w1 {:?}, w3 {:?}, w2 {:?} do not match experts of {df} x {dm}",
                                l.w1_shape, l.w3_shape, l.w2_shape
                            ),
                        ));
                    } else if let Some(why) = l.bad_moment_shape {
                        scan.damaged.push((uid, why));
                    } else {
                        scan.ok += 1;
                        scan.bytes += l.size;
                        match l.moments {
                            0 => {}
                            6 => {
                                scan.with_moments += 1;
                                scan.fp32_moments += usize::from(l.fp32_moments > 0);
                            }
                            _ => scan.partial_moments += 1,
                        }
                    }
                }
            }
        }
        let listed: BTreeSet<u64> = uids.iter().copied().collect();
        scan.orphans = on_disk.iter().copied().filter(|u| !listed.contains(u)).collect();
    }
    if !scan.missing.is_empty() {
        let first = scan.missing[0];
        issues.push(issue(
            Severity::Blocker,
            "expert_missing",
            format!(
                "{} of the {} experts have no file in experts/ (first: {}); a copy of the folder that left experts/ behind cannot be imported.",
                scan.missing.len(),
                n_experts,
                expert_file_name(first)
            ),
        ));
    }
    for (uid, why) in scan.damaged.iter().take(3) {
        issues.push(issue(
            Severity::Blocker,
            "expert_damaged",
            format!("Expert {} cannot be used: {why}.", expert_file_name(*uid)),
        ));
    }
    if scan.damaged.len() > 3 {
        issues.push(issue(
            Severity::Blocker,
            "expert_damaged",
            format!("{} more experts cannot be used.", scan.damaged.len() - 3),
        ));
    }
    if !scan.orphans.is_empty() {
        issues.push(issue(
            Severity::Note,
            "orphan_experts",
            format!(
                "{} expert file(s) in experts/ are not part of the pool (for example {}) and are not copied.",
                scan.orphans.len(),
                expert_file_name(scan.orphans[0])
            ),
        ));
    }
    if scan.fp32_moments > 0 {
        notes_extra.push(issue(
            Severity::Note,
            "fp32_moments",
            format!("{} expert file(s) still hold fp32 Adam moments; they are read as they are and stored as bf16 the next time each expert is written.", scan.fp32_moments),
        ));
    }
    if scan.partial_moments > 0 {
        issues.push(issue(
            Severity::Warning,
            "partial_moments",
            format!(
                "{} expert file(s) have only some of their Adam moments; the missing ones start at zero.",
                scan.partial_moments
            ),
        ));
    }
    issues.extend(notes_extra);
    issues.sort_by_key(|i| i.severity);

    let suggested_preset = config.as_ref().map_or(Preset::Imported, suggest_preset);
    let report = CompatReport {
        source: src.to_path_buf(),
        layout,
        step: man.step,
        val: man.val,
        n_experts,
        d_model: man.d_model,
        d_ff: man.d_ff,
        config,
        config_defaulted,
        suggested_preset,
        train_hints: hints,
        mapped,
        unmapped,
        experts: scan,
        optim,
        issues,
        total_bytes: fsio::dir_size(src),
    };
    Ok(Analysis { report, manifest: man, parsed, uids, flat_files })
}

/// Look at a Python-written weights directory and say what an import would do, without changing
/// anything. Errors are for a folder that is not a weights folder at all (or has an unparseable
/// manifest); everything else is in the returned report.
pub fn check_python_checkpoint(src: &Path) -> Result<CompatReport> {
    check_python_checkpoint_with(src, &ImportOptions::default())
}

/// [`check_python_checkpoint`] under the given import options (so a preview can show what
/// `override_halt_freeze` would do).
pub fn check_python_checkpoint_with(src: &Path, opts: &ImportOptions) -> Result<CompatReport> {
    Ok(analyse(src, opts)?.report)
}

/// Convert the Python weights directory `src` into this engine's layout at `dst`.
///
/// Returns the compatibility report; fails with [`StoreError::Incompatible`] (whose text is the
/// [`CompatReport::blocker_message`]) if there is a blocker, and with [`StoreError::AlreadyExists`]
/// if `dst` exists and `overwrite` is not set. Nothing is left at `dst` unless the whole import
/// succeeded.
pub fn import_python(src: &Path, dst: &Path, opts: &ImportOptions) -> Result<CompatReport> {
    let a = analyse(src, opts)?;
    if !a.report.is_importable() {
        return Err(StoreError::Incompatible(a.report.blocker_message()));
    }
    let Some(parsed) = a.parsed.clone() else {
        return Err(StoreError::Incompatible(a.report.blocker_message()));
    };
    if dst.exists() && !opts.overwrite {
        return Err(StoreError::AlreadyExists(dst.to_path_buf()));
    }
    let name = dst
        .file_name()
        .ok_or_else(|| StoreError::Invalid(format!("{} is not a usable destination", dst.display())))?
        .to_os_string();
    let parent = match dst.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    fs::create_dir_all(&parent).map_err(|e| StoreError::io_at(&parent, e))?;
    let mut tmp_name = std::ffi::OsString::from(".tmp-");
    tmp_name.push(&name);
    let tmp = parent.join(tmp_name);
    if tmp.exists() {
        fs::remove_dir_all(&tmp).map_err(|e| StoreError::io_at(&tmp, e))?;
    }
    fs::create_dir(&tmp).map_err(|e| StoreError::io_at(&tmp, e))?;
    if let Err(e) = build_import(src, &tmp, &a, &parsed) {
        let _ = fs::remove_dir_all(&tmp);
        return Err(e);
    }
    // put it in place (replacing, if allowed, only once the new one is complete)
    let mut aside = None;
    if dst.exists() {
        let mut o = std::ffi::OsString::from(".old-");
        o.push(&name);
        let o = parent.join(o);
        let _ = fs::remove_dir_all(&o);
        fs::rename(dst, &o).map_err(|e| {
            let _ = fs::remove_dir_all(&tmp);
            StoreError::io_at(dst, e)
        })?;
        aside = Some(o);
    }
    if let Err(e) = fsio::rename_replace(&tmp, dst) {
        if let Some(o) = &aside {
            let _ = fs::rename(o, dst);
        }
        let _ = fs::remove_dir_all(&tmp);
        return Err(StoreError::io_at(dst, e));
    }
    fsio::sync_dir(&parent);
    if let Some(o) = aside {
        let _ = fs::remove_dir_all(o);
    }
    Ok(a.report)
}

fn copy_file(src: &Path, dst: &Path) -> Result<u64> {
    fs::copy(src, dst).map_err(|e| StoreError::io_at(src, e))
}

fn build_import(src: &Path, tmp: &Path, a: &Analysis, parsed: &ParsedCfg) -> Result<()> {
    let model = &parsed.model;
    let (d_model, d_ff) = (model.d_model as usize, model.pool_d_ff as usize);
    let n = a.uids.len();
    let exp_dst = tmp.join(EXPERTS_DIR);
    fs::create_dir(&exp_dst).map_err(|e| StoreError::io_at(&exp_dst, e))?;

    // experts: copied byte for byte, or converted from the flat layout
    for &uid in &a.uids {
        let dst = exp_dst.join(expert_file_name(uid));
        if let Some((_, file)) = a.flat_files.iter().find(|(u, _)| *u == uid) {
            let entry = read_flat_expert(&src.join(EXPERTS_DIR).join(file), d_model, d_ff).map_err(|why| {
                StoreError::ExpertCorrupt { uid, path: src.join(EXPERTS_DIR).join(file), reason: why }
            })?;
            write_expert_file(&dst, &entry, d_model, d_ff, false)?;
        } else {
            copy_file(&src.join(EXPERTS_DIR).join(expert_file_name(uid)), &dst)?;
        }
    }

    // bundles: copied; a non-paged folder keeps the gate in core.npz, so those two are rewritten
    let gate_in_core = a.report.layout != PythonLayout::Paged && {
        let h = npz_headers(&src.join(CORE_FILE))?;
        h.iter().any(|(k, _, _)| k == GATE_KEY)
    };
    let routers_has_gate = npz_headers(&src.join(ROUTERS_FILE))?.iter().any(|(k, _, _)| k == GATE_KEY);
    if gate_in_core && !routers_has_gate {
        let ck = read_bundles_for_gate_move(src)?;
        write_bundle(&tmp.join(CORE_FILE), &ck.0)?;
        write_bundle(&tmp.join(ROUTERS_FILE), &ck.1)?;
    } else {
        copy_file(&src.join(CORE_FILE), &tmp.join(CORE_FILE))?;
        copy_file(&src.join(ROUTERS_FILE), &tmp.join(ROUTERS_FILE))?;
    }
    if src.join(OPTIM_FILE).is_file() {
        copy_file(&src.join(OPTIM_FILE), &tmp.join(OPTIM_FILE))?;
    }

    // the manifest, rebuilt around what is now on disk
    let routers = read_bundle(&tmp.join(ROUTERS_FILE))?;
    let core = read_bundle(&tmp.join(CORE_FILE))?;
    let gate: Vec<f32> = routers.get(GATE_KEY).map(|t| t.data.clone()).unwrap_or_default();
    let mut m = a.manifest.clone();
    let mut records = Vec::with_capacity(n);
    for (i, &uid) in a.uids.iter().enumerate() {
        let file = expert_file_name(uid);
        let path = exp_dst.join(&file);
        let info = super::tiers::describe_expert_file(&path)?;
        records.push(ExpertRecord {
            id: uid,
            file,
            bytes: info.size,
            params: (3 * d_model * d_ff) as u64,
            moments: info.has_moments,
            gate: f64::from(gate.get(i).copied().unwrap_or(1.0)),
            extra: Default::default(),
        });
    }
    let tel = m.telemetry.get_or_insert_with(|| {
        let mut t = Telemetry::fresh(&a.uids);
        t.gate = gate.iter().map(|&g| f64::from(g)).collect();
        t
    });
    tel.uid = a.uids.clone();
    tel.next_uid = Some(tel.next_uid.unwrap_or(0).max(a.uids.iter().max().map_or(0, |u| u + 1)));
    if let Some(ever) = a.manifest.cfg.as_ref().and_then(|c| c.get("pool_ever")).and_then(|v| v.as_array())
        && tel.ever.is_empty()
    {
        tel.ever = ever.iter().map(|b| b.as_bool().unwrap_or(false)).collect();
    }
    let mut cfg = parsed.to_cfg();
    cfg.insert("pool_max".into(), serde_json::json!(model.pool_max.max(n as u32)));
    m.cfg = Some(cfg);
    m.n_experts = n;
    m.d_model = Some(d_model);
    m.d_ff = Some(d_ff);
    m.paged = true;
    m.removed_expert_files = None;
    let mut core_names: Vec<String> = core.names().map(str::to_string).collect();
    core_names.sort();
    let mut router_names: Vec<String> = routers.names().map(str::to_string).collect();
    router_names.sort();
    m.core_tensors = core_names;
    m.router_tensors = router_names;
    m.experts = records;
    m.total_bytes = fsio::dir_size(tmp);
    let notes: Vec<String> =
        a.report.issues.iter().filter(|i| i.severity != Severity::Blocker).map(|i| i.message.clone()).collect();
    m.minagi_rs = Some(MinagiRs {
        format: RS_FORMAT,
        writer: Some(format!("minagi-core {}", env!("CARGO_PKG_VERSION"))),
        origin: Some(ORIGIN_PYTHON_IMPORT.into()),
        preset: Some(a.report.suggested_preset.as_str().to_string()),
        imported_from: Some(src.display().to_string()),
        notes,
        extra: Default::default(),
    });
    m.write(&tmp.join(MANIFEST_FILE), false)?;

    // prove the result is a healthy checkpoint before it is allowed to look like one
    let ck = Checkpoint::read(tmp)?;
    let problems = ck.verify();
    if let Some(first) = problems.first() {
        return Err(StoreError::Invalid(format!("the converted checkpoint failed its own check: {first}")));
    }
    fsio::write_atomic(&tmp.join(COMPLETE_FILE), false, |w| {
        use std::io::Write;
        w.write_all(b"ok")?;
        Ok(())
    })?;
    Ok(())
}

fn read_bundles_for_gate_move(src: &Path) -> Result<(NamedTensors, NamedTensors)> {
    let mut core = read_bundle(&src.join(CORE_FILE))?;
    let mut routers = read_bundle(&src.join(ROUTERS_FILE))?;
    if let Some(g) = core.remove(GATE_KEY) {
        routers.insert(GATE_KEY, g);
    }
    Ok((core, routers))
}

/// Copy a checkpoint to `dst` as a folder the Python reference can load (`train.py`, `serve.py`,
/// `store.load`). Our layout already is that layout, so this is a copy without the `COMPLETE`
/// marker (Python neither needs nor expects it); the `minagi_rs` block in the manifest is ignored
/// by Python. `dst` must not exist; the copy is built beside it and renamed into place.
pub fn export_to_python(ckpt_dir: &Path, dst: &Path) -> Result<()> {
    let info = Checkpoint::inspect(ckpt_dir)?;
    if let Some(p) = info.problems.first() {
        return Err(StoreError::Invalid(format!("this checkpoint cannot be exported: {p}")));
    }
    if dst.exists() {
        return Err(StoreError::AlreadyExists(dst.to_path_buf()));
    }
    let name = dst
        .file_name()
        .ok_or_else(|| StoreError::Invalid(format!("{} is not a usable destination", dst.display())))?
        .to_os_string();
    let parent = match dst.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    fs::create_dir_all(&parent).map_err(|e| StoreError::io_at(&parent, e))?;
    let mut tmp_name = std::ffi::OsString::from(".tmp-");
    tmp_name.push(&name);
    let tmp = parent.join(tmp_name);
    let _ = fs::remove_dir_all(&tmp);
    let result = (|| -> Result<()> {
        fs::create_dir(&tmp).map_err(|e| StoreError::io_at(&tmp, e))?;
        fs::create_dir(tmp.join(EXPERTS_DIR)).map_err(|e| StoreError::io_at(&tmp, e))?;
        for f in [CORE_FILE, ROUTERS_FILE, OPTIM_FILE, MANIFEST_FILE] {
            let s = ckpt_dir.join(f);
            if s.is_file() {
                copy_file(&s, &tmp.join(f))?;
            }
        }
        let rd = fs::read_dir(ckpt_dir.join(EXPERTS_DIR)).map_err(|e| StoreError::io_at(ckpt_dir, e))?;
        for e in rd.flatten() {
            let n = e.file_name();
            if n.to_str().and_then(parse_expert_file_name).is_some() {
                copy_file(&e.path(), &tmp.join(EXPERTS_DIR).join(&n))?;
            }
        }
        fsio::rename_replace(&tmp, dst).map_err(|e| StoreError::io_at(dst, e))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&tmp);
    }
    result
}
