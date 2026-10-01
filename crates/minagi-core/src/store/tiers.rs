//! The expert pool on disk, with a RAM cache in front of it.
//!
//! This is the Rust port of `Tiers` in the Python reference's `minagi/paged.py`. Every expert is
//! one file, `experts/e00012.npz`, named by the expert's stable **uid** (not by its position in the
//! pool: pruning does not renumber anything, so a file never moves). The file holds the expert's
//! three weight matrices in full precision and, once the expert has been trained, its own Adam
//! moments:
//!
//! | array              | dtype | shape                | meaning                              |
//! |--------------------|-------|----------------------|--------------------------------------|
//! | `w1`, `w3`         | `<f4` | `[d_ff, d_model]`    | the SwiGLU gate and value projections |
//! | `w2`               | `<f4` | `[d_model, d_ff]`    | the down projection                  |
//! | `w1_m` .. `w2_v`   | `<i2` | same as the weight   | Adam first / second moment, bf16 bits |
//!
//! Weights stay fp32 on disk because every page-out in a narrower format would be a fresh rounding
//! of what the expert has just learned; the moments are stored as bf16 because they need range, not
//! mantissa, and that halves the file (the measurements are in `minagi/precision.py`). A file
//! written before the moments were narrowed holds `<f4` moments; they are read (and rounded to bf16)
//! like the Python reference does.
//!
//! # The three tiers
//!
//! * **Disk** holds every expert.
//! * **RAM** ([`Tiers`]) keeps the most recently used `ram_capacity` experts as whole
//!   [`ExpertEntry`] values. When it is full, the entry untouched longest is dropped, and if it was
//!   changed since it was read it is written back first (*dirty write-back*).
//! * **The accelerator** is the engine's business; it copies what it needs out of an entry.
//!
//! # Sharing entries: `Arc`, not clones
//!
//! An expert is large: at the Full preset it is 3 x 2048 x 512 f32 values = **12.6 MB** of weights
//! plus 12.6 MB of bf16 moments. [`Tiers::fetch`] therefore hands out an `Arc<ExpertEntry>`: a hit
//! costs a reference-count bump, and the caller copies into device memory only what it needs.
//! Entries are never modified in place; replacing an expert's content means building a new entry
//! and [`put`](Tiers::put)ting it, which is exactly the Python behaviour ("an entry replaces the old
//! one whole"). The price of `Arc` is that memory of an evicted entry lives on while a caller still
//! holds it; at most `ram_capacity` + (entries in flight) entries are ever alive, and an in-flight
//! entry is by definition one the engine is using.
//!
//! # Atomic writes
//!
//! Every file is written to `<name>.tmp` next to its destination and renamed into place (with a
//! short retry on Windows, where a rename onto a file someone has open can fail for a moment), so an
//! interrupted write leaves either the old file or the new one, never half of one. Because nothing
//! is ever rewritten in place, a checkpoint can *hard-link* the live expert files instead of copying
//! them (see [`super::checkpoint`]).
//!
//! # Read-only mode
//!
//! A [`Tiers`] opened with `read_only` never writes, creates or deletes anything: `put` keeps the
//! entry in RAM but never marks it dirty, `flush` does nothing, and `create`/`delete` return
//! [`StoreError::ReadOnly`]. This is for tools that look at a directory a training run owns.

use super::npz::{NpzReader, NpzWriter};
use super::{Result, StoreError, bf16, fsio};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The file name of expert `uid`: `e00012.npz` (five digits, more if the uid needs them).
pub fn expert_file_name(uid: u64) -> String {
    format!("e{uid:05}.npz")
}

/// The uid in an expert file name, or `None` if the name is not `e<digits>.npz`.
pub fn parse_expert_file_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix('e')?.strip_suffix(".npz")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Names of the six moment arrays in an expert file, in the order they are written.
pub const MOMENT_KEYS: [&str; 6] = ["w1_m", "w1_v", "w3_m", "w3_v", "w2_m", "w2_v"];

/// An expert's Adam moments as bf16 bit patterns (the `<i2` arrays of the file, kept as they are).
///
/// Keeping the bits rather than widening to `f32` halves what a cached expert costs in RAM and
/// makes a read-then-write of an untouched expert bit-identical by construction. Convert at the
/// boundary with [`bf16::unpack_into`] / [`bf16::pack_into`], or with [`ExpertMoments::pack`] and
/// [`ExpertMoments::unpack`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertMoments {
    pub w1_m: Vec<i16>,
    pub w1_v: Vec<i16>,
    pub w3_m: Vec<i16>,
    pub w3_v: Vec<i16>,
    pub w2_m: Vec<i16>,
    pub w2_v: Vec<i16>,
}

impl ExpertMoments {
    /// All-zero moments for a weight matrix of `n` elements (what a fresh expert starts with).
    pub fn zeros(n: usize) -> Self {
        let z = || vec![0i16; n];
        Self { w1_m: z(), w1_v: z(), w3_m: z(), w3_v: z(), w2_m: z(), w2_v: z() }
    }

    /// Round six `f32` arrays to bf16 (ties to even, like torch). Order: `w1_m, w1_v, w3_m, w3_v,
    /// w2_m, w2_v`, the same as [`MOMENT_KEYS`].
    pub fn pack(f: [&[f32]; 6]) -> Self {
        Self {
            w1_m: bf16::pack(f[0]),
            w1_v: bf16::pack(f[1]),
            w3_m: bf16::pack(f[2]),
            w3_v: bf16::pack(f[3]),
            w2_m: bf16::pack(f[4]),
            w2_v: bf16::pack(f[5]),
        }
    }

    /// Widen to six `f32` arrays, in [`MOMENT_KEYS`] order (exact).
    pub fn unpack(&self) -> [Vec<f32>; 6] {
        let s = self.slices();
        [
            bf16::unpack(s[0]),
            bf16::unpack(s[1]),
            bf16::unpack(s[2]),
            bf16::unpack(s[3]),
            bf16::unpack(s[4]),
            bf16::unpack(s[5]),
        ]
    }

    /// The six arrays with their file names, in [`MOMENT_KEYS`] order.
    pub fn named(&self) -> [(&'static str, &[i16]); 6] {
        let s = self.slices();
        [
            (MOMENT_KEYS[0], s[0]),
            (MOMENT_KEYS[1], s[1]),
            (MOMENT_KEYS[2], s[2]),
            (MOMENT_KEYS[3], s[3]),
            (MOMENT_KEYS[4], s[4]),
            (MOMENT_KEYS[5], s[5]),
        ]
    }

    fn slices(&self) -> [&[i16]; 6] {
        [&self.w1_m, &self.w1_v, &self.w3_m, &self.w3_v, &self.w2_m, &self.w2_v]
    }
}

/// One expert, whole: weights (and optionally the Adam moments that belong to it).
///
/// Matrices are row-major `f32`: `w1` and `w3` are `[d_ff, d_model]`, `w2` is `[d_model, d_ff]`.
/// Their length is always `d_ff * d_model`; [`Tiers`] checks this on every `put`/`create`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertEntry {
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
    pub w2: Vec<f32>,
    /// `None` for an expert that has never been stepped (a freshly grown or created one).
    pub moments: Option<ExpertMoments>,
}

impl ExpertEntry {
    /// An expert of zeros, with zero moments if `with_moments`.
    pub fn zeros(d_model: usize, d_ff: usize, with_moments: bool) -> Self {
        let n = d_model * d_ff;
        Self {
            w1: vec![0.0; n],
            w3: vec![0.0; n],
            w2: vec![0.0; n],
            moments: with_moments.then(|| ExpertMoments::zeros(n)),
        }
    }

    /// Check the lengths against a pool's expert size.
    pub fn check(&self, d_model: usize, d_ff: usize) -> std::result::Result<(), String> {
        let n = d_model * d_ff;
        let mut arrays: Vec<(&str, usize)> = vec![("w1", self.w1.len()), ("w3", self.w3.len()), ("w2", self.w2.len())];
        if let Some(m) = &self.moments {
            for (k, a) in m.named() {
                arrays.push((k, a.len()));
            }
        }
        for (name, len) in arrays {
            if len != n {
                return Err(format!(
                    "{name} has {len} values but an expert here has d_ff x d_model = {d_ff} x {d_model} = {n}"
                ));
            }
        }
        Ok(())
    }

    /// Bytes this entry occupies in RAM (weights and moments).
    pub fn mem_bytes(&self) -> usize {
        (self.w1.len() + self.w3.len() + self.w2.len()) * 4
            + self.moments.as_ref().map_or(0, |m| m.named().iter().map(|(_, a)| a.len() * 2).sum())
    }

    /// Number of weights (3 x d_ff x d_model), moments excluded.
    pub fn params(&self) -> usize {
        self.w1.len() + self.w3.len() + self.w2.len()
    }
}

// ---------------------------------------------------------------------------------------------
// reading and writing one expert file
// ---------------------------------------------------------------------------------------------

/// Read an expert file, returning the entry and the file's size in bytes.
///
/// * A missing file is [`StoreError::ExpertMissing`] (naming the expert and the path).
/// * Anything else wrong (not a zip, no `w1`, wrong shape or dtype, truncated data) is
///   [`StoreError::ExpertCorrupt`] with the reason.
/// * Moments may be bf16 bits (`<i2`, the current format) or `<f4` (an older file; rounded to bf16).
///   If only some of the six are present the rest are zero, which is what the Python reference
///   does when it hands the optimiser an expert's moments.
pub fn read_expert_file_sized(path: &Path, uid: u64, d_model: usize, d_ff: usize) -> Result<(ExpertEntry, u64)> {
    let corrupt = |reason: String| StoreError::ExpertCorrupt { uid, path: path.to_path_buf(), reason };
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(StoreError::ExpertMissing { uid, path: path.to_path_buf() });
        }
        Err(e) => return Err(corrupt(format!("could not be opened ({e})"))),
    };
    let size = file.metadata().map(|m| m.len()).map_err(|e| corrupt(format!("could not be inspected ({e})")))?;
    let mut z = NpzReader::new(BufReader::new(file))
        .map_err(|e| corrupt(format!("is not a readable .npz archive ({e}); the file may be truncated or damaged")))?;
    let n = d_model * d_ff;
    let want = [d_ff, d_model];
    let want_w2 = [d_model, d_ff];

    let mut weight = |key: &str, shape: &[usize; 2]| -> Result<Vec<f32>> {
        if !z.contains(key) {
            return Err(corrupt(format!("has no array named {key:?} (found: {})", z.keys().join(", "))));
        }
        let a = z.get(key).map_err(|e| corrupt(format!("array {key:?} is unreadable ({e})")))?;
        if a.shape() != shape {
            return Err(corrupt(format!(
                "array {key:?} has shape {:?} but this pool's experts need {shape:?} (d_ff {d_ff}, d_model {d_model})",
                a.shape()
            )));
        }
        match a.into_data() {
            super::NpyData::F32(v) => Ok(v),
            other => Err(corrupt(format!("array {key:?} has dtype {} but weights are stored as <f4", other.descr()))),
        }
    };
    let w1 = weight("w1", &want)?;
    let w3 = weight("w3", &want)?;
    let w2 = weight("w2", &want_w2)?;

    let present: Vec<bool> = MOMENT_KEYS.iter().map(|k| z.contains(k)).collect();
    let moments = if present.iter().any(|&p| p) {
        let mut out: Vec<Vec<i16>> = Vec::with_capacity(6);
        for (key, here) in MOMENT_KEYS.iter().zip(&present) {
            if !here {
                out.push(vec![0i16; n]);
                continue;
            }
            let shape = if key.starts_with("w2") { want_w2 } else { want };
            let a = z.get(key).map_err(|e| corrupt(format!("array {key:?} is unreadable ({e})")))?;
            if a.shape() != shape {
                return Err(corrupt(format!("moment array {key:?} has shape {:?}, expected {shape:?}", a.shape())));
            }
            out.push(match a.into_data() {
                super::NpyData::I16(v) => v,
                super::NpyData::F32(v) => bf16::pack(&v),
                other => {
                    return Err(corrupt(format!(
                        "moment array {key:?} has dtype {} but moments are stored as <i2 (bf16 bits) or <f4",
                        other.descr()
                    )));
                }
            });
        }
        let mut it = out.into_iter();
        let mut next = || it.next().unwrap_or_default();
        Some(ExpertMoments { w1_m: next(), w1_v: next(), w3_m: next(), w3_v: next(), w2_m: next(), w2_v: next() })
    } else {
        None
    };
    Ok((ExpertEntry { w1, w3, w2, moments }, size))
}

/// Read an expert file (see [`read_expert_file_sized`]).
pub fn read_expert_file(path: &Path, uid: u64, d_model: usize, d_ff: usize) -> Result<ExpertEntry> {
    read_expert_file_sized(path, uid, d_model, d_ff).map(|(e, _)| e)
}

/// Write an expert file atomically (temporary file, then rename). Returns the file size.
///
/// `durable` additionally forces the bytes to disk before the rename.
pub fn write_expert_file(path: &Path, entry: &ExpertEntry, d_model: usize, d_ff: usize, durable: bool) -> Result<u64> {
    entry.check(d_model, d_ff).map_err(StoreError::Invalid)?;
    fsio::write_atomic(path, durable, |w| {
        let mut z = NpzWriter::new(w);
        z.add_slice("w1", &[d_ff, d_model], &entry.w1)?;
        z.add_slice("w3", &[d_ff, d_model], &entry.w3)?;
        z.add_slice("w2", &[d_model, d_ff], &entry.w2)?;
        if let Some(m) = &entry.moments {
            for (key, bits) in m.named() {
                let shape = if key.starts_with("w2") { [d_model, d_ff] } else { [d_ff, d_model] };
                z.add_slice(key, &shape, bits)?;
            }
        }
        z.finish()?;
        Ok(())
    })
}

/// What is inside an expert file, read from the archive's directory and array headers only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertFileInfo {
    pub size: u64,
    pub has_w1_w3_w2: bool,
    /// All six moment arrays present.
    pub has_moments: bool,
    /// Shape of `w1` as stored, `[d_ff, d_model]`.
    pub w1_shape: Option<Vec<usize>>,
}

/// Describe an expert file cheaply (no tensor is decoded). A file that is not a readable archive
/// is an error; a readable archive with odd contents is reported through the fields.
pub fn describe_expert_file(path: &Path) -> Result<ExpertFileInfo> {
    let file = File::open(path).map_err(|e| StoreError::io_at(path, e))?;
    let size = file.metadata().map_err(|e| StoreError::io_at(path, e))?.len();
    let mut z = NpzReader::new(BufReader::new(file))?;
    let has_w = ["w1", "w3", "w2"].iter().all(|k| z.contains(k));
    let has_moments = MOMENT_KEYS.iter().all(|k| z.contains(k));
    let w1_shape = if z.contains("w1") { z.header("w1").ok().map(|h| h.shape) } else { None };
    Ok(ExpertFileInfo { size, has_w1_w3_w2: has_w, has_moments, w1_shape })
}

// ---------------------------------------------------------------------------------------------
// the tiers
// ---------------------------------------------------------------------------------------------

/// How to open a [`Tiers`].
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// The `experts/` directory.
    pub dir: PathBuf,
    pub d_model: usize,
    pub d_ff: usize,
    /// How many whole experts the RAM tier keeps (Python default 256; the app's `ram_cache`).
    pub ram_capacity: usize,
    /// Never write, create or delete anything.
    pub read_only: bool,
    /// Force every expert write to disk before renaming it into place (slow; off by default).
    pub durable: bool,
}

impl TierConfig {
    pub fn new(dir: impl Into<PathBuf>, d_model: usize, d_ff: usize) -> Self {
        Self { dir: dir.into(), d_model, d_ff, ram_capacity: 256, read_only: false, durable: false }
    }

    pub fn ram_capacity(mut self, n: usize) -> Self {
        self.ram_capacity = n;
        self
    }

    pub fn read_only(mut self, yes: bool) -> Self {
        self.read_only = yes;
        self
    }

    pub fn durable(mut self, yes: bool) -> Self {
        self.durable = yes;
        self
    }
}

/// Counters and sizes of a [`Tiers`], for the "pool" panel and logs. Same names as the Python
/// `Tiers.report()` where they exist.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct TierReport {
    /// Experts currently held in RAM.
    pub ram_held: usize,
    pub ram_capacity: usize,
    /// Of those, how many are changed and not yet on disk.
    pub dirty: usize,
    /// Counted fetches answered from RAM.
    pub hits: u64,
    /// Counted fetches that had to read the disk (Python's `disk_reads`).
    pub misses: u64,
    /// Counted fetches in all: experts brought towards the accelerator (`hits + misses`).
    pub loads: u64,
    pub hit_rate: f64,
    /// Entries dropped from RAM to make room.
    pub evictions: u64,
    /// Expert files written because of an eviction or a flush.
    pub writebacks: u64,
    /// Number of expert files on disk.
    pub disk_files: usize,
    /// Their total size.
    pub disk_bytes: u64,
    /// Total bytes read from / written to expert files by this `Tiers`.
    pub bytes_read: u64,
    pub bytes_written: u64,
}

struct Slot {
    entry: Arc<ExpertEntry>,
    dirty: bool,
    /// Position in the recency order (larger = more recently used).
    tick: u64,
}

/// The RAM tier (an LRU of whole experts) over the `experts/` directory. See the module docs.
///
/// Not internally synchronised: methods take `&mut self`. Share one behind a `Mutex` if several
/// threads need it.
pub struct Tiers {
    cfg: TierConfig,
    ram: HashMap<u64, Slot>,
    /// `tick -> uid`, oldest first.
    order: BTreeMap<u64, u64>,
    clock: u64,
    /// `uid -> file size` of every expert file known to be on disk (scanned at open and kept up to
    /// date by this object; call [`Tiers::rescan`] if something else changes the directory).
    disk: BTreeMap<u64, u64>,
    hits: u64,
    misses: u64,
    evictions: u64,
    writebacks: u64,
    bytes_read: u64,
    bytes_written: u64,
}

impl Tiers {
    /// Open the experts directory. In write mode a missing directory is created; in read-only mode
    /// it must exist.
    pub fn open(cfg: TierConfig) -> Result<Self> {
        if cfg.read_only {
            if !cfg.dir.is_dir() {
                return Err(StoreError::Invalid(format!("the experts folder {} does not exist", cfg.dir.display())));
            }
        } else {
            fs::create_dir_all(&cfg.dir).map_err(|e| StoreError::io_at(&cfg.dir, e))?;
        }
        let mut t = Self {
            cfg,
            ram: HashMap::new(),
            order: BTreeMap::new(),
            clock: 0,
            disk: BTreeMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            writebacks: 0,
            bytes_read: 0,
            bytes_written: 0,
        };
        t.rescan()?;
        Ok(t)
    }

    pub fn dir(&self) -> &Path {
        &self.cfg.dir
    }

    pub fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    pub fn d_ff(&self) -> usize {
        self.cfg.d_ff
    }

    pub fn ram_capacity(&self) -> usize {
        self.cfg.ram_capacity
    }

    pub fn is_read_only(&self) -> bool {
        self.cfg.read_only
    }

    /// Where expert `uid`'s file lives (whether or not it exists).
    pub fn path_of(&self, uid: u64) -> PathBuf {
        self.cfg.dir.join(expert_file_name(uid))
    }

    /// Re-read the directory listing (after something else added or removed expert files).
    pub fn rescan(&mut self) -> Result<()> {
        self.disk.clear();
        let rd = fs::read_dir(&self.cfg.dir).map_err(|e| StoreError::io_at(&self.cfg.dir, e))?;
        for entry in rd {
            let entry = entry.map_err(|e| StoreError::io_at(&self.cfg.dir, e))?;
            let name = entry.file_name();
            let Some(uid) = name.to_str().and_then(parse_expert_file_name) else { continue };
            let meta = entry.metadata().map_err(|e| StoreError::io_at(entry.path(), e))?;
            if meta.is_file() {
                self.disk.insert(uid, meta.len());
            }
        }
        Ok(())
    }

    /// Delete leftover `*.npz.tmp` files from writes that were interrupted. Returns how many.
    /// (Never done automatically: another process might be mid-write.)
    pub fn sweep_temp_files(&mut self) -> Result<usize> {
        if self.cfg.read_only {
            return Err(StoreError::ReadOnly("delete leftover temporary files".into()));
        }
        let mut n = 0;
        let rd = fs::read_dir(&self.cfg.dir).map_err(|e| StoreError::io_at(&self.cfg.dir, e))?;
        for entry in rd.flatten() {
            let name = entry.file_name();
            let Some(stem) = name.to_str().and_then(|s| s.strip_suffix(".tmp")) else { continue };
            if parse_expert_file_name(stem).is_some() && fs::remove_file(entry.path()).is_ok() {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Whether expert `uid` exists: in RAM (it may not have been written yet) or on disk.
    pub fn contains(&self, uid: u64) -> bool {
        self.ram.contains_key(&uid) || self.disk.contains_key(&uid)
    }

    /// Whether expert `uid` has a file on disk.
    pub fn on_disk(&self, uid: u64) -> bool {
        self.disk.contains_key(&uid)
    }

    /// Whether expert `uid` is held in RAM with changes not yet written.
    pub fn is_dirty(&self, uid: u64) -> bool {
        self.ram.get(&uid).is_some_and(|s| s.dirty)
    }

    /// Uids held in RAM, least recently used first (the order they would be evicted in).
    pub fn ram_uids(&self) -> Vec<u64> {
        self.order.values().copied().collect()
    }

    /// Every uid known to exist (RAM or disk), ascending.
    pub fn uids(&self) -> Vec<u64> {
        let mut s: BTreeSet<u64> = self.disk.keys().copied().collect();
        s.extend(self.ram.keys().copied());
        s.into_iter().collect()
    }

    /// Change the RAM capacity; shrinking evicts (writing back dirty experts) right away.
    pub fn set_ram_capacity(&mut self, n: usize) -> Result<()> {
        self.cfg.ram_capacity = n;
        self.trim()
    }

    /// The expert's entry, from RAM if it is there, else read from disk. Counts as a load.
    ///
    /// If making room evicts a dirty expert and its write-back fails, the error is returned; the
    /// evicted expert is not lost (it stays in RAM, still dirty) and the next call retries.
    pub fn fetch(&mut self, uid: u64) -> Result<Arc<ExpertEntry>> {
        self.fetch_with(uid, true)
    }

    /// Like [`fetch`](Tiers::fetch) but not counted, for a fetch that does not put the expert on
    /// the accelerator (its moments, fetched for an optimiser step), so the hit rate stays a fact
    /// about loads.
    pub fn fetch_uncounted(&mut self, uid: u64) -> Result<Arc<ExpertEntry>> {
        self.fetch_with(uid, false)
    }

    fn fetch_with(&mut self, uid: u64, count: bool) -> Result<Arc<ExpertEntry>> {
        if let Some(slot) = self.ram.get(&uid) {
            let entry = Arc::clone(&slot.entry);
            self.hits += u64::from(count);
            self.touch(uid);
            return Ok(entry);
        }
        let (entry, size) = read_expert_file_sized(&self.path_of(uid), uid, self.cfg.d_model, self.cfg.d_ff)?;
        self.misses += u64::from(count);
        self.bytes_read += size;
        self.disk.insert(uid, size);
        let entry = Arc::new(entry);
        self.insert(uid, Arc::clone(&entry), false);
        self.trim()?;
        Ok(entry)
    }

    /// Hand an expert back after it was in use (or store a new content for it).
    ///
    /// `dirty` says the content differs from the file: it will be written when it is evicted or at
    /// [`flush`](Tiers::flush). Once an expert is dirty it stays dirty until written, even if it is
    /// `put` again with `dirty = false` (that cannot lose a change). In read-only mode nothing is
    /// ever marked dirty.
    pub fn put(&mut self, uid: u64, entry: impl Into<Arc<ExpertEntry>>, dirty: bool) -> Result<()> {
        let entry = entry.into();
        entry
            .check(self.cfg.d_model, self.cfg.d_ff)
            .map_err(|m| StoreError::Invalid(format!("cannot store expert e{uid:05}: {m}")))?;
        let dirty = dirty && !self.cfg.read_only;
        let was_dirty = self.ram.get(&uid).is_some_and(|s| s.dirty);
        self.insert(uid, entry, dirty || was_dirty);
        self.trim()
    }

    /// Add a brand-new expert (growth): its file is written immediately and the entry is kept in
    /// RAM, clean. It is an error if an expert with this uid already exists, because uids are never
    /// reused.
    pub fn create(&mut self, uid: u64, entry: impl Into<Arc<ExpertEntry>>) -> Result<()> {
        if self.cfg.read_only {
            return Err(StoreError::ReadOnly(format!("create expert e{uid:05}")));
        }
        let path = self.path_of(uid);
        if self.contains(uid) || path.exists() {
            return Err(StoreError::AlreadyExists(path));
        }
        let entry = entry.into();
        let size = write_expert_file(&path, &entry, self.cfg.d_model, self.cfg.d_ff, self.cfg.durable)?;
        self.disk.insert(uid, size);
        self.bytes_written += size;
        self.insert(uid, entry, false);
        self.trim()
    }

    /// Remove an expert for good (prune): it leaves RAM without being written back and its file is
    /// deleted. Returns whether anything was removed.
    pub fn delete(&mut self, uid: u64) -> Result<bool> {
        if self.cfg.read_only {
            return Err(StoreError::ReadOnly(format!("delete expert e{uid:05}")));
        }
        let mut removed = false;
        if let Some(slot) = self.ram.remove(&uid) {
            self.order.remove(&slot.tick);
            removed = true;
        }
        match fs::remove_file(self.path_of(uid)) {
            Ok(()) => removed = true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(StoreError::io_at(self.path_of(uid), e)),
        }
        removed |= self.disk.remove(&uid).is_some();
        Ok(removed)
    }

    /// Write every dirty expert to disk. Returns how many files were written. If some writes fail
    /// the others are still attempted, the failed ones stay dirty, and the first error is returned.
    /// Does nothing in read-only mode.
    pub fn flush(&mut self) -> Result<usize> {
        if self.cfg.read_only {
            return Ok(0);
        }
        let mut dirty: Vec<u64> = self.ram.iter().filter(|(_, s)| s.dirty).map(|(&u, _)| u).collect();
        dirty.sort_unstable();
        let mut written = 0;
        let mut first_err = None;
        for uid in dirty {
            let Some(entry) = self.ram.get(&uid).map(|s| Arc::clone(&s.entry)) else { continue };
            match self.write_back(uid, &entry) {
                Ok(()) => {
                    if let Some(s) = self.ram.get_mut(&uid) {
                        s.dirty = false;
                    }
                    written += 1;
                }
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        match first_err {
            None => Ok(written),
            Some(e) => Err(e),
        }
    }

    /// Hits, misses, evictions, write-backs and sizes. No i/o.
    pub fn report(&self) -> TierReport {
        let loads = self.hits + self.misses;
        TierReport {
            ram_held: self.ram.len(),
            ram_capacity: self.cfg.ram_capacity,
            dirty: self.ram.values().filter(|s| s.dirty).count(),
            hits: self.hits,
            misses: self.misses,
            loads,
            hit_rate: self.hits as f64 / loads.max(1) as f64,
            evictions: self.evictions,
            writebacks: self.writebacks,
            disk_files: self.disk.len(),
            disk_bytes: self.disk.values().sum(),
            bytes_read: self.bytes_read,
            bytes_written: self.bytes_written,
        }
    }

    // -- internals -----------------------------------------------------------------------

    fn insert(&mut self, uid: u64, entry: Arc<ExpertEntry>, dirty: bool) {
        if let Some(old) = self.ram.remove(&uid) {
            self.order.remove(&old.tick);
        }
        self.clock += 1;
        self.order.insert(self.clock, uid);
        self.ram.insert(uid, Slot { entry, dirty, tick: self.clock });
    }

    fn touch(&mut self, uid: u64) {
        if let Some(slot) = self.ram.get_mut(&uid) {
            self.order.remove(&slot.tick);
            self.clock += 1;
            slot.tick = self.clock;
            self.order.insert(self.clock, uid);
        }
    }

    /// Evict least-recently-used entries until the RAM tier fits its capacity, writing back dirty
    /// ones first. A failed write-back stops the eviction and leaves that expert in RAM.
    fn trim(&mut self) -> Result<()> {
        while self.ram.len() > self.cfg.ram_capacity {
            let Some((&tick, &uid)) = self.order.iter().next() else { break };
            let (entry, dirty) = match self.ram.get(&uid) {
                Some(s) => (Arc::clone(&s.entry), s.dirty),
                None => {
                    self.order.remove(&tick);
                    continue;
                }
            };
            if dirty {
                self.write_back(uid, &entry)?;
            }
            self.order.remove(&tick);
            self.ram.remove(&uid);
            self.evictions += 1;
        }
        Ok(())
    }

    fn write_back(&mut self, uid: u64, entry: &ExpertEntry) -> Result<()> {
        if self.cfg.read_only {
            return Err(StoreError::ReadOnly(format!("write expert e{uid:05}")));
        }
        let size = write_expert_file(&self.path_of(uid), entry, self.cfg.d_model, self.cfg.d_ff, self.cfg.durable)?;
        self.disk.insert(uid, size);
        self.bytes_written += size;
        self.writebacks += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::fsio::scratch_dir;
    use crate::store::npy::NpyArray;
    use crate::store::npz::write_npz_file;

    const DM: usize = 4;
    const DF: usize = 3;

    /// A deterministic entry whose every value depends on `seed`.
    fn entry(seed: u32, with_moments: bool) -> ExpertEntry {
        let n = DM * DF;
        let f = |salt: u32| -> Vec<f32> {
            (0..n).map(|i| (seed * 100 + salt * 10 + i as u32) as f32 * 0.25 - 3.0).collect()
        };
        ExpertEntry {
            w1: f(1),
            w3: f(2),
            w2: f(3),
            moments: with_moments.then(|| {
                let m: Vec<Vec<f32>> = (4..10).map(f).collect();
                ExpertMoments::pack([&m[0], &m[1], &m[2], &m[3], &m[4], &m[5]].map(|v| v.as_slice()))
            }),
        }
    }

    fn open(dir: &Path, cap: usize) -> Tiers {
        Tiers::open(TierConfig::new(dir, DM, DF).ram_capacity(cap)).unwrap()
    }

    fn listing(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> =
            fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }

    #[test]
    fn file_names_roundtrip() {
        assert_eq!(expert_file_name(12), "e00012.npz");
        assert_eq!(expert_file_name(123456), "e123456.npz");
        assert_eq!(parse_expert_file_name("e00012.npz"), Some(12));
        assert_eq!(parse_expert_file_name("e123456.npz"), Some(123456));
        for bad in ["e00012.npy", "00012.npz", "e.npz", "e00x12.npz", "e00012.npz.tmp", "core.npz", ""] {
            assert_eq!(parse_expert_file_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn create_fetch_roundtrip_is_bit_exact_including_moments() {
        let d = scratch_dir("tiers-rt");
        let mut t = open(&d, 8);
        let with = entry(1, true);
        t.create(7, with.clone()).unwrap();
        t.create(8, entry(2, false)).unwrap();
        assert_eq!(listing(&d), ["e00007.npz", "e00008.npz"]);
        // fresh Tiers: everything comes from disk
        let mut t2 = open(&d, 8);
        assert_eq!(t2.uids(), [7, 8]);
        assert_eq!(*t2.fetch(7).unwrap(), with);
        let e8 = t2.fetch(8).unwrap();
        assert_eq!(e8.moments, None);
        assert_eq!(e8.w1, entry(2, false).w1);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn lru_evicts_the_least_recently_used_and_hits_refresh_recency() {
        let d = scratch_dir("tiers-lru");
        let mut t = open(&d, 2);
        for u in [1, 2, 3] {
            t.create(u, entry(u as u32, false)).unwrap();
        }
        // creating 1, 2, 3 with capacity 2 already evicted 1
        assert_eq!(t.ram_uids(), [2, 3]);
        t.fetch(2).unwrap(); // hit: 2 becomes the most recent
        assert_eq!(t.ram_uids(), [3, 2]);
        t.fetch(1).unwrap(); // miss: loads 1, evicts 3 (the least recently used)
        assert_eq!(t.ram_uids(), [2, 1]);
        let r = t.report();
        assert_eq!((r.hits, r.misses, r.loads), (1, 1, 2));
        assert_eq!(r.evictions, 2);
        assert!((r.hit_rate - 0.5).abs() < 1e-12);
        assert_eq!((r.ram_held, r.ram_capacity), (2, 2));
        // an uncounted fetch moves recency but not the statistics
        t.fetch_uncounted(3).unwrap();
        let r2 = t.report();
        assert_eq!((r2.hits, r2.misses), (1, 1));
        assert_eq!(t.ram_uids(), [1, 3]);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn fetch_hands_out_shared_entries_not_copies() {
        let d = scratch_dir("tiers-arc");
        let mut t = open(&d, 4);
        t.create(1, entry(1, true)).unwrap();
        let a = t.fetch(1).unwrap();
        let b = t.fetch(1).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        // an evicted entry stays valid while someone holds it
        t.set_ram_capacity(0).unwrap();
        assert!(t.ram_uids().is_empty());
        assert_eq!(a.w1, entry(1, true).w1);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn dirty_entries_are_written_back_on_eviction_clean_ones_are_not() {
        let d = scratch_dir("tiers-dirty");
        let mut t = open(&d, 1);
        t.create(1, entry(1, false)).unwrap();
        t.create(2, entry(2, false)).unwrap(); // evicts 1 (clean, already on disk)
        assert_eq!(t.report().writebacks, 0);
        let before = fs::read(t.path_of(2)).unwrap();
        // 2 is trained: new weights and moments, marked dirty
        let trained = entry(9, true);
        t.put(2, trained.clone(), true).unwrap();
        assert!(t.is_dirty(2));
        assert_eq!(fs::read(t.path_of(2)).unwrap(), before, "nothing is written until eviction or flush");
        t.fetch(1).unwrap(); // evicts 2 -> write-back
        assert_eq!(t.report().writebacks, 1);
        assert!(!t.is_dirty(1));
        let mut again = open(&d, 4);
        assert_eq!(*again.fetch(2).unwrap(), trained);
        // put(dirty = false) after dirty keeps the pending change
        t.put(1, entry(5, false), true).unwrap();
        t.put(1, entry(6, false), false).unwrap();
        assert!(t.is_dirty(1));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn flush_writes_all_dirty_experts_and_leaves_them_clean() {
        let d = scratch_dir("tiers-flush");
        let mut t = open(&d, 8);
        for u in 0..3u64 {
            t.create(u, entry(u as u32, false)).unwrap();
        }
        t.put(0, entry(10, true), true).unwrap();
        t.put(2, entry(12, true), true).unwrap();
        assert_eq!(t.report().dirty, 2);
        assert_eq!(t.flush().unwrap(), 2);
        assert_eq!(t.report().dirty, 0);
        assert_eq!(t.flush().unwrap(), 0);
        let mut again = open(&d, 8);
        assert_eq!(*again.fetch(0).unwrap(), entry(10, true));
        assert_eq!(*again.fetch(1).unwrap(), entry(1, false));
        assert_eq!(*again.fetch(2).unwrap(), entry(12, true));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn writes_are_atomic_no_temp_files_remain_and_a_failed_write_loses_nothing() {
        let d = scratch_dir("tiers-atomic");
        let mut t = open(&d, 1);
        t.create(1, entry(1, false)).unwrap();
        assert_eq!(listing(&d), ["e00001.npz"], "no .tmp left after a successful write");
        let good = fs::read(t.path_of(1)).unwrap();

        // Make the temporary name unusable (a directory squats on it) so the write-back of expert 1
        // must fail part-way, then check that the old file is intact and the change is not lost.
        t.put(1, entry(7, true), true).unwrap();
        fs::create_dir(d.join("e00001.npz.tmp")).unwrap();
        // capacity is 1, so creating 2 evicts 1 (dirty): its write-back has to fail
        let err = t.create(2, entry(2, false));
        assert!(err.is_err(), "the write-back must fail");
        assert_eq!(fs::read(t.path_of(1)).unwrap(), good, "the old file is untouched");
        assert!(t.ram_uids().contains(&1) && t.is_dirty(1), "the change is still held in RAM, dirty");
        fs::remove_dir(d.join("e00001.npz.tmp")).unwrap();
        // the next operation retries and succeeds
        t.flush().unwrap();
        let mut again = open(&d, 4);
        assert_eq!(*again.fetch(1).unwrap(), entry(7, true));
        assert_eq!(listing(&d).iter().filter(|n| n.ends_with(".tmp")).count(), 0);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn sweep_removes_only_leftover_expert_temp_files() {
        let d = scratch_dir("tiers-sweep");
        let mut t = open(&d, 2);
        t.create(1, entry(1, false)).unwrap();
        fs::write(d.join("e00002.npz.tmp"), b"half").unwrap();
        fs::write(d.join("notes.tmp"), b"mine").unwrap();
        assert_eq!(t.sweep_temp_files().unwrap(), 1);
        assert_eq!(listing(&d), ["e00001.npz", "notes.tmp"]);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn read_only_never_writes_creates_or_deletes() {
        let d = scratch_dir("tiers-ro");
        {
            let mut t = open(&d, 4);
            t.create(1, entry(1, false)).unwrap();
            t.create(2, entry(2, false)).unwrap();
        }
        let snapshot = |d: &Path| -> Vec<(String, Vec<u8>)> {
            listing(d).into_iter().map(|n| (n.clone(), fs::read(d.join(&n)).unwrap())).collect()
        };
        let before = snapshot(&d);
        let mut ro = Tiers::open(TierConfig::new(&d, DM, DF).ram_capacity(1).read_only(true)).unwrap();
        assert!(ro.is_read_only());
        let e = ro.fetch(1).unwrap();
        // "training" it in a read-only pool changes the RAM copy but is never written, even when evicted
        ro.put(1, entry(9, true), true).unwrap();
        assert!(!ro.is_dirty(1));
        ro.fetch(2).unwrap(); // evicts 1
        assert_eq!(ro.report().writebacks, 0);
        assert_eq!(ro.flush().unwrap(), 0);
        assert!(matches!(ro.create(5, entry(5, false)), Err(StoreError::ReadOnly(_))));
        assert!(matches!(ro.delete(1), Err(StoreError::ReadOnly(_))));
        assert!(matches!(ro.sweep_temp_files(), Err(StoreError::ReadOnly(_))));
        assert_eq!(e.w1, entry(1, false).w1);
        assert_eq!(snapshot(&d), before, "the directory is byte for byte what it was");
        // read-only open of a missing folder is an error, not a created directory
        let missing = d.join("nope");
        assert!(Tiers::open(TierConfig::new(&missing, DM, DF).read_only(true)).is_err());
        assert!(!missing.exists());
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn delete_removes_file_ram_and_pending_changes() {
        let d = scratch_dir("tiers-del");
        let mut t = open(&d, 4);
        t.create(1, entry(1, false)).unwrap();
        t.put(1, entry(2, false), true).unwrap();
        assert!(t.delete(1).unwrap());
        assert!(!t.contains(1) && !t.is_dirty(1));
        assert!(listing(&d).is_empty());
        assert_eq!(t.flush().unwrap(), 0, "the deleted expert is not resurrected by a flush");
        assert!(!t.delete(1).unwrap(), "deleting again is not an error, and reports nothing removed");
        // uids are never reused: creating over an existing expert is an error
        t.create(2, entry(2, false)).unwrap();
        assert!(matches!(t.create(2, entry(3, false)), Err(StoreError::AlreadyExists(_))));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn missing_file_error_names_the_expert() {
        let d = scratch_dir("tiers-missing");
        let mut t = open(&d, 4);
        t.create(3, entry(3, false)).unwrap();
        let err = t.fetch(42).unwrap_err();
        assert!(matches!(err, StoreError::ExpertMissing { uid: 42, .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("e00042"), "{msg}");
        assert!(msg.contains("e00042.npz"), "{msg}");
        // and a failed fetch is not counted as a load
        assert_eq!(t.report().misses, 0);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn corrupt_files_are_reported_with_the_expert_and_a_reason() {
        let d = scratch_dir("tiers-corrupt");
        let mut t = open(&d, 4);
        // not an archive at all
        fs::write(t.path_of(7), b"this is not a zip file").unwrap();
        let msg = t.fetch(7).unwrap_err().to_string();
        assert!(msg.contains("e00007") && msg.contains("not a readable .npz"), "{msg}");
        // a truncated real file
        t.create(8, entry(8, true)).unwrap();
        let bytes = fs::read(t.path_of(8)).unwrap();
        fs::write(t.path_of(8), &bytes[..bytes.len() / 2]).unwrap();
        let mut fresh = open(&d, 4);
        let e = fresh.fetch(8).unwrap_err();
        assert!(matches!(e, StoreError::ExpertCorrupt { uid: 8, .. }), "{e:?}");
        // a valid archive of the wrong shape (a pool of a different size) says so
        let wrong = NpyArray::f32(vec![5, 5], vec![0.0; 25]).unwrap();
        write_npz_file(t.path_of(9), [("w1", &wrong), ("w3", &wrong), ("w2", &wrong)].iter().map(|(k, a)| (*k, *a)))
            .unwrap();
        let msg = fresh.fetch(9).unwrap_err().to_string();
        assert!(msg.contains("e00009") && msg.contains("[5, 5]") && msg.contains("[3, 4]"), "{msg}");
        // missing array
        let right = NpyArray::f32(vec![DF, DM], vec![0.0; DF * DM]).unwrap();
        write_npz_file(t.path_of(10), [("w1", &right)].iter().map(|(k, a)| (*k, *a))).unwrap();
        let msg = fresh.fetch(10).unwrap_err().to_string();
        assert!(msg.contains("e00010") && msg.contains("\"w3\""), "{msg}");
        // wrong dtype for weights
        let ints = NpyArray::i16(vec![DF, DM], vec![0; DF * DM]).unwrap();
        let ok2 = NpyArray::f32(vec![DM, DF], vec![0.0; DF * DM]).unwrap();
        write_npz_file(t.path_of(11), [("w1", &ints), ("w3", &ints), ("w2", &ok2)].iter().map(|(k, a)| (*k, *a)))
            .unwrap();
        let msg = fresh.fetch(11).unwrap_err().to_string();
        assert!(msg.contains("e00011") && msg.contains("<i2"), "{msg}");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn older_files_with_f32_moments_and_partial_moments_are_read_like_python() {
        let d = scratch_dir("tiers-legacy");
        let t = open(&d, 4);
        let w = |salt: f32, shape: Vec<usize>| {
            NpyArray::f32(shape.clone(), (0..DM * DF).map(|i| i as f32 * salt).collect()).unwrap()
        };
        let m = NpyArray::f32(vec![DF, DM], (0..DM * DF).map(|i| 1.0 + i as f32 * 1.1e-3).collect()).unwrap();
        let m2 = NpyArray::f32(vec![DM, DF], (0..DM * DF).map(|i| 2.0 + i as f32 * 1.1e-3).collect()).unwrap();
        // fp32 moments for w1 only: the rest are zero, the present ones are rounded to bf16
        write_npz_file(
            t.path_of(1),
            [
                ("w1", &w(1.0, vec![DF, DM])),
                ("w3", &w(2.0, vec![DF, DM])),
                ("w2", &w(3.0, vec![DM, DF])),
                ("w1_m", &m),
                ("w1_v", &m),
                ("w2_m", &m2),
            ]
            .iter()
            .map(|(k, a)| (*k, *a)),
        )
        .unwrap();
        let mut t = open(&d, 4);
        let e = t.fetch(1).unwrap();
        let mo = e.moments.as_ref().expect("moments present");
        assert_eq!(mo.w1_m, bf16::pack(m.as_f32().unwrap()));
        assert_eq!(mo.w1_v, bf16::pack(m.as_f32().unwrap()));
        assert_eq!(mo.w2_m, bf16::pack(m2.as_f32().unwrap()));
        assert!(mo.w3_m.iter().all(|&b| b == 0) && mo.w3_v.iter().all(|&b| b == 0) && mo.w2_v.iter().all(|&b| b == 0));
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn put_checks_sizes_and_describe_reads_headers_only() {
        let d = scratch_dir("tiers-size");
        let mut t = open(&d, 4);
        let mut bad = entry(1, false);
        bad.w2.pop();
        let msg = t.put(1, bad, true).unwrap_err().to_string();
        assert!(msg.contains("e00001") && msg.contains("w2"), "{msg}");
        t.create(2, entry(2, true)).unwrap();
        let info = describe_expert_file(&t.path_of(2)).unwrap();
        assert!(info.has_w1_w3_w2 && info.has_moments);
        assert_eq!(info.w1_shape, Some(vec![DF, DM]));
        assert_eq!(info.size, fs::metadata(t.path_of(2)).unwrap().len());
        assert_eq!(t.report().disk_bytes, info.size);
        assert_eq!(t.report().disk_files, 1);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn mem_bytes_matches_the_documented_full_size() {
        // Full preset: 3 x 2048 x 512 f32 = 12.58 MB of weights, + 12.58 MB of bf16 moments
        let n = 2048 * 512;
        let e = ExpertEntry {
            w1: vec![0.0; n],
            w3: vec![0.0; n],
            w2: vec![0.0; n],
            moments: Some(ExpertMoments::zeros(n)),
        };
        assert_eq!(e.mem_bytes(), 3 * n * 4 + 6 * n * 2);
        assert_eq!(3 * n * 4, 12_582_912);
        assert_eq!(e.params(), 3 * n);
    }
}
