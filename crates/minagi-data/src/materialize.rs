//! Build the dataset directory the engine reads from a [`SplitPlan`].
//!
//! The layout is `dest/train/<lane>/<rel>` and `dest/val/<lane>/<rel>` plus `dest/manifest.json`. Whole files are
//! placed with the cheapest method that works: a copy-on-write clone (**reflink**), else a **hard link**, else a
//! plain **copy**. Symbolic links are never used, so moving or deleting the original folder cannot break a dataset
//! that was prepared from it. Byte ranges (the head and tail of a split file) are always real copies.
//!
//! The work happens in `dest.tmp-<n>` and is renamed into place at the end, so `dest` either does not exist or is
//! complete. A dataset that is already shaped and used in place is not touched at all.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use minagi_types::{JobProgress, SplitConfig, UnixMs};
use serde::Serialize;

use crate::download::Side;
use crate::error::{DataError, DataResult};
use crate::split::{LanePlan, PlannedFile, SplitMethod, SplitPlan};
use crate::util::{CancelFlag, PROGRESS_INTERVAL, RateMeter, Throttle};

/// Name of the manifest written at the top of a prepared dataset.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Version of the manifest format.
pub const MANIFEST_VERSION: u32 = 1;

/// How a file got into the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkMethod {
    /// A copy-on-write clone: a separate file that shares disk blocks until either side is changed.
    Reflink,
    /// A second name for the same file.
    HardLink,
    /// The bytes were copied.
    Copy,
    /// A byte range of a source file was written out as its own file (always a copy).
    Slice,
}

impl LinkMethod {
    /// True when no new disk space was needed.
    pub fn is_free(self) -> bool {
        matches!(self, LinkMethod::Reflink | LinkMethod::HardLink)
    }
}

/// Which placement methods may be tried. Both on by default; turning them off exists for tests and for users who
/// want a fully independent copy.
#[derive(Debug, Clone, Copy)]
pub struct MaterializeOptions {
    pub allow_reflink: bool,
    pub allow_hardlink: bool,
}

impl Default for MaterializeOptions {
    fn default() -> Self {
        Self { allow_reflink: true, allow_hardlink: true }
    }
}

/// Per-lane totals of a materialized dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneReport {
    pub name: String,
    pub train_files: u64,
    pub train_bytes: u64,
    pub val_files: u64,
    pub val_bytes: u64,
}

/// What [`materialize`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializeReport {
    /// The directory holding `train/` and `val/` (the source folder itself for an in-place dataset).
    pub dest: PathBuf,
    /// True when nothing was written because the dataset is used where it is.
    pub in_place: bool,
    pub lanes: Vec<LaneReport>,
    pub files: u64,
    pub train_bytes: u64,
    pub val_bytes: u64,
    pub reflinked_files: u64,
    pub hardlinked_files: u64,
    pub copied_files: u64,
    /// Bytes that share storage with the originals.
    pub bytes_linked: u64,
    /// Bytes that were written to disk.
    pub bytes_copied: u64,
}

impl MaterializeReport {
    /// True when preparing the dataset took no extra disk space (everything was linked, or used in place).
    pub fn uses_no_extra_disk_space(&self) -> bool {
        self.bytes_copied == 0
    }
}

/// [`materialize_with`] with the default options.
pub fn materialize(
    plan: &SplitPlan,
    dest: &Path,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(JobProgress),
) -> DataResult<MaterializeReport> {
    materialize_with(plan, dest, &MaterializeOptions::default(), cancel, progress)
}

/// Build `dest` from `plan`. See the module docs.
///
/// `dest` must not exist (or be an empty directory). Progress counts bytes and is throttled to ten calls a second.
/// On error or cancellation nothing is left behind.
pub fn materialize_with(
    plan: &SplitPlan,
    dest: &Path,
    opts: &MaterializeOptions,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(JobProgress),
) -> DataResult<MaterializeReport> {
    if let Some(root) = &plan.in_place {
        return Ok(report_in_place(plan, root));
    }
    prepare_dest(dest)?;
    let tmp = reserve_tmp_dir(dest)?;
    match build(plan, &tmp, opts, cancel, progress) {
        Ok(mut report) => {
            let finish =
                remove_empty_dest(dest).and_then(|()| fs::rename(&tmp, dest).map_err(|e| DataError::io(dest, e)));
            if let Err(e) = finish {
                let _ = fs::remove_dir_all(&tmp);
                return Err(e);
            }
            report.dest = dest.to_path_buf();
            Ok(report)
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp);
            Err(e)
        }
    }
}

fn report_in_place(plan: &SplitPlan, root: &Path) -> MaterializeReport {
    let mut report = empty_report(root.to_path_buf());
    report.in_place = true;
    for lane in &plan.lanes {
        report.lanes.push(lane_report(lane));
        report.files += (lane.train.len() + lane.val.len()) as u64;
        report.train_bytes += lane.train_bytes();
        report.val_bytes += lane.val_bytes();
    }
    report
}

fn empty_report(dest: PathBuf) -> MaterializeReport {
    MaterializeReport {
        dest,
        in_place: false,
        lanes: Vec::new(),
        files: 0,
        train_bytes: 0,
        val_bytes: 0,
        reflinked_files: 0,
        hardlinked_files: 0,
        copied_files: 0,
        bytes_linked: 0,
        bytes_copied: 0,
    }
}

fn lane_report(lane: &LanePlan) -> LaneReport {
    LaneReport {
        name: lane.name.clone(),
        train_files: lane.train.len() as u64,
        train_bytes: lane.train_bytes(),
        val_files: lane.val.len() as u64,
        val_bytes: lane.val_bytes(),
    }
}

/// `dest` may be absent or an empty directory; anything else would be overwritten, so it is refused.
fn prepare_dest(dest: &Path) -> DataResult<()> {
    match fs::read_dir(dest) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                return Err(DataError::invalid(format!("{} already exists and is not empty", dest.display())));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        // A file in the way (NotADirectory) or an unreadable folder.
        Err(_) if dest.exists() => return Err(DataError::invalid(format!("{} already exists", dest.display()))),
        Err(e) => return Err(DataError::io(dest, e)),
    }
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
    }
    Ok(())
}

fn remove_empty_dest(dest: &Path) -> DataResult<()> {
    match fs::remove_dir(dest) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(DataError::io(dest, e)),
    }
}

/// Create `dest.tmp-<n>` for the first free `n`.
fn reserve_tmp_dir(dest: &Path) -> DataResult<PathBuf> {
    let base = dest.as_os_str().to_os_string();
    for n in 0..10_000u32 {
        let mut name = base.clone();
        name.push(format!(".tmp-{n}"));
        let candidate = PathBuf::from(name);
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(DataError::io(&candidate, e)),
        }
    }
    Err(DataError::invalid(format!("no free temporary folder next to {}", dest.display())))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ManifestFile {
    lane: String,
    side: Side,
    /// Relative to the dataset directory.
    path: String,
    bytes: u64,
    source: String,
    method: LinkMethod,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ManifestLane {
    name: String,
    display_name: String,
    color_slot: u32,
    split_method: SplitMethod,
    train_files: u64,
    train_bytes: u64,
    val_files: u64,
    val_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Manifest<'a> {
    version: u32,
    created_at: UnixMs,
    split: &'a SplitConfig,
    sources: Vec<String>,
    lanes: Vec<ManifestLane>,
    files: Vec<ManifestFile>,
    uses_no_extra_disk_space: bool,
    bytes_linked: u64,
    bytes_copied: u64,
}

/// What can be done per source device: hard links and reflinks only work within one filesystem, and a failure on one
/// file is taken to mean the next ones would fail too.
#[derive(Clone, Copy)]
struct Capabilities {
    reflink: bool,
    hardlink: bool,
}

struct Builder<'a> {
    tmp: &'a Path,
    opts: &'a MaterializeOptions,
    cancel: &'a CancelFlag,
    progress: &'a mut dyn FnMut(JobProgress),
    throttle: Throttle,
    meter: RateMeter,
    caps: HashMap<u64, Capabilities>,
    total_bytes: u64,
    done_bytes: u64,
    report: MaterializeReport,
    manifest_files: Vec<ManifestFile>,
}

fn build(
    plan: &SplitPlan,
    tmp: &Path,
    opts: &MaterializeOptions,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(JobProgress),
) -> DataResult<MaterializeReport> {
    let total_bytes = plan.train_bytes() + plan.val_bytes();
    let mut b = Builder {
        tmp,
        opts,
        cancel,
        progress,
        throttle: Throttle::new(PROGRESS_INTERVAL),
        meter: RateMeter::new(0.0),
        caps: HashMap::new(),
        total_bytes,
        done_bytes: 0,
        report: empty_report(tmp.to_path_buf()),
        manifest_files: Vec::new(),
    };
    for side in [Side::Train, Side::Val] {
        for lane in &plan.lanes {
            let files = if side == Side::Train { &lane.train } else { &lane.val };
            b.place_lane(side, lane, files)?;
        }
    }
    // Lanes with no files at all still get their (empty) directories so the layout is predictable.
    for lane in &plan.lanes {
        for side in [Side::Train, Side::Val] {
            let dir = tmp.join(side.dir()).join(&lane.name);
            fs::create_dir_all(&dir).map_err(|e| DataError::io(&dir, e))?;
        }
        b.report.lanes.push(lane_report(lane));
    }
    b.write_manifest(plan)?;
    b.emit(true, "Dataset ready");
    Ok(b.report)
}

impl Builder<'_> {
    fn place_lane(&mut self, side: Side, lane: &LanePlan, files: &[PlannedFile]) -> DataResult<()> {
        let lane_dir = self.tmp.join(side.dir()).join(&lane.name);
        let mut names = Namer::default();
        for file in files {
            self.cancel.check()?;
            let rel = names.unique(&file.dest_rel);
            let dst = lane_dir.join(&rel);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
            }
            let method = self.place(file, &dst)?;
            self.record(side, lane, file, &rel, method);
            self.done_bytes += file.size;
            let verb = if method.is_free() { "Linking" } else { "Copying" };
            self.emit(false, &format!("{verb} {}/{}", lane.name, rel));
        }
        Ok(())
    }

    fn record(&mut self, side: Side, lane: &LanePlan, file: &PlannedFile, rel: &str, method: LinkMethod) {
        let r = &mut self.report;
        r.files += 1;
        match side {
            Side::Train => r.train_bytes += file.size,
            Side::Val => r.val_bytes += file.size,
        }
        match method {
            LinkMethod::Reflink => r.reflinked_files += 1,
            LinkMethod::HardLink => r.hardlinked_files += 1,
            LinkMethod::Copy | LinkMethod::Slice => r.copied_files += 1,
        }
        if method.is_free() {
            r.bytes_linked += file.size;
        } else {
            r.bytes_copied += file.size;
        }
        self.manifest_files.push(ManifestFile {
            lane: lane.name.clone(),
            side,
            path: format!("{}/{}/{}", side.dir(), lane.name, rel),
            bytes: file.size,
            source: file.source.display().to_string(),
            method,
        });
    }

    /// Put one planned file at `dst` using the cheapest method that works.
    fn place(&mut self, file: &PlannedFile, dst: &Path) -> DataResult<LinkMethod> {
        if let Some((start, end)) = file.range {
            copy_range(&file.source, start, end, dst, self.cancel)?;
            return Ok(LinkMethod::Slice);
        }
        let device = device_id(&file.source);
        let initial = Capabilities { reflink: self.opts.allow_reflink, hardlink: self.opts.allow_hardlink };
        let caps = self.caps.entry(device).or_insert(initial);
        if caps.reflink {
            if reflink_copy::reflink(&file.source, dst).is_ok() {
                return Ok(LinkMethod::Reflink);
            }
            caps.reflink = false;
            let _ = fs::remove_file(dst); // a failed clone may leave an empty file behind
        }
        if caps.hardlink {
            if fs::hard_link(&file.source, dst).is_ok() {
                return Ok(LinkMethod::HardLink);
            }
            caps.hardlink = false;
        }
        fs::copy(&file.source, dst).map_err(|e| write_error(&file.source, dst, file.size, e))?;
        Ok(LinkMethod::Copy)
    }

    fn emit(&mut self, force: bool, message: &str) {
        if !force && !self.throttle.ready() {
            return;
        }
        let done = self.done_bytes as f64;
        let total = self.total_bytes as f64;
        let rate = self.meter.update(done);
        (self.progress)(JobProgress {
            done,
            total: Some(total),
            unit: "bytes".to_string(),
            message: message.to_string(),
            bytes_per_sec: rate,
            eta_seconds: RateMeter::eta(rate, done, Some(total)),
        });
    }

    fn write_manifest(&mut self, plan: &SplitPlan) -> DataResult<()> {
        let report = &self.report;
        let lanes = plan
            .lanes
            .iter()
            .map(|l| ManifestLane {
                name: l.name.clone(),
                display_name: l.display_name.clone(),
                color_slot: l.color_slot,
                split_method: l.method,
                train_files: l.train.len() as u64,
                train_bytes: l.train_bytes(),
                val_files: l.val.len() as u64,
                val_bytes: l.val_bytes(),
            })
            .collect();
        let manifest = Manifest {
            version: MANIFEST_VERSION,
            created_at: UnixMs::now(),
            split: &plan.cfg,
            sources: plan.sources.iter().map(|p| p.display().to_string()).collect(),
            lanes,
            files: std::mem::take(&mut self.manifest_files),
            uses_no_extra_disk_space: report.bytes_copied == 0,
            bytes_linked: report.bytes_linked,
            bytes_copied: report.bytes_copied,
        };
        let path = self.tmp.join(MANIFEST_FILE);
        let file = File::create(&path).map_err(|e| DataError::io(&path, e))?;
        let mut out = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut out, &manifest).map_err(|e| DataError::io(&path, io::Error::other(e)))?;
        out.flush().map_err(|e| DataError::io(&path, e))
    }
}

/// An identifier of the filesystem a file lives on (0 where the platform does not give one cheaply).
fn device_id(path: &Path) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(path).map(|m| m.dev()).unwrap_or(0)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        0
    }
}

/// Turn a failed write into [`DataError::DiskFull`] when the disk is the problem.
fn write_error(src: &Path, dst: &Path, need: u64, e: io::Error) -> DataError {
    if e.kind() == io::ErrorKind::StorageFull {
        let free = dst.parent().and_then(|p| fs4::available_space(p).ok()).unwrap_or(0);
        return DataError::DiskFull { need_bytes: need, free_bytes: free };
    }
    let culprit = if e.kind() == io::ErrorKind::NotFound && !src.exists() { src } else { dst };
    DataError::io(culprit, e)
}

/// Copy bytes `start..end` of `src` into a new file at `dst`, checking for cancellation between chunks.
fn copy_range(src: &Path, start: u64, end: u64, dst: &Path, cancel: &CancelFlag) -> DataResult<()> {
    let need = end.saturating_sub(start);
    let mut input = File::open(src).map_err(|e| DataError::io(src, e))?;
    input.seek(SeekFrom::Start(start)).map_err(|e| DataError::io(src, e))?;
    let mut output = BufWriter::new(File::create(dst).map_err(|e| DataError::io(dst, e))?);
    let mut chunk = vec![0u8; 1 << 20];
    let mut left = need;
    while left > 0 {
        cancel.check()?;
        let want = chunk.len().min(left as usize);
        let n = input.read(&mut chunk[..want]).map_err(|e| DataError::io(src, e))?;
        if n == 0 {
            return Err(DataError::io(
                src,
                io::Error::new(io::ErrorKind::UnexpectedEof, "file is shorter than when it was scanned"),
            ));
        }
        output.write_all(&chunk[..n]).map_err(|e| write_error(src, dst, need, e))?;
        left -= n as u64;
    }
    output.flush().map_err(|e| write_error(src, dst, need, e))
}

/// Gives each file of a lane a destination path that is unique even on a case-insensitive filesystem, and in which
/// no file sits where another needs a folder.
#[derive(Default)]
struct Namer {
    /// Lower-cased paths of files already placed.
    files: HashSet<String>,
    /// Lower-cased folders implied by those files.
    dirs: BTreeSet<String>,
}

impl Namer {
    /// `wanted` if free; otherwise `stem-2.ext`, `stem-3.ext`, ...; and when a folder/file clash makes the path
    /// unusable its `/` are flattened to `__` first.
    fn unique(&mut self, wanted: &str) -> String {
        let mut candidate = sanitize_rel(wanted);
        if self.conflicts_with_layout(&candidate) {
            candidate = candidate.replace('/', "__");
        }
        let base = candidate.clone();
        let mut n = 2;
        while self.files.contains(&candidate.to_lowercase()) || self.dirs.contains(&candidate.to_lowercase()) {
            candidate = with_suffix(&base, n);
            n += 1;
        }
        self.files.insert(candidate.to_lowercase());
        let mut prefix = String::new();
        let parts: Vec<&str> = candidate.split('/').collect();
        for part in &parts[..parts.len() - 1] {
            prefix.push_str(part);
            self.dirs.insert(prefix.to_lowercase());
            prefix.push('/');
        }
        candidate
    }

    /// True when a parent folder of `path` is already a file, or `path` itself is already a folder.
    fn conflicts_with_layout(&self, path: &str) -> bool {
        let lower = path.to_lowercase();
        if self.dirs.contains(&lower) {
            return true;
        }
        let mut prefix = String::new();
        let parts: Vec<&str> = lower.split('/').collect();
        for part in &parts[..parts.len() - 1] {
            prefix.push_str(part);
            if self.files.contains(&prefix) {
                return true;
            }
            prefix.push('/');
        }
        false
    }
}

/// A lane-relative path that cannot escape the lane folder: empty and `.` components are dropped and `..` becomes
/// `__`. Paths from a scan never contain these; plans built by hand might.
fn sanitize_rel(rel: &str) -> String {
    let cleaned: Vec<&str> = rel
        .split(['/', '\\'])
        .filter(|c| !c.is_empty() && *c != ".")
        .map(|c| if c == ".." { "__" } else { c })
        .collect();
    if cleaned.is_empty() { "file".to_string() } else { cleaned.join("/") }
}

/// `dir/name.ext` with `-n` before the extension: `dir/name-2.ext`.
fn with_suffix(path: &str, n: u32) -> String {
    let (dir, file) = match path.rsplit_once('/') {
        Some((d, f)) => (format!("{d}/"), f),
        None => (String::new(), path),
    };
    match file.rfind('.').filter(|&i| i > 0) {
        Some(i) => format!("{dir}{}-{n}{}", &file[..i], &file[i..]),
        None => format!("{dir}{file}-{n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, content: &[u8]) -> PathBuf {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();
        path
    }

    /// A plan with two lanes: `stories` (3 files, one in a subfolder, one held out) and `code` (1 file).
    fn sample_plan(src: &Path) -> SplitPlan {
        let a = write(src, "a.txt", b"alpha alpha alpha\n");
        let b = write(src, "sub/b.txt", b"bravo bravo\n");
        let v = write(src, "v.txt", b"validation text\n");
        let c = write(src, "c.rs", b"fn main() {}\n");
        let file = |p: &Path, rel: &str| PlannedFile {
            dest_rel: rel.to_string(),
            source: p.to_path_buf(),
            range: None,
            size: fs::metadata(p).unwrap().len(),
        };
        let lane = |name: &str, train: Vec<PlannedFile>, val: Vec<PlannedFile>| LanePlan {
            name: name.into(),
            display_name: name.into(),
            color_slot: 0,
            method: crate::split::SplitMethod::HoldOutFiles,
            train,
            val,
        };
        SplitPlan {
            sources: vec![src.to_path_buf()],
            cfg: SplitConfig::default(),
            in_place: None,
            lanes: vec![
                lane("code", vec![file(&c, "c.rs")], vec![]),
                lane("stories", vec![file(&a, "a.txt"), file(&b, "sub/b.txt")], vec![file(&v, "v.txt")]),
            ],
            warnings: vec![],
        }
    }

    fn run(plan: &SplitPlan, dest: &Path, opts: &MaterializeOptions) -> DataResult<MaterializeReport> {
        materialize_with(plan, dest, opts, &CancelFlag::new(), &mut |_| {})
    }

    #[cfg(unix)]
    fn inode(p: &Path) -> (u64, u64) {
        use std::os::unix::fs::MetadataExt;
        let m = fs::metadata(p).unwrap();
        (m.dev(), m.ino())
    }

    fn no_leftovers(dest: &Path) {
        let parent = dest.parent().unwrap();
        let stray: Vec<_> = fs::read_dir(parent)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(stray.is_empty(), "left behind: {stray:?}");
    }

    #[test]
    fn builds_the_engine_layout_and_a_manifest() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        let dest = dir.path().join("out/dataset");
        let report = run(&plan, &dest, &MaterializeOptions::default()).unwrap();

        assert_eq!(fs::read(dest.join("train/stories/a.txt")).unwrap(), b"alpha alpha alpha\n");
        assert_eq!(fs::read(dest.join("train/stories/sub/b.txt")).unwrap(), b"bravo bravo\n");
        assert_eq!(fs::read(dest.join("val/stories/v.txt")).unwrap(), b"validation text\n");
        assert_eq!(fs::read(dest.join("train/code/c.rs")).unwrap(), b"fn main() {}\n");
        assert!(dest.join("val/code").is_dir(), "a lane with no validation text still has its folder");
        no_leftovers(&dest);

        assert_eq!(report.dest, dest);
        assert!(!report.in_place);
        assert_eq!(report.files, 4);
        assert_eq!(report.train_bytes, 18 + 12 + 13);
        assert_eq!(report.val_bytes, 16);
        assert_eq!(report.lanes.len(), 2);
        assert_eq!(
            (report.lanes[1].name.as_str(), report.lanes[1].val_files, report.lanes[1].val_bytes),
            ("stories", 1, 16)
        );

        let manifest: serde_json::Value = serde_json::from_slice(&fs::read(dest.join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(manifest["version"], 1);
        assert_eq!(manifest["split"]["seed"], 42);
        assert_eq!(manifest["split"]["pct"], 2.0);
        assert_eq!(manifest["lanes"].as_array().unwrap().len(), 2);
        assert_eq!(manifest["lanes"][1]["name"], "stories");
        assert_eq!(manifest["lanes"][1]["valBytes"], 16);
        assert_eq!(manifest["lanes"][1]["splitMethod"], "hold_out_files");
        assert_eq!(manifest["sources"].as_array().unwrap().len(), 1);
        assert!(manifest["createdAt"].as_u64().unwrap() > 1_600_000_000_000);
        let files = manifest["files"].as_array().unwrap();
        assert_eq!(files.len(), 4);
        assert!(files.iter().any(|f| f["path"] == "val/stories/v.txt" && f["side"] == "val" && f["bytes"] == 16));
        assert_eq!(manifest["bytesCopied"], report.bytes_copied);
    }

    #[test]
    fn links_are_preferred_and_cost_no_extra_space() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        let dest = dir.path().join("ds");
        let report = run(&plan, &dest, &MaterializeOptions::default()).unwrap();
        assert_eq!(report.copied_files, 0, "{report:?}");
        assert_eq!(report.reflinked_files + report.hardlinked_files, 4);
        assert_eq!(report.bytes_copied, 0);
        assert_eq!(report.bytes_linked, report.train_bytes + report.val_bytes);
        assert!(report.uses_no_extra_disk_space());
    }

    #[cfg(unix)]
    #[test]
    fn hard_links_share_the_inode_when_reflinks_are_off() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        let plan = sample_plan(&src);
        let dest = dir.path().join("ds");
        let report = run(&plan, &dest, &MaterializeOptions { allow_reflink: false, allow_hardlink: true }).unwrap();
        assert_eq!((report.hardlinked_files, report.reflinked_files, report.copied_files), (4, 0, 0));
        assert_eq!(inode(&dest.join("train/stories/a.txt")), inode(&src.join("a.txt")));
    }

    #[cfg(unix)]
    #[test]
    fn plain_copies_when_links_are_off_are_independent_files() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        let plan = sample_plan(&src);
        let dest = dir.path().join("ds");
        let report = run(&plan, &dest, &MaterializeOptions { allow_reflink: false, allow_hardlink: false }).unwrap();
        assert_eq!((report.copied_files, report.hardlinked_files, report.reflinked_files), (4, 0, 0));
        assert_eq!(report.bytes_copied, 18 + 12 + 13 + 16);
        assert_eq!(report.bytes_linked, 0);
        assert!(!report.uses_no_extra_disk_space());
        assert_ne!(inode(&dest.join("train/stories/a.txt")), inode(&src.join("a.txt")));
        assert_eq!(fs::read(dest.join("train/stories/a.txt")).unwrap(), fs::read(src.join("a.txt")).unwrap());
    }

    #[test]
    fn symlinks_are_never_created() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        for (reflink, hardlink) in [(true, true), (false, true), (false, false)] {
            let dest = dir.path().join(format!("ds-{reflink}-{hardlink}"));
            run(&plan, &dest, &MaterializeOptions { allow_reflink: reflink, allow_hardlink: hardlink }).unwrap();
            for rel in ["train/stories/a.txt", "train/stories/sub/b.txt", "val/stories/v.txt", "train/code/c.rs"] {
                let meta = fs::symlink_metadata(dest.join(rel)).unwrap();
                assert!(meta.file_type().is_file() && !meta.file_type().is_symlink(), "{rel}");
            }
        }
    }

    #[test]
    fn the_prepared_dataset_survives_deleting_the_originals() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        let plan = sample_plan(&src);
        let dest = dir.path().join("ds");
        run(&plan, &dest, &MaterializeOptions::default()).unwrap();
        fs::remove_dir_all(&src).unwrap();
        assert_eq!(fs::read(dest.join("train/stories/a.txt")).unwrap(), b"alpha alpha alpha\n");
    }

    #[test]
    fn byte_ranges_are_written_as_managed_copies() {
        const TEXT: &str = "line one\nline two\nline three\nline four\n";
        let dir = TempDir::new().unwrap();
        let source = write(dir.path(), "src/book.txt", TEXT.as_bytes());
        let cut = TEXT.find("line four").unwrap() as u64;
        let len = TEXT.len() as u64;
        let range = |name: &str, r: (u64, u64)| PlannedFile {
            dest_rel: name.into(),
            source: source.clone(),
            range: Some(r),
            size: r.1 - r.0,
        };
        let plan = SplitPlan {
            sources: vec![],
            cfg: SplitConfig::default(),
            in_place: None,
            lanes: vec![LanePlan {
                name: "book".into(),
                display_name: "book".into(),
                color_slot: 0,
                method: crate::split::SplitMethod::TailSlice,
                train: vec![range("book.head.txt", (0, cut))],
                val: vec![range("book.tail.txt", (cut, len))],
            }],
            warnings: vec![],
        };

        let dest = dir.path().join("ds");
        let report = run(&plan, &dest, &MaterializeOptions::default()).unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("train/book/book.head.txt")).unwrap(),
            "line one\nline two\nline three\n"
        );
        assert_eq!(fs::read_to_string(dest.join("val/book/book.tail.txt")).unwrap(), "line four\n");
        assert_eq!(report.bytes_copied, len, "slices are real copies");
        assert_eq!(report.copied_files, 2);
        assert!(!report.uses_no_extra_disk_space());
    }

    #[test]
    fn a_source_that_shrank_is_reported_not_silently_truncated() {
        let dir = TempDir::new().unwrap();
        let source = write(dir.path(), "src/short.txt", b"tiny");
        let plan = SplitPlan {
            sources: vec![],
            cfg: SplitConfig::default(),
            in_place: None,
            lanes: vec![LanePlan {
                name: "l".into(),
                display_name: "l".into(),
                color_slot: 0,
                method: crate::split::SplitMethod::TailSlice,
                train: vec![PlannedFile {
                    dest_rel: "short.head.txt".into(),
                    source,
                    range: Some((0, 100)),
                    size: 100,
                }],
                val: vec![],
            }],
            warnings: vec![],
        };
        let dest = dir.path().join("ds");
        assert!(matches!(run(&plan, &dest, &MaterializeOptions::default()), Err(DataError::Io { .. })));
        assert!(!dest.exists());
    }

    #[test]
    fn colliding_names_are_made_unique_instead_of_overwriting() {
        let dir = TempDir::new().unwrap();
        let one = write(dir.path(), "x/notes.txt", b"first");
        let two = write(dir.path(), "y/Notes.txt", b"second");
        let three = write(dir.path(), "z/notes.txt", b"third");
        let file = |p: &Path, rel: &str| PlannedFile {
            dest_rel: rel.into(),
            source: p.to_path_buf(),
            range: None,
            size: fs::metadata(p).unwrap().len(),
        };
        let mut plan = sample_plan(&dir.path().join("src"));
        plan.lanes = vec![LanePlan {
            name: "my-files".into(),
            display_name: "My files".into(),
            color_slot: 0,
            method: crate::split::SplitMethod::HoldOutFiles,
            train: vec![file(&one, "notes.txt"), file(&two, "Notes.txt"), file(&three, "notes.txt")],
            val: vec![],
        }];
        let dest = dir.path().join("ds");
        run(&plan, &dest, &MaterializeOptions::default()).unwrap();
        let mut got: Vec<(String, String)> = fs::read_dir(dest.join("train/my-files"))
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                (e.file_name().to_string_lossy().into_owned(), fs::read_to_string(e.path()).unwrap())
            })
            .collect();
        got.sort();
        assert_eq!(got.len(), 3, "{got:?}");
        let contents: Vec<&str> = got.iter().map(|(_, c)| c.as_str()).collect();
        assert!(contents.contains(&"first") && contents.contains(&"second") && contents.contains(&"third"));
    }

    #[test]
    fn namer_handles_case_folding_suffixes_and_folder_file_clashes() {
        let mut n = Namer::default();
        assert_eq!(n.unique("a/b.txt"), "a/b.txt");
        assert_eq!(n.unique("A/B.txt"), "A/B-2.txt");
        assert_eq!(n.unique("a/b.txt"), "a/b-3.txt");
        assert_eq!(n.unique("noext"), "noext");
        assert_eq!(n.unique("NOEXT"), "NOEXT-2");
        assert_eq!(n.unique(".hidden"), ".hidden");
        assert_eq!(n.unique(".hidden"), ".hidden-2");
        // A file named like an existing folder, and a path through an existing file: flattened.
        assert_eq!(n.unique("a"), "a-2", "a is a folder already");
        assert_eq!(n.unique("noext/inner.txt"), "noext__inner.txt", "noext is a file already");
        // Flattened names are unique too.
        assert_eq!(n.unique("noext/inner.txt"), "noext__inner-2.txt");
    }

    #[test]
    fn relative_paths_cannot_escape_the_lane_folder() {
        assert_eq!(sanitize_rel("a/b.txt"), "a/b.txt");
        assert_eq!(sanitize_rel("../../etc/passwd"), "__/__/etc/passwd");
        assert_eq!(sanitize_rel("/abs/path.txt"), "abs/path.txt");
        assert_eq!(sanitize_rel("./x//y.txt"), "x/y.txt");
        assert_eq!(sanitize_rel("a\\..\\b"), "a/__/b");
        assert_eq!(sanitize_rel(""), "file");
        assert_eq!(sanitize_rel("."), "file");
        let mut n = Namer::default();
        assert_eq!(n.unique("../x.txt"), "__/x.txt");
    }

    #[test]
    fn with_suffix_keeps_directories_and_extensions() {
        assert_eq!(with_suffix("d/e/f.tar.gz", 2), "d/e/f.tar-2.gz");
        assert_eq!(with_suffix("f", 3), "f-3");
        assert_eq!(with_suffix(".bashrc", 2), ".bashrc-2");
    }

    #[test]
    fn in_place_datasets_are_not_touched() {
        let dir = TempDir::new().unwrap();
        let mut plan = sample_plan(&dir.path().join("src"));
        plan.in_place = Some(dir.path().join("src"));
        let dest = dir.path().join("never-created");
        let report = run(&plan, &dest, &MaterializeOptions::default()).unwrap();
        assert!(report.in_place);
        assert_eq!(report.dest, dir.path().join("src"));
        assert!(!dest.exists());
        assert!(report.uses_no_extra_disk_space());
        assert_eq!(report.files, 4);
        no_leftovers(&dest);
    }

    #[test]
    fn refuses_a_non_empty_destination_but_accepts_an_empty_one() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        let busy = dir.path().join("busy");
        write(&busy, "keep.txt", b"mine");
        assert!(matches!(run(&plan, &busy, &MaterializeOptions::default()), Err(DataError::Invalid(_))));
        assert_eq!(fs::read(busy.join("keep.txt")).unwrap(), b"mine");
        no_leftovers(&busy);

        let empty = dir.path().join("empty");
        fs::create_dir(&empty).unwrap();
        run(&plan, &empty, &MaterializeOptions::default()).unwrap();
        assert!(empty.join("train/stories/a.txt").exists());

        let file = write(dir.path(), "afile", b"x");
        assert!(matches!(run(&plan, &file, &MaterializeOptions::default()), Err(DataError::Invalid(_))));
    }

    #[test]
    fn a_failure_leaves_nothing_behind() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        let plan = sample_plan(&src);
        fs::remove_file(src.join("c.rs")).unwrap(); // vanishes between planning and building
        let dest = dir.path().join("ds");
        let err = run(&plan, &dest, &MaterializeOptions::default()).unwrap_err();
        assert!(matches!(err, DataError::Io { .. }), "{err:?}");
        assert!(!dest.exists());
        no_leftovers(&dest);
    }

    #[test]
    fn cancellation_stops_and_cleans_up() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        let dest = dir.path().join("ds");
        let cancel = CancelFlag::new();
        let calls = AtomicUsize::new(0);
        let result = materialize(&plan, &dest, &cancel, &mut |_| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                cancel.cancel();
            }
        });
        assert!(matches!(result, Err(DataError::Cancelled)));
        assert!(!dest.exists());
        no_leftovers(&dest);

        let pre = CancelFlag::new();
        pre.cancel();
        assert!(matches!(materialize(&plan, &dest, &pre, &mut |_| {}), Err(DataError::Cancelled)));
        no_leftovers(&dest);
    }

    #[test]
    fn progress_counts_bytes_and_ends_complete() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        let mut seen: Vec<JobProgress> = Vec::new();
        materialize(&plan, &dir.path().join("ds"), &CancelFlag::new(), &mut |p| seen.push(p)).unwrap();
        let last = seen.last().unwrap();
        assert_eq!(last.unit, "bytes");
        assert_eq!((last.done, last.total), (59.0, Some(59.0)));
        assert!(seen.windows(2).all(|w| w[0].done <= w[1].done));
    }

    #[test]
    fn two_runs_with_the_same_plan_give_the_same_tree() {
        let dir = TempDir::new().unwrap();
        let plan = sample_plan(&dir.path().join("src"));
        let tree = |dest: &Path| -> Vec<String> {
            let mut out = Vec::new();
            let mut stack = vec![dest.to_path_buf()];
            while let Some(d) = stack.pop() {
                for e in fs::read_dir(&d).unwrap() {
                    let p = e.unwrap().path();
                    if p.is_dir() {
                        stack.push(p);
                    } else {
                        out.push(p.strip_prefix(dest).unwrap().display().to_string());
                    }
                }
            }
            out.sort();
            out
        };
        run(&plan, &dir.path().join("a"), &MaterializeOptions::default()).unwrap();
        run(&plan, &dir.path().join("b"), &MaterializeOptions::default()).unwrap();
        assert_eq!(tree(&dir.path().join("a")), tree(&dir.path().join("b")));
    }
}
