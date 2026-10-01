//! Deterministic train/validation split.
//!
//! [`plan_split`] decides, lane by lane, which text is held out for validation. It only *plans*; nothing is written
//! until [`crate::materialize`] runs. The same lanes, config and seed always produce an identical [`SplitPlan`].
//!
//! * **Whole-file hold-out** (lanes with at least [`MIN_FILES_FOR_FILE_SPLIT`] files). Files are ordered by
//!   `blake3(seed_le_bytes ‖ lane_relative_path)` and moved to validation until about `pct` % of the lane's bytes is
//!   held out, never more than 20 % of the files and never fewer than one. A file that would overshoot the target by
//!   more than a factor of two is skipped in favour of the next one in the order, so a single huge file cannot turn a
//!   2 % hold-out into 60 %.
//! * **Tail split** (smaller lanes, including a lane that is one big file). The last
//!   `clamp(pct % of the lane, 50 kB, 5 MB)` bytes of the largest file (never more than a fifth of that file) become
//!   `<stem>.tail.txt` in validation and the rest `<stem>.head.txt` in training. The cut falls on a UTF-8 character
//!   boundary and, when there is one nearby, just after a newline.
//! * **Folder mode** uses the `val/` the dataset already has, exactly, and lets a mini-AGI layout stay in place.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use minagi_types::{Count, LaneInfo, SplitConfig, SplitMode};

use crate::error::{DataError, DataResult};
use crate::lanes::{Lane, LaneFile, LaneLayout, LayoutKind};

/// Lanes with fewer files than this are split inside a file instead of by file.
pub const MIN_FILES_FOR_FILE_SPLIT: usize = 5;
/// Hold-out never takes more than this share of a lane's files.
pub const MAX_FILE_SHARE: f64 = 0.20;
/// Bounds for the validation tail of a lane that is split inside a file, in bytes.
pub const TAIL_MIN_BYTES: u64 = 50_000;
pub const TAIL_MAX_BYTES: u64 = 5_000_000;
/// A tail may take at most this share of the file it is cut from.
pub const MAX_TAIL_SHARE: f64 = 0.20;
/// Below this a tail is not worth having; the lane gets no validation text and a warning.
const MIN_USEFUL_TAIL_BYTES: u64 = 256;
/// How far from the nominal cut the tail split looks for a newline, at most.
const CUT_SEARCH_BYTES: u64 = 64 * 1024;

/// Lowest and highest hold-out percentage a [`SplitConfig`] may ask for.
pub const MIN_PCT: f64 = 0.5;
pub const MAX_PCT: f64 = 10.0;

/// How a lane's validation text was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMethod {
    /// Whole files were held out.
    HoldOutFiles,
    /// The end of the largest file was cut off.
    TailSlice,
    /// The dataset's own `val/` folder is used as it is.
    Provided,
    /// The lane is too small to give anything to validation.
    NoValidation,
}

/// One file (or a byte range of one) in the prepared dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedFile {
    /// Path inside the lane directory, with `/` separators.
    pub dest_rel: String,
    /// The file on disk this comes from.
    pub source: PathBuf,
    /// `None` for the whole file (linked or copied as is); `Some((start, end))` for a managed copy of that byte range.
    pub range: Option<(u64, u64)>,
    /// Bytes this entry holds.
    pub size: u64,
}

impl PlannedFile {
    fn whole(f: &LaneFile) -> Self {
        Self { dest_rel: f.rel_path.clone(), source: f.path.clone(), range: None, size: f.size }
    }
}

/// The plan for one lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanePlan {
    pub name: String,
    pub display_name: String,
    pub color_slot: u32,
    pub method: SplitMethod,
    /// Sorted by `dest_rel`.
    pub train: Vec<PlannedFile>,
    /// Sorted by `dest_rel`.
    pub val: Vec<PlannedFile>,
}

impl LanePlan {
    pub fn train_bytes(&self) -> u64 {
        self.train.iter().map(|f| f.size).sum()
    }

    pub fn val_bytes(&self) -> u64 {
        self.val.iter().map(|f| f.size).sum()
    }
}

/// What to build, ready for [`crate::materialize`].
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPlan {
    /// The folders and files the dataset is built from.
    pub sources: Vec<PathBuf>,
    /// The settings actually used (the percentage is clamped to its range).
    pub cfg: SplitConfig,
    /// Set when the dataset is already `train/` + `val/` shaped and is used where it is, read-only.
    pub in_place: Option<PathBuf>,
    /// Lanes sorted by name.
    pub lanes: Vec<LanePlan>,
    pub warnings: Vec<String>,
}

impl SplitPlan {
    pub fn train_bytes(&self) -> u64 {
        self.lanes.iter().map(LanePlan::train_bytes).sum()
    }

    pub fn val_bytes(&self) -> u64 {
        self.lanes.iter().map(LanePlan::val_bytes).sum()
    }

    /// The lane table for the UI. Sample prompts are left empty; see [`crate::suggest_prompts`].
    pub fn lane_infos(&self) -> Vec<LaneInfo> {
        self.lanes
            .iter()
            .map(|l| LaneInfo {
                name: l.name.clone(),
                display_name: l.display_name.clone(),
                color_slot: l.color_slot,
                enabled: true,
                n_files: Count(l.train.len() as u64),
                train_bytes: Count(l.train_bytes()),
                val_files: Count(l.val.len() as u64),
                val_bytes: Count(l.val_bytes()),
                sample_prompt: None,
            })
            .collect()
    }
}

/// Choose what goes to validation. See the module docs for the algorithm.
///
/// Reads a little of the files it cuts a tail from (to find a clean cut). `cfg.pct` is clamped to 0.5..=10.
pub fn plan_split(layout: &LaneLayout, cfg: &SplitConfig) -> DataResult<SplitPlan> {
    let pct = if cfg.pct.is_finite() { cfg.pct.clamp(MIN_PCT, MAX_PCT) } else { SplitConfig::default().pct };
    let cfg = SplitConfig { pct, ..*cfg };

    if cfg.mode == SplitMode::Folder {
        return plan_provided(layout, cfg);
    }

    let mut warnings = Vec::new();
    let mut lanes = Vec::with_capacity(layout.lanes.len());
    for lane in &layout.lanes {
        lanes.push(plan_lane(lane, &cfg, &mut warnings)?);
    }
    Ok(SplitPlan { sources: layout.sources.clone(), cfg, in_place: None, lanes, warnings })
}

/// Folder mode: the dataset's own `val/` is used exactly.
fn plan_provided(layout: &LaneLayout, cfg: SplitConfig) -> DataResult<SplitPlan> {
    if !layout.provides_val() {
        return Err(DataError::invalid(
            "The folder split needs a val/ folder with text in it, and this dataset has none. Use the automatic split instead.",
        ));
    }
    let planned = |lane: &Lane| -> Vec<PlannedFile> {
        let mut files: Vec<PlannedFile> = lane.files.iter().map(PlannedFile::whole).collect();
        files.sort_by(|a, b| a.dest_rel.cmp(&b.dest_rel));
        files
    };
    let mut names: Vec<&str> = layout.lanes.iter().chain(&layout.val_lanes).map(|l| l.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    let lanes = names
        .into_iter()
        .filter_map(|name| {
            let train = layout.lanes.iter().find(|l| l.name == name);
            let val = layout.val_lanes.iter().find(|l| l.name == name);
            let meta = train.or(val)?;
            Some(LanePlan {
                name: meta.name.clone(),
                display_name: meta.display_name.clone(),
                color_slot: meta.color_slot,
                method: SplitMethod::Provided,
                train: train.map(planned).unwrap_or_default(),
                val: val.map(planned).unwrap_or_default(),
            })
        })
        .collect();
    let in_place = match &layout.kind {
        LayoutKind::MiniAgi { root } => Some(root.clone()),
        LayoutKind::Folders => None,
    };
    Ok(SplitPlan { sources: layout.sources.clone(), cfg, in_place, lanes, warnings: Vec::new() })
}

fn plan_lane(lane: &Lane, cfg: &SplitConfig, warnings: &mut Vec<String>) -> DataResult<LanePlan> {
    let mut plan = LanePlan {
        name: lane.name.clone(),
        display_name: lane.display_name.clone(),
        color_slot: lane.color_slot,
        method: SplitMethod::HoldOutFiles,
        train: Vec::new(),
        val: Vec::new(),
    };
    if lane.files.len() >= MIN_FILES_FOR_FILE_SPLIT {
        let held: Vec<bool> = hold_out_flags(&lane.files, cfg);
        for (file, held) in lane.files.iter().zip(held) {
            (if held { &mut plan.val } else { &mut plan.train }).push(PlannedFile::whole(file));
        }
    } else {
        plan.method = SplitMethod::TailSlice;
        split_largest_file(lane, cfg, &mut plan, warnings)?;
    }
    plan.train.sort_by(|a, b| a.dest_rel.cmp(&b.dest_rel));
    plan.val.sort_by(|a, b| a.dest_rel.cmp(&b.dest_rel));
    Ok(plan)
}

/// `blake3(seed_le_bytes ‖ rel_path)`, the order in which files are considered for hold-out.
fn order_key(seed: u32, rel_path: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(rel_path.as_bytes());
    *hasher.finalize().as_bytes()
}

/// For each file of a lane (in the lane's own order): is it held out? Independent of that order.
fn hold_out_flags(files: &[LaneFile], cfg: &SplitConfig) -> Vec<bool> {
    let lane_bytes: u64 = files.iter().map(|f| f.size).sum();
    let target = cfg.pct / 100.0 * lane_bytes as f64;
    let max_files = ((files.len() as f64 * MAX_FILE_SHARE).floor() as usize).max(1);

    let mut order: Vec<usize> = (0..files.len()).collect();
    order.sort_by(|&a, &b| {
        order_key(cfg.seed, &files[a].rel_path)
            .cmp(&order_key(cfg.seed, &files[b].rel_path))
            .then_with(|| files[a].rel_path.cmp(&files[b].rel_path))
    });

    let mut held = vec![false; files.len()];
    let (mut count, mut bytes) = (0usize, 0f64);
    for &i in &order {
        if count >= max_files || bytes >= target {
            break;
        }
        let size = files[i].size as f64;
        // Skip a file that would take us far past the target; a smaller one further down may fit.
        if bytes + size <= 2.0 * target {
            held[i] = true;
            count += 1;
            bytes += size;
        }
    }
    if count == 0 {
        // Every file is more than twice the target. A lane with this many files still needs a validation file:
        // take the smallest (ties go to the earlier in the hash order).
        if let Some(&smallest) = order.iter().min_by_key(|&&i| files[i].size) {
            held[smallest] = true;
        }
    }
    held
}

/// Lanes with few files: everything stays in training except the tail of the largest file.
fn split_largest_file(
    lane: &Lane,
    cfg: &SplitConfig,
    plan: &mut LanePlan,
    warnings: &mut Vec<String>,
) -> DataResult<()> {
    let Some(largest) = lane.files.iter().max_by(|a, b| a.size.cmp(&b.size).then_with(|| b.rel_path.cmp(&a.rel_path)))
    else {
        return Ok(());
    };
    let size = std::fs::metadata(&largest.path).map_err(|e| DataError::io(&largest.path, e))?.len();
    let wanted = (cfg.pct / 100.0 * lane.bytes() as f64).clamp(TAIL_MIN_BYTES as f64, TAIL_MAX_BYTES as f64) as u64;
    let tail_len = wanted.min((size as f64 * MAX_TAIL_SHARE) as u64);

    let others = || lane.files.iter().filter(|f| f.rel_path != largest.rel_path);
    if tail_len < MIN_USEFUL_TAIL_BYTES {
        plan.method = SplitMethod::NoValidation;
        plan.train.extend(lane.files.iter().map(PlannedFile::whole));
        warnings.push(format!(
            "The \"{}\" lane is too small to hold any text out for validation, so it has none.",
            lane.display_name
        ));
        return Ok(());
    }

    let cut = find_cut(&largest.path, size, tail_len).map_err(|e| DataError::io(&largest.path, e))?;
    let (head_rel, tail_rel) = split_names(&largest.rel_path);
    plan.train.push(PlannedFile { dest_rel: head_rel, source: largest.path.clone(), range: Some((0, cut)), size: cut });
    plan.val.push(PlannedFile {
        dest_rel: tail_rel,
        source: largest.path.clone(),
        range: Some((cut, size)),
        size: size - cut,
    });
    plan.train.extend(others().map(PlannedFile::whole));
    Ok(())
}

/// `dir/stem.ext` becomes (`dir/stem.head.txt`, `dir/stem.tail.txt`).
fn split_names(rel_path: &str) -> (String, String) {
    let (dir, file) = match rel_path.rsplit_once('/') {
        Some((d, f)) => (format!("{d}/"), f),
        None => (String::new(), rel_path),
    };
    let stem =
        Path::new(file).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| file.to_string());
    (format!("{dir}{stem}.head.txt"), format!("{dir}{stem}.tail.txt"))
}

/// Byte offset at which the tail of a `size`-byte file starts: about `tail_len` bytes from the end, moved to a
/// newline or character boundary nearby.
fn find_cut(path: &Path, size: u64, tail_len: u64) -> std::io::Result<u64> {
    let nominal = size - tail_len;
    // The search may move the cut by at most half the tail, so the tail stays between half and 1.5 times its size.
    let reach = CUT_SEARCH_BYTES.min(tail_len / 2);
    let start = nominal.saturating_sub(reach);
    let end = (nominal + reach).min(size);
    let mut window = vec![0u8; (end - start) as usize];
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    file.read_exact(&mut window)?;
    Ok(choose_cut(&window, start, nominal))
}

/// Pick the cut near `nominal` inside `window` (the bytes of the file from `window_start`). In order of
/// preference: just after the first newline at or after `nominal`; just after the last newline before it; the next
/// UTF-8 character boundary at or after it. The result is always a character boundary and at most `window_start +
/// window.len()`.
pub(crate) fn choose_cut(window: &[u8], window_start: u64, nominal: u64) -> u64 {
    let at = (nominal - window_start) as usize;
    if let Some(p) = window[at..].iter().position(|&b| b == b'\n') {
        return window_start + (at + p + 1) as u64;
    }
    if let Some(p) = window[..at].iter().rposition(|&b| b == b'\n') {
        return window_start + (p + 1) as u64;
    }
    let mut i = at;
    while i < window.len() && (window[i] & 0b1100_0000) == 0b1000_0000 {
        i += 1;
    }
    window_start + i as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;

    use tempfile::TempDir;

    use crate::lanes::{LaneLayout, LayoutKind};

    fn lane_file(name: &str, size: u64) -> LaneFile {
        LaneFile { path: PathBuf::from(format!("/nonexistent/{name}")), rel_path: name.to_string(), size, mtime_ns: 0 }
    }

    fn virtual_lane(name: &str, files: Vec<LaneFile>) -> Lane {
        Lane { name: name.into(), display_name: name.into(), color_slot: 0, files }
    }

    fn layout_of(lanes: Vec<Lane>) -> LaneLayout {
        LaneLayout { sources: vec![], kind: LayoutKind::Folders, lanes, val_lanes: vec![], warnings: vec![] }
    }

    fn auto(pct: f64, seed: u32) -> SplitConfig {
        SplitConfig { mode: SplitMode::Auto, pct, seed }
    }

    fn val_names(p: &LanePlan) -> Vec<&str> {
        p.val.iter().map(|f| f.dest_rel.as_str()).collect()
    }

    fn many_files(n: usize, size: u64) -> Vec<LaneFile> {
        (0..n).map(|i| lane_file(&format!("doc{i:03}.txt"), size)).collect()
    }

    #[test]
    fn hold_out_takes_about_pct_of_the_bytes() {
        let lane = virtual_lane("books", many_files(100, 1000));
        let plan = plan_split(&layout_of(vec![lane]), &auto(2.0, 7)).unwrap();
        let l = &plan.lanes[0];
        assert_eq!(l.method, SplitMethod::HoldOutFiles);
        assert_eq!(l.val.len(), 2, "2 % of 100 equal files");
        assert_eq!(l.train.len(), 98);
        assert_eq!(l.train_bytes() + l.val_bytes(), 100_000);
        let held: HashSet<_> = val_names(l).into_iter().collect();
        assert!(l.train.iter().all(|f| !held.contains(f.dest_rel.as_str())), "train and val are disjoint");
        assert!(l.val.iter().all(|f| f.range.is_none()), "whole files");
    }

    #[test]
    fn same_seed_same_plan_other_seed_other_plan() {
        let layout = layout_of(vec![virtual_lane("a", many_files(200, 500)), virtual_lane("b", many_files(50, 900))]);
        let one = plan_split(&layout, &auto(5.0, 1)).unwrap();
        assert_eq!(one, plan_split(&layout, &auto(5.0, 1)).unwrap());
        let two = plan_split(&layout, &auto(5.0, 2)).unwrap();
        assert_ne!(val_names(&one.lanes[0]), val_names(&two.lanes[0]));
    }

    #[test]
    fn plan_does_not_depend_on_the_order_files_were_scanned_in() {
        let files = many_files(60, 700);
        let mut reversed = files.clone();
        reversed.reverse();
        let a = plan_split(&layout_of(vec![virtual_lane("x", files)]), &auto(3.0, 9)).unwrap();
        let b = plan_split(&layout_of(vec![virtual_lane("x", reversed)]), &auto(3.0, 9)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn order_key_is_blake3_of_seed_then_path() {
        let mut h = blake3::Hasher::new();
        h.update(&5u32.to_le_bytes());
        h.update(b"dir/file.txt");
        assert_eq!(order_key(5, "dir/file.txt"), *h.finalize().as_bytes());
        assert_ne!(order_key(5, "dir/file.txt"), order_key(6, "dir/file.txt"));
    }

    #[test]
    fn hold_out_never_exceeds_a_fifth_of_the_files_and_has_at_least_one() {
        // 10 files, 10 %: the byte target is one file, the cap is 2 files.
        let plan = plan_split(&layout_of(vec![virtual_lane("l", many_files(10, 100))]), &auto(10.0, 3)).unwrap();
        assert_eq!(plan.lanes[0].val.len(), 1);
        // 5 equal files at 0.5 %: the target is a fraction of a file, but a lane with 5 files gets one.
        let plan = plan_split(&layout_of(vec![virtual_lane("l", many_files(5, 1000))]), &auto(0.5, 3)).unwrap();
        assert_eq!(plan.lanes[0].method, SplitMethod::HoldOutFiles);
        assert_eq!(plan.lanes[0].val.len(), 1);
        // The cap holds even when the target asks for more: 6 small files, 10 % of bytes would need 1 file anyway.
        let mut files = many_files(6, 10);
        files.push(lane_file("big.txt", 10_000));
        let plan = plan_split(&layout_of(vec![virtual_lane("l", files)]), &auto(10.0, 3)).unwrap();
        assert!(plan.lanes[0].val.len() <= 1);
    }

    #[test]
    fn a_huge_file_is_skipped_when_it_would_blow_the_target() {
        let mut files = many_files(30, 100);
        files.push(lane_file("huge.bin.txt", 1_000_000));
        let layout = layout_of(vec![virtual_lane("l", files)]);
        for seed in 0..20 {
            let plan = plan_split(&layout, &auto(2.0, seed)).unwrap();
            let l = &plan.lanes[0];
            assert!(l.val_bytes() <= 2 * (0.02 * 1_003_000.0) as u64, "seed {seed}: val {} bytes", l.val_bytes());
            assert!(!val_names(l).contains(&"huge.bin.txt"));
        }
    }

    #[test]
    fn when_every_file_is_far_over_the_target_the_smallest_is_held_out() {
        let files: Vec<LaneFile> = (0..5).map(|i| lane_file(&format!("f{i}.txt"), 1000 + i * 10)).collect();
        let plan = plan_split(&layout_of(vec![virtual_lane("l", files)]), &auto(0.5, 1)).unwrap();
        assert_eq!(val_names(&plan.lanes[0]), ["f0.txt"]);
    }

    #[test]
    fn pct_is_clamped_to_its_range() {
        let layout = layout_of(vec![virtual_lane("l", many_files(100, 100))]);
        assert_eq!(plan_split(&layout, &auto(0.0, 1)).unwrap().cfg.pct, MIN_PCT);
        assert_eq!(plan_split(&layout, &auto(50.0, 1)).unwrap().cfg.pct, MAX_PCT);
        assert_eq!(plan_split(&layout, &auto(f64::NAN, 1)).unwrap().cfg.pct, 2.0);
        let low = plan_split(&layout, &auto(0.5, 1)).unwrap().lanes[0].val.len();
        let high = plan_split(&layout, &auto(10.0, 1)).unwrap().lanes[0].val.len();
        assert_eq!((low, high), (1, 10));
    }

    // ----- tail split -----

    fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> LaneFile {
        let path = dir.join(name);
        fs::write(&path, bytes).unwrap();
        LaneFile { path, rel_path: name.to_string(), size: bytes.len() as u64, mtime_ns: 0 }
    }

    fn sentence_text(lines: usize) -> String {
        (0..lines)
            .map(|i| format!("Line number {i} of a rather ordinary story, with ünïcode and an emoji 🎉.\n"))
            .collect()
    }

    /// Read the planned entries back and return (train, val) concatenations.
    fn read_planned(f: &PlannedFile) -> Vec<u8> {
        let all = fs::read(&f.source).unwrap();
        match f.range {
            Some((s, e)) => all[s as usize..e as usize].to_vec(),
            None => all,
        }
    }

    #[test]
    fn a_single_big_file_is_split_into_head_and_tail_at_a_newline() {
        let dir = TempDir::new().unwrap();
        let text = sentence_text(8000); // ~ 650 kB
        let file = write_file(dir.path(), "novel.txt", text.as_bytes());
        let plan = plan_split(&layout_of(vec![virtual_lane("stories", vec![file])]), &auto(2.0, 1)).unwrap();
        let l = &plan.lanes[0];
        assert_eq!(l.method, SplitMethod::TailSlice);
        assert_eq!((l.train.len(), l.val.len()), (1, 1));
        assert_eq!(l.train[0].dest_rel, "novel.head.txt");
        assert_eq!(l.val[0].dest_rel, "novel.tail.txt");

        let (head, tail) = (read_planned(&l.train[0]), read_planned(&l.val[0]));
        assert_eq!([head.clone(), tail.clone()].concat(), text.as_bytes(), "nothing lost, nothing duplicated");
        assert!(String::from_utf8(tail.clone()).is_ok() && String::from_utf8(head.clone()).is_ok());
        assert_eq!(head.last(), Some(&b'\n'), "the cut is just after a newline");
        // 2 % of 650 kB is under the 50 kB floor, so the tail is about 50 kB (within the half-tail search reach).
        let tail_len = tail.len() as u64;
        assert!((25_000..=75_000).contains(&tail_len), "tail is {tail_len} bytes");
        assert_eq!(l.val[0].size, tail_len);
        assert_eq!(l.train[0].size + l.val[0].size, text.len() as u64);
    }

    #[test]
    fn the_tail_is_two_percent_of_a_larger_lane() {
        let dir = TempDir::new().unwrap();
        let text = sentence_text(60_000); // ~ 4.9 MB: 2 % is ~ 98 kB
        let file = write_file(dir.path(), "a.txt", text.as_bytes());
        let size = file.size;
        let plan = plan_split(&layout_of(vec![virtual_lane("l", vec![file])]), &auto(2.0, 1)).unwrap();
        let tail = plan.lanes[0].val[0].size as f64;
        assert!((tail - 0.02 * size as f64).abs() < 1000.0, "tail {tail} of {size}");
    }

    #[test]
    fn the_tail_is_capped_at_five_megabytes() {
        let dir = TempDir::new().unwrap();
        // A sparse 400 MB file: 2 % would be 8 MB, the cap says 5 MB. The zeros have no newline, so the cut is exact.
        let path = dir.path().join("huge.txt");
        let f = fs::File::create(&path).unwrap();
        f.set_len(400_000_000).unwrap();
        let file = LaneFile { path, rel_path: "huge.txt".into(), size: 400_000_000, mtime_ns: 0 };
        let plan = plan_split(&layout_of(vec![virtual_lane("l", vec![file])]), &auto(2.0, 1)).unwrap();
        assert_eq!(plan.lanes[0].val[0].size, TAIL_MAX_BYTES);
        assert_eq!(plan.lanes[0].val[0].range, Some((395_000_000, 400_000_000)));
    }

    #[test]
    fn the_cut_falls_on_a_char_boundary_when_there_is_no_newline() {
        let dir = TempDir::new().unwrap();
        // One giant line of 3-byte characters: no newline anywhere.
        let text = "日".repeat(100_000); // 300 kB
        let file = write_file(dir.path(), "line.txt", text.as_bytes());
        for pct in [0.5, 2.0, 7.3] {
            let plan = plan_split(&layout_of(vec![virtual_lane("l", vec![file.clone()])]), &auto(pct, 1)).unwrap();
            let l = &plan.lanes[0];
            let (head, tail) = (read_planned(&l.train[0]), read_planned(&l.val[0]));
            assert!(String::from_utf8(head).is_ok(), "pct {pct}: head must be valid UTF-8");
            assert!(String::from_utf8(tail).is_ok(), "pct {pct}: tail must be valid UTF-8");
        }
    }

    #[test]
    fn choose_cut_prefers_forward_newline_then_backward_then_char_boundary() {
        let w = b"aaaa\nbbbb\ncccc";
        assert_eq!(choose_cut(w, 100, 102), 105, "first newline at or after the nominal cut");
        assert_eq!(choose_cut(w, 100, 105), 110, "a newline just before the nominal cut does not count");
        // Nothing forward: fall back to the last newline before.
        assert_eq!(choose_cut(w, 100, 111), 110);
        // No newline at all: land on a char boundary (skip continuation bytes).
        let s = "ééééé".as_bytes(); // each char is 2 bytes
        assert_eq!(choose_cut(s, 0, 3), 4);
        assert_eq!(choose_cut(s, 0, 4), 4);
        assert_eq!(choose_cut(s, 0, 0), 0);
    }

    #[test]
    fn choose_cut_after_a_newline_exactly_at_the_nominal_offset() {
        // A newline byte exactly at the nominal offset ends the preceding line: the cut goes right after it.
        assert_eq!(choose_cut(b"ab\ncd", 0, 2), 3);
    }

    #[test]
    fn small_lanes_split_the_largest_file_and_keep_the_others() {
        let dir = TempDir::new().unwrap();
        let big = write_file(dir.path(), "big.txt", sentence_text(5000).as_bytes());
        let small = write_file(dir.path(), "small.txt", sentence_text(10).as_bytes());
        let other = write_file(dir.path(), "sub_other.txt", sentence_text(20).as_bytes());
        let plan = plan_split(&layout_of(vec![virtual_lane("l", vec![small, big, other])]), &auto(2.0, 1)).unwrap();
        let l = &plan.lanes[0];
        let train: Vec<&str> = l.train.iter().map(|f| f.dest_rel.as_str()).collect();
        assert_eq!(train, ["big.head.txt", "small.txt", "sub_other.txt"]);
        assert_eq!(val_names(l), ["big.tail.txt"]);
        assert!(l.train[1].range.is_none(), "untouched files stay whole");
    }

    #[test]
    fn exactly_five_files_use_the_whole_file_split_and_four_use_the_tail() {
        let dir = TempDir::new().unwrap();
        let mk = |n: usize| -> Vec<LaneFile> {
            (0..n).map(|i| write_file(dir.path(), &format!("f{i}.txt"), sentence_text(300).as_bytes())).collect()
        };
        let four = plan_split(&layout_of(vec![virtual_lane("l", mk(4))]), &auto(2.0, 1)).unwrap();
        assert_eq!(four.lanes[0].method, SplitMethod::TailSlice);
        let five = plan_split(&layout_of(vec![virtual_lane("l", mk(5))]), &auto(2.0, 1)).unwrap();
        assert_eq!(five.lanes[0].method, SplitMethod::HoldOutFiles);
    }

    #[test]
    fn tail_split_is_deterministic() {
        let dir = TempDir::new().unwrap();
        let file = write_file(dir.path(), "a.txt", sentence_text(4000).as_bytes());
        let layout = layout_of(vec![virtual_lane("l", vec![file])]);
        assert_eq!(plan_split(&layout, &auto(2.0, 5)).unwrap(), plan_split(&layout, &auto(2.0, 5)).unwrap());
    }

    #[test]
    fn a_lane_too_small_for_validation_gets_none_and_a_warning() {
        let dir = TempDir::new().unwrap();
        let file = write_file(dir.path(), "tiny.txt", b"just a few words here\n");
        let plan = plan_split(&layout_of(vec![virtual_lane("tiny", vec![file])]), &auto(2.0, 1)).unwrap();
        let l = &plan.lanes[0];
        assert_eq!(l.method, SplitMethod::NoValidation);
        assert!(l.val.is_empty());
        assert_eq!(l.train.len(), 1);
        assert_eq!(plan.warnings.len(), 1);
    }

    #[test]
    fn a_missing_source_file_is_an_error_not_a_panic() {
        let layout = layout_of(vec![virtual_lane("l", vec![lane_file("gone.txt", 1_000_000)])]);
        assert!(matches!(plan_split(&layout, &auto(2.0, 1)), Err(DataError::Io { .. })));
    }

    // ----- folder mode -----

    fn provided_layout(in_place: bool) -> LaneLayout {
        let kind = if in_place { LayoutKind::MiniAgi { root: PathBuf::from("/data/ds") } } else { LayoutKind::Folders };
        LaneLayout {
            sources: vec![],
            kind,
            lanes: vec![virtual_lane("stories", vec![lane_file("a.txt", 100), lane_file("b.txt", 100)])],
            val_lanes: vec![
                virtual_lane("extra", vec![lane_file("e.txt", 5)]),
                virtual_lane("stories", vec![lane_file("v.txt", 10)]),
            ],
            warnings: vec![],
        }
    }

    #[test]
    fn folder_mode_uses_the_provided_val_exactly_and_stays_in_place() {
        let cfg = SplitConfig { mode: SplitMode::Folder, ..SplitConfig::default() };
        let plan = plan_split(&provided_layout(true), &cfg).unwrap();
        assert_eq!(plan.in_place, Some(PathBuf::from("/data/ds")));
        let names: Vec<&str> = plan.lanes.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["extra", "stories"]);
        assert!(plan.lanes.iter().all(|l| l.method == SplitMethod::Provided));
        assert_eq!(val_names(&plan.lanes[1]), ["v.txt"]);
        assert_eq!(plan.lanes[1].train.len(), 2);
        assert!(plan.lanes[0].train.is_empty(), "a val-only domain has no training files");
        assert_eq!((plan.train_bytes(), plan.val_bytes()), (200, 15));

        let not_in_place = plan_split(&provided_layout(false), &cfg).unwrap();
        assert_eq!(not_in_place.in_place, None);
    }

    #[test]
    fn folder_mode_without_val_is_an_error() {
        let layout = layout_of(vec![virtual_lane("l", many_files(10, 10))]);
        let cfg = SplitConfig { mode: SplitMode::Folder, ..SplitConfig::default() };
        assert!(matches!(plan_split(&layout, &cfg), Err(DataError::Invalid(_))));
    }

    #[test]
    fn auto_mode_on_a_mini_agi_layout_is_not_in_place() {
        // Holding files out of train/ needs a managed copy, so Auto never claims the dataset can stay in place.
        let mut layout = provided_layout(true);
        layout.lanes[0].files = many_files(20, 100);
        let plan = plan_split(&layout, &auto(2.0, 1)).unwrap();
        assert_eq!(plan.in_place, None);
    }

    #[test]
    fn lane_infos_summarise_the_plan() {
        let plan = plan_split(&layout_of(vec![virtual_lane("books", many_files(100, 1000))]), &auto(2.0, 1)).unwrap();
        let infos = plan.lane_infos();
        assert_eq!(infos.len(), 1);
        let i = &infos[0];
        assert_eq!((i.name.as_str(), i.n_files, i.val_files), ("books", Count(98), Count(2)));
        assert_eq!((i.train_bytes, i.val_bytes), (Count(98_000), Count(2_000)));
        assert!(i.enabled && i.sample_prompt.is_none());
    }
}
