//! Lane detection: which folders of the user's text become which lanes.
//!
//! A lane is one kind of text. [`detect_lanes`] applies four rules, in this order:
//!
//! 1. A single dropped folder that already has a `train/` subfolder is a *mini-AGI layout*: the subfolders of `train/`
//!    are the lanes, the subfolders of `val/` are the validation domains, and loose files directly under `train/`
//!    form one lane named after the folder. Such a dataset can be used **in place, read-only**.
//! 2. A single folder with top-level subfolders that hold text, where loose files are at most 20 % of the text bytes:
//!    each top-level subfolder is a lane (deeper files belong to it) and the loose files form the lane `misc`.
//! 3. Any other single folder is one lane named after the folder.
//! 4. Several dropped paths: one lane per dropped folder, and all dropped loose files together form `my-files`.
//!
//! [`LaneMode`] lets the user override the choice. Lane names are slugified (`[a-z0-9_-]`) and de-duplicated.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use minagi_types::LaneMode;

use crate::scan::{ScanResult, ScannedFile};

/// Loose files may be at most this share of the text bytes for rule 2 to apply.
pub const LOOSE_SHARE_LIMIT: f64 = 0.20;
/// Name of the lane that collects loose files next to lane folders (rule 2).
pub const MISC_LANE: &str = "misc";
/// Name of the lane that collects dropped loose files (rule 4).
pub const LOOSE_FILES_LANE: &str = "my-files";
/// The first colour slot that is shared by all "other" lanes.
pub const OTHER_COLOR_SLOT: u32 = 8;

/// Lanes that always keep the same colour, so a chart of a dataset reads the same everywhere.
const FIXED_COLOR_SLOTS: [(&str, u32); 8] = [
    ("stories", 0),
    ("arithmetic", 1),
    ("wikipedia", 2),
    ("code", 3),
    ("chat", 4),
    ("reasoning", 5),
    ("chess", 6),
    ("self-knowledge", 7),
];

/// One text file as a lane sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneFile {
    /// Where the file lives on disk.
    pub path: PathBuf,
    /// Path inside the lane, with `/` separators.
    pub rel_path: String,
    pub size: u64,
    pub mtime_ns: u64,
}

/// One lane: a name and its files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane {
    /// Folder-safe name, unique within the layout.
    pub name: String,
    /// The original folder name (or a friendly name for generated lanes).
    pub display_name: String,
    pub color_slot: u32,
    pub files: Vec<LaneFile>,
}

impl Lane {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// Whether the dataset is used where it is or has to be rebuilt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutKind {
    /// Already `train/` + `val/` shaped; `root` can be used in place, read-only.
    MiniAgi { root: PathBuf },
    /// Folders that need a managed copy (or links) to be shaped for the engine.
    Folders,
}

/// The result of [`detect_lanes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneLayout {
    /// The folders and files the user dropped (recorded in the dataset's manifest).
    pub sources: Vec<PathBuf>,
    pub kind: LayoutKind,
    /// Training lanes, sorted by name.
    pub lanes: Vec<Lane>,
    /// Validation domains the dataset already provides (mini-AGI layout only), sorted by name.
    pub val_lanes: Vec<Lane>,
    pub warnings: Vec<String>,
}

impl LaneLayout {
    /// True for a mini-AGI layout, which is used in place.
    pub fn in_place(&self) -> bool {
        matches!(self.kind, LayoutKind::MiniAgi { .. })
    }

    /// True when the dataset ships its own validation files.
    pub fn provides_val(&self) -> bool {
        self.val_lanes.iter().any(|l| !l.files.is_empty())
    }

    pub fn lane(&self, name: &str) -> Option<&Lane> {
        self.lanes.iter().find(|l| l.name == name)
    }

    pub fn train_bytes(&self) -> u64 {
        self.lanes.iter().map(Lane::bytes).sum()
    }
}

/// Turn a folder name into a lane name: lower case, `[a-z0-9_-]` only, runs of other characters become one `-`.
/// A name with nothing usable in it becomes `lane`.
pub fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() { "lane".to_string() } else { trimmed.to_string() }
}

/// Hands out unique lane names. Each lane has an identity `key` (the original folder, say); asking again for the same
/// key returns the same name, while different keys whose preferred names slugify alike get `-2`, `-3`, ... suffixes.
#[derive(Default)]
struct Slugger {
    by_key: HashMap<String, String>,
    taken: HashSet<String>,
}

impl Slugger {
    fn name_for(&mut self, key: &str, preferred: &str) -> String {
        if let Some(existing) = self.by_key.get(key) {
            return existing.clone();
        }
        let base = slugify(preferred);
        let mut candidate = base.clone();
        let mut n = 2;
        while !self.taken.insert(candidate.clone()) {
            candidate = format!("{base}-{n}");
            n += 1;
        }
        self.by_key.insert(key.to_string(), candidate.clone());
        candidate
    }
}

/// Assign colour slots: the eight well-known lanes keep theirs; every other lane takes the lowest slot nobody has
/// claimed, in sorted-name order. Slots from [`OTHER_COLOR_SLOT`] up mean "folds into Other".
pub fn assign_color_slots(names: &[&str]) -> HashMap<String, u32> {
    let mut slots: HashMap<String, u32> = HashMap::new();
    let mut used: HashSet<u32> = HashSet::new();
    for name in names {
        if let Some(&(_, slot)) = FIXED_COLOR_SLOTS.iter().find(|(n, _)| n == name) {
            slots.insert((*name).to_string(), slot);
            used.insert(slot);
        }
    }
    let mut others: Vec<&str> = names.iter().copied().filter(|n| !slots.contains_key(*n)).collect();
    others.sort_unstable();
    let mut next = 0;
    for name in others {
        while used.contains(&next) {
            next += 1;
        }
        slots.insert(name.to_string(), next);
        used.insert(next);
    }
    slots
}

/// Group the scanned text files into lanes.
///
/// `scan` is the result of [`crate::scan_paths`] on the user's paths. See the module docs for the rules; `mode`
/// forces a single lane ([`LaneMode::OneLane`]) or one lane per top-level folder ([`LaneMode::PerFolder`]) instead of
/// letting the rules decide ([`LaneMode::Auto`]). Both overrides ignore a `train/` + `val/` layout.
pub fn detect_lanes(scan: &ScanResult, mode: LaneMode) -> LaneLayout {
    let mut warnings = Vec::new();
    if scan.files.is_empty() {
        warnings.push("No text files were found.".to_string());
    }
    let single_dir = (scan.roots.len() == 1 && scan.roots[0].is_dir).then(|| &scan.roots[0]);

    let (kind, mut lanes, mut val_lanes) = match single_dir {
        Some(root) => {
            let layout_files = || scan.files.iter().filter(|f| f.rel_path.starts_with("train/"));
            if mode == LaneMode::Auto && layout_files().next().is_some() {
                let (lanes, val, ignored) = mini_agi_lanes(scan, &root.name);
                if ignored > 0 {
                    warnings.push(format!(
                        "{ignored} text file{} outside train/ and val/ {} left out.",
                        if ignored == 1 { "" } else { "s" },
                        if ignored == 1 { "was" } else { "were" }
                    ));
                }
                (LayoutKind::MiniAgi { root: root.path.clone() }, lanes, val)
            } else {
                (LayoutKind::Folders, folder_lanes(scan, &root.name, mode), Vec::new())
            }
        }
        None => (LayoutKind::Folders, multi_root_lanes(scan, mode), Vec::new()),
    };

    lanes.sort_by(|a, b| a.name.cmp(&b.name));
    val_lanes.sort_by(|a, b| a.name.cmp(&b.name));

    // Colours are chosen over train and val-only names together, so a val-only domain still has a colour.
    let mut names: Vec<&str> = lanes.iter().map(|l| l.name.as_str()).collect();
    for lane in &val_lanes {
        if !names.contains(&lane.name.as_str()) {
            names.push(&lane.name);
        }
    }
    let slots = assign_color_slots(&names);
    for lane in lanes.iter_mut().chain(val_lanes.iter_mut()) {
        lane.color_slot = slots.get(&lane.name).copied().unwrap_or(OTHER_COLOR_SLOT);
    }
    let sources = scan.roots.iter().map(|r| r.path.clone()).collect();
    LaneLayout { sources, kind, lanes, val_lanes, warnings }
}

/// A set of lanes being filled, keyed by lane name.
#[derive(Default)]
struct LaneSet(BTreeMap<String, Lane>);

impl LaneSet {
    /// Add `file` to lane `name` (created on first use) under the lane-relative path `rel_path`.
    fn add(&mut self, name: &str, display: &str, file: &ScannedFile, rel_path: String) {
        let lane = self.0.entry(name.to_string()).or_insert_with(|| Lane {
            name: name.to_string(),
            display_name: display.to_string(),
            color_slot: 0,
            files: Vec::new(),
        });
        lane.files.push(LaneFile { path: file.path.clone(), rel_path, size: file.size, mtime_ns: file.mtime_ns });
    }

    fn into_lanes(self) -> Vec<Lane> {
        self.0.into_values().collect()
    }
}

/// Key of the lane made from loose files (so train and val agree on its name).
const LOOSE_KEY: &str = "\u{0}loose";

/// Rule 1. Returns train lanes, val domains and the number of text files outside `train/` and `val/`.
fn mini_agi_lanes(scan: &ScanResult, root_name: &str) -> (Vec<Lane>, Vec<Lane>, usize) {
    // One slugger for both sides, so `val/Stories` pairs with `train/Stories`.
    let mut slugger = Slugger::default();
    let (mut train, mut val) = (LaneSet::default(), LaneSet::default());
    let mut ignored = 0;
    // Real subfolders claim their names first; the loose-files lane takes what is left.
    let mut loose: Vec<(&ScannedFile, bool)> = Vec::new();
    for f in &scan.files {
        let parts: Vec<&str> = f.rel_path.split('/').collect();
        let is_val = parts[0] == "val";
        if parts[0] != "train" && !is_val {
            ignored += 1;
        } else if parts.len() == 2 {
            loose.push((f, is_val));
        } else if parts.len() >= 3 {
            let name = slugger.name_for(parts[1], parts[1]);
            let side = if is_val { &mut val } else { &mut train };
            side.add(&name, parts[1], f, parts[2..].join("/"));
        } else {
            ignored += 1;
        }
    }
    for (f, is_val) in loose {
        let name = slugger.name_for(LOOSE_KEY, root_name);
        let side = if is_val { &mut val } else { &mut train };
        side.add(&name, root_name, f, f.rel_path.rsplit('/').next().unwrap_or(&f.rel_path).to_string());
    }
    (train.into_lanes(), val.into_lanes(), ignored)
}

/// Rules 2 and 3 (and the `OneLane` / `PerFolder` overrides) for a single dropped folder.
fn folder_lanes(scan: &ScanResult, root_name: &str, mode: LaneMode) -> Vec<Lane> {
    let is_loose = |f: &ScannedFile| !f.rel_path.contains('/');
    let total_bytes: u64 = scan.files.iter().map(|f| f.size).sum();
    let loose_bytes: u64 = scan.files.iter().filter(|f| is_loose(f)).map(|f| f.size).sum();
    let has_subfolders = scan.files.iter().any(|f| !is_loose(f));
    let per_folder = match mode {
        LaneMode::OneLane => false,
        LaneMode::PerFolder => has_subfolders,
        LaneMode::Auto => has_subfolders && (loose_bytes as f64) <= LOOSE_SHARE_LIMIT * total_bytes as f64,
    };

    let mut slugger = Slugger::default();
    let mut lanes = LaneSet::default();
    if per_folder {
        let mut loose = Vec::new();
        for f in &scan.files {
            match f.rel_path.split_once('/') {
                Some((top, rest)) => lanes.add(&slugger.name_for(top, top), top, f, rest.to_string()),
                None => loose.push(f),
            }
        }
        // Real folders have their names by now, so a folder called "misc" is not overwritten.
        for f in loose {
            lanes.add(&slugger.name_for(LOOSE_KEY, MISC_LANE), MISC_LANE, f, f.rel_path.clone());
        }
    } else {
        let name = slugger.name_for(LOOSE_KEY, root_name);
        for f in &scan.files {
            lanes.add(&name, root_name, f, f.rel_path.clone());
        }
    }
    lanes.into_lanes()
}

/// Rule 4: one lane per dropped folder; dropped loose files share `my-files`.
fn multi_root_lanes(scan: &ScanResult, mode: LaneMode) -> Vec<Lane> {
    let mut slugger = Slugger::default();
    let mut lanes = LaneSet::default();
    // Folders claim their names first, then the loose files.
    for f in scan.files.iter().filter(|f| scan.roots[f.root].is_dir) {
        let root = &scan.roots[f.root];
        if mode == LaneMode::OneLane {
            // Prefix with the folder name so several folders merged into one lane keep distinct paths.
            let name = slugger.name_for(LOOSE_KEY, LOOSE_FILES_LANE);
            lanes.add(&name, "My files", f, format!("{}/{}", root.name, f.rel_path));
        } else {
            let name = slugger.name_for(&format!("root:{}", f.root), &root.name);
            lanes.add(&name, &root.name, f, f.rel_path.clone());
        }
    }
    for f in scan.files.iter().filter(|f| !scan.roots[f.root].is_dir) {
        let name = slugger.name_for(LOOSE_KEY, LOOSE_FILES_LANE);
        lanes.add(&name, "My files", f, f.rel_path.clone());
    }
    lanes.into_lanes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    use tempfile::TempDir;

    use crate::scan::{PrevIndex, ScanOptions, scan_paths};
    use crate::util::CancelFlag;

    fn write(root: &Path, rel: &str, bytes: usize) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "x".repeat(bytes)).unwrap();
    }

    fn scan(paths: &[PathBuf]) -> ScanResult {
        scan_paths(paths, &ScanOptions::default(), &PrevIndex::new(), &CancelFlag::new(), &mut |_| {})
    }

    fn layout(paths: &[PathBuf], mode: LaneMode) -> LaneLayout {
        detect_lanes(&scan(paths), mode)
    }

    /// `(lane name, sorted lane-relative paths)` for every lane.
    fn summary(lanes: &[Lane]) -> Vec<(String, Vec<String>)> {
        lanes
            .iter()
            .map(|l| {
                let mut rels: Vec<String> = l.files.iter().map(|f| f.rel_path.clone()).collect();
                rels.sort();
                (l.name.clone(), rels)
            })
            .collect()
    }

    fn lane_names(l: &LaneLayout) -> Vec<&str> {
        l.lanes.iter().map(|l| l.name.as_str()).collect()
    }

    fn entry(name: &str, rels: &[&str]) -> (String, Vec<String>) {
        (name.to_string(), rels.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn slugify_produces_safe_names() {
        assert_eq!(slugify("Stories"), "stories");
        assert_eq!(slugify("My Notes (2024)!"), "my-notes-2024");
        assert_eq!(slugify("self-knowledge"), "self-knowledge");
        assert_eq!(slugify("snake_case_name"), "snake_case_name");
        assert_eq!(slugify("  --weird--  "), "weird");
        assert_eq!(slugify("日本語"), "lane");
        assert_eq!(slugify(""), "lane");
        assert!(
            slugify("Ünï cödé / path")
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        );
    }

    #[test]
    fn slugger_dedupes_with_numeric_suffixes() {
        let mut s = Slugger::default();
        assert_eq!(s.name_for("a", "Data"), "data");
        assert_eq!(s.name_for("b", "data"), "data-2");
        assert_eq!(s.name_for("c", "DATA"), "data-3");
        assert_eq!(s.name_for("b", "anything"), "data-2", "same key, same name");
    }

    #[test]
    fn color_slots_follow_the_fixed_table_then_sorted_free_slots() {
        let slots = assign_color_slots(&["code", "stories", "zebra", "apple", "chess"]);
        assert_eq!((slots["stories"], slots["code"], slots["chess"]), (0, 3, 6));
        // Free slots are 1, 2, 4, 5, 7, 8, ...: apple comes first alphabetically.
        assert_eq!((slots["apple"], slots["zebra"]), (1, 2));

        let names: Vec<String> = (0..12).map(|i| format!("lane{i:02}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let slots = assign_color_slots(&refs);
        let mut values: Vec<u32> = slots.values().copied().collect();
        values.sort_unstable();
        assert_eq!(values, (0..12).collect::<Vec<u32>>());
        assert!(slots["lane11"] >= OTHER_COLOR_SLOT, "the ninth and later lanes fold into Other");

        let all = assign_color_slots(&["self-knowledge", "reasoning", "chat", "wikipedia", "arithmetic", "extra"]);
        assert_eq!(
            (all["arithmetic"], all["wikipedia"], all["chat"], all["reasoning"], all["self-knowledge"]),
            (1, 2, 4, 5, 7)
        );
        assert_eq!(all["extra"], 0, "stories is absent, so its slot is free");
    }

    #[test]
    fn rule_1_mini_agi_layout() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("MyCorpus");
        write(&root, "train/stories/a.txt", 100);
        write(&root, "train/stories/sub/b.txt", 100);
        write(&root, "train/code/c.rs", 100);
        write(&root, "train/loose.txt", 10);
        write(&root, "val/stories/v.txt", 10);
        write(&root, "val/extra/e.txt", 10);
        write(&root, "README.md", 10);
        write(&root, "other/notes.txt", 10);

        let l = layout(std::slice::from_ref(&root), LaneMode::Auto);
        assert_eq!(l.kind, LayoutKind::MiniAgi { root: root.clone() });
        assert!(l.in_place() && l.provides_val());
        assert_eq!(
            summary(&l.lanes),
            [entry("code", &["c.rs"]), entry("mycorpus", &["loose.txt"]), entry("stories", &["a.txt", "sub/b.txt"])]
        );
        assert_eq!(summary(&l.val_lanes), [entry("extra", &["e.txt"]), entry("stories", &["v.txt"])]);
        assert_eq!(l.lane("mycorpus").unwrap().display_name, "MyCorpus");
        assert_eq!(l.warnings.len(), 1, "{:?}", l.warnings);
        assert!(l.warnings[0].contains("2 text files outside train/ and val/ were left out"));
        // Colours: stories 0 and code 3 are fixed; "extra" (val only) and "mycorpus" take the free slots 1 and 2.
        assert_eq!((l.lane("stories").unwrap().color_slot, l.lane("code").unwrap().color_slot), (0, 3));
        assert_eq!(l.val_lanes[0].color_slot, 1);
        assert_eq!(l.lane("mycorpus").unwrap().color_slot, 2);
    }

    #[test]
    fn rule_1_needs_text_under_train_and_works_without_val() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("ds");
        write(&root, "train/chat/log.txt", 50);
        let l = layout(std::slice::from_ref(&root), LaneMode::Auto);
        assert!(l.in_place());
        assert!(!l.provides_val());
        assert_eq!(lane_names(&l), ["chat"]);

        // An empty train/ is not a layout.
        let empty = dir.path().join("empty");
        write(&empty, "train/.keep", 0);
        write(&empty, "notes/a.txt", 50);
        let l = layout(&[empty], LaneMode::Auto);
        assert!(!l.in_place());
    }

    #[test]
    fn rule_1_loose_lane_does_not_steal_a_real_lane_name() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("books");
        write(&root, "train/books/a.txt", 50);
        write(&root, "train/loose.txt", 5);
        write(&root, "val/loose.txt", 5);
        let l = layout(&[root], LaneMode::Auto);
        assert_eq!(summary(&l.lanes), [entry("books", &["a.txt"]), entry("books-2", &["loose.txt"])]);
        assert_eq!(
            summary(&l.val_lanes),
            [entry("books-2", &["loose.txt"])],
            "train and val agree on the loose lane's name"
        );
    }

    #[test]
    fn rule_2_subfolders_become_lanes_and_loose_files_become_misc() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("library");
        write(&root, "Fiction/a.txt", 400);
        write(&root, "Fiction/deep/er/b.txt", 400);
        write(&root, "Poems & Songs/c.txt", 150);
        write(&root, "readme.txt", 50); // 50 of 1000 = 5 %
        let l = layout(&[root], LaneMode::Auto);
        assert!(!l.in_place());
        assert_eq!(
            summary(&l.lanes),
            [
                entry("fiction", &["a.txt", "deep/er/b.txt"]),
                entry("misc", &["readme.txt"]),
                entry("poems-songs", &["c.txt"])
            ]
        );
        assert_eq!(l.lane("poems-songs").unwrap().display_name, "Poems & Songs");
    }

    #[test]
    fn rule_2_loose_share_boundary_is_twenty_percent_inclusive() {
        let dir = TempDir::new().unwrap();
        let at_limit = dir.path().join("at");
        write(&at_limit, "sub/a.txt", 800);
        write(&at_limit, "loose.txt", 200); // exactly 20 %
        assert_eq!(lane_names(&layout(&[at_limit], LaneMode::Auto)), ["misc", "sub"]);

        let over = dir.path().join("over");
        write(&over, "sub/a.txt", 799);
        write(&over, "loose.txt", 201); // just over 20 %: rule 3
        let l = layout(&[over], LaneMode::Auto);
        assert_eq!(summary(&l.lanes), [entry("over", &["loose.txt", "sub/a.txt"])]);
    }

    #[test]
    fn rule_2_a_real_misc_folder_keeps_its_name() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("r");
        write(&root, "misc/a.txt", 900);
        write(&root, "b/b.txt", 900);
        write(&root, "loose.txt", 10);
        let l = layout(&[root], LaneMode::Auto);
        assert_eq!(lane_names(&l), ["b", "misc", "misc-2"]);
        assert_eq!(summary(&l.lanes)[1], entry("misc", &["a.txt"]));
        assert_eq!(summary(&l.lanes)[2], entry("misc-2", &["loose.txt"]));
    }

    #[test]
    fn rule_2_without_loose_files_has_no_misc_lane() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("r");
        write(&root, "one/a.txt", 10);
        write(&root, "two/b.txt", 10);
        assert_eq!(lane_names(&layout(&[root], LaneMode::Auto)), ["one", "two"]);
    }

    #[test]
    fn rule_3_one_lane_named_after_the_folder() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("My Essays");
        write(&root, "a.txt", 10);
        write(&root, "b.txt", 10);
        let l = layout(&[root], LaneMode::Auto);
        assert_eq!(summary(&l.lanes), [entry("my-essays", &["a.txt", "b.txt"])]);
        assert_eq!(l.lanes[0].display_name, "My Essays");
        assert!(l.val_lanes.is_empty());
    }

    #[test]
    fn rule_4_one_lane_per_root_and_my_files_for_loose_files() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("blog");
        let b = dir.path().join("papers");
        write(&a, "x/p1.txt", 10);
        write(&b, "p2.txt", 10);
        write(dir.path(), "note1.txt", 5);
        write(dir.path(), "sub/note2.txt", 5);
        let l = layout(&[a, b, dir.path().join("note1.txt"), dir.path().join("sub/note2.txt")], LaneMode::Auto);
        assert_eq!(
            summary(&l.lanes),
            [
                entry("blog", &["x/p1.txt"]),
                entry("my-files", &["note1.txt", "note2.txt"]),
                entry("papers", &["p2.txt"])
            ]
        );
        assert_eq!(l.lane("my-files").unwrap().display_name, "My files");
    }

    #[test]
    fn rule_4_same_named_roots_are_deduped_and_a_single_loose_file_is_my_files() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("one/data");
        let b = dir.path().join("two/data");
        write(&a, "a.txt", 10);
        write(&b, "b.txt", 10);
        let l = layout(&[a, b], LaneMode::Auto);
        assert_eq!(summary(&l.lanes), [entry("data", &["a.txt"]), entry("data-2", &["b.txt"])]);

        write(dir.path(), "solo.txt", 5);
        let l = layout(&[dir.path().join("solo.txt")], LaneMode::Auto);
        assert_eq!(summary(&l.lanes), [entry("my-files", &["solo.txt"])]);
    }

    #[test]
    fn a_dropped_mini_agi_folder_next_to_another_root_is_just_a_lane() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("ds");
        let b = dir.path().join("other");
        write(&a, "train/stories/s.txt", 10);
        write(&b, "o.txt", 10);
        let l = layout(&[a, b], LaneMode::Auto);
        assert!(!l.in_place(), "rule 4 applies to several roots");
        assert_eq!(summary(&l.lanes), [entry("ds", &["train/stories/s.txt"]), entry("other", &["o.txt"])]);
    }

    #[test]
    fn mode_overrides() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("proj");
        write(&root, "a/one.txt", 100);
        write(&root, "b/two.txt", 100);
        write(&root, "loose.txt", 1000); // far above 20 %: Auto would pick one lane

        let auto = layout(std::slice::from_ref(&root), LaneMode::Auto);
        assert_eq!(lane_names(&auto), ["proj"]);
        let per_folder = layout(std::slice::from_ref(&root), LaneMode::PerFolder);
        assert_eq!(lane_names(&per_folder), ["a", "b", "misc"]);
        let one = layout(std::slice::from_ref(&root), LaneMode::OneLane);
        assert_eq!(lane_names(&one), ["proj"]);

        // OneLane / PerFolder ignore a train/ + val/ layout.
        let ds = dir.path().join("ds");
        write(&ds, "train/stories/s.txt", 10);
        write(&ds, "val/stories/v.txt", 10);
        let forced = layout(std::slice::from_ref(&ds), LaneMode::OneLane);
        assert!(!forced.in_place());
        assert_eq!(summary(&forced.lanes), [entry("ds", &["train/stories/s.txt", "val/stories/v.txt"])]);

        // OneLane over several roots merges them, keeping the paths apart.
        let (x, y) = (dir.path().join("x"), dir.path().join("y"));
        write(&x, "same.txt", 5);
        write(&y, "same.txt", 5);
        let merged = layout(&[x, y], LaneMode::OneLane);
        assert_eq!(summary(&merged.lanes), [entry("my-files", &["x/same.txt", "y/same.txt"])]);
    }

    #[test]
    fn nothing_found_gives_an_empty_layout_with_a_warning() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "pic.png", 5);
        let l = layout(&[dir.path().to_path_buf()], LaneMode::Auto);
        assert!(l.lanes.is_empty());
        assert!(l.warnings.iter().any(|w| w.contains("No text files")));
    }

    #[test]
    fn lanes_sorted_and_files_keep_their_sizes() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("r");
        write(&root, "zeta/a.txt", 7);
        write(&root, "alpha/b.txt", 9);
        let l = layout(&[root], LaneMode::Auto);
        assert_eq!(lane_names(&l), ["alpha", "zeta"]);
        assert_eq!((l.lanes[0].bytes(), l.lanes[1].bytes(), l.train_bytes()), (9, 7, 16));
        assert!(l.lanes[0].files[0].path.ends_with("alpha/b.txt"));
    }
}
