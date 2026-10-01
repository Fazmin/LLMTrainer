//! The whole local path a dropped folder takes: scan, lanes, split, materialize, prompts, preview.

use std::fs;
use std::path::{Path, PathBuf};

use minagi_data::{
    CancelFlag, LayoutKind, PrevIndex, ScanOptions, detect_lanes, materialize, plan_split, preview_text, probe_paths,
    scan_paths, suggest_prompts,
};
use minagi_types::{LaneMode, SplitConfig, SplitMode};
use tempfile::TempDir;

fn write(root: &Path, rel: &str, content: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn essay(i: usize) -> String {
    (0..40)
        .map(|l| {
            format!("Essay {i}, paragraph {l}: it was a quiet evening in the harbour town and nobody was in a hurry.\n")
        })
        .collect()
}

/// A library folder: two lanes of many files, one big single-file lane, some clutter, and a few loose files.
fn library(root: &Path) {
    for i in 0..60 {
        write(root, &format!("Essays/2024/essay{i:02}.txt"), &essay(i));
    }
    for i in 0..25 {
        write(
            root,
            &format!("Poems/poem{i:02}.md"),
            &format!("# Poem {i}\n\nroses are red, violets are blue, number {i} is a poem for you\n").repeat(30),
        );
    }
    write(
        root,
        "Novel/novel.txt",
        &(0..3000)
            .map(|l| format!("Chapter text, line {l}, with ünïcode ☃ and plenty of ordinary words.\n"))
            .collect::<String>(),
    );
    write(root, "readme.txt", "a short loose file\n");
    write(root, "Essays/cover.png", "not text");
    write(root, ".git/config", "[core]\n");
    write(root, "node_modules/x/index.js", "module.exports = 1\n");
}

fn tree(root: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((path.strip_prefix(root).unwrap().display().to_string(), fs::metadata(&path).unwrap().len()));
            }
        }
    }
    out.sort();
    out
}

fn prepare(src: &Path, dest: &Path, cfg: &SplitConfig) -> minagi_data::MaterializeReport {
    let scan =
        scan_paths(&[src.to_path_buf()], &ScanOptions::default(), &PrevIndex::new(), &CancelFlag::new(), &mut |_| {});
    let layout = detect_lanes(&scan, LaneMode::Auto);
    let plan = plan_split(&layout, cfg).unwrap();
    materialize(&plan, dest, &CancelFlag::new(), &mut |_| {}).unwrap()
}

#[test]
fn a_folder_becomes_a_train_and_val_dataset_the_engine_can_read() {
    let dir = TempDir::new().unwrap();
    let src = dir.path().join("library");
    library(&src);

    let probe = &probe_paths(std::slice::from_ref(&src))[0];
    // 60 essays, 25 poems, the novel, the readme and the png; .git and node_modules are not counted.
    assert!(probe.is_dir && probe.exact && !probe.looks_like_mini_agi_layout);
    assert_eq!(probe.approx_files.0, 88);

    let scan = scan_paths(
        std::slice::from_ref(&src),
        &ScanOptions::default(),
        &PrevIndex::new(),
        &CancelFlag::new(),
        &mut |_| {},
    );
    assert_eq!(scan.files.len(), 60 + 25 + 1 + 1, "the png, .git and node_modules are not text");
    assert_eq!(scan.skipped.binary, 1);

    let layout = detect_lanes(&scan, LaneMode::Auto);
    assert_eq!(layout.kind, LayoutKind::Folders);
    let names: Vec<&str> = layout.lanes.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["essays", "misc", "novel", "poems"]);

    let plan = plan_split(&layout, &SplitConfig::default()).unwrap();
    let dest = dir.path().join("dataset");
    let report = materialize(&plan, &dest, &CancelFlag::new(), &mut |_| {}).unwrap();

    // The engine's shape: train/<lane>/** and val/<domain>/**.
    for lane in ["essays", "novel", "poems"] {
        assert!(dest.join("train").join(lane).is_dir(), "train/{lane}");
        assert!(dest.join("val").join(lane).is_dir(), "val/{lane}");
    }
    assert!(dest.join("manifest.json").is_file());
    let files = tree(&dest);
    assert!(files.iter().any(|(p, _)| p.starts_with("train/essays/2024/essay")), "relative paths are kept");
    assert!(
        files.iter().any(|(p, _)| p == "train/novel/novel.head.txt")
            && files.iter().any(|(p, _)| p == "val/novel/novel.tail.txt")
    );
    assert!(!files.iter().any(|(p, _)| p.contains("cover.png") || p.contains(".git") || p.contains("node_modules")));

    // About 2 % of each many-file lane is held out, as whole files, and nothing is lost.
    let bytes_in = |prefix: &str| files.iter().filter(|(p, _)| p.starts_with(prefix)).map(|(_, n)| n).sum::<u64>();
    for lane in ["essays", "poems"] {
        let (train, val) = (bytes_in(&format!("train/{lane}/")), bytes_in(&format!("val/{lane}/")));
        let share = val as f64 / (train + val) as f64;
        assert!(val > 0 && share < 0.09, "{lane}: {share:.3} held out");
    }
    let src_bytes: u64 = scan.files.iter().map(|f| f.size).sum();
    assert_eq!(report.train_bytes + report.val_bytes, src_bytes);
    assert_eq!(report.files as usize, files.len() - 1, "every file but the manifest is accounted for");
    assert!(report.reflinked_files + report.hardlinked_files > 80, "whole files are linked, not copied: {report:?}");
    assert!(report.bytes_copied > 0, "the split novel is a managed copy");

    // The lane table for the UI.
    let infos = plan.lane_infos();
    assert_eq!(infos.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(), ["essays", "misc", "novel", "poems"]);
    assert!(infos.iter().all(|i| i.enabled && i.val_bytes.0 <= i.train_bytes.0));

    // Prompts and previews work on the prepared directory.
    let prompts = suggest_prompts(&dest);
    assert_eq!(prompts.len(), 4);
    let essays = prompts.iter().find(|(l, _)| l == "essays").unwrap();
    assert!(essays.1.starts_with("Essay "), "{essays:?}");
    let preview = preview_text(&dest, "novel", 150, 5).unwrap();
    assert!(preview.text.chars().count() <= 150 && preview.text.contains("Chapter text"));
}

#[test]
fn the_same_folder_and_seed_always_give_the_same_dataset() {
    let dir = TempDir::new().unwrap();
    let src = dir.path().join("library");
    library(&src);
    let cfg = SplitConfig { mode: SplitMode::Auto, pct: 3.0, seed: 99 };
    prepare(&src, &dir.path().join("a"), &cfg);
    prepare(&src, &dir.path().join("b"), &cfg);
    assert_eq!(tree(&dir.path().join("a")), tree(&dir.path().join("b")));
    // Another seed holds out different files.
    prepare(&src, &dir.path().join("c"), &SplitConfig { seed: 100, ..cfg });
    assert_ne!(tree(&dir.path().join("a")), tree(&dir.path().join("c")));
}

#[test]
fn a_mini_agi_folder_is_used_in_place_or_re_split() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("minidata");
    for i in 0..10 {
        write(
            &root,
            &format!("train/stories/s{i}.txt"),
            &format!("Once upon a time number {i}, a small story was told.\n").repeat(20),
        );
    }
    write(&root, "train/code/a.py", "def f(x):\n    return x + 1\n");
    write(&root, "val/stories/v.txt", "A held-out story that the model must not see during training.\n");

    let scan = scan_paths(
        std::slice::from_ref(&root),
        &ScanOptions::default(),
        &PrevIndex::new(),
        &CancelFlag::new(),
        &mut |_| {},
    );
    let layout = detect_lanes(&scan, LaneMode::Auto);
    assert!(layout.in_place() && layout.provides_val());

    // Folder mode: nothing is copied; the dataset is the folder itself.
    let in_place = plan_split(&layout, &SplitConfig { mode: SplitMode::Folder, ..SplitConfig::default() }).unwrap();
    let before = tree(&root);
    let never = dir.path().join("never-created");
    let report = materialize(&in_place, &never, &CancelFlag::new(), &mut |_| {}).unwrap();
    assert!(report.in_place && report.dest == root && !never.exists());
    assert_eq!(tree(&root), before, "the source folder is untouched");
    assert_eq!(report.val_bytes, 62);
    // Prompts and previews read straight from the folder.
    let prompts = suggest_prompts(&root);
    assert_eq!(prompts.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(), ["code", "stories"]);
    assert!(prompts[0].1.starts_with("def merge_sorted"));
    assert!(preview_text(&root, "stories", 80, 1).unwrap().text.contains("Once upon a time"));

    // Auto mode re-splits the training files and builds a managed copy next to it.
    let resplit = plan_split(&layout, &SplitConfig::default()).unwrap();
    assert!(resplit.in_place.is_none());
    let dest = dir.path().join("managed");
    let report = materialize(&resplit, &dest, &CancelFlag::new(), &mut |_| {}).unwrap();
    assert!(!report.in_place && dest.join("train/stories").is_dir());
}
