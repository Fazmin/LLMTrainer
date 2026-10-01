//! A quick look at what the user is dragging over the window, before any real scan.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use minagi_types::{Count, PathProbe};
use walkdir::WalkDir;

use crate::scan::is_ignored;

/// The whole probe takes at most this long, however many paths and files there are.
pub const PROBE_BUDGET: Duration = Duration::from_millis(300);

/// [`probe_paths_within`] with the standard 300 ms budget.
pub fn probe_paths(paths: &[PathBuf]) -> Vec<PathProbe> {
    probe_paths_within(paths, PROBE_BUDGET)
}

/// For each path: is it a folder, roughly how many files does it hold, and is it already shaped like a mini-AGI
/// dataset (`train/` inside)?
///
/// The folders share `budget` evenly. When a walk runs out of time its count is a lower bound and `exact` is false.
/// Folders and files that the scan would skip (hidden, `node_modules`, ...) are not counted; nothing is opened.
pub fn probe_paths_within(paths: &[PathBuf], budget: Duration) -> Vec<PathProbe> {
    let started = Instant::now();
    let n = paths.len().max(1) as u32;
    paths
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let deadline = started + budget * (i as u32 + 1) / n;
            probe_one(path, deadline)
        })
        .collect()
}

fn probe_one(path: &Path, deadline: Instant) -> PathProbe {
    let shown = path.display().to_string();
    let Ok(meta) = std::fs::metadata(path) else {
        return PathProbe {
            path: shown,
            is_dir: false,
            approx_files: Count(0),
            exact: true,
            looks_like_mini_agi_layout: false,
        };
    };
    if !meta.is_dir() {
        return PathProbe {
            path: shown,
            is_dir: false,
            approx_files: Count(1),
            exact: true,
            looks_like_mini_agi_layout: false,
        };
    }
    let looks_like_layout = std::fs::read_dir(path.join("train")).is_ok_and(|mut entries| entries.next().is_some());
    let (mut files, mut exact) = (0u64, true);
    for entry in WalkDir::new(path).follow_links(false).into_iter().filter_entry(|e| !is_ignored(e)).flatten() {
        if Instant::now() >= deadline {
            exact = false;
            break;
        }
        if entry.file_type().is_file() {
            files += 1;
        }
    }
    PathProbe {
        path: shown,
        is_dir: true,
        approx_files: Count(files),
        exact,
        looks_like_mini_agi_layout: looks_like_layout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    fn touch(root: &Path, rel: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "x").unwrap();
    }

    #[test]
    fn counts_files_and_detects_the_layout() {
        let dir = TempDir::new().unwrap();
        let plain = dir.path().join("plain");
        for i in 0..7 {
            touch(&plain, &format!("sub/f{i}.txt"));
        }
        touch(&plain, ".hidden/skip.txt");
        touch(&plain, "node_modules/skip.js");
        touch(&plain, ".DS_Store");
        let layout = dir.path().join("layout");
        touch(&layout, "train/stories/a.txt");
        touch(&layout, "val/stories/b.txt");
        let loose = dir.path().join("loose.txt");
        fs::write(&loose, "hello").unwrap();
        let missing = dir.path().join("missing");

        let probes = probe_paths(&[plain.clone(), layout.clone(), loose.clone(), missing.clone()]);
        assert_eq!(probes.len(), 4);
        assert_eq!(
            probes[0],
            PathProbe {
                path: plain.display().to_string(),
                is_dir: true,
                approx_files: Count(7),
                exact: true,
                looks_like_mini_agi_layout: false
            }
        );
        assert_eq!(
            (probes[1].is_dir, probes[1].approx_files, probes[1].looks_like_mini_agi_layout),
            (true, Count(2), true)
        );
        assert_eq!((probes[2].is_dir, probes[2].approx_files, probes[2].exact), (false, Count(1), true));
        assert_eq!((probes[3].is_dir, probes[3].approx_files), (false, Count(0)));
    }

    #[test]
    fn an_empty_train_folder_is_not_a_layout() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("train")).unwrap();
        assert!(!probe_paths(&[dir.path().to_path_buf()])[0].looks_like_mini_agi_layout);
    }

    #[test]
    fn walks_stop_at_the_time_budget() {
        let dir = TempDir::new().unwrap();
        for i in 0..200 {
            touch(dir.path(), &format!("d{}/f{i}.txt", i % 5));
        }
        let probe = &probe_paths_within(&[dir.path().to_path_buf()], Duration::ZERO)[0];
        assert!(!probe.exact);
        assert!(probe.approx_files.0 < 200);
        let probe = &probe_paths_within(&[dir.path().to_path_buf()], Duration::from_secs(10))[0];
        assert!(probe.exact);
        assert_eq!(probe.approx_files, Count(200));
    }

    #[test]
    fn the_standard_probe_returns_within_its_budget() {
        let dir = TempDir::new().unwrap();
        for i in 0..300 {
            touch(dir.path(), &format!("d{}/f{i}.txt", i % 10));
        }
        let started = Instant::now();
        let probes = probe_paths(&vec![dir.path().to_path_buf(); 3]);
        assert!(started.elapsed() < PROBE_BUDGET + Duration::from_millis(250), "{:?}", started.elapsed());
        assert_eq!(probes.len(), 3);
    }
}
