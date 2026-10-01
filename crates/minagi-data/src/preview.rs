//! A random passage from a lane, for the "peek inside" panel.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use minagi_types::{Count, TextPreview};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use walkdir::WalkDir;

use crate::error::{DataError, DataResult};
use crate::lanes::LaneLayout;
use crate::scan::is_ignored;

/// The longest passage that can be asked for, in characters.
pub const MAX_PREVIEW_CHARS: usize = 100_000;
/// How far past the chosen offset the passage may move to start at the beginning of a line, in bytes.
const LINE_START_REACH: usize = 256;
/// Draws before giving up on finding readable text.
const ATTEMPTS: usize = 8;

/// A file of a lane: where it is and its path relative to the lane.
#[derive(Debug, Clone)]
pub struct PreviewFile {
    pub path: PathBuf,
    pub rel_path: String,
    pub size: u64,
}

/// A random passage of about `n_chars` characters from `<root>/train/<lane>`.
///
/// The file is drawn in proportion to its size, the offset uniformly, and the passage starts at the beginning of a
/// line when one is close by, always on a character boundary. The same arguments on the same files give the same
/// passage. A lane (or file) shorter than `n_chars` is returned whole.
pub fn preview_text(root: &Path, lane: &str, n_chars: usize, seed: u32) -> DataResult<TextPreview> {
    let dir = root.join("train").join(lane);
    if !dir.is_dir() {
        return Err(DataError::NotFound(format!("lane \"{lane}\"")));
    }
    let mut files: Vec<PreviewFile> = WalkDir::new(&dir)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| !is_ignored(e))
        .flatten()
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let size = e.metadata().ok()?.len();
            let rel = e.path().strip_prefix(&dir).ok()?;
            Some(PreviewFile { path: e.path().to_path_buf(), rel_path: crate::util::rel_to_string(rel), size })
        })
        .collect();
    files.retain(|f| f.size > 0);
    preview_files(&files, lane, n_chars, seed)
}

/// Like [`preview_text`] for a lane that has not been prepared yet, straight from a detected [`LaneLayout`].
pub fn preview_layout_lane(layout: &LaneLayout, lane: &str, n_chars: usize, seed: u32) -> DataResult<TextPreview> {
    let found = layout.lane(lane).ok_or_else(|| DataError::NotFound(format!("lane \"{lane}\"")))?;
    let mut files: Vec<PreviewFile> = found
        .files
        .iter()
        .map(|f| PreviewFile { path: f.path.clone(), rel_path: f.rel_path.clone(), size: f.size })
        .collect();
    files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    preview_files(&files, lane, n_chars, seed)
}

/// The shared core: draw a file by size, then a passage in it.
pub fn preview_files(files: &[PreviewFile], lane: &str, n_chars: usize, seed: u32) -> DataResult<TextPreview> {
    let total: u64 = files.iter().map(|f| f.size).sum();
    if total == 0 {
        return Err(DataError::NotFound(format!("text in lane \"{lane}\"")));
    }
    let n_chars = n_chars.clamp(1, MAX_PREVIEW_CHARS);
    let mut rng = ChaCha8Rng::seed_from_u64(u64::from(seed));
    for _ in 0..ATTEMPTS {
        let mut pick = rng.random_range(0..total);
        let file = files
            .iter()
            .find(|f| {
                if pick < f.size {
                    true
                } else {
                    pick -= f.size;
                    false
                }
            })
            .unwrap_or(&files[files.len() - 1]);
        // Leave room after the offset for a full passage; a file shorter than that is read from its start.
        let room = file.size.saturating_sub(max_bytes(n_chars));
        let offset = if room == 0 { 0 } else { rng.random_range(0..=room) };
        if let Some((start, text)) = read_passage(&file.path, offset, n_chars) {
            return Ok(TextPreview { file: file.rel_path.clone(), offset: Count(start), text });
        }
    }
    Err(DataError::NotFound(format!("readable text in lane \"{lane}\"")))
}

/// The most bytes a passage of `n_chars` characters plus its alignment can take.
fn max_bytes(n_chars: usize) -> u64 {
    // A character takes at most 4 bytes; the line-start search may skip up to LINE_START_REACH bytes first.
    (n_chars * 4 + LINE_START_REACH) as u64
}

/// Up to `n_chars` characters of UTF-8 text starting at (or just after) byte `offset`; also the byte offset where
/// the text really starts. `None` when nothing readable is found there.
fn read_passage(path: &Path, offset: u64, n_chars: usize) -> Option<(u64, String)> {
    // One byte of look-behind tells whether `offset` already is the start of a line.
    let from = offset.saturating_sub(1);
    let mut file = File::open(path).ok()?;
    file.seek(SeekFrom::Start(from)).ok()?;
    let mut buf = Vec::new();
    file.take(max_bytes(n_chars) + 8).read_to_end(&mut buf).ok()?;

    let mut at = (offset - from) as usize;
    // Start on a character boundary…
    while at < buf.len() && (buf[at] & 0b1100_0000) == 0b1000_0000 {
        at += 1;
    }
    // …and at the start of a line when one begins soon.
    let at_line_start = at == 0 || buf.get(at - 1) == Some(&b'\n');
    if !at_line_start {
        let reach = buf.len().min(at + LINE_START_REACH);
        if let Some(p) = buf[at..reach].iter().position(|&b| b == b'\n') {
            at += p + 1;
        }
    }
    let rest = buf.get(at..)?;
    // Only the valid UTF-8 up to the first break is used (a passage may end inside a character).
    let valid = match std::str::from_utf8(rest) {
        Ok(t) => t,
        Err(e) => std::str::from_utf8(&rest[..e.valid_up_to()]).ok()?,
    };
    let end = valid.char_indices().nth(n_chars).map_or(valid.len(), |(i, _)| i);
    let text = &valid[..end];
    (!text.trim().is_empty()).then(|| (from + at as u64, text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;

    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    fn lane_text(lines: usize) -> String {
        (0..lines)
            .map(|i| format!("Line {i}: naïve café résumé 日本語 🎉 and some ordinary words to read.\n"))
            .collect()
    }

    #[test]
    fn returns_a_real_passage_that_matches_the_file_at_the_offset() {
        let dir = TempDir::new().unwrap();
        let text = lane_text(2000);
        write(dir.path(), "train/books/one.txt", &text);
        for seed in 0..40 {
            let p = preview_text(dir.path(), "books", 300, seed).unwrap();
            assert_eq!(p.file, "one.txt");
            assert!(p.text.chars().count() <= 300 && p.text.chars().count() >= 250, "{} chars", p.text.chars().count());
            let off = p.offset.0 as usize;
            assert!(text.is_char_boundary(off), "offset {off} is on a boundary");
            assert!(text[off..].starts_with(&p.text), "the passage is what the file holds at its offset");
        }
    }

    #[test]
    fn passages_start_at_the_beginning_of_a_line_when_one_is_close() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/books/one.txt", &lane_text(2000));
        let starts_at_line = (0..40)
            .filter(|&seed| preview_text(dir.path(), "books", 200, seed).unwrap().text.starts_with("Line "))
            .count();
        assert!(starts_at_line >= 38, "{starts_at_line} of 40");
    }

    #[test]
    fn same_arguments_same_passage_other_seeds_vary() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/books/one.txt", &lane_text(3000));
        let a = preview_text(dir.path(), "books", 200, 7).unwrap();
        assert_eq!(a, preview_text(dir.path(), "books", 200, 7).unwrap());
        let offsets: HashSet<u64> =
            (0..30).map(|s| preview_text(dir.path(), "books", 200, s).unwrap().offset.0).collect();
        assert!(offsets.len() > 20, "{} distinct offsets", offsets.len());
    }

    #[test]
    fn files_are_drawn_in_proportion_to_their_size() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/mix/big.txt", &lane_text(3000));
        write(dir.path(), "train/mix/sub/small.txt", &lane_text(30));
        let big = (0..300).filter(|&s| preview_text(dir.path(), "mix", 100, s).unwrap().file == "big.txt").count();
        assert!(big > 270, "big file was drawn {big} of 300 times");
        let small =
            (0..3000).filter(|&s| preview_text(dir.path(), "mix", 100, s).unwrap().file == "sub/small.txt").count();
        assert!((10..100).contains(&small), "small file was drawn {small} of 3000 times");
    }

    #[test]
    fn short_files_are_returned_whole_and_passages_stay_valid_utf8() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/tiny/a.txt", "héllo wörld, a short text\n");
        for seed in 0..20 {
            let p = preview_text(dir.path(), "tiny", 500, seed).unwrap();
            assert_eq!(p.offset, Count(0), "seed {seed}");
            assert_eq!(p.text, "héllo wörld, a short text\n");
        }
        // Dense multi-byte text: every offset lands inside characters of 3 or 4 bytes.
        write(dir.path(), "train/cjk/a.txt", &"日本語の文章です。🎉".repeat(2000));
        for seed in 0..60 {
            let p = preview_text(dir.path(), "cjk", 77, seed).unwrap();
            assert_eq!(p.text.chars().count(), 77);
        }
    }

    #[test]
    fn invalid_bytes_end_a_passage_instead_of_corrupting_it() {
        let dir = TempDir::new().unwrap();
        let mut data = "valid text before the damage ".repeat(10).into_bytes();
        data.push(0xFF);
        data.extend_from_slice("valid text after the damage ".repeat(10).as_bytes());
        fs::create_dir_all(dir.path().join("train/dmg")).unwrap();
        fs::write(dir.path().join("train/dmg/a.txt"), &data).unwrap();
        for seed in 0..60 {
            let p = preview_text(dir.path(), "dmg", 400, seed).unwrap();
            assert!(!p.text.contains('\u{FFFD}') && !p.text.is_empty());
            let at = p.offset.0 as usize;
            assert_eq!(
                &data[at..at + p.text.len()],
                p.text.as_bytes(),
                "seed {seed}: the passage is the file's own bytes"
            );
        }
    }

    #[test]
    fn missing_or_empty_lanes_are_not_found() {
        let dir = TempDir::new().unwrap();
        assert!(matches!(preview_text(dir.path(), "nope", 100, 1), Err(DataError::NotFound(_))));
        fs::create_dir_all(dir.path().join("train/empty")).unwrap();
        assert!(matches!(preview_text(dir.path(), "empty", 100, 1), Err(DataError::NotFound(_))));
        write(dir.path(), "train/blank/a.txt", "   \n  \n");
        assert!(matches!(preview_text(dir.path(), "blank", 100, 1), Err(DataError::NotFound(_))));
        write(dir.path(), "train/hidden/.secret.txt", "text nobody should see in a preview\n");
        assert!(matches!(preview_text(dir.path(), "hidden", 100, 1), Err(DataError::NotFound(_))));
    }

    #[test]
    fn the_size_of_the_passage_is_bounded() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "train/l/a.txt", &lane_text(10));
        assert!(preview_text(dir.path(), "l", 0, 1).unwrap().text.chars().count() == 1);
        let all = preview_text(dir.path(), "l", usize::MAX, 1).unwrap();
        assert_eq!(all.text, lane_text(10));
    }

    #[test]
    fn a_lane_can_be_previewed_before_it_is_prepared() {
        use crate::lanes::{Lane, LaneFile, LayoutKind};
        let dir = TempDir::new().unwrap();
        write(dir.path(), "a.txt", &lane_text(500));
        let layout = LaneLayout {
            sources: vec![],
            kind: LayoutKind::Folders,
            lanes: vec![Lane {
                name: "raw".into(),
                display_name: "raw".into(),
                color_slot: 0,
                files: vec![LaneFile {
                    path: dir.path().join("a.txt"),
                    rel_path: "a.txt".into(),
                    size: lane_text(500).len() as u64,
                    mtime_ns: 0,
                }],
            }],
            val_lanes: vec![],
            warnings: vec![],
        };
        let p = preview_layout_lane(&layout, "raw", 120, 3).unwrap();
        assert_eq!(p.file, "a.txt");
        assert!(lane_text(500).contains(&p.text));
        assert!(matches!(preview_layout_lane(&layout, "other", 120, 3), Err(DataError::NotFound(_))));
    }
}
