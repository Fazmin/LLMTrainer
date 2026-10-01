//! The training reader.
//!
//! Lanes (top-level folders under `train/`) are read round-robin with equal weight, whatever their size. Each visit
//! to a lane picks a random file and a random offset in it, then takes consecutive chunks from there. Every step
//! yields the *window* of text that ends at the next chunk: up to `ctx` tokens, so the model re-reads the recent past
//! and is scored on all of it. This follows mini-AGI's reader.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::error::{TextError, TextResult};
use crate::tokenizer::ByteTokenizer;

/// Files bigger than this are not read whole: a random window of [`StreamConfig::large_window_bytes`] is used.
const DEFAULT_MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_LARGE_WINDOW_BYTES: u64 = 64 * 1024 * 1024;
const CACHE_FILES: usize = 4;

#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Tokens read per step.
    pub chunk: usize,
    /// Tokens in a visit to a lane (rounded up to whole chunks, and to at least one context window).
    pub passage: usize,
    pub shuffle_seed: u32,
    pub max_file_bytes: u64,
    pub large_window_bytes: u64,
}

impl StreamConfig {
    pub fn new(chunk: usize, passage: usize, shuffle_seed: u32) -> Self {
        Self {
            chunk,
            passage,
            shuffle_seed,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            large_window_bytes: DEFAULT_LARGE_WINDOW_BYTES,
        }
    }
}

/// One training step's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    /// Input tokens: the window ending at this step's chunk.
    pub x: Vec<u16>,
    /// Targets: `x` shifted by one, so `y[i]` is the token after `x[i]`.
    pub y: Vec<u16>,
    /// Index of the lane this came from.
    pub lane: usize,
    /// First step of a visit: the engine starts a fresh text (new expert vote, empty caches).
    pub first_in_visit: bool,
    /// Tokens of new text read this step (the chunk size, less at the end of a file).
    pub new_tokens: usize,
}

struct LaneFiles {
    name: String,
    files: Vec<PathBuf>,
}

struct Visit {
    tokens: Arc<Vec<u16>>,
    lane: usize,
    /// Index into `tokens` where this visit's text starts.
    start: usize,
    step: usize,
    steps: usize,
}

pub struct TrainStream {
    lanes: Vec<LaneFiles>,
    cfg: StreamConfig,
    rng: ChaCha8Rng,
    next_lane: usize,
    visit: Option<Visit>,
    cache: Vec<(PathBuf, Arc<Vec<u16>>)>,
    tokenizer: ByteTokenizer,
    /// For tests and diagnostics: how many visits have started.
    visits_started: u64,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let ty = e.file_type()?;
        if ty.is_dir() {
            walk(&e.path(), out)?;
        } else if ty.is_file() {
            out.push(e.path());
        }
    }
    Ok(())
}

impl TrainStream {
    /// Open `root/train/<lane>/**`. `enabled` limits which lanes are read (all when `None`). `chars_before` is how much
    /// text earlier sessions already read, so a resumed run does not replay the same sequence.
    pub fn open(root: &Path, enabled: Option<&[String]>, cfg: StreamConfig, chars_before: u64) -> TextResult<Self> {
        if cfg.chunk == 0 {
            return Err(TextError::Invalid("a step must read at least one character".into()));
        }
        let train = root.join("train");
        let mut lanes = Vec::new();
        let mut dirs: Vec<_> = std::fs::read_dir(&train)
            .map_err(|e| TextError::io(&train, e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
            .collect();
        dirs.sort_by_key(|e| e.file_name());
        for d in dirs {
            let name = d.file_name().to_string_lossy().to_string();
            if enabled.is_some_and(|en| !en.contains(&name)) {
                continue;
            }
            let mut files = Vec::new();
            walk(&d.path(), &mut files).map_err(|e| TextError::io(d.path(), e))?;
            if !files.is_empty() {
                lanes.push(LaneFiles { name, files });
            }
        }
        // Files placed directly in train/ form one lane named after the folder.
        if enabled.is_none() {
            let loose: Vec<PathBuf> = std::fs::read_dir(&train)
                .map_err(|e| TextError::io(&train, e))?
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .map(|e| e.path())
                .collect();
            if !loose.is_empty() {
                let mut loose = loose;
                loose.sort();
                lanes.push(LaneFiles { name: "train".into(), files: loose });
            }
        }
        if lanes.is_empty() {
            return Err(TextError::NoText(train.display().to_string()));
        }
        let seed = (cfg.shuffle_seed as u64) ^ chars_before.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Ok(Self {
            lanes,
            cfg,
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_lane: 0,
            visit: None,
            cache: Vec::new(),
            tokenizer: ByteTokenizer::new(),
            visits_started: 0,
        })
    }

    pub fn lane_names(&self) -> Vec<String> {
        self.lanes.iter().map(|l| l.name.clone()).collect()
    }

    pub fn visits_started(&self) -> u64 {
        self.visits_started
    }

    fn load(&mut self, path: &Path) -> TextResult<Arc<Vec<u16>>> {
        let size = std::fs::metadata(path).map_err(|e| TextError::io(path, e))?.len();
        if size <= self.cfg.max_file_bytes {
            if let Some((_, t)) = self.cache.iter().find(|(p, _)| p == path) {
                return Ok(t.clone());
            }
            let bytes = std::fs::read(path).map_err(|e| TextError::io(path, e))?;
            let tokens = Arc::new(self.tokenizer.encode(&bytes));
            if self.cache.len() >= CACHE_FILES {
                self.cache.remove(0);
            }
            self.cache.push((path.to_path_buf(), tokens.clone()));
            return Ok(tokens);
        }
        // A huge file: read only a window of it, from a random offset, starting on a character boundary.
        let window = self.cfg.large_window_bytes.min(size);
        let offset = self.rng.random_range(0..=(size - window));
        let mut f = std::fs::File::open(path).map_err(|e| TextError::io(path, e))?;
        f.seek(SeekFrom::Start(offset)).map_err(|e| TextError::io(path, e))?;
        let mut bytes = vec![0u8; window as usize];
        f.read_exact(&mut bytes).map_err(|e| TextError::io(path, e))?;
        let skip = bytes.iter().take(4).take_while(|b| (**b & 0xC0) == 0x80).count();
        Ok(Arc::new(self.tokenizer.encode(&bytes[skip..])))
    }

    fn start_visit(&mut self, ctx: usize) -> TextResult<()> {
        let chunk = self.cfg.chunk;
        let steps = ctx.div_ceil(chunk).max(self.cfg.passage.div_ceil(chunk)).max(1);
        let span = steps * chunk;
        for _ in 0..self.lanes.len() {
            let lane = self.next_lane;
            self.next_lane = (self.next_lane + 1) % self.lanes.len();
            let n = self.lanes[lane].files.len();
            // A few random tries, then a sweep, so one tiny file does not hide a usable one.
            let mut order: Vec<usize> = (0..6).map(|_| self.rng.random_range(0..n)).collect();
            order.extend(0..n);
            for fi in order {
                let path = self.lanes[lane].files[fi].clone();
                let tokens = self.load(&path)?;
                if tokens.len() < chunk + 1 {
                    continue;
                }
                let start = if tokens.len() > span + 1 { self.rng.random_range(0..tokens.len() - span - 1) } else { 0 };
                self.visit = Some(Visit { tokens, lane, start, step: 0, steps });
                self.visits_started += 1;
                return Ok(());
            }
        }
        Err(TextError::TooShort { chunk })
    }

    /// The next step's text. `ctx` is the current context window; it can grow during training.
    pub fn next_batch(&mut self, ctx: usize) -> TextResult<Batch> {
        let chunk = self.cfg.chunk;
        let ctx = ctx.max(chunk);
        loop {
            let need_new = match &self.visit {
                None => true,
                Some(v) => v.step >= v.steps || v.start + (v.step + 1) * chunk >= v.tokens.len(),
            };
            if need_new {
                self.start_visit(ctx)?;
            }
            let v = self.visit.as_mut().expect("a visit was just started");
            let prev_end = v.start + v.step * chunk;
            let end = (v.start + (v.step + 1) * chunk).min(v.tokens.len() - 1);
            if end <= prev_end {
                // The file ended exactly here; start over elsewhere.
                v.step = v.steps;
                continue;
            }
            let window_start = end.saturating_sub(ctx);
            let batch = Batch {
                x: v.tokens[window_start..end].to_vec(),
                y: v.tokens[window_start + 1..=end].to_vec(),
                lane: v.lane,
                first_in_visit: v.step == 0,
                new_tokens: end - prev_end,
            };
            v.step += 1;
            return Ok(batch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Files whose bytes are a counter, so a window can be checked against what it must contain.
    fn counter_file(len: usize, offset: u8) -> Vec<u8> {
        (0..len).map(|i| (((i % 90) as u8) + 33).wrapping_add(offset % 3)).collect()
    }

    fn tree(lanes: &[(&str, usize, usize)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, files, len) in lanes {
            let d = dir.path().join("train").join(name);
            std::fs::create_dir_all(&d).unwrap();
            for i in 0..*files {
                std::fs::write(d.join(format!("part-{i:04}.txt")), counter_file(*len, i as u8)).unwrap();
            }
        }
        dir
    }

    fn cfg(chunk: usize, passage: usize) -> StreamConfig {
        StreamConfig::new(chunk, passage, 7)
    }

    #[test]
    fn windows_end_at_the_chunk_and_targets_are_shifted_by_one() {
        let dir = tree(&[("stories", 2, 5000)]);
        let mut s = TrainStream::open(dir.path(), None, cfg(64, 256), 0).unwrap();
        let first = s.next_batch(64).unwrap();
        assert_eq!(first.x.len(), first.y.len());
        assert!(first.first_in_visit);
        assert_eq!(first.new_tokens, 64);
        // y is x shifted by one, so y[i] == x[i + 1].
        assert_eq!(&first.y[..first.y.len() - 1], &first.x[1..]);
        // The next step in the same visit continues right after.
        let second = s.next_batch(64).unwrap();
        assert!(!second.first_in_visit);
        // With ctx == chunk the windows tile the text: this step starts where the last target left off.
        assert_eq!(second.x[0], *first.y.last().unwrap());
        assert_eq!(second.lane, first.lane);
    }

    #[test]
    fn a_context_longer_than_the_chunk_includes_history() {
        let dir = tree(&[("stories", 1, 20_000)]);
        let mut s = TrainStream::open(dir.path(), None, cfg(64, 64), 0).unwrap();
        // With ctx = 256 and chunk = 64, a visit lasts 4 steps and windows grow toward 256 tokens.
        let sizes: Vec<usize> = (0..4).map(|_| s.next_batch(256).unwrap().x.len()).collect();
        assert!(sizes.iter().all(|&n| (64..=256).contains(&n)), "{sizes:?}");
        assert_eq!(*sizes.last().unwrap(), 256, "by the last chunk the window is a full context: {sizes:?}");
        assert_eq!(s.visits_started(), 1);
        // The fifth step starts a new visit.
        let fifth = s.next_batch(256).unwrap();
        assert!(fifth.first_in_visit);
        assert_eq!(s.visits_started(), 2);
    }

    #[test]
    fn lanes_take_turns_equally_whatever_their_size() {
        let dir = tree(&[("a", 1, 50_000), ("b", 8, 3_000), ("c", 2, 8_000)]);
        let mut s = TrainStream::open(dir.path(), None, cfg(32, 32), 0).unwrap();
        assert_eq!(s.lane_names(), vec!["a", "b", "c"]);
        let lanes: Vec<usize> = (0..9).map(|_| s.next_batch(32).unwrap().lane).collect();
        assert_eq!(lanes, vec![0, 1, 2, 0, 1, 2, 0, 1, 2], "one visit (one step) per lane, round robin");
    }

    #[test]
    fn same_seed_same_text_and_resume_does_not_replay() {
        let dir = tree(&[("a", 6, 9000), ("b", 6, 9000)]);
        let run = |seed: u32, before: u64| {
            let mut s = TrainStream::open(dir.path(), None, StreamConfig::new(64, 128, seed), before).unwrap();
            (0..12).map(|_| s.next_batch(128).unwrap().x).collect::<Vec<_>>()
        };
        assert_eq!(run(1, 0), run(1, 0), "deterministic");
        assert_ne!(run(1, 0), run(2, 0), "a different shuffle seed reads different passages");
        assert_ne!(run(1, 0), run(1, 123_456), "a resumed run starts somewhere new");
    }

    #[test]
    fn disabled_lanes_are_not_read() {
        let dir = tree(&[("stories", 2, 4000), ("code", 2, 4000)]);
        let only = vec!["code".to_string()];
        let mut s = TrainStream::open(dir.path(), Some(&only), cfg(32, 32), 0).unwrap();
        assert_eq!(s.lane_names(), vec!["code"]);
        assert!((0..5).all(|_| s.next_batch(32).unwrap().lane == 0));
    }

    #[test]
    fn tiny_files_are_skipped_and_all_tiny_is_an_error() {
        let dir = tree(&[("a", 1, 10)]);
        let err = TrainStream::open(dir.path(), None, cfg(64, 64), 0).unwrap().next_batch(64).unwrap_err();
        assert!(matches!(err, TextError::TooShort { chunk: 64 }), "{err}");
        // A good file next to a tiny one is found.
        std::fs::write(dir.path().join("train/a/big.txt"), counter_file(4000, 0)).unwrap();
        let mut s = TrainStream::open(dir.path(), None, cfg(64, 64), 0).unwrap();
        assert!(s.next_batch(64).is_ok());
    }

    #[test]
    fn a_dataset_without_text_is_a_plain_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("train").join("empty")).unwrap();
        assert!(matches!(TrainStream::open(dir.path(), None, cfg(8, 8), 0), Err(TextError::NoText(_))));
        assert!(matches!(TrainStream::open(&dir.path().join("nope"), None, cfg(8, 8), 0), Err(TextError::Io { .. })));
    }

    #[test]
    fn markers_in_files_become_single_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().join("train").join("chat");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("a.txt"), "<user>\nhi\n</user>\n<bot>\nhello\n</bot>\n".repeat(50)).unwrap();
        let mut s = TrainStream::open(dir.path(), None, cfg(40, 40), 0).unwrap();
        let b = s.next_batch(40).unwrap();
        assert!(b.x.iter().any(|&t| t >= 256), "markers are tokens: {:?}", &b.x[..10]);
    }

    #[test]
    fn huge_files_are_read_by_window_not_whole() {
        let dir = tree(&[("big", 1, 40_000)]);
        let mut c = cfg(64, 64);
        c.max_file_bytes = 10_000; // treat the 40 KB file as huge
        c.large_window_bytes = 8_000;
        let mut s = TrainStream::open(dir.path(), None, c, 0).unwrap();
        for _ in 0..20 {
            let b = s.next_batch(64).unwrap();
            assert_eq!(b.x.len(), b.y.len());
        }
        assert!(s.cache.is_empty(), "windows are not cached as whole files");
    }

    #[test]
    fn the_end_of_a_file_never_yields_an_empty_step() {
        // A file just over one chunk long: every batch is non-empty even as visits hit the end.
        let dir = tree(&[("a", 1, 100)]);
        let mut s = TrainStream::open(dir.path(), None, cfg(64, 640), 0).unwrap();
        for _ in 0..30 {
            let b = s.next_batch(64).unwrap();
            assert!(!b.x.is_empty() && b.new_tokens > 0);
        }
    }
}
