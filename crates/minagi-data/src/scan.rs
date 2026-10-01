//! Folder scan and text sniffing.
//!
//! [`scan_paths`] walks the folders and files a user dropped, keeps the ones that look like UTF-8 text and counts the
//! rest by reason. Nothing is ever followed through a symbolic link, hidden and build folders are skipped, and files
//! with a binary extension are never even opened. Content is judged by [`sniff`] on the first 4 KiB.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use minagi_types::{SkippedExample, SkippedSummary};
use rayon::prelude::*;
use walkdir::{DirEntry, WalkDir};

use crate::util::{CancelFlag, PROGRESS_INTERVAL, Throttle, mtime_ns, rel_to_string};

/// How many leading bytes decide whether a file is text.
pub const SNIFF_BYTES: usize = 4096;
/// At most this many skipped files are listed as examples.
pub const MAX_SKIPPED_EXAMPLES: usize = 50;
/// Files at least this large get a warning (they are still accepted).
pub const LARGE_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Worker threads used to sniff files.
pub const SCAN_THREADS: usize = 4;

/// Folders that never hold training text (dot-folders are skipped on top of these).
pub const SKIP_DIR_NAMES: &[&str] =
    &[".git", "node_modules", "target", "build", "dist", "__pycache__", ".venv", "venv"];
/// Files that are never training text.
pub const SKIP_FILE_NAMES: &[&str] = &[".DS_Store", "Thumbs.db"];

/// Extensions of files that are certainly not text. Matched case-insensitively and never opened.
pub const BINARY_EXTENSIONS: &[&str] = &[
    // images
    "png",
    "jpg",
    "jpeg",
    "jfif",
    "gif",
    "webp",
    "avif",
    "bmp",
    "ico",
    "icns",
    "tif",
    "tiff",
    "heic",
    "heif",
    "psd",
    "svg",
    "raw",
    "cr2",
    "nef",
    // audio
    "mp3",
    "wav",
    "flac",
    "ogg",
    "oga",
    "opus",
    "m4a",
    "aac",
    "wma",
    "aiff",
    "aif",
    "mid",
    "midi",
    // video
    "mp4",
    "m4v",
    "mkv",
    "mov",
    "avi",
    "webm",
    "flv",
    "wmv",
    "mpg",
    "mpeg",
    "3gp",
    // archives and disk images
    "zip",
    "gz",
    "tgz",
    "tar",
    "xz",
    "bz2",
    "7z",
    "rar",
    "zst",
    "lz4",
    "lzma",
    "iso",
    "dmg",
    // executables and libraries
    "exe",
    "dll",
    "so",
    "dylib",
    "o",
    "a",
    "lib",
    "obj",
    "class",
    "jar",
    "pyc",
    "pyo",
    "bin",
    "wasm",
    "msi",
    "apk",
    // documents that are containers
    "pdf",
    "docx",
    "xlsx",
    "pptx",
    "odt",
    "ods",
    "odp",
    // machine-learning and numeric data
    "npy",
    "npz",
    "safetensors",
    "pt",
    "pth",
    "ckpt",
    "onnx",
    "gguf",
    "h5",
    "hdf5",
    "pkl",
    "pickle",
    // databases and columnar data
    "sqlite",
    "sqlite3",
    "db",
    "parquet",
    "arrow",
    "feather",
    "orc",
    // fonts
    "ttf",
    "otf",
    "woff",
    "woff2",
    "eot",
];

/// What the first bytes of a file say about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sniff {
    /// Valid UTF-8 text (a UTF-8 byte-order mark is allowed).
    Text,
    /// No content at all (or nothing but a byte-order mark).
    Empty,
    /// Not text: NUL bytes, invalid UTF-8, or mostly control characters.
    Binary,
    /// UTF-16 or UTF-32 text, recognised by its byte-order mark.
    Utf16,
}

/// Decide whether `head` (the first up to [`SNIFF_BYTES`] bytes of a file) is UTF-8 text.
///
/// The rules, in order: nothing at all is [`Sniff::Empty`]; a UTF-16/32 byte-order mark is [`Sniff::Utf16`] (checked
/// before NUL bytes, which UTF-16 text is full of); a UTF-8 byte-order mark is stripped; any NUL byte is
/// [`Sniff::Binary`]; invalid UTF-8 is binary, except that a code point cut off at the very end (the head is only a
/// prefix of the file) is tolerated; finally more than 90 % of the bytes must be printable (space to `~`, tab, CR, LF,
/// and every byte of a valid multi-byte sequence).
pub fn sniff(head: &[u8]) -> Sniff {
    if head.is_empty() {
        return Sniff::Empty;
    }
    // UTF-32 BE starts with two NULs, so the BOM checks come before the NUL check.
    const UTF16_UTF32_BOMS: [&[u8]; 4] =
        [&[0x00, 0x00, 0xFE, 0xFF], &[0xFF, 0xFE, 0x00, 0x00], &[0xFF, 0xFE], &[0xFE, 0xFF]];
    if UTF16_UTF32_BOMS.iter().any(|bom| head.starts_with(bom)) {
        return Sniff::Utf16;
    }
    let body = head.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(head);
    if body.is_empty() {
        return Sniff::Empty;
    }
    if body.contains(&0) {
        return Sniff::Binary;
    }
    let valid = match std::str::from_utf8(body) {
        Ok(s) => s.as_bytes(),
        // `error_len() == None` means "ran out of bytes inside a code point": the head just ends mid-character.
        Err(e) if e.error_len().is_none() => &body[..e.valid_up_to()],
        Err(_) => return Sniff::Binary,
    };
    if valid.is_empty() {
        return Sniff::Binary;
    }
    let printable = valid.iter().filter(|&&b| matches!(b, 0x20..=0x7E | b'\t' | b'\n' | b'\r') || b >= 0x80).count();
    if printable * 10 > valid.len() * 9 { Sniff::Text } else { Sniff::Binary }
}

/// The lower-case extension when it is on the binary denylist.
fn binary_extension(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    BINARY_EXTENSIONS.contains(&ext.as_str()).then_some(ext)
}

/// Knobs for [`scan_paths`].
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// Files larger than this are skipped as `too_large`. `None` accepts any size (with a warning above
    /// [`ScanOptions::warn_file_bytes`]).
    pub max_file_bytes: Option<u64>,
    /// Accepted files at least this large add a warning.
    pub warn_file_bytes: u64,
    /// How many skipped files are kept as examples.
    pub max_examples: usize,
    /// Threads used to sniff files in parallel.
    pub threads: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            max_file_bytes: None,
            warn_file_bytes: LARGE_FILE_BYTES,
            max_examples: MAX_SKIPPED_EXAMPLES,
            threads: SCAN_THREADS,
        }
    }
}

/// What a previous scan learned about one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrevEntry {
    pub size: u64,
    pub mtime_ns: u64,
    pub verdict: Sniff,
}

impl PrevEntry {
    pub fn was_text(&self) -> bool {
        self.verdict == Sniff::Text
    }
}

/// Cache of earlier sniff results, keyed by path. A file whose size and modification time are unchanged is not
/// opened again. [`ScanResult::index`] is the cache to pass to the next scan.
#[derive(Debug, Clone, Default)]
pub struct PrevIndex(HashMap<PathBuf, PrevEntry>);

impl PrevIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, path: PathBuf, entry: PrevEntry) {
        self.0.insert(path, entry);
    }

    pub fn get(&self, path: &Path) -> Option<&PrevEntry> {
        self.0.get(path)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PathBuf, &PrevEntry)> {
        self.0.iter()
    }

    /// The cached verdict, only when the file is unchanged.
    fn lookup(&self, path: &Path, size: u64, mtime_ns: u64) -> Option<Sniff> {
        self.0.get(path).filter(|e| e.size == size && e.mtime_ns == mtime_ns).map(|e| e.verdict)
    }
}

impl FromIterator<(PathBuf, PrevEntry)> for PrevIndex {
    fn from_iter<T: IntoIterator<Item = (PathBuf, PrevEntry)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A snapshot handed to the progress callback (at most ten times a second).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanProgress {
    /// Files looked at so far, text or not.
    pub files_seen: u64,
    /// Text files found so far.
    pub text_files: u64,
    /// Total size of the text files found so far.
    pub text_bytes: u64,
    /// The folder being walked.
    pub current_dir: String,
}

/// One path the user supplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanRoot {
    pub path: PathBuf,
    /// False for a loose file (and for a path that could not be read).
    pub is_dir: bool,
    /// The folder's or file's own name.
    pub name: String,
}

/// A text file found by the scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedFile {
    pub path: PathBuf,
    /// Index into [`ScanResult::roots`].
    pub root: usize,
    /// Path relative to its root, with `/` separators. A loose file's relative path is its own name.
    pub rel_path: String,
    pub size: u64,
    pub mtime_ns: u64,
}

/// Everything [`scan_paths`] learned.
#[derive(Debug, Clone, Default)]
pub struct ScanResult {
    pub roots: Vec<ScanRoot>,
    /// Text files in walk order (roots in the order given, each folder sorted by name).
    pub files: Vec<ScannedFile>,
    pub skipped: SkippedSummary,
    pub warnings: Vec<String>,
    /// Fresh cache to pass as `prev` next time.
    pub index: PrevIndex,
    /// Files looked at, text or not.
    pub files_seen: u64,
    /// True when the scan was cancelled; the lists above are then partial.
    pub cancelled: bool,
}

impl ScanResult {
    /// Total size of the text files.
    pub fn text_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// Scan `paths` (folders and/or loose files) for text.
///
/// `prev` lets unchanged files skip the sniff. `progress` is called at most ten times a second and once more at the
/// end. When `cancel` is set the walk stops and the partial result has `cancelled == true`.
pub fn scan_paths(
    paths: &[PathBuf],
    opts: &ScanOptions,
    prev: &PrevIndex,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(ScanProgress),
) -> ScanResult {
    let mut scanner = Scanner::new(opts, prev, cancel, progress);
    for (idx, path) in paths.iter().enumerate() {
        scanner.scan_root(idx, path);
    }
    scanner.finish()
}

/// How a file ended up skipped.
#[derive(Debug, Clone, Copy)]
enum SkipKind {
    Binary,
    Empty,
    Utf16,
    TooLarge,
    Unreadable,
}

/// Counts skipped files and keeps a bounded list of examples.
struct SkipTally {
    summary: SkippedSummary,
    max_examples: usize,
}

impl SkipTally {
    fn add(&mut self, kind: SkipKind, path: &Path, reason: impl Into<String>) {
        let counter = match kind {
            SkipKind::Binary => &mut self.summary.binary,
            SkipKind::Empty => &mut self.summary.empty,
            SkipKind::Utf16 => &mut self.summary.utf16,
            SkipKind::TooLarge => &mut self.summary.too_large,
            SkipKind::Unreadable => &mut self.summary.unreadable,
        };
        *counter = counter.saturating_add(1);
        if self.summary.examples.len() < self.max_examples {
            self.summary.examples.push(SkippedExample { path: path.display().to_string(), reason: reason.into() });
        }
    }
}

/// A file waiting for its verdict.
struct Pending {
    path: PathBuf,
    root: usize,
    rel_path: String,
    size: u64,
    mtime_ns: u64,
    /// Filled from the cache up front, or by the parallel sniff.
    verdict: Option<io::Result<Sniff>>,
}

/// Sniff files in batches so the parallel part stays simple and results keep their walk order.
const BATCH: usize = 256;

struct Scanner<'a> {
    opts: &'a ScanOptions,
    prev: &'a PrevIndex,
    cancel: &'a CancelFlag,
    progress: &'a mut dyn FnMut(ScanProgress),
    pool: Option<rayon::ThreadPool>,
    throttle: Throttle,
    result: ScanResult,
    tally: SkipTally,
    pending: Vec<Pending>,
    text_bytes: u64,
    current_dir: String,
    large: Vec<(PathBuf, u64)>,
}

impl<'a> Scanner<'a> {
    fn new(
        opts: &'a ScanOptions,
        prev: &'a PrevIndex,
        cancel: &'a CancelFlag,
        progress: &'a mut dyn FnMut(ScanProgress),
    ) -> Self {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(opts.threads.clamp(1, SCAN_THREADS)).build().ok();
        Self {
            opts,
            prev,
            cancel,
            progress,
            pool,
            throttle: Throttle::new(PROGRESS_INTERVAL),
            result: ScanResult::default(),
            tally: SkipTally { summary: SkippedSummary::default(), max_examples: opts.max_examples },
            pending: Vec::with_capacity(BATCH),
            text_bytes: 0,
            current_dir: String::new(),
            large: Vec::new(),
        }
    }

    fn scan_root(&mut self, idx: usize, path: &Path) {
        let name = display_name(path);
        let meta = fs::metadata(path);
        let is_dir = meta.as_ref().is_ok_and(|m| m.is_dir());
        self.result.roots.push(ScanRoot { path: path.to_path_buf(), is_dir, name: name.clone() });
        if self.cancel.is_cancelled() {
            self.result.cancelled = true;
            return;
        }
        let meta = match meta {
            Ok(m) => m,
            Err(e) => {
                self.result.files_seen += 1;
                self.tally.add(SkipKind::Unreadable, path, format!("cannot be opened: {e}"));
                return;
            }
        };
        if !is_dir {
            // A file named explicitly is used even when hidden; only its type and content can rule it out.
            self.consider_file(idx, path.to_path_buf(), name, Ok(meta));
            self.flush();
            return;
        }
        self.current_dir = path.display().to_string();
        let walker =
            WalkDir::new(path).follow_links(false).sort_by_file_name().into_iter().filter_entry(|e| !is_ignored(e));
        for entry in walker {
            if self.cancel.is_cancelled() {
                self.result.cancelled = true;
                break;
            }
            match entry {
                Err(e) => {
                    let at = e.path().unwrap_or(path).to_path_buf();
                    self.result.files_seen += 1;
                    self.tally.add(SkipKind::Unreadable, &at, format!("cannot be read: {}", io::Error::from(e)));
                }
                Ok(e) if e.depth() == 0 => {}
                Ok(e) if e.file_type().is_dir() => {
                    self.current_dir = e.path().display().to_string();
                    self.tick();
                }
                // Symbolic links are never followed; sockets and pipes would block if opened.
                Ok(e) if !e.file_type().is_file() => {}
                Ok(e) => {
                    let rel = rel_to_string(e.path().strip_prefix(path).unwrap_or(e.path()));
                    let meta = e.metadata().map_err(io::Error::from);
                    self.consider_file(idx, e.into_path(), rel, meta);
                }
            }
            if self.pending.len() >= BATCH {
                self.flush();
            }
        }
        self.flush();
    }

    /// Cheap checks that need no file content; queues the file for sniffing when they pass.
    fn consider_file(&mut self, root: usize, path: PathBuf, rel_path: String, meta: io::Result<fs::Metadata>) {
        self.result.files_seen += 1;
        if let Some(ext) = binary_extension(&path) {
            self.tally.add(SkipKind::Binary, &path, format!("binary file type (.{ext})"));
            self.tick();
            return;
        }
        let meta = match meta {
            Ok(m) => m,
            Err(e) => {
                self.tally.add(SkipKind::Unreadable, &path, format!("cannot be read: {e}"));
                return;
            }
        };
        let size = meta.len();
        if size == 0 {
            self.tally.add(SkipKind::Empty, &path, "empty file");
            return;
        }
        if self.opts.max_file_bytes.is_some_and(|max| size > max) {
            self.tally.add(
                SkipKind::TooLarge,
                &path,
                format!("larger than the {}-byte limit", self.opts.max_file_bytes.unwrap_or(0)),
            );
            return;
        }
        let mtime = mtime_ns(&meta);
        let verdict = self.prev.lookup(&path, size, mtime).map(Ok);
        self.pending.push(Pending { path, root, rel_path, size, mtime_ns: mtime, verdict });
    }

    /// Sniff everything queued (in parallel where the cache had no answer) and record the verdicts in order.
    fn flush(&mut self) {
        let todo: Vec<usize> = (0..self.pending.len()).filter(|&i| self.pending[i].verdict.is_none()).collect();
        if !todo.is_empty() {
            let sniffed: Vec<(usize, io::Result<Sniff>)> = {
                let pending = &self.pending;
                let run = || todo.par_iter().map(|&i| (i, sniff_file(&pending[i].path))).collect();
                match &self.pool {
                    Some(pool) => pool.install(run),
                    None => todo.iter().map(|&i| (i, sniff_file(&pending[i].path))).collect(),
                }
            };
            for (i, verdict) in sniffed {
                self.pending[i].verdict = Some(verdict);
            }
        }
        for p in std::mem::take(&mut self.pending) {
            match p.verdict {
                Some(Ok(Sniff::Text)) => self.accept(p),
                Some(Ok(Sniff::Empty)) => {
                    self.record(&p, Sniff::Empty);
                    self.tally.add(SkipKind::Empty, &p.path, "empty file");
                }
                Some(Ok(Sniff::Binary)) => {
                    self.record(&p, Sniff::Binary);
                    self.tally.add(SkipKind::Binary, &p.path, "contains binary data, not text");
                }
                Some(Ok(Sniff::Utf16)) => {
                    self.record(&p, Sniff::Utf16);
                    self.tally.add(SkipKind::Utf16, &p.path, "UTF-16 or UTF-32 text; save it as UTF-8 to use it");
                }
                Some(Err(e)) => self.tally.add(SkipKind::Unreadable, &p.path, format!("cannot be read: {e}")),
                None => {}
            }
        }
        self.tick();
    }

    fn record(&mut self, p: &Pending, verdict: Sniff) {
        self.result.index.insert(p.path.clone(), PrevEntry { size: p.size, mtime_ns: p.mtime_ns, verdict });
    }

    fn accept(&mut self, p: Pending) {
        self.record(&p, Sniff::Text);
        if p.size >= self.opts.warn_file_bytes {
            self.large.push((p.path.clone(), p.size));
        }
        self.text_bytes += p.size;
        self.result.files.push(ScannedFile {
            path: p.path,
            root: p.root,
            rel_path: p.rel_path,
            size: p.size,
            mtime_ns: p.mtime_ns,
        });
    }

    fn snapshot(&self) -> ScanProgress {
        ScanProgress {
            files_seen: self.result.files_seen,
            text_files: self.result.files.len() as u64,
            text_bytes: self.text_bytes,
            current_dir: self.current_dir.clone(),
        }
    }

    fn tick(&mut self) {
        if self.throttle.ready() {
            let snap = self.snapshot();
            (self.progress)(snap);
        }
    }

    fn finish(mut self) -> ScanResult {
        self.flush();
        if !self.large.is_empty() {
            let gb = |bytes: u64| bytes as f64 / 1e9;
            let (first, size) = &self.large[0];
            let warning = if self.large.len() == 1 {
                format!(
                    "{} is {:.1} GB. Very large files make preparing the dataset slower.",
                    first.display(),
                    gb(*size)
                )
            } else {
                format!(
                    "{} files are larger than {:.0} GB (for example {}). Very large files make preparing the dataset slower.",
                    self.large.len(),
                    gb(self.opts.warn_file_bytes),
                    first.display()
                )
            };
            self.result.warnings.push(warning);
        }
        let snap = self.snapshot();
        (self.progress)(snap);
        self.result.skipped = self.tally.summary;
        self.result
    }
}

/// Dot-entries and well-known build/vendor folders are skipped (never the root the user chose).
pub(crate) fn is_ignored(e: &DirEntry) -> bool {
    if e.depth() == 0 {
        return false;
    }
    let name = e.file_name().to_string_lossy();
    if name.starts_with('.') {
        return true;
    }
    if e.file_type().is_dir() {
        SKIP_DIR_NAMES.contains(&name.as_ref())
    } else {
        SKIP_FILE_NAMES.contains(&name.as_ref())
    }
}

/// A path's own name, falling back to the canonical path for things like `.` or `/`.
fn display_name(path: &Path) -> String {
    let own = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    own(path).or_else(|| fs::canonicalize(path).ok().and_then(|p| own(&p))).unwrap_or_else(|| "root".to_string())
}

/// Read up to [`SNIFF_BYTES`] bytes from the start of a file and sniff them.
fn sniff_file(path: &Path) -> io::Result<Sniff> {
    let mut head = Vec::with_capacity(SNIFF_BYTES);
    File::open(path)?.take(SNIFF_BYTES as u64).read_to_end(&mut head)?;
    Ok(sniff(&head))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::TempDir;

    // ----- sniff -----

    #[test]
    fn sniff_plain_text_and_empty() {
        assert_eq!(sniff(b""), Sniff::Empty);
        assert_eq!(sniff(b"hello world\n"), Sniff::Text);
        assert_eq!(sniff(b"   \n\t\r\n"), Sniff::Text);
        assert_eq!(sniff("日本語のテキスト、これは文章です。".as_bytes()), Sniff::Text);
        assert_eq!(sniff("emoji 🎉 and café".as_bytes()), Sniff::Text);
    }

    #[test]
    fn sniff_rejects_nul_bytes() {
        assert_eq!(sniff(b"abc\0def"), Sniff::Binary);
        assert_eq!(sniff(&[0u8; 64]), Sniff::Binary);
    }

    #[test]
    fn sniff_recognises_utf16_and_utf32_boms() {
        assert_eq!(sniff(&[0xFF, 0xFE, b'h', 0, b'i', 0]), Sniff::Utf16);
        assert_eq!(sniff(&[0xFE, 0xFF, 0, b'h', 0, b'i']), Sniff::Utf16);
        assert_eq!(sniff(&[0xFF, 0xFE, 0, 0, b'h', 0, 0, 0]), Sniff::Utf16);
        assert_eq!(sniff(&[0, 0, 0xFE, 0xFF, 0, 0, 0, b'h']), Sniff::Utf16);
    }

    #[test]
    fn sniff_strips_utf8_bom() {
        let mut data = vec![0xEF, 0xBB, 0xBF];
        data.extend_from_slice(b"text after a byte-order mark");
        assert_eq!(sniff(&data), Sniff::Text);
        assert_eq!(sniff(&[0xEF, 0xBB, 0xBF]), Sniff::Empty, "a bare BOM carries no text");
        // The BOM is not counted towards the printable ratio.
        let mut mostly_control = vec![0xEF, 0xBB, 0xBF];
        mostly_control.extend_from_slice(&[1u8; 20]);
        assert_eq!(sniff(&mostly_control), Sniff::Binary);
    }

    #[test]
    fn sniff_tolerates_a_truncated_final_code_point() {
        let full = "naïve café 🎉".as_bytes();
        // Cut inside the 4-byte emoji, inside the 2-byte "é", and right after the lead byte.
        for cut in 1..=3 {
            let head = &full[..full.len() - cut];
            assert_eq!(sniff(head), Sniff::Text, "cut {cut} bytes");
        }
        let mut head = b"hello wor".to_vec();
        head.push(0xE2); // lead byte of a 3-byte sequence, nothing after it
        assert_eq!(sniff(&head), Sniff::Text);
    }

    #[test]
    fn sniff_rejects_invalid_utf8_elsewhere() {
        // Latin-1 "é" followed by ASCII is invalid UTF-8 in the middle of the data.
        assert_eq!(sniff(b"caf\xE9 au lait, s'il vous plait"), Sniff::Binary);
        assert_eq!(sniff(b"\xFF\xFFnot utf8"), Sniff::Binary, "FF FE and FE FF are BOMs, FF FF is just invalid");
        assert_eq!(sniff(b"\x80\x81\x82 continuation bytes only"), Sniff::Binary);
        assert_eq!(sniff(&[0xC3]), Sniff::Binary, "a lone truncated byte proves nothing");
    }

    #[test]
    fn sniff_requires_more_than_ninety_percent_printable() {
        let mut ninety = vec![b'a'; 90];
        ninety.extend_from_slice(&[0x01; 10]);
        assert_eq!(sniff(&ninety), Sniff::Binary, "exactly 90 % is not enough");
        let mut ninety_one = vec![b'a'; 91];
        ninety_one.extend_from_slice(&[0x01; 9]);
        assert_eq!(sniff(&ninety_one), Sniff::Text);
        // DEL and the escape character are not printable either.
        assert_eq!(sniff(&[0x1B, 0x7F, 0x1B, 0x7F, b'a']), Sniff::Binary);
    }

    // ----- fixtures -----

    fn write(root: &Path, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn scan(paths: &[PathBuf]) -> ScanResult {
        scan_paths(paths, &ScanOptions::default(), &PrevIndex::new(), &CancelFlag::new(), &mut |_| {})
    }

    fn rels(r: &ScanResult) -> Vec<&str> {
        r.files.iter().map(|f| f.rel_path.as_str()).collect()
    }

    fn is_root_user() -> bool {
        #[cfg(unix)]
        {
            // A cheap proxy: root can read a mode-000 file.
            let dir = TempDir::new().unwrap();
            let p = write(dir.path(), "probe", b"x");
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&p, fs::Permissions::from_mode(0o000)).unwrap();
            fs::read(&p).is_ok()
        }
        #[cfg(not(unix))]
        {
            true
        }
    }

    // ----- scan -----

    #[test]
    fn scan_classifies_a_mixed_tree() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write(root, "a/one.txt", b"first file\n");
        write(root, "a/deep/er/two.md", b"# second\n");
        write(root, "b/three.rs", b"fn main() {}\n");
        write(root, "UPPER/Four.TXT", "ünïcode text\n".as_bytes());
        write(root, ".hidden/secret.txt", b"hidden dir\n");
        write(root, "a/.dotfile", b"hidden file\n");
        write(root, "node_modules/pkg/index.js", b"module.exports = 1;\n");
        write(root, ".git/config", b"[core]\n");
        write(root, "target/debug/x.txt", b"build output\n");
        write(root, "build/y.txt", b"build\n");
        write(root, "dist/z.txt", b"dist\n");
        write(root, "__pycache__/m.txt", b"cache\n");
        write(root, "venv/lib.txt", b"venv\n");
        write(root, ".DS_Store", b"\0\0\0\x01Bud1");
        write(root, "a/Thumbs.db", b"thumbs");
        write(root, "img/photo.PNG", b"\x89PNG\r\n\x1a\n");
        write(root, "img/clip.mp4", b"\0\0\0\x18ftypmp42");
        write(root, "bin/data.dat", &[0u8, 1, 2, 3, 0, 0, 255, 254]);
        write(root, "bin/latin1.txt", b"caf\xE9 au lait, s'il vous plait");
        write(root, "bin/utf16.txt", &[0xFF, 0xFE, b'h', 0, b'i', 0]);
        write(root, "empty/nothing.txt", b"");
        write(root, "docs/archive.zip", b"PK\x03\x04");

        let r = scan(&[root.to_path_buf()]);
        assert_eq!(rels(&r), ["UPPER/Four.TXT", "a/deep/er/two.md", "a/one.txt", "b/three.rs"]);
        assert!(r.files.iter().all(|f| f.root == 0 && f.size > 0 && f.path.is_absolute()));

        let s = &r.skipped;
        // png + mp4 + zip by extension, data.dat + latin1.txt by content.
        assert_eq!(s.binary, 5, "{s:?}");
        assert_eq!(s.utf16, 1);
        assert_eq!(s.empty, 1);
        assert_eq!((s.too_large, s.unreadable), (0, 0));
        assert!(s.examples.iter().any(|e| e.path.ends_with("photo.PNG") && e.reason.contains(".png")));
        assert!(s.examples.iter().any(|e| e.path.ends_with("utf16.txt") && e.reason.contains("UTF-16")));
        assert_eq!(r.files_seen, 4 + 5 + 1 + 1);
        assert!(!r.cancelled);
    }

    #[test]
    fn binary_extensions_are_never_opened() {
        let dir = TempDir::new().unwrap();
        let p = write(dir.path(), "pic.png", b"not really a png");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&p, fs::Permissions::from_mode(0o000)).unwrap();
        }
        let r = scan(&[dir.path().to_path_buf()]);
        assert_eq!(r.skipped.binary, 1);
        assert_eq!(r.skipped.unreadable, 0, "an unreadable image must not be opened, so it cannot fail to open");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_not_followed_and_loops_terminate() {
        use std::os::unix::fs::symlink;
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write(root, "real/a.txt", b"real\n");
        symlink(root, root.join("real/loop")).unwrap(); // points back at the root
        symlink(root.join("real/a.txt"), root.join("real/link.txt")).unwrap();
        symlink(root.join("real"), root.join("alias")).unwrap();
        symlink(root.join("missing"), root.join("dangling")).unwrap();
        let r = scan(&[root.to_path_buf()]);
        assert_eq!(rels(&r), ["real/a.txt"]);
        assert_eq!(r.skipped.total(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_files_and_folders_are_counted() {
        use std::os::unix::fs::PermissionsExt;
        if is_root_user() {
            return; // root ignores permission bits
        }
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write(root, "ok.txt", b"fine\n");
        let locked_file = write(root, "locked.txt", b"secret\n");
        let locked_dir = root.join("locked_dir");
        write(root, "locked_dir/inside.txt", b"inside\n");
        fs::set_permissions(&locked_file, fs::Permissions::from_mode(0o000)).unwrap();
        fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o000)).unwrap();

        let r = scan(&[root.to_path_buf()]);

        fs::set_permissions(&locked_file, fs::Permissions::from_mode(0o644)).unwrap();
        fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(rels(&r), ["ok.txt"]);
        assert_eq!(r.skipped.unreadable, 2, "{:?}", r.skipped);
        assert!(r.skipped.examples.iter().all(|e| e.reason.contains("cannot be")));
    }

    #[test]
    fn a_giant_single_line_is_text_and_only_its_head_is_read() {
        let dir = TempDir::new().unwrap();
        let line = vec![b'x'; 8 * 1024 * 1024];
        write(dir.path(), "giant.txt", &line);
        let r = scan(&[dir.path().to_path_buf()]);
        assert_eq!(rels(&r), ["giant.txt"]);
        assert_eq!(r.files[0].size, 8 * 1024 * 1024);
    }

    #[test]
    fn explicit_files_and_multiple_roots_keep_their_own_relative_paths() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("alpha");
        let b = dir.path().join("beta");
        write(&a, "x/one.txt", b"1\n");
        write(&b, "two.txt", b"2\n");
        let loose = write(dir.path(), ".notes.txt", b"hidden but chosen explicitly\n");
        let image = write(dir.path(), "cover.jpg", b"\xFF\xD8\xFF");
        let missing = dir.path().join("nope.txt");

        let r = scan(&[a.clone(), b.clone(), loose.clone(), image, missing.clone()]);
        assert_eq!(r.roots.len(), 5);
        assert_eq!((r.roots[0].name.as_str(), r.roots[0].is_dir), ("alpha", true));
        assert_eq!((r.roots[2].name.as_str(), r.roots[2].is_dir), (".notes.txt", false));
        let got: Vec<(usize, &str)> = r.files.iter().map(|f| (f.root, f.rel_path.as_str())).collect();
        assert_eq!(got, [(0, "x/one.txt"), (1, "two.txt"), (2, ".notes.txt")]);
        assert_eq!(r.skipped.binary, 1);
        assert_eq!(r.skipped.unreadable, 1);
        assert!(r.skipped.examples.iter().any(|e| e.path == missing.display().to_string()));
    }

    #[test]
    fn skipped_examples_are_capped_but_counts_are_exact() {
        let dir = TempDir::new().unwrap();
        for i in 0..80 {
            write(dir.path(), &format!("img{i:02}.png"), b"x");
        }
        let r = scan(&[dir.path().to_path_buf()]);
        assert_eq!(r.skipped.binary, 80);
        assert_eq!(r.skipped.examples.len(), MAX_SKIPPED_EXAMPLES);
    }

    #[test]
    fn size_limit_marks_files_too_large_and_big_files_warn() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "small.txt", b"tiny\n");
        write(dir.path(), "big.txt", &vec![b'a'; 5000]);
        let opts = ScanOptions { max_file_bytes: Some(1000), ..ScanOptions::default() };
        let r = scan_paths(&[dir.path().to_path_buf()], &opts, &PrevIndex::new(), &CancelFlag::new(), &mut |_| {});
        assert_eq!(rels(&r), ["small.txt"]);
        assert_eq!(r.skipped.too_large, 1);

        let opts = ScanOptions { warn_file_bytes: 1000, ..ScanOptions::default() };
        let r = scan_paths(&[dir.path().to_path_buf()], &opts, &PrevIndex::new(), &CancelFlag::new(), &mut |_| {});
        assert_eq!(r.files.len(), 2, "big files are accepted");
        assert!(r.warnings.len() == 1 && r.warnings[0].contains("big.txt"), "{:?}", r.warnings);
    }

    #[test]
    fn the_cache_avoids_re_sniffing_unchanged_files() {
        let dir = TempDir::new().unwrap();
        let p = write(dir.path(), "a.txt", b"plain text\n");
        let paths = [dir.path().to_path_buf()];
        let first = scan(&paths);
        assert_eq!(first.files.len(), 1);
        assert_eq!(first.index.len(), 1);
        assert!(first.index.get(&p).unwrap().was_text());

        // Swap in binary content of the same size, then restore the old modification time.
        let old_mtime = fs::metadata(&p).unwrap().modified().unwrap();
        fs::write(&p, [0u8; 11]).unwrap();
        OpenOptions::new().write(true).open(&p).unwrap().set_modified(old_mtime).unwrap();

        let cached = scan_paths(&paths, &ScanOptions::default(), &first.index, &CancelFlag::new(), &mut |_| {});
        assert_eq!(cached.files.len(), 1, "unchanged size and mtime: the cached verdict is reused");
        let fresh = scan(&paths);
        assert_eq!(fresh.files.len(), 0, "without the cache the file is sniffed again");
        assert_eq!(fresh.skipped.binary, 1);

        // A different size invalidates the cache entry.
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"more").unwrap();
        drop(f);
        let changed = scan_paths(&paths, &ScanOptions::default(), &first.index, &CancelFlag::new(), &mut |_| {});
        assert_eq!(changed.files.len(), 0);
    }

    #[test]
    fn cached_skips_keep_their_reason() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "u.txt", &[0xFF, 0xFE, b'a', 0]);
        let paths = [dir.path().to_path_buf()];
        let first = scan(&paths);
        let second = scan_paths(&paths, &ScanOptions::default(), &first.index, &CancelFlag::new(), &mut |_| {});
        assert_eq!(second.skipped.utf16, 1);
    }

    #[test]
    fn cancellation_returns_a_partial_result() {
        let dir = TempDir::new().unwrap();
        for i in 0..50 {
            write(dir.path(), &format!("f{i}.txt"), b"x\n");
        }
        let cancel = CancelFlag::new();
        cancel.cancel();
        let r =
            scan_paths(&[dir.path().to_path_buf()], &ScanOptions::default(), &PrevIndex::new(), &cancel, &mut |_| {});
        assert!(r.cancelled);
        assert!(r.files.is_empty());

        // Cancel from inside the progress callback of a bigger scan.
        for i in 0..2000 {
            write(dir.path(), &format!("g/{i:04}.txt"), b"y\n");
        }
        let cancel = CancelFlag::new();
        let mut calls = 0;
        let r =
            scan_paths(&[dir.path().to_path_buf()], &ScanOptions::default(), &PrevIndex::new(), &cancel, &mut |_| {
                calls += 1;
                cancel.cancel();
            });
        assert!(calls >= 1);
        assert!(r.cancelled);
        assert!(r.files.len() < 2050);
    }

    #[test]
    fn progress_is_throttled_and_ends_with_the_final_counts() {
        let dir = TempDir::new().unwrap();
        for i in 0..1500 {
            write(dir.path(), &format!("d{}/f{i}.txt", i % 10), b"line\n");
        }
        let mut seen: Vec<ScanProgress> = Vec::new();
        let started = std::time::Instant::now();
        let r = scan_paths(
            &[dir.path().to_path_buf()],
            &ScanOptions::default(),
            &PrevIndex::new(),
            &CancelFlag::new(),
            &mut |p| seen.push(p),
        );
        let elapsed = started.elapsed();
        let max_calls = (elapsed.as_millis() / PROGRESS_INTERVAL.as_millis()) as usize + 2;
        assert!(seen.len() <= max_calls, "{} callbacks in {elapsed:?}", seen.len());
        let last = seen.last().unwrap();
        assert_eq!((last.files_seen, last.text_files), (1500, 1500));
        assert_eq!(last.text_bytes, r.text_bytes());
        assert!(seen.windows(2).all(|w| w[0].files_seen <= w[1].files_seen));
    }

    #[test]
    fn results_are_deterministic() {
        let dir = TempDir::new().unwrap();
        for name in ["z.txt", "a.txt", "m/q.txt", "m/b.txt", "k.txt"] {
            write(dir.path(), name, b"x\n");
        }
        let a = scan(&[dir.path().to_path_buf()]);
        let b = scan(&[dir.path().to_path_buf()]);
        assert_eq!(rels(&a), rels(&b));
        assert_eq!(rels(&a), ["a.txt", "k.txt", "m/b.txt", "m/q.txt", "z.txt"]);
    }
}
