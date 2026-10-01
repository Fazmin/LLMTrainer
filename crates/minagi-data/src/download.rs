//! Resumable download of starter text into engine-ready shards.
//!
//! A [`RemoteFile`] is fetched with HTTP `Range` requests and cut into shards of at most 8 MiB named
//! `part-0000.txt`, `part-0001.txt`, ... in `<dest>/<train|val>/<lane>/`. Shards are cut at the last
//! `"\n<|endoftext|>\n"` in each window, so a story is never split between two files, and every shard is valid UTF-8.
//!
//! What makes it safe to interrupt at any moment:
//!
//! * A shard is written as `part-NNNN.txt.part`, synced, then renamed, so a shard file is either absent or whole.
//! * After every shard a cursor (`sourceOffset`, `shardIdx`, and the CDN's `etag`) is saved atomically in
//!   `<dest>/.download-state.json`. A restart continues from the cursor; at most one shard of network traffic is lost.
//! * On restart the finished shards are re-read to rebuild the running SHA-256 (and checked for the right total size),
//!   so the whole-file checksum still covers the whole stream.
//!
//! Hugging Face answers `resolve/<revision>/<file>` with a 302 to a *signed* CDN URL that expires, so every attempt,
//! first or retried, starts with a fresh `HEAD` against the original URL (never following the redirect) and uses the
//! new `Location`. The `HEAD` also returns `x-linked-size` and `x-linked-etag` (the file's SHA-256), which are checked
//! against the manifest. Ranged `GET`s must answer `206` with a `Content-Range` that matches exactly; a `200` to a
//! ranged request (a server that ignores `Range`) is an error, never silently accepted.
//!
//! The [`Client`] passed in must **not** follow redirects; [`http_client`] builds a suitable one.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use minagi_types::JobProgress;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_LENGTH, CONTENT_RANGE, ETAG, LOCATION, RANGE};
use reqwest::{Client, Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::error::{DataError, DataResult};
use crate::util::{PROGRESS_INTERVAL, RateMeter, Throttle, hex};

/// Largest shard written, in bytes.
pub const SHARD_BYTES: u64 = 8 * 1024 * 1024;
/// Stories are separated by a line holding just this marker.
pub const STORY_BOUNDARY: &[u8] = b"\n<|endoftext|>\n";
/// File holding the resume cursor while a download is in progress.
pub const STATE_FILE: &str = ".download-state.json";
/// Slack required on top of the download itself, in bytes.
pub const SLACK_BYTES: u64 = 1024 * 1024 * 1024;
/// Free space required, as a multiple of the download's size.
pub const SPACE_FACTOR: f64 = 1.2;

/// Which half of the dataset a source file feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Train,
    Val,
}

impl Side {
    /// The directory name under the dataset root.
    pub fn dir(self) -> &'static str {
        match self {
            Side::Train => "train",
            Side::Val => "val",
        }
    }
}

/// One file to download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    /// The `resolve/<revision>/<name>` URL.
    pub url: String,
    /// The file's name, used in messages and as the key in the resume state.
    pub name: String,
    /// The lane the text goes to (`stories`).
    pub lane: String,
    pub side: Side,
    /// Size of the whole remote file in bytes, as published (`x-linked-size`).
    pub size: u64,
    /// SHA-256 of the whole remote file, as published (`x-linked-etag`).
    pub sha256: String,
    /// When set, only the first this-many bytes are used (cut back to a story boundary). The slice has no published
    /// checksum, so it is checked through `Content-Range`, the file's identity (`x-linked-etag`), the CDN's `etag`
    /// across resumes and UTF-8 validity, and its own SHA-256 is recorded in the manifest.
    pub prefix_bytes: Option<u64>,
}

impl RemoteFile {
    /// Bytes of the source that will be read.
    pub fn effective_bytes(&self) -> u64 {
        self.prefix_bytes.map_or(self.size, |p| p.min(self.size))
    }

    /// Where this file's shards go, below `dest`.
    pub fn shard_dir(&self, dest: &Path) -> PathBuf {
        dest.join(self.side.dir()).join(&self.lane)
    }
}

/// Free space of the filesystem holding `path`, in bytes.
pub type FreeSpaceFn = fn(&Path) -> io::Result<u64>;

/// Knobs for the downloader. The defaults are what the app uses; tests shrink the shards and the delays.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub shard_bytes: u64,
    /// Attempts without any progress before giving up. A shard completing resets the count.
    pub max_attempts: u32,
    /// First retry delay; doubles each time, up to 30 s.
    pub backoff_base: Duration,
    pub slack_bytes: u64,
    pub space_factor: f64,
    pub progress_interval: Duration,
    pub free_space: FreeSpaceFn,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            shard_bytes: SHARD_BYTES,
            max_attempts: 5,
            backoff_base: Duration::from_secs(1),
            slack_bytes: SLACK_BYTES,
            space_factor: SPACE_FACTOR,
            progress_interval: PROGRESS_INTERVAL,
            free_space: disk_free_bytes,
        }
    }
}

/// Free space available to this user on the filesystem holding `path`.
pub fn disk_free_bytes(path: &Path) -> io::Result<u64> {
    fs4::available_space(path)
}

/// An HTTP client that is safe to use for downloads: rustls, no redirect following, timeouts.
pub fn http_client() -> DataResult<Client> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("llm-trainer/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(DataError::from)
}

/// The resume cursor of one source file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileState {
    /// Bytes of the source consumed so far (finished shards, and for a prefix also the cut-off tail once done).
    pub source_offset: u64,
    /// Bytes held by the finished shards. Equal to `source_offset` until a prefix drops its unfinished last story.
    pub kept_bytes: u64,
    /// Number of finished shards (the next one is `part-<shard_idx>.txt`).
    pub shard_idx: u32,
    /// The `etag` the CDN sent with the first ranged response; it must not change between resumes.
    pub etag: Option<String>,
    pub complete: bool,
    /// SHA-256 of the bytes kept (the whole file, or the prefix), once complete.
    pub sha256: Option<String>,
}

/// Contents of `.download-state.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadState {
    pub version: u32,
    pub starter_id: String,
    pub files: BTreeMap<String, FileState>,
}

impl DownloadState {
    pub const VERSION: u32 = 1;

    pub fn new(starter_id: &str) -> Self {
        Self { version: Self::VERSION, starter_id: starter_id.to_string(), files: BTreeMap::new() }
    }

    /// Load the state in `dest`, or `None` when there is none (or it is unreadable and must be ignored).
    pub fn load(dest: &Path) -> Option<Self> {
        let bytes = std::fs::read(dest.join(STATE_FILE)).ok()?;
        let state: DownloadState = serde_json::from_slice(&bytes).ok()?;
        (state.version == Self::VERSION).then_some(state)
    }

    /// Write the state atomically (temp file, sync, rename).
    pub fn save(&self, dest: &Path) -> DataResult<()> {
        let path = dest.join(STATE_FILE);
        let tmp = dest.join(format!("{STATE_FILE}.tmp"));
        let json = serde_json::to_vec_pretty(self).map_err(|e| DataError::io(&path, io::Error::other(e)))?;
        let write = || -> io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            io::Write::write_all(&mut f, &json)?;
            f.sync_all()?;
            std::fs::rename(&tmp, &path)
        };
        write().map_err(|e| DataError::io(&path, e))
    }
}

/// How one source file ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOutcome {
    pub name: String,
    pub side: Side,
    pub lane: String,
    pub shards: u32,
    /// Bytes in the shards.
    pub kept_bytes: u64,
    /// SHA-256 of those bytes.
    pub sha256: String,
    /// True when the SHA-256 was compared against the published checksum (a whole file, not a prefix).
    pub verified_against_published: bool,
}

/// What a failed attempt means for the retry loop.
enum Attempt {
    /// Worth trying again after a delay (the next attempt re-resolves the URL).
    Retry(DataError),
    /// Trying again would give the same result.
    Fatal(DataError),
    /// Fatal, and what was downloaded so far cannot be trusted: it is thrown away so the next run starts clean.
    Discard(DataError),
}

impl From<DataError> for Attempt {
    fn from(e: DataError) -> Self {
        // Cancellation, disk trouble and invalid data are never fixed by waiting.
        Attempt::Fatal(e)
    }
}

/// Retry on a network-level failure (connection, timeout, truncated body).
fn transient(e: impl Into<DataError>) -> Attempt {
    Attempt::Retry(e.into())
}

/// A download in progress: owns the progress reporting and the resume state.
pub(crate) struct Session<'a, P: FnMut(JobProgress)> {
    client: &'a Client,
    opts: &'a DownloadOptions,
    cancel: &'a CancellationToken,
    progress: P,
    throttle: Throttle,
    meter: RateMeter,
    dest: &'a Path,
    pub state: DownloadState,
    /// Bytes to read over all files.
    total_bytes: u64,
    /// Bytes of finished files.
    finished_bytes: u64,
    /// Bytes fetched over the network in this run.
    pub fetched: u64,
}

impl<'a, P: FnMut(JobProgress)> Session<'a, P> {
    pub fn new(
        client: &'a Client,
        opts: &'a DownloadOptions,
        cancel: &'a CancellationToken,
        progress: P,
        dest: &'a Path,
        state: DownloadState,
        files: &[RemoteFile],
    ) -> Self {
        let total_bytes = files.iter().map(RemoteFile::effective_bytes).sum();
        let done: u64 = files
            .iter()
            .map(|f| match state.files.get(&f.name) {
                Some(s) if s.complete => f.effective_bytes(),
                Some(s) => s.source_offset,
                None => 0,
            })
            .sum();
        Self {
            client,
            opts,
            cancel,
            progress,
            throttle: Throttle::new(opts.progress_interval),
            meter: RateMeter::new(done as f64),
            dest,
            state,
            total_bytes,
            finished_bytes: 0,
            fetched: 0,
        }
    }

    /// Bytes still to read across `files`, given the saved cursors.
    pub fn remaining_bytes(&self, files: &[RemoteFile]) -> u64 {
        files
            .iter()
            .map(|f| match self.state.files.get(&f.name) {
                Some(s) if s.complete => 0,
                Some(s) => f.effective_bytes().saturating_sub(s.source_offset),
                None => f.effective_bytes(),
            })
            .sum()
    }

    /// Refuse to start without room for the download and some slack.
    pub fn check_space(&self, remaining: u64) -> DataResult<()> {
        let need = (remaining as f64 * self.opts.space_factor).ceil() as u64 + self.opts.slack_bytes;
        let free = (self.opts.free_space)(self.dest).map_err(|e| DataError::io(self.dest, e))?;
        if free < need { Err(DataError::DiskFull { need_bytes: need, free_bytes: free }) } else { Ok(()) }
    }

    fn emit(&mut self, current_done: u64, message: &str, force: bool) {
        if !force && !self.throttle.ready() {
            return;
        }
        let done = (self.finished_bytes + current_done) as f64;
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

    /// Download one file completely (or continue it), retrying with backoff.
    pub async fn download_file(&mut self, file: &RemoteFile) -> DataResult<FileOutcome> {
        let dir = file.shard_dir(self.dest);
        tokio::fs::create_dir_all(&dir).await.map_err(|e| DataError::io(&dir, e))?;

        let mut st = self.state.files.get(&file.name).cloned().unwrap_or_default();
        if st.complete {
            // Finished earlier: make sure the shards are all still there, without hashing them again.
            match shards_total_size(&dir, st.shard_idx).await {
                Ok(total) if total == st.kept_bytes => {
                    let sha = st.sha256.clone().unwrap_or_default();
                    self.finished_bytes += file.effective_bytes();
                    return Ok(outcome(file, &st, sha));
                }
                _ => st = FileState::default(),
            }
        }
        // The hash of everything already on disk; rebuilt from the shards because SHA-256 state cannot be saved.
        let mut hasher = Sha256::new();
        if st.shard_idx > 0 {
            self.emit(st.source_offset, &format!("Checking what was already downloaded of {}", file.name), true);
            match replay_shards(&dir, st.shard_idx, st.kept_bytes).await {
                Ok(h) => hasher = h,
                Err(_) => st = FileState::default(),
            }
        }
        if st.shard_idx == 0 {
            st = FileState::default();
            clear_shards(&dir).await?;
        } else {
            remove_partial_shards(&dir).await?;
        }

        let mut attempts = 0u32;
        loop {
            if self.cancel.is_cancelled() {
                return Err(DataError::Cancelled);
            }
            let shards_before = st.shard_idx;
            let result = self.attempt(file, &dir, &mut st, &mut hasher).await;
            self.state.files.insert(file.name.clone(), st.clone());
            match result {
                Ok(()) => break,
                Err(Attempt::Fatal(e)) => return Err(e),
                Err(Attempt::Discard(e)) => {
                    self.discard(file, &dir).await;
                    return Err(e);
                }
                Err(Attempt::Retry(e)) => {
                    if st.shard_idx > shards_before {
                        attempts = 0;
                    }
                    attempts += 1;
                    if attempts >= self.opts.max_attempts {
                        return Err(e);
                    }
                    let delay =
                        self.opts.backoff_base.saturating_mul(1 << (attempts - 1).min(10)).min(Duration::from_secs(30));
                    self.emit(
                        st.source_offset,
                        &format!("Connection problem, retrying in {} s", delay.as_secs().max(1)),
                        true,
                    );
                    tokio::select! {
                        _ = self.cancel.cancelled() => return Err(DataError::Cancelled),
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
            }
        }

        let actual = hex(&hasher.finalize());
        let verified = file.prefix_bytes.is_none();
        if verified && !actual.eq_ignore_ascii_case(&file.sha256) {
            self.discard(file, &dir).await;
            return Err(DataError::Checksum { file: file.name.clone(), expected: file.sha256.clone(), actual });
        }
        st.complete = true;
        st.sha256 = Some(actual.clone());
        self.state.files.insert(file.name.clone(), st.clone());
        self.state.save(self.dest)?;
        self.finished_bytes += file.effective_bytes();
        self.emit(0, &format!("Finished {}", file.name), true);
        Ok(outcome(file, &st, actual))
    }

    /// Throw away everything downloaded of `file` and forget its cursor.
    async fn discard(&mut self, file: &RemoteFile, dir: &Path) {
        let _ = clear_shards(dir).await;
        self.state.files.remove(&file.name);
        let _ = self.state.save(self.dest);
    }

    /// One pass: resolve the URL, request the remaining range, stream it into shards.
    async fn attempt(
        &mut self,
        file: &RemoteFile,
        dir: &Path,
        st: &mut FileState,
        hasher: &mut Sha256,
    ) -> Result<(), Attempt> {
        let end_exclusive = file.effective_bytes();
        if st.source_offset >= end_exclusive {
            return Ok(());
        }
        let url = self.resolve(file).await?;
        let start = st.source_offset;
        let last = end_exclusive - 1;
        let request = self
            .client
            .get(url)
            .header(RANGE, format!("bytes={start}-{last}"))
            .header(ACCEPT_ENCODING, "identity")
            .send();
        let response = tokio::select! {
            _ = self.cancel.cancelled() => return Err(DataError::Cancelled.into()),
            r = request => r.map_err(transient)?,
        };
        check_ranged_response(&response, file, start, last)?;
        // The CDN's own etag must stay the same for as long as we keep adding to the same download.
        let etag = response.headers().get(ETAG).and_then(|v| v.to_str().ok()).map(normalize_etag);
        match (&st.etag, etag) {
            (Some(saved), Some(now)) if *saved != now => {
                return Err(Attempt::Discard(DataError::invalid(format!(
                    "{} changed on the server since the download started; it has to be downloaded again",
                    file.name
                ))));
            }
            (None, Some(now)) => st.etag = Some(now),
            _ => {}
        }

        let mut stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::with_capacity(self.opts.shard_bytes as usize + 64 * 1024);
        let mut received = 0u64;
        let expected = last - start + 1;
        let message = format!("Downloading {}", file.name);
        loop {
            let chunk = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => return Err(DataError::Cancelled.into()),
                c = stream.next() => c,
            };
            match chunk {
                None => break,
                Some(Err(e)) => return Err(transient(e)),
                Some(Ok(bytes)) => {
                    received += bytes.len() as u64;
                    self.fetched += bytes.len() as u64;
                    if received > expected {
                        return Err(Attempt::Fatal(DataError::Network(format!(
                            "the server sent more than the {expected} bytes that were asked for"
                        ))));
                    }
                    buf.extend_from_slice(&bytes);
                    while buf.len() as u64 >= self.opts.shard_bytes {
                        let window = &buf[..self.opts.shard_bytes as usize];
                        let cut = story_cut(window).ok_or_else(|| {
                            Attempt::Fatal(DataError::invalid(format!(
                                "{} has no \"<|endoftext|>\" line in {} bytes, so it cannot be cut into whole stories",
                                file.name, self.opts.shard_bytes
                            )))
                        })?;
                        self.commit_shard(file, dir, st, hasher, &buf[..cut]).await?;
                        buf.drain(..cut);
                    }
                    self.emit(st.source_offset + buf.len() as u64, &message, false);
                }
            }
        }
        if received != expected {
            return Err(transient(DataError::Network(format!(
                "the connection closed after {received} of {expected} bytes"
            ))));
        }

        // The stream is complete; what is left in the buffer is the last shard.
        let last_shard: &[u8] = if file.prefix_bytes.is_some() {
            // A prefix ends mid-story; keep whole stories only.
            story_cut(&buf).map_or(&[][..], |cut| &buf[..cut])
        } else {
            &buf
        };
        if !last_shard.is_empty() {
            self.commit_shard(file, dir, st, hasher, last_shard).await?;
        }
        if file.prefix_bytes.is_some() {
            // The cut-off tail of the prefix counts as consumed.
            st.source_offset = end_exclusive;
            self.state.files.insert(file.name.clone(), st.clone());
            self.state.save(self.dest)?;
        }
        Ok(())
    }

    /// `HEAD` the original URL without following the redirect: learn the fresh signed URL and check the file's
    /// identity against the manifest.
    async fn resolve(&mut self, file: &RemoteFile) -> Result<Url, Attempt> {
        let url = Url::parse(&file.url).map_err(|e| DataError::invalid(format!("bad URL {}: {e}", file.url)))?;
        let request = self.client.head(url.clone()).header(ACCEPT_ENCODING, "identity").send();
        let response = tokio::select! {
            _ = self.cancel.cancelled() => return Err(DataError::Cancelled.into()),
            r = request => r.map_err(transient)?,
        };
        let status = response.status();
        let headers = response.headers();
        let target = if status.is_redirection() {
            let location = headers.get(LOCATION).and_then(|v| v.to_str().ok()).ok_or_else(|| {
                Attempt::Fatal(DataError::Network("the server redirected without saying where".into()))
            })?;
            url.join(location).map_err(|e| DataError::Network(format!("bad redirect target: {e}")))?
        } else if status.is_success() {
            url.clone()
        } else {
            // A login wall or a missing file will not go away by asking again.
            return Err(match status.as_u16() {
                401 | 403 => Attempt::Fatal(DataError::Network(format!(
                    "the server refused access to {} ({status}); it may need a login",
                    file.name
                ))),
                _ => status_attempt(status, "looking up", &file.name),
            });
        };

        let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_string);
        let published_size = header("x-linked-size")
            .or_else(|| if status.is_success() { header(CONTENT_LENGTH.as_str()) } else { None })
            .and_then(|s| s.trim().parse::<u64>().ok());
        if let Some(size) = published_size
            && size != file.size
        {
            return Err(Attempt::Fatal(DataError::invalid(format!(
                "{} is {size} bytes on the server but {} bytes were expected; the starter list is out of date",
                file.name, file.size
            ))));
        }
        if let Some(etag) = header("x-linked-etag").map(|e| normalize_etag(&e))
            && !etag.eq_ignore_ascii_case(&file.sha256)
        {
            return Err(Attempt::Fatal(DataError::invalid(format!(
                "{} has a different checksum on the server ({etag}) than the starter list expects ({}); the file was changed",
                file.name, file.sha256
            ))));
        }
        Ok(target)
    }

    /// Validate, write and record one finished shard.
    async fn commit_shard(
        &mut self,
        file: &RemoteFile,
        dir: &Path,
        st: &mut FileState,
        hasher: &mut Sha256,
        data: &[u8],
    ) -> Result<(), Attempt> {
        if std::str::from_utf8(data).is_err() {
            return Err(Attempt::Fatal(DataError::invalid(format!("{} is not valid UTF-8 text", file.name))));
        }
        write_shard(dir, st.shard_idx, data).await.map_err(|e| DataError::io(dir, e))?;
        hasher.update(data);
        st.source_offset += data.len() as u64;
        st.kept_bytes += data.len() as u64;
        st.shard_idx += 1;
        self.state.files.insert(file.name.clone(), st.clone());
        self.state.save(self.dest)?;
        Ok(())
    }
}

fn outcome(file: &RemoteFile, st: &FileState, sha256: String) -> FileOutcome {
    FileOutcome {
        name: file.name.clone(),
        side: file.side,
        lane: file.lane.clone(),
        shards: st.shard_idx,
        kept_bytes: st.kept_bytes,
        sha256,
        verified_against_published: file.prefix_bytes.is_none(),
    }
}

/// Map an unexpected HTTP status to a retry or a fatal error.
fn status_attempt(status: StatusCode, what: &str, name: &str) -> Attempt {
    match status.as_u16() {
        // The signed URL expired (403 from the CDN) or the server is struggling: ask again, from the top.
        401 | 403 | 408 | 410 | 425 | 429 | 500..=599 => {
            Attempt::Retry(DataError::Network(format!("the server answered {status} while {what} {name}")))
        }
        404 => Attempt::Fatal(DataError::NotFound(format!("{name} is no longer available at its download address"))),
        _ => Attempt::Fatal(DataError::Network(format!("the server answered {status} while {what} {name}"))),
    }
}

/// A ranged `GET` must be a `206` whose `Content-Range` is exactly what was asked for.
fn check_ranged_response(response: &Response, file: &RemoteFile, start: u64, last: u64) -> Result<(), Attempt> {
    let status = response.status();
    if status == StatusCode::OK {
        return Err(Attempt::Fatal(DataError::Network(format!(
            "the server ignored the byte range for {} and tried to send the whole file; refusing it",
            file.name
        ))));
    }
    if status != StatusCode::PARTIAL_CONTENT {
        return Err(status_attempt(status, "downloading", &file.name));
    }
    let content_range =
        response.headers().get(CONTENT_RANGE).and_then(|v| v.to_str().ok()).and_then(parse_content_range).ok_or_else(
            || Attempt::Fatal(DataError::Network(format!("the server sent no usable Content-Range for {}", file.name))),
        )?;
    if content_range != (start, last, file.size) {
        return Err(Attempt::Fatal(DataError::Network(format!(
            "unexpected Content-Range for {}: got bytes {}-{}/{}, asked for {start}-{last} of {}",
            file.name, content_range.0, content_range.1, content_range.2, file.size
        ))));
    }
    if let Some(len) = response.content_length()
        && len != last - start + 1
    {
        return Err(Attempt::Fatal(DataError::Network(format!(
            "Content-Length {len} does not match the requested range for {}",
            file.name
        ))));
    }
    Ok(())
}

/// `bytes 0-99/1000` into `(0, 99, 1000)`.
pub(crate) fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.trim().strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    Some((start.trim().parse().ok()?, end.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// An ETag with its quotes and weak marker removed.
pub(crate) fn normalize_etag(raw: &str) -> String {
    raw.trim().trim_start_matches("W/").trim_matches('"').to_string()
}

/// End (exclusive) of the last whole story in `window`: just after the last `"\n<|endoftext|>\n"`.
pub(crate) fn story_cut(window: &[u8]) -> Option<usize> {
    memchr::memmem::rfind(window, STORY_BOUNDARY).map(|i| i + STORY_BOUNDARY.len())
}

pub(crate) fn shard_name(idx: u32) -> String {
    format!("part-{idx:04}.txt")
}

/// Write a shard as `.part`, sync it, and rename it into place.
async fn write_shard(dir: &Path, idx: u32, data: &[u8]) -> io::Result<()> {
    let path = dir.join(shard_name(idx));
    let tmp = dir.join(format!("{}.part", shard_name(idx)));
    let mut file = tokio::fs::File::create(&tmp).await?;
    file.write_all(data).await?;
    file.sync_data().await?;
    drop(file);
    tokio::fs::rename(&tmp, &path).await
}

/// Hash the first `count` shards of `dir` and check that together they hold exactly `expected_bytes`.
async fn replay_shards(dir: &Path, count: u32, expected_bytes: u64) -> io::Result<Sha256> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        let mut chunk = vec![0u8; 1 << 20];
        for idx in 0..count {
            let mut file = std::fs::File::open(dir.join(shard_name(idx)))?;
            loop {
                let n = io::Read::read(&mut file, &mut chunk)?;
                if n == 0 {
                    break;
                }
                hasher.update(&chunk[..n]);
                total += n as u64;
            }
        }
        if total != expected_bytes && count > 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "saved shards do not add up to the saved cursor"));
        }
        Ok(hasher)
    })
    .await
    .map_err(io::Error::other)?
}

/// Total size of the first `count` shards of `dir`; an error if one is missing.
async fn shards_total_size(dir: &Path, count: u32) -> io::Result<u64> {
    let mut total = 0;
    for idx in 0..count {
        total += tokio::fs::metadata(dir.join(shard_name(idx))).await?.len();
    }
    Ok(total)
}

/// Remove every shard (finished or partial) in `dir`.
async fn clear_shards(dir: &Path) -> DataResult<()> {
    remove_matching(dir, |name| name.starts_with("part-")).await
}

/// Remove unfinished `.part` shards left by a kill.
async fn remove_partial_shards(dir: &Path) -> DataResult<()> {
    remove_matching(dir, |name| name.starts_with("part-") && name.ends_with(".part")).await
}

async fn remove_matching(dir: &Path, matches: impl Fn(&str) -> bool) -> DataResult<()> {
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(DataError::io(dir, e)),
    };
    while let Some(entry) = entries.next_entry().await.map_err(|e| DataError::io(dir, e))? {
        if matches(&entry.file_name().to_string_lossy()) {
            tokio::fs::remove_file(entry.path()).await.map_err(|e| DataError::io(entry.path(), e))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_range() {
        assert_eq!(parse_content_range("bytes 0-1999/19447282"), Some((0, 1999, 19447282)));
        assert_eq!(parse_content_range(" bytes 5-9/10 "), Some((5, 9, 10)));
        assert_eq!(parse_content_range("bytes */10"), None);
        assert_eq!(parse_content_range("bytes 0-9/*"), None);
        assert_eq!(parse_content_range("items 0-9/10"), None);
        assert_eq!(parse_content_range(""), None);
    }

    #[test]
    fn normalizes_etags() {
        assert_eq!(normalize_etag("\"abc\""), "abc");
        assert_eq!(normalize_etag("W/\"abc\""), "abc");
        assert_eq!(normalize_etag("abc"), "abc");
    }

    #[test]
    fn story_cut_is_after_the_last_boundary() {
        let text = b"one\n<|endoftext|>\ntwo\n<|endoftext|>\nthree partial";
        let cut = story_cut(text).unwrap();
        assert_eq!(&text[..cut], b"one\n<|endoftext|>\ntwo\n<|endoftext|>\n");
        assert_eq!(story_cut(b"no marker here"), None);
        // A marker cut in half by the window end is not a boundary.
        assert_eq!(story_cut(b"story\n<|endoftext|"), None);
        // Back-to-back markers share a newline.
        let double = b"a\n<|endoftext|>\n<|endoftext|>\nb";
        assert_eq!(&double[..story_cut(double).unwrap()], b"a\n<|endoftext|>\n<|endoftext|>\n");
    }

    #[test]
    fn effective_bytes_and_dirs() {
        let f = RemoteFile {
            url: "http://x/y".into(),
            name: "y".into(),
            lane: "stories".into(),
            side: Side::Val,
            size: 100,
            sha256: String::new(),
            prefix_bytes: Some(40),
        };
        assert_eq!(f.effective_bytes(), 40);
        assert_eq!(RemoteFile { prefix_bytes: Some(400), ..f.clone() }.effective_bytes(), 100);
        assert_eq!(RemoteFile { prefix_bytes: None, ..f.clone() }.effective_bytes(), 100);
        assert_eq!(f.shard_dir(Path::new("/d")), Path::new("/d/val/stories"));
    }

    #[test]
    fn state_round_trips_and_ignores_garbage() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(DownloadState::load(dir.path()).is_none());
        let mut s = DownloadState::new("tinystories-quick");
        let file = FileState {
            source_offset: 7,
            kept_bytes: 7,
            shard_idx: 2,
            etag: Some("e".into()),
            complete: false,
            sha256: None,
        };
        s.files.insert("a.txt".into(), file);
        s.save(dir.path()).unwrap();
        assert_eq!(DownloadState::load(dir.path()), Some(s));
        assert!(!dir.path().join(format!("{STATE_FILE}.tmp")).exists());
        std::fs::write(dir.path().join(STATE_FILE), b"{ not json").unwrap();
        assert!(DownloadState::load(dir.path()).is_none());
        std::fs::write(dir.path().join(STATE_FILE), br#"{"version":99,"starterId":"x","files":{}}"#).unwrap();
        assert!(DownloadState::load(dir.path()).is_none(), "an unknown version is ignored");
    }

    #[tokio::test]
    async fn shards_are_written_atomically_and_replayed() {
        let dir = tempfile::TempDir::new().unwrap();
        write_shard(dir.path(), 0, b"hello ").await.unwrap();
        write_shard(dir.path(), 1, b"world").await.unwrap();
        assert!(dir.path().join("part-0000.txt").is_file() && dir.path().join("part-0001.txt").is_file());
        assert!(!dir.path().join("part-0000.txt.part").exists());
        let hasher = replay_shards(dir.path(), 2, 11).await.unwrap();
        assert_eq!(hex(&hasher.finalize()), hex(&Sha256::digest(b"hello world")));
        assert!(replay_shards(dir.path(), 2, 12).await.is_err(), "size mismatch");
        assert!(replay_shards(dir.path(), 3, 11).await.is_err(), "missing shard");
        std::fs::write(dir.path().join("part-0002.txt.part"), b"x").unwrap();
        remove_partial_shards(dir.path()).await.unwrap();
        assert!(!dir.path().join("part-0002.txt.part").exists() && dir.path().join("part-0001.txt").exists());
        clear_shards(dir.path()).await.unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
