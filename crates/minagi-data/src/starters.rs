//! The starter datasets the app offers, and how to install them.
//!
//! | id                 | what                                                                   | network |
//! |--------------------|------------------------------------------------------------------------|---------|
//! | `tinystories-quick`| the first 64 MiB of TinyStories (train) plus its whole validation file | yes     |
//! | `tinystories-full` | all of TinyStories (1.9 GB) plus its validation file                   | yes     |
//! | `arithmetic`       | synthetic arithmetic problems, generated locally                       | no      |
//! | `sampler`          | a few short bundled stories plus a small arithmetic lane               | no      |
//!
//! [`install_starter`] builds the dataset directory (`train/<lane>/part-NNNN.txt`, `val/<lane>/part-NNNN.txt`,
//! `manifest.json`) in `dest`. Downloads are resumable: see [`crate::download`].

use std::path::{Path, PathBuf};

use minagi_types::{ArithmeticParams, Count, DatasetKind, JobProgress, LaneInfo, StarterInfo, UnixMs};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::download::{DownloadOptions, DownloadState, FileOutcome, RemoteFile, STATE_FILE, Session, Side, shard_name};
use crate::error::{DataError, DataResult};
use crate::lanes::assign_color_slots;
use crate::materialize::LaneReport;
use crate::synth::arithmetic;
use crate::util::{CancelFlag, hex};

/// The pinned TinyStories revision every download comes from.
pub const TINYSTORIES_REVISION: &str = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64";
/// Base URL of the pinned revision; file names are appended.
pub const TINYSTORIES_BASE_URL: &str =
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64";
/// How much of the TinyStories training file the quick starter reads.
pub const QUICK_PREFIX_BYTES: u64 = 64 * 1024 * 1024;

const TRAIN_FILE: &str = "TinyStories-train.txt";
const TRAIN_BYTES: u64 = 1_924_281_556;
const TRAIN_SHA256: &str = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f";
const VALID_FILE: &str = "TinyStories-valid.txt";
const VALID_BYTES: u64 = 19_447_282;
const VALID_SHA256: &str = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4";

/// Text of the bundled `sampler` stories (training and validation halves), in the TinyStories shape: one story per
/// block, blocks separated by a line holding `<|endoftext|>`.
const SAMPLER_TRAIN_TEXT: &str = include_str!("../assets/sampler_stories_train.txt");
const SAMPLER_VAL_TEXT: &str = include_str!("../assets/sampler_stories_val.txt");

/// What installing a starter involves.
#[derive(Debug, Clone, PartialEq)]
pub enum Recipe {
    /// Download these files.
    Download(Vec<RemoteFile>),
    /// Generate arithmetic problems.
    Arithmetic(ArithmeticParams),
    /// Write the bundled stories and generate a small arithmetic lane.
    Sampler(ArithmeticParams),
}

/// A starter dataset: what the UI shows and how to build it.
#[derive(Debug, Clone, PartialEq)]
pub struct Starter {
    pub info: StarterInfo,
    pub recipe: Recipe,
}

impl Starter {
    pub fn id(&self) -> &str {
        &self.info.id
    }

    /// The kind a dataset created from this starter has.
    pub fn kind(&self) -> DatasetKind {
        match self.recipe {
            Recipe::Download(_) => DatasetKind::Starter,
            Recipe::Arithmetic(_) => DatasetKind::Generated,
            Recipe::Sampler(_) => DatasetKind::Bundled,
        }
    }

    /// Point the downloads at another server: every file becomes `<base_url>/<file name>`. Used by tests, and by
    /// anyone mirroring the files.
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        if let Recipe::Download(files) = &mut self.recipe {
            let base = base_url.trim_end_matches('/');
            for file in files {
                file.url = format!("{base}/{}", file.name);
            }
        }
        self
    }
}

fn tinystories_files(prefix: Option<u64>) -> Vec<RemoteFile> {
    let file = |name: &str, side, size, sha: &str, prefix| RemoteFile {
        url: format!("{TINYSTORIES_BASE_URL}/{name}"),
        name: name.to_string(),
        lane: "stories".to_string(),
        side,
        size,
        sha256: sha.to_string(),
        prefix_bytes: prefix,
    };
    vec![
        file(TRAIN_FILE, Side::Train, TRAIN_BYTES, TRAIN_SHA256, prefix),
        file(VALID_FILE, Side::Val, VALID_BYTES, VALID_SHA256, None),
    ]
}

const TINYSTORIES_ATTRIBUTION: &str = "Ronen Eldan and Yuanzhi Li, \"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?\" (Microsoft Research, 2023)";
const TINYSTORIES_PAGE: &str = "https://huggingface.co/datasets/roneneldan/TinyStories";
const OWN_ATTRIBUTION: &str =
    "Generated on your computer with a port of the arithmetic generator from mini-AGI by Alexey Borsky (MIT License)";

/// Parameters of the standalone arithmetic starter.
fn arithmetic_starter_params() -> ArithmeticParams {
    ArithmeticParams::default()
}

/// Parameters of the small arithmetic lane inside the sampler.
fn sampler_arithmetic_params() -> ArithmeticParams {
    ArithmeticParams { train_problems: 4_000, val_problems: 200, seed: 7, notes_frac: 0.3 }
}

/// Every starter, with its recipe.
pub fn all_starters() -> Vec<Starter> {
    let arithmetic = arithmetic_starter_params();
    let sampler = sampler_arithmetic_params();
    vec![
        Starter {
            info: StarterInfo {
                id: "tinystories-quick".into(),
                title: "TinyStories, quick start".into(),
                description: "The first 64 MB of TinyStories, simple stories written for young children, plus its validation \
                              set. A good first run: it downloads in a few minutes and a small model learns to tell short \
                              stories from it."
                    .into(),
                size_bytes: Count(QUICK_PREFIX_BYTES + VALID_BYTES),
                license: "CDLA-Sharing-1.0".into(),
                attribution: TINYSTORIES_ATTRIBUTION.into(),
                source_url: TINYSTORIES_PAGE.into(),
                offline: false,
            },
            recipe: Recipe::Download(tinystories_files(Some(QUICK_PREFIX_BYTES))),
        },
        Starter {
            info: StarterInfo {
                id: "tinystories-full".into(),
                title: "TinyStories, full".into(),
                description: "All of TinyStories (about 2 million short stories, 1.9 GB) plus its validation set. Needs a \
                              fast connection and about 2.4 GB of free space."
                    .into(),
                size_bytes: Count(TRAIN_BYTES + VALID_BYTES),
                license: "CDLA-Sharing-1.0".into(),
                attribution: TINYSTORIES_ATTRIBUTION.into(),
                source_url: TINYSTORIES_PAGE.into(),
                offline: false,
            },
            recipe: Recipe::Download(tinystories_files(None)),
        },
        Starter {
            info: StarterInfo {
                id: "arithmetic".into(),
                title: "Arithmetic".into(),
                description: "Sums, differences, products, comparisons, remainders and more, generated on your computer. \
                              Some lines show their working, so you can watch a model learn to carry digits."
                    .into(),
                size_bytes: Count(estimated_arithmetic_bytes(&arithmetic)),
                license: "CC0-1.0".into(),
                attribution: OWN_ATTRIBUTION.into(),
                source_url: "https://github.com/volotat/mini-AGI".into(),
                offline: true,
            },
            recipe: Recipe::Arithmetic(arithmetic),
        },
        Starter {
            info: StarterInfo {
                id: "sampler".into(),
                title: "Sampler".into(),
                description: "A few short fables and a little arithmetic that come with the app, so you can try a first \
                              training run in seconds without downloading anything."
                    .into(),
                size_bytes: Count(SAMPLER_TRAIN_TEXT.len() as u64 + SAMPLER_VAL_TEXT.len() as u64 + estimated_arithmetic_bytes(&sampler)),
                license: "CC0-1.0".into(),
                attribution: "Stories written for LLM Trainer; arithmetic generated with a port of the generator from mini-AGI by Alexey Borsky (MIT License)".into(),
                source_url: "https://github.com/volotat/mini-AGI".into(),
                offline: true,
            },
            recipe: Recipe::Sampler(sampler),
        },
    ]
}

/// Average bytes per generated line (measured over many seeds), used for the size shown in the UI.
const ARITHMETIC_BYTES_PER_PROBLEM: u64 = 37;

fn estimated_arithmetic_bytes(p: &ArithmeticParams) -> u64 {
    (u64::from(p.train_problems) + u64::from(p.val_problems)) * ARITHMETIC_BYTES_PER_PROBLEM
}

/// What the UI lists.
pub fn starters() -> Vec<StarterInfo> {
    all_starters().into_iter().map(|s| s.info).collect()
}

/// Look a starter up by id.
pub fn find_starter(id: &str) -> Option<Starter> {
    all_starters().into_iter().find(|s| s.info.id == id)
}

/// How a source file's bytes were verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Integrity {
    /// The whole file's SHA-256 matched the published one.
    Sha256Verified,
    /// Only a prefix was used: size, identity and encoding were checked and the prefix's own hash recorded.
    PrefixRecorded,
    /// Generated or bundled text; there is nothing to verify against.
    Local,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestLane {
    name: String,
    display_name: String,
    color_slot: u32,
    train_files: u64,
    train_bytes: u64,
    val_files: u64,
    val_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestSource {
    /// Name of the remote file or generated part.
    name: String,
    side: Side,
    lane: String,
    url: Option<String>,
    published_size: Option<u64>,
    published_sha256: Option<String>,
    prefix_bytes: Option<u64>,
    shards: u32,
    kept_bytes: u64,
    /// SHA-256 of the shards laid end to end.
    kept_sha256: String,
    integrity: Integrity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StarterManifest {
    version: u32,
    kind: String,
    starter_id: String,
    title: String,
    created_at: UnixMs,
    license: String,
    attribution: String,
    source_url: String,
    lanes: Vec<ManifestLane>,
    sources: Vec<ManifestSource>,
    generated: Option<ArithmeticParams>,
}

/// Name of the manifest at the top of a prepared dataset.
pub use crate::materialize::MANIFEST_FILE;

/// What [`install_starter`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallReport {
    pub dest: PathBuf,
    pub starter_id: String,
    pub lanes: Vec<LaneReport>,
    pub train_bytes: u64,
    pub val_bytes: u64,
    /// Bytes fetched over the network in this run (0 for generated starters and for a repeat install).
    pub downloaded_bytes: u64,
    /// True when this run continued an earlier, interrupted download.
    pub resumed: bool,
    /// True when `dest` already held this starter completely and nothing was done.
    pub already_installed: bool,
}

impl InstallReport {
    /// The lane table for the UI (`n_files` counts shards).
    pub fn lane_infos(&self) -> Vec<LaneInfo> {
        let names: Vec<&str> = self.lanes.iter().map(|l| l.name.as_str()).collect();
        let slots = assign_color_slots(&names);
        self.lanes
            .iter()
            .map(|l| LaneInfo {
                name: l.name.clone(),
                display_name: l.name.clone(),
                color_slot: slots.get(&l.name).copied().unwrap_or(crate::lanes::OTHER_COLOR_SLOT),
                enabled: true,
                n_files: Count(l.train_files),
                train_bytes: Count(l.train_bytes),
                val_files: Count(l.val_files),
                val_bytes: Count(l.val_bytes),
                sample_prompt: None,
            })
            .collect()
    }
}

/// [`install_starter_with`] with the default options.
pub async fn install_starter(
    starter: &Starter,
    dest: &Path,
    client: &Client,
    cancel: CancellationToken,
    progress: impl FnMut(JobProgress),
) -> DataResult<InstallReport> {
    install_starter_with(starter, dest, client, &DownloadOptions::default(), cancel, progress).await
}

/// Build `dest` from `starter`.
///
/// Downloads resume from `dest/.download-state.json` after a cancel, a crash or a lost connection; cancelling
/// (`cancel`) returns [`DataError::Cancelled`] with the state kept. `client` must not follow redirects (see
/// [`crate::http_client`]); generated and bundled starters never touch it. Progress is throttled to ten calls a
/// second. Installing into a folder that already holds the finished starter does nothing.
pub async fn install_starter_with<P: FnMut(JobProgress)>(
    starter: &Starter,
    dest: &Path,
    client: &Client,
    opts: &DownloadOptions,
    cancel: CancellationToken,
    mut progress: P,
) -> DataResult<InstallReport> {
    tokio::fs::create_dir_all(dest).await.map_err(|e| DataError::io(dest, e))?;
    if let Some(report) = already_installed(starter, dest).await {
        return Ok(report);
    }
    match &starter.recipe {
        Recipe::Download(files) => install_download(starter, files, dest, client, opts, &cancel, progress).await,
        Recipe::Arithmetic(params) => {
            let params = *params;
            let dest_owned = dest.to_path_buf();
            let report = run_blocking(&cancel, &mut progress, move |flag, progress| {
                let r = arithmetic::generate(&params, &dest_owned, flag, progress)?;
                Ok(vec![
                    generated_source(Side::Train, "arithmetic", r.train_shards, r.train_bytes, &dest_owned)?,
                    generated_source(Side::Val, "arithmetic", r.val_shards, r.val_bytes, &dest_owned)?,
                ])
            })
            .await?;
            finish_generated(starter, dest, report, Some(params)).await
        }
        Recipe::Sampler(params) => {
            let params = *params;
            let dest_owned = dest.to_path_buf();
            let sources = run_blocking(&cancel, &mut progress, move |flag, progress| {
                let mut out = write_sampler_stories(&dest_owned)?;
                flag.check()?;
                let r = arithmetic::generate(&params, &dest_owned, flag, progress)?;
                out.push(generated_source(Side::Train, "arithmetic", r.train_shards, r.train_bytes, &dest_owned)?);
                out.push(generated_source(Side::Val, "arithmetic", r.val_shards, r.val_bytes, &dest_owned)?);
                Ok(out)
            })
            .await?;
            finish_generated(starter, dest, sources, Some(params)).await
        }
    }
}

/// A finished install of this starter in `dest`, if there is one.
async fn already_installed(starter: &Starter, dest: &Path) -> Option<InstallReport> {
    if dest.join(STATE_FILE).exists() {
        return None;
    }
    let manifest = read_manifest(dest).await.ok()?;
    if manifest.starter_id != starter.info.id {
        return None;
    }
    for source in &manifest.sources {
        let dir = dest.join(source.side.dir()).join(&source.lane);
        let mut total = 0;
        for idx in 0..source.shards {
            total += tokio::fs::metadata(dir.join(shard_name(idx))).await.ok()?.len();
        }
        if total != source.kept_bytes {
            return None;
        }
    }
    Some(report_from_manifest(dest, &manifest, false, true))
}

fn report_from_manifest(dest: &Path, m: &StarterManifest, resumed: bool, already: bool) -> InstallReport {
    let lanes: Vec<LaneReport> = m
        .lanes
        .iter()
        .map(|l| LaneReport {
            name: l.name.clone(),
            train_files: l.train_files,
            train_bytes: l.train_bytes,
            val_files: l.val_files,
            val_bytes: l.val_bytes,
        })
        .collect();
    InstallReport {
        dest: dest.to_path_buf(),
        starter_id: m.starter_id.clone(),
        train_bytes: lanes.iter().map(|l| l.train_bytes).sum(),
        val_bytes: lanes.iter().map(|l| l.val_bytes).sum(),
        lanes,
        downloaded_bytes: 0,
        resumed,
        already_installed: already,
    }
}

async fn read_manifest(dest: &Path) -> DataResult<StarterManifest> {
    let path = dest.join(MANIFEST_FILE);
    let bytes = tokio::fs::read(&path).await.map_err(|e| DataError::io(&path, e))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| DataError::invalid(format!("{} is not a starter manifest: {e}", path.display())))
}

async fn install_download<P: FnMut(JobProgress)>(
    starter: &Starter,
    files: &[RemoteFile],
    dest: &Path,
    client: &Client,
    opts: &DownloadOptions,
    cancel: &CancellationToken,
    progress: P,
) -> DataResult<InstallReport> {
    if files.is_empty() {
        return Err(DataError::invalid("this starter has nothing to download"));
    }
    let state = DownloadState::load(dest)
        .filter(|s| s.starter_id == starter.info.id)
        .unwrap_or_else(|| DownloadState::new(&starter.info.id));
    let resumed = state.files.values().any(|f| f.shard_idx > 0 || f.complete);
    let mut session = Session::new(client, opts, cancel, progress, dest, state, files);
    session.check_space(session.remaining_bytes(files))?;

    let mut outcomes = Vec::with_capacity(files.len());
    for file in files {
        outcomes.push(session.download_file(file).await?);
    }
    let downloaded = session.fetched;

    let sources: Vec<ManifestSource> = outcomes
        .iter()
        .zip(files)
        .map(|(o, f)| ManifestSource {
            name: o.name.clone(),
            side: o.side,
            lane: o.lane.clone(),
            url: Some(f.url.clone()),
            published_size: Some(f.size),
            published_sha256: Some(f.sha256.clone()),
            prefix_bytes: f.prefix_bytes,
            shards: o.shards,
            kept_bytes: o.kept_bytes,
            kept_sha256: o.sha256.clone(),
            integrity: integrity_of(o),
        })
        .collect();
    let manifest = build_manifest(starter, sources, None);
    write_manifest(dest, &manifest).await?;
    // The download is complete; the cursor has done its job.
    let _ = tokio::fs::remove_file(dest.join(STATE_FILE)).await;
    let mut report = report_from_manifest(dest, &manifest, resumed, false);
    report.downloaded_bytes = downloaded;
    Ok(report)
}

fn integrity_of(o: &FileOutcome) -> Integrity {
    if o.verified_against_published { Integrity::Sha256Verified } else { Integrity::PrefixRecorded }
}

fn build_manifest(
    starter: &Starter,
    sources: Vec<ManifestSource>,
    generated: Option<ArithmeticParams>,
) -> StarterManifest {
    let mut lanes: Vec<ManifestLane> = Vec::new();
    for s in &sources {
        let idx = match lanes.iter().position(|l| l.name == s.lane) {
            Some(i) => i,
            None => {
                lanes.push(ManifestLane {
                    name: s.lane.clone(),
                    display_name: s.lane.clone(),
                    color_slot: 0,
                    train_files: 0,
                    train_bytes: 0,
                    val_files: 0,
                    val_bytes: 0,
                });
                lanes.len() - 1
            }
        };
        let lane = &mut lanes[idx];
        match s.side {
            Side::Train => {
                lane.train_files += u64::from(s.shards);
                lane.train_bytes += s.kept_bytes;
            }
            Side::Val => {
                lane.val_files += u64::from(s.shards);
                lane.val_bytes += s.kept_bytes;
            }
        }
    }
    lanes.sort_by(|a, b| a.name.cmp(&b.name));
    let names: Vec<&str> = lanes.iter().map(|l| l.name.as_str()).collect();
    let slots = assign_color_slots(&names);
    for lane in &mut lanes {
        lane.color_slot = slots.get(&lane.name).copied().unwrap_or(crate::lanes::OTHER_COLOR_SLOT);
    }
    StarterManifest {
        version: 1,
        kind: "starter".into(),
        starter_id: starter.info.id.clone(),
        title: starter.info.title.clone(),
        created_at: UnixMs::now(),
        license: starter.info.license.clone(),
        attribution: starter.info.attribution.clone(),
        source_url: starter.info.source_url.clone(),
        lanes,
        sources,
        generated,
    }
}

/// Write `manifest.json` atomically.
async fn write_manifest(dest: &Path, manifest: &StarterManifest) -> DataResult<()> {
    let path = dest.join(MANIFEST_FILE);
    let tmp = dest.join(format!("{MANIFEST_FILE}.tmp"));
    let json = serde_json::to_vec_pretty(manifest).map_err(|e| DataError::io(&path, std::io::Error::other(e)))?;
    tokio::fs::write(&tmp, json).await.map_err(|e| DataError::io(&tmp, e))?;
    tokio::fs::rename(&tmp, &path).await.map_err(|e| DataError::io(&path, e))
}

async fn finish_generated(
    starter: &Starter,
    dest: &Path,
    sources: Vec<ManifestSource>,
    generated: Option<ArithmeticParams>,
) -> DataResult<InstallReport> {
    let manifest = build_manifest(starter, sources, generated);
    write_manifest(dest, &manifest).await?;
    Ok(report_from_manifest(dest, &manifest, false, false))
}

/// Describe shards that were written locally (hashing them so [`verify_installed`] can check them later).
fn generated_source(side: Side, lane: &str, shards: u32, bytes: u64, dest: &Path) -> DataResult<ManifestSource> {
    let dir = dest.join(side.dir()).join(lane);
    let sha = hash_shards(&dir, shards)?;
    Ok(ManifestSource {
        name: format!("{lane}-{}", side.dir()),
        side,
        lane: lane.to_string(),
        url: None,
        published_size: None,
        published_sha256: None,
        prefix_bytes: None,
        shards,
        kept_bytes: bytes,
        kept_sha256: sha,
        integrity: Integrity::Local,
    })
}

/// SHA-256 of the first `count` shards of `dir`, laid end to end.
fn hash_shards(dir: &Path, count: u32) -> DataResult<String> {
    let mut hasher = Sha256::new();
    for idx in 0..count {
        let path = dir.join(shard_name(idx));
        let mut file = std::fs::File::open(&path).map_err(|e| DataError::io(&path, e))?;
        std::io::copy(&mut file, &mut hasher).map_err(|e| DataError::io(&path, e))?;
    }
    Ok(hex(&hasher.finalize()))
}

/// Write the bundled sampler stories as the `stories` lane.
fn write_sampler_stories(dest: &Path) -> DataResult<Vec<ManifestSource>> {
    let mut out = Vec::new();
    for (side, text) in [(Side::Train, SAMPLER_TRAIN_TEXT), (Side::Val, SAMPLER_VAL_TEXT)] {
        let dir = dest.join(side.dir()).join("stories");
        std::fs::create_dir_all(&dir).map_err(|e| DataError::io(&dir, e))?;
        let path = dir.join(shard_name(0));
        let tmp = dir.join(format!("{}.part", shard_name(0)));
        std::fs::write(&tmp, text).map_err(|e| DataError::io(&tmp, e))?;
        std::fs::rename(&tmp, &path).map_err(|e| DataError::io(&path, e))?;
        out.push(generated_source(side, "stories", 1, text.len() as u64, dest)?);
    }
    Ok(out)
}

/// Run blocking work on a worker thread, forwarding its progress and turning `cancel` into the job's [`CancelFlag`].
async fn run_blocking<T, F>(cancel: &CancellationToken, progress: &mut impl FnMut(JobProgress), job: F) -> DataResult<T>
where
    T: Send + 'static,
    F: FnOnce(&CancelFlag, &mut dyn FnMut(JobProgress)) -> DataResult<T> + Send + 'static,
{
    let flag = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<JobProgress>();
    let job_flag = flag.clone();
    let mut handle = tokio::task::spawn_blocking(move || {
        job(&job_flag, &mut |p| {
            let _ = tx.send(p);
        })
    });
    let result = loop {
        tokio::select! {
            Some(p) = rx.recv() => progress(p),
            _ = cancel.cancelled(), if !flag.is_cancelled() => flag.cancel(),
            joined = &mut handle => break joined,
        }
    };
    while let Ok(p) = rx.try_recv() {
        progress(p);
    }
    result.map_err(|e| DataError::Io { path: None, source: std::io::Error::other(e) })?
}

/// What [`verify_installed`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub files_checked: u32,
    pub bytes_checked: u64,
    /// One plain-English line per problem; empty when everything checks out.
    pub problems: Vec<String>,
}

impl VerifyReport {
    pub fn is_ok(&self) -> bool {
        self.problems.is_empty()
    }
}

/// Re-read an installed starter and check every shard against `manifest.json`: shard counts, sizes, the SHA-256 of
/// the shards laid end to end, and (for whole files) the published checksum.
pub fn verify_installed(dest: &Path) -> DataResult<VerifyReport> {
    let bytes = std::fs::read(dest.join(MANIFEST_FILE)).map_err(|e| DataError::io(dest.join(MANIFEST_FILE), e))?;
    let manifest: StarterManifest = serde_json::from_slice(&bytes)
        .map_err(|e| DataError::invalid(format!("{MANIFEST_FILE} is not a starter manifest: {e}")))?;
    let mut report = VerifyReport { files_checked: 0, bytes_checked: 0, problems: Vec::new() };
    for source in &manifest.sources {
        let dir = dest.join(source.side.dir()).join(&source.lane);
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        let mut missing = false;
        for idx in 0..source.shards {
            let path = dir.join(shard_name(idx));
            match std::fs::File::open(&path) {
                Ok(mut f) => total += std::io::copy(&mut f, &mut hasher).map_err(|e| DataError::io(&path, e))?,
                Err(_) => {
                    report.problems.push(format!("{} is missing", path.display()));
                    missing = true;
                    break;
                }
            }
        }
        report.files_checked += 1;
        report.bytes_checked += total;
        if missing {
            continue;
        }
        if total != source.kept_bytes {
            report.problems.push(format!("{}: expected {} bytes but found {total}", source.name, source.kept_bytes));
        }
        let actual = hex(&hasher.finalize());
        if actual != source.kept_sha256 {
            report
                .problems
                .push(format!("{}: contents differ from what was installed (checksum mismatch)", source.name));
        }
        if source.integrity == Integrity::Sha256Verified
            && let Some(published) = &source.published_sha256
            && !actual.eq_ignore_ascii_case(published)
        {
            report.problems.push(format!("{}: does not match the published checksum", source.name));
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_starters_have_the_documented_ids_and_flags() {
        let list = starters();
        let ids: Vec<&str> = list.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["tinystories-quick", "tinystories-full", "arithmetic", "sampler"]);
        let offline: Vec<bool> = list.iter().map(|s| s.offline).collect();
        assert_eq!(offline, [false, false, true, true]);
        assert!(list.iter().all(|s| !s.title.is_empty()
            && !s.description.is_empty()
            && !s.license.is_empty()
            && !s.attribution.is_empty()
            && s.size_bytes.0 > 0));
        assert_eq!(list[0].license, "CDLA-Sharing-1.0");
        assert!(find_starter("sampler").is_some() && find_starter("nope").is_none());
    }

    #[test]
    fn tinystories_sources_are_pinned_with_the_published_facts() {
        let quick = find_starter("tinystories-quick").unwrap();
        let Recipe::Download(files) = &quick.recipe else { panic!("quick is a download") };
        assert_eq!(files.len(), 2);
        let (train, val) = (&files[0], &files[1]);
        assert_eq!((train.side, val.side), (Side::Train, Side::Val));
        assert_eq!(train.prefix_bytes, Some(67_108_864));
        assert_eq!(val.prefix_bytes, None);
        assert_eq!((train.size, val.size), (1_924_281_556, 19_447_282));
        assert_eq!(train.sha256, "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f");
        assert_eq!(val.sha256, "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4");
        assert!(train.url.contains(TINYSTORIES_REVISION) && train.url.ends_with("/TinyStories-train.txt"));
        assert!(!train.url.contains("/main/"), "the revision is pinned");
        assert!(files.iter().all(|f| f.lane == "stories"));
        assert_eq!(quick.kind(), DatasetKind::Starter);
        assert_eq!(quick.info.size_bytes, Count(67_108_864 + 19_447_282));

        let full = find_starter("tinystories-full").unwrap();
        let Recipe::Download(files) = &full.recipe else { panic!("full is a download") };
        assert!(files.iter().all(|f| f.prefix_bytes.is_none()));
        assert_eq!(full.info.size_bytes, Count(1_924_281_556 + 19_447_282));
    }

    #[test]
    fn kinds_follow_the_recipe() {
        assert_eq!(find_starter("arithmetic").unwrap().kind(), DatasetKind::Generated);
        assert_eq!(find_starter("sampler").unwrap().kind(), DatasetKind::Bundled);
    }

    #[test]
    fn with_base_url_rewrites_every_download() {
        let s = find_starter("tinystories-quick").unwrap().with_base_url("http://127.0.0.1:9/some/dir/");
        let Recipe::Download(files) = &s.recipe else { panic!() };
        assert_eq!(files[0].url, "http://127.0.0.1:9/some/dir/TinyStories-train.txt");
        assert_eq!(files[1].url, "http://127.0.0.1:9/some/dir/TinyStories-valid.txt");
        let a = find_starter("arithmetic").unwrap();
        assert_eq!(a.clone().with_base_url("http://x"), a, "generated starters have no URLs");
    }

    #[test]
    fn the_arithmetic_size_estimate_is_realistic() {
        // Measure the real average line length and compare with the constant used for the estimate.
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let n = 50_000;
        let total: usize = (0..n).map(|_| arithmetic::sample_line(&mut rng, 0.3).len() + 1).sum();
        let avg = total as f64 / n as f64;
        let assumed = ARITHMETIC_BYTES_PER_PROBLEM as f64;
        assert!((avg - assumed).abs() / assumed < 0.1, "average line is {avg:.1} bytes, estimate uses {assumed}");
    }

    #[test]
    fn the_bundled_sampler_text_is_in_the_expected_shape() {
        for (name, text) in [("train", SAMPLER_TRAIN_TEXT), ("val", SAMPLER_VAL_TEXT)] {
            assert!(text.ends_with("\n<|endoftext|>\n"), "{name} ends with a story separator");
            assert!(text.matches("\n<|endoftext|>\n").count() >= 2, "{name} has several stories");
            assert!(!text.contains('\r') && !text.contains('\0'));
        }
        let total = SAMPLER_TRAIN_TEXT.len() + SAMPLER_VAL_TEXT.len();
        assert!((30_000..=60_000).contains(&total), "bundled text is {total} bytes");
        assert!(SAMPLER_VAL_TEXT.len() * 20 > SAMPLER_TRAIN_TEXT.len(), "validation is at least about 5 % of the text");
    }
}
