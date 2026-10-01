//! The starter downloader against a local server that behaves like Hugging Face's `resolve` endpoint and its CDN.

mod support;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use minagi_data::download::{DownloadState, STATE_FILE};
use minagi_data::{
    DataError, DownloadOptions, Recipe, RemoteFile, Side, Starter, http_client, install_starter, install_starter_with,
    verify_installed,
};
use minagi_types::{AppError, Count, JobProgress, StarterInfo};
use reqwest::Client;
use support::{BOUNDARY, Server, expected_shards, sha256_hex, stories};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

const SHARD: usize = 16 * 1024;

fn test_opts() -> DownloadOptions {
    DownloadOptions {
        shard_bytes: SHARD as u64,
        max_attempts: 5,
        backoff_base: Duration::from_millis(5),
        slack_bytes: 0,
        space_factor: 1.2,
        progress_interval: Duration::ZERO,
        free_space: |_| Ok(u64::MAX / 2),
    }
}

fn info() -> StarterInfo {
    StarterInfo {
        id: "test-stories".into(),
        title: "Test stories".into(),
        description: "for tests".into(),
        size_bytes: Count(1),
        license: "CC0-1.0".into(),
        attribution: "tests".into(),
        source_url: "http://localhost".into(),
        offline: false,
    }
}

fn remote(server: &Server, name: &str, body: &[u8], side: Side, prefix: Option<u64>) -> RemoteFile {
    RemoteFile {
        url: server.url(name),
        name: name.to_string(),
        lane: "stories".into(),
        side,
        size: body.len() as u64,
        sha256: sha256_hex(body),
        prefix_bytes: prefix,
    }
}

fn starter(files: Vec<RemoteFile>) -> Starter {
    Starter { info: info(), recipe: Recipe::Download(files) }
}

fn client() -> Client {
    http_client().unwrap()
}

fn shards_in(dir: &Path) -> Vec<Vec<u8>> {
    let mut names: Vec<String> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect(),
        Err(_) => return Vec::new(),
    };
    names.sort();
    names.iter().map(|n| std::fs::read(dir.join(n)).unwrap()).collect()
}

fn shard_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> =
        std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}

fn train_dir(dest: &Path) -> PathBuf {
    dest.join("train/stories")
}

fn val_dir(dest: &Path) -> PathBuf {
    dest.join("val/stories")
}

/// A train and a validation file on a fresh server.
struct Fixture {
    server: Server,
    train: Vec<u8>,
    val: Vec<u8>,
    dest: TempDir,
}

impl Fixture {
    async fn new() -> Fixture {
        let train = stories(1, 200_000, true);
        let val = stories(2, 40_000, false);
        let server = Server::start(vec![("train.txt", train.clone()), ("val.txt", val.clone())]).await;
        Fixture { server, train, val, dest: TempDir::new().unwrap() }
    }

    fn starter(&self, train_prefix: Option<u64>) -> Starter {
        starter(vec![
            remote(&self.server, "train.txt", &self.train, Side::Train, train_prefix),
            remote(&self.server, "val.txt", &self.val, Side::Val, None),
        ])
    }

    async fn install(&self, s: &Starter) -> Result<minagi_data::InstallReport, DataError> {
        install_starter_with(s, self.dest.path(), &client(), &test_opts(), CancellationToken::new(), |_| {}).await
    }

    fn assert_complete(&self, train_prefix: Option<usize>) {
        assert_eq!(shards_in(&train_dir(self.dest.path())), expected_shards(&self.train, SHARD, train_prefix));
        assert_eq!(shards_in(&val_dir(self.dest.path())), expected_shards(&self.val, SHARD, None));
    }
}

#[tokio::test]
async fn downloads_whole_files_into_story_aligned_shards() {
    let f = Fixture::new().await;
    let report = f.install(&f.starter(None)).await.unwrap();

    f.assert_complete(None);
    let train_shards = shards_in(&train_dir(f.dest.path()));
    assert!(train_shards.len() > 8, "{} shards", train_shards.len());
    for (i, shard) in train_shards.iter().enumerate() {
        assert!(shard.len() <= SHARD);
        assert!(std::str::from_utf8(shard).is_ok(), "shard {i} is valid UTF-8");
        if i + 1 < train_shards.len() {
            assert!(shard.ends_with(BOUNDARY), "shard {i} ends on a story boundary");
        }
    }
    assert_eq!(shard_names(&train_dir(f.dest.path()))[..2], ["part-0000.txt", "part-0001.txt"]);
    assert_eq!(train_shards.concat(), f.train, "nothing lost or duplicated");
    assert_eq!(shards_in(&val_dir(f.dest.path())).concat(), f.val);

    // The report, manifest and cleanup.
    assert_eq!((report.train_bytes, report.val_bytes), (f.train.len() as u64, f.val.len() as u64));
    assert!(!report.resumed && !report.already_installed);
    assert_eq!(report.downloaded_bytes, (f.train.len() + f.val.len()) as u64);
    assert_eq!(report.lanes.len(), 1);
    assert_eq!(report.lanes[0].train_files as usize, train_shards.len());
    let lane = &report.lane_infos()[0];
    assert_eq!((lane.name.as_str(), lane.color_slot), ("stories", 0));
    assert!(!f.dest.path().join(STATE_FILE).exists(), "the cursor is removed when the download is complete");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.dest.path().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["starterId"], "test-stories");
    assert_eq!(manifest["sources"][0]["integrity"], "sha256_verified");
    assert_eq!(manifest["sources"][0]["keptSha256"], sha256_hex(&f.train));
    assert_eq!(manifest["lanes"][0]["valBytes"], f.val.len());

    // What the server saw: a HEAD then a ranged GET per file, asking for identity encoding.
    let log = f.server.requests();
    assert_eq!(f.server.heads().len(), 2);
    let gets = f.server.cdn_gets();
    assert_eq!(gets.len(), 2);
    assert_eq!(gets[0].range, Some((0, f.train.len() as u64 - 1)));
    assert!(log.iter().all(|r| r.accept_encoding.as_deref() == Some("identity")), "{log:?}");
    assert_eq!(f.server.stale_token_requests(), 0);

    assert!(verify_installed(f.dest.path()).unwrap().is_ok());
}

#[tokio::test]
async fn story_boundaries_hold_for_many_shard_sizes_and_data_shapes() {
    for (shard, trailing) in [(4096usize, true), (7001, false), (64 * 1024, true)] {
        let body = stories(shard as u64, 300_000, trailing);
        let server = Server::start(vec![("a.txt", body.clone())]).await;
        let dest = TempDir::new().unwrap();
        let s = starter(vec![remote(&server, "a.txt", &body, Side::Train, None)]);
        let opts = DownloadOptions { shard_bytes: shard as u64, ..test_opts() };
        install_starter_with(&s, dest.path(), &client(), &opts, CancellationToken::new(), |_| {}).await.unwrap();
        let got = shards_in(&train_dir(dest.path()));
        assert_eq!(got, expected_shards(&body, shard, None), "shard {shard}");
        let markers_in_file = body.windows(BOUNDARY.len()).filter(|w| *w == BOUNDARY).count();
        let markers_in_shards: usize =
            got.iter().map(|g| g.windows(BOUNDARY.len()).filter(|w| *w == BOUNDARY).count()).sum();
        assert_eq!(markers_in_file, markers_in_shards, "no marker was cut in two");
        for (i, g) in got.iter().enumerate() {
            assert!(g.len() <= shard);
            assert!(std::str::from_utf8(g).is_ok());
            // Every shard begins at the start of a story: the previous one ended with the marker.
            if i > 0 {
                assert!(got[i - 1].ends_with(BOUNDARY));
            }
        }
        assert_eq!(got.concat(), body);
    }
}

#[tokio::test]
async fn a_prefix_keeps_whole_stories_only_and_records_its_own_hash() {
    let f = Fixture::new().await;
    let prefix = 100_003usize; // lands inside a story
    let s = f.starter(Some(prefix as u64));
    let report = f.install(&s).await.unwrap();

    f.assert_complete(Some(prefix));
    let kept: Vec<u8> = shards_in(&train_dir(f.dest.path())).concat();
    assert!(kept.len() <= prefix && kept.len() > prefix - 2000, "kept {} of {prefix}", kept.len());
    assert!(kept.ends_with(BOUNDARY), "the prefix ends on a story boundary");
    assert_eq!(&kept[..], &f.train[..kept.len()]);
    // Only the prefix was requested.
    let train_get = &f.server.cdn_gets()[0];
    assert_eq!(train_get.range, Some((0, prefix as u64 - 1)));
    assert_eq!(report.downloaded_bytes, (prefix + f.val.len()) as u64);

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.dest.path().join("manifest.json")).unwrap()).unwrap();
    let src = &manifest["sources"][0];
    assert_eq!(src["integrity"], "prefix_recorded");
    assert_eq!(src["keptSha256"], sha256_hex(&kept), "the prefix's own checksum is recorded");
    assert_ne!(src["keptSha256"], src["publishedSha256"], "it is a slice, so it differs from the file's");
    assert_eq!(src["prefixBytes"], prefix);
    assert_eq!(src["keptBytes"], kept.len());
    // The recorded hash lets a later check pass, and catches damage.
    assert!(verify_installed(f.dest.path()).unwrap().is_ok());
    let shard = train_dir(f.dest.path()).join("part-0001.txt");
    let mut bytes = std::fs::read(&shard).unwrap();
    bytes[10] ^= 0x01;
    std::fs::write(&shard, bytes).unwrap();
    let verdict = verify_installed(f.dest.path()).unwrap();
    assert!(!verdict.is_ok() && verdict.problems.iter().any(|p| p.contains("checksum")), "{verdict:?}");
}

#[tokio::test]
async fn cancel_then_resume_gives_byte_identical_output_without_redownloading() {
    // The reference: an uninterrupted install.
    let clean = Fixture::new().await;
    clean.install(&clean.starter(None)).await.unwrap();

    let f = Fixture::new().await;
    let s = f.starter(None);
    let token = CancellationToken::new();
    let t2 = token.clone();
    let threshold = 4.0 * SHARD as f64;
    let first = install_starter_with(&s, f.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
        if p.done >= threshold {
            t2.cancel();
        }
    })
    .await;
    assert!(matches!(first, Err(DataError::Cancelled)), "{first:?}");

    // The cursor survived and matches what is on disk.
    let state = DownloadState::load(f.dest.path()).expect("state file");
    let cursor = &state.files["train.txt"];
    assert!(cursor.shard_idx >= 3 && cursor.source_offset >= 3 * 8000 && !cursor.complete, "{cursor:?}");
    let on_disk = shards_in(&train_dir(f.dest.path()));
    assert_eq!(on_disk.len() as u32, cursor.shard_idx);
    assert_eq!(on_disk.iter().map(Vec::len).sum::<usize>() as u64, cursor.source_offset);
    assert!(!shard_names(&train_dir(f.dest.path())).iter().any(|n| n.ends_with(".part")));

    // Resume: the first ranged request starts at the cursor, never at 0.
    f.server.clear_log();
    let report = f.install(&s).await.unwrap();
    assert!(report.resumed);
    let gets = f.server.cdn_gets();
    assert_eq!(gets[0].range.unwrap().0, cursor.source_offset, "continues exactly where it stopped");
    assert!(gets.iter().all(|g| g.range.unwrap().0 > 0 || g.path.ends_with("val.txt")));
    assert!(report.downloaded_bytes < (f.train.len() + f.val.len()) as u64, "finished shards were not fetched again");

    for rel in ["train/stories", "val/stories"] {
        assert_eq!(shards_in(&f.dest.path().join(rel)), shards_in(&clean.dest.path().join(rel)), "{rel}");
    }
    f.assert_complete(None);
    assert!(verify_installed(f.dest.path()).unwrap().is_ok());
}

#[tokio::test]
async fn a_cancel_during_the_second_file_keeps_the_first_finished() {
    let f = Fixture::new().await;
    let s = f.starter(None);
    let token = CancellationToken::new();
    let t2 = token.clone();
    let train_len = f.train.len() as f64;
    let first = install_starter_with(&s, f.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
        if p.done > train_len + 20_000.0 {
            t2.cancel();
        }
    })
    .await;
    assert!(matches!(first, Err(DataError::Cancelled)));
    let state = DownloadState::load(f.dest.path()).unwrap();
    assert!(state.files["train.txt"].complete);
    assert!(!state.files["val.txt"].complete && state.files["val.txt"].shard_idx > 0);

    f.server.clear_log();
    f.install(&s).await.unwrap();
    let gets = f.server.cdn_gets();
    assert_eq!(gets.len(), 1, "only the unfinished file is fetched: {gets:?}");
    assert!(gets[0].path.ends_with("val.txt") && gets[0].range.unwrap().0 > 0);
    f.assert_complete(None);
}

#[tokio::test]
async fn a_connection_dropped_mid_stream_is_retried_from_the_cursor_with_a_fresh_url() {
    let f = Fixture::new().await;
    f.server.set_faults(|x| x.drop_after = [70_000, 30_000].into());
    let report = f.install(&f.starter(None)).await.unwrap();

    f.assert_complete(None);
    assert!(!report.resumed, "that word is for a restart of the whole install");
    let gets: Vec<_> = f.server.cdn_gets().into_iter().filter(|g| g.path.ends_with("train.txt")).collect();
    assert_eq!(gets.len(), 3, "two drops, then success: {gets:?}");
    let starts: Vec<u64> = gets.iter().map(|g| g.range.unwrap().0).collect();
    assert_eq!(starts[0], 0);
    assert!(starts[1] > 0 && starts[1] <= 70_000, "{starts:?}");
    assert!(starts[2] > starts[1], "the second drop made progress too: {starts:?}");
    assert_eq!(f.server.heads().len(), 4, "every attempt starts with a fresh HEAD (3 for the train file, 1 for val)");
    assert_eq!(f.server.stale_token_requests(), 0, "no signed URL was ever reused");
    let paths: std::collections::HashSet<_> = gets.iter().map(|g| g.path.clone()).collect();
    assert_eq!(paths.len(), 3, "three different signed URLs");
}

#[tokio::test]
async fn an_expired_signature_is_re_resolved_and_retried() {
    let f = Fixture::new().await;
    f.server.set_faults(|x| x.forbid_next_cdn = 2);
    f.install(&f.starter(None)).await.unwrap();
    f.assert_complete(None);
    let statuses: Vec<u16> = f.server.cdn_gets().iter().map(|g| g.status).collect();
    assert_eq!(&statuses[..3], [403, 403, 206], "{statuses:?}");
    assert!(f.server.heads().len() >= 4);
    assert_eq!(f.server.stale_token_requests(), 0, "no URL was used twice");
}

#[tokio::test]
async fn retries_give_up_after_max_attempts_without_progress() {
    let f = Fixture::new().await;
    // Every attempt dies before a single shard completes.
    f.server.set_faults(|x| x.drop_after = std::iter::repeat_n(1000, 20).collect());
    let err = f.install(&f.starter(None)).await.unwrap_err();
    assert!(matches!(err, DataError::Network(_)), "{err:?}");
    assert_eq!(f.server.cdn_gets().len(), 5, "max_attempts");
    assert!(shards_in(&train_dir(f.dest.path())).is_empty());
}

#[tokio::test]
async fn a_200_to_a_ranged_request_is_rejected() {
    let f = Fixture::new().await;
    f.server.set_faults(|x| x.ignore_range = true);
    let err = f.install(&f.starter(None)).await.unwrap_err();
    match &err {
        DataError::Network(m) => assert!(m.contains("ignored the byte range"), "{m}"),
        other => panic!("expected a network error, got {other:?}"),
    }
    assert_eq!(f.server.cdn_gets().len(), 1, "not retried: asking again would get the same answer");
    assert!(shards_in(&train_dir(f.dest.path())).is_empty(), "nothing was written");
    assert!(matches!(AppError::from(err), AppError::Network(_)));
}

#[tokio::test]
async fn a_checksum_mismatch_is_detected_reported_and_cleaned_up() {
    let f = Fixture::new().await;
    f.server.set_faults(|x| x.tamper = true);
    let err = f.install(&f.starter(None)).await.unwrap_err();
    match &err {
        DataError::Checksum { file, expected, actual } => {
            assert_eq!(file, "train.txt");
            assert_eq!(expected, &sha256_hex(&f.train));
            assert_ne!(actual, expected);
            assert_eq!(actual.len(), 64);
        }
        other => panic!("expected a checksum error, got {other:?}"),
    }
    assert!(err.to_string().contains("checksum mismatch"));
    assert!(shards_in(&train_dir(f.dest.path())).is_empty(), "corrupt shards are removed");
    let state = DownloadState::load(f.dest.path());
    assert!(state.is_none_or(|s| !s.files.contains_key("train.txt")), "the bad file starts over next time");
    assert!(matches!(AppError::from(err), AppError::Network(m) if m.contains("checksum")));

    // Once the server is healthy again a retry downloads cleanly.
    f.server.reset_faults();
    f.install(&f.starter(None)).await.unwrap();
    f.assert_complete(None);
}

#[tokio::test]
async fn a_changed_published_checksum_or_size_is_refused_before_any_download() {
    let f = Fixture::new().await;
    f.server.set_faults(|x| x.linked_etag = Some("0".repeat(64)));
    let err = f.install(&f.starter(None)).await.unwrap_err();
    assert!(matches!(&err, DataError::Invalid(m) if m.contains("different checksum")), "{err:?}");
    assert!(f.server.cdn_gets().is_empty(), "the file was never requested");

    f.server.reset_faults();
    f.server.set_faults(|x| x.linked_size = Some(f.train.len() as u64 + 1));
    let err = f.install(&f.starter(None)).await.unwrap_err();
    assert!(matches!(&err, DataError::Invalid(m) if m.contains("bytes on the server")), "{err:?}");
    assert!(f.server.cdn_gets().is_empty());
}

#[tokio::test]
async fn a_content_range_total_that_disagrees_is_rejected() {
    let f = Fixture::new().await;
    f.server.set_faults(|x| x.wrong_total = Some(f.train.len() as u64 + 5));
    let err = f.install(&f.starter(None)).await.unwrap_err();
    assert!(matches!(&err, DataError::Network(m) if m.contains("Content-Range")), "{err:?}");
    assert!(shards_in(&train_dir(f.dest.path())).is_empty());
}

#[tokio::test]
async fn a_missing_file_is_not_found_and_not_retried() {
    let f = Fixture::new().await;
    let mut files = vec![remote(&f.server, "train.txt", &f.train, Side::Train, None)];
    files[0].url = f.server.url("does-not-exist.txt");
    let err = f.install(&starter(files)).await.unwrap_err();
    assert!(matches!(err, DataError::NotFound(_)), "{err:?}");
    assert_eq!(f.server.heads().len(), 1);
}

#[tokio::test]
async fn data_without_story_separators_is_rejected_not_cut_blindly() {
    let body = vec![b'x'; 5 * SHARD];
    let server = Server::start(vec![("flat.txt", body.clone())]).await;
    let dest = TempDir::new().unwrap();
    let s = starter(vec![remote(&server, "flat.txt", &body, Side::Train, None)]);
    let err = install_starter_with(&s, dest.path(), &client(), &test_opts(), CancellationToken::new(), |_| {})
        .await
        .unwrap_err();
    assert!(matches!(&err, DataError::Invalid(m) if m.contains("endoftext")), "{err:?}");
}

#[tokio::test]
async fn invalid_utf8_is_rejected() {
    let mut body = stories(9, 40_000, true);
    body[5] = 0xFF;
    let server = Server::start(vec![("bad.txt", body.clone())]).await;
    let dest = TempDir::new().unwrap();
    let s = starter(vec![remote(&server, "bad.txt", &body, Side::Train, None)]);
    let err = install_starter_with(&s, dest.path(), &client(), &test_opts(), CancellationToken::new(), |_| {})
        .await
        .unwrap_err();
    assert!(matches!(&err, DataError::Invalid(m) if m.contains("UTF-8")), "{err:?}");
}

#[tokio::test]
async fn refuses_to_start_without_enough_free_space() {
    let f = Fixture::new().await;
    let total = (f.train.len() + f.val.len()) as u64;

    // 1.2 x the download plus the slack must fit.
    let slack = 1_000_000u64;
    let need = (total as f64 * 1.2).ceil() as u64 + slack;
    let tight = DownloadOptions { slack_bytes: slack, free_space: |_| Ok(1_000_100), ..test_opts() };
    let err =
        install_starter_with(&f.starter(None), f.dest.path(), &client(), &tight, CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
    match &err {
        DataError::DiskFull { need_bytes, free_bytes } => assert_eq!((*need_bytes, *free_bytes), (need, 1_000_100)),
        other => panic!("expected DiskFull, got {other:?}"),
    }
    assert!(f.server.requests().is_empty(), "nothing was requested");
    assert!(shards_in(&train_dir(f.dest.path())).is_empty());
    assert!(
        matches!(AppError::from(err), AppError::DiskFull { need_bytes, free_bytes } if need_bytes == Count(need) && free_bytes == Count(1_000_100))
    );

    // Just enough space passes.
    let enough = DownloadOptions { slack_bytes: slack, free_space: |_| Ok(1_000_000 + 300_000), ..test_opts() };
    install_starter_with(&f.starter(None), f.dest.path(), &client(), &enough, CancellationToken::new(), |_| {})
        .await
        .unwrap();
    f.assert_complete(None);
}

#[tokio::test]
async fn the_default_slack_is_one_gigabyte_and_the_real_free_space_check_works() {
    let f = Fixture::new().await;
    let defaults = DownloadOptions::default();
    assert_eq!((defaults.slack_bytes, defaults.space_factor, defaults.shard_bytes), (1 << 30, 1.2, 8 << 20));
    assert_eq!((defaults.max_attempts, defaults.progress_interval), (5, Duration::from_millis(100)));
    // 900 MB free is not enough for a tiny download plus 1 GiB of slack.
    let low = DownloadOptions {
        free_space: |_| Ok(900_000_000),
        shard_bytes: SHARD as u64,
        backoff_base: Duration::from_millis(1),
        ..defaults
    };
    let err = install_starter_with(&f.starter(None), f.dest.path(), &client(), &low, CancellationToken::new(), |_| {})
        .await
        .unwrap_err();
    assert!(matches!(err, DataError::DiskFull { .. }));
    // The platform call reports something sane for a real folder.
    assert!(minagi_data::download::disk_free_bytes(f.dest.path()).unwrap() > 0);
}

#[tokio::test]
async fn the_free_space_check_counts_only_what_is_left_on_a_resume() {
    static FREE: AtomicU64 = AtomicU64::new(0);
    let f = Fixture::new().await;
    let s = f.starter(None);
    let token = CancellationToken::new();
    let t2 = token.clone();
    let first = install_starter_with(&s, f.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
        if p.done > 150_000.0 {
            t2.cancel();
        }
    })
    .await;
    assert!(matches!(first, Err(DataError::Cancelled)));
    let done = DownloadState::load(f.dest.path()).unwrap().files["train.txt"].source_offset;
    let remaining = (f.train.len() + f.val.len()) as u64 - done;
    // Enough for the remainder (plus a little), though not for the whole download from scratch.
    FREE.store((remaining as f64 * 1.2) as u64 + 2_000, Ordering::Relaxed);
    assert!(FREE.load(Ordering::Relaxed) < ((f.train.len() + f.val.len()) as f64 * 1.2) as u64);
    let opts = DownloadOptions { free_space: |_| Ok(FREE.load(Ordering::Relaxed)), ..test_opts() };
    install_starter_with(&s, f.dest.path(), &client(), &opts, CancellationToken::new(), |_| {}).await.unwrap();
    f.assert_complete(None);
}

#[tokio::test]
async fn a_changed_cdn_etag_between_resumes_is_detected_and_the_download_restarts_clean() {
    let f = Fixture::new().await;
    let s = f.starter(None);
    let token = CancellationToken::new();
    let t2 = token.clone();
    let first = install_starter_with(&s, f.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
        if p.done > 60_000.0 {
            t2.cancel();
        }
    })
    .await;
    assert!(matches!(first, Err(DataError::Cancelled)));
    assert!(DownloadState::load(f.dest.path()).unwrap().files["train.txt"].etag.is_some());

    f.server.set_cdn_etag("train.txt", "a-different-file-now");
    let err = f.install(&s).await.unwrap_err();
    assert!(matches!(&err, DataError::Invalid(m) if m.contains("changed on the server")), "{err:?}");
    assert!(shards_in(&train_dir(f.dest.path())).is_empty(), "the stale shards were discarded");

    // The next run starts over from byte 0 and now agrees with the server.
    f.server.clear_log();
    f.install(&s).await.unwrap();
    assert_eq!(f.server.cdn_gets()[0].range.unwrap().0, 0);
    f.assert_complete(None);
}

#[tokio::test]
async fn a_hard_kill_between_the_shard_rename_and_the_cursor_write_is_harmless() {
    let f = Fixture::new().await;
    let s = f.starter(None);
    let token = CancellationToken::new();
    let t2 = token.clone();
    let _ = install_starter_with(&s, f.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
        if p.done > 5.0 * SHARD as f64 {
            t2.cancel();
        }
    })
    .await;
    // Rewind the cursor by one shard (the shard file stays on disk, as if the kill came just after the rename) and
    // leave a half-written next shard behind.
    let mut state = DownloadState::load(f.dest.path()).unwrap();
    let cur = state.files.get_mut("train.txt").unwrap();
    let last_len =
        std::fs::metadata(train_dir(f.dest.path()).join(format!("part-{:04}.txt", cur.shard_idx - 1))).unwrap().len();
    cur.shard_idx -= 1;
    cur.source_offset -= last_len;
    cur.kept_bytes -= last_len;
    let stray = train_dir(f.dest.path()).join(format!("part-{:04}.txt.part", cur.shard_idx + 1));
    state.save(f.dest.path()).unwrap();
    std::fs::write(stray, b"half a shard").unwrap();

    f.install(&s).await.unwrap();
    f.assert_complete(None);
    assert!(verify_installed(f.dest.path()).unwrap().is_ok());
}

#[tokio::test]
async fn a_cursor_that_does_not_match_the_files_on_disk_restarts_that_file() {
    let f = Fixture::new().await;
    let s = f.starter(None);
    let token = CancellationToken::new();
    let t2 = token.clone();
    let _ = install_starter_with(&s, f.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
        if p.done > 4.0 * SHARD as f64 {
            t2.cancel();
        }
    })
    .await;
    let mut state = DownloadState::load(f.dest.path()).unwrap();
    state.files.get_mut("train.txt").unwrap().kept_bytes += 7; // lies about what is on disk
    state.save(f.dest.path()).unwrap();

    f.server.clear_log();
    f.install(&s).await.unwrap();
    assert_eq!(f.server.cdn_gets()[0].range.unwrap().0, 0, "nothing could be trusted, so it started over");
    f.assert_complete(None);

    // A missing shard is caught the same way.
    let g = Fixture::new().await;
    let token = CancellationToken::new();
    let t2 = token.clone();
    let _ =
        install_starter_with(&g.starter(None), g.dest.path(), &client(), &test_opts(), token, move |p: JobProgress| {
            if p.done > 4.0 * SHARD as f64 {
                t2.cancel();
            }
        })
        .await;
    std::fs::remove_file(train_dir(g.dest.path()).join("part-0001.txt")).unwrap();
    g.install(&g.starter(None)).await.unwrap();
    g.assert_complete(None);
}

#[tokio::test]
async fn installing_again_into_a_finished_folder_does_nothing() {
    let f = Fixture::new().await;
    let s = f.starter(None);
    f.install(&s).await.unwrap();
    f.server.clear_log();
    let again = f.install(&s).await.unwrap();
    assert!(again.already_installed);
    assert_eq!(again.downloaded_bytes, 0);
    assert_eq!((again.train_bytes, again.val_bytes), (f.train.len() as u64, f.val.len() as u64));
    assert!(f.server.requests().is_empty());

    // Damage is noticed: with a shard missing it downloads again.
    std::fs::remove_file(train_dir(f.dest.path()).join("part-0002.txt")).unwrap();
    let repaired = f.install(&s).await.unwrap();
    assert!(!repaired.already_installed);
    f.assert_complete(None);
}

#[tokio::test]
async fn verify_installed_reports_missing_and_changed_shards() {
    let f = Fixture::new().await;
    f.install(&f.starter(None)).await.unwrap();
    let ok = verify_installed(f.dest.path()).unwrap();
    assert!(ok.is_ok() && ok.files_checked == 2 && ok.bytes_checked == (f.train.len() + f.val.len()) as u64, "{ok:?}");

    let shard = val_dir(f.dest.path()).join("part-0000.txt");
    std::fs::remove_file(&shard).unwrap();
    assert!(verify_installed(f.dest.path()).unwrap().problems.iter().any(|p| p.contains("missing")));
}

#[tokio::test]
async fn progress_reports_bytes_rate_and_eta_and_is_throttled() {
    let f = Fixture::new().await;
    let s = f.starter(None);
    let total = (f.train.len() + f.val.len()) as f64;

    // Unthrottled: every chunk reports.
    let seen = Arc::new(Mutex::new(Vec::<JobProgress>::new()));
    let sink = seen.clone();
    install_starter_with(&s, f.dest.path(), &client(), &test_opts(), CancellationToken::new(), move |p| {
        sink.lock().unwrap().push(p)
    })
    .await
    .unwrap();
    {
        let seen = seen.lock().unwrap();
        assert!(seen.len() > 10);
        assert!(seen.iter().all(|p| p.unit == "bytes" && p.total == Some(total)));
        assert!(seen.windows(2).all(|w| w[0].done <= w[1].done), "progress never goes backwards");
        assert_eq!(seen.last().unwrap().done, total);
        assert!(seen.iter().any(|p| p.message.starts_with("Downloading train.txt")));
        assert!(seen.iter().all(|p| p.eta_seconds.is_none_or(|e| e >= 0.0) && p.bytes_per_sec.is_none_or(|r| r > 0.0)));
    }

    // Throttled to 10 Hz.
    let g = Fixture::new().await;
    let seen = Arc::new(Mutex::new(Vec::<JobProgress>::new()));
    let sink = seen.clone();
    let opts = DownloadOptions { progress_interval: Duration::from_millis(100), ..test_opts() };
    let started = std::time::Instant::now();
    install_starter_with(&g.starter(None), g.dest.path(), &client(), &opts, CancellationToken::new(), move |p| {
        sink.lock().unwrap().push(p)
    })
    .await
    .unwrap();
    let elapsed = started.elapsed().as_millis() as usize;
    let seen = seen.lock().unwrap();
    // Forced events (start of a file, retry notices, "Finished") bypass the throttle; the stream itself does not.
    assert!(seen.len() <= elapsed / 100 + 8, "{} callbacks in {elapsed} ms", seen.len());
    assert_eq!(seen.last().unwrap().done, total);
}

#[tokio::test]
async fn offline_starters_install_without_the_network() {
    // The client is never used: point it nowhere.
    let client = client();

    let dir = TempDir::new().unwrap();
    let sampler = minagi_data::find_starter("sampler").unwrap();
    let mut seen = 0;
    let report = install_starter(&sampler, dir.path(), &client, CancellationToken::new(), |p| {
        seen += 1;
        assert_eq!(p.unit, "problems");
    })
    .await
    .unwrap();
    assert!(seen > 0);
    let names: Vec<&str> = report.lanes.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["arithmetic", "stories"]);
    assert_eq!(report.downloaded_bytes, 0);
    assert!(
        dir.path().join("train/stories/part-0000.txt").is_file()
            && dir.path().join("val/stories/part-0000.txt").is_file()
    );
    assert!(
        dir.path().join("train/arithmetic/part-0000.txt").is_file()
            && dir.path().join("val/arithmetic/part-0000.txt").is_file()
    );
    assert!(
        std::fs::read_to_string(dir.path().join("train/stories/part-0000.txt")).unwrap().contains("Once upon a time")
    );
    assert!(verify_installed(dir.path()).unwrap().is_ok());
    let infos = report.lane_infos();
    assert_eq!(
        (infos[0].name.as_str(), infos[0].color_slot, infos[1].name.as_str(), infos[1].color_slot),
        ("arithmetic", 1, "stories", 0)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["license"], "CC0-1.0");
    // A second call is a no-op.
    let again = install_starter(&sampler, dir.path(), &client, CancellationToken::new(), |_| {}).await.unwrap();
    assert!(again.already_installed);

    // A small arithmetic starter.
    let mut arithmetic = minagi_data::find_starter("arithmetic").unwrap();
    arithmetic.recipe = Recipe::Arithmetic(minagi_types::ArithmeticParams {
        train_problems: 3000,
        val_problems: 300,
        seed: 3,
        notes_frac: 0.3,
    });
    let dir = TempDir::new().unwrap();
    let report = install_starter(&arithmetic, dir.path(), &client, CancellationToken::new(), |_| {}).await.unwrap();
    assert_eq!(report.lanes.len(), 1);
    let first_line = std::fs::read_to_string(dir.path().join("val/arithmetic/part-0000.txt")).unwrap();
    assert!(first_line.lines().next().unwrap().contains(" = "));
    assert!(verify_installed(dir.path()).unwrap().is_ok());
}

#[tokio::test]
async fn cancelling_a_generated_starter_stops_it() {
    let mut arithmetic = minagi_data::find_starter("arithmetic").unwrap();
    arithmetic.recipe = Recipe::Arithmetic(minagi_types::ArithmeticParams {
        train_problems: 50_000_000,
        val_problems: 1000,
        seed: 1,
        notes_frac: 0.3,
    });
    let dir = TempDir::new().unwrap();
    let token = CancellationToken::new();
    let t2 = token.clone();
    let result = install_starter(&arithmetic, dir.path(), &client(), token, move |_| t2.cancel()).await;
    assert!(matches!(result, Err(DataError::Cancelled)), "{result:?}");
    assert!(!dir.path().join("manifest.json").exists());
}
