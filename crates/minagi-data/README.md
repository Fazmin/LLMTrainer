# minagi-data

The dataset manager of LLM Trainer. It prepares the directory the training engine reads,

```
<root>/train/<lane>/**     one folder per kind of text (a "lane")
<root>/val/<domain>/**     held-out text, one folder per domain
<root>/manifest.json       what is in it, how it was split, where it came from
```

from the user's own folders and from starter datasets. No database, no Tauri: filesystem and HTTP logic only, with
`minagi-types` for the DTOs that cross into the UI (`dataset.rs` there).

## Public API

| area | function | notes |
|------|----------|-------|
| scan | `scan_paths(paths, &ScanOptions, &PrevIndex, &CancelFlag, &mut dyn FnMut(ScanProgress)) -> ScanResult` | folders and loose files; never follows symlinks; skips hidden and build folders and binary extensions; sniffs 4 KiB; cached by (path, size, mtime) |
| scan | `sniff(&[u8]) -> Sniff` | pure: `Text`, `Empty`, `Binary`, `Utf16` |
| lanes | `detect_lanes(&ScanResult, LaneMode) -> LaneLayout` | four rules; `slugify`, `assign_color_slots` |
| split | `plan_split(&LaneLayout, &SplitConfig) -> DataResult<SplitPlan>` | seeded, deterministic; whole files or a tail cut; folder mode uses the dataset's own `val/` |
| build | `materialize(&SplitPlan, dest, &CancelFlag, &mut dyn FnMut(JobProgress)) -> DataResult<MaterializeReport>` | reflink, then hard link, then copy; never symlinks; atomic via `dest.tmp-<n>`; in-place datasets are not touched |
| synth | `synth::arithmetic::generate(&ArithmeticParams, dest, &CancelFlag, progress) -> DataResult<ArithmeticReport>` | port of mini-AGI's `corpora/arithmetic.py`, with no train/val leakage |
| starters | `starters() -> Vec<StarterInfo>`, `find_starter(id) -> Option<Starter>` | `tinystories-quick`, `tinystories-full`, `arithmetic`, `sampler` |
| starters | `install_starter(&Starter, dest, &Client, CancellationToken, progress) -> DataResult<InstallReport>` (async) | resumable downloads; `install_starter_with` takes `DownloadOptions`; `verify_installed(dest)` re-checks the shards |
| starters | `http_client() -> DataResult<Client>` | rustls, redirects off (the downloader resolves them itself) |
| prompts | `suggest_prompts(root) -> Vec<(lane, prompt)>` | fixed prompts for stories/arithmetic/code/chat, a held-out passage otherwise |
| preview | `preview_text(root, lane, n_chars, seed) -> DataResult<TextPreview>` | random, UTF-8 safe passage; `preview_layout_lane` for unprepared lanes |
| probe | `probe_paths(paths) -> Vec<PathProbe>` | at most 300 ms in total |
| errors | `DataError` and `impl From<DataError> for AppError` | io, network, disk full, cancelled, invalid, not found |

All long jobs report `JobProgress` at most ten times a second and stop promptly on cancellation.

## Lanes

1. A single folder with `train/` is a mini-AGI layout: lanes are the subfolders of `train/`, validation domains the
   subfolders of `val/`, loose files in `train/` form a lane named after the folder. Such a dataset is used in place,
   read-only, when the split mode is `folder`.
2. A single folder with subfolders holding text, loose files at most 20 % of the text bytes: one lane per top-level
   subfolder, loose files in `misc`.
3. Any other single folder: one lane named after it.
4. Several dropped paths: one lane per folder, loose files together in `my-files`.

Names are slugified (`[a-z0-9_-]`) and de-duplicated (`-2`, `-3`). Colour slots: stories 0, arithmetic 1, wikipedia 2,
code 3, chat 4, reasoning 5, chess 6, self-knowledge 7; every other lane takes the lowest slot nobody claimed, in sorted
name order. Slots from 8 up fold into "Other".

## Split

Whole-file hold-out for lanes with at least five files (about `pct` % of the bytes, at most 20 % of the files, at least
one file, ordered by `blake3(seed ‖ path)`); otherwise the last `clamp(pct % of the lane, 50 kB, 5 MB)` bytes of the
largest file become `<stem>.tail.txt` in `val/` and the rest `<stem>.head.txt` in `train/`, cut at a newline or UTF-8
boundary. A tail never exceeds a fifth of its file; a lane too small for that gets no validation text and a warning.

## Starters and the downloader

`tinystories-*` come from a pinned Hugging Face revision. Each attempt starts with a `HEAD` (no redirect following) to
read `x-linked-size` / `x-linked-etag` and the fresh signed CDN URL, then a ranged `GET` that must answer `206` with an
exact `Content-Range`. The stream is cut into shards of at most 8 MiB at the last `\n<|endoftext|>\n` of each window,
each written as `.part` and renamed, with the resume cursor saved in `<dest>/.download-state.json` after every shard.
Whole files are verified against the published SHA-256; the 64 MiB prefix has its own SHA-256 recorded in
`manifest.json`. Free space of 1.2 x the download plus 1 GiB is required before starting.

## Layout produced by a starter

```
<dest>/train/stories/part-0000.txt ...
<dest>/val/stories/part-0000.txt ...
<dest>/manifest.json
```

Tests never touch the real network: the downloader suite runs against a loopback server that imitates the Hugging Face
`resolve` redirect, its one-use signed URLs and the failures listed above (`cargo test -p minagi-data`).
