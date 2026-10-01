//! Dataset manager for LLM Trainer: prepares the directory the training engine reads.
//!
//! The engine reads text from `<root>/train/<lane>/**` and `<root>/val/<domain>/**`, where a *lane* is a top-level
//! folder holding one kind of text. This crate builds that directory from the user's own folders and from starter
//! datasets. It is plain filesystem and HTTP logic: no database, no Tauri, everything unit-tested.
//!
//! # From a dropped folder
//!
//! ```no_run
//! use std::path::PathBuf;
//! use minagi_data::{CancelFlag, PrevIndex, ScanOptions, detect_lanes, materialize, plan_split, scan_paths};
//! use minagi_types::{LaneMode, SplitConfig};
//!
//! # fn main() -> minagi_data::DataResult<()> {
//! let cancel = CancelFlag::new();
//! // 1. Find the text (skips binaries, hidden folders, UTF-16, ...).
//! let scan = scan_paths(&[PathBuf::from("/home/me/notes")], &ScanOptions::default(), &PrevIndex::new(), &cancel, &mut |_| {});
//! // 2. Decide the lanes, 3. decide what is held out for validation (deterministic, seeded).
//! let layout = detect_lanes(&scan, LaneMode::Auto);
//! let plan = plan_split(&layout, &SplitConfig::default())?;
//! // 4. Build train/ and val/ with reflinks or hard links where possible (never symlinks).
//! let report = materialize(&plan, std::path::Path::new("/data/dataset-1"), &cancel, &mut |_| {})?;
//! println!("extra disk space needed: {} bytes", report.bytes_copied);
//! # Ok(()) }
//! ```
//!
//! # From a starter
//!
//! [`starters()`] lists what is on offer; [`install_starter`] downloads (resumably) or generates one. See
//! [`download`] for how the downloader survives being killed at any moment.
//!
//! # Modules
//!
//! | module            | what it does                                                                      |
//! |-------------------|-----------------------------------------------------------------------------------|
//! | [`scan`]          | walk folders, sniff text, cache verdicts, count what was skipped and why           |
//! | [`lanes`]         | group files into lanes (four rules), slugify names, assign colour slots            |
//! | [`split`]         | seeded train/validation split: whole files, or a tail cut from big files           |
//! | [`materialize`]   | build the dataset directory atomically with reflink, hard link or copy             |
//! | [`synth`]         | locally generated text (the arithmetic corpus)                                     |
//! | [`starters`]      | the starter list and [`install_starter`]                                           |
//! | [`download`]      | the resumable, checksum-verified, story-aligned downloader                         |
//! | [`prompts`]       | suggested sampling prompts per lane                                                |
//! | [`preview`]       | a random passage from a lane                                                       |
//! | [`probe`]         | instant drag-and-drop feedback                                                     |
//! | [`error`]         | [`DataError`] and its mapping to `minagi_types::AppError`                          |

pub mod download;
pub mod error;
pub mod lanes;
pub mod materialize;
pub mod preview;
pub mod probe;
pub mod prompts;
pub mod scan;
pub mod split;
pub mod starters;
pub mod synth;
mod util;

pub use download::{DownloadOptions, RemoteFile, Side, http_client};
pub use error::{DataError, DataResult};
pub use lanes::*;
pub use materialize::*;
pub use preview::{PreviewFile, preview_files, preview_layout_lane, preview_text};
pub use probe::{probe_paths, probe_paths_within};
pub use prompts::{default_prompt, fill_sample_prompts, suggest_prompts};
pub use scan::*;
pub use split::*;
pub use starters::*;
pub use util::CancelFlag;
