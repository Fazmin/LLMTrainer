//! Dataset-manager types: lanes, scan results, starters, background-job progress and the split/lane settings.
//!
//! A *dataset* is a directory shaped `<root>/train/<lane>/**` and `<root>/val/<domain>/**`. A *lane* is one kind of
//! text (stories, code, chat, ...) and is what the UI draws as a coloured chip. These types are produced by the
//! `minagi-data` crate and consumed by the store, the Tauri commands and the UI.

use serde::{Deserialize, Serialize};
use specta::Type;

use crate::units::{Count, UnixMs};

/// One lane of a dataset, as shown in the lane table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct LaneInfo {
    /// Folder-safe name (`[a-z0-9_-]`), also the directory name under `train/` and `val/`.
    pub name: String,
    /// Human-friendly name (the original folder name).
    pub display_name: String,
    /// Fixed colour slot. 0..=7 are the eight lane colours, 8 and above fold into the shared "Other" colour.
    pub color_slot: u32,
    /// Whether the lane is read during training.
    pub enabled: bool,
    pub n_files: Count,
    pub train_bytes: Count,
    pub val_files: Count,
    pub val_bytes: Count,
    /// A prompt that suits this lane, offered when sampling from the model.
    pub sample_prompt: Option<String>,
}

/// One file that was not used, with the reason in plain words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SkippedExample {
    pub path: String,
    pub reason: String,
}

/// What a folder scan left out, counted by reason. `examples` holds at most 50 entries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SkippedSummary {
    /// Images, archives, executables, or files whose content is not text.
    pub binary: u32,
    pub empty: u32,
    /// UTF-16 or UTF-32 text (recognised by its byte-order mark); the engine reads UTF-8 only.
    pub utf16: u32,
    pub too_large: u32,
    /// Files or folders that could not be opened (permissions, vanished mid-scan).
    pub unreadable: u32,
    pub examples: Vec<SkippedExample>,
}

impl SkippedSummary {
    /// Total number of skipped files, over all reasons.
    pub fn total(&self) -> u32 {
        self.binary + self.empty + self.utf16 + self.too_large + self.unreadable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DatasetKind {
    /// Downloaded or generated from the built-in starter list.
    Starter,
    /// Built from the user's own folders.
    Linked,
    /// Generated locally (synthetic text).
    Generated,
    /// Shipped inside the app.
    Bundled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DatasetStatus {
    /// Created but not yet scanned or prepared.
    Draft,
    Scanning,
    /// Downloading, generating or linking files.
    Preparing,
    Ready,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct DatasetSummary {
    pub id: i64,
    pub name: String,
    pub kind: DatasetKind,
    /// Set when this dataset was installed from the starter list.
    pub starter_id: Option<String>,
    pub status: DatasetStatus,
    pub train_bytes: Count,
    pub val_bytes: Count,
    pub lanes: u32,
    pub license: Option<String>,
    pub attribution: Option<String>,
    pub source_url: Option<String>,
    pub error: Option<String>,
    pub created_at: UnixMs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct DatasetDetail {
    pub summary: DatasetSummary,
    pub lanes: Vec<LaneInfo>,
    pub skipped: SkippedSummary,
    pub warnings: Vec<String>,
}

/// A starter dataset the app can install.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct StarterInfo {
    pub id: String,
    pub title: String,
    pub description: String,
    /// Approximate size on disk once installed.
    pub size_bytes: Count,
    pub license: String,
    pub attribution: String,
    pub source_url: String,
    /// True when no network is needed (generated locally or bundled).
    pub offline: bool,
}

/// Progress of a background job (download, generate, link files).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct JobProgress {
    pub done: f64,
    pub total: Option<f64>,
    /// What `done` and `total` count: `"bytes"`, `"files"` or `"problems"`.
    pub unit: String,
    /// One plain-English line: "Downloading TinyStories-train.txt".
    pub message: String,
    pub bytes_per_sec: Option<f64>,
    pub eta_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    /// Hold a small share of the files (or the end of a big file) out for validation.
    Auto,
    /// Use the `val/` folder the dataset already has, exactly as it is.
    Folder,
}

/// How a dataset's text is divided into training and validation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SplitConfig {
    pub mode: SplitMode,
    /// Share of every lane held out for validation, in percent (2.0 means 2 %). Clamped to 0.5..=10.
    pub pct: f64,
    /// Seed for the file shuffle; the same inputs and seed always give the same split.
    pub seed: u32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self { mode: SplitMode::Auto, pct: 2.0, seed: 42 }
    }
}

/// How the folders a user drops are mapped to lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LaneMode {
    /// Pick the best of the four rules (mini-AGI layout, one lane per subfolder, one lane, one lane per root).
    #[default]
    Auto,
    /// Everything becomes a single lane.
    OneLane,
    /// Each top-level subfolder becomes a lane, whatever the share of loose files.
    PerFolder,
}

/// Settings for the synthetic arithmetic lane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct ArithmeticParams {
    pub train_problems: u32,
    pub val_problems: u32,
    pub seed: u32,
    /// Share of problems that carry a `<think> ... </think>` scratchpad, 0.0..=1.0.
    pub notes_frac: f64,
}

impl Default for ArithmeticParams {
    fn default() -> Self {
        Self { train_problems: 500_000, val_problems: 10_000, seed: 1, notes_frac: 0.3 }
    }
}

/// A random passage from a lane, for the "peek inside" panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct TextPreview {
    /// Path of the file relative to the lane.
    pub file: String,
    /// Byte offset of the passage inside the file.
    pub offset: Count,
    pub text: String,
}

/// Quick answer about something the user is dragging over the window, before any real scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct PathProbe {
    pub path: String,
    pub is_dir: bool,
    /// Files seen during the (time-boxed) walk; a lower bound when `exact` is false.
    pub approx_files: Count,
    /// False when the walk stopped at its time budget.
    pub exact: bool,
    /// True when the folder has a `train/` subfolder, i.e. it is already in the shape the engine reads.
    pub looks_like_mini_agi_layout: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtos_use_camel_case_and_plain_numbers() {
        let lane = LaneInfo {
            name: "stories".into(),
            display_name: "Stories".into(),
            color_slot: 0,
            enabled: true,
            n_files: Count(3),
            train_bytes: Count(1000),
            val_files: Count(1),
            val_bytes: Count(20),
            sample_prompt: None,
        };
        let json = serde_json::to_value(&lane).unwrap();
        assert_eq!(json["displayName"], "Stories");
        assert_eq!(json["colorSlot"], 0);
        assert_eq!(json["trainBytes"], 1000);
        assert_eq!(json["samplePrompt"], serde_json::Value::Null);
        let back: LaneInfo = serde_json::from_value(json).unwrap();
        assert_eq!(back, lane);
    }

    #[test]
    fn enums_are_snake_case() {
        assert_eq!(serde_json::to_string(&DatasetStatus::Preparing).unwrap(), "\"preparing\"");
        assert_eq!(serde_json::to_string(&LaneMode::PerFolder).unwrap(), "\"per_folder\"");
        assert_eq!(serde_json::to_string(&DatasetKind::Bundled).unwrap(), "\"bundled\"");
    }

    #[test]
    fn defaults_are_sensible() {
        let split = SplitConfig::default();
        assert_eq!((split.mode, split.pct), (SplitMode::Auto, 2.0));
        assert!((0.0..=1.0).contains(&ArithmeticParams::default().notes_frac));
    }

    #[test]
    fn skipped_total_adds_all_reasons() {
        let s = SkippedSummary { binary: 1, empty: 2, utf16: 3, too_large: 4, unreadable: 5, examples: vec![] };
        assert_eq!(s.total(), 15);
    }
}
