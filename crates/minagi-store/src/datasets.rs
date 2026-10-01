//! Datasets, their lanes, and background jobs.

use minagi_types::{
    Count, DatasetDetail, DatasetKind, DatasetStatus, DatasetSummary, JobInfo, JobState, LaneInfo, SkippedSummary,
    SplitConfig, SplitMode, UnixMs,
};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use crate::db::{Store, StoreError, StoreResult};

#[derive(Debug, Clone)]
pub struct NewDataset {
    pub name: String,
    pub kind: DatasetKind,
    pub starter_id: Option<String>,
    pub split: SplitConfig,
    pub license: Option<String>,
    pub attribution: Option<String>,
    pub source_url: Option<String>,
}

/// Where a dataset's files live: a folder inside the app data directory, or a user folder used in place.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetRoot {
    pub root_rel: Option<String>,
    pub root_abs: Option<String>,
}

/// Result of preparing a dataset: everything the lane table and the summary show.
#[derive(Debug, Clone)]
pub struct DatasetFinal {
    pub root: DatasetRoot,
    pub train_bytes: u64,
    pub val_bytes: u64,
    pub lanes: Vec<LaneInfo>,
    pub skipped: SkippedSummary,
    pub warnings: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct Extras {
    skipped: SkippedSummary,
    warnings: Vec<String>,
}

fn enum_str<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
}

fn parse_enum<T: serde::de::DeserializeOwned>(s: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).ok()
}

const SUMMARY_COLUMNS: &str = "d.id, d.name, d.kind, d.status, d.train_bytes, d.val_bytes, \
    (SELECT count(*) FROM lanes l WHERE l.dataset_id = d.id), d.license, d.attribution, d.source_url, d.error, d.created_at, d.starter_id";

fn summary_from_row(r: &Row<'_>) -> rusqlite::Result<DatasetSummary> {
    let kind: String = r.get(2)?;
    let status: String = r.get(3)?;
    Ok(DatasetSummary {
        id: r.get(0)?,
        name: r.get(1)?,
        kind: parse_enum(&kind).unwrap_or(DatasetKind::Linked),
        status: parse_enum(&status).unwrap_or(DatasetStatus::Error),
        train_bytes: Count(r.get::<_, i64>(4)? as u64),
        val_bytes: Count(r.get::<_, i64>(5)? as u64),
        lanes: r.get::<_, i64>(6)? as u32,
        license: r.get(7)?,
        attribution: r.get(8)?,
        source_url: r.get(9)?,
        error: r.get(10)?,
        created_at: UnixMs(r.get::<_, i64>(11)? as u64),
        starter_id: r.get(12)?,
    })
}

fn get_summary(conn: &Connection, id: i64) -> StoreResult<DatasetSummary> {
    conn.query_row(&format!("SELECT {SUMMARY_COLUMNS} FROM datasets d WHERE d.id = ?1"), [id], summary_from_row)
        .optional()?
        .ok_or_else(|| StoreError::NotFound(format!("dataset {id}")))
}

fn lanes_of(conn: &Connection, id: i64) -> StoreResult<Vec<LaneInfo>> {
    let mut stmt = conn.prepare(
        "SELECT name, display_name, color_slot, enabled, n_files, train_bytes, val_files, val_bytes, sample_prompt \
         FROM lanes WHERE dataset_id = ?1 ORDER BY color_slot, name",
    )?;
    let rows = stmt
        .query_map([id], |r| {
            Ok(LaneInfo {
                name: r.get(0)?,
                display_name: r.get(1)?,
                color_slot: r.get::<_, i64>(2)? as u32,
                enabled: r.get::<_, i64>(3)? != 0,
                n_files: Count(r.get::<_, i64>(4)? as u64),
                train_bytes: Count(r.get::<_, i64>(5)? as u64),
                val_files: Count(r.get::<_, i64>(6)? as u64),
                val_bytes: Count(r.get::<_, i64>(7)? as u64),
                sample_prompt: r.get(8)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

impl Store {
    pub fn create_dataset(&self, new: NewDataset) -> StoreResult<DatasetSummary> {
        self.write(move |conn| {
            let now = UnixMs::now().0 as i64;
            conn.execute(
                "INSERT INTO datasets (name, kind, starter_id, status, split_mode, split_pct, split_seed, license, attribution, \
                 source_url, created_at, updated_at) VALUES (?1, ?2, ?3, 'draft', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)",
                params![
                    new.name,
                    enum_str(&new.kind),
                    new.starter_id,
                    if new.split.mode == SplitMode::Folder { "folder" } else { "auto" },
                    new.split.pct,
                    new.split.seed as i64,
                    new.license,
                    new.attribution,
                    new.source_url,
                    now
                ],
            )?;
            get_summary(conn, conn.last_insert_rowid())
        })
    }

    pub fn list_datasets(&self) -> StoreResult<Vec<DatasetSummary>> {
        self.read(|c| {
            let mut stmt =
                c.prepare(&format!("SELECT {SUMMARY_COLUMNS} FROM datasets d ORDER BY d.created_at DESC, d.id DESC"))?;
            let rows = stmt.query_map([], summary_from_row)?.collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    pub fn get_dataset(&self, id: i64) -> StoreResult<DatasetDetail> {
        self.read(|c| {
            let summary = get_summary(c, id)?;
            let json: String = c.query_row("SELECT skipped_json FROM datasets WHERE id = ?1", [id], |r| r.get(0))?;
            let extras: Extras = serde_json::from_str(&json).unwrap_or_default();
            Ok(DatasetDetail { summary, lanes: lanes_of(c, id)?, skipped: extras.skipped, warnings: extras.warnings })
        })
    }

    pub fn set_dataset_status(
        &self,
        id: i64,
        status: DatasetStatus,
        error: Option<String>,
    ) -> StoreResult<DatasetSummary> {
        self.write(move |conn| {
            conn.execute(
                "UPDATE datasets SET status = ?2, error = ?3, updated_at = ?4 WHERE id = ?1",
                params![id, enum_str(&status), error, UnixMs::now().0 as i64],
            )?;
            get_summary(conn, id)
        })
    }

    pub fn dataset_root(&self, id: i64) -> StoreResult<DatasetRoot> {
        self.read(|c| {
            c.query_row("SELECT root_rel, root_abs FROM datasets WHERE id = ?1", [id], |r| {
                Ok(DatasetRoot { root_rel: r.get(0)?, root_abs: r.get(1)? })
            })
            .optional()?
            .ok_or_else(|| StoreError::NotFound(format!("dataset {id}")))
        })
    }

    pub fn dataset_split(&self, id: i64) -> StoreResult<SplitConfig> {
        self.read(|c| {
            c.query_row("SELECT split_mode, split_pct, split_seed FROM datasets WHERE id = ?1", [id], |r| {
                let mode: String = r.get(0)?;
                Ok(SplitConfig {
                    mode: if mode == "folder" { SplitMode::Folder } else { SplitMode::Auto },
                    pct: r.get(1)?,
                    seed: r.get::<_, i64>(2)? as u32,
                })
            })
            .optional()?
            .ok_or_else(|| StoreError::NotFound(format!("dataset {id}")))
        })
    }

    pub fn set_dataset_split(&self, id: i64, split: SplitConfig) -> StoreResult<()> {
        self.write(move |conn| {
            let n = conn.execute(
                "UPDATE datasets SET split_mode = ?2, split_pct = ?3, split_seed = ?4, updated_at = ?5 WHERE id = ?1",
                params![
                    id,
                    if split.mode == SplitMode::Folder { "folder" } else { "auto" },
                    split.pct,
                    split.seed as i64,
                    UnixMs::now().0 as i64
                ],
            )?;
            if n == 0 { Err(StoreError::NotFound(format!("dataset {id}"))) } else { Ok(()) }
        })
    }

    /// Record the outcome of preparing a dataset: its location, sizes, lanes and what was skipped. Marks it ready.
    pub fn finish_dataset(&self, id: i64, fin: DatasetFinal) -> StoreResult<DatasetDetail> {
        self.write(move |conn| {
            let tx = conn.transaction()?;
            let extras = serde_json::to_string(&Extras { skipped: fin.skipped.clone(), warnings: fin.warnings.clone() })
                .map_err(|e| StoreError::Other(e.to_string()))?;
            tx.execute(
                "UPDATE datasets SET status = 'ready', error = NULL, root_rel = ?2, root_abs = ?3, train_bytes = ?4, val_bytes = ?5, \
                 skipped_json = ?6, scanned_at = ?7, updated_at = ?7 WHERE id = ?1",
                params![
                    id,
                    fin.root.root_rel,
                    fin.root.root_abs,
                    fin.train_bytes as i64,
                    fin.val_bytes as i64,
                    extras,
                    UnixMs::now().0 as i64
                ],
            )?;
            // Keep the user's choices (enabled, custom prompt) for lanes that are still there.
            let previous = lanes_of(&tx, id)?;
            tx.execute("DELETE FROM lanes WHERE dataset_id = ?1", [id])?;
            for lane in &fin.lanes {
                let before = previous.iter().find(|p| p.name == lane.name);
                tx.execute(
                    "INSERT INTO lanes (dataset_id, name, display_name, color_slot, enabled, n_files, train_bytes, val_files, val_bytes, sample_prompt) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        id,
                        lane.name,
                        lane.display_name,
                        lane.color_slot as i64,
                        before.map_or(lane.enabled, |b| b.enabled) as i64,
                        lane.n_files.0 as i64,
                        lane.train_bytes.0 as i64,
                        lane.val_files.0 as i64,
                        lane.val_bytes.0 as i64,
                        before.and_then(|b| b.sample_prompt.clone()).or_else(|| lane.sample_prompt.clone()),
                    ],
                )?;
            }
            tx.commit()?;
            let summary = get_summary(conn, id)?;
            Ok(DatasetDetail { summary, lanes: lanes_of(conn, id)?, skipped: fin.skipped, warnings: fin.warnings })
        })
    }

    /// Change how a lane is used: switch it off, rename it, or set the prompt used when sampling.
    pub fn update_lane(
        &self,
        dataset_id: i64,
        lane: String,
        enabled: Option<bool>,
        display_name: Option<String>,
        sample_prompt: Option<String>,
    ) -> StoreResult<DatasetDetail> {
        self.write(move |conn| {
            let n = conn.execute(
                "UPDATE lanes SET enabled = COALESCE(?3, enabled), display_name = COALESCE(?4, display_name), \
                 sample_prompt = COALESCE(?5, sample_prompt) WHERE dataset_id = ?1 AND name = ?2",
                params![dataset_id, lane, enabled.map(|e| e as i64), display_name, sample_prompt],
            )?;
            if n == 0 {
                return Err(StoreError::NotFound(format!("lane {lane}")));
            }
            let summary = get_summary(conn, dataset_id)?;
            let json: String =
                conn.query_row("SELECT skipped_json FROM datasets WHERE id = ?1", [dataset_id], |r| r.get(0))?;
            let extras: Extras = serde_json::from_str(&json).unwrap_or_default();
            Ok(DatasetDetail {
                summary,
                lanes: lanes_of(conn, dataset_id)?,
                skipped: extras.skipped,
                warnings: extras.warnings,
            })
        })
    }

    /// Remember the folders a dataset was built from, so it can be rescanned or re-split later.
    pub fn set_dataset_sources(&self, id: i64, paths: Vec<String>) -> StoreResult<()> {
        self.write(move |conn| {
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM dataset_sources WHERE dataset_id = ?1", [id])?;
            for path in &paths {
                let kind = if std::path::Path::new(path).is_dir() { "folder" } else { "file" };
                tx.execute(
                    "INSERT OR IGNORE INTO dataset_sources (dataset_id, path, kind) VALUES (?1, ?2, ?3)",
                    params![id, path, kind],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    pub fn dataset_sources(&self, id: i64) -> StoreResult<Vec<String>> {
        self.read(|c| {
            let mut stmt = c.prepare("SELECT path FROM dataset_sources WHERE dataset_id = ?1 ORDER BY id")?;
            let rows = stmt.query_map([id], |r| r.get::<_, String>(0))?.collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    pub fn delete_dataset(&self, id: i64) -> StoreResult<()> {
        self.write(move |conn| {
            let n = conn.execute("DELETE FROM datasets WHERE id = ?1", [id])?;
            if n == 0 { Err(StoreError::NotFound(format!("dataset {id}"))) } else { Ok(()) }
        })
    }

    // ───────── jobs ─────────

    pub fn create_job(&self, kind: &str, subject: Option<String>) -> StoreResult<String> {
        let kind = kind.to_string();
        self.write(move |conn| {
            let id = uuid::Uuid::now_v7().to_string();
            let now = UnixMs::now().0 as i64;
            conn.execute(
                "INSERT INTO jobs (id, kind, state, subject, created_at, updated_at) VALUES (?1, ?2, 'queued', ?3, ?4, ?4)",
                params![id, kind, subject, now],
            )?;
            Ok(id)
        })
    }

    pub fn update_job(
        &self,
        id: &str,
        state: JobState,
        progress: Option<f64>,
        message: Option<String>,
        error: Option<String>,
    ) {
        let id = id.to_string();
        self.write_batched(move |conn| {
            let now = UnixMs::now().0 as i64;
            conn.execute(
                "UPDATE jobs SET state = ?2, progress = COALESCE(?3, progress), message = COALESCE(?4, message), error = ?5, \
                 updated_at = ?6, finished_at = CASE WHEN ?2 IN ('done','failed','cancelled') THEN ?6 ELSE finished_at END WHERE id = ?1",
                params![id, state.as_str(), progress, message, error, now],
            )?;
            Ok(())
        });
    }

    pub fn list_jobs(&self) -> StoreResult<Vec<JobInfo>> {
        self.read(|c| {
            let mut stmt = c.prepare(
                "SELECT id, kind, state, subject, progress, message, error, created_at, updated_at FROM jobs \
                 ORDER BY created_at DESC, id DESC LIMIT 100",
            )?;
            let rows = stmt
                .query_map([], |r| {
                    let state: String = r.get(2)?;
                    Ok(JobInfo {
                        id: r.get(0)?,
                        kind: r.get(1)?,
                        state: JobState::parse(&state).unwrap_or(JobState::Failed),
                        subject: r.get(3)?,
                        progress: r.get(4)?,
                        message: r.get(5)?,
                        error: r.get(6)?,
                        created_at: UnixMs(r.get::<_, i64>(7)? as u64),
                        updated_at: UnixMs(r.get::<_, i64>(8)? as u64),
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// A job still marked running or queued when the app starts was cut off: mark it failed so nothing looks stuck.
    pub fn fail_interrupted_jobs(&self) -> StoreResult<usize> {
        self.write(|conn| {
            let now = UnixMs::now().0 as i64;
            Ok(conn.execute(
                "UPDATE jobs SET state = 'failed', error = 'The app closed before this finished.', updated_at = ?1, finished_at = ?1 \
                 WHERE state IN ('queued','running')",
                [now],
            )?)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        (dir, store)
    }

    fn new_ds(store: &Store, name: &str) -> DatasetSummary {
        store
            .create_dataset(NewDataset {
                name: name.into(),
                kind: DatasetKind::Linked,
                starter_id: None,
                split: SplitConfig::default(),
                license: None,
                attribution: None,
                source_url: None,
            })
            .unwrap()
    }

    fn lane(name: &str, slot: u32, bytes: u64) -> LaneInfo {
        LaneInfo {
            name: name.into(),
            display_name: name.to_uppercase(),
            color_slot: slot,
            enabled: true,
            n_files: Count(3),
            train_bytes: Count(bytes),
            val_files: Count(1),
            val_bytes: Count(bytes / 50),
            sample_prompt: None,
        }
    }

    fn fin(lanes: Vec<LaneInfo>) -> DatasetFinal {
        DatasetFinal {
            root: DatasetRoot { root_rel: Some("datasets/1-x".into()), root_abs: None },
            train_bytes: lanes.iter().map(|l| l.train_bytes.0).sum(),
            val_bytes: lanes.iter().map(|l| l.val_bytes.0).sum(),
            lanes,
            skipped: SkippedSummary { binary: 2, ..Default::default() },
            warnings: vec!["A file is over 4 GB.".into()],
        }
    }

    #[test]
    fn datasets_are_created_as_drafts_and_finished_with_lanes() {
        let (_d, store) = open();
        let ds = new_ds(&store, "My notes");
        assert_eq!(ds.status, DatasetStatus::Draft);
        assert_eq!(ds.lanes, 0);

        let detail = store.finish_dataset(ds.id, fin(vec![lane("stories", 0, 1000), lane("code", 3, 5000)])).unwrap();
        assert_eq!(detail.summary.status, DatasetStatus::Ready);
        assert_eq!(detail.summary.train_bytes.0, 6000);
        assert_eq!(detail.summary.lanes, 2);
        assert_eq!(detail.skipped.binary, 2);
        assert_eq!(detail.warnings, vec!["A file is over 4 GB.".to_string()]);
        assert_eq!(detail.lanes[0].name, "stories");
        assert_eq!(store.dataset_root(ds.id).unwrap().root_rel.as_deref(), Some("datasets/1-x"));
        // The detail read back matches what finish returned.
        assert_eq!(store.get_dataset(ds.id).unwrap(), detail);
    }

    #[test]
    fn rescanning_keeps_the_users_lane_choices() {
        let (_d, store) = open();
        let ds = new_ds(&store, "x");
        store.finish_dataset(ds.id, fin(vec![lane("stories", 0, 1000), lane("code", 3, 5000)])).unwrap();
        store.update_lane(ds.id, "code".into(), Some(false), None, Some("def f():".into())).unwrap();
        // A rescan reports fresh sizes and the default (enabled) state.
        let again = store.finish_dataset(ds.id, fin(vec![lane("stories", 0, 2000), lane("code", 3, 6000)])).unwrap();
        let code = again.lanes.iter().find(|l| l.name == "code").unwrap();
        assert!(!code.enabled, "a lane the user switched off stays off");
        assert_eq!(code.sample_prompt.as_deref(), Some("def f():"));
        assert_eq!(code.train_bytes.0, 6000);
    }

    #[test]
    fn split_settings_round_trip_and_lane_updates_validate() {
        let (_d, store) = open();
        let ds = new_ds(&store, "x");
        assert_eq!(store.dataset_split(ds.id).unwrap(), SplitConfig::default());
        let split = SplitConfig { mode: SplitMode::Folder, pct: 5.0, seed: 7 };
        store.set_dataset_split(ds.id, split).unwrap();
        assert_eq!(store.dataset_split(ds.id).unwrap(), split);
        assert!(matches!(
            store.update_lane(ds.id, "nope".into(), Some(true), None, None),
            Err(StoreError::NotFound(_))
        ));
    }

    #[test]
    fn sources_are_remembered_in_order_and_replaced() {
        let (_d, store) = open();
        let ds = new_ds(&store, "x");
        store.set_dataset_sources(ds.id, vec!["/a/notes".into(), "/b/code".into()]).unwrap();
        assert_eq!(store.dataset_sources(ds.id).unwrap(), vec!["/a/notes".to_string(), "/b/code".to_string()]);
        store.set_dataset_sources(ds.id, vec!["/c".into()]).unwrap();
        assert_eq!(store.dataset_sources(ds.id).unwrap(), vec!["/c".to_string()]);
    }

    #[test]
    fn status_errors_and_deletion() {
        let (_d, store) = open();
        let a = new_ds(&store, "a");
        let b = new_ds(&store, "b");
        let failed = store.set_dataset_status(a.id, DatasetStatus::Error, Some("No readable text.".into())).unwrap();
        assert_eq!(failed.error.as_deref(), Some("No readable text."));
        assert_eq!(store.list_datasets().unwrap().len(), 2);
        store.delete_dataset(a.id).unwrap();
        let left = store.list_datasets().unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id, b.id);
        assert!(store.delete_dataset(a.id).is_err());
    }

    #[test]
    fn jobs_track_progress_and_interrupted_ones_fail() {
        let (_d, store) = open();
        let id = store.create_job("download", Some("dataset:1".into())).unwrap();
        store.update_job(&id, JobState::Running, Some(0.25), Some("Downloading".into()), None);
        store.flush();
        let j = &store.list_jobs().unwrap()[0];
        assert_eq!((j.state, j.progress), (JobState::Running, Some(0.25)));
        assert_eq!(j.message.as_deref(), Some("Downloading"));

        assert_eq!(store.fail_interrupted_jobs().unwrap(), 1);
        let j = &store.list_jobs().unwrap()[0];
        assert_eq!(j.state, JobState::Failed);
        assert!(j.error.as_deref().unwrap().contains("closed"));

        store.update_job(&id, JobState::Done, Some(1.0), None, None);
        store.flush();
        assert!(store.list_jobs().unwrap()[0].state.is_finished());
    }

    #[test]
    fn a_run_keeps_its_history_when_its_dataset_is_deleted() {
        let (_d, store) = open();
        let ds = new_ds(&store, "gone soon");
        let run = store
            .create_run(crate::NewRun {
                name: "r".into(),
                preset: minagi_types::Preset::Tiny,
                dataset_id: Some(ds.id),
                dataset_snapshot: serde_json::json!({ "name": "gone soon" }),
                model: minagi_types::Preset::Tiny.model(),
                train: minagi_types::Preset::Tiny.train(),
                goal: None,
                engine_version: "t".into(),
                app_version: "0".into(),
                origin: crate::RunOrigin::Trained,
                parent_run_id: None,
                backend: None,
                chars_total: None,
            })
            .unwrap();
        assert_eq!(run.dataset_name.as_deref(), Some("gone soon"));
        store.delete_dataset(ds.id).unwrap();
        let after = store.get_run(run.id).unwrap();
        assert_eq!(after.dataset_id, None);
        assert_eq!(after.dataset_name, None);
    }
}
