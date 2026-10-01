//! Runs: creation, lifecycle state, progress, and crash recovery.

use minagi_types::{Chars, Count, Goal, ModelConfig, Preset, RunState, RunSummary, Step, TrainConfig, UnixMs, Verdict};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::db::{Store, StoreError, StoreResult};

pub const CONFIG_SCHEMA_VERSION: i64 = 1;

/// Everything needed to record a new run.
#[derive(Debug, Clone)]
pub struct NewRun {
    pub name: String,
    pub preset: Preset,
    pub dataset_id: Option<i64>,
    pub dataset_snapshot: serde_json::Value,
    pub model: ModelConfig,
    pub train: TrainConfig,
    pub goal: Option<Goal>,
    pub engine_version: String,
    pub app_version: String,
    pub origin: RunOrigin,
    pub parent_run_id: Option<i64>,
    pub backend: Option<String>,
    /// Size of one pass over the training text, for the "how far through the text" display.
    pub chars_total: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOrigin {
    Trained,
    Imported,
    Forked,
}

impl RunOrigin {
    fn as_str(&self) -> &'static str {
        match self {
            RunOrigin::Trained => "trained",
            RunOrigin::Imported => "imported",
            RunOrigin::Forked => "forked",
        }
    }
}

/// Coalesced progress written about once a second while a run is active.
#[derive(Debug, Clone, Default)]
pub struct RunProgress {
    pub run_id: i64,
    pub step: u64,
    pub chars_read: u64,
    pub chars_total: Option<u64>,
    pub active_ms: u64,
    pub last_train_nats: Option<f64>,
    pub last_heldout_nats: Option<f64>,
    pub n_experts: Option<u32>,
    pub stage: Option<String>,
    pub verdict: Option<Verdict>,
}

fn to_json<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
}

fn enum_str<T: Serialize>(v: &T) -> Option<String> {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(str::to_string))
}

fn parse_enum<T: DeserializeOwned>(s: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).ok()
}

pub fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "run".into() } else { out.chars().take(40).collect() }
}

const RUN_COLUMNS: &str = "r.id, r.uid, r.name, r.notes, r.status, r.preset, r.dataset_id, d.name, r.step, r.chars_read, \
    r.chars_total, r.active_ms, r.best_heldout_nats, r.last_heldout_nats, r.last_train_nats, r.n_experts, r.verdict, \
    r.stage, r.backend, r.goal_json, r.error, r.created_at, r.started_at, r.ended_at";

fn run_from_row(r: &Row<'_>) -> rusqlite::Result<RunSummary> {
    let status: String = r.get(4)?;
    let preset: String = r.get(5)?;
    let verdict: Option<String> = r.get(16)?;
    let goal: Option<String> = r.get(19)?;
    Ok(RunSummary {
        id: r.get(0)?,
        uid: r.get(1)?,
        name: r.get(2)?,
        notes: r.get(3)?,
        status: RunState::parse(&status).unwrap_or(RunState::Failed),
        preset: Preset::parse(&preset).unwrap_or(Preset::Custom),
        dataset_id: r.get(6)?,
        dataset_name: r.get(7)?,
        step: Step(r.get::<_, i64>(8)? as u64),
        chars_read: Chars(r.get::<_, i64>(9)? as u64),
        chars_total: r.get::<_, Option<i64>>(10)?.map(|v| Chars(v as u64)),
        active_ms: Count(r.get::<_, i64>(11)? as u64),
        best_heldout_nats: r.get(12)?,
        last_heldout_nats: r.get(13)?,
        last_train_nats: r.get(14)?,
        n_experts: r.get::<_, Option<i64>>(15)?.map(|v| v as u32),
        verdict: verdict.and_then(|s| parse_enum(&s)),
        stage: r.get(17)?,
        backend: r.get(18)?,
        goal: goal.and_then(|s| serde_json::from_str(&s).ok()),
        error: r.get(20)?,
        created_at: UnixMs(r.get::<_, i64>(21)? as u64),
        started_at: r.get::<_, Option<i64>>(22)?.map(|v| UnixMs(v as u64)),
        ended_at: r.get::<_, Option<i64>>(23)?.map(|v| UnixMs(v as u64)),
    })
}

fn get_run_conn(conn: &Connection, id: i64) -> StoreResult<RunSummary> {
    let sql = format!("SELECT {RUN_COLUMNS} FROM runs r LEFT JOIN datasets d ON d.id = r.dataset_id WHERE r.id = ?1");
    conn.query_row(&sql, [id], run_from_row).optional()?.ok_or_else(|| StoreError::NotFound(format!("run {id}")))
}

impl Store {
    /// Record a new run (and its deduplicated configuration) and return it.
    pub fn create_run(&self, new: NewRun) -> StoreResult<RunSummary> {
        self.write(move |conn| {
            let now = UnixMs::now().0 as i64;
            let config_json = to_json(&serde_json::json!({ "model": new.model, "train": new.train }));
            let hash = format!("{:x}", Sha256::digest(config_json.as_bytes()));
            let tx = conn.transaction()?;
            tx.execute(
                "INSERT OR IGNORE INTO run_configs (preset, config_json, config_hash, schema_version, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![new.preset.as_str(), config_json, hash, CONFIG_SCHEMA_VERSION, now],
            )?;
            let config_id: i64 =
                tx.query_row("SELECT id FROM run_configs WHERE config_hash = ?1", [&hash], |r| r.get(0))?;
            let uid = uuid::Uuid::now_v7().to_string();
            // dir_rel is unique; use the uid prefix so concurrent or repeated names never collide.
            let dir_rel = format!("runs/{}-{}", slugify(&new.name), &uid[uid.len() - 8..]);
            tx.execute(
                "INSERT INTO runs (uid, name, origin, dataset_id, dataset_snapshot_json, parent_run_id, preset, config_id, \
                 dir_rel, status, backend, engine_version, app_version, goal_json, chars_total, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'created', ?10, ?11, ?12, ?13, ?15, ?14, ?14)",
                params![
                    uid,
                    new.name,
                    new.origin.as_str(),
                    new.dataset_id,
                    to_json(&new.dataset_snapshot),
                    new.parent_run_id,
                    new.preset.as_str(),
                    config_id,
                    dir_rel,
                    new.backend,
                    new.engine_version,
                    new.app_version,
                    new.goal.as_ref().map(to_json),
                    now,
                    new.chars_total.map(|c| c as i64),
                ],
            )?;
            let id = tx.last_insert_rowid();
            tx.commit()?;
            get_run_conn(conn, id)
        })
    }

    pub fn get_run(&self, id: i64) -> StoreResult<RunSummary> {
        self.read(|c| get_run_conn(c, id))
    }

    pub fn list_runs(&self) -> StoreResult<Vec<RunSummary>> {
        self.read(|c| {
            let sql = format!(
                "SELECT {RUN_COLUMNS} FROM runs r LEFT JOIN datasets d ON d.id = r.dataset_id ORDER BY r.created_at DESC, r.id DESC"
            );
            let mut stmt = c.prepare(&sql)?;
            let rows = stmt.query_map([], run_from_row)?.collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// The configuration a run was created with. Fields added to the config since the run was created are taken from
    /// the run's preset, so old runs keep loading.
    pub fn run_config(&self, run_id: i64) -> StoreResult<(ModelConfig, TrainConfig)> {
        self.read(|c| {
            let (json, preset): (String, String) = c
                .query_row(
                    "SELECT rc.config_json, r.preset FROM runs r JOIN run_configs rc ON rc.id = r.config_id WHERE r.id = ?1",
                    [run_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?
                .ok_or_else(|| StoreError::NotFound(format!("run {run_id}")))?;
            let preset = Preset::parse(&preset).unwrap_or(Preset::Small);
            let stored: serde_json::Value = serde_json::from_str(&json).map_err(|e| StoreError::Other(e.to_string()))?;
            let bad = |e: serde_json::Error| StoreError::Other(format!("stored configuration is invalid: {e}"));
            let model = preset.load_model(&stored["model"]).map_err(bad)?;
            let train = preset.load_train(&stored["train"]).map_err(bad)?;
            Ok((model, train))
        })
    }

    /// Directory of the run, relative to the app data directory.
    pub fn run_dir_rel(&self, run_id: i64) -> StoreResult<String> {
        self.read(|c| {
            c.query_row("SELECT dir_rel FROM runs WHERE id = ?1", [run_id], |r| r.get(0))
                .optional()?
                .ok_or_else(|| StoreError::NotFound(format!("run {run_id}")))
        })
    }

    /// Change a run's lifecycle state. Sets started/ended timestamps as appropriate.
    pub fn set_run_state(&self, id: i64, state: RunState, error: Option<String>) -> StoreResult<RunSummary> {
        self.write(move |conn| {
            let now = UnixMs::now().0 as i64;
            conn.execute(
                "UPDATE runs SET status = ?2, error = COALESCE(?3, error), updated_at = ?4, \
                 started_at = CASE WHEN ?2 IN ('running') AND started_at IS NULL THEN ?4 ELSE started_at END, \
                 ended_at = CASE WHEN ?2 IN ('completed','stopped','failed','interrupted') THEN ?4 \
                                 WHEN ?2 IN ('running','paused','preparing') THEN NULL ELSE ended_at END \
                 WHERE id = ?1",
                params![id, state.as_str(), error, now],
            )?;
            get_run_conn(conn, id)
        })
    }

    /// Queue a coalesced progress update (batched, never blocks the caller).
    pub fn update_run_progress(&self, p: RunProgress) {
        self.write_batched(move |conn| {
            let now = UnixMs::now().0 as i64;
            conn.execute(
                "UPDATE runs SET step = ?2, chars_read = ?3, chars_total = COALESCE(?4, chars_total), active_ms = ?5, \
                 last_train_nats = COALESCE(?6, last_train_nats), n_experts = COALESCE(?7, n_experts), \
                 stage = COALESCE(?8, stage), verdict = COALESCE(?9, verdict), \
                 last_heldout_nats = COALESCE(?10, last_heldout_nats), \
                 best_heldout_nats = CASE WHEN ?10 IS NOT NULL AND (best_heldout_nats IS NULL OR ?10 < best_heldout_nats) \
                                     THEN ?10 ELSE best_heldout_nats END, \
                 updated_at = ?11 WHERE id = ?1",
                params![
                    p.run_id,
                    p.step as i64,
                    p.chars_read as i64,
                    p.chars_total.map(|v| v as i64),
                    p.active_ms as i64,
                    p.last_train_nats.filter(|v| v.is_finite()),
                    p.n_experts,
                    p.stage,
                    p.verdict.as_ref().and_then(enum_str),
                    p.last_heldout_nats.filter(|v| v.is_finite()),
                    now,
                ],
            )?;
            Ok(())
        });
    }

    pub fn rename_run(&self, id: i64, name: String, notes: String) -> StoreResult<RunSummary> {
        self.write(move |conn| {
            let n = conn.execute(
                "UPDATE runs SET name = ?2, notes = ?3, updated_at = ?4 WHERE id = ?1",
                params![id, name, notes, UnixMs::now().0 as i64],
            )?;
            if n == 0 {
                return Err(StoreError::NotFound(format!("run {id}")));
            }
            get_run_conn(conn, id)
        })
    }

    /// Delete a run and its telemetry.
    pub fn delete_run(&self, id: i64) -> StoreResult<()> {
        self.write(move |conn| {
            // `metrics` is clustered by (run_id, key_id, step), so deleting one key at a time is a contiguous
            // range delete that stays fast and never holds the writer for long.
            let keys: Vec<i64> = {
                let mut stmt = conn.prepare("SELECT DISTINCT key_id FROM metrics WHERE run_id = ?1")?;
                stmt.query_map([id], |r| r.get(0))?.collect::<Result<_, _>>()?
            };
            for key in keys {
                conn.execute("DELETE FROM metrics WHERE run_id = ?1 AND key_id = ?2", params![id, key])?;
            }
            let n = conn.execute("DELETE FROM runs WHERE id = ?1", [id])?;
            if n == 0 {
                return Err(StoreError::NotFound(format!("run {id}")));
            }
            Ok(())
        })
    }

    /// At startup: any run still marked active was cut off by a close or crash. Returns the affected run ids.
    pub fn mark_interrupted_on_startup(&self) -> StoreResult<Vec<i64>> {
        self.write(|conn| {
            let ids: Vec<i64> = {
                let mut stmt =
                    conn.prepare("SELECT id FROM runs WHERE status IN ('preparing','running','paused','stopping')")?;
                stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?
            };
            let now = UnixMs::now().0 as i64;
            for id in &ids {
                conn.execute(
                    "UPDATE runs SET status = 'interrupted', ended_at = ?2, updated_at = ?2 WHERE id = ?1",
                    params![id, now],
                )?;
            }
            Ok(ids)
        })
    }
}

#[cfg(test)]
pub(crate) fn test_run(store: &Store) -> RunSummary {
    store
        .create_run(NewRun {
            name: "My first run".into(),
            preset: Preset::Tiny,
            dataset_id: None,
            dataset_snapshot: serde_json::json!({}),
            model: Preset::Tiny.model(),
            train: Preset::Tiny.train(),
            goal: Some(Goal::Minutes { value: 30.0 }),
            engine_version: "test".into(),
            app_version: "0.0.0".into(),
            origin: RunOrigin::Trained,
            parent_run_id: None,
            backend: Some("cpu".into()),
            chars_total: None,
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        (dir, store)
    }

    #[test]
    fn create_and_read_back() {
        let (_d, store) = open();
        let run = test_run(&store);
        assert_eq!(run.status, RunState::Created);
        assert_eq!(run.preset, Preset::Tiny);
        assert!(matches!(run.goal, Some(Goal::Minutes { .. })));
        let (m, t) = store.run_config(run.id).unwrap();
        assert_eq!(m, Preset::Tiny.model());
        assert_eq!(t, Preset::Tiny.train());
        assert!(store.run_dir_rel(run.id).unwrap().starts_with("runs/my-first-run-"));
        assert_eq!(store.list_runs().unwrap().len(), 1);
    }

    #[test]
    fn identical_configs_share_one_row_and_names_never_collide() {
        let (_d, store) = open();
        let a = test_run(&store);
        let b = test_run(&store);
        assert_ne!(a.id, b.id);
        assert_ne!(store.run_dir_rel(a.id).unwrap(), store.run_dir_rel(b.id).unwrap());
        let n: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM run_configs", [], |r| r.get(0))?)).unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn state_transitions_set_timestamps() {
        let (_d, store) = open();
        let run = test_run(&store);
        let r = store.set_run_state(run.id, RunState::Running, None).unwrap();
        assert!(r.started_at.is_some() && r.ended_at.is_none());
        let r = store.set_run_state(run.id, RunState::Completed, None).unwrap();
        assert!(r.ended_at.is_some());
    }

    #[test]
    fn active_runs_become_interrupted_after_a_crash() {
        let (_d, store) = open();
        let running = test_run(&store);
        let done = test_run(&store);
        store.set_run_state(running.id, RunState::Running, None).unwrap();
        store.set_run_state(done.id, RunState::Completed, None).unwrap();
        let ids = store.mark_interrupted_on_startup().unwrap();
        assert_eq!(ids, vec![running.id]);
        assert_eq!(store.get_run(running.id).unwrap().status, RunState::Interrupted);
        assert_eq!(store.get_run(done.id).unwrap().status, RunState::Completed);
    }

    #[test]
    fn progress_updates_track_best_heldout() {
        let (_d, store) = open();
        let run = test_run(&store);
        for (step, held) in [(10u64, 3.0), (20, 2.0), (30, 2.5)] {
            store.update_run_progress(RunProgress {
                run_id: run.id,
                step,
                chars_read: step * 512,
                last_heldout_nats: Some(held),
                ..Default::default()
            });
        }
        store.flush();
        let r = store.get_run(run.id).unwrap();
        assert_eq!(r.step.0, 30);
        assert_eq!(r.best_heldout_nats, Some(2.0));
        assert_eq!(r.last_heldout_nats, Some(2.5));
    }

    #[test]
    fn delete_removes_everything() {
        let (_d, store) = open();
        let run = test_run(&store);
        store.delete_run(run.id).unwrap();
        assert!(matches!(store.get_run(run.id), Err(StoreError::NotFound(_))));
        assert!(store.delete_run(run.id).is_err());
    }

    #[test]
    fn slugs_are_filesystem_safe() {
        assert_eq!(slugify("My first run!"), "my-first-run");
        assert_eq!(slugify("  ../etc/passwd "), "etc-passwd");
        assert_eq!(slugify("日本語"), "run");
    }
}
