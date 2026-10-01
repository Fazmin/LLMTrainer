//! Expert-pool snapshots and checkpoint listing.

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use minagi_types::{Chars, CheckpointInfo, CheckpointKind, Count, PoolSnapshot, Step, UnixMs};
use rusqlite::{OptionalExtension, params};

use crate::db::{Store, StoreError, StoreResult};

impl Store {
    /// Persist a pool snapshot (taken at evaluation rounds and on birth/prune events).
    pub fn record_pool_snapshot(&self, run_id: i64, snap: PoolSnapshot) {
        self.write_batched(move |conn| {
            let json = |v: &dyn erased::Json| v.to_json();
            conn.execute(
                "INSERT OR REPLACE INTO pool_snapshots (run_id, step, chars, n_experts, resident_json, usage_b64, brakes_json, halting_hist_json) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    run_id,
                    snap.step.0 as i64,
                    snap.chars.0 as i64,
                    snap.n_experts,
                    json(&snap.resident),
                    B64.encode(&snap.usage),
                    json(&snap.brakes),
                    json(&snap.halting_hist),
                ],
            )?;
            Ok(())
        });
    }

    /// The snapshot at `step`, or the nearest one at or before it (the latest when `step` is `None`).
    pub fn get_pool_snapshot(&self, run_id: i64, step: Option<u64>) -> StoreResult<Option<PoolSnapshot>> {
        self.read(|c| {
            let row = c
                .query_row(
                    "SELECT step, chars, n_experts, resident_json, usage_b64, brakes_json, halting_hist_json FROM pool_snapshots \
                     WHERE run_id = ?1 AND step <= ?2 ORDER BY step DESC LIMIT 1",
                    params![run_id, step.map(|s| s as i64).unwrap_or(i64::MAX)],
                    |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, String>(4)?,
                            r.get::<_, String>(5)?,
                            r.get::<_, String>(6)?,
                        ))
                    },
                )
                .optional()?;
            let Some((step, chars, n, resident, usage, brakes, hist)) = row else { return Ok(None) };
            let bad = |what: &str, e: String| StoreError::Other(format!("corrupt pool snapshot ({what}): {e}"));
            Ok(Some(PoolSnapshot {
                step: Step(step as u64),
                chars: Chars(chars as u64),
                n_experts: n as u32,
                resident: serde_json::from_str(&resident).map_err(|e| bad("resident", e.to_string()))?,
                usage: B64.decode(usage).map_err(|e| bad("usage", e.to_string()))?,
                brakes: serde_json::from_str(&brakes).map_err(|e| bad("brakes", e.to_string()))?,
                halting_hist: serde_json::from_str(&hist).map_err(|e| bad("hist", e.to_string()))?,
            }))
        })
    }

    /// Steps at which pool snapshots exist, oldest first.
    pub fn list_pool_steps(&self, run_id: i64) -> StoreResult<Vec<(Step, Chars)>> {
        self.read(|c| {
            let mut stmt = c.prepare("SELECT step, chars FROM pool_snapshots WHERE run_id = ?1 ORDER BY step")?;
            let rows = stmt
                .query_map([run_id], |r| Ok((Step(r.get::<_, i64>(0)? as u64), Chars(r.get::<_, i64>(1)? as u64))))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    pub fn list_checkpoints(&self, run_id: i64) -> StoreResult<Vec<CheckpointInfo>> {
        self.read(|c| {
            let mut stmt = c.prepare(
                "SELECT id, step, chars, kind, path_rel, bytes, heldout_nats, n_experts, is_best, pinned, created_at \
                 FROM checkpoints WHERE run_id = ?1 AND state = 'ok' ORDER BY step DESC, id DESC",
            )?;
            let rows = stmt
                .query_map([run_id], |r| {
                    let kind: String = r.get(3)?;
                    Ok(CheckpointInfo {
                        id: r.get(0)?,
                        step: Step(r.get::<_, i64>(1)? as u64),
                        chars: Chars(r.get::<_, i64>(2)? as u64),
                        kind: serde_json::from_value(serde_json::Value::String(kind)).unwrap_or(CheckpointKind::Auto),
                        path: r.get(4)?,
                        bytes: Count(r.get::<_, i64>(5)? as u64),
                        heldout_nats: r.get(6)?,
                        n_experts: r.get::<_, Option<i64>>(7)?.map(|v| v as u32),
                        is_best: r.get::<_, i64>(8)? != 0,
                        pinned: r.get::<_, i64>(9)? != 0,
                        created_at: UnixMs(r.get::<_, i64>(10)? as u64),
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Active training time (ms) at the last recorded tick at or before `step`, so a resumed run's timer continues.
    pub fn active_ms_at(&self, run_id: i64, step: u64) -> StoreResult<u64> {
        self.read(|c| {
            Ok(c.query_row(
                "SELECT t_ms FROM ticks WHERE run_id = ?1 AND step <= ?2 ORDER BY step DESC LIMIT 1",
                params![run_id, step as i64],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
            .unwrap_or(0) as u64)
        })
    }
}

/// Tiny helper so snapshot serialisation reads uniformly in the closure above.
mod erased {
    pub trait Json {
        fn to_json(&self) -> String;
    }
    impl<T: serde::Serialize> Json for T {
        fn to_json(&self) -> String {
            serde_json::to_string(self).unwrap_or_else(|_| "null".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::test_run;
    use minagi_types::{Brake, BrakeReport, ExpertInfo};

    fn snap(step: u64, n: u32) -> PoolSnapshot {
        let b = |ok| Brake { ok, why: "x".into() };
        PoolSnapshot {
            step: Step(step),
            chars: Chars(step * 512),
            n_experts: n,
            resident: vec![ExpertInfo {
                uid: 1,
                slot: Some(0),
                gate: 1.0,
                use_share: 0.5,
                admits: 3,
                age_chars: Chars(10),
                on_trial: false,
                staleness: 0.1,
                dying: false,
            }],
            usage: (0..n).map(|i| (255 - i * 3) as u8).collect(),
            brakes: BrakeReport { room: b(true), used: b(true), earning: b(false), fits: b(true), honest: b(true) },
            halting_hist: vec![0.1, 0.9],
        }
    }

    #[test]
    fn pool_snapshots_round_trip_and_nearest_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        let run = test_run(&store);
        store.record_pool_snapshot(run.id, snap(100, 16));
        store.record_pool_snapshot(run.id, snap(300, 17));
        store.flush();
        let got = store.get_pool_snapshot(run.id, Some(250)).unwrap().unwrap();
        assert_eq!(got, snap(100, 16));
        assert_eq!(store.get_pool_snapshot(run.id, None).unwrap().unwrap().n_experts, 17);
        assert!(store.get_pool_snapshot(run.id, Some(50)).unwrap().is_none());
        assert_eq!(store.list_pool_steps(run.id).unwrap().len(), 2);
    }
}
