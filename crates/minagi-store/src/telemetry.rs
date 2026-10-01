//! Time-series telemetry: ticks and metrics, held-out evaluations, samples, timeline events, checkpoints,
//! resume truncation, and the downsampled series the charts read.

use minagi_types::{
    Chars, CheckpointMeta, Count, EvalPoint, EvalResult, SampleRound, SeriesData, SeriesRequest, Step, TickPoint,
    UnixMs, XAxis,
};
use rusqlite::{Connection, OptionalExtension, params};

use crate::db::{Store, StoreError, StoreResult};

/// A point of interest on a run's timeline (checkpoint, pause, expert born, warning...).
#[derive(Debug, Clone)]
pub struct NewEvent {
    pub step: u64,
    pub chars: u64,
    pub kind: String,
    pub expert_uid: Option<u32>,
    pub brake: Option<String>,
    pub payload: Option<serde_json::Value>,
}

impl NewEvent {
    pub fn new(step: u64, chars: u64, kind: &str) -> Self {
        Self { step, chars, kind: kind.to_string(), expert_uid: None, brake: None, payload: None }
    }

    pub fn expert(mut self, uid: Option<u32>) -> Self {
        self.expert_uid = uid;
        self
    }

    pub fn brake(mut self, brake: Option<String>) -> Self {
        self.brake = brake;
        self
    }

    pub fn payload(mut self, payload: Option<serde_json::Value>) -> Self {
        self.payload = payload;
        self
    }
}

/// Stable metric ids; must match the seed rows in `0001_init.sql`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKey {
    TrainNats = 1,
    GradNorm = 2,
    LrScale = 3,
    LrEffective = 4,
    ReadCps = 5,
    WriteCps = 6,
    CtxNow = 7,
    NExperts = 8,
    AvgRows = 9,
    Rep8Pct = 10,
}

impl MetricKey {
    pub const ALL: [MetricKey; 10] = [
        MetricKey::TrainNats,
        MetricKey::GradNorm,
        MetricKey::LrScale,
        MetricKey::LrEffective,
        MetricKey::ReadCps,
        MetricKey::WriteCps,
        MetricKey::CtxNow,
        MetricKey::NExperts,
        MetricKey::AvgRows,
        MetricKey::Rep8Pct,
    ];

    pub fn id(self) -> i64 {
        self as i64
    }

    pub fn name(self) -> &'static str {
        match self {
            MetricKey::TrainNats => "train.nats",
            MetricKey::GradNorm => "grad.norm",
            MetricKey::LrScale => "lr.scale",
            MetricKey::LrEffective => "lr.effective",
            MetricKey::ReadCps => "speed.read_cps",
            MetricKey::WriteCps => "speed.write_cps",
            MetricKey::CtxNow => "ctx.now",
            MetricKey::NExperts => "pool.n_experts",
            MetricKey::AvgRows => "halt.avg_rows",
            MetricKey::Rep8Pct => "text.rep8_pct",
        }
    }

    pub fn from_name(name: &str) -> Option<MetricKey> {
        MetricKey::ALL.into_iter().find(|k| k.name() == name)
    }
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

/// Snap a bucket width up to {1, 2, 5} x 10^k so bucket edges stay stable as data arrives and zooming merges cleanly.
pub fn snap_bucket(raw: u64) -> u64 {
    if raw <= 1 {
        return 1;
    }
    let mut mag = 1u64;
    while mag * 10 <= raw {
        mag *= 10;
    }
    for m in [1, 2, 5, 10] {
        if mag * m >= raw {
            return mag * m;
        }
    }
    mag * 10
}

/// Metrics-table rows for one tick.
fn metric_rows(p: &TickPoint) -> Vec<(MetricKey, Option<f64>)> {
    let mut v = vec![
        (MetricKey::TrainNats, finite(p.train_nats)),
        (MetricKey::GradNorm, finite(p.grad_norm)),
        (MetricKey::LrScale, finite(p.lr_scale)),
        (MetricKey::LrEffective, finite(p.lr_effective)),
        (MetricKey::ReadCps, finite(p.read_cps)),
        (MetricKey::CtxNow, Some(p.context_now as f64)),
        (MetricKey::NExperts, Some(p.n_experts as f64)),
        (MetricKey::AvgRows, finite(p.avg_rows as f64)),
    ];
    if let Some(w) = p.write_cps {
        v.push((MetricKey::WriteCps, finite(w)));
    }
    if let Some(r) = p.rep8_pct {
        v.push((MetricKey::Rep8Pct, finite(r as f64)));
    }
    v
}

impl Store {
    /// Persist a batch of ticks (batched; returns immediately). Non-finite values are stored as NULL.
    pub fn record_ticks(&self, run_id: i64, points: Vec<TickPoint>) {
        if points.is_empty() {
            return;
        }
        self.write_batched(move |conn| {
            let wall = UnixMs::now().0 as i64;
            let mut tick = conn.prepare_cached(
                "INSERT OR REPLACE INTO ticks (run_id, step, chars, t_ms, wall_ts) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let mut metric = conn.prepare_cached(
                "INSERT OR REPLACE INTO metrics (run_id, key_id, step, value) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for p in &points {
                tick.execute(params![run_id, p.step.0 as i64, p.chars.0 as i64, p.t_ms.0 as i64, wall])?;
                for (key, value) in metric_rows(p) {
                    metric.execute(params![run_id, key.id(), p.step.0 as i64, value])?;
                }
            }
            Ok(())
        });
    }

    /// Persist one held-out evaluation round with its per-domain scores.
    pub fn record_eval(&self, run_id: i64, result: EvalResult, t_ms: u64) {
        self.write_batched(move |conn| {
            let wall = UnixMs::now().0 as i64;
            conn.execute(
                "INSERT OR REPLACE INTO eval_rounds (run_id, step, chars, t_ms, wall_ts, overall_nats, overall_se, train_nats_ema) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    run_id,
                    result.step.0 as i64,
                    result.chars.0 as i64,
                    t_ms as i64,
                    wall,
                    finite(result.overall_nats),
                    finite(result.overall_se),
                    finite(result.train_nats_ema)
                ],
            )?;
            let round_id: i64 =
                conn.query_row("SELECT id FROM eval_rounds WHERE run_id = ?1 AND step = ?2", params![run_id, result.step.0 as i64], |r| r.get(0))?;
            conn.execute("DELETE FROM eval_domain WHERE round_id = ?1", [round_id])?;
            let mut stmt = conn.prepare_cached(
                "INSERT INTO eval_domain (round_id, domain, nats, se, n_chars) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for d in &result.domains {
                stmt.execute(params![round_id, d.domain, finite(d.nats), finite(d.se), d.n_chars.0 as i64])?;
            }
            Ok(())
        });
    }

    /// All held-out evaluations of a run, oldest first (input to the verdict, ETA and charts).
    pub fn eval_points(&self, run_id: i64) -> StoreResult<Vec<EvalPoint>> {
        self.read(|c| {
            let mut stmt = c.prepare(
                "SELECT chars, overall_nats, overall_se, train_nats_ema FROM eval_rounds WHERE run_id = ?1 ORDER BY step",
            )?;
            let rows = stmt
                .query_map([run_id], |r| {
                    Ok(EvalPoint {
                        chars: r.get::<_, i64>(0)? as f64,
                        nats: r.get::<_, Option<f64>>(1)?.unwrap_or(f64::NAN),
                        se: r.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                        train_nats: r.get(3)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Names of the held-out domains scored in a run, alphabetically (colours follow the name, not the position).
    pub fn eval_domains(&self, run_id: i64) -> StoreResult<Vec<String>> {
        self.read(|c| {
            let mut stmt = c.prepare(
                "SELECT DISTINCT d.domain FROM eval_domain d JOIN eval_rounds e ON e.id = d.round_id \
                 WHERE e.run_id = ?1 ORDER BY d.domain",
            )?;
            let rows = stmt.query_map([run_id], |r| r.get::<_, String>(0))?.collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    pub fn record_samples(&self, run_id: i64, round: SampleRound) {
        self.write_batched(move |conn| {
            let mut stmt = conn.prepare_cached(
                "INSERT OR REPLACE INTO samples (run_id, step, idx, chars, domain, prompt, raw, adapted, raw_rep8_pct, adapted_rep8_pct, note) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )?;
            for (i, s) in round.items.iter().enumerate() {
                stmt.execute(params![
                    run_id,
                    round.step.0 as i64,
                    i as i64,
                    round.chars.0 as i64,
                    s.domain,
                    s.prompt,
                    s.raw,
                    s.adapted,
                    s.raw_rep8,
                    s.adapted_rep8,
                    s.note
                ])?;
            }
            Ok(())
        });
    }

    /// Steps at which sample rounds exist, oldest first (drives the "watch it learn to write" slider).
    pub fn list_sample_steps(&self, run_id: i64) -> StoreResult<Vec<(Step, Chars)>> {
        self.read(|c| {
            let mut stmt = c.prepare("SELECT DISTINCT step, chars FROM samples WHERE run_id = ?1 ORDER BY step")?;
            let rows = stmt
                .query_map([run_id], |r| Ok((Step(r.get::<_, i64>(0)? as u64), Chars(r.get::<_, i64>(1)? as u64))))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Sample round at `step`, or the latest one when `step` is `None`.
    pub fn get_samples(&self, run_id: i64, step: Option<u64>) -> StoreResult<Option<SampleRound>> {
        self.read(|c| {
            let step = match step {
                Some(s) => Some(s as i64),
                None => c.query_row("SELECT MAX(step) FROM samples WHERE run_id = ?1", [run_id], |r| {
                    r.get::<_, Option<i64>>(0)
                })?,
            };
            let Some(step) = step else { return Ok(None) };
            let mut stmt = c.prepare(
                "SELECT domain, prompt, raw, adapted, raw_rep8_pct, adapted_rep8_pct, note, chars \
                 FROM samples WHERE run_id = ?1 AND step = ?2 ORDER BY idx",
            )?;
            let mut chars = 0u64;
            let items = stmt
                .query_map(params![run_id, step], |r| {
                    chars = r.get::<_, i64>(7)? as u64;
                    Ok(minagi_types::SampleItem {
                        domain: r.get(0)?,
                        prompt: r.get(1)?,
                        raw: r.get(2)?,
                        adapted: r.get(3)?,
                        raw_rep8: r.get::<_, Option<f64>>(4)?.map(|v| v as f32),
                        adapted_rep8: r.get::<_, Option<f64>>(5)?.map(|v| v as f32),
                        note: r.get(6)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            if items.is_empty() {
                return Ok(None);
            }
            Ok(Some(SampleRound { step: Step(step as u64), chars: Chars(chars), items }))
        })
    }

    /// Append a timeline annotation.
    pub fn record_event(&self, run_id: i64, ev: NewEvent) {
        self.write_batched(move |conn| {
            conn.execute(
                "INSERT INTO run_events (run_id, step, chars, wall_ts, kind, expert_uid, brake, payload_json) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    run_id,
                    ev.step as i64,
                    ev.chars as i64,
                    UnixMs::now().0 as i64,
                    ev.kind,
                    ev.expert_uid,
                    ev.brake,
                    ev.payload.map(|p| p.to_string())
                ],
            )?;
            Ok(())
        });
    }

    pub fn get_events(&self, run_id: i64, kinds: Option<Vec<String>>) -> StoreResult<Vec<minagi_types::RunEvent>> {
        self.read(|c| {
            let mut stmt = c.prepare(
                "SELECT id, step, chars, wall_ts, kind, expert_uid, brake, payload_json FROM run_events \
                 WHERE run_id = ?1 ORDER BY step, id",
            )?;
            let all = stmt
                .query_map([run_id], |r| {
                    Ok(minagi_types::RunEvent {
                        id: r.get(0)?,
                        step: Step(r.get::<_, i64>(1)? as u64),
                        chars: Chars(r.get::<_, i64>(2)? as u64),
                        at: UnixMs(r.get::<_, i64>(3)? as u64),
                        kind: r.get(4)?,
                        expert_uid: r.get::<_, Option<i64>>(5)?.map(|v| v as u32),
                        brake: r.get(6)?,
                        payload: r.get(7)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(match kinds {
                Some(k) if !k.is_empty() => all.into_iter().filter(|e| k.contains(&e.kind)).collect(),
                _ => all,
            })
        })
    }

    /// Record a checkpoint. Durable immediately: checkpoints are what a crash resumes from.
    pub fn record_checkpoint(&self, run_id: i64, meta: CheckpointMeta) -> StoreResult<i64> {
        self.write(move |conn| {
            let kind = serde_json::to_value(meta.kind)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| "auto".into());
            conn.execute(
                "INSERT INTO checkpoints (run_id, step, chars, kind, path_rel, bytes, heldout_nats, n_experts, engine_format, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    run_id,
                    meta.step.0 as i64,
                    meta.chars.0 as i64,
                    kind,
                    meta.path,
                    meta.bytes.0 as i64,
                    meta.heldout_nats.and_then(finite),
                    meta.n_experts,
                    meta.engine_format,
                    UnixMs::now().0 as i64
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    /// Latest checkpoint a run can resume from: (step, chars, path relative to the run directory).
    pub fn latest_checkpoint(&self, run_id: i64) -> StoreResult<Option<(Step, Chars, String)>> {
        self.read(|c| {
            Ok(c.query_row(
                "SELECT step, chars, path_rel FROM checkpoints WHERE run_id = ?1 AND state = 'ok' ORDER BY step DESC LIMIT 1",
                [run_id],
                |r| Ok((Step(r.get::<_, i64>(0)? as u64), Chars(r.get::<_, i64>(1)? as u64), r.get(2)?)),
            )
            .optional()?)
        })
    }

    /// Delete telemetry newer than `step` so curves stay honest after resuming from an older checkpoint.
    /// Returns the number of tick rows removed.
    pub fn truncate_after(&self, run_id: i64, step: u64) -> StoreResult<usize> {
        self.write(move |conn| {
            let tx = conn.transaction()?;
            let s = step as i64;
            let ticks = tx.execute("DELETE FROM ticks WHERE run_id = ?1 AND step > ?2", params![run_id, s])?;
            tx.execute("DELETE FROM metrics WHERE run_id = ?1 AND step > ?2", params![run_id, s])?;
            tx.execute("DELETE FROM eval_rounds WHERE run_id = ?1 AND step > ?2", params![run_id, s])?; // cascades to eval_domain
            tx.execute("DELETE FROM samples WHERE run_id = ?1 AND step > ?2", params![run_id, s])?;
            tx.execute("DELETE FROM run_events WHERE run_id = ?1 AND step > ?2", params![run_id, s])?;
            tx.execute("DELETE FROM pool_snapshots WHERE run_id = ?1 AND step > ?2", params![run_id, s])?;
            tx.commit()?;
            Ok(ticks)
        })
    }

    /// Chart series, downsampled on the server. Keys are metric names (`train.nats`, ...) or evaluation series
    /// (`eval.overall`, `eval.train`, `eval.gap`, `eval.domain.<name>`).
    pub fn get_series(&self, req: &SeriesRequest) -> StoreResult<Vec<SeriesData>> {
        let max_points = if req.max_points == 0 { 1000 } else { req.max_points.min(5000) } as u64;
        self.read(|c| {
            let mut out = Vec::with_capacity(req.keys.len());
            for key in &req.keys {
                let data = if let Some(rest) = key.strip_prefix("eval.") {
                    eval_series(c, req, key, rest)?
                } else if let Some(mk) = MetricKey::from_name(key) {
                    metric_series(c, req, mk, max_points)?
                } else {
                    return Err(StoreError::NotFound(format!("metric '{key}'")));
                };
                out.push(data);
            }
            Ok(out)
        })
    }
}

fn x_column(x: XAxis) -> &'static str {
    match x {
        XAxis::Chars => "chars",
        XAxis::Step => "step",
        XAxis::Time => "t_ms",
    }
}

/// Convert a UI x value (seconds for time) into the column's unit.
fn x_to_db(x: XAxis, v: f64) -> f64 {
    if x == XAxis::Time { v * 1000.0 } else { v }
}

fn x_from_db(x: XAxis, step: i64, chars: i64, t_ms: i64) -> f64 {
    match x {
        XAxis::Chars => chars as f64,
        XAxis::Step => step as f64,
        XAxis::Time => t_ms as f64 / 1000.0,
    }
}

fn metric_series(c: &Connection, req: &SeriesRequest, key: MetricKey, max_points: u64) -> StoreResult<SeriesData> {
    let col = x_column(req.x);
    let lo = req.from.map(|v| x_to_db(req.x, v)).unwrap_or(f64::MIN);
    let hi = req.to.map(|v| x_to_db(req.x, v)).unwrap_or(f64::MAX);
    let bounds: (Option<i64>, Option<i64>) = c.query_row(
        &format!("SELECT MIN(step), MAX(step) FROM ticks WHERE run_id = ?1 AND {col} >= ?2 AND {col} <= ?3"),
        params![req.run_id, lo, hi],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let mut data =
        SeriesData { key: key.name().to_string(), x: vec![], y: vec![], y_lo: vec![], y_hi: vec![], bucket: 1 };
    let (Some(s0), Some(s1)) = bounds else { return Ok(data) };

    // Aligned bucket edges can make a range touch one more bucket than span / width, so size buckets against
    // `max_points - 1` to make the point budget a hard guarantee.
    let span = (s1 - s0 + 1) as u64;
    let bw = snap_bucket(span.div_ceil(max_points.saturating_sub(1).max(1)));
    data.bucket = bw as u32;

    let mut stmt = c.prepare_cached(
        "SELECT MAX(step), AVG(value), MIN(value), MAX(value) FROM metrics \
         WHERE run_id = ?1 AND key_id = ?2 AND step BETWEEN ?3 AND ?4 \
         GROUP BY step / ?5 ORDER BY step / ?5",
    )?;
    let mut lookup = c.prepare_cached("SELECT chars, t_ms FROM ticks WHERE run_id = ?1 AND step = ?2")?;
    let mut rows = stmt.query(params![req.run_id, key.id(), s0, s1, bw as i64])?;
    while let Some(r) = rows.next()? {
        let step: i64 = r.get(0)?;
        let (chars, t_ms): (i64, i64) =
            match lookup.query_row(params![req.run_id, step], |t| Ok((t.get(0)?, t.get(1)?))) {
                Ok(v) => v,
                Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                Err(e) => return Err(e.into()),
            };
        data.x.push(x_from_db(req.x, step, chars, t_ms));
        data.y.push(r.get(1)?);
        data.y_lo.push(r.get(2)?);
        data.y_hi.push(r.get(3)?);
    }
    Ok(data)
}

fn eval_series(c: &Connection, req: &SeriesRequest, full_key: &str, rest: &str) -> StoreResult<SeriesData> {
    let mut data =
        SeriesData { key: full_key.to_string(), x: vec![], y: vec![], y_lo: vec![], y_hi: vec![], bucket: 1 };
    let lo = req.from.map(|v| x_to_db(req.x, v)).unwrap_or(f64::MIN);
    let hi = req.to.map(|v| x_to_db(req.x, v)).unwrap_or(f64::MAX);
    let col = match req.x {
        XAxis::Chars => "e.chars",
        XAxis::Step => "e.step",
        XAxis::Time => "e.t_ms",
    };
    // (value, se) per round, depending on which evaluation series is requested.
    let sql = if let Some(domain) = rest.strip_prefix("domain.") {
        let _ = domain;
        format!(
            "SELECT e.step, e.chars, e.t_ms, d.nats, d.se FROM eval_rounds e JOIN eval_domain d ON d.round_id = e.id \
             WHERE e.run_id = ?1 AND d.domain = ?2 AND {col} >= ?3 AND {col} <= ?4 ORDER BY e.step"
        )
    } else {
        let value = match rest {
            "overall" => "e.overall_nats",
            "train" => "e.train_nats_ema",
            "gap" => "(e.overall_nats - e.train_nats_ema)",
            other => return Err(StoreError::NotFound(format!("series 'eval.{other}'"))),
        };
        format!(
            "SELECT e.step, e.chars, e.t_ms, {value}, e.overall_se FROM eval_rounds e \
             WHERE e.run_id = ?1 AND ?2 = ?2 AND {col} >= ?3 AND {col} <= ?4 ORDER BY e.step"
        )
    };
    let domain = rest.strip_prefix("domain.").unwrap_or("");
    let mut stmt = c.prepare(&sql)?;
    let mut rows = stmt.query(params![req.run_id, domain, lo, hi])?;
    while let Some(r) = rows.next()? {
        let (step, chars, t_ms): (i64, i64, i64) = (r.get(0)?, r.get(1)?, r.get(2)?);
        let v: Option<f64> = r.get(3)?;
        let se: Option<f64> = r.get(4)?;
        data.x.push(x_from_db(req.x, step, chars, t_ms));
        data.y.push(v);
        // For evaluation series the envelope is the standard error band.
        data.y_lo.push(v.zip(se).map(|(v, s)| v - s));
        data.y_hi.push(v.zip(se).map(|(v, s)| v + s));
    }
    Ok(data)
}

#[allow(unused)]
fn _assert_types(_: Count) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::test_run;
    use minagi_types::{DomainScore, SampleItem};
    use std::time::Instant;

    fn open() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("t.db")).unwrap();
        (dir, store)
    }

    fn tick(step: u64) -> TickPoint {
        TickPoint {
            step: Step(step),
            chars: Chars(step * 512),
            t_ms: Count(step * 10),
            train_nats: 5.0 - (step as f64 + 1.0).ln() * 0.3,
            grad_norm: 1.0,
            lr_scale: 1.0,
            lr_effective: 1e-3,
            read_cps: 9000.0,
            write_cps: None,
            context_now: 512,
            n_experts: 16,
            avg_rows: 3.0,
            rep8_pct: None,
        }
    }

    fn req(run_id: i64, keys: &[&str], x: XAxis, max_points: u32) -> SeriesRequest {
        SeriesRequest {
            run_id,
            keys: keys.iter().map(|s| s.to_string()).collect(),
            x,
            from: None,
            to: None,
            max_points,
        }
    }

    #[test]
    fn bucket_widths_snap_to_1_2_5() {
        assert_eq!(snap_bucket(0), 1);
        assert_eq!(snap_bucket(1), 1);
        assert_eq!(snap_bucket(2), 2);
        assert_eq!(snap_bucket(3), 5);
        assert_eq!(snap_bucket(6), 10);
        assert_eq!(snap_bucket(11), 20);
        assert_eq!(snap_bucket(51), 100);
        assert_eq!(snap_bucket(500), 500);
        assert_eq!(snap_bucket(501), 1000);
    }

    #[test]
    fn raw_series_round_trips_when_it_fits() {
        let (_d, store) = open();
        let run = test_run(&store);
        store.record_ticks(run.id, (1..=50).map(tick).collect());
        store.flush();
        let s = &store.get_series(&req(run.id, &["train.nats"], XAxis::Chars, 1000)).unwrap()[0];
        assert_eq!(s.x.len(), 50);
        assert_eq!(s.bucket, 1);
        assert_eq!(s.x[0], 512.0);
        assert!((s.y[0].unwrap() - tick(1).train_nats).abs() < 1e-12);
    }

    #[test]
    fn downsampling_respects_the_budget_and_keeps_spikes() {
        let (_d, store) = open();
        let run = test_run(&store);
        let mut pts: Vec<TickPoint> = (1..=20_000).map(tick).collect();
        pts[10_000].grad_norm = 500.0; // a spike
        store.record_ticks(run.id, pts);
        store.flush();
        let s = &store.get_series(&req(run.id, &["grad.norm"], XAxis::Step, 400)).unwrap()[0];
        assert!(s.x.len() <= 400, "got {} points", s.x.len());
        assert!(s.x.windows(2).all(|w| w[0] < w[1]), "x must be strictly increasing");
        let max_hi = s.y_hi.iter().flatten().cloned().fold(f64::MIN, f64::max);
        assert_eq!(max_hi, 500.0, "min/max envelope must preserve the spike");
        assert!(s.y.iter().flatten().all(|v| *v < 500.0), "the mean line must not be dominated by one spike");
    }

    #[test]
    fn bucket_edges_are_stable_as_data_arrives() {
        let (_d, store) = open();
        let run = test_run(&store);
        store.record_ticks(run.id, (1..=10_000).map(tick).collect());
        store.flush();
        let before = store.get_series(&req(run.id, &["train.nats"], XAxis::Step, 100)).unwrap().remove(0);
        store.record_ticks(run.id, (10_001..=10_020).map(tick).collect());
        store.flush();
        let after = store.get_series(&req(run.id, &["train.nats"], XAxis::Step, 100)).unwrap().remove(0);
        assert_eq!(before.bucket, after.bucket, "a few more points must not change the bucket width");
        let n = before.x.len() - 1; // all but the still-filling last bucket
        assert_eq!(&before.x[..n], &after.x[..n]);
        assert_eq!(&before.y[..n], &after.y[..n]);
    }

    #[test]
    fn non_finite_values_are_stored_as_null() {
        let (_d, store) = open();
        let run = test_run(&store);
        let mut t = tick(1);
        t.train_nats = f64::NAN;
        store.record_ticks(run.id, vec![t, tick(2)]);
        store.flush();
        let s = &store.get_series(&req(run.id, &["train.nats"], XAxis::Step, 100)).unwrap()[0];
        assert_eq!(s.y[0], None);
        assert!(s.y[1].is_some());
    }

    #[test]
    fn time_axis_is_in_seconds_and_ranges_filter() {
        let (_d, store) = open();
        let run = test_run(&store);
        store.record_ticks(run.id, (1..=100).map(tick).collect()); // t_ms = step * 10
        store.flush();
        let mut r = req(run.id, &["train.nats"], XAxis::Time, 1000);
        r.from = Some(0.5); // seconds -> steps 50..
        r.to = Some(0.8);
        let s = &store.get_series(&r).unwrap()[0];
        assert_eq!(s.x.first().copied(), Some(0.5));
        assert_eq!(s.x.last().copied(), Some(0.8));
        assert_eq!(s.x.len(), 31);
    }

    #[test]
    fn eval_series_overall_domain_and_gap() {
        let (_d, store) = open();
        let run = test_run(&store);
        for (i, nats) in [3.0, 2.5, 2.2].into_iter().enumerate() {
            store.record_eval(
                run.id,
                EvalResult {
                    step: Step((i as u64 + 1) * 100),
                    chars: Chars((i as u64 + 1) * 51_200),
                    overall_nats: nats,
                    overall_se: 0.02,
                    train_nats_ema: nats - 0.1,
                    domains: vec![
                        DomainScore { domain: "stories".into(), nats: nats - 0.2, se: 0.03, n_chars: Count(30_000) },
                        DomainScore { domain: "arithmetic".into(), nats: nats + 0.5, se: 0.05, n_chars: Count(30_000) },
                    ],
                },
                i as u64 * 1000,
            );
        }
        store.flush();
        let got = store
            .get_series(&req(run.id, &["eval.overall", "eval.domain.arithmetic", "eval.gap"], XAxis::Chars, 100))
            .unwrap();
        assert_eq!(got[0].y, vec![Some(3.0), Some(2.5), Some(2.2)]);
        assert_eq!(got[0].y_lo[0], Some(3.0 - 0.02));
        assert_eq!(got[1].y, vec![Some(3.5), Some(3.0), Some(2.7)]);
        for v in got[2].y.iter().flatten() {
            assert!((v - 0.1).abs() < 1e-9);
        }
        assert_eq!(store.eval_domains(run.id).unwrap(), vec!["arithmetic".to_string(), "stories".to_string()]);
        let pts = store.eval_points(run.id).unwrap();
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[2].nats, 2.2);
    }

    #[test]
    fn resume_truncation_removes_only_newer_rows() {
        let (_d, store) = open();
        let run = test_run(&store);
        store.record_ticks(run.id, (1..=200).map(tick).collect());
        store.record_eval(
            run.id,
            EvalResult {
                step: Step(100),
                chars: Chars(51_200),
                overall_nats: 3.0,
                overall_se: 0.0,
                train_nats_ema: 3.0,
                domains: vec![],
            },
            0,
        );
        store.record_eval(
            run.id,
            EvalResult {
                step: Step(200),
                chars: Chars(102_400),
                overall_nats: 2.0,
                overall_se: 0.0,
                train_nats_ema: 2.0,
                domains: vec![],
            },
            0,
        );
        store.record_samples(
            run.id,
            SampleRound {
                step: Step(200),
                chars: Chars(102_400),
                items: vec![SampleItem {
                    domain: "d".into(),
                    prompt: "p".into(),
                    raw: "r".into(),
                    adapted: "a".into(),
                    raw_rep8: None,
                    adapted_rep8: None,
                    note: None,
                }],
            },
        );
        store.flush();
        let removed = store.truncate_after(run.id, 100).unwrap();
        assert_eq!(removed, 100);
        let s = &store.get_series(&req(run.id, &["train.nats"], XAxis::Step, 1000)).unwrap()[0];
        assert_eq!(s.x.last().copied(), Some(100.0));
        assert_eq!(store.eval_points(run.id).unwrap().len(), 1);
        assert!(store.get_samples(run.id, None).unwrap().is_none());
    }

    #[test]
    fn samples_round_trip_and_latest_is_default() {
        let (_d, store) = open();
        let run = test_run(&store);
        let item = |t: &str| SampleItem {
            domain: "stories".into(),
            prompt: "Once upon a time".into(),
            raw: t.into(),
            adapted: format!("{t}!"),
            raw_rep8: Some(10.0),
            adapted_rep8: Some(2.0),
            note: Some("n".into()),
        };
        for step in [100u64, 200] {
            store.record_samples(
                run.id,
                SampleRound { step: Step(step), chars: Chars(step * 512), items: vec![item(&format!("s{step}"))] },
            );
        }
        store.flush();
        let latest = store.get_samples(run.id, None).unwrap().unwrap();
        assert_eq!(latest.step.0, 200);
        assert_eq!(latest.items[0].raw, "s200");
        assert_eq!(latest.items[0].adapted_rep8, Some(2.0));
        assert_eq!(store.list_sample_steps(run.id).unwrap().len(), 2);
        assert_eq!(store.get_samples(run.id, Some(100)).unwrap().unwrap().items[0].raw, "s100");
    }

    #[test]
    fn checkpoints_and_latest_lookup() {
        let (_d, store) = open();
        let run = test_run(&store);
        for step in [100u64, 300, 200] {
            store
                .record_checkpoint(
                    run.id,
                    CheckpointMeta {
                        step: Step(step),
                        chars: Chars(step * 512),
                        kind: minagi_types::CheckpointKind::Auto,
                        path: format!("checkpoints/step-{step:09}"),
                        bytes: Count(1000),
                        heldout_nats: Some(2.0),
                        n_experts: 16,
                        engine_format: 1,
                    },
                )
                .unwrap();
        }
        let (step, _chars, path) = store.latest_checkpoint(run.id).unwrap().unwrap();
        assert_eq!(step.0, 300);
        assert_eq!(path, "checkpoints/step-000000300");
    }

    #[test]
    fn writing_200k_ticks_is_fast_and_a_million_row_series_queries_quickly() {
        let (_d, store) = open();
        let run = test_run(&store);
        let t0 = Instant::now();
        for chunk in (1..=100_000u64).collect::<Vec<_>>().chunks(1000) {
            store.record_ticks(run.id, chunk.iter().map(|s| tick(*s)).collect());
        }
        store.flush();
        let write = t0.elapsed();
        assert!(write.as_secs_f64() < 20.0, "writing 100k ticks (~1M metric rows) took {write:?}");
        let t1 = Instant::now();
        let s = &store
            .get_series(&req(run.id, &["train.nats", "grad.norm", "speed.read_cps"], XAxis::Chars, 2000))
            .unwrap();
        let q = t1.elapsed();
        assert!(s[0].x.len() <= 2000);
        assert!(q.as_millis() < 600, "3 series over 100k ticks took {q:?}");
    }
}
