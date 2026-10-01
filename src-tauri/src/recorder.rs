//! The recorder turns the engine's event stream into three things: durable telemetry in SQLite, a throttled live
//! stream for the UI, and the insight (verdict, estimate, milestone) shown on the dashboard. It also watches the
//! run's goal and asks the engine to stop when it is reached.
//!
//! Throttling: `Step` events can arrive hundreds of times a second; the UI gets a tiny `Pulse` at most every 250 ms
//! and an averaged `Tick` at most once a second, and the database gets one row per tick.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use minagi_store::{NewEvent, RunProgress, Store};
use minagi_types::{
    AppError, Chars, Count, Ctl, EngineEvent, EvalPoint, EvalResult, Goal, GrowthEvent, Insight, LiveMsg, Outcome,
    RunState, Schedule, Stage, Step, StepMetrics, TickPoint, UnixMs, bits_to_nats, eta_to_target, milestone,
    nats_to_bits, start_bits, verdict,
};

use crate::hub::LiveHub;
use crate::state::{SessionFlags, SessionHandle};

const PULSE_EVERY: Duration = Duration::from_millis(250);
const TICK_EVERY: Duration = Duration::from_millis(1000);
/// Held-out score (bits per character) used for the "time to a good result" estimate when the goal has no target.
const DEFAULT_TARGET_BITS: f64 = 2.0;

pub struct RecorderCtx {
    pub run_id: i64,
    pub store: Store,
    pub hub: LiveHub,
    pub ctl: flume::Sender<Ctl>,
    pub flags: Arc<SessionFlags>,
    pub session_slot: Arc<Mutex<Option<SessionHandle>>>,
    pub goal: Option<Goal>,
    pub chars_total: Option<u64>,
    /// Active training time already spent before this session (when resuming).
    pub start_active_ms: u64,
    /// Held-out evaluations already recorded (when resuming).
    pub eval_points: Vec<EvalPoint>,
    pub initial_state: RunState,
    /// Tells the person a long run has ended (they will often be looking at something else).
    pub notify: Option<crate::state::Notifier>,
    pub run_name: String,
}

/// Averages `Step` events over a tick interval.
#[derive(Default)]
struct Acc {
    n: u32,
    train_sum: f64,
    train_n: u32,
    grad_sum: f64,
    grad_n: u32,
    cps_sum: f64,
    last: Option<StepMetrics>,
}

impl Acc {
    fn add(&mut self, m: &StepMetrics) {
        self.n += 1;
        if m.train_nats.is_finite() {
            self.train_sum += m.train_nats;
            self.train_n += 1;
        }
        if m.grad_norm.is_finite() {
            self.grad_sum += m.grad_norm;
            self.grad_n += 1;
        }
        self.cps_sum += m.read_cps;
        self.last = Some(m.clone());
    }

    fn mean_cps(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.cps_sum / self.n as f64 }
    }

    fn take_tick(&mut self, active_ms: f64) -> Option<TickPoint> {
        let last = self.last.take()?;
        let tick = TickPoint {
            step: last.step,
            chars: last.chars_read,
            t_ms: Count(active_ms as u64),
            // A tick whose steps were all non-finite stays NaN so the chart shows the gap.
            train_nats: if self.train_n > 0 { self.train_sum / self.train_n as f64 } else { f64::NAN },
            grad_norm: if self.grad_n > 0 { self.grad_sum / self.grad_n as f64 } else { f64::NAN },
            lr_scale: last.lr_scale,
            lr_effective: last.lr_experts,
            read_cps: self.mean_cps(),
            write_cps: last.write_cps,
            context_now: last.context_now,
            n_experts: last.n_experts,
            avg_rows: last.avg_rows,
            rep8_pct: last.rep8_pct,
        };
        *self = Acc::default();
        Some(tick)
    }
}

struct Recorder {
    ctx: RecorderCtx,
    state: RunState,
    active_ms: f64,
    acc: Acc,
    last_pulse: Instant,
    last_tick: Instant,
    last_step: u64,
    last_chars: u64,
    train_ema: f64,
    schedule: Schedule,
    n_experts: u32,
    stage_name: Option<String>,
    last_heldout: Option<f64>,
    last_verdict: Option<minagi_types::Verdict>,
    goal_reached: bool,
    last_cps: f64,
}

/// Run until the engine reports it has finished. Blocks the calling (recorder) thread.
pub fn run(ctx: RecorderCtx, events: flume::Receiver<EngineEvent>) {
    let mut r = Recorder {
        state: ctx.initial_state,
        active_ms: ctx.start_active_ms as f64,
        acc: Acc::default(),
        last_pulse: Instant::now(),
        last_tick: Instant::now(),
        last_step: 0,
        last_chars: 0,
        train_ema: f64::NAN,
        schedule: Schedule::default(),
        n_experts: 0,
        stage_name: None,
        last_heldout: None,
        last_verdict: None,
        goal_reached: false,
        last_cps: 0.0,
        ctx,
    };
    while let Ok(event) = events.recv() {
        if r.handle(event) {
            break;
        }
    }
    // Whatever happened, the session is over: free the slot so a new run can start.
    let mut slot = r.ctx.session_slot.lock().unwrap();
    if slot.as_ref().is_some_and(|s| s.run_id == r.ctx.run_id) {
        *slot = None;
    }
}

impl Recorder {
    fn run_id(&self) -> i64 {
        self.ctx.run_id
    }

    /// Returns true when the run is over.
    fn handle(&mut self, event: EngineEvent) -> bool {
        match event {
            EngineEvent::Stage(info) => {
                self.stage_name = Some(format!("{:?}", info.stage));
                self.ctx.hub.send(LiveMsg::Stage { run_id: self.run_id(), info: info.clone(), at: UnixMs::now() });
                self.on_stage(info.stage);
            }
            EngineEvent::Step(m) => self.on_step(m),
            EngineEvent::Eval(r) => self.on_eval(r),
            EngineEvent::Samples(round) => {
                self.ctx.store.record_samples(self.run_id(), round.clone());
                self.ctx.hub.send(LiveMsg::Samples { run_id: self.run_id(), round });
            }
            EngineEvent::Pool(snapshot) => {
                self.ctx.store.record_pool_snapshot(self.run_id(), snapshot.clone());
                self.ctx.hub.send(LiveMsg::Pool { run_id: self.run_id(), snapshot });
            }
            EngineEvent::Growth(event) => {
                let (kind, uid, brake) = match &event {
                    GrowthEvent::Born { uid, .. } => ("expert_born", Some(*uid), None),
                    GrowthEvent::Pruned { uid, .. } => ("expert_pruned", Some(*uid), None),
                    GrowthEvent::Blocked { brakes } => {
                        let first = [
                            ("room", &brakes.room),
                            ("used", &brakes.used),
                            ("earning", &brakes.earning),
                            ("fits", &brakes.fits),
                            ("honest", &brakes.honest),
                        ]
                        .into_iter()
                        .find(|(_, b)| !b.ok)
                        .map(|(n, _)| n.to_string());
                        ("growth_blocked", None, first)
                    }
                };
                self.ctx.store.record_event(
                    self.run_id(),
                    NewEvent::new(self.last_step, self.last_chars, kind)
                        .expert(uid)
                        .brake(brake)
                        .payload(serde_json::to_value(&event).ok()),
                );
                self.ctx.hub.send(LiveMsg::Growth { run_id: self.run_id(), event });
            }
            EngineEvent::Plasticity(event) => {
                if event.jumped {
                    self.ctx.store.record_event(
                        self.run_id(),
                        NewEvent::new(self.last_step, self.last_chars, "plasticity_jump")
                            .payload(serde_json::to_value(&event).ok()),
                    );
                }
                self.ctx.hub.send(LiveMsg::Plasticity { run_id: self.run_id(), event });
            }
            EngineEvent::ContextGrow { from, to } => {
                self.ctx.store.record_event(
                    self.run_id(),
                    NewEvent::new(self.last_step, self.last_chars, "context_grow")
                        .payload(Some(serde_json::json!({ "from": from, "to": to }))),
                );
            }
            EngineEvent::CheckpointSaved(meta) => {
                if let Err(e) = self.ctx.store.record_checkpoint(self.run_id(), meta.clone()) {
                    eprintln!("[recorder] could not record checkpoint: {e}");
                }
                self.ctx.store.record_event(
                    self.run_id(),
                    NewEvent::new(meta.step.0, meta.chars.0, "checkpoint").payload(serde_json::to_value(&meta).ok()),
                );
                self.ctx.hub.send(LiveMsg::Checkpoint { run_id: self.run_id(), checkpoint: meta });
            }
            EngineEvent::Memory(_) => {}
            EngineEvent::Warn(d) => {
                self.ctx.store.record_event(
                    self.run_id(),
                    NewEvent::new(self.last_step, self.last_chars, "warning").payload(serde_json::to_value(&d).ok()),
                );
                self.ctx.hub.send(LiveMsg::Warning {
                    run_id: self.run_id(),
                    code: d.code,
                    message: d.message,
                    hint: d.hint,
                });
            }
            EngineEvent::Finished(outcome) => {
                self.finish(outcome);
                return true;
            }
        }
        false
    }

    /// Map the engine's stage to the run's lifecycle state, and persist/announce changes.
    fn on_stage(&mut self, stage: Stage) {
        let next = match stage {
            Stage::PreparingData | Stage::CreatingModel | Stage::LoadingCheckpoint => RunState::Preparing,
            Stage::Reading => RunState::Running,
            Stage::Evaluating | Stage::Sampling | Stage::GrowPrune | Stage::Checkpointing => {
                if self.state == RunState::Preparing { RunState::Running } else { self.state }
            }
            Stage::Paused => RunState::Paused,
            Stage::Stopping => RunState::Stopping,
            Stage::Idle | Stage::Finished | Stage::Failed => self.state,
        };
        if next != self.state {
            let (prev, now) = (self.state, next);
            self.state = next;
            match self.ctx.store.set_run_state(self.run_id(), now, None) {
                Ok(run) => self.ctx.hub.set_run(Some(run)),
                Err(e) => eprintln!("[recorder] could not update run state: {e}"),
            }
            self.ctx.hub.send(LiveMsg::State { run_id: self.run_id(), state: now, reason: None });
            let kind = match (prev, now) {
                (_, RunState::Paused) => Some("pause"),
                (RunState::Paused, RunState::Running) => Some("resume"),
                _ => None,
            };
            if let Some(kind) = kind {
                self.ctx.store.record_event(self.run_id(), NewEvent::new(self.last_step, self.last_chars, kind));
            }
        }
    }

    fn on_step(&mut self, m: StepMetrics) {
        self.active_ms += m.step_ms as f64;
        self.last_step = m.step.0;
        self.last_chars = m.chars_read.0;
        self.n_experts = m.n_experts;
        self.schedule = m.schedule.clone();
        if m.train_nats.is_finite() {
            self.train_ema = if self.train_ema.is_finite() {
                self.train_ema + 0.05 * (m.train_nats - self.train_ema)
            } else {
                m.train_nats
            };
        }
        self.last_cps = m.read_cps;
        self.acc.add(&m);
        let now = Instant::now();

        if now.duration_since(self.last_pulse) >= PULSE_EVERY {
            self.last_pulse = now;
            self.ctx.hub.send(LiveMsg::Pulse {
                run_id: self.run_id(),
                step: Step(self.last_step),
                chars: Chars(self.last_chars),
                chars_total: self.ctx.chars_total.map(Chars),
                train_nats_ema: self.train_ema,
                read_cps: self.acc.mean_cps(),
                active_ms: Count(self.active_ms as u64),
                schedule: self.schedule.clone(),
            });
            self.check_goal_progress();
        }
        if now.duration_since(self.last_tick) >= TICK_EVERY {
            self.last_tick = now;
            if let Some(tick) = self.acc.take_tick(self.active_ms) {
                self.ctx.hub.send(LiveMsg::Ticks { run_id: self.run_id(), points: vec![tick.clone()] });
                self.ctx.store.record_ticks(self.run_id(), vec![tick]);
                self.persist_progress();
            }
        }
    }

    fn on_eval(&mut self, r: EvalResult) {
        self.ctx.store.record_eval(self.run_id(), r.clone(), self.active_ms as u64);
        let point = EvalPoint {
            chars: r.chars.0 as f64,
            nats: r.overall_nats,
            se: r.overall_se,
            train_nats: r.train_nats_ema.is_finite().then_some(r.train_nats_ema),
        };
        self.ctx.eval_points.push(point);
        self.last_heldout = r.overall_nats.is_finite().then_some(r.overall_nats);
        self.ctx.hub.send(LiveMsg::Eval { run_id: self.run_id(), result: r });

        let insight = self.insight();
        self.last_verdict = Some(insight.report.verdict);
        self.ctx.hub.send(LiveMsg::Insight { run_id: self.run_id(), insight: insight.clone() });
        self.persist_progress();

        if let (Some(Goal::TargetBits { value }), Some(bits)) = (&self.ctx.goal, insight.bits_per_char)
            && bits <= *value
        {
            self.reach_goal();
        }
    }

    fn insight(&self) -> Insight {
        let points = &self.ctx.eval_points;
        let report = verdict(points);
        let bits = points.last().filter(|p| p.nats.is_finite()).map(|p| nats_to_bits(p.nats));
        let target = match &self.ctx.goal {
            Some(Goal::TargetBits { value }) => *value,
            _ => DEFAULT_TARGET_BITS,
        };
        Insight {
            report,
            eta: eta_to_target(points, bits_to_nats(target), self.last_cps),
            milestone: milestone(bits.unwrap_or_else(start_bits)),
            bits_per_char: bits,
        }
    }

    fn persist_progress(&self) {
        self.ctx.store.update_run_progress(RunProgress {
            run_id: self.run_id(),
            step: self.last_step,
            chars_read: self.last_chars,
            chars_total: self.ctx.chars_total,
            active_ms: self.active_ms as u64,
            last_train_nats: self.train_ema.is_finite().then_some(self.train_ema),
            last_heldout_nats: self.last_heldout,
            n_experts: (self.n_experts > 0).then_some(self.n_experts),
            stage: self.stage_name.clone(),
            verdict: self.last_verdict,
        });
    }

    /// Time- and character-based goals are checked as the run progresses.
    fn check_goal_progress(&mut self) {
        let reached = match &self.ctx.goal {
            Some(Goal::Minutes { value }) => self.active_ms >= value * 60_000.0,
            Some(Goal::Chars { value }) => self.last_chars >= value.0,
            _ => false,
        };
        if reached {
            self.reach_goal();
        }
    }

    fn reach_goal(&mut self) {
        if !self.goal_reached {
            self.goal_reached = true;
            let _ = self.ctx.ctl.send(Ctl::Stop { save: true });
        }
    }

    fn finish(&mut self, outcome: Outcome) {
        // Flush what is left of the current tick so the curves end where the run did.
        if let Some(tick) = self.acc.take_tick(self.active_ms) {
            self.ctx.hub.send(LiveMsg::Ticks { run_id: self.run_id(), points: vec![tick.clone()] });
            self.ctx.store.record_ticks(self.run_id(), vec![tick]);
        }
        let user_stopped = self.ctx.flags.user_stop.load(Ordering::SeqCst);
        let (state, error) = match &outcome {
            Outcome::Failed { error } => (RunState::Failed, Some(error.clone())),
            Outcome::Completed => (RunState::Completed, None),
            Outcome::Stopped if self.goal_reached && !user_stopped => (RunState::Completed, None),
            Outcome::Stopped => (RunState::Stopped, None),
        };
        self.state = state;
        self.persist_progress();
        self.ctx.store.flush();
        match self.ctx.store.set_run_state(self.run_id(), state, error.as_ref().map(|e: &AppError| e.to_string())) {
            Ok(run) => self.ctx.hub.set_run(Some(run)),
            Err(e) => eprintln!("[recorder] could not finalise run state: {e}"),
        }
        if let Some(error) = error {
            self.ctx.hub.send(LiveMsg::Error { run_id: self.run_id(), error, fatal: true });
        }
        self.ctx.hub.send(LiveMsg::State { run_id: self.run_id(), state, reason: None });
        if let Some(notify) = &self.ctx.notify {
            let name = &self.ctx.run_name;
            match state {
                RunState::Completed => {
                    notify("Training finished", &format!("“{name}” reached its goal and was saved."))
                }
                RunState::Failed => notify(
                    "Training stopped",
                    &format!("“{name}” ran into a problem. Open LLM Trainer to see what happened."),
                ),
                _ => {}
            }
        }
    }
}
