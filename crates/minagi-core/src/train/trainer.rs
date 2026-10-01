//! The run loop: read a window of text, guess every next character, learn from the mistakes, and every so often stop to
//! measure, write a sample, save, and decide whether the expert pool should grow or shrink.
//!
//! One step is one optimiser update on the window of text ending at the next chunk. The model re-reads its whole window
//! each step (so the gradient reaches all of it) and is scored on all of it. Everything the loop does besides training
//! is scheduled in wall-clock time (evaluation and samples, checkpoints) or in characters (growth), exactly as the
//! original's reader does, and is announced to the app as [`EngineEvent`]s.

use std::cell::Cell;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use flume::{Receiver, RecvTimeoutError};
use minagi_text::{ByteTokenizer, EvalSet, StreamConfig, TextError, TrainStream};
use minagi_types::{
    AppError, BrakeReport, Chars, CheckpointKind, CheckpointMeta, Count, Ctl, Diagnostic, EngineEvent, EvalResult,
    EventSink, ExpertInfo, GrowthEvent, MemoryReport, Outcome, PoolSnapshot, Preset, RunSpec, SampleItem, SampleRound,
    Schedule, Stage, StageInfo, Step, StepMetrics, TrainConfig,
};

use super::context::{ContextRamp, RampSettings, chars_to_steps};
use super::eval::Evaluator;
use super::session::{RunPaths, Saved, create_model, load_model, preset_of, read_checkpoint, write_checkpoint};
use super::step::{StepSettings, train_on_window};
use crate::backend;
use crate::chat::{Decode, Writer, repeat_rate};
use crate::error::{EngineError, Result};
use crate::model::{Model, ModelOpts, sample_depth};
use crate::ops::Kernel;
use crate::optim::adamw::{AdamHyper, AdamW};
use crate::optim::brakes::{GrowthInputs, decide};
use crate::optim::plasticity::Plasticity;
use crate::rng::HostRng;

const READING: &str = "Reading text and learning from its mistakes";
/// Characters each prompt is continued by in a sample round.
const SAMPLE_CHARS: usize = 140;
/// Recent training losses kept for the held-out gap and the train curve at evaluation time.
const RECENT: usize = 400;
/// After this many steps in a row with a non-finite loss the run is stopped as diverged.
const MAX_NAN_STREAK: u32 = 20;
/// Restore-from-checkpoint hint appended to a divergence error.
const DIVERGED: &str = "The training score became invalid (NaN) over and over, so the run was stopped. Resume from \
                        the last saved progress with a lower learning rate.";

fn kernel_for(train: &TrainConfig, metal: bool) -> Kernel {
    if train.row_checkpointing {
        Kernel { hand: true, q_block: 512, ckpt: true, sync_tiles: metal }
    } else {
        Kernel::hand()
    }
}

/// Control messages from the app, applied between steps (and polled during long evaluations so Stop is prompt).
struct Control<'a> {
    rx: &'a Receiver<Ctl>,
    stop: Cell<Option<bool>>,
    paused: Cell<bool>,
    eval_now: Cell<bool>,
    sample_now: Cell<bool>,
    ckpt_now: Cell<bool>,
    sample_every_min: Cell<Option<f64>>,
    ckpt_every_min: Cell<Option<f64>>,
}

impl<'a> Control<'a> {
    fn new(rx: &'a Receiver<Ctl>) -> Self {
        Self {
            rx,
            stop: Cell::new(None),
            paused: Cell::new(false),
            eval_now: Cell::new(false),
            sample_now: Cell::new(false),
            ckpt_now: Cell::new(false),
            sample_every_min: Cell::new(None),
            ckpt_every_min: Cell::new(None),
        }
    }

    fn apply(&self, msg: Ctl) {
        match msg {
            Ctl::Pause => self.paused.set(true),
            Ctl::Resume => self.paused.set(false),
            Ctl::Stop { save } => self.stop.set(Some(save)),
            Ctl::CheckpointNow => self.ckpt_now.set(true),
            Ctl::SampleNow => self.sample_now.set(true),
            Ctl::EvalNow => self.eval_now.set(true),
            Ctl::SetSampleEveryMin { minutes } => self.sample_every_min.set(Some(minutes)),
            Ctl::SetCheckpointEveryMin { minutes } => self.ckpt_every_min.set(Some(minutes)),
        }
    }

    /// Apply everything waiting. If the app dropped its end, nobody is listening: stop without saving.
    fn poll(&self) {
        loop {
            match self.rx.try_recv() {
                Ok(m) => self.apply(m),
                Err(flume::TryRecvError::Empty) => return,
                Err(flume::TryRecvError::Disconnected) => {
                    if self.stop.get().is_none() {
                        self.stop.set(Some(false));
                    }
                    return;
                }
            }
        }
    }

    fn stop_requested(&self) -> bool {
        self.poll();
        self.stop.get().is_some()
    }
}

struct StepReport {
    timing: super::step::StepTiming,
    loss: f64,
    grad_norm: f64,
    rows: f32,
    row_mass: Vec<f32>,
    new_tokens: usize,
    skipped: bool,
}

struct Trainer<'a> {
    spec: &'a RunSpec,
    sink: &'a dyn EventSink,
    ctl: Control<'a>,
    model: Model,
    opt: AdamW,
    plast: Plasticity,
    ramp: ContextRamp,
    stream: TrainStream,
    evaluator: Option<Evaluator>,
    prompts: Vec<(String, String)>,
    paths: RunPaths,
    preset: Preset,
    gpu_budget_gb: f64,
    rng: HostRng,
    grow_rng: HostRng,

    step: u64,
    /// Characters read over the model's whole life.
    chars: u64,
    nats: f64,
    chunk_in_visit: u32,
    recent: VecDeque<f64>,
    last_heldout: Option<f64>,
    last_gap: Option<f64>,
    last_rows: Vec<f32>,
    read_cps: f64,
    write_cps: Option<f64>,
    rep8: Option<f32>,
    nan_streak: u32,
    /// `MINAGI_PROFILE=1` prints where each step's time goes.
    profile: bool,
    warned_no_eval: bool,
    brakes: Option<BrakeReport>,

    survival_steps: u64,
    grow_every_steps: u64,
    next_eval: Instant,
    next_ckpt: Instant,
}

/// Run a training session to the end. `gpu_budget_gb` is the memory the engine may use (for the memory brake).
pub fn run(spec: &RunSpec, ctl: &Receiver<Ctl>, sink: &dyn EventSink, gpu_budget_gb: f64) -> Outcome {
    let stage = |s: Stage, detail: &str| {
        sink.emit(EngineEvent::Stage(StageInfo { stage: s, detail: Some(detail.to_string()), progress: None }))
    };
    let result = Trainer::start(spec, ctl, sink, gpu_budget_gb).and_then(|mut t| t.run_loop());
    match result {
        Ok(outcome) => outcome,
        Err(EngineError::Cancelled) => {
            stage(Stage::Finished, "Stopped");
            Outcome::Stopped
        }
        Err(e) => {
            let suggest = (spec.model.d_model > Preset::Tiny.model().d_model).then_some(Preset::Tiny);
            let error = e.to_app(suggest);
            stage(Stage::Failed, &failure_text(&error));
            Outcome::Failed { error }
        }
    }
}

fn failure_text(e: &AppError) -> String {
    match e {
        AppError::OutOfMemory { .. } => "Not enough memory".into(),
        AppError::DiskFull { .. } => "Not enough disk space".into(),
        AppError::DatasetNotReady => "There is no text to read".into(),
        other => other.to_string(),
    }
}

fn warn(sink: &dyn EventSink, code: &str, message: impl Into<String>, hint: Option<&str>) {
    sink.emit(EngineEvent::Warn(Diagnostic {
        code: code.into(),
        message: message.into(),
        hint: hint.map(String::from),
    }));
}

impl<'a> Trainer<'a> {
    fn stage(&self, s: Stage, detail: &str) {
        self.sink.emit(EngineEvent::Stage(StageInfo { stage: s, detail: Some(detail.to_string()), progress: None }));
    }

    // ---- setup -----------------------------------------------------------------------------------------------

    fn start(spec: &'a RunSpec, rx: &'a Receiver<Ctl>, sink: &'a dyn EventSink, gpu_budget_gb: f64) -> Result<Self> {
        let stage = |s: Stage, d: &str| {
            sink.emit(EngineEvent::Stage(StageInfo { stage: s, detail: Some(d.to_string()), progress: None }))
        };
        let ctl = Control::new(rx);
        let train = &spec.train;
        spec.model.validate().map_err(EngineError::Config)?;
        train.validate(&spec.model).map_err(EngineError::Config)?;
        stage(Stage::PreparingData, "Reading the list of text files");
        let dev = backend::device(spec.backend)?;
        let metal = spec.backend == minagi_types::BackendKind::Metal;
        let opts = ModelOpts::from_train(train, kernel_for(train, metal));
        let paths = RunPaths::new(&spec.run_dir);
        std::fs::create_dir_all(paths.checkpoints())?;

        // the model: a fresh one, or a saved one to continue
        let (mut model, opt, saved) = match &spec.resume_from {
            Some(dir) => {
                stage(Stage::LoadingCheckpoint, "Loading your saved progress");
                let ck = read_checkpoint(dir)?;
                load_model(&dev, &ck, train, opts, &paths)?
            }
            None => {
                stage(Stage::CreatingModel, "Setting up the model with random starting values");
                let model = backend::pool(|| create_model(&dev, &spec.model, train, opts, &paths, spec.seed))?;
                let hyper = AdamHyper { beta1: train.beta1, beta2: train.beta2, eps: 1e-8 };
                (model, AdamW::new(hyper, train.weight_decay), Saved::default())
            }
        };
        let preset = preset_of(&model.cfg);
        let chunk = train.chunk;

        // pool settings that belong to how the model trains, not to its weights
        let survival_steps = chars_to_steps(u64::from(train.prune.survival_chars), chunk);
        model.pool.book.trial = survival_steps as f64;
        model.pool.book.now = saved.step as f64;
        model.pool.book.dying_at = train.prune.dying_at;
        model.pool.book.explore_bias = train.explore_bias;
        model.pool.book.explore_steps = f64::from(train.explore_steps);

        let lanes = spec.lanes.as_deref();
        let stream = TrainStream::open(
            &spec.data_root,
            lanes,
            StreamConfig::new(chunk as usize, train.passage as usize, train.shuffle_seed),
            saved.chars,
        )?;
        let evaluator = match EvalSet::open(&spec.data_root, lanes) {
            Ok(set) => Some(Evaluator::open(set, chunk as usize, model.cfg.block as usize)),
            Err(TextError::NoText(_)) | Err(TextError::Io { .. }) => None,
            Err(e) => return Err(e.into()),
        };
        let prompts: Vec<(String, String)> = minagi_data::suggest_prompts(&spec.data_root)
            .into_iter()
            .filter(|(lane, _)| lanes.is_none_or(|l| l.contains(lane)))
            .collect();

        let ramp = ContextRamp::new(&RampSettings::from_config(train, model.cfg.block), saved.context_now);
        let plast = saved.plasticity.as_ref().map(Plasticity::restore).unwrap_or_default();
        let now = Instant::now();
        let t = Trainer {
            spec,
            sink,
            ctl,
            model,
            opt,
            plast,
            ramp,
            stream,
            evaluator,
            prompts,
            paths,
            preset,
            gpu_budget_gb,
            rng: HostRng::fork(spec.seed, 2),
            grow_rng: HostRng::fork(spec.seed, 3),
            step: saved.step,
            chars: saved.chars,
            nats: saved.nats,
            chunk_in_visit: 0,
            recent: VecDeque::with_capacity(RECENT),
            last_heldout: saved.val,
            last_gap: None,
            last_rows: Vec::new(),
            read_cps: 0.0,
            write_cps: None,
            rep8: None,
            nan_streak: 0,
            profile: std::env::var("MINAGI_PROFILE").is_ok_and(|v| v == "1"),
            warned_no_eval: false,
            brakes: None,
            survival_steps,
            grow_every_steps: chars_to_steps(u64::from(train.growth.every_chars), chunk).max(1),
            next_eval: now + Duration::from_secs_f64(train.sample_every_min * 60.0),
            next_ckpt: now + Duration::from_secs_f64(train.save_every_min * 60.0),
        };
        t.sink.emit(EngineEvent::Memory(t.memory_report()));
        Ok(t)
    }

    // ---- the loop --------------------------------------------------------------------------------------------

    fn run_loop(&mut self) -> Result<Outcome> {
        self.stage(Stage::Reading, READING);
        loop {
            self.ctl.poll();
            if let Some(save) = self.ctl.stop.get() {
                return self.finish(save);
            }
            if self.ctl.paused.get() {
                self.wait_while_paused();
                continue;
            }
            let began = Instant::now();
            let r = backend::pool(|| self.train_step())?;
            self.after_step(&r, began.elapsed());
            if self.nan_streak >= MAX_NAN_STREAK {
                self.stage(Stage::Failed, "The training score became invalid");
                return Ok(Outcome::Failed { error: AppError::Engine(DIVERGED.into()) });
            }
            self.scheduled()?;
        }
    }

    fn finish(&mut self, save: bool) -> Result<Outcome> {
        self.stage(Stage::Stopping, if save { "Saving your progress before stopping" } else { "Stopping" });
        if save {
            self.checkpoint(CheckpointKind::Stop);
        }
        self.stage(Stage::Finished, "Stopped");
        Ok(Outcome::Stopped)
    }

    /// Hold still until resumed or stopped; the schedule's clocks do not run while paused.
    fn wait_while_paused(&mut self) {
        self.stage(Stage::Paused, "Paused. Nothing is being read.");
        let began = Instant::now();
        while self.ctl.paused.get() && self.ctl.stop.get().is_none() {
            match self.ctl.rx.recv_timeout(Duration::from_millis(50)) {
                Ok(m) => self.ctl.apply(m),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => self.ctl.stop.set(Some(false)),
            }
        }
        let away = began.elapsed();
        self.next_eval += away;
        self.next_ckpt += away;
        if self.ctl.stop.get().is_none() {
            self.stage(Stage::Reading, READING);
        }
    }

    // ---- one step ---------------------------------------------------------------------------------------------

    fn lr_scale(&self) -> f64 {
        self.plast.factor()
    }

    fn train_step(&mut self) -> Result<StepReport> {
        let train = &self.spec.train;
        let batch = self.stream.next_batch(self.ramp.context() as usize)?;
        if batch.first_in_visit {
            self.chunk_in_visit = 0;
        }
        let (x, y): (Vec<u32>, Vec<u32>) =
            (batch.x.iter().map(|&t| t as u32).collect(), batch.y.iter().map(|&t| t as u32).collect());
        let n_steps = sample_depth(&self.model.cfg, true, &mut self.rng);
        self.model.pool.book.now = self.step as f64;
        let settings = StepSettings {
            lr: train.lr,
            trunk_lr_mult: train.trunk_lr_mult,
            clip: train.clip,
            pool_aux: train.pool_aux,
            row_ckpt: train.row_checkpointing,
        };
        let scale = self.lr_scale();
        let out = train_on_window(&mut self.model, &mut self.opt, &x, &y, n_steps, scale, &settings)?;
        Ok(StepReport {
            timing: out.timing,
            loss: out.loss,
            grad_norm: out.grad_norm,
            rows: out.rows,
            row_mass: out.row_mass,
            new_tokens: batch.new_tokens,
            skipped: out.skipped,
        })
    }

    fn after_step(&mut self, r: &StepReport, elapsed: Duration) {
        let secs = elapsed.as_secs_f64().max(1e-6);
        if r.skipped {
            self.nan_streak += 1;
            if self.nan_streak == 1 || self.nan_streak.is_multiple_of(10) {
                warn(
                    self.sink,
                    "loss_not_finite",
                    "The training score became invalid (NaN), so this step was skipped.",
                    Some("If it keeps happening, resume from the last saved progress with a lower learning rate."),
                );
            }
        } else {
            self.nan_streak = 0;
            self.step += 1;
            self.chars += r.new_tokens as u64;
            self.nats += r.loss * r.new_tokens as f64;
            self.model.pool.book.now = self.step as f64;
            self.plast.tick();
            self.ramp.record(self.chunk_in_visit, r.loss);
            self.chunk_in_visit += 1;
            if self.recent.len() == RECENT {
                self.recent.pop_front();
            }
            self.recent.push_back(r.loss);
            if !r.row_mass.is_empty() {
                self.last_rows = r.row_mass.clone();
            }
            if let Some(to) = self.ramp.after_step(self.step) {
                self.sink.emit(EngineEvent::ContextGrow {
                    from: self.ramp.context().min(to.saturating_sub(self.spec.train.context_step)),
                    to,
                });
            }
        }
        if self.profile && self.step.is_multiple_of(20) {
            let t = r.timing;
            eprintln!(
                "[profile] step {} total {:.0} ms: forward {:.0}, backward {:.0}, readback {:.0}, update {:.0}",
                self.step,
                secs * 1000.0,
                t.forward * 1000.0,
                t.backward * 1000.0,
                t.readback * 1000.0,
                t.update * 1000.0
            );
        }
        let cps = r.new_tokens as f64 / secs;
        self.read_cps = if self.read_cps == 0.0 { cps } else { 0.9 * self.read_cps + 0.1 * cps };
        let metrics = self.step_metrics(r, secs);
        self.sink.emit(EngineEvent::Step(metrics));
    }

    fn step_metrics(&self, r: &StepReport, secs: f64) -> StepMetrics {
        let t = &self.spec.train;
        let scale = self.lr_scale();
        let max_rows = self.model.cfg.max_steps as usize;
        let hist = (self.step.is_multiple_of(20) && !r.row_mass.is_empty()).then(|| {
            let mut h = r.row_mass.clone();
            h.resize(max_rows, 0.0);
            let total: f32 = h.iter().sum::<f32>().max(1e-9);
            h.iter().map(|v| v / total).collect::<Vec<f32>>()
        });
        let now = Instant::now();
        let ms_until = |at: Instant| at.saturating_duration_since(now).as_millis() as u64;
        StepMetrics {
            step: Step(self.step),
            chars_read: Chars(self.chars),
            train_nats: r.loss,
            grad_norm: r.grad_norm,
            clip: t.clip,
            lr_scale: scale,
            lr_trunk: t.lr * t.trunk_lr_mult * scale,
            lr_experts: t.lr * scale,
            context_now: self.ramp.context(),
            read_cps: self.read_cps,
            write_cps: self.write_cps,
            avg_rows: r.rows,
            halt_hist: hist,
            rep8_pct: self.rep8,
            ctx_gain: self.ramp.gain(),
            n_experts: self.model.pool.n_experts() as u32,
            step_ms: (secs * 1000.0) as f32,
            schedule: Schedule {
                next_eval_chars: None,
                next_sample_ms: Some(Count(ms_until(self.next_eval))),
                next_checkpoint_ms: Some(Count(ms_until(self.next_ckpt))),
                next_growth_chars: Some(Chars(
                    (self.step / self.grow_every_steps + 1) * self.grow_every_steps * u64::from(t.chunk),
                )),
            },
        }
    }

    // ---- scheduled work --------------------------------------------------------------------------------------

    fn scheduled(&mut self) -> Result<()> {
        let now = Instant::now();
        let due = now >= self.next_eval;
        let (force_eval, force_sample) = (self.ctl.eval_now.replace(false), self.ctl.sample_now.replace(false));
        if due || force_eval || force_sample {
            let (do_eval, do_sample) = (due || force_eval, due || force_sample);
            if do_eval {
                self.evaluate()?;
            }
            if do_sample && self.ctl.stop.get().is_none() {
                self.sample()?;
            }
            if do_eval && self.ctl.stop.get().is_none() {
                self.emit_pool();
            }
            let every = self.ctl.sample_every_min.take().unwrap_or(self.spec.train.sample_every_min);
            self.next_eval = Instant::now() + Duration::from_secs_f64(every * 60.0);
            if self.ctl.stop.get().is_none() {
                self.stage(Stage::Reading, READING);
            }
        }
        if self.step > 0 && self.step.is_multiple_of(self.grow_every_steps) && self.ctl.stop.get().is_none() {
            // growth is checked once per multiple of the cadence, even if several steps pass in between
            self.growth_check()?;
        }
        let force_ckpt = self.ctl.ckpt_now.replace(false);
        if (Instant::now() >= self.next_ckpt || force_ckpt) && self.ctl.stop.get().is_none() {
            let every = self.ctl.ckpt_every_min.take().unwrap_or(self.spec.train.save_every_min);
            self.stage(Stage::Checkpointing, "Saving your progress");
            self.checkpoint(if force_ckpt { CheckpointKind::Manual } else { CheckpointKind::Auto });
            self.next_ckpt = Instant::now() + Duration::from_secs_f64(every * 60.0);
            self.stage(Stage::Reading, READING);
        }
        if self.step.is_multiple_of(400) {
            self.sink.emit(EngineEvent::Memory(self.memory_report()));
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<()> {
        let Some(ev) = &self.evaluator else {
            if !self.warned_no_eval {
                self.warned_no_eval = true;
                warn(
                    self.sink,
                    "no_heldout",
                    "There is no held-out text, so the model cannot be scored on text it has not seen.",
                    Some("Prepare the dataset again with some text held back for testing."),
                );
            }
            return Ok(());
        };
        self.stage(Stage::Evaluating, "Testing on text it has never trained on");
        let ctl = &self.ctl;
        let stop = || ctl.stop_requested();
        let outcome = match ev.with_context(self.ramp.context() as usize).run(
            &mut self.model,
            self.spec.train.eval_chars as usize,
            &stop,
        ) {
            Ok(o) => o,
            Err(EngineError::Cancelled) => return Ok(()),
            Err(e) => return Err(e),
        };
        let train_ema =
            if self.recent.is_empty() { f64::NAN } else { self.recent.iter().sum::<f64>() / self.recent.len() as f64 };
        if outcome.overall_nats.is_finite() {
            self.last_heldout = Some(outcome.overall_nats);
            if train_ema.is_finite() {
                self.last_gap = Some(outcome.overall_nats - train_ema);
            }
        }
        self.sink.emit(EngineEvent::Eval(EvalResult {
            step: Step(self.step),
            chars: Chars(self.chars),
            overall_nats: outcome.overall_nats,
            overall_se: outcome.overall_se,
            train_nats_ema: train_ema,
            domains: outcome.domains,
        }));
        // the held-out score steers the learning rate
        let obs = self.plast.observe(outcome.overall_nats, outcome.overall_se);
        self.sink.emit(EngineEvent::Plasticity(obs.to_event()));
        Ok(())
    }

    fn sample(&mut self) -> Result<()> {
        if self.prompts.is_empty() {
            return Ok(());
        }
        self.stage(Stage::Sampling, "Asking the model to write something");
        let tok = ByteTokenizer::new();
        let decode_adapted = Decode::adapted(&self.spec.train.decoding);
        let started = Instant::now();
        let mut written = 0usize;
        let mut items = Vec::new();
        let (mut raw_rep, mut n_rep) = (0f32, 0f32);
        for (domain, prompt) in self.prompts.clone() {
            if self.ctl.stop_requested() {
                return Ok(());
            }
            let ids: Vec<u32> = tok.encode_str(&prompt).iter().map(|&t| u32::from(t)).collect();
            let mut texts = Vec::new();
            for d in [Decode::RAW, decode_adapted] {
                let toks = backend::pool(|| Writer::new(&self.model, &ids).write(&mut self.model, SAMPLE_CHARS, &d))?;
                written += toks.len();
                let text = tok.decode(&toks.iter().map(|&t| t as u16).collect::<Vec<u16>>());
                texts.push(text);
            }
            let rep = |s: &str| repeat_rate(s.as_bytes(), 8);
            let (raw, adapted) = (texts.remove(0), texts.remove(0));
            let (raw_r, ad_r) = (rep(&raw), rep(&adapted));
            raw_rep += raw_r;
            n_rep += 1.0;
            items.push(SampleItem {
                domain,
                prompt,
                raw_rep8: Some(raw_r),
                adapted_rep8: Some(ad_r),
                raw,
                adapted,
                note: None,
            });
        }
        self.write_cps = Some(written as f64 / started.elapsed().as_secs_f64().max(1e-6));
        if n_rep > 0.0 {
            self.rep8 = Some(100.0 * raw_rep / n_rep);
        }
        self.sink.emit(EngineEvent::Samples(SampleRound { step: Step(self.step), chars: Chars(self.chars), items }));
        Ok(())
    }

    // ---- growing and pruning the pool ------------------------------------------------------------------------

    fn growth_inputs_brakes(&self) -> crate::optim::brakes::GrowthDecision {
        let pool = &self.model.pool;
        let g = &self.spec.train.growth;
        let dying = pool.book.dying();
        let (_, peak) = backend::process_memory_gb();
        let mem_frac = if self.gpu_budget_gb > 0.0 { (peak / self.gpu_budget_gb).clamp(0.0, 1.0) } else { 0.0 };
        decide(
            g,
            &GrowthInputs {
                model_step: self.step as f64,
                max_experts: self.model.cfg.pool_max,
                dying: &dying,
                dying_at: pool.book.dying_at,
                born: &pool.book.born,
                trial: pool.book.trial,
                mem_frac,
                disk_bytes_now: pool.disk_bytes(0),
                disk_bytes_after: pool.disk_bytes(g.k as usize),
                last_gap: self.last_gap,
            },
        )
    }

    fn growth_check(&mut self) -> Result<()> {
        self.stage(Stage::GrowPrune, "Checking whether the expert pool should grow or shrink");
        let step = self.step as f64;
        // prune first: delete experts nothing has addressed for a whole survival window
        let before = self.model.pool.book.uid.clone();
        if self.model.prune_pool(step, self.survival_steps as f64, 0, &mut self.opt)?.is_some() {
            let after: std::collections::HashSet<u64> = self.model.pool.book.uid.iter().copied().collect();
            for uid in before.into_iter().filter(|u| !after.contains(u)) {
                self.sink.emit(EngineEvent::Growth(GrowthEvent::Pruned {
                    uid: uid as u32,
                    reason: "Nothing asked for this expert for a very long time.".into(),
                }));
            }
        }
        let decision = self.growth_inputs_brakes();
        self.brakes = Some(decision.brakes.clone());
        if decision.grow > 0 {
            let seed_from = self
                .model
                .pool
                .book
                .use_
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);
            let first_new = self.model.pool.n_experts();
            let birth = self.spec.train.growth.birth_gate as f32;
            let parents_by_pos = self.model.pool.book.uid.clone();
            let lineage = self.model.grow_pool(
                decision.grow as usize,
                seed_from,
                step,
                birth,
                &mut self.opt,
                &mut self.grow_rng,
            )?;
            for (i, lin) in lineage.iter().enumerate() {
                let parents = lin
                    .as_ref()
                    .map(|l| l.parents.iter().filter_map(|&p| parents_by_pos.get(p).map(|&u| u as u32)).collect())
                    .unwrap_or_default();
                self.sink.emit(EngineEvent::Growth(GrowthEvent::Born {
                    uid: self.model.pool.book.uid[first_new + i] as u32,
                    parents,
                }));
            }
        } else {
            self.sink.emit(EngineEvent::Growth(GrowthEvent::Blocked { brakes: decision.brakes }));
        }
        self.emit_pool();
        self.stage(Stage::Reading, READING);
        Ok(())
    }

    fn emit_pool(&mut self) {
        let snap = self.pool_snapshot();
        self.sink.emit(EngineEvent::Pool(snap));
    }

    fn pool_snapshot(&mut self) -> PoolSnapshot {
        let brakes = match &self.brakes {
            Some(b) => b.clone(),
            None => {
                let d = self.growth_inputs_brakes();
                self.brakes = Some(d.brakes.clone());
                d.brakes
            }
        };
        let book = &self.model.pool.book;
        let gate = self.model.pool.gate_values().unwrap_or_default();
        let shares = book.usage_shares();
        let dying = book.dying();
        let chunk = u64::from(self.spec.train.chunk);
        let resident = book
            .slots
            .iter()
            .enumerate()
            .filter(|(_, e)| **e >= 0)
            .map(|(slot, &e)| {
                let e = e as usize;
                ExpertInfo {
                    uid: book.uid[e] as u32,
                    slot: Some(slot as u8),
                    gate: gate.get(e).copied().unwrap_or(1.0),
                    use_share: shares[e] as f32,
                    admits: book.admits[e] as u32,
                    age_chars: Chars((self.step.saturating_sub(book.born[e] as u64)) * chunk),
                    on_trial: book.trial > 0.0 && (self.step as f64 - book.born[e]) < book.trial,
                    staleness: dying[e].clamp(0.0, 1.0) as f32,
                    dying: dying[e] >= book.dying_at,
                }
            })
            .collect();
        let mut sorted = shares;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top = sorted.first().copied().unwrap_or(0.0).max(1e-12);
        let usage = sorted.iter().map(|s| (s / top * 255.0).round().clamp(0.0, 255.0) as u8).collect();
        let mut hist = self.last_rows.clone();
        hist.resize(self.model.cfg.max_steps as usize, 0.0);
        let total: f32 = hist.iter().sum::<f32>().max(1e-9);
        PoolSnapshot {
            step: Step(self.step),
            chars: Chars(self.chars),
            n_experts: book.n as u32,
            resident,
            usage,
            brakes,
            halting_hist: hist.iter().map(|v| v / total).collect(),
        }
    }

    // ---- checkpoints and memory ------------------------------------------------------------------------------

    fn saved_state(&self) -> Saved {
        Saved {
            step: self.step,
            chars: self.chars,
            nats: self.nats,
            val: self.last_heldout.filter(|v| v.is_finite()),
            plasticity: Some(self.plast.state()),
            context_now: Some(self.ramp.context()),
        }
    }

    /// Save a checkpoint; a failure is a warning, never the end of the run.
    fn checkpoint(&mut self, kind: CheckpointKind) -> Option<CheckpointMeta> {
        let rel = RunPaths::checkpoint_rel(self.step);
        let dest = self.paths.run_dir.join(&rel);
        let saved = self.saved_state();
        let result = (|| -> Result<_> {
            if dest.exists() {
                std::fs::remove_dir_all(&dest)?; // a second checkpoint at the same step replaces the first
            }
            write_checkpoint(&mut self.model, &self.opt, &saved, Some(self.preset), &dest)
        })();
        match result {
            Ok(report) => {
                let meta = CheckpointMeta {
                    step: Step(self.step),
                    chars: Chars(self.chars),
                    kind,
                    path: rel,
                    bytes: Count(report.total_bytes),
                    heldout_nats: saved.val,
                    n_experts: report.n_experts as u32,
                    engine_format: crate::store::manifest::RS_FORMAT,
                };
                self.sink.emit(EngineEvent::CheckpointSaved(meta.clone()));
                Some(meta)
            }
            Err(e) => {
                let app = e.to_app(None);
                let (message, hint) = match &app {
                    AppError::Io(m) if m.contains("disk is full") => {
                        ("Could not save: the disk is full.".to_string(), Some("Free some space, or delete old runs."))
                    }
                    _ => (format!("Could not save a checkpoint: {e}"), None),
                };
                warn(self.sink, "checkpoint_failed", message, hint);
                None
            }
        }
    }

    fn memory_report(&self) -> MemoryReport {
        let (resident, _) = backend::process_memory_gb();
        let pool = &self.model.pool;
        let rep = pool.store().report();
        let per_expert = pool.params_per_expert() as f64 * 4.0;
        MemoryReport {
            resident_gb: resident,
            ram_gb: rep.ram_held as f64 * per_expert / 1e9,
            disk_gb: pool.store().disk_bytes().map(|b| b as f64).unwrap_or_else(|| pool.disk_bytes(0) as f64) / 1e9,
            gpu_budget_gb: self.gpu_budget_gb,
        }
    }
}
