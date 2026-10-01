//! The simulated training engine.
//!
//! Time is simulated: each step advances a virtual clock by `chunk / read_cps`, and the real sleep between steps is
//! that duration divided by `speed`. Scheduled work (evaluation, samples, checkpoints) follows the virtual clock, so a
//! 10-minute schedule at 20x speed fires every 30 real seconds.

use std::path::Path;
use std::time::Duration;

use flume::RecvTimeoutError;
use minagi_types::{
    AppError, Brake, BrakeReport, Chars, CheckpointKind, CheckpointMeta, Count, Ctl, Diagnostic, DomainScore, Engine,
    EngineEvent, EvalResult, EventSink, ExpertInfo, GrowthEvent, MemoryReport, Outcome, PlasticityEvent, PoolSnapshot,
    Preset, RunSpec, SampleItem, SampleRound, Schedule, Stage, StageInfo, Step, StepMetrics,
};

use crate::sim::{Curve, Rng, Scenario, default_prompt, rep8, write_text};

const READING: &str = "Reading text and learning from its mistakes";

pub struct MockEngine {
    pub speed: f64,
    pub scenario: Scenario,
}

/// Subfolder names under `root/sub`, or a single default domain when there are none.
fn list_domains(root: &Path, sub: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(root.join(sub))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    if v.is_empty() {
        v.push("stories".into());
    }
    v
}

/// Normalised histogram of rows used, centred on `avg_rows`.
fn halting_hist(max_rows: u32, avg_rows: f64) -> Vec<f32> {
    let center = avg_rows.max(1.0);
    let mut hist: Vec<f32> = (0..max_rows)
        .map(|i| {
            let d = (i as f64 + 1.0 - center) / (center * 0.5 + 0.8);
            (-0.5 * d * d).exp() as f32
        })
        .collect();
    let sum: f32 = hist.iter().sum();
    hist.iter_mut().for_each(|h| *h /= sum.max(1e-6));
    hist
}

fn avg_rows_at(max_rows: u32, chars: f64) -> f64 {
    1.2 + (max_rows as f64 * 0.45 - 1.2) * (1.0 - (-chars / 2.0e6).exp())
}

struct State {
    step: u64,
    chars: u64,
    sim_ms: f64,
    n_experts: u32,
    context: u32,
    lr_scale: f64,
    next_eval_ms: f64,
    next_checkpoint_ms: f64,
    next_growth_chars: u64,
    last_heldout: Option<f64>,
    /// Exponential moving average of the training loss (what the real engine reports at evaluation time).
    train_ema: f64,
    growth_was_blocked: bool,
}

impl Engine for MockEngine {
    fn run(&mut self, spec: RunSpec, ctl: flume::Receiver<Ctl>, sink: &dyn EventSink) -> Outcome {
        let speed = self.speed;
        let stage = |s: Stage, detail: &str| {
            sink.emit(EngineEvent::Stage(StageInfo { stage: s, detail: Some(detail.to_string()), progress: None }))
        };
        let sleep = move |ms: f64| std::thread::sleep(Duration::from_secs_f64((ms / 1000.0 / speed).max(0.0)));

        stage(Stage::PreparingData, "Reading the list of text files");
        sleep(400.0);
        match self.scenario {
            Scenario::OomAtStart => {
                stage(Stage::Failed, "Not enough memory");
                return Outcome::Failed { error: AppError::OutOfMemory { suggested_preset: Some(Preset::Tiny) } };
            }
            Scenario::DiskFull => {
                stage(Stage::Failed, "Not enough disk space");
                return Outcome::Failed {
                    error: AppError::DiskFull { need_bytes: Count(2_000_000_000), free_bytes: Count(150_000_000) },
                };
            }
            _ => {}
        }
        let _ = std::fs::create_dir_all(spec.run_dir.join("checkpoints"));

        let curve = Curve::for_scenario(self.scenario);
        let mut rng = Rng::new(spec.seed.wrapping_add(17));
        let mut domains = list_domains(&spec.data_root, "val");
        if let Some(lanes) = &spec.lanes {
            domains.retain(|d| lanes.contains(d));
            if domains.is_empty() {
                domains.push("stories".into());
            }
        }
        let offsets: Vec<f64> =
            (0..domains.len()).map(|i| [-0.1, 0.45, 0.2, 0.7, 0.35, 0.55, 0.1, 0.8][i % 8]).collect();
        let (train, model) = (&spec.train, &spec.model);
        let chunk = train.chunk as u64;

        let mut st = State {
            step: 0,
            chars: 0,
            sim_ms: 0.0,
            n_experts: model.pool_experts,
            context: train.context_start,
            lr_scale: 1.0,
            next_eval_ms: train.sample_every_min * 60_000.0,
            next_checkpoint_ms: train.save_every_min * 60_000.0,
            next_growth_chars: train.growth.every_chars as u64,
            last_heldout: None,
            train_ema: curve.start,
            growth_was_blocked: false,
        };

        stage(Stage::CreatingModel, "Setting up the model with random starting values");
        sleep(500.0);
        if let Some(resume) = &spec.resume_from {
            stage(Stage::LoadingCheckpoint, "Loading your saved progress");
            let saved = std::fs::read_to_string(resume.join("mock-state.json"))
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
            if let Some(v) = saved {
                st.step = v["step"].as_u64().unwrap_or(0);
                st.chars = v["chars"].as_u64().unwrap_or(0);
                st.sim_ms = v["simMs"].as_f64().unwrap_or(0.0);
                st.n_experts = v["nExperts"].as_u64().unwrap_or(st.n_experts as u64) as u32;
                st.next_eval_ms = st.sim_ms + train.sample_every_min * 60_000.0;
                st.next_checkpoint_ms = st.sim_ms + train.save_every_min * 60_000.0;
                st.next_growth_chars = st.chars + train.growth.every_chars as u64;
            }
            sleep(400.0);
        }
        sink.emit(EngineEvent::Memory(MemoryReport {
            resident_gb: 1.2,
            ram_gb: 0.8,
            disk_gb: 0.2,
            gpu_budget_gb: 14.0,
        }));
        stage(Stage::Reading, READING);

        let base_cps = 9_000.0 * (0.8 + 0.4 * rng.f64());
        let (mut paused, mut stopping) = (false, None::<bool>);
        let (mut force_eval, mut force_sample, mut force_ckpt) = (false, false, false);

        loop {
            // ── control messages ──────────────────────────────────────────────────────────────────────────
            loop {
                let next = if paused {
                    ctl.recv_timeout(Duration::from_millis(50))
                } else {
                    ctl.try_recv().map_err(|e| match e {
                        flume::TryRecvError::Empty => RecvTimeoutError::Timeout,
                        flume::TryRecvError::Disconnected => RecvTimeoutError::Disconnected,
                    })
                };
                match next {
                    Ok(Ctl::Pause) if !paused => {
                        paused = true;
                        stage(Stage::Paused, "Paused. Nothing is being read.");
                    }
                    Ok(Ctl::Resume) if paused => {
                        paused = false;
                        stage(Stage::Reading, READING);
                    }
                    Ok(Ctl::Stop { save }) => {
                        stopping = Some(save);
                        break;
                    }
                    Ok(Ctl::CheckpointNow) => force_ckpt = true,
                    Ok(Ctl::SampleNow) => force_sample = true,
                    Ok(Ctl::EvalNow) => force_eval = true,
                    Ok(Ctl::SetSampleEveryMin { minutes }) => st.next_eval_ms = st.sim_ms + minutes * 60_000.0,
                    Ok(Ctl::SetCheckpointEveryMin { minutes }) => {
                        st.next_checkpoint_ms = st.sim_ms + minutes * 60_000.0
                    }
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) if paused => {}
                    Err(RecvTimeoutError::Timeout) => break,
                    // The app dropped its end: nobody is listening, so stop without saving.
                    Err(RecvTimeoutError::Disconnected) => {
                        stopping = Some(false);
                        break;
                    }
                }
            }
            if let Some(save) = stopping {
                stage(Stage::Stopping, if save { "Saving your progress before stopping" } else { "Stopping" });
                if save {
                    self.checkpoint(&spec, sink, &st, CheckpointKind::Stop);
                }
                stage(Stage::Finished, "Stopped");
                return Outcome::Stopped;
            }

            // ── one step ──────────────────────────────────────────────────────────────────────────────────
            let progress = st.chars as f64;
            let cps = base_cps * (1.0 + 0.04 * rng.normal()).max(0.5);
            let step_ms = chunk as f64 / cps * 1000.0;
            st.step += 1;
            st.chars += chunk;
            st.sim_ms += step_ms;

            let diverged = curve.is_nan_at(progress);
            let train_nats = if diverged {
                f64::NAN
            } else {
                curve.train(progress, offsets[0]) + 0.05 * rng.normal() * (1.5 - (progress / 3.0e6).min(1.0))
            };
            if train_nats.is_finite() {
                st.train_ema += 0.02 * (train_nats - st.train_ema);
            }
            let grad_norm = if diverged { f64::NAN } else { 0.9 + 0.5 * rng.f64() + 3.0 * (-progress / 4.0e5).exp() };
            if st.step.is_multiple_of(49) && st.context < train.context_end && progress > 5.0e5 {
                let from = st.context;
                st.context = (st.context + 16).min(train.context_end);
                sink.emit(EngineEvent::ContextGrow { from, to: st.context });
            }
            let avg_rows = avg_rows_at(model.max_steps, progress);
            sink.emit(EngineEvent::Step(StepMetrics {
                step: Step(st.step),
                chars_read: Chars(st.chars),
                train_nats,
                grad_norm,
                clip: train.clip,
                lr_scale: st.lr_scale,
                lr_trunk: train.lr * train.trunk_lr_mult * st.lr_scale,
                lr_experts: train.lr * st.lr_scale,
                context_now: st.context,
                read_cps: cps,
                write_cps: None,
                avg_rows: avg_rows as f32,
                halt_hist: st.step.is_multiple_of(20).then(|| halting_hist(model.max_steps, avg_rows)),
                rep8_pct: None,
                ctx_gain: None,
                n_experts: st.n_experts,
                step_ms: step_ms as f32,
                schedule: Schedule {
                    next_eval_chars: None,
                    next_sample_ms: Some(Count((st.next_eval_ms - st.sim_ms).max(0.0) as u64)),
                    next_checkpoint_ms: Some(Count((st.next_checkpoint_ms - st.sim_ms).max(0.0) as u64)),
                    next_growth_chars: Some(Chars(st.next_growth_chars)),
                },
            }));
            if diverged && st.step.is_multiple_of(50) {
                sink.emit(EngineEvent::Warn(Diagnostic {
                    code: "loss_not_finite".into(),
                    message: "The training loss became invalid (NaN).".into(),
                    hint: Some("Resume from the last good checkpoint with a lower learning rate.".into()),
                }));
            }
            sleep(step_ms);

            // ── scheduled work ────────────────────────────────────────────────────────────────────────────
            let due = st.sim_ms >= st.next_eval_ms;
            if due || force_eval || force_sample {
                let (do_eval, do_sample) = (due || force_eval, due || force_sample);
                (force_eval, force_sample) = (false, false);
                if do_eval {
                    st.next_eval_ms = st.sim_ms + train.sample_every_min * 60_000.0;
                    stage(Stage::Evaluating, "Testing on text it has never trained on");
                    st.last_heldout = Some(self.evaluate(sink, &curve, &mut rng, &st, &domains, &offsets));
                    self.plasticity(sink, &mut rng, &mut st);
                }
                if do_sample {
                    stage(Stage::Sampling, "Asking the model to write something");
                    self.sample(sink, &mut rng, &st, &domains, &offsets, &curve);
                }
                if do_eval {
                    stage(Stage::GrowPrune, "Checking whether the expert pool should grow");
                    self.pool(sink, &mut rng, &spec, &st);
                }
                stage(Stage::Reading, READING);
            }
            if st.chars >= st.next_growth_chars {
                st.next_growth_chars += train.growth.every_chars as u64;
                self.grow(sink, &mut rng, &mut st, model.pool_max);
            }
            if force_ckpt || st.sim_ms >= st.next_checkpoint_ms {
                force_ckpt = false;
                st.next_checkpoint_ms = st.sim_ms + train.save_every_min * 60_000.0;
                stage(Stage::Checkpointing, "Saving your progress");
                self.checkpoint(&spec, sink, &st, CheckpointKind::Auto);
                stage(Stage::Reading, READING);
            }
            if st.step.is_multiple_of(400) {
                sink.emit(EngineEvent::Memory(MemoryReport {
                    resident_gb: 1.2 + 0.0005 * (st.step % 40) as f64,
                    ram_gb: 0.8,
                    disk_gb: 0.2 + st.n_experts as f64 * 0.002,
                    gpu_budget_gb: 14.0,
                }));
            }
        }
    }
}

impl MockEngine {
    /// Emit a held-out evaluation round and return the overall score it reported.
    fn evaluate(
        &self,
        sink: &dyn EventSink,
        curve: &Curve,
        rng: &mut Rng,
        st: &State,
        domains: &[String],
        offsets: &[f64],
    ) -> f64 {
        let progress = st.chars as f64;
        let domain_scores: Vec<DomainScore> = domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let nats = if curve.is_nan_at(progress) {
                    f64::NAN
                } else {
                    curve.heldout(progress, offsets[i]) + 0.012 * rng.normal()
                };
                DomainScore {
                    domain: d.clone(),
                    nats,
                    se: 0.01 + 0.03 / (1.0 + progress / 1.0e6),
                    n_chars: Count(30_720),
                }
            })
            .collect();
        let n = domain_scores.len() as f64;
        let overall = domain_scores.iter().map(|s| s.nats).sum::<f64>() / n;
        sink.emit(EngineEvent::Eval(EvalResult {
            step: Step(st.step),
            chars: Chars(st.chars),
            overall_nats: overall,
            overall_se: domain_scores.iter().map(|s| s.se).sum::<f64>() / n / n.sqrt(),
            train_nats_ema: st.train_ema,
            domains: domain_scores,
        }));
        overall
    }

    /// The learning-rate controller: eases the rate down, and sometimes jumps it back up when progress stalls.
    fn plasticity(&self, sink: &dyn EventSink, rng: &mut Rng, st: &mut State) {
        let old = st.lr_scale;
        st.lr_scale = (st.lr_scale * if rng.f64() < 0.12 { 2.0 } else { 0.93 }).clamp(0.05, 1.0);
        let jumped = st.lr_scale > old * 1.5;
        sink.emit(EngineEvent::Plasticity(PlasticityEvent {
            rate_scale: st.lr_scale,
            evidence_t: 2.5 - (st.chars as f64 / 2.0e6).min(3.0) + 0.3 * rng.normal(),
            reason: if jumped {
                "The score stopped improving, so the learning rate was raised.".into()
            } else {
                "Improving steadily; easing the learning rate.".into()
            },
            jumped,
        }));
    }

    fn sample(
        &self,
        sink: &dyn EventSink,
        rng: &mut Rng,
        st: &State,
        domains: &[String],
        offsets: &[f64],
        curve: &Curve,
    ) {
        let items: Vec<SampleItem> = domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let nats = curve.heldout(st.chars as f64, offsets[i]);
                let raw = write_text(rng, d, nats, false, 140);
                let adapted = write_text(rng, d, nats, true, 140);
                SampleItem {
                    domain: d.clone(),
                    prompt: default_prompt(d),
                    raw_rep8: Some(rep8(&raw)),
                    adapted_rep8: Some(rep8(&adapted)),
                    raw,
                    adapted,
                    note: None,
                }
            })
            .collect();
        sink.emit(EngineEvent::Samples(SampleRound { step: Step(st.step), chars: Chars(st.chars), items }));
    }

    fn brakes(&self, rng: &mut Rng, st: &State, pool_max: u32) -> BrakeReport {
        let b = |ok: bool, good: &str, bad: &str| Brake { ok, why: (if ok { good } else { bad }).to_string() };
        let gap_ok = st.last_heldout.map(|h| h - st.train_ema < 0.4).unwrap_or(true);
        BrakeReport {
            room: b(
                st.n_experts < pool_max,
                "There is room for another expert.",
                "The expert pool is at its size limit.",
            ),
            used: b(true, "Most experts are being used.", "Too many experts sit idle."),
            earning: b(
                rng.f64() > 0.35,
                "Recent newcomers are earning their keep.",
                "3 of 8 newer experts are not yet useful.",
            ),
            fits: b(true, "Few newcomers are still on trial.", "Too many newcomers are still on trial."),
            honest: b(
                gap_ok,
                "The model does about as well on new text as on training text.",
                "It does much better on training text than on new text.",
            ),
        }
    }

    fn grow(&self, sink: &dyn EventSink, rng: &mut Rng, st: &mut State, pool_max: u32) {
        let brakes = self.brakes(rng, st, pool_max);
        if brakes.all_ok() || st.growth_was_blocked {
            st.n_experts = (st.n_experts + 1).min(pool_max);
            st.growth_was_blocked = false;
            sink.emit(EngineEvent::Growth(GrowthEvent::Born {
                uid: 1000 + st.n_experts,
                parents: (0..4).map(|i| (i * 3 + rng.range(0, 3)) as u32).collect(),
            }));
        } else {
            st.growth_was_blocked = true;
            sink.emit(EngineEvent::Growth(GrowthEvent::Blocked { brakes }));
        }
    }

    fn pool(&self, sink: &dyn EventSink, rng: &mut Rng, spec: &RunSpec, st: &State) {
        let resident = spec.train.resident.min(st.n_experts) as usize;
        let mut shares: Vec<f32> =
            (0..st.n_experts).map(|i| 1.0 / (1.0 + i as f32 * 0.35) * (0.7 + 0.6 * rng.f64() as f32)).collect();
        let total: f32 = shares.iter().sum();
        shares.iter_mut().for_each(|s| *s /= total);
        shares.sort_by(|a, b| b.total_cmp(a));
        let max = shares[0].max(1e-6);
        let usage = shares.iter().map(|s| (s / max * 255.0) as u8).collect();
        let experts = (0..resident)
            .map(|i| ExpertInfo {
                uid: i as u32 + 1,
                slot: Some(i as u8),
                gate: 0.8 + 0.4 * rng.f64() as f32,
                use_share: shares[i],
                admits: rng.range(5, 900) as u32,
                age_chars: Chars(st.chars.saturating_sub(rng.range(0, 1_000_000) as u64)),
                on_trial: i + 3 >= resident,
                staleness: if i + 3 >= resident { 0.4 + 0.5 * rng.f64() as f32 } else { 0.05 * rng.f64() as f32 },
                dying: i + 1 == resident && st.n_experts > 20,
            })
            .collect();
        sink.emit(EngineEvent::Pool(PoolSnapshot {
            step: Step(st.step),
            chars: Chars(st.chars),
            n_experts: st.n_experts,
            resident: experts,
            usage,
            brakes: self.brakes(rng, st, spec.model.pool_max),
            halting_hist: halting_hist(spec.model.max_steps, avg_rows_at(spec.model.max_steps, st.chars as f64)),
        }));
    }

    /// Write a checkpoint atomically: build in a temp directory, mark it COMPLETE, then rename into place.
    fn checkpoint(&self, spec: &RunSpec, sink: &dyn EventSink, st: &State, kind: CheckpointKind) {
        let rel = format!("checkpoints/step-{:09}", st.step);
        let dir = spec.run_dir.join(&rel);
        let tmp = spec.run_dir.join(format!("checkpoints/.tmp-step-{:09}", st.step));
        let warn = |message: &str| {
            sink.emit(EngineEvent::Warn(Diagnostic {
                code: "checkpoint_failed".into(),
                message: message.into(),
                hint: None,
            }))
        };
        let _ = std::fs::remove_dir_all(&tmp);
        if std::fs::create_dir_all(&tmp).is_err() {
            return warn("Could not save a checkpoint.");
        }
        let state =
            serde_json::json!({ "step": st.step, "chars": st.chars, "simMs": st.sim_ms, "nExperts": st.n_experts });
        let payload = serde_json::to_vec_pretty(&state).unwrap_or_default();
        if std::fs::write(tmp.join("mock-state.json"), &payload).is_err()
            || std::fs::write(tmp.join("COMPLETE"), b"ok").is_err()
        {
            return warn("Could not write the checkpoint files.");
        }
        let _ = std::fs::remove_dir_all(&dir);
        if std::fs::rename(&tmp, &dir).is_err() {
            return warn("Could not finish saving the checkpoint.");
        }
        sink.emit(EngineEvent::CheckpointSaved(CheckpointMeta {
            step: Step(st.step),
            chars: Chars(st.chars),
            kind,
            path: rel,
            bytes: Count(payload.len() as u64 + 2),
            heldout_nats: st.last_heldout.filter(|h| h.is_finite()),
            n_experts: st.n_experts,
            engine_format: 0,
        }));
    }
}
