//! Starting, controlling and ending training sessions.
//!
//! A session is two threads: the engine (blocking, owns the model) and the recorder (owns telemetry). They talk over
//! an unbounded event channel, so the engine never waits on the database or the UI.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use minagi_types::{AppError, AppResult, Ctl, EngineEvent, Outcome, RunSpec, RunState, RunSummary};

use minagi_store::NewEvent;

use crate::power::PowerGuard;
use crate::recorder::{self, RecorderCtx};
use crate::state::{AppState, SessionFlags, SessionHandle};

/// What the engine reads for a run: the folder, and which lanes of it. A run created without a dataset reads a
/// small built-in demo folder.
pub fn data_for(state: &AppState, run: &RunSummary) -> AppResult<(std::path::PathBuf, Option<Vec<String>>)> {
    match run.dataset_id {
        Some(id) => {
            let detail = state.store.get_dataset(id)?;
            if detail.summary.status != minagi_types::DatasetStatus::Ready {
                return Err(AppError::DatasetNotReady);
            }
            let root = crate::datasets::dataset_root_path(state, id)?;
            if !root.join("train").is_dir() {
                return Err(AppError::NotFound("the text this run was set up with (its folder is missing)".into()));
            }
            Ok((root, crate::datasets::enabled_lanes(&detail)))
        }
        None => {
            let root = state.paths.datasets().join("demo");
            let _ = std::fs::create_dir_all(root.join("train").join("stories"));
            let _ = std::fs::create_dir_all(root.join("val").join("stories"));
            Ok((root, None))
        }
    }
}

pub fn start_session(state: &AppState, run_id: i64) -> AppResult<RunSummary> {
    let mut slot = state.session.lock().unwrap();
    if slot.is_some() {
        return Err(AppError::RunActive);
    }
    let run = state.store.get_run(run_id)?;
    if run.status.is_active() {
        return Err(AppError::RunActive);
    }
    let (model, train) = state.store.run_config(run_id)?;
    model.validate().map_err(AppError::Invalid)?;
    train.validate(&model).map_err(AppError::Invalid)?;

    let run_dir = state.paths.resolve(&state.store.run_dir_rel(run_id)?);
    std::fs::create_dir_all(&run_dir)?;

    // Resume from the latest checkpoint if there is one, discarding telemetry recorded after it so curves stay honest.
    let mut resume_from = None;
    let mut start_active_ms = 0;
    if let Some((step, _chars, rel)) = state.store.latest_checkpoint(run_id)? {
        let dir = run_dir.join(&rel);
        if dir.join("COMPLETE").exists() {
            state.store.truncate_after(run_id, step.0)?;
            start_active_ms = state.store.active_ms_at(run_id, step.0)?;
            state.store.record_event(
                run_id,
                NewEvent::new(step.0, 0, "resumed").payload(Some(serde_json::json!({ "fromStep": step.0 }))),
            );
            resume_from = Some(dir);
        }
    }

    let (data_root, lanes) = data_for(state, &run)?;
    let spec = RunSpec {
        run_dir,
        data_root,
        lanes,
        model,
        train,
        seed: run_id as u64,
        resume_from,
        backend: state.hardware().selected,
    };

    let (ev_tx, ev_rx) = flume::unbounded::<EngineEvent>();
    let (ctl_tx, ctl_rx) = flume::unbounded::<Ctl>();
    let flags = Arc::new(SessionFlags::default());

    let started = state.store.set_run_state(run_id, RunState::Preparing, None)?;
    state.live.reset_snapshot(started.clone());
    state.live.send(minagi_types::LiveMsg::State { run_id, state: RunState::Preparing, reason: None });

    // Engine thread.
    let mut engine = state.factory.new_engine();
    let engine_tx = ev_tx.clone();
    std::thread::Builder::new()
        .name("minagi-engine".into())
        .spawn(move || {
            let outcome =
                catch_unwind(AssertUnwindSafe(|| engine.run(spec, ctl_rx, &engine_tx))).unwrap_or_else(|_| {
                    Outcome::Failed { error: AppError::Engine("The training engine stopped unexpectedly.".into()) }
                });
            let _ = engine_tx.send(EngineEvent::Finished(outcome));
        })
        .map_err(|e| AppError::Engine(e.to_string()))?;
    drop(ev_tx);

    // Recorder thread.
    let ctx = RecorderCtx {
        run_id,
        store: state.store.clone(),
        hub: state.live.clone(),
        ctl: ctl_tx.clone(),
        flags: flags.clone(),
        session_slot: state.session.clone(),
        goal: run.goal.clone(),
        chars_total: run.chars_total.map(|c| c.0),
        start_active_ms,
        eval_points: state.store.eval_points(run_id).unwrap_or_default(),
        initial_state: RunState::Preparing,
        notify: state.notifier.get().cloned(),
        run_name: run.name.clone(),
    };
    std::thread::Builder::new()
        .name("minagi-recorder".into())
        .spawn(move || recorder::run(ctx, ev_rx))
        .map_err(|e| AppError::Engine(e.to_string()))?;

    *slot = Some(SessionHandle { run_id, _power: PowerGuard::acquire(), ctl: ctl_tx, flags });
    Ok(started)
}

/// Send a control message to the active session.
pub fn send_ctl(state: &AppState, ctl: Ctl) -> AppResult<()> {
    let slot = state.session.lock().unwrap();
    let session = slot.as_ref().ok_or(AppError::NoActiveRun)?;
    if matches!(ctl, Ctl::Stop { .. }) {
        session.flags.user_stop.store(true, Ordering::SeqCst);
    }
    session.ctl.send(ctl).map_err(|_| AppError::NoActiveRun)
}

pub fn active_run_id(state: &AppState) -> Option<i64> {
    state.session.lock().unwrap().as_ref().map(|s| s.run_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    use minagi_mock::{MockFactory, Scenario};
    use minagi_store::{NewRun, RunOrigin, Store};
    use minagi_types::{CheckpointKind, Goal, Preset, SeriesRequest, XAxis};

    use crate::state::AppPaths;

    fn app(dir: &std::path::Path, speed: f64) -> AppState {
        let paths = AppPaths::new(dir.to_path_buf());
        paths.ensure().unwrap();
        let store = Store::open(&paths.db()).unwrap();
        AppState::new(paths, store, Arc::new(MockFactory::new(speed, Scenario::Normal)))
    }

    fn new_run(state: &AppState, goal: Goal) -> i64 {
        let mut train = Preset::Tiny.train();
        train.sample_every_min = 0.05;
        train.save_every_min = 0.1;
        state
            .store
            .create_run(NewRun {
                name: "test".into(),
                preset: Preset::Tiny,
                dataset_id: None,
                dataset_snapshot: serde_json::json!({}),
                model: Preset::Tiny.model(),
                train,
                goal: Some(goal),
                engine_version: "test".into(),
                app_version: "0".into(),
                origin: RunOrigin::Trained,
                parent_run_id: None,
                backend: None,
                chars_total: None,
            })
            .unwrap()
            .id
    }

    fn wait_for(state: &AppState, run_id: i64, f: impl Fn(RunState) -> bool) -> RunSummary {
        let t0 = Instant::now();
        loop {
            let r = state.store.get_run(run_id).unwrap();
            if f(r.status) && active_run_id(state).is_none() {
                return r;
            }
            assert!(t0.elapsed() < Duration::from_secs(60), "timed out waiting; status = {:?}", r.status);
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn series(state: &AppState, run_id: i64, key: &str) -> minagi_types::SeriesData {
        state
            .store
            .get_series(&SeriesRequest {
                run_id,
                keys: vec![key.into()],
                x: XAxis::Chars,
                from: None,
                to: None,
                max_points: 2000,
            })
            .unwrap()
            .remove(0)
    }

    #[test]
    fn a_run_trains_to_its_goal_and_everything_is_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let state = app(dir.path(), 40.0);
        let id = new_run(&state, Goal::Chars { value: minagi_types::Chars(1_000_000) });
        start_session(&state, id).unwrap();
        assert!(matches!(start_session(&state, id), Err(AppError::RunActive)), "only one run at a time");

        let run = wait_for(&state, id, |s| s == RunState::Completed);
        state.store.flush();
        assert!(run.chars_read.0 >= 1_000_000, "read {}", run.chars_read.0);
        assert!(run.ended_at.is_some() && run.started_at.is_some());
        assert!(run.verdict.is_some() && run.best_heldout_nats.is_some() && run.last_train_nats.is_some());
        assert!(run.active_ms.0 > 0);

        assert!(series(&state, id, "train.nats").x.len() >= 2, "ticks must be persisted about once a second");
        assert!(state.store.eval_points(id).unwrap().len() >= 5);
        assert!(state.store.get_samples(id, None).unwrap().is_some());
        assert!(state.store.get_pool_snapshot(id, None).unwrap().is_some());
        let ckpts = state.store.list_checkpoints(id).unwrap();
        assert!(ckpts.iter().any(|c| c.kind == CheckpointKind::Auto), "periodic checkpoints");
        assert!(ckpts.iter().any(|c| c.kind == CheckpointKind::Stop), "a final checkpoint when the goal stops the run");
        assert!(ckpts.iter().all(|c| {
            state.paths.resolve(&state.store.run_dir_rel(id).unwrap()).join(&c.path).join("COMPLETE").exists()
        }));
        assert!(!state.store.get_events(id, Some(vec!["checkpoint".into()])).unwrap().is_empty());
    }

    #[test]
    fn a_user_stop_ends_in_stopped_and_can_be_resumed() {
        let dir = tempfile::tempdir().unwrap();
        let state = app(dir.path(), 20.0);
        let id = new_run(&state, Goal::UntilStopped);
        start_session(&state, id).unwrap();
        // let it get going, then pause and resume it, then stop
        std::thread::sleep(Duration::from_millis(900));
        send_ctl(&state, Ctl::Pause).unwrap();
        wait_state(&state, id, RunState::Paused);
        send_ctl(&state, Ctl::Resume).unwrap();
        wait_state(&state, id, RunState::Running);
        send_ctl(&state, Ctl::Stop { save: true }).unwrap();
        let first = wait_for(&state, id, |s| s == RunState::Stopped);
        assert!(first.chars_read.0 > 0);
        assert!(state.store.latest_checkpoint(id).unwrap().is_some());
        let kinds: Vec<String> = state.store.get_events(id, None).unwrap().into_iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&"pause".to_string()) && kinds.contains(&"resume".to_string()), "{kinds:?}");

        // Starting again resumes from the checkpoint and records that it did.
        start_session(&state, id).unwrap();
        wait_state(&state, id, RunState::Running);
        send_ctl(&state, Ctl::Stop { save: false }).unwrap();
        let second = wait_for(&state, id, |s| s == RunState::Stopped);
        state.store.flush();
        let kinds: Vec<String> = state.store.get_events(id, None).unwrap().into_iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&"resumed".to_string()), "{kinds:?}");
        assert!(second.step.0 >= first.step.0.saturating_sub(1), "a resumed run continues from its checkpoint");
    }

    fn wait_state(state: &AppState, run_id: i64, want: RunState) {
        let t0 = Instant::now();
        while state.store.get_run(run_id).unwrap().status != want {
            assert!(t0.elapsed() < Duration::from_secs(30), "never reached {want:?}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn controlling_with_no_session_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let state = app(dir.path(), 1.0);
        assert!(matches!(send_ctl(&state, Ctl::Pause), Err(AppError::NoActiveRun)));
    }

    #[test]
    fn engine_failures_become_a_failed_run_with_an_actionable_error() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::new(dir.path().to_path_buf());
        paths.ensure().unwrap();
        let store = Store::open(&paths.db()).unwrap();
        let state = AppState::new(paths, store, Arc::new(MockFactory::new(100.0, Scenario::OomAtStart)));
        let id = new_run(&state, Goal::UntilStopped);
        start_session(&state, id).unwrap();
        let run = wait_for(&state, id, |s| s == RunState::Failed);
        assert!(run.error.as_deref().unwrap_or("").to_lowercase().contains("memory"), "{:?}", run.error);
        // The slot is free again, so the user can try a smaller model right away.
        assert!(active_run_id(&state).is_none());
    }
}
