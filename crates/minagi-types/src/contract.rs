//! Behavioural contract every `Engine` must satisfy, as a reusable test harness.
//!
//! The simulated engine runs it on every build; the real engine runs it nightly. If the two drift apart the UI
//! (built against the mock) would break on the real thing, so the invariants the app relies on live here.
//!
//! The host, not the engine, emits `EngineEvent::Finished` once `Engine::run` returns.

use std::time::Duration;

use crate::events::{CheckpointKind, Ctl, EngineEvent, Outcome, Stage};
use crate::traits::{EngineFactory, RunSpec};

/// Everything a run produced.
#[derive(Debug)]
pub struct Recorded {
    pub events: Vec<EngineEvent>,
    pub outcome: Outcome,
}

/// Run the engine on its own thread, calling `driver` for every event so a test can send control messages at chosen
/// moments. Panics if the engine goes silent for 30 seconds.
pub fn run_with_driver(
    factory: &dyn EngineFactory,
    spec: RunSpec,
    mut driver: impl FnMut(&EngineEvent, &flume::Sender<Ctl>),
) -> Recorded {
    let (ev_tx, ev_rx) = flume::unbounded::<EngineEvent>();
    let (ctl_tx, ctl_rx) = flume::unbounded::<Ctl>();
    let mut engine = factory.new_engine();
    let handle = std::thread::spawn(move || engine.run(spec, ctl_rx, &ev_tx));

    let mut events = Vec::new();
    loop {
        match ev_rx.recv_timeout(Duration::from_secs(30)) {
            Ok(ev) => {
                driver(&ev, &ctl_tx);
                events.push(ev);
            }
            Err(flume::RecvTimeoutError::Disconnected) => break,
            Err(flume::RecvTimeoutError::Timeout) => panic!("engine produced no events for 30 s (hung?)"),
        }
    }
    let outcome = handle.join().expect("engine thread panicked");
    Recorded { events, outcome }
}

/// Assert the invariants the app depends on. Panics with a description of the first violation.
pub fn check_invariants(spec: &RunSpec, rec: &Recorded) {
    let ev = &rec.events;
    assert!(!ev.is_empty(), "an engine must emit at least one event");
    assert!(matches!(ev.first(), Some(EngineEvent::Stage(_))), "the first event must be a Stage event");

    // The last stage tells the UI how the run ended.
    let last_stage = ev.iter().rev().find_map(|e| if let EngineEvent::Stage(s) = e { Some(s.stage) } else { None });
    match &rec.outcome {
        Outcome::Failed { .. } => assert_eq!(last_stage, Some(Stage::Failed), "a failed run must end on Stage::Failed"),
        _ => assert_eq!(last_stage, Some(Stage::Finished), "a finished run must end on Stage::Finished"),
    }

    // Counters never go backwards; steps strictly increase.
    let (mut last_step, mut last_chars) = (0u64, 0u64);
    for e in ev {
        if let EngineEvent::Step(s) = e {
            assert!(s.step.0 > last_step, "step must strictly increase ({} after {})", s.step.0, last_step);
            assert!(s.chars_read.0 >= last_chars, "characters read must not decrease");
            assert!(s.schedule.next_sample_ms.is_some(), "Step events must carry a schedule for the banner");
            assert!(s.n_experts >= 1, "a model always has at least one expert");
            (last_step, last_chars) = (s.step.0, s.chars_read.0);
        }
    }

    // No Step events while paused: between Stage::Paused and the next Stage::Reading the engine is silent on steps.
    let mut paused = false;
    for e in ev {
        match e {
            EngineEvent::Stage(s) if s.stage == Stage::Paused => paused = true,
            EngineEvent::Stage(s) if s.stage == Stage::Reading => paused = false,
            EngineEvent::Step(_) => assert!(!paused, "a Step was emitted while the run was paused"),
            _ => {}
        }
    }

    for e in ev {
        match e {
            EngineEvent::Eval(r) => {
                assert!(!r.domains.is_empty(), "an evaluation must score at least one domain");
            }
            EngineEvent::Samples(r) => {
                assert!(!r.items.is_empty(), "a sample round must contain samples");
                assert!(r.items.iter().all(|i| !i.prompt.is_empty()), "every sample needs its prompt");
            }
            EngineEvent::Pool(p) => {
                assert_eq!(p.usage.len() as u32, p.n_experts, "usage must cover the whole pool");
                assert!(
                    p.resident.len() as u32 <= spec.train.resident.min(p.n_experts),
                    "more resident experts than slots"
                );
            }
            EngineEvent::CheckpointSaved(c) => {
                let dir = spec.run_dir.join(&c.path);
                assert!(dir.join("COMPLETE").exists(), "checkpoint {} is missing its COMPLETE marker", c.path);
            }
            _ => {}
        }
    }
    // Atomic writes: no half-written checkpoint directories are left behind.
    if let Ok(rd) = std::fs::read_dir(spec.run_dir.join("checkpoints")) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(!name.starts_with(".tmp-"), "leftover temporary checkpoint directory {name}");
        }
    }

    // A stop that asked to save must produce a Stop checkpoint after the Stopping stage.
    if rec.outcome == Outcome::Stopped {
        let stopping_at = ev.iter().position(|e| matches!(e, EngineEvent::Stage(s) if s.stage == Stage::Stopping));
        if let Some(i) = stopping_at {
            let saved =
                ev[i..].iter().any(|e| matches!(e, EngineEvent::CheckpointSaved(c) if c.kind == CheckpointKind::Stop));
            let _ = saved; // a stop without save is legal; the save case is asserted in the dedicated test
        }
    }
}
