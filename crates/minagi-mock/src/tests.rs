//! Contract tests: the mock must satisfy the same behavioural contract the real engine will be held to.

use std::path::Path;

use minagi_types::contract::{Recorded, check_invariants, run_with_driver};
use minagi_types::{
    AppError, BackendKind, CheckpointKind, Ctl, EngineEvent, EngineFactory, EvalPoint, Outcome, Preset, RunSpec, Stage,
    Verdict, insight,
};

use crate::{MockFactory, Scenario};

fn spec(dir: &Path, resume: Option<std::path::PathBuf>) -> RunSpec {
    let mut train = Preset::Tiny.train();
    train.sample_every_min = 0.05; // an evaluation roughly every 27k characters
    train.save_every_min = 0.15;
    RunSpec {
        run_dir: dir.join("run"),
        data_root: dir.join("data"),
        lanes: None,
        model: Preset::Tiny.model(),
        train,
        seed: 1,
        resume_from: resume,
        backend: BackendKind::Cpu,
    }
}

fn factory(scenario: Scenario) -> MockFactory {
    MockFactory::new(20_000.0, scenario)
}

fn evals(rec: &Recorded) -> Vec<EvalPoint> {
    rec.events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::Eval(r) => Some(EvalPoint {
                chars: r.chars.0 as f64,
                nats: r.overall_nats,
                se: r.overall_se,
                train_nats: Some(r.train_nats_ema),
            }),
            _ => None,
        })
        .collect()
}

/// Run until `n_evals` evaluations have happened, then stop (saving).
fn run_until_evals(scenario: Scenario, n_evals: usize) -> Recorded {
    let dir = tempfile::tempdir().unwrap();
    let s = spec(dir.path(), None);
    let mut seen = 0;
    let rec = run_with_driver(&factory(scenario), s.clone(), |ev, ctl| {
        if matches!(ev, EngineEvent::Eval(_)) {
            seen += 1;
            if seen == n_evals {
                ctl.send(Ctl::Stop { save: true }).unwrap();
            }
        }
    });
    check_invariants(&s, &rec);
    rec
}

#[test]
fn normal_run_with_pause_resume_and_stop_satisfies_the_contract() {
    let dir = tempfile::tempdir().unwrap();
    let s = spec(dir.path(), None);
    let (mut evals_seen, mut paused_once, mut saw_ckpt) = (0, false, false);
    let rec = run_with_driver(&factory(Scenario::Normal), s.clone(), |ev, ctl| match ev {
        EngineEvent::Eval(_) => {
            evals_seen += 1;
            if evals_seen == 2 && !paused_once {
                paused_once = true;
                ctl.send(Ctl::Pause).unwrap();
            }
        }
        EngineEvent::Stage(st) if st.stage == Stage::Paused => {
            std::thread::sleep(std::time::Duration::from_millis(150));
            ctl.send(Ctl::Resume).unwrap();
        }
        EngineEvent::CheckpointSaved(_) if !saw_ckpt => {
            saw_ckpt = true;
            ctl.send(Ctl::Stop { save: true }).unwrap();
        }
        _ => {}
    });
    check_invariants(&s, &rec);
    assert_eq!(rec.outcome, Outcome::Stopped);
    assert!(paused_once && saw_ckpt);
    assert!(rec.events.iter().any(|e| matches!(e, EngineEvent::Stage(st) if st.stage == Stage::Paused)));
    assert!(
        rec.events.iter().any(|e| matches!(e, EngineEvent::CheckpointSaved(c) if c.kind == CheckpointKind::Stop)),
        "stopping with save must write a Stop checkpoint"
    );
    assert!(rec.events.iter().any(|e| matches!(e, EngineEvent::Samples(_))));
    assert!(rec.events.iter().any(|e| matches!(e, EngineEvent::Pool(_))));
}

#[test]
fn stop_without_save_writes_no_stop_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let s = spec(dir.path(), None);
    let mut steps = 0;
    let rec = run_with_driver(&factory(Scenario::Normal), s.clone(), |ev, ctl| {
        if matches!(ev, EngineEvent::Step(_)) {
            steps += 1;
            if steps == 30 {
                ctl.send(Ctl::Stop { save: false }).unwrap();
            }
        }
    });
    check_invariants(&s, &rec);
    assert!(!rec.events.iter().any(|e| matches!(e, EngineEvent::CheckpointSaved(c) if c.kind == CheckpointKind::Stop)));
}

#[test]
fn resume_continues_from_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let s1 = spec(dir.path(), None);
    let mut stop_sent = false;
    let mut first = run_with_driver(&factory(Scenario::Normal), s1.clone(), |ev, ctl| {
        if matches!(ev, EngineEvent::CheckpointSaved(_)) && !stop_sent {
            stop_sent = true;
            ctl.send(Ctl::Stop { save: true }).unwrap();
        }
    });
    check_invariants(&s1, &first);
    let saved = first
        .events
        .drain(..)
        .filter_map(|e| if let EngineEvent::CheckpointSaved(c) = e { Some(c) } else { None })
        .next_back()
        .expect("a checkpoint");
    let s2 = spec(dir.path(), Some(s1.run_dir.join(&saved.path)));
    let mut steps = 0;
    let second = run_with_driver(&factory(Scenario::Normal), s2.clone(), |ev, ctl| {
        if matches!(ev, EngineEvent::Step(_)) {
            steps += 1;
            if steps == 10 {
                ctl.send(Ctl::Stop { save: false }).unwrap();
            }
        }
    });
    let first_step =
        second.events.iter().find_map(|e| if let EngineEvent::Step(s) = e { Some(s.step.0) } else { None }).unwrap();
    assert_eq!(first_step, saved.step.0 + 1, "a resumed run continues right after the checkpoint");
    assert!(second.events.iter().any(|e| matches!(e, EngineEvent::Stage(s) if s.stage == Stage::LoadingCheckpoint)));
}

#[test]
fn each_scenario_produces_the_verdict_a_user_should_see() {
    let verdict_of = |scenario, n| insight::verdict(&evals(&run_until_evals(scenario, n))).verdict;
    assert_eq!(verdict_of(Scenario::Normal, 12), Verdict::Learning);
    assert_eq!(verdict_of(Scenario::Plateau, 60), Verdict::Plateau);
    assert_eq!(verdict_of(Scenario::Overfit, 110), Verdict::Overfitting);
    assert_eq!(verdict_of(Scenario::Diverge, 140), Verdict::Diverging);
}

#[test]
fn failure_scenarios_fail_with_actionable_errors() {
    for (scenario, check) in [
        (
            Scenario::OomAtStart,
            (|e: &AppError| matches!(e, AppError::OutOfMemory { suggested_preset: Some(_) })) as fn(&AppError) -> bool,
        ),
        (Scenario::DiskFull, |e: &AppError| matches!(e, AppError::DiskFull { .. })),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let s = spec(dir.path(), None);
        let rec = run_with_driver(&factory(scenario), s.clone(), |_, _| {});
        check_invariants(&s, &rec);
        match rec.outcome {
            Outcome::Failed { error } => assert!(check(&error), "unexpected error {error:?}"),
            other => panic!("expected failure, got {other:?}"),
        }
    }
}

#[test]
fn no_gpu_scenario_only_offers_the_cpu() {
    let f = factory(Scenario::NoGpu);
    let b = f.probe_backends();
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].kind, BackendKind::Cpu);
    assert!(f.is_mock());
}

#[test]
fn generator_streams_text_and_supports_learning() {
    let dir = tempfile::tempdir().unwrap();
    let f = factory(Scenario::Normal);
    let mut g = f.open_generator(dir.path(), BackendKind::Cpu).unwrap();
    let mut text = String::new();
    let stats = g
        .generate(
            &minagi_types::GenRequest { prompt: "Once".into(), max_new: 60, adapt: true, stop_at: None },
            &mut |c| {
                text.push_str(&c.text);
                assert!(c.rows >= 1 && !c.experts.is_empty());
                true
            },
        )
        .unwrap();
    assert!(!text.is_empty() && stats.chars.0 as usize == text.chars().count());
    let learned = g.learn("hello").unwrap();
    assert!(learned.nats_after < learned.nats_before);
    // Cancellation stops generation early.
    let mut n = 0;
    g.generate(
        &minagi_types::GenRequest { prompt: "x".into(), max_new: 100, adapt: false, stop_at: None },
        &mut |_| {
            n += 1;
            n < 5
        },
    )
    .unwrap();
    assert_eq!(n, 5);
}
