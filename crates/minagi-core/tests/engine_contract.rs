//! The real engine, run end to end on a toy dataset (CPU) and held to the same contract as the simulated one.

use std::path::{Path, PathBuf};

use minagi_core::RealFactory;
use minagi_types::contract::{Recorded, check_invariants, run_with_driver};
use minagi_types::{
    BackendKind, Ctl, EngineEvent, EngineFactory, GrowthEvent, ModelConfig, Outcome, Preset, RunSpec, Stage,
};

/// A few kilobytes of regular English-like text per file, so a tiny model can visibly learn it in a few hundred steps.
fn prose(seed: usize, bytes: usize) -> String {
    const WORDS: [&str; 24] = [
        "the", "little", "river", "runs", "past", "old", "stone", "bridge", "and", "a", "quiet", "garden", "grows",
        "near", "it", "while", "children", "play", "under", "bright", "morning", "light", "every", "day",
    ];
    let mut out = String::new();
    let mut i = seed;
    while out.len() < bytes {
        for k in 0..7 {
            out.push_str(WORDS[(i * 7 + k * 5 + seed) % WORDS.len()]);
            out.push(' ');
        }
        out.push_str(".\n");
        i += 1;
    }
    out
}

fn dataset(dir: &Path) -> PathBuf {
    let root = dir.join("data");
    for (side, files) in [("train", 3), ("val", 1)] {
        let lane = root.join(side).join("stories");
        std::fs::create_dir_all(&lane).unwrap();
        for f in 0..files {
            std::fs::write(lane.join(format!("{f}.txt")), prose(f + if side == "val" { 11 } else { 0 }, 6_000))
                .unwrap();
        }
    }
    root
}

fn small_model() -> ModelConfig {
    let mut c = Preset::Tiny.model();
    c.d_model = 32;
    c.n_head = 2;
    c.d_ff = 64;
    c.n_prelude = 1;
    c.max_steps = 4;
    c.bptt_window = 4;
    c.train_steps_mean = 0.0;
    c.halt_prior = 0.4;
    c.block = 96;
    c.pool_experts = 8;
    c.pool_max = 12;
    c.pool_d_ff = 16;
    c.pool_top_k = 2;
    c
}

fn spec(dir: &Path, resume: Option<PathBuf>) -> RunSpec {
    let mut t = Preset::Tiny.train();
    t.chunk = 32;
    t.passage = 192;
    t.context_start = 64;
    t.context_end = 96;
    t.resident = 4;
    t.ram_cache = 6;
    t.lr = 4e-3;
    t.sample_every_min = 0.03;
    t.save_every_min = 0.06;
    t.eval_chars = 192;
    t.growth.every_chars = 160; // a growth check every 5 steps
    t.prune.survival_chars = 3_200;
    RunSpec {
        run_dir: dir.join("run"),
        data_root: dir.join("data"),
        lanes: None,
        model: small_model(),
        train: t,
        seed: 7,
        resume_from: resume,
        backend: BackendKind::Cpu,
    }
}

fn steps(rec: &Recorded) -> Vec<(u64, f64, u64)> {
    rec.events
        .iter()
        .filter_map(
            |e| if let EngineEvent::Step(s) = e { Some((s.step.0, s.train_nats, s.chars_read.0)) } else { None },
        )
        .collect()
}

#[test]
fn the_real_engine_trains_evaluates_samples_grows_saves_and_resumes() {
    let dir = tempfile::tempdir().unwrap();
    dataset(dir.path());
    let factory = RealFactory::new(16.0);
    let s = spec(dir.path(), None);
    let (mut evals, mut samples, mut ckpts, mut growth) = (0, 0, 0, 0);
    let rec = run_with_driver(&factory, s.clone(), |ev, ctl| {
        match ev {
            EngineEvent::Eval(_) => evals += 1,
            EngineEvent::Samples(_) => samples += 1,
            EngineEvent::CheckpointSaved(_) => ckpts += 1,
            EngineEvent::Growth(_) => growth += 1,
            _ => {}
        }
        if evals >= 2 && samples >= 1 && ckpts >= 1 && growth >= 1 {
            let _ = ctl.send(Ctl::Stop { save: true });
        }
    });
    check_invariants(&s, &rec);
    assert_eq!(rec.outcome, Outcome::Stopped);

    // it learned: the training score fell well below ln(265) = 5.58 and held-out followed
    let st = steps(&rec);
    assert!(st.len() > 30, "only {} steps", st.len());
    let head: f64 = st[..10].iter().map(|s| s.1).sum::<f64>() / 10.0;
    let tail: f64 = st[st.len() - 10..].iter().map(|s| s.1).sum::<f64>() / 10.0;
    assert!(head > 5.0, "a fresh model should start near ln(265): {head}");
    assert!(tail < head - 0.5, "training loss did not fall: {head} -> {tail}");
    let held: Vec<f64> = rec
        .events
        .iter()
        .filter_map(|e| if let EngineEvent::Eval(r) = e { Some(r.overall_nats) } else { None })
        .collect();
    assert!(held.iter().all(|h| h.is_finite()) && held.last().unwrap() < &5.4, "held-out scores {held:?}");
    assert!(rec.events.iter().any(|e| matches!(e, EngineEvent::Plasticity(_))));
    assert!(rec.events.iter().any(|e| matches!(e, EngineEvent::Pool(_))));
    assert!(
        rec.events
            .iter()
            .any(|e| matches!(e, EngineEvent::Growth(GrowthEvent::Born { .. } | GrowthEvent::Blocked { .. })))
    );
    for e in &rec.events {
        if let EngineEvent::Samples(r) = e {
            assert!(r.items.iter().all(|i| i.raw.chars().count() > 0 && i.adapted.chars().count() > 0));
        }
    }
    // the stop saved: the last checkpoint is a Stop checkpoint and is complete
    let last_ckpt = rec
        .events
        .iter()
        .rev()
        .find_map(|e| if let EngineEvent::CheckpointSaved(c) = e { Some(c.clone()) } else { None })
        .expect("a checkpoint");
    assert_eq!(last_ckpt.kind, minagi_types::CheckpointKind::Stop);
    let ck_dir = s.run_dir.join(&last_ckpt.path);
    assert!(
        ck_dir.join("COMPLETE").is_file() && ck_dir.join("manifest.json").is_file() && ck_dir.join("experts").is_dir()
    );
    let last_step = st.last().unwrap().0;
    assert_eq!(last_ckpt.step.0, last_step);

    // resume: continues from the saved step with the saved characters and pool
    let s2 = spec(dir.path(), Some(ck_dir.clone()));
    let mut seen = 0;
    let rec2 = run_with_driver(&factory, s2.clone(), |ev, ctl| {
        if matches!(ev, EngineEvent::Step(_)) {
            seen += 1;
            if seen == 8 {
                let _ = ctl.send(Ctl::Stop { save: false });
            }
        }
    });
    check_invariants(&s2, &rec2);
    let st2 = steps(&rec2);
    assert_eq!(st2.first().unwrap().0, last_step + 1, "the step counter continues");
    assert!(st2.first().unwrap().2 > st.last().unwrap().2 - 1, "characters read continue");
    assert!(st2[0].1 < head - 0.3, "a resumed model has kept what it learned: {} vs fresh {head}", st2[0].1);
}

#[test]
fn pause_resume_and_stop_without_saving_follow_the_contract() {
    let dir = tempfile::tempdir().unwrap();
    dataset(dir.path());
    let s = spec(dir.path(), None);
    let (mut n, mut paused) = (0, false);
    let rec = run_with_driver(&RealFactory::new(16.0), s.clone(), |ev, ctl| match ev {
        EngineEvent::Step(_) => {
            n += 1;
            if n == 5 && !paused {
                paused = true;
                let _ = ctl.send(Ctl::Pause);
            }
            if n == 15 {
                let _ = ctl.send(Ctl::Stop { save: false });
            }
        }
        EngineEvent::Stage(st) if st.stage == Stage::Paused => {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let _ = ctl.send(Ctl::Resume);
        }
        _ => {}
    });
    check_invariants(&s, &rec);
    assert_eq!(rec.outcome, Outcome::Stopped);
    assert!(rec.events.iter().any(|e| matches!(e, EngineEvent::Stage(s) if s.stage == Stage::Paused)));
    assert!(
        !rec.events
            .iter()
            .any(|e| matches!(e, EngineEvent::CheckpointSaved(c) if c.kind == minagi_types::CheckpointKind::Stop))
    );
}

#[test]
fn a_dataset_with_no_text_fails_with_a_plain_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("data").join("train")).unwrap();
    let s = spec(dir.path(), None);
    let rec = run_with_driver(&RealFactory::new(16.0), s.clone(), |_, _| {});
    check_invariants(&s, &rec);
    assert!(matches!(rec.outcome, Outcome::Failed { .. }), "{:?}", rec.outcome);
}

#[test]
fn the_generator_writes_learns_on_a_copy_and_saves_it() {
    use minagi_types::GenRequest;
    let dir = tempfile::tempdir().unwrap();
    dataset(dir.path());
    let factory = RealFactory::new(16.0);
    let s = spec(dir.path(), None);
    let rec = run_with_driver(&factory, s.clone(), |ev, ctl| {
        if let EngineEvent::Step(st) = ev
            && st.step.0 >= 40
        {
            let _ = ctl.send(Ctl::Stop { save: true });
        }
    });
    let ck = rec
        .events
        .iter()
        .rev()
        .find_map(|e| if let EngineEvent::CheckpointSaved(c) = e { Some(s.run_dir.join(&c.path)) } else { None })
        .unwrap();
    let before: Vec<_> =
        std::fs::read_dir(ck.join("experts")).unwrap().flatten().map(|e| e.metadata().unwrap().len()).collect();

    let mut g = factory.open_generator(&ck, BackendKind::Cpu).unwrap();
    assert!(g.info().supports_learn);
    let req = GenRequest { prompt: "the little river ".into(), max_new: 40, adapt: false, stop_at: None };
    let mut text = String::new();
    let mut rows = Vec::new();
    let stats = g
        .generate(&req, &mut |c| {
            text.push_str(&c.text);
            rows.push(c.rows);
            true
        })
        .unwrap();
    assert_eq!(stats.chars.0, 40);
    assert_eq!(rows.len(), 40);
    assert!(rows.iter().all(|&r| (1..=4).contains(&r)));
    // cancelling from the callback ends writing at once
    let mut count = 0;
    let stats = g
        .generate(&req, &mut |_| {
            count += 1;
            count < 5
        })
        .unwrap();
    assert_eq!(stats.chars.0, 5);

    // learning moves the copy and reports a before/after score
    let learn_text = prose(3, 600);
    let rep = g.learn(&learn_text).unwrap();
    assert!(rep.nats_before.is_finite() && rep.nats_after.is_finite());
    assert!(rep.nats_after < rep.nats_before + 0.05, "{rep:?}");
    let dest = dir.path().join("adapted");
    let meta = g.save_adapted(&dest).unwrap();
    assert!(dest.join("COMPLETE").is_file());
    assert!(meta.step.0 >= 40);
    drop(g);

    // the original checkpoint was never touched
    let after: Vec<_> =
        std::fs::read_dir(ck.join("experts")).unwrap().flatten().map(|e| e.metadata().unwrap().len()).collect();
    assert_eq!(before, after);
    // and the adapted one opens and writes too
    let mut g2 = factory.open_generator(&dest, BackendKind::Cpu).unwrap();
    g2.generate(&GenRequest { prompt: "the ".into(), max_new: 5, adapt: true, stop_at: None }, &mut |_| true).unwrap();
}
