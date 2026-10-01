//! Golden tests for the three pure-logic ports: the learning-rate controller, the growth brakes and the context
//! ramp. The fixtures in `tests/fixtures/pure/` are written by `tools/golden/pure_*.py`, which drive the real
//! reference code (`minagi/plasticity.py`, `minagi/pool.py::AutoGrow`, and the ramp statements extracted from
//! `train.py`). Continuous quantities must agree to 1e-9 (relative); decisions must agree exactly.

use minagi_core::optim::brakes::{BrakeKind, GrowthInputs, decide};
use minagi_core::optim::plasticity::{GradSnr, Plasticity};
use minagi_core::train::context::{ContextRamp, RampSettings, chars_to_steps, ramp_schedule, visit_chunks};
use minagi_types::GrowthConfig;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::path::PathBuf;

const REL: f64 = 1e-9;
/// Below this magnitude two numbers are equal for our purposes (a relative test is meaningless near zero).
const ABS: f64 = 1e-12;

fn load<T: DeserializeOwned>(name: &str) -> T {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pure").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + ABS
}

/// First difference between two JSON values (numbers compared to `REL`), as a path and message.
fn json_diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64()?, y.as_f64()?);
            (!close(x, y, REL)).then(|| format!("{path}: {x} vs {y}"))
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: length {} vs {}", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().find_map(|(i, (p, q))| json_diff(p, q, &format!("{path}[{i}]")))
        }
        (Value::Object(x), Value::Object(y)) => {
            if x.len() != y.len() || x.keys().any(|k| !y.contains_key(k)) {
                return Some(format!(
                    "{path}: keys {:?} vs {:?}",
                    x.keys().collect::<Vec<_>>(),
                    y.keys().collect::<Vec<_>>()
                ));
            }
            x.iter().find_map(|(k, v)| json_diff(v, &y[k], &format!("{path}.{k}")))
        }
        _ => (a != b).then(|| format!("{path}: {a} vs {b}")),
    }
}

fn assert_json_close(got: &Value, want: &Value, what: &str) {
    if let Some(d) = json_diff(got, want, "") {
        panic!("{what}: {d}");
    }
}

/// A fixture number: a JSON number, `"nan"`, `"inf"`, `"-inf"`, or `null` for "missing".
fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "nan" => Some(f64::NAN),
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

// ============================================================================ plasticity

#[derive(Deserialize)]
struct Obs {
    v: Value,
    s: Value,
    k: u32,
}

#[derive(Deserialize)]
struct Expect {
    scale: f64,
    t: f64,
    e: f64,
    n: f64,
    #[serde(default)]
    jumped: bool,
    #[serde(default = "yes")]
    counted: bool,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    factor: Option<f64>,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    obs: Vec<Obs>,
    initial_state: Option<Value>,
    expect: Vec<Expect>,
    final_state: Value,
    #[serde(default)]
    resume_after: Option<usize>,
    #[serde(default)]
    resume_state: Option<Value>,
}

#[derive(Deserialize)]
struct RestoreCase {
    name: String,
    state: Value,
    state_after_restore: Value,
    factor: f64,
    tail_obs: Vec<Obs>,
    tail: Vec<Expect>,
}

#[derive(Deserialize)]
struct Fit {
    vals: Vec<f64>,
    t: f64,
    n: f64,
    e: f64,
}

#[derive(Deserialize)]
struct Factor {
    scale: f64,
    step: u64,
    factor: f64,
}

#[derive(Deserialize)]
struct Describe {
    scale: f64,
    se: Vec<f64>,
    text: String,
}

#[derive(Deserialize)]
struct PlasticityFixture {
    constants: std::collections::BTreeMap<String, f64>,
    scenarios: Vec<Scenario>,
    restore_cases: Vec<RestoreCase>,
    factors: Vec<Factor>,
    fits: Vec<Fit>,
    describe: Vec<Describe>,
}

/// Feed `obs[range]` to `p`, checking every evaluation against `expect`; returns nothing, panics on a mismatch.
fn replay(p: &mut Plasticity, obs: &[Obs], expect: &[Expect], first_step: u64, what: &str) {
    let mut step = first_step;
    for (i, (o, want)) in obs.iter().zip(expect).enumerate() {
        for _ in 0..o.k {
            p.tick();
        }
        step += u64::from(o.k);
        assert_eq!(p.step(), step, "{what} #{i}: step");
        let val = num(&o.v).unwrap_or(f64::NAN);
        let got = p.observe(val, num(&o.s));
        let at = format!("{what} #{i}");
        assert!(close(got.scale, want.scale, REL), "{at}: scale {} vs {}", got.scale, want.scale);
        assert!(close(got.t, want.t, REL), "{at}: t {} vs {}", got.t, want.t);
        assert!(close(got.effect, want.e, REL), "{at}: e {} vs {}", got.effect, want.e);
        assert!(close(got.n_eff, want.n, REL), "{at}: n_eff {} vs {}", got.n_eff, want.n);
        assert_eq!(got.jumped, want.jumped, "{at}: jumped");
        assert_eq!(got.counted, want.counted, "{at}: counted");
        assert_eq!(got.note, want.note, "{at}: note");
        assert!(!got.reason.is_empty(), "{at}: reason");
        match want.factor {
            Some(f) => assert!(close(p.factor(), f, REL), "{at}: factor {} vs {f}", p.factor()),
            None => assert_eq!(p.factor(), got.scale, "{at}: factor past the warmup is the scale"),
        }
    }
}

#[test]
fn plasticity_constants_match_the_reference() {
    let fx: PlasticityFixture = load("plasticity.json");
    let ours = [
        ("FLOOR", Plasticity::FLOOR),
        ("CEIL", Plasticity::CEIL),
        ("LAM", Plasticity::LAM),
        ("LAM_FAST", Plasticity::LAM_FAST),
        ("EFFECT", Plasticity::EFFECT),
        ("MIN_EFF", Plasticity::MIN_EFF),
        ("NUDGE", Plasticity::NUDGE),
        ("NUDGE_DOWN", Plasticity::NUDGE_DOWN),
        ("T_W", Plasticity::T_W),
        ("T_W_DOWN", Plasticity::T_W_DOWN),
        ("T_MID", Plasticity::T_MID),
        ("JUMP_SE", Plasticity::JUMP_SE),
        ("HOLD_SE", Plasticity::HOLD_SE),
        ("FLOOR_JUMP", Plasticity::FLOOR_JUMP),
        ("UP", Plasticity::UP),
        ("WARMUP", Plasticity::WARMUP as f64),
    ];
    assert_eq!(ours.len(), fx.constants.len());
    for (name, v) in ours {
        assert_eq!(fx.constants.get(name), Some(&v), "{name}");
    }
}

#[test]
fn plasticity_matches_python_over_every_scenario() {
    let fx: PlasticityFixture = load("plasticity.json");
    let mut evaluations = 0;
    for sc in &fx.scenarios {
        let start = |s: &Option<Value>| s.as_ref().map_or_else(Plasticity::new, Plasticity::restore);
        let first_step = sc.initial_state.as_ref().and_then(|s| s["step"].as_u64()).unwrap_or(0);
        let mut p = start(&sc.initial_state);
        evaluations += sc.obs.len();
        match sc.resume_after {
            None => {
                replay(&mut p, &sc.obs, &sc.expect, first_step, &sc.name);
                assert_json_close(&p.state(), &sc.final_state, &format!("{}: final state", sc.name));
            }
            Some(k) => {
                // run to the checkpoint, and compare what we would have written with what Python wrote
                replay(&mut p, &sc.obs[..k], &sc.expect[..k], first_step, &sc.name);
                let ours = p.state();
                let theirs = sc.resume_state.as_ref().expect("resume_state");
                assert_json_close(&ours, theirs, &format!("{}: state at the checkpoint", sc.name));
                let at = first_step + sc.obs[..k].iter().map(|o| u64::from(o.k)).sum::<u64>();
                // resume from the state PYTHON wrote, and from the state WE wrote: both must carry on identically
                for (label, saved) in [("python's state", theirs), ("our state", &ours)] {
                    let mut q = Plasticity::restore(saved);
                    replay(&mut q, &sc.obs[k..], &sc.expect[k..], at, &format!("{} resumed from {label}", sc.name));
                    assert_json_close(&q.state(), &sc.final_state, &format!("{} ({label}): final state", sc.name));
                }
            }
        }
    }
    assert!(evaluations >= 2000, "only {evaluations} evaluations replayed");
}

#[test]
fn plasticity_restores_every_checkpoint_shape_like_python() {
    let fx: PlasticityFixture = load("plasticity.json");
    for c in &fx.restore_cases {
        let mut p = Plasticity::restore(&c.state);
        assert_json_close(&p.state(), &c.state_after_restore, &format!("{}: state after restore", c.name));
        assert!(close(p.factor(), c.factor, REL), "{}: factor", c.name);
        let first_step = p.step();
        replay(&mut p, &c.tail_obs, &c.tail, first_step, &c.name);
    }
}

#[test]
fn plasticity_evidence_on_tiny_histories_matches_python() {
    let fx: PlasticityFixture = load("plasticity.json");
    for f in &fx.fits {
        let p = if f.vals.is_empty() { Plasticity::new() } else { Plasticity::restore(&json!({ "hist": f.vals })) };
        let e = p.evidence();
        assert!(close(e.t, f.t, REL) && close(e.n_eff, f.n, REL) && close(e.effect, f.e, REL), "{:?}", f.vals);
    }
}

#[test]
fn plasticity_warmup_factor_matches_python() {
    let fx: PlasticityFixture = load("plasticity.json");
    assert!(fx.factors.len() > 300);
    for f in &fx.factors {
        let p = Plasticity::restore(&json!({ "scale": f.scale, "step": f.step }));
        assert!(
            close(p.factor(), f.factor, 1e-15),
            "scale {} step {}: {} vs {}",
            f.scale,
            f.step,
            p.factor(),
            f.factor
        );
    }
}

#[test]
fn plasticity_describe_matches_python() {
    let fx: PlasticityFixture = load("plasticity.json");
    for d in &fx.describe {
        let p = Plasticity::restore(&json!({ "scale": d.scale, "se": d.se }));
        assert_eq!(p.describe(), d.text);
    }
}

// ============================================================================ gradient signal

#[derive(Deserialize)]
struct SnrRow {
    grads: Vec<Option<Vec<f64>>>,
    ratio: Option<f64>,
}

#[derive(Deserialize)]
struct SnrScenario {
    name: String,
    beta: f64,
    rows: Vec<SnrRow>,
}

#[derive(Deserialize)]
struct SnrFixture {
    scenarios: Vec<SnrScenario>,
}

#[test]
fn grad_snr_matches_python() {
    let fx: SnrFixture = load("gradsnr.json");
    for sc in &fx.scenarios {
        let mut m = GradSnr::new(sc.beta);
        for (i, row) in sc.rows.iter().enumerate() {
            let grads: Vec<Vec<f32>> =
                row.grads.iter().flatten().map(|g| g.iter().map(|&x| x as f32).collect()).collect();
            let refs: Vec<&[f32]> = grads.iter().map(Vec::as_slice).collect();
            let got = m.observe(&refs);
            // the reference works in float32; ours accumulates the sums in float64
            match (got, row.ratio) {
                (None, None) => {}
                (Some(g), Some(w)) => assert!(close(g, w, 1e-4), "{} #{i}: {g} vs {w}", sc.name),
                (g, w) => panic!("{} #{i}: {g:?} vs {w:?}", sc.name),
            }
        }
    }
}

// ============================================================================ growth brakes

#[derive(Deserialize)]
struct BrakeCfg {
    k: u32,
    max_experts: u32,
    dying_frac_max: f64,
    keep_ratio_min: f64,
    mem_frac: f64,
    max_disk_gb: f64,
    max_in_flight: u32,
    recent_mult: f64,
    max_gap: f64,
}

#[derive(Deserialize)]
struct BrakeIn {
    step: f64,
    born: Vec<f64>,
    dying: Vec<f64>,
    dying_at: f64,
    trial: f64,
    mem: f64,
    per: u64,
    gap: Option<f64>,
    has_disk: bool,
}

#[derive(Deserialize)]
struct BrakeOut {
    grew: u32,
    blocked: String,
    in_flight: usize,
    idle: usize,
    keep_ratio: f64,
}

#[derive(Deserialize)]
struct BrakeCase {
    cfg: BrakeCfg,
    #[serde(rename = "in")]
    inp: BrakeIn,
    out: BrakeOut,
}

#[derive(Deserialize)]
struct Sim {
    name: String,
    rounds: Vec<BrakeCase>,
}

#[derive(Deserialize)]
struct BrakeFixture {
    cases: Vec<BrakeCase>,
    simulations: Vec<Sim>,
}

fn check_brake_case(c: &BrakeCase, what: &str) {
    let cfg = GrowthConfig {
        k: c.cfg.k,
        max_gap: c.cfg.max_gap,
        dying_frac_max: c.cfg.dying_frac_max,
        max_in_flight: c.cfg.max_in_flight,
        keep_ratio_min: c.cfg.keep_ratio_min,
        recent_mult: c.cfg.recent_mult,
        max_disk_gb: c.cfg.max_disk_gb,
        mem_frac: c.cfg.mem_frac,
        ..GrowthConfig::default()
    };
    let n = c.inp.dying.len() as u64;
    let (now, after) = if c.inp.has_disk { (n * c.inp.per, (n + u64::from(c.cfg.k)) * c.inp.per) } else { (0, 0) };
    let inp = GrowthInputs {
        model_step: c.inp.step,
        max_experts: c.cfg.max_experts,
        dying: &c.inp.dying,
        dying_at: c.inp.dying_at,
        born: &c.inp.born,
        trial: c.inp.trial,
        mem_frac: c.inp.mem,
        disk_bytes_now: now,
        disk_bytes_after: after,
        last_gap: c.inp.gap,
    };
    let d = decide(&cfg, &inp);

    assert_eq!(d.grow, c.out.grew, "{what}: grow");
    let want = match c.out.blocked.as_str() {
        "none" => None,
        "honest" => Some(BrakeKind::Honest),
        "fits" => Some(BrakeKind::Fits),
        "room" => Some(BrakeKind::Room),
        "used" => Some(BrakeKind::Used),
        "earning" => Some(BrakeKind::Earning),
        other => panic!("unknown brake {other}"),
    };
    assert_eq!(d.blocked_by, want, "{what}: which brake refused");
    assert_eq!(d.measures.in_flight, c.out.in_flight, "{what}: in flight");
    assert_eq!(d.measures.idle, c.out.idle, "{what}: idle");
    assert_eq!(d.measures.keep_ratio, c.out.keep_ratio, "{what}: keep ratio");
    assert_eq!(d.brakes.all_ok(), c.out.grew > 0, "{what}: all_ok");

    // Every brake the reference ranks ahead of the one that refused said yes, and the refusing one said no.
    let order = [BrakeKind::Honest, BrakeKind::Fits, BrakeKind::Room, BrakeKind::Used, BrakeKind::Earning];
    let oks = [d.brakes.honest.ok, d.brakes.fits.ok, d.brakes.room.ok, d.brakes.used.ok, d.brakes.earning.ok];
    for (kind, ok) in order.iter().zip(oks) {
        match want {
            Some(w) if w == *kind => assert!(!ok, "{what}: {kind:?} should refuse"),
            Some(w) if order.iter().position(|x| *x == w) < order.iter().position(|x| x == kind) => {}
            _ => assert!(ok, "{what}: {kind:?} should agree"),
        }
    }
    assert!(!d.reason.is_empty());
}

#[test]
fn growth_brakes_match_python_on_every_case() {
    let fx: BrakeFixture = load("brakes.json");
    assert!(fx.cases.len() >= 250);
    for (i, c) in fx.cases.iter().enumerate() {
        check_brake_case(c, &format!("case {i}"));
    }
    // all five refusals and growth itself are exercised by the fixture
    for blocked in ["none", "honest", "fits", "room", "used", "earning"] {
        assert!(fx.cases.iter().any(|c| c.out.blocked == blocked), "no case blocked by {blocked}");
    }
}

#[test]
fn growth_brakes_match_python_over_simulated_runs() {
    let fx: BrakeFixture = load("brakes.json");
    assert_eq!(fx.simulations.len(), 2);
    for sim in &fx.simulations {
        for (i, c) in sim.rounds.iter().enumerate() {
            check_brake_case(c, &format!("{} round {i}", sim.name));
        }
        assert!(sim.rounds.iter().any(|c| c.out.grew > 0), "{} never grew", sim.name);
    }
}

// ============================================================================ context ramp

#[derive(Deserialize)]
struct RampArgs {
    chunk: u32,
    passage: u32,
    context: u32,
    context_start: u32,
    context_step: u32,
    context_gain_min: f64,
    context_grow_every: u32,
    context_every: u32,
}

#[derive(Deserialize)]
struct GainSample {
    step: u64,
    gain: Option<f64>,
    wants_more: bool,
    ctx: u32,
}

#[derive(Deserialize)]
struct RampScenario {
    name: String,
    args: RampArgs,
    block: u32,
    saved: Option<u32>,
    nsteps: u64,
    losses: Vec<f64>,
    start_ctx: u32,
    max_ctx: u32,
    every_steps: u64,
    grow_every_steps: u64,
    changes: Vec<(u64, u32)>,
    samples: Vec<GainSample>,
    final_ctx: u32,
    final_turn: u32,
}

#[derive(Deserialize)]
struct StartCase {
    block: u32,
    context: u32,
    start: u32,
    saved: Option<u32>,
    ctx_now: u32,
    ctx_max: u32,
    was_above: bool,
}

#[derive(Deserialize)]
struct GainCase {
    chunk: u32,
    ctx_now: u32,
    samples: Vec<(u32, f64)>,
    gain: Option<f64>,
}

#[derive(Deserialize)]
struct InSteps {
    chars: u64,
    chunk: u32,
    steps: u64,
}

#[derive(Deserialize)]
struct Visit {
    ctx: u32,
    chunk: u32,
    passage: u32,
    turn: u32,
}

#[derive(Deserialize)]
struct Ramp {
    step: u64,
    total: u64,
    start: u32,
    end: u32,
    warm: f64,
    granularity: u32,
    ctx: u32,
}

#[derive(Deserialize)]
struct ContextFixture {
    scenarios: Vec<RampScenario>,
    starts: Vec<StartCase>,
    gains: Vec<GainCase>,
    in_steps: Vec<InSteps>,
    visits: Vec<Visit>,
    ramps: Vec<Ramp>,
}

fn settings(a: &RampArgs, block: u32) -> RampSettings {
    RampSettings {
        chunk: a.chunk,
        passage: a.passage,
        block,
        context_start: a.context_start,
        context_end: a.context,
        context_step: a.context_step,
        grow_every_chars: a.context_grow_every,
        check_every_chars: a.context_every,
        gain_min: a.context_gain_min,
    }
}

#[test]
fn context_ramp_follows_python_step_for_step() {
    let fx: ContextFixture = load("context.json");
    for sc in &fx.scenarios {
        let mut r = ContextRamp::new(&settings(&sc.args, sc.block), sc.saved);
        assert_eq!((r.context(), r.ceiling()), (sc.start_ctx, sc.max_ctx), "{}: start", sc.name);
        assert_eq!((r.check_every_steps(), r.grow_every_steps()), (sc.every_steps, sc.grow_every_steps), "{}", sc.name);
        let sample_every = sc.samples.get(1).zip(sc.samples.first()).map_or(70, |(b, a)| b.step - a.step);
        let (mut step, mut at, mut changes, mut samples) = (0u64, 0usize, Vec::new(), sc.samples.iter());
        while step < sc.nsteps {
            let turn = r.visit_chunks(); // a visit keeps the length it started with
            for j in 0..turn {
                if step >= sc.nsteps {
                    break;
                }
                r.record(j, sc.losses[at]);
                at += 1;
                step += 1;
                if let Some(c) = r.after_step(step) {
                    changes.push((step, c));
                }
                if step % sample_every == 0 {
                    let want = samples.next().unwrap_or_else(|| panic!("{}: extra sample at {step}", sc.name));
                    assert_eq!(want.step, step, "{}", sc.name);
                    match (r.gain(), want.gain) {
                        (None, None) => {}
                        (Some(g), Some(w)) => assert!(close(g, w, REL), "{} step {step}: gain {g} vs {w}", sc.name),
                        (g, w) => panic!("{} step {step}: gain {g:?} vs {w:?}", sc.name),
                    }
                    assert_eq!(r.wants_more(), want.wants_more, "{} step {step}: wants_more", sc.name);
                    assert_eq!(r.context(), want.ctx, "{} step {step}: context", sc.name);
                }
            }
        }
        assert!(samples.next().is_none(), "{}: samples missing", sc.name);
        assert_eq!(changes, sc.changes, "{}: when the window grew", sc.name);
        assert_eq!(r.context(), sc.final_ctx, "{}: final window", sc.name);
        assert_eq!(r.visit_chunks(), sc.final_turn, "{}: visit length", sc.name);
    }
}

#[test]
fn context_start_and_ceiling_match_python() {
    let fx: ContextFixture = load("context.json");
    assert!(fx.starts.len() > 200);
    for c in &fx.starts {
        let s = RampSettings {
            chunk: 512,
            passage: 16384,
            block: c.block,
            context_start: c.start,
            context_end: c.context,
            context_step: 1,
            grow_every_chars: 100_000,
            check_every_chars: 65_536,
            gain_min: 0.015,
        };
        let r = ContextRamp::new(&s, c.saved);
        assert_eq!((r.context(), r.ceiling()), (c.ctx_now, c.ctx_max), "{:?}", (c.block, c.context, c.start, c.saved));
        assert_eq!(r.held_from().is_some(), c.was_above, "{:?}", (c.block, c.context, c.start, c.saved));
    }
}

#[test]
fn context_gain_matches_python() {
    let fx: ContextFixture = load("context.json");
    assert!(fx.gains.len() > 70);
    for (i, c) in fx.gains.iter().enumerate() {
        let s = RampSettings {
            chunk: c.chunk,
            passage: 1,
            block: u32::MAX,
            context_start: c.ctx_now,
            context_end: u32::MAX,
            context_step: 1,
            grow_every_chars: 1,
            check_every_chars: 1,
            gain_min: 0.0,
        };
        let mut r = ContextRamp::new(&s, None);
        assert_eq!(r.context(), c.ctx_now);
        for &(p, loss) in &c.samples {
            r.record(p, loss);
        }
        match (r.gain(), c.gain) {
            (None, None) => {}
            (Some(g), Some(w)) => assert!(close(g, w, REL), "gain case {i}: {g} vs {w}"),
            (g, w) => panic!("gain case {i}: {g:?} vs {w:?}"),
        }
    }
    assert!(fx.gains.iter().any(|c| c.gain.is_some()) && fx.gains.iter().any(|c| c.gain.is_none()));
}

#[test]
fn context_cadence_visit_and_schedule_helpers_match_python() {
    let fx: ContextFixture = load("context.json");
    for c in &fx.in_steps {
        assert_eq!(chars_to_steps(c.chars, c.chunk), c.steps, "chars {} chunk {}", c.chars, c.chunk);
    }
    for v in &fx.visits {
        assert_eq!(visit_chunks(v.ctx, v.chunk, v.passage), v.turn, "{:?}", (v.ctx, v.chunk, v.passage));
    }
    assert!(fx.ramps.len() > 300);
    for r in &fx.ramps {
        assert_eq!(
            ramp_schedule(r.step, r.total, r.start, r.end, r.warm, r.granularity),
            r.ctx,
            "{:?}",
            (r.step, r.total, r.start, r.end, r.warm, r.granularity)
        );
    }
}
