//! Whole-model tests on a tiny network (CPU, in-memory expert store).

use candle_core::Device;
use minagi_types::{ModelConfig, MoeMode, Preset};

use super::*;
use crate::moe::MemStore;
use crate::ops::testutil::max_abs_diff;

pub(crate) fn tiny_cfg() -> ModelConfig {
    let mut c = Preset::Tiny.model();
    c.d_model = 32;
    c.n_head = 2;
    c.d_ff = 64;
    c.n_prelude = 1;
    c.max_steps = 4;
    c.bptt_window = 4;
    c.train_steps_mean = 0.0;
    c.block = 64;
    c.pool_experts = 8;
    c.pool_max = 16;
    c.pool_d_ff = 16;
    c.pool_top_k = 2;
    c.halt_prior = 0.4;
    c
}

pub(crate) fn tiny_model(cfg: ModelConfig, resident: usize, mode: MoeMode, seed: u64) -> Model {
    let opts = ModelOpts { kernel: Kernel::plain(), capacity_factor: 0.0, moe_mode: mode, z_weight: 1e-3 };
    Model::create(&Device::Cpu, cfg, opts, resident, Box::new(MemStore::new()), seed).unwrap()
}

pub(crate) fn text(n: usize, mul: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * mul + 3) % 120 + 97) as u32 % 265).collect()
}

fn fwd(m: &mut Model, ids: &[u32], targets: Option<&[u32]>, train: bool, steps: usize) -> FwdOut {
    m.forward(FwdIn { ids, targets, caches: None, pos_offset: 0, n_steps: steps, train, want_logits: true }).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn a_fresh_model_scores_about_ln_vocab() {
    let mut m = tiny_model(tiny_cfg(), 4, MoeMode::DenseMasked, 1);
    let (x, y) = (text(48, 7), text(48, 11));
    let out = fwd(&mut m, &x, Some(&y), false, 4);
    let loss = crate::ops::scalar(out.loss.as_ref().unwrap()).unwrap();
    assert!((loss - (265f32).ln()).abs() < 0.15, "loss {loss}");
    assert_eq!(out.row_mass.len(), 4);
    assert!((out.row_mass.iter().sum::<f32>() - 1.0).abs() < 1e-4, "the halting distribution sums to one");
    assert_eq!(out.logits.unwrap().dims(), &[48, 265]);
}

#[test]
fn every_parameter_receives_a_finite_nonzero_gradient() {
    for mode in [MoeMode::DenseMasked, MoeMode::SparseDispatch] {
        let mut m = tiny_model(tiny_cfg(), 4, mode, 2);
        // larger weights so no gradient is accidentally ~0 at the tiny default init
        let (x, y) = (text(48, 7), text(48, 11));
        let out = fwd(&mut m, &x, Some(&y), true, 4);
        let total = (out.loss.unwrap() + out.aux.unwrap().affine(0.01, 0.0).unwrap()).unwrap();
        let g = total.backward().unwrap();
        for p in m.named_vars().iter().chain(
            m.pool
                .slot_vars()
                .iter()
                .map(|v| NamedVar { name: "slot".into(), var: (*v).clone(), group: Group::Pool })
                .collect::<Vec<_>>()
                .iter(),
        ) {
            let grad = g.get(&p.var).unwrap_or_else(|| panic!("no gradient for {} ({mode:?})", p.name));
            let v = host(grad);
            assert!(v.iter().all(|x| x.is_finite()), "{} has non-finite gradient", p.name);
            assert!(v.iter().any(|x| *x != 0.0), "{} has an all-zero gradient ({mode:?})", p.name);
        }
    }
}

#[test]
fn dense_masked_and_sparse_dispatch_give_the_same_loss_and_gradients() {
    let (x, y) = (text(48, 7), text(48, 11));
    let run = |mode| {
        let mut m = tiny_model(tiny_cfg(), 4, mode, 3);
        let out = fwd(&mut m, &x, Some(&y), true, 4);
        let loss = out.loss.unwrap();
        let l = crate::ops::scalar(&loss).unwrap();
        let g = loss.backward().unwrap();
        let grads: Vec<Vec<f32>> = m.named_vars().iter().map(|p| host(g.get(&p.var).unwrap())).collect();
        (l, grads)
    };
    let (la, ga) = run(MoeMode::DenseMasked);
    let (lb, gb) = run(MoeMode::SparseDispatch);
    assert!((la - lb).abs() < 1e-5, "{la} vs {lb}");
    for (i, (a, b)) in ga.iter().zip(&gb).enumerate() {
        let scale = b.iter().fold(1e-9f32, |m, x| m.max(x.abs()));
        let diff = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(diff / scale < 2e-3, "gradient {i}: {}", diff / scale);
    }
}

#[test]
fn with_every_expert_on_the_card_the_model_is_causal() {
    // (Admission lets the whole window vote for experts, so causality only holds when nothing can be left out.)
    let mut cfg = tiny_cfg();
    cfg.pool_top_k = cfg.pool_experts; // every token asks for every expert
    let mut m = tiny_model(cfg, 8, MoeMode::SparseDispatch, 4);
    let a = text(40, 7);
    let mut b = a.clone();
    for t in b[20..].iter_mut() {
        *t = (*t + 5) % 265;
    }
    let la = host(&fwd(&mut m, &a, None, false, 4).logits.unwrap());
    let lb = host(&fwd(&mut m, &b, None, false, 4).logits.unwrap());
    let v = 265;
    let diff = |lo: usize, hi: usize| {
        la[lo * v..hi * v].iter().zip(&lb[lo * v..hi * v]).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    };
    assert!(diff(0, 20) < 1e-5, "logits before the edit changed: {}", diff(0, 20));
    assert!(diff(20, 40) > 1e-4, "logits at the edit did not change");
}

#[test]
fn cached_chunks_equal_one_pass() {
    // Writing and measuring read a text chunk by chunk through the attention caches; it must equal one long pass.
    let mut cfg = tiny_cfg();
    cfg.pool_top_k = cfg.pool_experts;
    let mut m = tiny_model(cfg, 8, MoeMode::SparseDispatch, 5);
    // make halting non-trivial so the freeze-and-carry path is exercised: push the halt bias up
    m.trunk.halt_b.set(&Tensor::from_vec(vec![0.5f32], 1, &m.dev).unwrap()).unwrap();
    let (x, y) = (text(36, 7), text(36, 11));
    let whole = fwd(&mut m, &x, Some(&y), false, 4);
    let mut caches = m.new_caches();
    let mut parts = Vec::new();
    let mut pos = 0;
    let mut loss_rows = Vec::new();
    for len in [10usize, 7, 19] {
        let o = m
            .forward(FwdIn {
                ids: &x[pos..pos + len],
                targets: Some(&y[pos..pos + len]),
                caches: Some(&mut caches),
                pos_offset: pos,
                n_steps: 4,
                train: false,
                want_logits: true,
            })
            .unwrap();
        parts.push(host(&o.logits.unwrap()));
        loss_rows.push(o.row_mass);
        pos += len;
    }
    let chunked: Vec<f32> = parts.concat();
    let one = host(&whole.logits.unwrap());
    let diff = chunked.iter().zip(&one).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff < 2e-4, "chunked reading differs from one pass by {diff}");
    assert_eq!(caches.positions(), 36);
}

#[test]
fn writing_with_a_cache_matches_uncached_writing() {
    let mut cfg = tiny_cfg();
    cfg.pool_top_k = cfg.pool_experts;
    let mut m = tiny_model(cfg, 8, MoeMode::SparseDispatch, 6);
    m.trunk.halt_b.set(&Tensor::from_vec(vec![0.0f32], 1, &m.dev).unwrap()).unwrap();
    let x = text(30, 7);
    let full = fwd(&mut m, &x, None, false, 4);
    let full_last = host(&full.logits.unwrap())[29 * 265..30 * 265].to_vec();
    let mut caches = m.new_caches();
    let mut last = Vec::new();
    for (pos, chunk) in [(0usize, &x[..29]), (29, &x[29..30])] {
        let o = m
            .forward(FwdIn {
                ids: chunk,
                targets: None,
                caches: Some(&mut caches),
                pos_offset: pos,
                n_steps: 4,
                train: false,
                want_logits: false,
            })
            .unwrap();
        last = host(&o.logits.unwrap());
    }
    let d = last.iter().zip(&full_last).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(d < 2e-4, "{d}");
}

#[test]
fn halted_characters_cost_no_expert_work_and_stop_changing() {
    let mut cfg = tiny_cfg();
    cfg.pool_top_k = 2;
    let mut m = tiny_model(cfg, 4, MoeMode::SparseDispatch, 7);
    let (x, y) = (text(40, 7), text(40, 11));
    // pondering: nothing halts early, every row routes every character
    let slow = fwd(&mut m, &x, Some(&y), false, 4);
    assert!(slow.route.tokens >= 3 * 40, "rows run {}, tokens routed {}", slow.rows_run, slow.route.tokens);
    // eager halting: the halting head says "stop" immediately, so after row 1 almost nothing routes
    m.trunk.halt_b.set(&Tensor::from_vec(vec![6.0f32], 1, &m.dev).unwrap()).unwrap();
    let eager = fwd(&mut m, &x, Some(&y), false, 4);
    assert!(eager.route.tokens < slow.route.tokens / 2, "{} vs {}", eager.route.tokens, slow.route.tokens);
    assert!(eager.row_mass[0] > 0.9, "most of the mass halts at row 1: {:?}", eager.row_mass);
    assert!(eager.expected_rows() < 1.3);
    let written = fwd(&mut m, &x, None, false, 4);
    assert!(written.steps.iter().all(|&s| s == 1.0), "{:?}", written.steps);
    assert!(host(&written.logits.unwrap()).iter().all(|v| v.is_finite()));
}

#[test]
fn halting_probabilities_reach_the_halting_head() {
    let mut m = tiny_model(tiny_cfg(), 4, MoeMode::DenseMasked, 8);
    let (x, y) = (text(40, 7), text(40, 11));
    let out = fwd(&mut m, &x, Some(&y), true, 4);
    let g = out.loss.unwrap().backward().unwrap();
    let gb = host(g.get(&m.trunk.halt_b).unwrap());
    assert!(gb[0] != 0.0 && gb[0].is_finite());
}

#[test]
fn reading_past_the_position_tables_is_a_clear_error() {
    let mut m = tiny_model(tiny_cfg(), 4, MoeMode::DenseMasked, 9);
    let x = text(65, 7);
    let err = m.forward(FwdIn {
        ids: &x,
        targets: None,
        caches: None,
        pos_offset: 0,
        n_steps: 2,
        train: false,
        want_logits: false,
    });
    assert!(err.err().unwrap().to_string().contains("position tables"));
}

#[test]
fn depth_sampling_follows_the_config() {
    let mut cfg = tiny_cfg();
    let mut rng = HostRng::new(1);
    assert_eq!(sample_depth(&cfg, true, &mut rng), 4, "mean 0 always runs the ceiling");
    cfg.train_steps_mean = 2.0;
    cfg.max_steps = 6;
    let draws: Vec<usize> = (0..400).map(|_| sample_depth(&cfg, true, &mut rng)).collect();
    assert!(draws.iter().all(|&n| (1..=6).contains(&n)));
    let mean = draws.iter().sum::<usize>() as f64 / 400.0;
    assert!((2.4..3.4).contains(&mean), "mean {mean}");
    assert_eq!(sample_depth(&cfg, false, &mut rng), 6, "evaluation and writing use the ceiling");
}

#[test]
fn growth_and_pruning_keep_router_gate_and_optimiser_in_step() {
    let mut m = tiny_model(tiny_cfg(), 4, MoeMode::DenseMasked, 10);
    let mut opt = AdamW::new(Default::default(), 0.1);
    // give the router and gate some moments so there is state to carry
    opt.put_moments(
        ROUTER_NAME,
        Tensor::ones((8, 32), candle_core::DType::F32, &m.dev).unwrap(),
        Tensor::ones((8, 32), candle_core::DType::F32, &m.dev).unwrap(),
    );
    opt.put_moments(
        GATE_NAME,
        Tensor::ones(8, candle_core::DType::F32, &m.dev).unwrap(),
        Tensor::ones(8, candle_core::DType::F32, &m.dev).unwrap(),
    );
    let mut rng = HostRng::new(2);
    let lin = m.grow_pool(2, Some(3), 100.0, 0.001, &mut opt, &mut rng).unwrap();
    assert_eq!(lin.len(), 2);
    assert_eq!(m.pool.n_experts(), 10);
    assert_eq!(m.trunk.router.dims(), &[10, 32]);
    assert_eq!(opt.moments(ROUTER_NAME).unwrap().0.dims(), &[10, 32]);
    assert_eq!(opt.moments(GATE_NAME).unwrap().0.dims(), &[10]);
    // a newborn's router row is the unit-weighted average of its parents' rows
    let router = m.trunk.router.to_vec2::<f32>().unwrap();
    let l = lin[0].as_ref().unwrap();
    for (c, got) in router[8].iter().enumerate() {
        let want: f32 = l.parents.iter().zip(&l.share).map(|(&p, &s)| s * router[p][c]).sum();
        assert!((got - want).abs() < 1e-5);
    }
    // the grown model still runs
    let (x, y) = (text(40, 7), text(40, 11));
    let out = fwd(&mut m, &x, Some(&y), true, 4);
    assert!(crate::ops::scalar(out.loss.as_ref().unwrap()).unwrap().is_finite());
    // prune: every expert was asked for recently, so nothing goes
    m.pool.book.segments = 1000;
    m.pool.book.born = vec![0.0; 10];
    m.pool.book.last_seen = vec![1000.0; 10];
    assert!(m.prune_pool(1000.0, 100.0, 0, &mut opt).unwrap().is_none());
    // make experts 8 and 9 stale and unprotected
    m.pool.book.last_seen[8] = 0.0;
    m.pool.book.last_seen[9] = 0.0;
    m.pool.book.slots = vec![-1; 4];
    let keep = m.prune_pool(1000.0, 100.0, 0, &mut opt).unwrap().unwrap();
    assert_eq!(keep.len(), 8);
    assert_eq!(m.trunk.router.dims(), &[8, 32]);
    assert_eq!(opt.moments(ROUTER_NAME).unwrap().0.dims(), &[8, 32]);
    assert_eq!(m.pool.gate_values().unwrap().len(), 8);
    m.pool.check().unwrap();
}

#[test]
fn a_shift_of_the_whole_window_changes_nothing_but_position() {
    // Sanity: rotary positions are relative, so reading the same text at pos_offset 0 in a fresh cache is the same
    // computation whatever the absolute table position (here compared through the cached path with offset 0 vs one pass).
    let mut cfg = tiny_cfg();
    cfg.pool_top_k = cfg.pool_experts;
    let mut m = tiny_model(cfg, 8, MoeMode::SparseDispatch, 11);
    let x = text(20, 7);
    let a = host(&fwd(&mut m, &x, None, false, 4).logits.unwrap());
    let mut c = m.new_caches();
    let b = host(
        &m.forward(FwdIn {
            ids: &x,
            targets: None,
            caches: Some(&mut c),
            pos_offset: 0,
            n_steps: 4,
            train: false,
            want_logits: false,
        })
        .unwrap()
        .logits
        .unwrap(),
    );
    assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-5));
    let _ = max_abs_diff;
}

/// Gradients of one training step by the ordinary path and by row checkpointing, on identical models.
#[allow(clippy::type_complexity)]
fn grads_both_ways(
    cfg: ModelConfig,
    mode: MoeMode,
    resident: usize,
    steps: usize,
    explore: bool,
) -> (f64, Vec<(String, Vec<f32>, Vec<f32>)>) {
    let (x, y) = (text(48, 7), text(48, 11));
    let make = || {
        let mut m = tiny_model(cfg.clone(), resident, mode, 5);
        // non-trivial halting so rows go quiet part-way
        m.trunk.halt_b.set(&Tensor::from_vec(vec![-0.4f32], 1, &m.dev).unwrap()).unwrap();
        if explore {
            m.pool.book.explore_bias = 0.65;
        }
        m
    };
    let mut a = make();
    let out = a
        .forward(FwdIn {
            ids: &x,
            targets: Some(&y),
            caches: None,
            pos_offset: 0,
            n_steps: steps,
            train: true,
            want_logits: false,
        })
        .unwrap();
    let loss_a = f64::from(crate::ops::scalar(out.loss.as_ref().unwrap()).unwrap());
    let total = (out.loss.unwrap() + out.aux.unwrap().affine(0.01, 0.0).unwrap()).unwrap();
    let ga = total.backward().unwrap();
    let mut b = make();
    let (co, gb) = b.forward_backward(&x, &y, steps, 0.01).unwrap();
    assert!((co.loss - loss_a).abs() < 1e-4 * loss_a.abs().max(1.0), "loss {} vs {loss_a}", co.loss);
    let mut out = Vec::new();
    for (va, vb) in a.all_vars().iter().zip(b.all_vars().iter()) {
        let name = format!("{:?}", va.dims());
        out.push((
            name,
            host(ga.get(va).unwrap()),
            host(gb.get(vb).unwrap_or_else(|| panic!("no checkpointed gradient for a {:?} tensor", vb.dims()))),
        ));
    }
    (co.loss, out)
}

#[test]
fn row_checkpointing_gives_the_same_gradients_as_the_ordinary_step() {
    for (mode, resident, steps, explore) in [
        (MoeMode::DenseMasked, 4, 4, false),
        (MoeMode::SparseDispatch, 4, 4, true),
        (MoeMode::SparseDispatch, 8, 3, false),
    ] {
        let mut cfg = tiny_cfg();
        cfg.max_steps = 4;
        let (_, pairs) = grads_both_ways(cfg, mode, resident, steps, explore);
        for (name, a, b) in pairs {
            let scale = a.iter().fold(1e-9f32, |m, x| m.max(x.abs()));
            let diff = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                diff / scale < 2e-3,
                "{mode:?} resident {resident} steps {steps} explore {explore}: gradient of {name} differs by {}",
                diff / scale
            );
        }
    }
}

#[test]
fn row_checkpointing_respects_backprop_truncation_and_min_rows() {
    let mut cfg = tiny_cfg();
    cfg.bptt_window = 2;
    cfg.min_steps = 2;
    let (_, pairs) = grads_both_ways(cfg, MoeMode::SparseDispatch, 4, 4, false);
    for (name, a, b) in pairs {
        let scale = a.iter().fold(1e-9f32, |m, x| m.max(x.abs()));
        let diff = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(diff / scale < 2e-3, "bptt/min_steps: gradient of {name} differs by {}", diff / scale);
    }
}

#[test]
fn a_checkpointed_step_trains_as_well_as_an_ordinary_one() {
    use crate::optim::adamw::AdamHyper;
    use crate::train::step::{StepSettings, train_on_window};
    let run = |ckpt: bool| {
        let mut m = tiny_model(tiny_cfg(), 4, MoeMode::SparseDispatch, 1);
        let mut opt = AdamW::new(AdamHyper::default(), 0.0);
        let s = StepSettings { lr: 3e-3, trunk_lr_mult: 0.5, clip: 1.0, pool_aux: 0.01, row_ckpt: ckpt };
        let (x, y) = (text(48, 7), text(48, 11));
        let mut last = 0.0;
        for _ in 0..40 {
            last = train_on_window(&mut m, &mut opt, &x, &y, 4, 1.0, &s).unwrap().loss;
        }
        last
    };
    let (a, b) = (run(false), run(true));
    assert!(a < 3.5 && b < 3.5, "both learn: {a} {b}");
    assert!((a - b).abs() < 1e-3, "the same trajectory: {a} {b}");
}
