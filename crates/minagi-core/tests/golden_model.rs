//! Parity of the whole recurrent model (halting, freezing, the PonderNet loss, caches) with the original Python
//! implementation.
//!
//! The reference fixtures use a *dense* recurrent MLP; this engine always routes the recurrent block through the expert
//! pool. A pool with one expert, top-1 routing, gate 1 and every character admitted computes exactly the same function
//! (the single expert is the dense MLP), so the dense fixtures apply unchanged: the dense MLP's `w1/w3/w2` become that
//! expert's weights.

#![allow(clippy::type_complexity, clippy::too_many_arguments)]

mod golden_support;

use golden_support::*;
use minagi_core::candle_core::{Device, Tensor};
use minagi_core::model::{Caches, FwdIn, Model, ModelOpts, Trunk};
use minagi_core::moe::{ExpertEntry, ExpertStore, MemStore, PagedPool};
use minagi_core::ops::Kernel;
use minagi_core::store::checkpoint::{NamedTensors, Tensor as HostTensor};
use minagi_types::{ModelConfig, MoeMode, Preset};
use serde_json::Value;

const PARAMS: [(&str, &[usize]); 19] = [
    ("tok_emb.weight", &[265, 32]),
    ("prelude.0.ln1.weight", &[32]),
    ("prelude.0.attn.qkv.weight", &[96, 32]),
    ("prelude.0.attn.proj.weight", &[32, 32]),
    ("prelude.0.ln2.weight", &[32]),
    ("prelude.0.mlp.w1.weight", &[64, 32]),
    ("prelude.0.mlp.w3.weight", &[64, 32]),
    ("prelude.0.mlp.w2.weight", &[32, 64]),
    ("recur.0.ln1.weight", &[32]),
    ("recur.0.attn.qkv.weight", &[96, 32]),
    ("recur.0.attn.proj.weight", &[32, 32]),
    ("recur.0.ln2.weight", &[32]),
    ("recur.0.mlp.w1.weight", &[64, 32]),
    ("recur.0.mlp.w3.weight", &[64, 32]),
    ("recur.0.mlp.w2.weight", &[32, 64]),
    ("adapter.weight", &[32, 64]),
    ("ln_f.weight", &[32]),
    ("halt.weight", &[1, 32]),
    ("halt.bias", &[1]),
];

/// The model config the fixtures use, with a variant's overrides applied.
fn config(over: &Value) -> ModelConfig {
    let mut c = Preset::Tiny.model();
    c.vocab_size = 265;
    c.d_model = 32;
    c.n_head = 2;
    c.d_ff = 64;
    c.n_prelude = 1;
    c.n_recur = 1;
    c.n_coda = 0;
    c.max_steps = 4;
    c.min_steps = 1;
    c.halt_prior = 0.4;
    c.halt_thresh = 0.9;
    c.halt_freeze = true;
    c.ponder_beta = 0.01;
    c.bptt_window = 4;
    c.train_steps_mean = 0.0;
    c.block = 64;
    c.pool_experts = 1;
    c.pool_max = 1;
    c.pool_d_ff = 64;
    c.pool_depth = 1;
    c.pool_top_k = 1;
    if let Some(o) = over.as_object() {
        for (k, v) in o {
            match k.as_str() {
                "bptt_window" => c.bptt_window = v.as_u64().unwrap() as u32,
                "min_steps" => c.min_steps = v.as_u64().unwrap() as u32,
                "halt_freeze" => c.halt_freeze = v.as_bool().unwrap(),
                other => panic!("unhandled config override {other}"),
            }
        }
    }
    c
}

/// A model whose weights come from `weight(name)`.
fn build(weight: &dyn Fn(&str) -> Vec<f32>, cfg: ModelConfig, mode: MoeMode, kernel: Kernel) -> Model {
    let dev = Device::Cpu;
    let mut core = NamedTensors::new();
    for (name, shape) in PARAMS {
        if !name.starts_with("recur.0.mlp") {
            core.insert(name, HostTensor::new(shape.to_vec(), weight(name)).unwrap());
        }
    }
    let mut routers = NamedTensors::new();
    routers.insert("recur.0.mlp.router.weight", HostTensor::new(vec![1, 32], vec![0.1; 32]).unwrap());
    routers.insert("recur.0.mlp.depth_emb", HostTensor::zeros(vec![32]));
    routers.insert("pool.gate", HostTensor::new(vec![1], vec![1.0]).unwrap());
    let trunk = Trunk::load(&cfg, &core, &routers, &dev).unwrap();
    let mut store = MemStore::new();
    store
        .put(
            0,
            ExpertEntry {
                w1: weight("recur.0.mlp.w1.weight"),
                w3: weight("recur.0.mlp.w3.weight"),
                w2: weight("recur.0.mlp.w2.weight"),
                moments: None,
            },
            false,
        )
        .unwrap();
    let pool = PagedPool::new(&dev, Box::new(store), 32, 64, 1, 1).unwrap();
    let opts = ModelOpts { kernel, capacity_factor: 0.0, moe_mode: mode, z_weight: 1e-3 };
    Model::assemble(&dev, cfg, opts, trunk, pool).unwrap()
}

/// Which stored weights a variant uses: `(variant that holds them, optional override variant for the halting head)`.
fn weights_of<'a>(f: &'a Npz, variant: &str) -> impl Fn(&str) -> Vec<f32> + 'a {
    let base = match variant {
        "default_init" | "perturbed" | "halting" => variant.to_string(),
        _ => "perturbed".to_string(),
    };
    let halting_all = variant == "halting_all";
    move |name: &str| {
        if halting_all && name.starts_with("halt.") {
            f.f32(&format!("halting_all/{name}"))
        } else {
            f.f32(&format!("{base}/{name}"))
        }
    }
}

fn host_param(m: &Model, name: &str) -> Option<Tensor> {
    match name {
        "recur.0.mlp.w1.weight" => Some(m.pool.w1.as_tensor().clone()),
        "recur.0.mlp.w3.weight" => Some(m.pool.w3.as_tensor().clone()),
        "recur.0.mlp.w2.weight" => Some(m.pool.w2.as_tensor().clone()),
        _ => m.trunk.named().into_iter().find(|p| p.name == name).map(|p| p.var.as_tensor().clone()),
    }
}

fn load_expert(m: &mut Model) {
    // put the single expert on the card, as the first forward will; nothing else to do
    let _ = m;
}

#[test]
fn the_training_forward_loss_and_gradients_match_for_every_variant() {
    let f = Npz::load("dense_forward.npz");
    let j = json("dense_forward.json");
    let ids = f.ids("idx");
    let targets = f.ids("targets");
    let variants = ["default_init", "perturbed", "halting", "halting_all", "bptt", "min_steps2", "n2", "no_freeze"];
    for variant in variants {
        let vj = &j["variants"][variant];
        let n_steps = vj["n_steps"].as_u64().unwrap() as usize;
        for mode in [MoeMode::DenseMasked, MoeMode::SparseDispatch] {
            for (kname, kernel) in [("composed", Kernel::plain()), ("hand", Kernel::hand())] {
                let w = weights_of(&f, variant);
                let mut m = build(&w, config(&vj["cfg_overrides"]), mode, kernel);
                load_expert(&mut m);
                let out = m
                    .forward(FwdIn {
                        ids: &ids,
                        targets: Some(&targets),
                        caches: None,
                        pos_offset: 0,
                        n_steps,
                        train: true,
                        want_logits: true,
                    })
                    .unwrap();
                let tag = format!("{variant}/{mode:?}/{kname}");
                let loss = out.loss.clone().unwrap();
                let got = host(&loss)[0];
                let want = vj["loss"].as_f64().unwrap() as f32;
                assert!((got - want).abs() < 1e-4 * want.abs().max(1.0), "{tag}: loss {got} vs {want}");
                if vj["mixture_logits_stored"].as_bool().unwrap_or(false) {
                    close_fwd(
                        &format!("{tag} mixture logits"),
                        &host(&out.logits.clone().unwrap()),
                        &f.f32(&format!("{variant}/mixture_logits")),
                    );
                }
                // the expected number of rows per character
                let want_rows = vj["last_steps"].as_f64().unwrap() as f32;
                assert!(
                    (out.expected_rows() - want_rows).abs() < 1e-3,
                    "{tag}: rows {} vs {want_rows}",
                    out.expected_rows()
                );
                if vj["grads_stored"].as_bool().unwrap_or(false) {
                    let grads = loss.backward().unwrap();
                    for (name, _) in PARAMS {
                        let var = if name.starts_with("recur.0.mlp") {
                            match name {
                                "recur.0.mlp.w1.weight" => &m.pool.w1,
                                "recur.0.mlp.w3.weight" => &m.pool.w3,
                                _ => &m.pool.w2,
                            }
                            .clone()
                        } else {
                            m.trunk.named().into_iter().find(|p| p.name == name).unwrap().var
                        };
                        let g = grads.get(&var).unwrap_or_else(|| panic!("{tag}: no gradient for {name}"));
                        close_grad(&format!("{tag} grad {name}"), &host(g), &f.f32(&format!("{variant}/grad/{name}")));
                    }
                }
                let _ = host_param(&m, "tok_emb.weight");
            }
        }
    }
}

#[test]
fn bptt_truncation_changes_the_gradients_but_not_the_loss() {
    let f = Npz::load("dense_forward.npz");
    let j = json("dense_forward.json");
    let (ids, targets) = (f.ids("idx"), f.ids("targets"));
    let run = |over: &Value| {
        let w = weights_of(&f, "perturbed");
        let mut m = build(&w, config(over), MoeMode::SparseDispatch, Kernel::plain());
        let out = m
            .forward(FwdIn {
                ids: &ids,
                targets: Some(&targets),
                caches: None,
                pos_offset: 0,
                n_steps: 4,
                train: true,
                want_logits: false,
            })
            .unwrap();
        let loss = out.loss.unwrap();
        let g = loss.backward().unwrap();
        (host(&loss)[0], host(g.get(&m.trunk.adapter).unwrap()))
    };
    let (l_full, g_full) = run(&Value::Null);
    let (l_cut, g_cut) = run(&j["variants"]["bptt"]["cfg_overrides"]);
    assert!((l_full - l_cut).abs() < 1e-5);
    let diff = g_full.iter().zip(&g_cut).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff > 1e-4, "detaching h must change the adapter's gradient: {diff}");
}

#[test]
fn writing_matches_one_shot_and_chunked_through_the_caches() {
    let f = Npz::load("dense_forward.npz");
    let inf = Npz::load("dense_infer.npz");
    let j = json("dense_infer.json");
    let ids = inf.ids("idx");
    for (sc, variant) in [("perturbed", "perturbed"), ("halting", "halting"), ("allhalt", "allhalt")] {
        let w: Box<dyn Fn(&str) -> Vec<f32>> = if variant == "allhalt" {
            let f2 = &f;
            Box::new(move |name: &str| {
                if name.starts_with("halt.") {
                    f2.f32(&format!("halting_all/{name}"))
                } else {
                    f2.f32(&format!("perturbed/{name}"))
                }
            })
        } else {
            let f2 = &f;
            let v = variant.to_string();
            Box::new(move |name: &str| f2.f32(&format!("{v}/{name}")))
        };
        for mode in [MoeMode::DenseMasked, MoeMode::SparseDispatch] {
            let mut m = build(&*w, config(&Value::Null), mode, Kernel::hand());
            // one shot
            let out = m
                .forward(FwdIn {
                    ids: &ids,
                    targets: None,
                    caches: None,
                    pos_offset: 0,
                    n_steps: 4,
                    train: false,
                    want_logits: false,
                })
                .unwrap();
            let tag = format!("{sc}/{mode:?}");
            close_fwd(
                &format!("{tag} one-shot logits"),
                &host(&out.logits.unwrap()),
                &inf.f32(&format!("{sc}/oneshot/logits")),
            );
            assert_eq!(out.steps, inf.f32(&format!("{sc}/oneshot/steps")), "{tag}: rows used per token");
            // chunked, through the per-row caches
            let mut caches: Caches = m.new_caches();
            for (i, c) in j["scenarios"][sc]["chunks"].as_array().unwrap().iter().enumerate() {
                let (s, e, off) = (
                    c["start"].as_u64().unwrap() as usize,
                    c["end"].as_u64().unwrap() as usize,
                    c["offset"].as_u64().unwrap() as usize,
                );
                let o = m
                    .forward(FwdIn {
                        ids: &ids[s..e],
                        targets: None,
                        caches: Some(&mut caches),
                        pos_offset: off,
                        n_steps: 4,
                        train: false,
                        want_logits: false,
                    })
                    .unwrap();
                close_fwd(
                    &format!("{tag} chunk{i} logits"),
                    &host(&o.logits.unwrap()),
                    &inf.f32(&format!("{sc}/chunk{i}/logits")),
                );
                assert_eq!(o.steps, inf.f32(&format!("{sc}/chunk{i}/steps")), "{tag}: chunk {i} rows");
            }
            assert_eq!(caches.positions(), 16, "{tag}: every slot holds every position, even rows that were skipped");
        }
    }
}
