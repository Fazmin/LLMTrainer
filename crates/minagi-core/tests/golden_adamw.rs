//! Parity of a whole paged training step (forward, backward, clipping, AdamW with experts paged in and out of the card)
//! with the original Python implementation: four real steps of the paged tiny model, teacher-forced through the
//! expert-admission decisions the reference made.

#![allow(clippy::type_complexity, clippy::too_many_arguments)]

mod golden_support;

use golden_support::*;
use minagi_core::candle_core::{Device, Tensor, Var};
use minagi_core::model::{FwdIn, Model, ModelOpts, Trunk};
use minagi_core::moe::{ExpertEntry, ExpertStore, PagedPool, TiersStore};
use minagi_core::ops::Kernel;
use minagi_core::optim::adamw::{AdamHyper, AdamW};
use minagi_core::store::checkpoint::{NamedTensors, Tensor as HostTensor};
use minagi_core::store::tiers::{TierConfig, Tiers};
use minagi_core::train::step::{StepSettings, train_on_window};
use minagi_types::{ModelConfig, MoeMode, Preset};

const TRUNK: [(&str, &[usize]); 16] = [
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
    ("adapter.weight", &[32, 64]),
    ("ln_f.weight", &[32]),
    ("halt.weight", &[1, 32]),
    ("halt.bias", &[1]),
];

fn pooled_config() -> ModelConfig {
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
    c.pool_experts = 8;
    c.pool_max = 8;
    c.pool_d_ff = 16;
    c.pool_depth = 1;
    c.pool_top_k = 2;
    c
}

fn build(f: &Npz, dir: &std::path::Path, mode: MoeMode, kernel: Kernel) -> Model {
    let dev = Device::Cpu;
    let cfg = pooled_config();
    let mut core = NamedTensors::new();
    for (name, shape) in TRUNK {
        core.insert(name, HostTensor::new(shape.to_vec(), f.f32(&format!("init/{name}"))).unwrap());
    }
    let mut routers = NamedTensors::new();
    routers.insert(
        "recur.0.mlp.router.weight",
        HostTensor::new(vec![8, 32], f.f32("init/recur.0.mlp.router.weight")).unwrap(),
    );
    routers.insert("recur.0.mlp.depth_emb", HostTensor::new(vec![32], f.f32("init/recur.0.mlp.depth_emb")).unwrap());
    routers.insert("pool.gate", HostTensor::new(vec![8], f.f32("init/pool.gate")).unwrap());
    let trunk = Trunk::load(&cfg, &core, &routers, &dev).unwrap();
    // the experts' files, behind a RAM tier of three, exactly as the reference ran
    let tiers = Tiers::open(TierConfig::new(dir, 32, 16).ram_capacity(3)).unwrap();
    let mut store = TiersStore::new(tiers);
    let (w1, w3, w2) = (f.f32("expert/w1"), f.f32("expert/w3"), f.f32("expert/w2"));
    for e in 0..8 {
        store
            .put(
                e as u64,
                ExpertEntry {
                    w1: w1[e * 512..(e + 1) * 512].to_vec(),
                    w3: w3[e * 512..(e + 1) * 512].to_vec(),
                    w2: w2[e * 512..(e + 1) * 512].to_vec(),
                    moments: None,
                },
                true,
            )
            .unwrap();
    }
    store.flush().unwrap();
    let mut pool = PagedPool::new(&dev, Box::new(store), 32, 16, 8, 4).unwrap();
    pool.gate = Var::from_tensor(&Tensor::from_vec(f.f32("init/pool.gate"), 8, &dev).unwrap()).unwrap();
    pool.book.explore_bias = 0.65;
    pool.book.explore_steps = 1000.0;
    let opts = ModelOpts { kernel, capacity_factor: 1.5, moe_mode: mode, z_weight: 1e-3 };
    Model::assemble(&dev, cfg, opts, trunk, pool).unwrap()
}

fn slots_of(f: &Npz, key: &str) -> Vec<i64> {
    f.i64(key)
}

#[test]
fn four_paged_training_steps_match_the_reference() {
    let f = Npz::load("adamw_paged.npz");
    let cfg_json = json("adamw_paged.json");
    let cj = &cfg_json["config"];
    let settings = StepSettings {
        lr: cj["lr_pool"].as_f64().unwrap(),
        trunk_lr_mult: cj["trunk_lr_mult"].as_f64().unwrap(),
        clip: cj["clip"].as_f64().unwrap(),
        pool_aux: cj["pool_aux_weight"].as_f64().unwrap(),
        row_ckpt: false,
    };
    for (mode, kernel) in [(MoeMode::SparseDispatch, Kernel::hand()), (MoeMode::DenseMasked, Kernel::plain())] {
        let dir = tempfile::tempdir().unwrap();
        let mut m = build(&f, dir.path(), mode, kernel);
        let mut opt =
            AdamW::new(AdamHyper { beta1: 0.9, beta2: 0.95, eps: 1e-8 }, cj["weight_decay"].as_f64().unwrap());
        let trainable: Vec<(String, Var)> = m
            .named_vars()
            .into_iter()
            .map(|p| (p.name, p.var))
            .chain([
                ("pool.w1".to_string(), m.pool.w1.clone()),
                ("pool.w3".to_string(), m.pool.w3.clone()),
                ("pool.w2".to_string(), m.pool.w2.clone()),
            ])
            .collect();
        for k in 0..4 {
            let tag = format!("{mode:?} step {k}");
            if k == 3 {
                // two held-out forwards between steps 2 and 3: the router's own choice, no learning
                for i in 0..2 {
                    let (x, y) = (f.ids(&format!("eval{i}/idx")), f.ids(&format!("eval{i}/targets")));
                    m.forward(FwdIn {
                        ids: &x,
                        targets: Some(&y),
                        caches: None,
                        pos_offset: 0,
                        n_steps: 4,
                        train: false,
                        want_logits: false,
                    })
                    .unwrap();
                    assert_eq!(
                        m.pool.book.slots,
                        slots_of(&f, &format!("eval{i}/slots_after")),
                        "{tag}: card after held-out forward {i}"
                    );
                }
            }
            // steer the exploration bonus to the set of experts the reference admitted
            m.pool.book.recent = f.f32(&format!("s{k}/recent_set")).into_iter().map(f64::from).collect();
            assert_eq!(m.pool.book.slots, slots_of(&f, &format!("s{k}/slots_before")), "{tag}: card before");
            let (x, y) = (f.ids(&format!("s{k}/idx")), f.ids(&format!("s{k}/targets")));
            let out = train_on_window(&mut m, &mut opt, &x, &y, 4, 1.0, &settings).unwrap();
            assert_eq!(m.pool.book.slots, slots_of(&f, &format!("s{k}/slots")), "{tag}: card used");
            let want_loss = f.scalar(&format!("s{k}/loss_model")) as f64;
            assert!(
                (out.loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0),
                "{tag}: loss {} vs {want_loss}",
                out.loss
            );
            let want_norm = f.scalar(&format!("s{k}/grad_norm")) as f64;
            assert!(
                (out.grad_norm - want_norm).abs() < 2e-3 * want_norm.max(1.0),
                "{tag}: grad norm {} vs {want_norm}",
                out.grad_norm
            );
            for (name, var) in &trainable {
                close_grad(
                    &format!("{tag}: param {name}"),
                    &host(var.as_tensor()),
                    &f.f32(&format!("s{k}/param/{name}")),
                );
            }
            // the moments of the slot tensors, per expert on the card
            let (m_avg, m_sq) = m.pool.slot_moments();
            for (i, name) in ["pool.w1", "pool.w3", "pool.w2"].into_iter().enumerate() {
                close_grad(
                    &format!("{tag}: exp_avg {name}"),
                    &host(&m_avg[i]),
                    &f.f32(&format!("s{k}/exp_avg/{name}")),
                );
                close_grad(
                    &format!("{tag}: exp_avg_sq {name}"),
                    &host(&m_sq[i]),
                    &f.f32(&format!("s{k}/exp_avg_sq/{name}")),
                );
            }
        }
        // the non-slot moments after the last step
        for (name, _) in &trainable {
            if name.starts_with("pool.w") {
                continue;
            }
            let (a, b) = opt.moments(name).unwrap_or_else(|| panic!("no moments for {name}"));
            close_grad(&format!("{mode:?}: exp_avg {name}"), &host(a), &f.f32(&format!("s3/exp_avg/{name}")));
            close_grad(&format!("{mode:?}: exp_avg_sq {name}"), &host(b), &f.f32(&format!("s3/exp_avg_sq/{name}")));
        }
        // flush: every expert's file holds its final weights, and moments rounded to bf16
        m.pool.flush().unwrap();
        for e in 0..8u64 {
            let key = format!("file/{e}/w1");
            if !f.has(&key) {
                continue;
            }
            let entry = m.pool.store_mut().fetch(e, false).unwrap();
            close_grad(&format!("{mode:?}: file {e} w1"), &entry.w1, &f.f32(&key));
            if let (Some(mm), true) = (entry.moments, f.has(&format!("file/{e}/w1_m"))) {
                close(&format!("{mode:?}: file {e} w1_m"), &mm.w1_m, &f.f32(&format!("file/{e}/w1_m")), 1e-7, 1e-2);
                close(&format!("{mode:?}: file {e} w2_v"), &mm.w2_v, &f.f32(&format!("file/{e}/w2_v")), 1e-9, 1e-2);
            }
        }
    }
}
