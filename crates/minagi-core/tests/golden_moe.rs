//! Parity of the expert pool (admission, routing, dispatch, the balancing loss) with the original Python implementation.

#![allow(clippy::type_complexity, clippy::too_many_arguments)]

mod golden_support;

use golden_support::*;
use minagi_core::candle_core::{DType, Device, Tensor, Var};
use minagi_core::moe::{ExpertEntry, ExpertStore, MemStore, PagedPool, PoolBook, RouteCfg};
use minagi_core::ops::Kernel;
use minagi_types::MoeMode;
use serde_json::Value;

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn idxs(v: &Value) -> Vec<usize> {
    let mut o: Vec<usize> = v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    o.sort();
    o
}

fn close64(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-5 + 1e-5 * w.abs(), "{what}[{i}]: {g} vs {w}");
    }
}

#[test]
fn admission_replays_every_scenario_state_by_state() {
    let j = json("moe_admission.json");
    for sc in j["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let c = &sc["config"];
        let mut book =
            PoolBook::new(c["n_experts"].as_u64().unwrap() as usize, c["resident"].as_u64().unwrap() as usize);
        book.explore_bias = c["explore_bias"].as_f64().unwrap();
        book.explore_steps = c["explore_steps"].as_f64().unwrap();
        for (i, step) in sc["steps"].as_array().unwrap().iter().enumerate() {
            let tag = format!("{name} step {i} ({})", step["op"]);
            match step["op"].as_str().unwrap() {
                "init" => {}
                "begin_text" => book.begin_text(step["explore"].as_bool().unwrap()),
                "begin_forward" => book.begin_forward(step["explore"].as_bool().unwrap()),
                "set_recent" => book.recent = f64s(&step["values"]),
                "set_explore" => {
                    book.explore_bias = step["explore_bias"].as_f64().unwrap();
                    book.explore_steps = step["explore_steps"].as_f64().unwrap();
                }
                "admit" => {
                    let mass = f32s(&step["mass"]);
                    let merit = (!step["merit"].is_null()).then(|| f32s(&step["merit"]));
                    let before = book.loads;
                    if let Some(plan) = book.admit(&mass, merit.as_deref()) {
                        book.set_slots(plan.new);
                    }
                    assert_eq!(book.loads - before, step["returned_loads"].as_u64().unwrap(), "{tag}: loads returned");
                }
                other => panic!("unknown op {other}"),
            }
            let s = &step["state"];
            let slots: Vec<i64> = s["slots"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
            assert_eq!(book.slots, slots, "{tag}: slots");
            assert_eq!(book.admitted(), idxs(&s["admitted"]), "{tag}: admitted");
            assert_eq!(book.merited(), idxs(&s["merited"]), "{tag}: merited");
            assert_eq!(book.admitting(), s["admitting"].as_bool().unwrap(), "{tag}: admitting");
            let mask: Vec<bool> = s["admitted_mask"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect();
            assert_eq!(book.admitted_mask(), mask, "{tag}: admitted mask");
            close64(&format!("{tag}: last_seen"), &book.last_seen, &f64s(&s["last_seen"]));
            close64(&format!("{tag}: admits"), &book.admits, &f64s(&s["admits"]));
            let ever: Vec<bool> = s["ever"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect();
            assert_eq!(book.ever, ever, "{tag}: ever");
            assert_eq!(book.loads, s["loads"].as_u64().unwrap(), "{tag}: loads");
            assert_eq!(book.swaps, s["swaps"].as_u64().unwrap(), "{tag}: swaps");
            assert_eq!(book.segments, s["segments"].as_u64().unwrap(), "{tag}: segments");
            close64(&format!("{tag}: recent"), &book.recent, &f64s(&s["recent"]));
            match (book.selection_bias(), &s["bias"]) {
                (None, Value::Null) => {}
                (Some(got), want) if !want.is_null() => {
                    let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
                    close64(&format!("{tag}: bias"), &got64, &f64s(want));
                }
                (got, want) => panic!("{tag}: bias {got:?} vs {want}"),
            }
        }
    }
}

#[test]
fn staleness_reads_like_the_reference() {
    let j = json("moe_admission.json");
    for c in j["dying_cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let born = f64s(&c["born"]);
        let mut book = PoolBook::new(born.len(), 4);
        book.born = born;
        book.last_seen = f64s(&c["last_seen"]);
        book.segments = c["segments"].as_u64().unwrap();
        book.now = c["now"].as_f64().unwrap();
        book.trial = c["trial"].as_f64().unwrap();
        book.dying_at = c["dying_at"].as_f64().unwrap();
        let got = book.dying();
        close64(&format!("dying {name}"), &got, &f64s(&c["dying"]));
        let idle = got.iter().filter(|&&d| d >= book.dying_at).count();
        assert_eq!(idle as u64, c["saturation_idle"].as_u64().unwrap(), "{name}: idle experts");
    }
}

/// A pool of the fixture's 8 experts (the card empty), with `gate` as given.
fn paged_pool(f: &Npz, gate: Vec<f32>) -> PagedPool {
    let (w1, w3, w2) = (f.f32("expert/w1"), f.f32("expert/w3"), f.f32("expert/w2"));
    let mut store = MemStore::new();
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
                false,
            )
            .unwrap();
    }
    let mut p = PagedPool::new(&Device::Cpu, Box::new(store), 32, 16, 8, 4).unwrap();
    p.gate = Var::from_tensor(&Tensor::from_vec(gate, 8, &Device::Cpu).unwrap()).unwrap();
    p.book.explore_bias = 0.65;
    p.book.explore_steps = 1000.0;
    p
}

#[test]
fn routing_into_a_paged_pool_matches_for_every_case() {
    let f = Npz::load("moe_paged_forward.npz");
    for case in ["a", "b", "c"] {
        for (mode, kernel) in [(MoeMode::DenseMasked, Kernel::plain()), (MoeMode::SparseDispatch, Kernel::hand())] {
            let tag = format!("{case}/{mode:?}");
            let mut pool = paged_pool(&f, f.f32(&format!("{case}/gate")));
            let explore = case == "b";
            if explore {
                pool.book.recent = f.f32(&format!("{case}/recent_before")).into_iter().map(f64::from).collect();
            }
            pool.book.begin_text(explore);
            let x = Var::from_tensor(&f.tensor(&format!("{case}/x")).reshape((12, 32)).unwrap()).unwrap();
            let router = Var::from_tensor(&f.tensor(&format!("{case}/router"))).unwrap();
            let depth = Var::from_tensor(&f.tensor(&format!("{case}/depth_emb"))).unwrap();
            let cfg = RouteCfg { top_k: 2, capacity_factor: 0.0, mode, z_weight: 1e-3, track: true, kernel };
            let out = pool.route(x.as_tensor(), router.as_tensor(), depth.as_tensor(), &cfg).unwrap();

            // what the card looks like
            let slots: Vec<i64> = f.i64(&format!("{case}/slots"));
            assert_eq!(pool.book.slots, slots, "{tag}: slots");
            let mask: Vec<bool> = f.u8(&format!("{case}/admitted_mask")).into_iter().map(|b| b != 0).collect();
            assert_eq!(pool.book.admitted_mask(), mask, "{tag}: admitted mask");
            assert_eq!(pool.book.loads as i64, f.int(&format!("{case}/loads")), "{tag}: loads");
            assert_eq!(pool.book.swaps as i64, f.int(&format!("{case}/swaps")), "{tag}: swaps");
            assert_eq!(pool.book.segments as i64, f.int(&format!("{case}/segments")), "{tag}: segments");
            let ever: Vec<bool> = f.u8(&format!("{case}/ever")).into_iter().map(|b| b != 0).collect();
            assert_eq!(pool.book.ever, ever, "{tag}: ever");
            for (key, got) in
                [("last_seen", &pool.book.last_seen), ("admits", &pool.book.admits), ("use", &pool.book.use_)]
            {
                let want: Vec<f64> = f.f32(&format!("{case}/{key}")).into_iter().map(f64::from).collect();
                close64(&format!("{tag}: {key}"), got, &want);
            }
            if explore {
                let want: Vec<f64> = f.f32(&format!("{case}/recent_after")).into_iter().map(f64::from).collect();
                close64(&format!("{tag}: recent after"), &pool.book.recent, &want);
            }

            // values
            close_fwd(&format!("{tag}: out"), &host(&out.y), &f.f32(&format!("{case}/out")));
            let aux = host(&out.aux)[0];
            let want_aux = f.scalar(&format!("{case}/aux"));
            assert!((aux - want_aux).abs() < 1e-4 * want_aux.abs().max(1.0), "{tag}: aux {aux} vs {want_aux}");

            // gradients of sum(out * g)
            let g = f.tensor(&format!("{case}/g")).reshape((12, 32)).unwrap();
            let grads = out.y.mul(&g).unwrap().sum_all().unwrap().backward().unwrap();
            close_grad(&format!("{tag}: dx"), &host(grads.get(&x).unwrap()), &f.f32(&format!("{case}/grad/x")));
            close_grad(
                &format!("{tag}: drouter"),
                &host(grads.get(&router).unwrap()),
                &f.f32(&format!("{case}/grad/router")),
            );
            close_grad(
                &format!("{tag}: ddepth"),
                &host(grads.get(&depth).unwrap()),
                &f.f32(&format!("{case}/grad/depth_emb")),
            );
            close_grad(
                &format!("{tag}: dgate"),
                &host(grads.get(&pool.gate).unwrap()),
                &f.f32(&format!("{case}/grad/gate")),
            );
            close_grad(
                &format!("{tag}: dslot_w1"),
                &host(grads.get(&pool.w1).unwrap()),
                &f.f32(&format!("{case}/grad/slot_w1")),
            );
            close_grad(
                &format!("{tag}: dslot_w3"),
                &host(grads.get(&pool.w3).unwrap()),
                &f.f32(&format!("{case}/grad/slot_w3")),
            );
            close_grad(
                &format!("{tag}: dslot_w2"),
                &host(grads.get(&pool.w2).unwrap()),
                &f.f32(&format!("{case}/grad/slot_w2")),
            );
            // gradients of the balancing loss alone
            let ga = out.aux.backward().unwrap();
            close_grad(&format!("{tag}: aux dx"), &host(ga.get(&x).unwrap()), &f.f32(&format!("{case}/grad_aux/x")));
            close_grad(
                &format!("{tag}: aux drouter"),
                &host(ga.get(&router).unwrap()),
                &f.f32(&format!("{case}/grad_aux/router")),
            );
            close_grad(
                &format!("{tag}: aux ddepth"),
                &host(ga.get(&depth).unwrap()),
                &f.f32(&format!("{case}/grad_aux/depth_emb")),
            );
        }
    }
}

/// Run the shared-pool fixture through a paged pool with every expert resident. Slot `s` holds expert `slots[s]`.
fn shared_run(
    f: &Npz,
    case: &str,
    mode: MoeMode,
    active: Option<&[bool]>,
    capacity: f64,
) -> (Tensor, Vec<i64>, PagedPool, Var, Var, Var, Var, usize) {
    let (w1, w3, w2) = (f.f32("w1"), f.f32("w3"), f.f32("w2"));
    let mut store = MemStore::new();
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
                false,
            )
            .unwrap();
    }
    let mut pool = PagedPool::new(&Device::Cpu, Box::new(store), 32, 16, 8, 8).unwrap();
    pool.gate = Var::from_tensor(&f.tensor(&format!("{case}/gate"))).unwrap();
    pool.book.begin_text(false);
    let x_full = f.tensor(&format!("{case}/x")).reshape((12, 32)).unwrap();
    let x = Var::from_tensor(&x_full).unwrap();
    let router = Var::from_tensor(&f.tensor(&format!("{case}/router"))).unwrap();
    let depth = Var::from_tensor(&f.tensor(&format!("{case}/depth_emb"))).unwrap();
    let cfg =
        RouteCfg { top_k: 2, capacity_factor: capacity, mode, z_weight: 1e-3, track: true, kernel: Kernel::plain() };
    let gate = pool.gate.clone();
    let (y, dropped) = match active {
        None => {
            let o = pool.route(x.as_tensor(), router.as_tensor(), depth.as_tensor(), &cfg).unwrap();
            (o.y, o.stats.dropped)
        }
        Some(a) => {
            let idx: Vec<u32> = a.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u32).collect();
            let it = Tensor::from_vec(idx.clone(), idx.len(), &Device::Cpu).unwrap();
            let o = pool
                .route(&x.as_tensor().index_select(&it, 0).unwrap(), router.as_tensor(), depth.as_tensor(), &cfg)
                .unwrap();
            let mut inv = vec![u32::MAX; 12];
            for (p, &i) in idx.iter().enumerate() {
                inv[i as usize] = p as u32;
            }
            (o.y.index_select(&Tensor::from_vec(inv, 12, &Device::Cpu).unwrap(), 0).unwrap(), o.stats.dropped)
        }
    };
    let slots = pool.book.slots.clone();
    (y, slots, pool, x, router, depth, gate, dropped)
}

/// Reorder a `[8, ...]` expert-indexed array into slot order.
fn by_slot(data: &[f32], slots: &[i64], per: usize) -> Vec<f32> {
    slots.iter().flat_map(|&e| data[e as usize * per..(e as usize + 1) * per].to_vec()).collect()
}

#[test]
fn a_resident_pool_matches_the_shared_pool_with_and_without_inactive_characters() {
    let f = Npz::load("moe_shared.npz");
    for case in ["base", "active"] {
        let active: Option<Vec<bool>> =
            (case == "active").then(|| f.u8("active/active").into_iter().map(|b| b != 0).collect());
        for mode in [MoeMode::DenseMasked, MoeMode::SparseDispatch] {
            let tag = format!("{case}/{mode:?}");
            let (y, slots, pool, x, router, depth, gate, _) = shared_run(&f, case, mode, active.as_deref(), 0.0);
            // (only experts someone asked for are on the card; the others cannot be in anyone's top-k anyway)
            close_fwd(&format!("{tag}: out"), &host(&y), &f.f32(&format!("{case}/out")));
            let g = f.tensor(&format!("{case}/g")).reshape((12, 32)).unwrap();
            let grads = y.mul(&g).unwrap().sum_all().unwrap().backward().unwrap();
            close_grad(&format!("{tag}: dx"), &host(grads.get(&x).unwrap()), &f.f32(&format!("{case}/grad/x")));
            close_grad(
                &format!("{tag}: drouter"),
                &host(grads.get(&router).unwrap()),
                &f.f32(&format!("{case}/grad/router")),
            );
            close_grad(
                &format!("{tag}: ddepth"),
                &host(grads.get(&depth).unwrap()),
                &f.f32(&format!("{case}/grad/depth_emb")),
            );
            // per-expert gradients arrive in slot order; experts nobody asked for have none
            let on_card: Vec<i64> = slots.iter().copied().filter(|&e| e >= 0).collect();
            let want_gate = f.f32(&format!("{case}/grad/gate"));
            let got_gate = host(grads.get(&gate).unwrap());
            for e in 0..8 {
                let want = if on_card.contains(&(e as i64)) { want_gate[e] } else { 0.0 };
                assert!(
                    (got_gate[e] - want).abs() < 1e-5 + 1e-3 * want.abs(),
                    "{tag}: dgate[{e}] {} vs {want}",
                    got_gate[e]
                );
            }
            for (key, var, per) in [("w1", &pool.w1, 512usize), ("w3", &pool.w3, 512), ("w2", &pool.w2, 512)] {
                let want = f.f32(&format!("{case}/grad/{key}"));
                let mut slot_want = by_slot(&want, &slots.iter().map(|&e| e.max(0)).collect::<Vec<_>>(), per);
                for (s, &e) in slots.iter().enumerate() {
                    if e < 0 {
                        slot_want[s * per..(s + 1) * per].iter_mut().for_each(|v| *v = 0.0);
                    }
                }
                close_grad(&format!("{tag}: d{key}"), &host(grads.get(var).unwrap()), &slot_want);
            }
        }
    }
    // inactive characters produce exactly zero
    let (y, ..) = shared_run(
        &f,
        "active",
        MoeMode::SparseDispatch,
        Some(&f.u8("active/active").iter().map(|&b| b != 0).collect::<Vec<_>>()),
        0.0,
    );
    let rows = y.to_vec2::<f32>().unwrap();
    for (i, a) in f.u8("active/active").iter().enumerate() {
        if *a == 0 {
            assert!(rows[i].iter().all(|&v| v == 0.0), "inactive character {i} must output exactly zero");
        }
    }
}

#[test]
fn the_capacity_limit_drops_what_the_reference_drops() {
    let f = Npz::load("moe_shared.npz");
    let (y, slots, pool, x, _, _, _, dropped) = shared_run(&f, "cap", MoeMode::SparseDispatch, None, 1.5);
    // the order-free facts: how many were dropped, and the output under the stable convention
    let counts: Vec<i64> = f.i64("cap/counts_before");
    let limit = f.int("cap/limit");
    let want_dropped: i64 = counts.iter().map(|&c| (c - limit).max(0)).sum();
    assert_eq!(dropped as i64, want_dropped, "dropped assignments");
    assert_eq!(f.int("cap/dropped"), want_dropped);
    close_fwd("capped out", &host(&y), &f.f32("cap/out_stable"));
    let g = f.tensor("cap/g").reshape((12, 32)).unwrap();
    let grads = y.mul(&g).unwrap().sum_all().unwrap().backward().unwrap();
    close_grad("capped dx", &host(grads.get(&x).unwrap()), &f.f32("cap/grad_stable/x"));
    let _ = (slots, pool);
}

#[test]
fn a_gated_off_pool_and_a_one_expert_pool_never_nan() {
    // the degenerate shapes the fixtures do not cover must still be well-defined
    let dev = Device::Cpu;
    let mut store = MemStore::new();
    store
        .put(0, ExpertEntry { w1: vec![0.1; 12], w3: vec![0.1; 12], w2: vec![0.1; 12], moments: None }, false)
        .unwrap();
    let mut pool = PagedPool::new(&dev, Box::new(store), 4, 3, 1, 1).unwrap();
    pool.book.begin_text(false);
    let x = Tensor::ones((5, 4), DType::F32, &dev).unwrap();
    let router = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
    let cfg = RouteCfg {
        top_k: 2,
        capacity_factor: 1.5,
        mode: MoeMode::SparseDispatch,
        z_weight: 1e-3,
        track: true,
        kernel: Kernel::plain(),
    };
    let out = pool.route(&x, &router, &Tensor::zeros(4, DType::F32, &dev).unwrap(), &cfg).unwrap();
    assert!(host(&out.y).iter().all(|v| v.is_finite()));
}
