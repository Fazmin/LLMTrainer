//! Parity of the tensor primitives (and the decoding rule) with the original Python implementation.

#![allow(clippy::type_complexity, clippy::too_many_arguments)]

mod golden_support;

use golden_support::*;
use minagi_core::candle_core::{Device, Tensor, Var};
use minagi_core::chat::{Decode, adjust, pick_next};
use minagi_core::ops::{self, Kernel};

const KERNELS: [(&str, Kernel); 2] = [("composed", Kernel::plain()), ("hand", Kernel::hand())];

fn weighted_sum(y: &Tensor, g: &Tensor) -> Tensor {
    y.mul(g).unwrap().sum_all().unwrap()
}

#[test]
fn rmsnorm_forward_and_backward() {
    let f = Npz::load("ops_rmsnorm.npz");
    let eps = f.scalar("eps");
    let w = f.tensor("weight");
    for case in ["base", "big", "tiny", "batch2"] {
        let shape = f.shape(&format!("{case}/x"));
        let flat = (shape[0] * shape[1], shape[2]);
        let x = f.tensor(&format!("{case}/x")).reshape(flat).unwrap();
        let g = f.tensor(&format!("{case}/g")).reshape(flat).unwrap();
        for (name, k) in KERNELS {
            let (xv, wv) = (Var::from_tensor(&x).unwrap(), Var::from_tensor(&w).unwrap());
            let y = ops::rms_norm(xv.as_tensor(), wv.as_tensor(), eps, k).unwrap();
            let grads = weighted_sum(&y, &g).backward().unwrap();
            close_fwd(&format!("{case}/{name} out"), &host(&y), &f.f32(&format!("{case}/out")));
            close_grad(&format!("{case}/{name} dx"), &host(grads.get(&xv).unwrap()), &f.f32(&format!("{case}/dx")));
            close_grad(
                &format!("{case}/{name} dweight"),
                &host(grads.get(&wv).unwrap()),
                &f.f32(&format!("{case}/dweight")),
            );
        }
    }
}

#[test]
fn rope_tables_and_rotation() {
    let f = Npz::load("ops_rope.npz");
    let (cos, sin) = ops::rope_tables(64, 16, 10000.0, &Device::Cpu).unwrap();
    close("cos table", &host(&cos), &f.f32("cos"), 2e-6, 0.0);
    close("sin table", &host(&sin), &f.f32("sin"), 2e-6, 0.0);
    // use the reference's own tables for the rotation so only the rotation is under test
    let (rc, rs) = (f.tensor("cos"), f.tensor("sin"));
    for case in ["pos0", "pos7", "pos59"] {
        let off = f.int(&format!("{case}/pos_offset")) as usize;
        let x = f.tensor(&format!("{case}/x"));
        let g = f.tensor(&format!("{case}/g"));
        let (c, s) =
            (rc.narrow(0, off, 5).unwrap().contiguous().unwrap(), rs.narrow(0, off, 5).unwrap().contiguous().unwrap());
        for (name, k) in KERNELS {
            let xv = Var::from_tensor(&x).unwrap();
            let y = ops::rope_i(xv.as_tensor(), &c, &s, k).unwrap();
            let grads = weighted_sum(&y, &g).backward().unwrap();
            close_fwd(&format!("{case}/{name} out"), &host(&y), &f.f32(&format!("{case}/out")));
            close_grad(&format!("{case}/{name} dx"), &host(grads.get(&xv).unwrap()), &f.f32(&format!("{case}/dx")));
        }
    }
}

#[test]
fn swiglu_forward_and_backward() {
    let f = Npz::load("ops_swiglu.npz");
    for case in ["base", "big"] {
        let x2 = f.tensor(&format!("{case}/x")).reshape((5, 32)).unwrap();
        let g = f.tensor(&format!("{case}/g")).reshape((5, 32)).unwrap();
        let (xv, w1, w3, w2) = (
            Var::from_tensor(&x2).unwrap(),
            Var::from_tensor(&f.tensor("w1")).unwrap(),
            Var::from_tensor(&f.tensor("w3")).unwrap(),
            Var::from_tensor(&f.tensor("w2")).unwrap(),
        );
        let y = ops::swiglu(xv.as_tensor(), w1.as_tensor(), w3.as_tensor(), w2.as_tensor()).unwrap();
        let grads = weighted_sum(&y, &g).backward().unwrap();
        close_fwd(&format!("{case} out"), &host(&y), &f.f32(&format!("{case}/out")));
        close_grad(&format!("{case} dx"), &host(grads.get(&xv).unwrap()), &f.f32(&format!("{case}/dx")));
        close_grad(&format!("{case} dw1"), &host(grads.get(&w1).unwrap()), &f.f32(&format!("{case}/dw1")));
        close_grad(&format!("{case} dw3"), &host(grads.get(&w3).unwrap()), &f.f32(&format!("{case}/dw3")));
        close_grad(&format!("{case} dw2"), &host(grads.get(&w2).unwrap()), &f.f32(&format!("{case}/dw2")));
    }
}

/// `x [T,32]` -> `(q, k, v)` each `[1,2,T,16]`, with rope at absolute positions `off..off+T` from the reference tables.
fn qkv_heads(x: &Tensor, qkv: &Tensor, off: usize, cos: &Tensor, sin: &Tensor, k: Kernel) -> (Tensor, Tensor, Tensor) {
    let t = x.dim(0).unwrap();
    let all = ops::linear(x, qkv).unwrap();
    let split = |i: usize| {
        all.narrow(1, i * 32, 32)
            .unwrap()
            .reshape((1, t, 2, 16))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let (c, s) =
        (cos.narrow(0, off, t).unwrap().contiguous().unwrap(), sin.narrow(0, off, t).unwrap().contiguous().unwrap());
    (ops::rope_i(&split(0), &c, &s, k).unwrap(), ops::rope_i(&split(1), &c, &s, k).unwrap(), split(2))
}

#[test]
fn attention_with_and_without_a_cache() {
    let f = Npz::load("ops_attention.npz");
    let rope = Npz::load("ops_rope.npz");
    let (cos, sin) = (rope.tensor("cos"), rope.tensor("sin"));
    let mask = ops::causal_mask(64, &Device::Cpu).unwrap();
    let (qkv_w, proj_w) = (f.tensor("qkv.weight"), f.tensor("proj.weight"));
    for (name, k) in KERNELS {
        // case a: no cache
        let x = f.tensor("a/x").reshape((8, 32)).unwrap();
        let (xv, wq, wp) =
            (Var::from_tensor(&x).unwrap(), Var::from_tensor(&qkv_w).unwrap(), Var::from_tensor(&proj_w).unwrap());
        let (q, kk, v) = qkv_heads(xv.as_tensor(), wq.as_tensor(), 0, &cos, &sin, k);
        close_fwd(&format!("a/{name} q_rope"), &host(&q), &f.f32("a/q_rope"));
        close_fwd(&format!("a/{name} k_rope"), &host(&kk), &f.f32("a/k_rope"));
        let y = ops::attention(&q, &kk, &v, &mask, 0, k).unwrap();
        let y = y.transpose(1, 2).unwrap().contiguous().unwrap().reshape((8, 32)).unwrap();
        close_fwd(&format!("a/{name} y_pre_proj"), &host(&y), &f.f32("a/y_pre_proj"));
        let out = ops::linear(&y, wp.as_tensor()).unwrap();
        let g = f.tensor("a/g").reshape((8, 32)).unwrap();
        let grads = weighted_sum(&out, &g).backward().unwrap();
        close_fwd(&format!("a/{name} out"), &host(&out), &f.f32("a/out"));
        close_grad(&format!("a/{name} dx"), &host(grads.get(&xv).unwrap()), &f.f32("a/dx"));
        close_grad(&format!("a/{name} dqkv"), &host(grads.get(&wq).unwrap()), &f.f32("a/dqkv.weight"));
        close_grad(&format!("a/{name} dproj"), &host(grads.get(&wp).unwrap()), &f.f32("a/dproj.weight"));

        // cases b and c: P cached positions, T new ones (the cache is a constant input)
        for case in ["b", "c"] {
            let (p, t) = (f.int(&format!("{case}/P")) as usize, f.int(&format!("{case}/T")) as usize);
            let x = f.tensor(&format!("{case}/x_new")).reshape((t, 32)).unwrap();
            let (xv, wq, wp) =
                (Var::from_tensor(&x).unwrap(), Var::from_tensor(&qkv_w).unwrap(), Var::from_tensor(&proj_w).unwrap());
            let (q, kn, vn) = qkv_heads(xv.as_tensor(), wq.as_tensor(), p, &cos, &sin, k);
            let (ck, cv) = (f.tensor(&format!("{case}/cache_k_in")), f.tensor(&format!("{case}/cache_v_in")));
            let (k_all, v_all) = (Tensor::cat(&[&ck, &kn], 2).unwrap(), Tensor::cat(&[&cv, &vn], 2).unwrap());
            close_fwd(&format!("{case}/{name} cache_k_out"), &host(&k_all), &f.f32(&format!("{case}/cache_k_out")));
            let y = ops::attention(&q, &k_all, &v_all, &mask, p, k).unwrap();
            let y = y.transpose(1, 2).unwrap().contiguous().unwrap().reshape((t, 32)).unwrap();
            let out = ops::linear(&y, wp.as_tensor()).unwrap();
            let g = f.tensor(&format!("{case}/g")).reshape((t, 32)).unwrap();
            let grads = weighted_sum(&out, &g).backward().unwrap();
            close_fwd(&format!("{case}/{name} out"), &host(&out), &f.f32(&format!("{case}/out")));
            close_grad(
                &format!("{case}/{name} dx_new"),
                &host(grads.get(&xv).unwrap()),
                &f.f32(&format!("{case}/dx_new")),
            );
            close_grad(
                &format!("{case}/{name} dproj"),
                &host(grads.get(&wp).unwrap()),
                &f.f32(&format!("{case}/dproj.weight")),
            );
        }
    }
}

#[test]
fn tiled_checkpointed_attention_matches_the_reference_too() {
    let f = Npz::load("ops_attention.npz");
    let rope = Npz::load("ops_rope.npz");
    let (cos, sin) = (rope.tensor("cos"), rope.tensor("sin"));
    let mask = ops::causal_mask(64, &Device::Cpu).unwrap();
    let k = Kernel::hand().with_tiles(3, true);
    let x = f.tensor("a/x").reshape((8, 32)).unwrap();
    let (xv, wq) = (Var::from_tensor(&x).unwrap(), Var::from_tensor(&f.tensor("qkv.weight")).unwrap());
    let (q, kk, v) = qkv_heads(xv.as_tensor(), wq.as_tensor(), 0, &cos, &sin, k);
    let y = ops::attention(&q, &kk, &v, &mask, 0, k).unwrap();
    let y = y.transpose(1, 2).unwrap().contiguous().unwrap().reshape((8, 32)).unwrap();
    close_fwd("tiled y_pre_proj", &host(&y), &f.f32("a/y_pre_proj"));
}

#[test]
fn greedy_decoding_with_the_adaptation_trace() {
    let j = json("decode.json");
    let cases = j["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let logits: Vec<f32> = c["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let prefix: Vec<u32> = c["prefix"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let p = &c["params"];
        let d = Decode {
            adapt_strength: p["adapt_strength"].as_f64().unwrap_or(0.0),
            adapt_decay: p["adapt_decay"].as_f64().unwrap_or(0.88),
            rep_penalty: p["rep_penalty"].as_f64().unwrap_or(1.0),
        };
        let want: Vec<f32> = c["adjusted"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        close(&format!("{name}: adjusted logits"), &adjust(&logits, &prefix, &d), &want, 1e-5, 1e-6);
        if !c["argmax_is_a_tie"].as_bool().unwrap_or(false) {
            assert_eq!(pick_next(&logits, &prefix, &d) as u64, c["expected_next"].as_u64().unwrap(), "{name}");
        }
    }
}

#[test]
fn the_tokenizer_matches_the_reference() {
    use minagi_text::ByteTokenizer;
    let j = json("tokenizer.json");
    let tok = ByteTokenizer::new();
    for c in j["encode_cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u16> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u16).collect();
        assert_eq!(tok.encode_str(text), want, "encode {text:?}");
        if c["roundtrip_decode_equals_text"].as_bool().unwrap_or(false) {
            assert_eq!(tok.decode(&want), text, "round trip {text:?}");
        }
    }
    for c in j["decode_cases"].as_array().unwrap() {
        let ids: Vec<i64> = c["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        // the reference silently drops ids outside 0..264
        let kept: Vec<u16> = ids.iter().filter(|&&i| (0..265).contains(&i)).map(|&i| i as u16).collect();
        assert_eq!(tok.decode(&kept), c["text"].as_str().unwrap(), "decode case {}", c["name"]);
    }
}
