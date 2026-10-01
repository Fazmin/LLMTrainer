//! Times the building blocks of one recurrent row (forward + backward) at a preset's sizes, to see where a step goes.
//!   bench_parts [--device cpu|metal] [--d 384] [--heads 6] [--t 1024] [--experts 16] [--ff 768] [--k 4] [--dense-ff 1024]

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use minagi_core::candle_core::{DType, Device, Tensor, Var};
use minagi_core::moe::{ExpertEntry, ExpertStore, MemStore, PagedPool, RouteCfg};
use minagi_core::ops::{self, Kernel};
use minagi_core::rng::HostRng;
use minagi_types::MoeMode;

fn arg(a: &HashMap<String, String>, k: &str, d: usize) -> usize {
    a.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn randn(shape: &[usize], std: f32, seed: u64, dev: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(HostRng::new(seed).normal_vec(n, std), shape.to_vec(), dev).unwrap()
}

/// Mean seconds over `n` runs (after `warm` untimed ones); each run is synchronised.
fn time(dev: &Device, warm: usize, n: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warm {
        f();
        dev.synchronize().unwrap();
    }
    let t = Instant::now();
    for _ in 0..n {
        f();
        dev.synchronize().unwrap();
    }
    t.elapsed().as_secs_f64() / n as f64
}

fn main() -> Result<()> {
    let mut kv = HashMap::new();
    let rest: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < rest.len() {
        if let Some(k) = rest[i].strip_prefix("--") {
            kv.insert(k.to_string(), rest[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    let dev = match kv.get("device").map(String::as_str) {
        Some("cpu") => Device::Cpu,
        _ => Device::new_metal(0)?,
    };
    let (d, h, t) = (arg(&kv, "d", 384), arg(&kv, "heads", 6), arg(&kv, "t", 1024));
    let (e, ff, k, dff) = (arg(&kv, "experts", 16), arg(&kv, "ff", 768), arg(&kv, "k", 4), arg(&kv, "dense-ff", 1024));
    let dh = d / h;
    println!("d {d} heads {h} T {t} experts {e} expert ff {ff} top-k {k} dense ff {dff}");
    let kern = Kernel::hand();
    let mask = ops::causal_mask(t, &dev)?;
    let (cos, sin) = ops::rope_tables(t, dh, 10000.0, &dev)?;

    let x = Var::from_tensor(&randn(&[t, d], 1.0, 1, &dev))?;
    let (ln, qkv, proj) = (
        Var::from_tensor(&Tensor::ones(d, DType::F32, &dev)?)?,
        Var::from_tensor(&randn(&[3 * d, d], 0.02, 2, &dev))?,
        Var::from_tensor(&randn(&[d, d], 0.02, 3, &dev))?,
    );
    let attn = |kern: Kernel| {
        let xn = ops::rms_norm(x.as_tensor(), ln.as_tensor(), 1e-6, kern).unwrap();
        let all = ops::linear(&xn, qkv.as_tensor()).unwrap();
        let split = |i: usize| {
            all.narrow(1, i * d, d)
                .unwrap()
                .reshape((1, t, h, dh))
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let (q, kk) =
            (ops::rope_i(&split(0), &cos, &sin, kern).unwrap(), ops::rope_i(&split(1), &cos, &sin, kern).unwrap());
        let y = ops::attention(&q, &kk, &split(2), &mask, 0, kern).unwrap();
        let y = y.transpose(1, 2).unwrap().contiguous().unwrap().reshape((t, d)).unwrap();
        ops::linear(&y, proj.as_tensor()).unwrap().sum_all().unwrap()
    };
    let fwd = time(&dev, 3, 10, || {
        let _ = attn(kern);
    });
    let both = time(&dev, 3, 10, || {
        let _ = attn(kern).backward().unwrap();
    });
    println!("attention (hand ops)          fwd {:6.1} ms   fwd+bwd {:6.1} ms", fwd * 1e3, both * 1e3);
    let both_c = time(&dev, 3, 10, || {
        let _ = attn(Kernel::plain()).backward().unwrap();
    });
    println!("attention (composed ops)      fwd+bwd {:6.1} ms", both_c * 1e3);

    let (w1, w3, w2) = (
        Var::from_tensor(&randn(&[dff, d], 0.02, 4, &dev))?,
        Var::from_tensor(&randn(&[dff, d], 0.02, 5, &dev))?,
        Var::from_tensor(&randn(&[d, dff], 0.02, 6, &dev))?,
    );
    let mlp = time(&dev, 3, 10, || {
        let _ = ops::swiglu(x.as_tensor(), w1.as_tensor(), w3.as_tensor(), w2.as_tensor())
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
    });
    println!("dense SwiGLU (ff {dff})        fwd+bwd {:6.1} ms", mlp * 1e3);

    let mut store = MemStore::new();
    for u in 0..e {
        let mut r = HostRng::new(100 + u as u64);
        store.put(
            u as u64,
            ExpertEntry {
                w1: r.normal_vec(ff * d, 0.02),
                w3: r.normal_vec(ff * d, 0.02),
                w2: r.normal_vec(d * ff, 0.02),
                moments: None,
            },
            false,
        )?;
    }
    let router = Var::from_tensor(&randn(&[e, d], 0.02, 7, &dev))?;
    let depth = Var::from_tensor(&Tensor::zeros(d, DType::F32, &dev)?)?;
    for mode in [MoeMode::SparseDispatch, MoeMode::DenseMasked] {
        // (fill the card once so routing, not paging, is what is timed)
        let mut st = MemStore::new();
        for u in 0..e {
            let mut r = HostRng::new(100 + u as u64);
            st.put(
                u as u64,
                ExpertEntry {
                    w1: r.normal_vec(ff * d, 0.02),
                    w3: r.normal_vec(ff * d, 0.02),
                    w2: r.normal_vec(d * ff, 0.02),
                    moments: None,
                },
                false,
            )?;
        }
        let mut pool = PagedPool::new(&dev, Box::new(st), d, ff, e, e)?;
        let cfg = RouteCfg { top_k: k, capacity_factor: 1.5, mode, z_weight: 1e-3, track: true, kernel: kern };
        pool.book.begin_text(false);
        let _ = pool.route(x.as_tensor(), router.as_tensor(), depth.as_tensor(), &cfg)?;
        let r = time(&dev, 3, 10, || {
            pool.book.begin_forward(false);
            let o = pool.route(x.as_tensor(), router.as_tensor(), depth.as_tensor(), &cfg).unwrap();
            let _ = (o.y.sum_all().unwrap() + o.aux).unwrap().backward().unwrap();
        });
        println!("MoE {mode:?}  ({e} experts, top-{k})  fwd+bwd {:6.1} ms", r * 1e3);
    }

    let emb = Var::from_tensor(&randn(&[265, d], 0.02, 8, &dev))?;
    let tg = Tensor::from_vec((0..t as u32).map(|i| i % 265).collect::<Vec<_>>(), t, &dev)?;
    let ids = tg.clone();
    let head = time(&dev, 3, 10, || {
        let xe = emb.as_tensor().index_select(&ids, 0).unwrap();
        let logits = ops::linear(&xe, emb.as_tensor()).unwrap();
        let _ = ops::cross_entropy_rows(&logits, &tg).unwrap().sum_all().unwrap().backward().unwrap();
    });
    println!("embed + tied head + CE        fwd+bwd {:6.1} ms", head * 1e3);

    let adapter = Var::from_tensor(&randn(&[d, 2 * d], 0.02, 9, &dev))?;
    let ad = time(&dev, 3, 10, || {
        let u = ops::linear(&Tensor::cat(&[x.as_tensor(), x.as_tensor()], 1).unwrap(), adapter.as_tensor()).unwrap();
        let _ = u.sum_all().unwrap().backward().unwrap();
    });
    println!("adapter (cat + matmul)        fwd+bwd {:6.1} ms", ad * 1e3);
    Ok(())
}
