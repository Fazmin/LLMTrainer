use candle_core::{DType, Device, Result, Tensor, Var, D};

fn nonfinite(t: &Tensor) -> Result<bool> {
    Ok(!t.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?.is_finite())
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?.detach();
    let e = x.broadcast_sub(&m)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    e.broadcast_div(&s)
}

fn gqa_trials(dev: &Device, dt: DType) -> Result<usize> {
    let (t, dm, hd) = (64usize, 64usize, 32usize);
    let mut bad = 0;
    for trial in 0..16u64 {
        let mk = |r: usize, c: usize, seed: u64| -> Result<Var> {
            let _ = seed;
            Var::from_tensor(&(Tensor::randn(0f32, 0.1f32, (r, c), dev)?).to_dtype(dt)?)
        };
        let (wq, wk, wv, wo) = (mk(dm, 2 * hd, trial)?, mk(dm, hd, trial)?, mk(dm, hd, trial)?, mk(2 * hd, dm, trial)?);
        let x = Tensor::randn(0f32, 1f32, (t, dm), dev)?.to_dtype(dt)?;
        let mut mask = vec![0f32; t * t];
        for i in 0..t { for j in i + 1..t { mask[i * t + j] = -1e9; } }
        let mask = Tensor::from_vec(mask, (t, t), dev)?.to_dtype(dt)?;
        let mut nan = false;
        for _ in 0..80 {
            let q = x.matmul(wq.as_tensor())?.reshape((1, t, 2, hd))?.transpose(1, 2)?.contiguous()?;
            let kv = |w: &Var| -> Result<Tensor> {
                x.matmul(w.as_tensor())?.reshape((1, 1, t, hd))?.unsqueeze(2)?.expand((1usize, 1, 2, t, hd))?.contiguous()?.reshape((1, 2, t, hd))
            };
            let (k, v) = (kv(&wk)?, kv(&wv)?);
            let att = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (hd as f64).sqrt()))?.broadcast_add(&mask)?;
            let o = softmax_last(&att)?.matmul(&v)?;
            let o = o.transpose(1, 2)?.contiguous()?.reshape((t, 2 * hd))?;
            let loss = o.matmul(wo.as_tensor())?.to_dtype(DType::F32)?.sqr()?.mean_all()?;
            let g = loss.backward()?;
            let mut bad_now = false;
            for p in [&wq, &wk, &wv, &wo] {
                if nonfinite(g.get(p).unwrap())? { bad_now = true; }
            }
            if bad_now { nan = true; break; }
            for p in [&wq, &wk, &wv, &wo] {
                p.set(&(p.as_tensor() - (g.get(p).unwrap() * 1e-3)?)?)?;
            }
        }
        bad += nan as usize;
    }
    Ok(bad)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    for (label, dt) in [("bf16", DType::BF16), ("f32", DType::F32)] {
        let mut bad = 0;
        for _ in 0..16 {
            let w = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1usize, 1, 512, 64), &dev)?.to_dtype(dt)?)?;
            let mut nan = false;
            for _ in 0..80 {
                let e = w.as_tensor().unsqueeze(2)?.expand((1usize, 1, 2, 512, 64))?;
                let g = (e.sqr()?.mean_all()? * 0.5)?.backward()?;
                let gw = g.get(&w).unwrap();
                if nonfinite(gw)? { nan = true; break; }
                w.set(&(w.as_tensor() - (gw * 1e-3)?)?)?;
            }
            bad += nan as usize;
        }
        println!("candle-core 0.11.0 (crates.io) bare expand {label}: {bad}/16 trials non-finite");
    }
    for (label, dt) in [("f32", DType::F32), ("bf16", DType::BF16)] {
        println!("candle-core 0.11.0 (crates.io) GQA block (interior rank-5 expand) {label}: {}/16 trials non-finite", gqa_trials(&dev, dt)?);
    }
    Ok(())
}
