//! The trunk's parameters: embeddings, the dense prelude blocks, the recurrent block's attention, the adapter, the
//! halting head and the router. Named exactly as the Python reference names them in its `state_dict`, so checkpoints map
//! one to one.
//!
//! The expert pool's weights live in `moe::pool`; the router and the depth embedding live here because they are
//! parameters of the recurrent block's call site (one router row per *expert*, never per slot).

use candle_core::{DType, Device, Tensor, Var};
use minagi_types::ModelConfig;

use crate::error::Result;
use crate::rng::HostRng;

/// Which learning rate and which part of the checkpoint a parameter belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// The shared body: embeddings, attention, norms, dense MLPs, adapter, halting head. Learns at `lr * trunk_lr_mult`.
    Trunk,
    /// The pool: router, depth embedding, gates, expert slots. Learns at the base rate.
    Pool,
}

#[derive(Clone)]
pub struct NamedVar {
    pub name: String,
    pub var: Var,
    pub group: Group,
}

/// One block's attention half: pre-norm, fused qkv, output projection, and the pre-norm of the MLP half.
#[derive(Clone)]
pub struct AttnParams {
    pub ln1: Var,
    pub qkv: Var,
    pub proj: Var,
    pub ln2: Var,
}

/// The dense SwiGLU of a prelude block (`w1`, `w3`: `[d_ff, d_model]`; `w2`: `[d_model, d_ff]`).
#[derive(Clone)]
pub struct MlpParams {
    pub w1: Var,
    pub w3: Var,
    pub w2: Var,
}

#[derive(Clone)]
pub struct Trunk {
    /// `[vocab, d_model]`, also the output head (tied).
    pub emb: Var,
    pub prelude: Vec<(AttnParams, MlpParams)>,
    /// The one weight-shared recurrent block; its MLP is the expert pool.
    pub recur: AttnParams,
    /// `[n_experts, d_model]`: one row per expert in the pool.
    pub router: Var,
    pub depth_emb: Var,
    /// `[d_model, 2 * d_model]`, starts as `[I | I]`.
    pub adapter: Var,
    pub ln_f: Var,
    /// `[1, d_model]`.
    pub halt_w: Var,
    /// `[1]`.
    pub halt_b: Var,
}

fn var(data: Vec<f32>, shape: &[usize], dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::from_vec(data, shape.to_vec(), dev)?)?)
}

fn ones(n: usize, dev: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::ones(n, DType::F32, dev)?)?)
}

impl Trunk {
    /// Fresh weights, initialised as the reference does: `N(0, 0.02)` for every matrix and the embedding, the output
    /// projection and every `w2` at `0.02 / sqrt(2 * depth)`, norms at one, the depth embedding at zero, the adapter at
    /// `[I | I]` (so the first pass sees the text itself and later passes accumulate on top), the halting weight at 1% of
    /// `N(0, 0.02)` and its bias at -2 (start biased toward pondering rather than halting instantly).
    pub fn init(cfg: &ModelConfig, rng: &mut HostRng, dev: &Device) -> Result<Self> {
        let (d, ff, v) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.vocab_size as usize);
        let n_exp = cfg.pool_experts as usize;
        let depth = (cfg.n_prelude + cfg.n_recur + cfg.n_coda) as f32;
        let (std, std_out) = (0.02f32, 0.02 / (2.0 * depth).sqrt());
        let attn = |rng: &mut HostRng| -> Result<AttnParams> {
            Ok(AttnParams {
                ln1: ones(d, dev)?,
                qkv: var(rng.normal_vec(3 * d * d, std), &[3 * d, d], dev)?,
                proj: var(rng.normal_vec(d * d, std_out), &[d, d], dev)?,
                ln2: ones(d, dev)?,
            })
        };
        let emb = var(rng.normal_vec(v * d, std), &[v, d], dev)?;
        let mut prelude = Vec::new();
        for _ in 0..cfg.n_prelude {
            let a = attn(rng)?;
            let m = MlpParams {
                w1: var(rng.normal_vec(ff * d, std), &[ff, d], dev)?,
                w3: var(rng.normal_vec(ff * d, std), &[ff, d], dev)?,
                w2: var(rng.normal_vec(d * ff, std_out), &[d, ff], dev)?,
            };
            prelude.push((a, m));
        }
        let recur = attn(rng)?;
        let router = var(rng.normal_vec(n_exp * d, std), &[n_exp, d], dev)?;
        let depth_emb = Var::from_tensor(&Tensor::zeros(d, DType::F32, dev)?)?;
        let mut adapter = vec![0f32; d * 2 * d];
        for i in 0..d {
            adapter[i * 2 * d + i] = 1.0;
            adapter[i * 2 * d + d + i] = 1.0;
        }
        let adapter = var(adapter, &[d, 2 * d], dev)?;
        let halt_w = var(rng.normal_vec(d, std * 0.01), &[1, d], dev)?;
        let halt_b = var(vec![-2.0], &[1], dev)?;
        Ok(Self { emb, prelude, recur, router, depth_emb, adapter, ln_f: ones(d, dev)?, halt_w, halt_b })
    }

    /// Weights read from a checkpoint's `core.npz` and `routers.npz` (named as the Python `state_dict()`).
    pub fn load(
        cfg: &ModelConfig,
        core: &crate::store::checkpoint::NamedTensors,
        routers: &crate::store::checkpoint::NamedTensors,
        dev: &Device,
    ) -> Result<Self> {
        let get = |name: &str, shape: &[usize]| -> Result<Var> {
            let t = core.get(name).or_else(|| routers.get(name)).ok_or_else(|| {
                crate::EngineError::Checkpoint(format!("the checkpoint has no tensor named {name:?}"))
            })?;
            if t.shape != shape {
                return Err(crate::EngineError::Checkpoint(format!(
                    "tensor {name:?} has shape {:?} but this model needs {shape:?}",
                    t.shape
                )));
            }
            var(t.data.clone(), shape, dev)
        };
        let (d, ff, v) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.vocab_size as usize);
        let attn = |prefix: &str| -> Result<AttnParams> {
            Ok(AttnParams {
                ln1: get(&format!("{prefix}.ln1.weight"), &[d])?,
                qkv: get(&format!("{prefix}.attn.qkv.weight"), &[3 * d, d])?,
                proj: get(&format!("{prefix}.attn.proj.weight"), &[d, d])?,
                ln2: get(&format!("{prefix}.ln2.weight"), &[d])?,
            })
        };
        let mut prelude = Vec::new();
        for i in 0..cfg.n_prelude {
            let p = format!("prelude.{i}");
            let m = MlpParams {
                w1: get(&format!("{p}.mlp.w1.weight"), &[ff, d])?,
                w3: get(&format!("{p}.mlp.w3.weight"), &[ff, d])?,
                w2: get(&format!("{p}.mlp.w2.weight"), &[d, ff])?,
            };
            prelude.push((attn(&p)?, m));
        }
        let n_experts = routers
            .get("recur.0.mlp.router.weight")
            .map(|t| t.shape.first().copied().unwrap_or(0))
            .ok_or_else(|| crate::EngineError::Checkpoint("the checkpoint has no router".into()))?;
        Ok(Self {
            emb: get("tok_emb.weight", &[v, d])?,
            prelude,
            recur: attn("recur.0")?,
            router: get("recur.0.mlp.router.weight", &[n_experts, d])?,
            depth_emb: get("recur.0.mlp.depth_emb", &[d])?,
            adapter: get("adapter.weight", &[d, 2 * d])?,
            ln_f: get("ln_f.weight", &[d])?,
            halt_w: get("halt.weight", &[1, d])?,
            halt_b: get("halt.bias", &[1])?,
        })
    }

    /// Every trunk-side parameter with its checkpoint name and group, in a fixed order.
    pub fn named(&self) -> Vec<NamedVar> {
        let mut out = Vec::new();
        let mut add = |name: String, var: &Var, group: Group| out.push(NamedVar { name, var: var.clone(), group });
        add("tok_emb.weight".into(), &self.emb, Group::Trunk);
        for (i, (a, m)) in self.prelude.iter().enumerate() {
            add(format!("prelude.{i}.ln1.weight"), &a.ln1, Group::Trunk);
            add(format!("prelude.{i}.attn.qkv.weight"), &a.qkv, Group::Trunk);
            add(format!("prelude.{i}.attn.proj.weight"), &a.proj, Group::Trunk);
            add(format!("prelude.{i}.ln2.weight"), &a.ln2, Group::Trunk);
            add(format!("prelude.{i}.mlp.w1.weight"), &m.w1, Group::Trunk);
            add(format!("prelude.{i}.mlp.w3.weight"), &m.w3, Group::Trunk);
            add(format!("prelude.{i}.mlp.w2.weight"), &m.w2, Group::Trunk);
        }
        add("recur.0.ln1.weight".into(), &self.recur.ln1, Group::Trunk);
        add("recur.0.attn.qkv.weight".into(), &self.recur.qkv, Group::Trunk);
        add("recur.0.attn.proj.weight".into(), &self.recur.proj, Group::Trunk);
        add("recur.0.ln2.weight".into(), &self.recur.ln2, Group::Trunk);
        add("recur.0.mlp.router.weight".into(), &self.router, Group::Pool);
        add("recur.0.mlp.depth_emb".into(), &self.depth_emb, Group::Pool);
        add("adapter.weight".into(), &self.adapter, Group::Trunk);
        add("ln_f.weight".into(), &self.ln_f, Group::Trunk);
        add("halt.weight".into(), &self.halt_w, Group::Trunk);
        add("halt.bias".into(), &self.halt_b, Group::Trunk);
        out
    }

    /// Parameters outside the pool, in the order `named` lists them.
    pub fn n_params(&self) -> usize {
        self.named().iter().map(|p| p.var.elem_count()).sum()
    }

    /// The weights as plain tensors. With `track` they keep their variable identity (an autograd graph is recorded);
    /// without, they are detached views of the same storage (nothing is recorded, and later in-place optimiser updates
    /// remain visible through them).
    pub fn view(&self, track: bool) -> TrunkView {
        let t = |v: &Var| if track { v.as_tensor().clone() } else { v.as_tensor().detach() };
        let attn = |a: &AttnParams| AttnView { ln1: t(&a.ln1), qkv: t(&a.qkv), proj: t(&a.proj), ln2: t(&a.ln2) };
        TrunkView {
            emb: t(&self.emb),
            prelude: self
                .prelude
                .iter()
                .map(|(a, m)| (attn(a), MlpView { w1: t(&m.w1), w3: t(&m.w3), w2: t(&m.w2) }))
                .collect(),
            recur: attn(&self.recur),
            router: t(&self.router),
            depth_emb: t(&self.depth_emb),
            adapter: t(&self.adapter),
            ln_f: t(&self.ln_f),
            halt_w: t(&self.halt_w),
            halt_b: t(&self.halt_b),
        }
    }
}

pub struct AttnView {
    pub ln1: Tensor,
    pub qkv: Tensor,
    pub proj: Tensor,
    pub ln2: Tensor,
}

pub struct MlpView {
    pub w1: Tensor,
    pub w3: Tensor,
    pub w2: Tensor,
}

pub struct TrunkView {
    pub emb: Tensor,
    pub prelude: Vec<(AttnView, MlpView)>,
    pub recur: AttnView,
    pub router: Tensor,
    pub depth_emb: Tensor,
    pub adapter: Tensor,
    pub ln_f: Tensor,
    pub halt_w: Tensor,
    pub halt_b: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use minagi_types::Preset;

    fn tiny() -> ModelConfig {
        let mut c = Preset::Tiny.model();
        c.d_model = 32;
        c.n_head = 2;
        c.d_ff = 64;
        c.pool_experts = 8;
        c.pool_d_ff = 16;
        c
    }

    fn host(v: &Var) -> Vec<f32> {
        v.flatten_all().unwrap().to_vec1().unwrap()
    }

    fn std_of(x: &[f32]) -> f32 {
        let m = x.iter().sum::<f32>() / x.len() as f32;
        (x.iter().map(|v| (v - m).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn initialisation_follows_the_reference_recipe() {
        let cfg = tiny();
        let t = Trunk::init(&cfg, &mut HostRng::new(1), &Device::Cpu).unwrap();
        let depth = (cfg.n_prelude + cfg.n_recur + cfg.n_coda) as f32;
        assert!((std_of(&host(&t.emb)) - 0.02).abs() < 0.002);
        assert!((std_of(&host(&t.prelude[0].0.qkv)) - 0.02).abs() < 0.002);
        let want_out = 0.02 / (2.0 * depth).sqrt();
        assert!((std_of(&host(&t.prelude[0].0.proj)) - want_out).abs() < 0.002);
        assert!((std_of(&host(&t.prelude[0].1.w2)) - want_out).abs() < 0.002);
        assert!(
            (std_of(&host(&t.router)) - 0.02).abs() < 0.003,
            "the router is re-initialised at 0.02 by the model-wide init"
        );
        assert!(host(&t.depth_emb).iter().all(|&x| x == 0.0));
        assert!(host(&t.ln_f).iter().all(|&x| x == 1.0));
        assert_eq!(host(&t.halt_b), vec![-2.0]);
        assert!(std_of(&host(&t.halt_w)) < 0.001, "the halting weight starts at 1% of the usual scale");
        // the adapter is [I | I]: out = h + x on the first pass
        let a = t.adapter.to_vec2::<f32>().unwrap();
        let d = cfg.d_model as usize;
        for (i, row) in a.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let want = if j == i || j == d + i { 1.0 } else { 0.0 };
                assert_eq!(v, want, "adapter[{i}][{j}]");
            }
        }
    }

    #[test]
    fn same_seed_same_weights() {
        let cfg = tiny();
        let a = Trunk::init(&cfg, &mut HostRng::new(7), &Device::Cpu).unwrap();
        let b = Trunk::init(&cfg, &mut HostRng::new(7), &Device::Cpu).unwrap();
        let c = Trunk::init(&cfg, &mut HostRng::new(8), &Device::Cpu).unwrap();
        assert_eq!(host(&a.emb), host(&b.emb));
        assert_ne!(host(&a.emb), host(&c.emb));
    }

    #[test]
    fn names_match_the_reference_state_dict_and_groups_split_trunk_from_pool() {
        let cfg = tiny();
        let t = Trunk::init(&cfg, &mut HostRng::new(1), &Device::Cpu).unwrap();
        let named = t.named();
        let names: Vec<&str> = named.iter().map(|p| p.name.as_str()).collect();
        for want in [
            "tok_emb.weight",
            "prelude.0.attn.qkv.weight",
            "prelude.0.mlp.w2.weight",
            "recur.0.attn.proj.weight",
            "recur.0.mlp.router.weight",
            "recur.0.mlp.depth_emb",
            "adapter.weight",
            "halt.bias",
        ] {
            assert!(names.contains(&want), "missing {want}");
        }
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
        let pool: Vec<&str> = named.iter().filter(|p| p.group == Group::Pool).map(|p| p.name.as_str()).collect();
        assert_eq!(pool, vec!["recur.0.mlp.router.weight", "recur.0.mlp.depth_emb"]);
    }

    #[test]
    fn untracked_views_record_nothing_and_still_see_updates() {
        let cfg = tiny();
        let t = Trunk::init(&cfg, &mut HostRng::new(1), &Device::Cpu).unwrap();
        let v = t.view(false);
        assert!(!v.emb.is_variable());
        let before = v.halt_b.to_vec1::<f32>().unwrap();
        t.halt_b.set(&Tensor::from_vec(vec![5.0f32], 1, &Device::Cpu).unwrap()).unwrap();
        assert_eq!(before, vec![-2.0]);
        assert_eq!(v.halt_b.to_vec1::<f32>().unwrap(), vec![5.0], "a detached view shares storage with the variable");
        assert!(t.view(true).emb.is_variable());
    }
}
