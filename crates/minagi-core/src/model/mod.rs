//! The network: trunk parameters ([`params`]), the forward pass ([`forward`]) and the model that ties them to the
//! expert pool.

pub mod ckpt;
pub mod forward;
pub mod params;
#[cfg(test)]
pub(crate) mod tests;

use candle_core::{Device, Tensor, Var};
use minagi_types::{ModelConfig, MoeMode, TrainConfig};

pub use ckpt::CkptOut;
pub use forward::{Caches, FwdIn, FwdOut, sample_depth};
pub use params::{Group, NamedVar, Trunk, TrunkView};

use crate::error::{EngineError, Result};
use crate::moe::{ExpertEntry, ExpertStore, Lineage, PagedPool};
use crate::ops::{self, Kernel};
use crate::optim::adamw::AdamW;
use crate::rng::HostRng;

/// Name the pool's gate parameter goes by in optimiser state and checkpoints.
pub const GATE_NAME: &str = "pool.gate";
/// Name of the router weight (one row per expert).
pub const ROUTER_NAME: &str = "recur.0.mlp.router.weight";

/// How the network is *run* (as opposed to its shape, which is fixed in [`ModelConfig`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelOpts {
    pub kernel: Kernel,
    /// Largest share of a step one expert may take, as a multiple of its fair share; 0 removes the limit.
    pub capacity_factor: f64,
    pub moe_mode: MoeMode,
    /// Weight of the router z-loss inside the auxiliary loss.
    pub z_weight: f32,
}

impl ModelOpts {
    pub fn from_train(t: &TrainConfig, kernel: Kernel) -> Self {
        Self { kernel, capacity_factor: t.capacity_factor, moe_mode: t.moe_mode, z_weight: 1e-3 }
    }
}

impl Default for ModelOpts {
    fn default() -> Self {
        Self { kernel: Kernel::hand(), capacity_factor: 1.5, moe_mode: MoeMode::SparseDispatch, z_weight: 1e-3 }
    }
}

pub struct Model {
    pub cfg: ModelConfig,
    pub opts: ModelOpts,
    pub trunk: Trunk,
    pub pool: PagedPool,
    pub dev: Device,
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
    pub(crate) mask: Tensor,
}

impl Model {
    /// A fresh model: random weights, and every expert written to `store`. `resident` is the number of slots on the card.
    pub fn create(
        dev: &Device,
        cfg: ModelConfig,
        opts: ModelOpts,
        resident: usize,
        mut store: Box<dyn ExpertStore>,
        seed: u64,
    ) -> Result<Self> {
        cfg.validate().map_err(EngineError::Config)?;
        let mut rng = HostRng::fork(seed, 0);
        let trunk = Trunk::init(&cfg, &mut rng, dev)?;
        let (d, ff, n) = (cfg.d_model as usize, cfg.pool_d_ff as usize, cfg.pool_experts as usize);
        let depth = (cfg.n_prelude + cfg.n_recur + cfg.n_coda) as f32;
        let std_out = 0.02 / (2.0 * depth).sqrt();
        let mut erng = HostRng::fork(seed, 1);
        for uid in 0..n {
            store.put(
                uid as u64,
                ExpertEntry {
                    w1: erng.normal_vec(ff * d, 0.02),
                    w3: erng.normal_vec(ff * d, 0.02),
                    w2: erng.normal_vec(d * ff, std_out),
                    moments: None,
                },
                true,
            )?;
        }
        store.flush()?;
        let pool = PagedPool::new(dev, store, d, ff, n, resident)?;
        Self::assemble(dev, cfg, opts, trunk, pool)
    }

    /// A model from parts (used when loading a checkpoint).
    pub fn assemble(dev: &Device, cfg: ModelConfig, opts: ModelOpts, trunk: Trunk, pool: PagedPool) -> Result<Self> {
        let head_dim = (cfg.d_model / cfg.n_head) as usize;
        let block = cfg.block as usize;
        let (cos, sin) = ops::rope_tables(block, head_dim, 10_000.0, dev)?;
        let mask = ops::causal_mask(block, dev)?;
        Ok(Self { cfg, opts, trunk, pool, dev: dev.clone(), cos, sin, mask })
    }

    pub fn new_caches(&self) -> Caches {
        Caches::new(self.cfg.kv_slots() as usize)
    }

    /// Every parameter outside the expert slots, with its name and group.
    pub fn named_vars(&self) -> Vec<NamedVar> {
        let mut v = self.trunk.named();
        v.push(NamedVar { name: GATE_NAME.into(), var: self.pool.gate.clone(), group: Group::Pool });
        v
    }

    /// Every variable that takes part in clipping and the optimiser step: the named ones and the three slot tensors.
    pub fn all_vars(&self) -> Vec<Var> {
        let mut v: Vec<Var> = self.named_vars().into_iter().map(|p| p.var).collect();
        v.extend(self.pool.slot_vars().into_iter().cloned());
        v
    }

    /// Parameters in the trunk (everything except the expert pool's experts and gates).
    pub fn trunk_params(&self) -> usize {
        self.trunk.named().iter().filter(|p| p.group == Group::Trunk).map(|p| p.var.elem_count()).sum()
    }

    /// Parameters in the whole pool as stored (every expert, not just the ones on the card).
    pub fn total_params(&self) -> usize {
        self.trunk.n_params() + self.pool.n_experts() * self.pool.params_per_expert() + self.pool.n_experts()
    }

    // ---- growing and pruning the pool -------------------------------------------------------------------------

    /// Add `k` experts by recombination and give every router a row for each: the unit-weighted average of its
    /// parents' rows, so the newborn is asked for where its parents are. Optimiser state grows to match.
    pub fn grow_pool(
        &mut self,
        k: usize,
        seed_from: Option<usize>,
        step: f64,
        birth_gate: f32,
        opt: &mut AdamW,
        rng: &mut HostRng,
    ) -> Result<Vec<Option<Lineage>>> {
        let lineage = self.pool.add_experts(k, seed_from, step, birth_gate, 16, rng)?;
        let d = self.cfg.d_model as usize;
        let old = self.trunk.router.as_tensor().clone();
        let n_old = old.dim(0)?;
        let host: Vec<f32> = old.flatten_all()?.to_vec1()?;
        let mut extra = Vec::with_capacity(k * d);
        for l in &lineage {
            match l {
                Some(l) => {
                    let mut row = vec![0f32; d];
                    for (&p, &share) in l.parents.iter().zip(&l.share) {
                        for (r, &w) in row.iter_mut().zip(&host[p * d..(p + 1) * d]) {
                            *r += share * w;
                        }
                    }
                    extra.extend(row);
                }
                None => extra.extend(rng.normal_vec(d, 0.01)),
            }
        }
        let extra = Tensor::from_vec(extra, (k, d), &self.dev)?;
        self.trunk.router = Var::from_tensor(&Tensor::cat(&[&old, &extra], 0)?.contiguous()?)?;
        debug_assert_eq!(self.trunk.router.dim(0)?, n_old + k);
        opt.grow_rows(ROUTER_NAME, k)?;
        opt.grow_rows(GATE_NAME, k)?;
        Ok(lineage)
    }

    /// Delete experts nothing has addressed for a whole survival window; routers and optimiser state follow.
    /// Returns the positions kept, or `None` when nothing was deleted.
    pub fn prune_pool(
        &mut self,
        step: f64,
        survival: f64,
        protect: usize,
        opt: &mut AdamW,
    ) -> Result<Option<Vec<usize>>> {
        let Some(keep) = self.pool.prune(step, survival, protect)? else { return Ok(None) };
        let idx = Tensor::from_vec(keep.iter().map(|&i| i as u32).collect::<Vec<_>>(), keep.len(), &self.dev)?;
        let router = self.trunk.router.as_tensor().index_select(&idx, 0)?.contiguous()?;
        // A fresh Var rather than reshaping the old one: autograd sizes a gradient from the tensor it saved.
        self.trunk.router = Var::from_tensor(&router)?;
        opt.select_rows(ROUTER_NAME, &keep)?;
        opt.select_rows(GATE_NAME, &keep)?;
        Ok(Some(keep))
    }
}
