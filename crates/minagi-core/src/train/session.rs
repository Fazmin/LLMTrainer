//! Creating, saving and loading a run's model: the glue between [`Model`] and the on-disk checkpoint layout.
//!
//! A run owns two kinds of expert files. The **live** directory (`<run>/live/experts`) is the training working set:
//! the tiers write experts back to it as they are evicted. A **checkpoint** (`<run>/checkpoints/step-N`) is a complete
//! snapshot whose expert files are hard links to the live ones. That is safe because the tiers never rewrite a file in
//! place (a write-back replaces the *name*), so a checkpoint's file stays as it was. Resuming rebuilds the live
//! directory from a checkpoint and never trusts what was left there.

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor as CTensor, Var};
use minagi_types::{ModelConfig, Preset, TrainConfig};
use serde_json::Value;

use crate::error::{EngineError, Result};
use crate::model::{GATE_NAME, Model, ModelOpts, Trunk};
use crate::moe::{PagedPool, TiersStore};
use crate::optim::adamw::{AdamHyper, AdamW};
use crate::store::checkpoint::{
    Checkpoint, CheckpointData, ExpertSource, NamedTensors, OptimState, ParamState, ReadOptions, Tensor, WriteOptions,
    WriteReport, is_slot_indexed,
};
use crate::store::manifest::{Manifest, MinagiRs, ParsedCfg, suggest_preset};
use crate::store::tiers::{TierConfig, Tiers};

/// Where a run keeps its files.
#[derive(Debug, Clone)]
pub struct RunPaths {
    pub run_dir: PathBuf,
}

impl RunPaths {
    pub fn new(run_dir: impl Into<PathBuf>) -> Self {
        Self { run_dir: run_dir.into() }
    }

    pub fn live_experts(&self) -> PathBuf {
        self.run_dir.join("live").join("experts")
    }

    pub fn checkpoints(&self) -> PathBuf {
        self.run_dir.join("checkpoints")
    }

    /// Directory name of the checkpoint for `step`, relative to the run directory.
    pub fn checkpoint_rel(step: u64) -> String {
        format!("checkpoints/step-{step:09}")
    }
}

/// What the training loop carries that is not a tensor.
#[derive(Debug, Clone, Default)]
pub struct Saved {
    pub step: u64,
    /// Characters read over the model's whole life.
    pub chars: u64,
    /// Information read, in nats (the loss integrated over what was read).
    pub nats: f64,
    /// Held-out score of this state, if one was measured.
    pub val: Option<f64>,
    pub plasticity: Option<Value>,
    pub context_now: Option<u32>,
}

fn open_live(paths: &RunPaths, cfg: &ModelConfig, ram_cache: u32) -> Result<Tiers> {
    let tc = TierConfig::new(paths.live_experts(), cfg.d_model as usize, cfg.pool_d_ff as usize)
        .ram_capacity(ram_cache.max(1) as usize);
    Ok(Tiers::open(tc)?)
}

/// A new model with random weights; its experts are written to the run's live directory.
pub fn create_model(
    dev: &Device,
    cfg: &ModelConfig,
    train: &TrainConfig,
    opts: ModelOpts,
    paths: &RunPaths,
    seed: u64,
) -> Result<Model> {
    let live = paths.live_experts();
    if live.exists() {
        std::fs::remove_dir_all(live.parent().unwrap_or(&live))?;
    }
    let tiers = open_live(paths, cfg, train.ram_cache)?;
    Model::create(dev, cfg.clone(), opts, train.resident as usize, Box::new(TiersStore::new(tiers)), seed)
}

fn to_host(v: &Var) -> Result<Tensor> {
    let shape = v.dims().to_vec();
    Ok(Tensor { shape, data: v.as_tensor().flatten_all()?.to_vec1::<f32>()? })
}

fn host_of(t: &CTensor) -> Result<Tensor> {
    Ok(Tensor { shape: t.dims().to_vec(), data: t.flatten_all()?.to_vec1::<f32>()? })
}

/// `core.npz`: the trunk, with the tied embedding stored twice as the Python reference does.
pub fn export_core(model: &Model) -> Result<NamedTensors> {
    let mut core = NamedTensors::new();
    for p in model.trunk.named() {
        if p.group == crate::model::Group::Trunk {
            core.insert(p.name.clone(), to_host(&p.var)?);
            if p.name == "tok_emb.weight" {
                core.insert("head.weight", to_host(&p.var)?);
            }
        }
    }
    Ok(core)
}

/// `routers.npz`: one router row per expert, the depth embedding and the gates.
pub fn export_routers(model: &Model) -> Result<NamedTensors> {
    let mut r = NamedTensors::new();
    r.insert(crate::model::ROUTER_NAME, to_host(&model.trunk.router)?);
    r.insert("recur.0.mlp.depth_emb", to_host(&model.trunk.depth_emb)?);
    r.insert(GATE_NAME, to_host(&model.pool.gate)?);
    Ok(r)
}

/// `optim.npz`: Adam's moments for everything that is not an expert, and the step count of the slot tensors (their
/// moments belong to the experts, which keep them in their own files).
pub fn export_optim(model: &Model, opt: &AdamW) -> Result<OptimState> {
    let t = opt.t() as f64;
    let mut params = Vec::new();
    for p in model.named_vars() {
        let (m, v) = match opt.moments(&p.name) {
            Some((m, v)) => (Some(host_of(m)?), Some(host_of(v)?)),
            None => (None, None),
        };
        params.push(ParamState { name: p.name, m, v, step: Some(t) });
    }
    for name in ["pool.w1", "pool.w3", "pool.w2"] {
        params.push(ParamState {
            name: name.into(),
            m: None,
            v: None,
            step: Some(model.pool.steps().max(opt.t()) as f64),
        });
    }
    Ok(OptimState { params })
}

/// Write a complete checkpoint at `dest` (which must not exist). Resident experts that changed are parked first, so
/// the files the checkpoint links to are current.
pub fn write_checkpoint(
    model: &mut Model,
    opt: &AdamW,
    saved: &Saved,
    preset: Option<Preset>,
    dest: &Path,
) -> Result<WriteReport> {
    model.pool.park_changed()?;
    let n = model.pool.n_experts();
    let mut cfg = model.cfg.clone();
    cfg.pool_max = cfg.pool_max.max(n as u32);
    let resident = model.pool.resident() as u32;
    let mut manifest = Manifest::new(saved.step as i64, saved.val, &ParsedCfg::from_model(cfg, resident));
    manifest.telemetry = Some(model.pool.book.telemetry(&model.pool.gate_values()?));
    manifest.read_chars = Some(saved.chars);
    manifest.read_nats = Some(saved.nats);
    manifest.plasticity = saved.plasticity.clone();
    manifest.context_now = saved.context_now.map(u64::from);
    manifest.minagi_rs = Some(MinagiRs::native(preset));
    let data = CheckpointData {
        manifest,
        core: export_core(model)?,
        routers: export_routers(model)?,
        optim: export_optim(model, opt)?,
    };
    let opts = WriteOptions { overwrite: false, durable: true };
    let uids = model.pool.book.uid.clone();
    let store = model.pool.store_mut();
    if let Some(tiers) = store.tiers_mut() {
        Ok(Checkpoint::write_with(dest, &data, ExpertSource::Tiers(tiers), &opts)?)
    } else {
        let mut entries = Vec::with_capacity(uids.len());
        for uid in uids {
            let e = store.fetch(uid, false)?;
            entries.push((uid, std::sync::Arc::new(crate::moe::tiers_store::to_tiers(e))));
        }
        Ok(Checkpoint::write_with(dest, &data, ExpertSource::Entries(&entries), &opts)?)
    }
}

/// Load a checkpoint into a model that can keep training: the experts are linked into the run's live directory, the
/// trunk and routers become variables, and Adam's moments are restored.
pub fn load_model(
    dev: &Device,
    ck: &Checkpoint,
    train: &TrainConfig,
    opts: ModelOpts,
    paths: &RunPaths,
) -> Result<(Model, AdamW, Saved)> {
    let man = &ck.data.manifest;
    let parsed = man
        .parsed_cfg()
        .map_err(|e| EngineError::Checkpoint(format!("the checkpoint's settings are unreadable: {e}")))?
        .ok_or_else(|| EngineError::Checkpoint("the checkpoint does not say what shape the model has".into()))?;
    if let Some(problem) = parsed.python_only_problems().into_iter().next() {
        return Err(EngineError::Unsupported(problem));
    }
    let cfg = parsed.model.clone();
    cfg.validate().map_err(EngineError::Unsupported)?;
    let missing = ck.missing_experts();
    if !missing.is_empty() {
        return Err(EngineError::Checkpoint(format!(
            "{} expert file(s) are missing from the checkpoint (first: e{:05})",
            missing.len(),
            missing[0]
        )));
    }
    let live = paths.live_experts();
    if let Some(parent) = live.parent()
        && parent.exists()
    {
        std::fs::remove_dir_all(parent)?;
    }
    ck.materialize_experts(&live)?;
    let tiers = open_live(paths, &cfg, train.ram_cache)?;
    let trunk = Trunk::load(&cfg, &ck.data.core, &ck.data.routers, dev)?;
    let n = man.n_experts.max(trunk.router.dim(0)?);
    let gate =
        ck.data.routers.get(GATE_NAME).ok_or_else(|| EngineError::Checkpoint("the checkpoint has no gates".into()))?;
    if gate.shape != [n] {
        return Err(EngineError::Checkpoint(format!(
            "the checkpoint has {} gates for {n} experts",
            gate.shape.iter().product::<usize>()
        )));
    }
    let resident = (train.resident as usize).min(n).max(1);
    let mut pool = PagedPool::new(
        dev,
        Box::new(TiersStore::new(tiers)),
        cfg.d_model as usize,
        cfg.pool_d_ff as usize,
        n,
        resident,
    )?;
    pool.gate = Var::from_tensor(&CTensor::from_vec(gate.data.clone(), n, dev)?)?;
    if let Some(t) = &man.telemetry {
        pool.book.load_telemetry(t);
    }
    pool.reset_clock(0);
    let mut model = Model::assemble(dev, cfg, opts, trunk, pool)?;

    // Adam: moments of everything but the experts, and the shared step count.
    let hyper = AdamHyper { beta1: train.beta1, beta2: train.beta2, eps: 1e-8 };
    let mut opt = AdamW::new(hyper, train.weight_decay);
    let mut t_seen = 0f64;
    for p in model.named_vars() {
        let Some(st) = ck.data.optim.get(&p.name) else { continue };
        t_seen = t_seen.max(st.step.unwrap_or(0.0));
        if let (Some(m), Some(v)) = (&st.m, &st.v) {
            let shape = p.var.dims().to_vec();
            if m.shape == shape && v.shape == shape {
                opt.put_moments(
                    &p.name,
                    CTensor::from_vec(m.data.clone(), shape.clone(), dev)?,
                    CTensor::from_vec(v.data.clone(), shape, dev)?,
                );
            }
        }
    }
    for st in ck.data.optim.params.iter().filter(|p| is_slot_indexed(&p.name)) {
        t_seen = t_seen.max(st.step.unwrap_or(0.0));
    }
    opt.set_t(t_seen as u64);
    model.pool.reset_clock(t_seen as u64);

    let saved = Saved {
        step: man.step.unwrap_or(0).max(0) as u64,
        chars: man.read_chars.unwrap_or(0),
        nats: man.read_nats.unwrap_or(0.0),
        val: man.val,
        plasticity: man.plasticity.clone(),
        context_now: man.context_now.map(|c| c as u32),
    };
    Ok((model, opt, saved))
}

/// Read a checkpoint directory (without its slot-indexed moments, which nothing needs).
pub fn read_checkpoint(dir: &Path) -> Result<Checkpoint> {
    Ok(Checkpoint::read_with(dir, &ReadOptions::default())?)
}

/// The preset a checkpoint's shape matches, for display.
pub fn preset_of(cfg: &ModelConfig) -> Preset {
    suggest_preset(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FwdIn;
    use crate::model::tests::{text, tiny_cfg};
    use crate::ops::Kernel;
    use minagi_types::MoeMode;

    fn train_cfg() -> TrainConfig {
        let mut t = Preset::Tiny.train();
        t.resident = 4;
        t.ram_cache = 8;
        t
    }

    fn opts() -> ModelOpts {
        ModelOpts { kernel: Kernel::plain(), capacity_factor: 0.0, moe_mode: MoeMode::SparseDispatch, z_weight: 1e-3 }
    }

    fn loss_of(m: &mut Model, x: &[u32], y: &[u32]) -> f32 {
        let out = m
            .forward(FwdIn {
                ids: x,
                targets: Some(y),
                caches: None,
                pos_offset: 0,
                n_steps: 4,
                train: false,
                want_logits: false,
            })
            .unwrap();
        crate::ops::scalar(&out.loss.unwrap()).unwrap()
    }

    #[test]
    fn a_checkpoint_round_trips_weights_experts_telemetry_and_optimiser() {
        let dir = tempfile::tempdir().unwrap();
        let paths = RunPaths::new(dir.path());
        let cfg = tiny_cfg();
        let mut model = create_model(&Device::Cpu, &cfg, &train_cfg(), opts(), &paths, 3).unwrap();
        // put some history into the pool and some state into the optimiser
        let (x, y) = (text(40, 7), text(40, 11));
        model.pool.book.now = 17.0;
        let mut opt = AdamW::new(AdamHyper::default(), 0.1);
        let g = model.trunk.halt_b.as_tensor().sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let t = opt.advance();
        opt.update(&[crate::optim::adamw::Param { name: "halt.bias", var: &model.trunk.halt_b, lr: 0.01 }], &g, t, 1.0)
            .unwrap();
        let before = loss_of(&mut model, &x, &y); // after the optimiser step: that is the state being saved
        model.pool.book.last_seen[2] = 5.0;
        let saved = Saved {
            step: 17,
            chars: 8704,
            nats: 123.5,
            val: Some(4.2),
            plasticity: Some(serde_json::json!({"scale": 0.5})),
            context_now: Some(40),
        };
        let dest = paths.checkpoints().join("step-000000017");
        std::fs::create_dir_all(paths.checkpoints()).unwrap();
        let report = write_checkpoint(&mut model, &opt, &saved, Some(Preset::Tiny), &dest).unwrap();
        assert_eq!(report.n_experts, 8);
        assert!(dest.join("COMPLETE").is_file());

        let ck = read_checkpoint(&dest).unwrap();
        assert_eq!(ck.data.manifest.step, Some(17));
        assert_eq!(ck.data.manifest.read_chars, Some(8704));
        assert!(ck.data.core.contains("head.weight") && ck.data.core.contains("tok_emb.weight"));
        assert!(ck.verify().is_empty(), "{:?}", ck.verify());

        // load into a fresh run directory and compare
        let dir2 = tempfile::tempdir().unwrap();
        let paths2 = RunPaths::new(dir2.path());
        let (mut back, opt2, saved2) = load_model(&Device::Cpu, &ck, &train_cfg(), opts(), &paths2).unwrap();
        assert_eq!(saved2.step, 17);
        assert_eq!(saved2.context_now, Some(40));
        assert_eq!(saved2.plasticity, saved.plasticity);
        assert_eq!(saved2.val, Some(4.2));
        assert_eq!(opt2.t(), 1);
        assert!(opt2.moments("halt.bias").is_some(), "Adam's moments came back");
        assert_eq!(back.pool.book.last_seen[2], 5.0, "the prune clock came back");
        assert_eq!(back.pool.n_experts(), 8);
        let after = loss_of(&mut back, &x, &y);
        assert!((before - after).abs() < 1e-5, "the same model scores the same: {before} vs {after}");
        // the checkpoint is independent of the live directory: training on does not change it
        model.pool.book.begin_text(false);
        let again = read_checkpoint(&dest).unwrap();
        assert!(again.verify().is_empty());
    }

    #[test]
    fn a_checkpoint_missing_an_expert_is_refused_with_a_clear_message() {
        let dir = tempfile::tempdir().unwrap();
        let paths = RunPaths::new(dir.path());
        let mut model = create_model(&Device::Cpu, &tiny_cfg(), &train_cfg(), opts(), &paths, 3).unwrap();
        let opt = AdamW::new(AdamHyper::default(), 0.1);
        let dest = paths.checkpoints().join("c");
        std::fs::create_dir_all(paths.checkpoints()).unwrap();
        write_checkpoint(&mut model, &opt, &Saved::default(), None, &dest).unwrap();
        std::fs::remove_file(dest.join("experts").join("e00003.npz")).unwrap();
        let ck = read_checkpoint(&dest).unwrap();
        let err =
            load_model(&Device::Cpu, &ck, &train_cfg(), opts(), &RunPaths::new(tempfile::tempdir().unwrap().path()))
                .err()
                .unwrap();
        assert!(err.to_string().contains("missing"), "{err}");
    }
}

#[cfg(test)]
mod safetensors_tests {
    use super::*;
    use crate::model::tests::tiny_cfg;
    use crate::store::safetensors::{export_safetensors, read_header};

    #[test]
    fn a_checkpoint_exports_as_a_valid_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        let paths = RunPaths::new(dir.path());
        let mut train = Preset::Tiny.train();
        train.resident = 4;
        let opts = ModelOpts {
            kernel: crate::ops::Kernel::plain(),
            capacity_factor: 0.0,
            moe_mode: minagi_types::MoeMode::SparseDispatch,
            z_weight: 1e-3,
        };
        let mut model = create_model(&Device::Cpu, &tiny_cfg(), &train, opts, &paths, 3).unwrap();
        let opt = AdamW::new(AdamHyper::default(), 0.1);
        let ck = paths.checkpoints().join("c");
        std::fs::create_dir_all(paths.checkpoints()).unwrap();
        write_checkpoint(&mut model, &opt, &Saved::default(), None, &ck).unwrap();
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let rep = export_safetensors(&ck, &out).unwrap();
        assert_eq!(rep.experts, 8);
        let file = out.join("model.safetensors");
        assert_eq!(std::fs::metadata(&file).unwrap().len(), rep.bytes);
        let hdr = read_header(&file).unwrap();
        assert_eq!(hdr.len(), rep.tensors);
        let find = |n: &str| hdr.iter().find(|h| h.0 == n).unwrap_or_else(|| panic!("missing {n}"));
        assert_eq!(find("tok_emb.weight").2, vec![265, 32]);
        assert_eq!(find("pool.experts.3.w2").2, vec![32, 16]);
        assert!(hdr.iter().all(|h| h.1 == "F32" && h.4 - h.3 == h.2.iter().product::<usize>() as u64 * 4));
        assert!(hdr.iter().all(|h| h.0 != "head.weight"), "the tied head is stored once");
        // ranges tile the data section exactly
        let mut spans: Vec<(u64, u64)> = hdr.iter().map(|h| (h.3, h.4)).collect();
        spans.sort();
        assert_eq!(spans[0].0, 0);
        for w in spans.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        // the bytes are the weights: compare one expert's w1 with the store
        let bytes = std::fs::read(&file).unwrap();
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(hlen % 8, 0);
        let e3 = find("pool.experts.3.w1");
        let start = 8 + hlen + e3.3 as usize;
        let got: Vec<f32> =
            bytes[start..start + 4 * 16 * 32].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let want = model.pool.store_mut().fetch(3, false).unwrap().w1;
        assert_eq!(got, want);
    }
}
