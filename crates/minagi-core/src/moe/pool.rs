//! The expert pool on the accelerator: a fixed number of *slots* whose contents are swapped, in front of RAM and disk.
//!
//! Only `resident` experts exist as tensors at any moment (`w1`, `w3`, `w2` stack one expert per slot). Which expert
//! sits in which slot is decided by [`PoolBook`]; this module moves the bytes. Adam's moments belong to the *expert*,
//! not to the slot it happens to occupy, so they travel with it, but only to an optimiser step: an expert arrives on the
//! card with its weights alone, and just before each step every expert on the card that is not already holding its own
//! moments gets them from RAM or disk. A reply loads experts at almost every character and steps none of them, so
//! writing moves no moments at all.
//!
//! An expert nothing has trained since it was loaded is still exactly its copy in RAM or on disk, so parking it copies
//! nothing back. One that was stepped goes back with its weights and its moments.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};

use super::book::{PoolBook, SlotPlan};
use super::store::{ExpertEntry, ExpertMoments, ExpertStore};
use crate::error::{EngineError, Result};
use crate::optim::adamw::{AdamHyper, adam_update};
use crate::rng::HostRng;

/// Where a newborn expert came from: the parents its hidden units were taken from and how many units each gave.
#[derive(Debug, Clone, PartialEq)]
pub struct Lineage {
    /// Pool positions of the parents.
    pub parents: Vec<usize>,
    /// Share of the child's hidden units from each parent (sums to 1).
    pub share: Vec<f32>,
}

/// Adam moments of the three slot tensors (`[resident, ...]` each).
struct SlotMoments {
    m: [Tensor; 3],
    v: [Tensor; 3],
}

pub struct PagedPool {
    pub book: PoolBook,
    pub d_model: usize,
    pub d_ff: usize,
    /// Slot tensors: `w1`, `w3` are `[resident, d_ff, d_model]`, `w2` is `[resident, d_model, d_ff]`.
    pub w1: Var,
    pub w3: Var,
    pub w2: Var,
    /// One scale per expert in the pool (`[n]`); a newborn starts near zero and earns its way up.
    pub gate: Var,
    pub(crate) store: Box<dyn ExpertStore>,
    pub(crate) dev: Device,
    mom: SlotMoments,
    /// Whether each slot's moments are its expert's own.
    own: Vec<bool>,
    /// Optimiser steps taken on the slots.
    steps: u64,
    /// `steps` when each slot's expert was loaded; equal to `steps` means "unchanged since".
    loaded_at: Vec<u64>,
}

fn zeros3(res: usize, d_model: usize, d_ff: usize, dev: &Device) -> Result<[Tensor; 3]> {
    Ok([
        Tensor::zeros((res, d_ff, d_model), DType::F32, dev)?,
        Tensor::zeros((res, d_ff, d_model), DType::F32, dev)?,
        Tensor::zeros((res, d_model, d_ff), DType::F32, dev)?,
    ])
}

impl PagedPool {
    /// A pool of `n` experts (already in `store`) with `resident` slots, all empty.
    pub fn new(
        dev: &Device,
        store: Box<dyn ExpertStore>,
        d_model: usize,
        d_ff: usize,
        n: usize,
        resident: usize,
    ) -> Result<Self> {
        let book = PoolBook::new(n, resident);
        let res = book.resident;
        let [a, b, c] = zeros3(res, d_model, d_ff, dev)?;
        let mom = SlotMoments { m: zeros3(res, d_model, d_ff, dev)?, v: zeros3(res, d_model, d_ff, dev)? };
        Ok(Self {
            d_model,
            d_ff,
            w1: Var::from_tensor(&a)?,
            w3: Var::from_tensor(&b)?,
            w2: Var::from_tensor(&c)?,
            gate: Var::from_tensor(&Tensor::ones(n, DType::F32, dev)?)?,
            store,
            dev: dev.clone(),
            mom,
            own: vec![false; res],
            steps: 0,
            loaded_at: vec![0; res],
            book,
        })
    }

    pub fn n_experts(&self) -> usize {
        self.book.n
    }

    pub fn resident(&self) -> usize {
        self.book.resident
    }

    pub fn store_mut(&mut self) -> &mut dyn ExpertStore {
        self.store.as_mut()
    }

    pub fn store(&self) -> &dyn ExpertStore {
        self.store.as_ref()
    }

    /// Parameters of one expert: three matrices.
    pub fn params_per_expert(&self) -> usize {
        3 * self.d_model * self.d_ff
    }

    /// What the pool costs on disk (8 bytes per parameter: an f32 weight plus two bf16 moments) for `extra` more experts.
    pub fn disk_bytes(&self, extra: usize) -> u64 {
        (self.book.n + extra) as u64 * self.params_per_expert() as u64 * 8
    }

    pub fn vram_params(&self) -> usize {
        self.book.resident * self.params_per_expert() + self.book.n
    }

    /// The slot tensors (weights).
    pub fn slot_vars(&self) -> [&Var; 3] {
        [&self.w1, &self.w3, &self.w2]
    }

    fn uid(&self, expert: i64) -> u64 {
        self.book.uid[expert as usize]
    }

    // ---- moving experts between the card and the tiers ------------------------------------------------------

    fn read_rows(var: &Var, s: usize) -> Result<Vec<f32>> {
        Ok(var.as_tensor().narrow(0, s, 1)?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn read_moment_rows(t: &Tensor, s: usize) -> Result<Vec<f32>> {
        Ok(t.narrow(0, s, 1)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Slot `s` as expert `e`'s entry: its weights, and its moments (the optimiser's when they are its own there,
    /// otherwise the ones its entry already holds). An entry replaces the old one whole, so it never loses them.
    fn entry_from_slot(&mut self, s: usize, e: i64) -> Result<ExpertEntry> {
        let mut entry = ExpertEntry {
            w1: Self::read_rows(&self.w1, s)?,
            w3: Self::read_rows(&self.w3, s)?,
            w2: Self::read_rows(&self.w2, s)?,
            moments: None,
        };
        if self.own[s] {
            entry.moments = Some(ExpertMoments {
                w1_m: Self::read_moment_rows(&self.mom.m[0], s)?,
                w1_v: Self::read_moment_rows(&self.mom.v[0], s)?,
                w3_m: Self::read_moment_rows(&self.mom.m[1], s)?,
                w3_v: Self::read_moment_rows(&self.mom.v[1], s)?,
                w2_m: Self::read_moment_rows(&self.mom.m[2], s)?,
                w2_v: Self::read_moment_rows(&self.mom.v[2], s)?,
            });
        } else {
            entry.moments = self.store.fetch(self.uid(e), false)?.moments;
        }
        Ok(entry)
    }

    fn write_rows(&self, var: &Var, s: usize, data: &[f32], dims: (usize, usize)) -> Result<()> {
        let src = Tensor::from_slice(data, (1, dims.0, dims.1), &self.dev)?;
        var.as_tensor().slice_set(&src, 0, s)?;
        Ok(())
    }

    /// Put the card into the state `plan` describes: park what is leaving, fetch what is arriving, leave the rest.
    pub fn apply_plan(&mut self, plan: &SlotPlan) -> Result<()> {
        for (s, e) in plan.leaving() {
            if self.loaded_at[s] == self.steps {
                continue; // unchanged since it was loaded: its copy in RAM or on disk is still the truth
            }
            let entry = self.entry_from_slot(s, e)?;
            self.store.put(self.uid(e), entry, true)?;
        }
        let (dm, df) = (self.d_model, self.d_ff);
        for (s, e) in plan.arriving() {
            let entry = self.store.fetch(self.uid(e), true)?;
            entry.check(dm, df)?;
            self.write_rows(&self.w1, s, &entry.w1, (df, dm))?;
            self.write_rows(&self.w3, s, &entry.w3, (df, dm))?;
            self.write_rows(&self.w2, s, &entry.w2, (dm, df))?;
            self.own[s] = false;
            self.loaded_at[s] = self.steps;
        }
        self.book.set_slots(plan.new.clone());
        Ok(())
    }

    /// Run admission for this pass's requests and move the experts it brings in. Returns how many were loaded.
    pub fn admit(&mut self, mass: &[f32], merit: Option<&[f32]>) -> Result<usize> {
        let before = self.book.loads;
        if let Some(plan) = self.book.admit(mass, merit) {
            self.apply_plan(&plan)?;
        }
        Ok((self.book.loads - before) as usize)
    }

    // ---- the optimiser's view -----------------------------------------------------------------------------------

    /// Before every optimiser step: each expert on the card holds its own moments, and one that does not yet gets them
    /// now from its entry in RAM or on disk (or zeros if it has none).
    pub fn own_moments(&mut self) -> Result<()> {
        let (dm, df) = (self.d_model, self.d_ff);
        let slots = self.book.slots.clone();
        for (s, &e) in slots.iter().enumerate() {
            if e < 0 || self.own[s] {
                continue;
            }
            let entry = self.store.fetch(self.uid(e), false)?;
            let pairs: [(&Option<ExpertMoments>, usize); 3] =
                [(&entry.moments, 0), (&entry.moments, 1), (&entry.moments, 2)];
            for (mo, i) in pairs {
                let dims = if i < 2 { (1, df, dm) } else { (1, dm, df) };
                let (m, v): (Option<&[f32]>, Option<&[f32]>) = match mo {
                    Some(x) => match i {
                        0 => (Some(&x.w1_m), Some(&x.w1_v)),
                        1 => (Some(&x.w3_m), Some(&x.w3_v)),
                        _ => (Some(&x.w2_m), Some(&x.w2_v)),
                    },
                    None => (None, None),
                };
                let n = dims.1 * dims.2;
                let put = |dst: &Tensor, src: Option<&[f32]>| -> Result<()> {
                    let t = match src {
                        Some(d) => Tensor::from_slice(d, dims, &self.dev)?,
                        None => Tensor::zeros(dims, DType::F32, &self.dev)?,
                    };
                    debug_assert_eq!(t.elem_count(), n);
                    dst.slice_set(&t, 0, s)?;
                    Ok(())
                };
                put(&self.mom.m[i], m)?;
                put(&self.mom.v[i], v)?;
            }
            self.own[s] = true;
        }
        Ok(())
    }

    /// One AdamW step on the slot tensors, with every expert on the card using its own moments. `t` is the global step
    /// count from the optimiser; `lr`, `wd`, `gscale` as in [`adam_update`].
    pub fn step_slots(
        &mut self,
        grads: &GradStore,
        h: &AdamHyper,
        t: u64,
        lr: f64,
        wd: f64,
        gscale: f64,
    ) -> Result<()> {
        self.own_moments()?;
        for (i, var) in [&self.w1, &self.w3, &self.w2].into_iter().enumerate() {
            if let Some(g) = grads.get(var) {
                let (m, v) = adam_update(var, &self.mom.m[i], &self.mom.v[i], g, h, t, lr, wd, gscale)?;
                self.mom.m[i] = m;
                self.mom.v[i] = v;
            }
        }
        // Whatever each slot's moments were going in, they are its expert's now.
        self.steps += 1;
        self.own = self.book.slots.iter().map(|&e| e >= 0).collect();
        Ok(())
    }

    /// Optimiser steps taken on the slots.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Adam's moments of the three slot tensors (first and second), in `w1, w3, w2` order (for tests and diagnostics).
    pub fn slot_moments(&self) -> (&[Tensor; 3], &[Tensor; 3]) {
        (&self.mom.m, &self.mom.v)
    }

    /// Park every resident expert that has changed into RAM (dirty), so the store holds current weights.
    pub fn park_changed(&mut self) -> Result<()> {
        for (s, e) in self.book.slots.clone().into_iter().enumerate() {
            if e < 0 || self.loaded_at[s] == self.steps {
                continue;
            }
            let entry = self.entry_from_slot(s, e)?;
            self.store.put(self.uid(e), entry, true)?;
            self.loaded_at[s] = self.steps;
        }
        Ok(())
    }

    /// Park every resident expert that has changed, then write everything dirty to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.park_changed()?;
        self.store.flush()
    }

    /// Restore the step counter after loading a checkpoint (so slots read as "unchanged since loaded").
    pub fn reset_clock(&mut self, steps: u64) {
        self.steps = steps;
        self.loaded_at = vec![steps; self.book.resident];
        self.own = vec![false; self.book.resident];
    }

    // ---- gate ---------------------------------------------------------------------------------------------------

    pub fn gate_values(&self) -> Result<Vec<f32>> {
        Ok(self.gate.as_tensor().to_vec1::<f32>()?)
    }

    /// Gates of the resident experts in slot order (`[resident]`), differentiable.
    pub fn resident_gate(&self, rows: &Tensor) -> Result<Tensor> {
        Ok(self.gate.as_tensor().index_select(rows, 0)?)
    }

    // ---- growth and pruning -------------------------------------------------------------------------------------

    /// Write `k` new experts to the store and make them part of the pool. They are not loaded: a new expert only
    /// reaches the card if a forward admits it.
    ///
    /// Each is built by **recombination**: a hidden unit is the triple `(w1[u], w3[u], w2[:, u])` and units are
    /// interchangeable, so a child assembled from whole units taken from different parents keeps every unit's learned
    /// feature intact while computing a function nothing in the pool computes. Parents are a random sample of
    /// `recombine` experts with `seed_from` (the busiest) always among them. With fewer than two experts, or
    /// `recombine < 2`, the child is a noisy copy of `seed_from`, or small random weights if there is none.
    ///
    /// Returns each child's lineage (for its router row). The caller extends the routers and the optimiser state.
    pub fn add_experts(
        &mut self,
        k: usize,
        seed_from: Option<usize>,
        step: f64,
        birth_gate: f32,
        recombine: usize,
        rng: &mut HostRng,
    ) -> Result<Vec<Option<Lineage>>> {
        // The copies in RAM or on disk are current only for experts nothing has trained since they were loaded.
        self.park_changed()?;
        let (dm, df) = (self.d_model, self.d_ff);
        let n0 = self.book.n;
        let seed = seed_from.filter(|&s| s < n0);
        let parents: Option<(Vec<usize>, Vec<ExpertEntry>)> = if recombine >= 2 && n0 >= 2 {
            let m = recombine.min(n0);
            let mut pick = rng.sample_indices(n0, m);
            if let Some(s) = seed
                && !pick.contains(&s)
            {
                pick[0] = s;
            }
            let got = pick.iter().map(|&i| self.store.fetch(self.book.uid[i], true)).collect::<Result<Vec<_>>>()?;
            Some((pick, got))
        } else {
            None
        };
        let seed_entry = match (&parents, seed) {
            (None, Some(s)) => Some(self.store.fetch(self.book.uid[s], true)?),
            _ => None,
        };
        let mut lineage = Vec::with_capacity(k);
        let uids = self.book.grow(k, step);
        for &uid in &uids {
            let (entry, lin) = if let Some((pick, got)) = &parents {
                let who: Vec<usize> = (0..df).map(|_| rng.below(got.len())).collect();
                let mut e = ExpertEntry {
                    w1: vec![0.0; df * dm],
                    w3: vec![0.0; df * dm],
                    w2: vec![0.0; dm * df],
                    moments: None,
                };
                for (u, &w) in who.iter().enumerate() {
                    e.w1[u * dm..(u + 1) * dm].copy_from_slice(&got[w].w1[u * dm..(u + 1) * dm]);
                    e.w3[u * dm..(u + 1) * dm].copy_from_slice(&got[w].w3[u * dm..(u + 1) * dm]);
                    for r in 0..dm {
                        e.w2[r * df + u] = got[w].w2[r * df + u];
                    }
                }
                let mut share = vec![0f32; got.len()];
                for &w in &who {
                    share[w] += 1.0;
                }
                share.iter_mut().for_each(|s| *s /= df as f32);
                (e, Some(Lineage { parents: pick.clone(), share }))
            } else if let Some(src) = &seed_entry {
                let noisy =
                    |v: &[f32], rng: &mut HostRng| v.iter().map(|x| x + 0.02 * rng.normal() as f32).collect::<Vec<_>>();
                let e = ExpertEntry {
                    w1: noisy(&src.w1, rng),
                    w3: noisy(&src.w3, rng),
                    w2: noisy(&src.w2, rng),
                    moments: None,
                };
                (e, seed.map(|s| Lineage { parents: vec![s], share: vec![1.0] }))
            } else {
                let fresh = |n: usize, rng: &mut HostRng| rng.normal_vec(n, 0.02);
                (
                    ExpertEntry {
                        w1: fresh(df * dm, rng),
                        w3: fresh(df * dm, rng),
                        w2: fresh(dm * df, rng),
                        moments: None,
                    },
                    None,
                )
            };
            self.store.put(uid, entry, true)?;
            lineage.push(lin);
        }
        self.store.flush()?;
        let old = self.gate.as_tensor().clone();
        let extra = Tensor::full(birth_gate, k, &self.dev)?;
        self.gate = Var::from_tensor(&Tensor::cat(&[&old, &extra], 0)?.contiguous()?)?;
        Ok(lineage)
    }

    /// Delete experts that have atrophied (see [`PoolBook::prune_keep`]); their files go with them. Returns the
    /// positions kept, or `None` when nothing was deleted. The caller shrinks the routers and the optimiser state.
    pub fn prune(&mut self, step: f64, survival: f64, protect: usize) -> Result<Option<Vec<usize>>> {
        let keep = self.book.prune_keep(step, survival, protect);
        if keep.len() == self.book.n {
            return Ok(None);
        }
        for i in 0..self.book.n {
            if !keep.contains(&i) {
                self.store.delete(self.book.uid[i])?;
            }
        }
        let idx = Tensor::from_vec(keep.iter().map(|&i| i as u32).collect::<Vec<_>>(), keep.len(), &self.dev)?;
        self.gate = Var::from_tensor(&self.gate.as_tensor().index_select(&idx, 0)?.contiguous()?)?;
        self.book.select(&keep);
        Ok(Some(keep))
    }

    /// Check the invariants a healthy pool keeps (used by tests and by `debug_assert`s in the trainer).
    pub fn check(&self) -> Result<()> {
        let n = self.book.n;
        if self.gate.dims1()? != n || self.book.uid.len() != n || self.book.last_seen.len() != n {
            return Err(EngineError::other("pool arrays are out of step with each other"));
        }
        let mut seen = std::collections::HashSet::new();
        for &e in &self.book.slots {
            if e >= 0 && (e as usize >= n || !seen.insert(e)) {
                return Err(EngineError::other("a slot holds an expert twice or one that does not exist"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::store::MemStore;

    const DM: usize = 4;
    const DF: usize = 3;

    /// Expert `i` has all weights equal to `i + 1` (w1), `10 (i + 1)` (w3), `100 (i + 1)` (w2) so a slot's contents
    /// say exactly which expert is there.
    fn entry(i: usize) -> ExpertEntry {
        let f = (i + 1) as f32;
        ExpertEntry { w1: vec![f; DM * DF], w3: vec![10.0 * f; DM * DF], w2: vec![100.0 * f; DM * DF], moments: None }
    }

    fn pool(n: usize, resident: usize) -> PagedPool {
        let mut store = MemStore::new();
        for i in 0..n {
            store.put(i as u64, entry(i), false).unwrap();
        }
        PagedPool::new(&Device::Cpu, Box::new(store), DM, DF, n, resident).unwrap()
    }

    fn slot_ids(p: &PagedPool) -> Vec<i64> {
        // read back which expert each slot holds from its weights
        (0..p.resident())
            .map(|s| {
                let v = PagedPool::read_rows(&p.w1, s).unwrap()[0];
                if v == 0.0 { -1 } else { v as i64 - 1 }
            })
            .collect()
    }

    #[test]
    fn admission_pages_the_right_experts_into_the_slots() {
        let mut p = pool(6, 3);
        p.book.begin_text(false);
        let mut m = vec![0.0; 6];
        m[4] = 3.0;
        m[1] = 2.0;
        m[5] = 1.0;
        assert_eq!(p.admit(&m, None).unwrap(), 3);
        assert_eq!(p.book.slots, vec![4, 1, 5]);
        assert_eq!(slot_ids(&p), vec![4, 1, 5]);
        assert_eq!(PagedPool::read_rows(&p.w2, 0).unwrap()[0], 500.0);
        p.check().unwrap();
    }

    #[test]
    fn a_trained_expert_is_parked_with_its_weights_and_moments_when_it_leaves() {
        let mut p = pool(4, 2);
        p.book.begin_text(false);
        let mut m = vec![0.0; 4];
        m[0] = 2.0;
        m[1] = 1.0;
        p.admit(&m, None).unwrap();
        // pretend an optimiser step changed slot 0: new weights and moments
        let dev = Device::Cpu;
        p.w1.as_tensor().slice_set(&Tensor::full(7.0f32, (1, DF, DM), &dev).unwrap(), 0, 0).unwrap();
        p.mom.m[0].slice_set(&Tensor::full(0.5f32, (1, DF, DM), &dev).unwrap(), 0, 0).unwrap();
        p.steps += 1;
        p.own = vec![true, true];
        // a new text wants 2 and 3: expert 0 and 1 leave (0 changed, 1 changed too since steps moved)
        p.book.begin_text(false);
        let mut m2 = vec![0.0; 4];
        m2[2] = 2.0;
        m2[3] = 1.0;
        p.admit(&m2, None).unwrap();
        assert_eq!(slot_ids(&p), vec![2, 3]);
        let parked = p.store.fetch(0, false).unwrap();
        assert!(parked.w1.iter().all(|&x| x == 7.0), "the stepped weights were written back");
        assert!(parked.moments.unwrap().w1_m.iter().all(|&x| x == 0.5), "and its own moments with them");
    }

    #[test]
    fn an_expert_nothing_trained_is_not_written_back() {
        let mut p = pool(4, 2);
        p.book.begin_text(false);
        p.admit(&[2.0, 1.0, 0.0, 0.0], None).unwrap();
        // corrupt the store copy: if parking wrote the (unchanged) slot back it would overwrite this
        p.store
            .put(
                0,
                ExpertEntry {
                    w1: vec![-1.0; DM * DF],
                    w3: vec![-1.0; DM * DF],
                    w2: vec![-1.0; DM * DF],
                    moments: None,
                },
                false,
            )
            .unwrap();
        p.book.begin_text(false);
        p.admit(&[0.0, 0.0, 2.0, 1.0], None).unwrap();
        assert!(p.store.fetch(0, false).unwrap().w1.iter().all(|&x| x == -1.0));
    }

    #[test]
    fn moments_follow_the_expert_not_the_slot() {
        let mut p = pool(4, 2);
        // expert 3's stored moments are 9s; expert 0 has none
        let mut e3 = entry(3);
        e3.moments = Some(ExpertMoments {
            w1_m: vec![9.0; DM * DF],
            w1_v: vec![9.0; DM * DF],
            w3_m: vec![9.0; DM * DF],
            w3_v: vec![9.0; DM * DF],
            w2_m: vec![9.0; DM * DF],
            w2_v: vec![9.0; DM * DF],
        });
        p.store.put(3, e3, false).unwrap();
        p.book.begin_text(false);
        p.admit(&[1.0, 0.0, 0.0, 2.0], None).unwrap(); // slots: [3, 0]
        p.own_moments().unwrap();
        let m0 = PagedPool::read_moment_rows(&p.mom.m[0], 0).unwrap();
        let m1 = PagedPool::read_moment_rows(&p.mom.m[0], 1).unwrap();
        assert!(m0.iter().all(|&x| x == 9.0), "slot 0 now holds expert 3, with expert 3's moments");
        assert!(m1.iter().all(|&x| x == 0.0), "expert 0 has none: it starts from zero, not from a stranger's");
        assert_eq!(p.own, vec![true, true]);
    }

    #[test]
    fn growth_recombines_whole_units_and_the_gate_grows() {
        let mut p = pool(5, 2);
        let mut rng = HostRng::new(1);
        let lin = p.add_experts(2, Some(2), 40.0, 0.001, 4, &mut rng).unwrap();
        assert_eq!(p.n_experts(), 7);
        assert_eq!(p.book.uid.len(), 7);
        assert_eq!(p.book.born[5], 40.0);
        let g = p.gate_values().unwrap();
        assert_eq!(g.len(), 7);
        assert_eq!(&g[..5], &[1.0; 5]);
        assert_eq!(&g[5..], &[0.001, 0.001]);
        for (i, l) in lin.iter().enumerate() {
            let l = l.as_ref().unwrap();
            assert!(l.parents.contains(&2), "the busiest expert is always a parent");
            assert!((l.share.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            // every hidden unit of the child is a whole unit of one parent: w1 value identifies the parent, and
            // the matching w3 and w2 entries must come from the same parent
            let child = p.store.fetch(p.book.uid[5 + i], false).unwrap();
            for u in 0..DF {
                let w1 = child.w1[u * DM];
                let parent_value = w1; // expert j has w1 = j + 1
                assert!(child.w1[u * DM..(u + 1) * DM].iter().all(|&x| x == w1));
                assert!(child.w3[u * DM..(u + 1) * DM].iter().all(|&x| x == 10.0 * parent_value));
                for r in 0..DM {
                    assert_eq!(child.w2[r * DF + u], 100.0 * parent_value);
                }
            }
        }
        p.check().unwrap();
    }

    #[test]
    fn growth_uses_the_latest_weights_of_resident_experts() {
        let mut p = pool(3, 2);
        p.book.begin_text(false);
        p.admit(&[2.0, 1.0, 0.0], None).unwrap();
        p.w1.as_tensor().slice_set(&Tensor::full(50.0f32, (1, DF, DM), &Device::Cpu).unwrap(), 0, 0).unwrap();
        p.steps += 1;
        let mut rng = HostRng::new(3);
        p.add_experts(1, Some(0), 1.0, 0.001, 0, &mut rng).unwrap(); // recombine off: a noisy copy of expert 0
        let child = p.store.fetch(p.book.uid[3], false).unwrap();
        assert!(
            child.w1.iter().all(|&x| (x - 50.0).abs() < 0.5),
            "copied from the trained weights, not the stale file"
        );
    }

    #[test]
    fn pruning_deletes_files_and_renumbers() {
        let mut p = pool(5, 2);
        p.book.begin_text(false);
        p.admit(&[2.0, 0.0, 0.0, 1.0, 0.0], None).unwrap(); // slots [0, 3]
        p.book.segments = 1000;
        p.book.born = vec![0.0; 5];
        p.book.last_seen = vec![1000.0, 0.0, 0.0, 1000.0, 1000.0];
        let keep = p.prune(1000.0, 100.0, 0).unwrap().unwrap();
        assert_eq!(keep, vec![0, 3, 4]);
        assert_eq!(p.n_experts(), 3);
        assert_eq!(p.book.slots, vec![0, 1]);
        assert_eq!(p.book.uid, vec![0, 3, 4]);
        assert_eq!(p.gate_values().unwrap().len(), 3);
        assert!(p.store.fetch(1, false).is_err() && p.store.fetch(2, false).is_err());
        assert!(p.store.fetch(4, false).is_ok());
        assert!(p.prune(1000.0, 100.0, 0).unwrap().is_none(), "nothing more to delete");
        p.check().unwrap();
    }

    #[test]
    fn flush_makes_the_store_current() {
        let mut p = pool(2, 2);
        p.book.begin_text(false);
        p.admit(&[2.0, 1.0], None).unwrap();
        p.w3.as_tensor().slice_set(&Tensor::full(3.5f32, (1, DF, DM), &Device::Cpu).unwrap(), 0, 1).unwrap();
        p.steps += 1;
        p.flush().unwrap();
        assert!(p.store.fetch(1, false).unwrap().w3.iter().all(|&x| x == 3.5));
        // and a second flush with nothing new does nothing harmful
        p.flush().unwrap();
    }
}
