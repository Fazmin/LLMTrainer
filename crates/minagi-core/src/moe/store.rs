//! Where experts live when they are not on the card.
//!
//! The pool sees an expert as one [`ExpertEntry`]: its three weight matrices and, once an optimiser step has touched it,
//! its Adam moments. Every expert is a file on disk with a RAM cache in front (`store::tiers`); the pool only needs
//! the small [`ExpertStore`] interface, which also lets tests run against an in-memory store with nothing on disk.

use std::collections::HashMap;

use crate::error::{EngineError, Result};

/// An expert's Adam moments, as f32 (they are narrowed to bf16 only when written to disk).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertMoments {
    pub w1_m: Vec<f32>,
    pub w1_v: Vec<f32>,
    pub w3_m: Vec<f32>,
    pub w3_v: Vec<f32>,
    pub w2_m: Vec<f32>,
    pub w2_v: Vec<f32>,
}

/// One expert: `w1`, `w3` are `[d_ff, d_model]` and `w2` is `[d_model, d_ff]`, row-major.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertEntry {
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
    pub w2: Vec<f32>,
    pub moments: Option<ExpertMoments>,
}

impl ExpertEntry {
    /// Elements per matrix.
    pub fn numel(&self) -> usize {
        self.w1.len()
    }

    /// Check the three matrices have `d_ff * d_model` elements each (and the moments too, when present).
    pub fn check(&self, d_model: usize, d_ff: usize) -> Result<()> {
        let n = d_model * d_ff;
        let ok = self.w1.len() == n && self.w3.len() == n && self.w2.len() == n;
        let ok_m = self
            .moments
            .as_ref()
            .is_none_or(|m| [&m.w1_m, &m.w1_v, &m.w3_m, &m.w3_v, &m.w2_m, &m.w2_v].iter().all(|v| v.len() == n));
        if ok && ok_m {
            Ok(())
        } else {
            Err(EngineError::Checkpoint(format!(
                "an expert has the wrong size (expected 3 matrices of {d_model} x {d_ff} values)"
            )))
        }
    }
}

/// Counters a store reports for the UI and the tests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoreReport {
    pub ram_held: usize,
    pub ram_capacity: usize,
    pub disk_reads: u64,
    pub ram_hits: u64,
    pub evictions: u64,
    pub writebacks: u64,
}

impl StoreReport {
    pub fn hit_rate(&self) -> f64 {
        let total = self.disk_reads + self.ram_hits;
        self.ram_hits as f64 / total.max(1) as f64
    }
}

/// The two storage tiers behind the card (RAM and disk), addressed by an expert's uid.
pub trait ExpertStore: Send {
    /// The expert, from RAM if it is there. `count = false` for a fetch that does not put the expert on the card (its
    /// moments, fetched for a step), so the hit rate stays a fact about loads.
    fn fetch(&mut self, uid: u64, count: bool) -> Result<ExpertEntry>;
    /// Hand an expert back after it has been resident (or create it). `dirty` entries are written to disk on eviction.
    fn put(&mut self, uid: u64, entry: ExpertEntry, dirty: bool) -> Result<()>;
    /// Remove an expert for good (pruning).
    fn delete(&mut self, uid: u64) -> Result<()>;
    /// Write everything dirty to disk.
    fn flush(&mut self) -> Result<()>;
    fn report(&self) -> StoreReport;
    /// The real tiers behind this store, when it has them (for writing checkpoints by hard link).
    fn tiers_mut(&mut self) -> Option<&mut crate::store::tiers::Tiers> {
        None
    }
    /// Bytes the experts occupy on disk, when known.
    fn disk_bytes(&self) -> Option<u64> {
        None
    }
}

/// An in-memory store for tests and for models that never leave RAM.
#[derive(Debug, Default)]
pub struct MemStore {
    map: HashMap<u64, ExpertEntry>,
    reads: u64,
}

impl MemStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn contains(&self, uid: u64) -> bool {
        self.map.contains_key(&uid)
    }

    pub fn get(&self, uid: u64) -> Option<&ExpertEntry> {
        self.map.get(&uid)
    }
}

impl ExpertStore for MemStore {
    fn fetch(&mut self, uid: u64, count: bool) -> Result<ExpertEntry> {
        if count {
            self.reads += 1;
        }
        self.map.get(&uid).cloned().ok_or_else(|| EngineError::Checkpoint(format!("expert {uid} is missing")))
    }

    fn put(&mut self, uid: u64, entry: ExpertEntry, _dirty: bool) -> Result<()> {
        self.map.insert(uid, entry);
        Ok(())
    }

    fn delete(&mut self, uid: u64) -> Result<()> {
        self.map.remove(&uid);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn report(&self) -> StoreReport {
        StoreReport { ram_held: self.map.len(), ram_capacity: usize::MAX, disk_reads: self.reads, ..Default::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_store_round_trips_and_deletes() {
        let mut s = MemStore::new();
        let e = ExpertEntry { w1: vec![1.0; 6], w3: vec![2.0; 6], w2: vec![3.0; 6], moments: None };
        e.check(3, 2).unwrap();
        assert!(e.check(3, 3).is_err());
        s.put(7, e.clone(), true).unwrap();
        assert_eq!(s.fetch(7, true).unwrap(), e);
        assert_eq!(s.report().disk_reads, 1);
        s.delete(7).unwrap();
        assert!(s.fetch(7, true).unwrap_err().to_string().contains("expert 7"));
    }
}
