//! The real expert store: RAM tier plus one file per expert, from `store::tiers`.
//!
//! The pool works with f32 moments; the tiers keep them as bf16 bit patterns (half the RAM, and an untouched expert
//! round-trips bit for bit). The conversion happens here, at the boundary.

use super::store::{ExpertEntry, ExpertMoments, ExpertStore, StoreReport};
use crate::error::Result;
use crate::store::tiers::{self, Tiers};

pub struct TiersStore {
    pub tiers: Tiers,
}

impl TiersStore {
    pub fn new(tiers: Tiers) -> Self {
        Self { tiers }
    }
}

pub(crate) fn to_tiers(e: ExpertEntry) -> tiers::ExpertEntry {
    let moments = e.moments.map(|m| tiers::ExpertMoments::pack([&m.w1_m, &m.w1_v, &m.w3_m, &m.w3_v, &m.w2_m, &m.w2_v]));
    tiers::ExpertEntry { w1: e.w1, w3: e.w3, w2: e.w2, moments }
}

pub(crate) fn from_tiers(e: &tiers::ExpertEntry) -> ExpertEntry {
    let moments = e.moments.as_ref().map(|m| {
        let [w1_m, w1_v, w3_m, w3_v, w2_m, w2_v] = m.unpack();
        ExpertMoments { w1_m, w1_v, w3_m, w3_v, w2_m, w2_v }
    });
    ExpertEntry { w1: e.w1.clone(), w3: e.w3.clone(), w2: e.w2.clone(), moments }
}

impl ExpertStore for TiersStore {
    fn fetch(&mut self, uid: u64, count: bool) -> Result<ExpertEntry> {
        let e = if count { self.tiers.fetch(uid)? } else { self.tiers.fetch_uncounted(uid)? };
        Ok(from_tiers(&e))
    }

    fn put(&mut self, uid: u64, entry: ExpertEntry, dirty: bool) -> Result<()> {
        Ok(self.tiers.put(uid, to_tiers(entry), dirty)?)
    }

    fn delete(&mut self, uid: u64) -> Result<()> {
        self.tiers.delete(uid)?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.tiers.flush()?;
        Ok(())
    }

    fn report(&self) -> StoreReport {
        let r = self.tiers.report();
        StoreReport {
            ram_held: r.ram_held,
            ram_capacity: r.ram_capacity,
            disk_reads: r.misses,
            ram_hits: r.hits,
            evictions: r.evictions,
            writebacks: r.writebacks,
        }
    }

    fn tiers_mut(&mut self) -> Option<&mut Tiers> {
        Some(&mut self.tiers)
    }

    fn disk_bytes(&self) -> Option<u64> {
        Some(self.tiers.report().disk_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tiers::TierConfig;

    #[test]
    fn entries_survive_ram_eviction_and_disk_with_bf16_moments() {
        let dir = tempfile_dir("tiers_store");
        let mut s = TiersStore::new(Tiers::open(TierConfig::new(&dir, 4, 3).ram_capacity(1)).unwrap());
        let n = 12;
        let e = |k: f32| ExpertEntry {
            w1: vec![k; n],
            w3: vec![k + 1.0; n],
            w2: vec![k + 2.0; n],
            moments: Some(ExpertMoments {
                w1_m: vec![0.1 * k; n],
                w1_v: vec![0.01 * k; n],
                w3_m: vec![0.2 * k; n],
                w3_v: vec![0.02 * k; n],
                w2_m: vec![0.3 * k; n],
                w2_v: vec![0.03 * k; n],
            }),
        };
        s.put(0, e(1.0), true).unwrap();
        s.put(1, e(2.0), true).unwrap(); // capacity 1: expert 0 is evicted and written to disk
        let back = s.fetch(0, true).unwrap();
        assert_eq!(back.w1, vec![1.0; n]);
        let m = back.moments.unwrap();
        assert!((m.w1_m[0] - 0.1).abs() < 1e-3, "moments are rounded to bf16 on the way to disk: {}", m.w1_m[0]);
        assert!(s.report().evictions >= 1);
        assert!(s.disk_bytes().unwrap() > 0);
        s.delete(1).unwrap();
        assert!(s.fetch(1, true).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn tempfile_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("minagi-{name}-{}-{}", std::process::id(), rand_suffix()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn rand_suffix() -> u64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
    }
}
