//! Seeded host-side randomness.
//!
//! Everything random in the engine (initial weights, how many rows to run, which parents a new expert recombines,
//! which window of text to read) is drawn on the host from one of these, never on the accelerator. That makes a run
//! reproducible from its seed and gives identical starting weights on every backend.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A small deterministic generator with the distributions the engine needs.
#[derive(Debug, Clone)]
pub struct HostRng {
    inner: ChaCha8Rng,
    /// Second value of the last Box-Muller pair.
    spare: Option<f64>,
}

impl HostRng {
    pub fn new(seed: u64) -> Self {
        Self { inner: ChaCha8Rng::seed_from_u64(seed), spare: None }
    }

    /// An independent stream derived from this seed and a label, so unrelated consumers do not shift each other.
    pub fn fork(seed: u64, stream: u64) -> Self {
        let mut r = ChaCha8Rng::seed_from_u64(seed);
        r.set_stream(stream);
        Self { inner: r, spare: None }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.inner.random::<f64>()
    }

    /// Uniform integer in `[0, n)`; `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        self.inner.random_range(0..n)
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }

    /// `n` samples of `N(0, std^2)` as f32.
    pub fn normal_vec(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() as f32 * std).collect()
    }

    /// A Poisson draw (Knuth's product method for small means, a normal approximation above 60).
    pub fn poisson(&mut self, mean: f64) -> u32 {
        if mean <= 0.0 {
            return 0;
        }
        if mean > 60.0 {
            return (mean + mean.sqrt() * self.normal()).round().max(0.0) as u32;
        }
        let limit = (-mean).exp();
        let (mut k, mut p) = (0u32, 1.0f64);
        loop {
            p *= self.uniform();
            if p <= limit {
                return k;
            }
            k += 1;
        }
    }

    /// The first `m` entries of a random permutation of `0..n` (partial Fisher-Yates).
    pub fn sample_indices(&mut self, n: usize, m: usize) -> Vec<usize> {
        let m = m.min(n);
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..m {
            let j = i + self.below(n - i);
            pool.swap(i, j);
        }
        pool.truncate(m);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let (mut a, mut b) = (HostRng::new(7), HostRng::new(7));
        for _ in 0..50 {
            assert_eq!(a.normal().to_bits(), b.normal().to_bits());
        }
        assert_ne!(HostRng::new(7).next_u64(), HostRng::new(8).next_u64());
    }

    #[test]
    fn forked_streams_are_independent_and_reproducible() {
        let a = HostRng::fork(1, 1).next_u64();
        assert_eq!(a, HostRng::fork(1, 1).next_u64());
        assert_ne!(a, HostRng::fork(1, 2).next_u64());
    }

    #[test]
    fn normal_has_unit_moments() {
        let mut r = HostRng::new(3);
        let v: Vec<f64> = (0..40_000).map(|_| r.normal()).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn poisson_mean_matches() {
        let mut r = HostRng::new(4);
        for mean in [0.5, 3.2, 12.8, 80.0] {
            let n = 20_000;
            let m = (0..n).map(|_| r.poisson(mean) as f64).sum::<f64>() / n as f64;
            assert!((m - mean).abs() < 0.1 * mean.max(1.0), "mean {mean} got {m}");
        }
        assert_eq!(r.poisson(0.0), 0);
    }

    #[test]
    fn sample_indices_are_distinct_and_in_range() {
        let mut r = HostRng::new(5);
        let s = r.sample_indices(20, 16);
        assert_eq!(s.len(), 16);
        let mut sorted = s.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 16);
        assert!(s.iter().all(|&i| i < 20));
        assert_eq!(r.sample_indices(3, 10).len(), 3);
    }
}
