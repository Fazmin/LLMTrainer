//! Helpers for the golden-parity tests: read the fixtures the Python reference wrote and compare numbers.
#![allow(dead_code, clippy::type_complexity)]

use std::collections::HashMap;
use std::path::PathBuf;

use minagi_core::candle_core::{Device, Tensor};
use minagi_core::store::NpzReader;
use minagi_core::store::npy::{NpyArray, NpyData};
use serde_json::Value;

pub fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden").join(name)
}

pub fn json(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path(name)).unwrap()).unwrap()
}

/// All arrays of one `.npz` fixture, by key.
pub struct Npz {
    map: HashMap<String, NpyArray>,
}

impl Npz {
    pub fn load(name: &str) -> Self {
        let mut r = NpzReader::open(path(name)).unwrap();
        let map = r.read_all().unwrap().into_iter().collect();
        Self { map }
    }

    pub fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.map.keys().cloned().collect();
        k.sort();
        k
    }

    fn arr(&self, key: &str) -> &NpyArray {
        self.map.get(key).unwrap_or_else(|| panic!("fixture has no key {key:?}"))
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.arr(key).shape().to_vec()
    }

    pub fn f32(&self, key: &str) -> Vec<f32> {
        match self.arr(key).data() {
            NpyData::F32(v) => v.clone(),
            other => panic!("{key}: expected f32, found {}", other.descr()),
        }
    }

    pub fn i64(&self, key: &str) -> Vec<i64> {
        match self.arr(key).data() {
            NpyData::I64(v) => v.clone(),
            other => panic!("{key}: expected i64, found {}", other.descr()),
        }
    }

    pub fn u8(&self, key: &str) -> Vec<u8> {
        match self.arr(key).data() {
            NpyData::U8(v) => v.clone(),
            other => panic!("{key}: expected u8, found {}", other.descr()),
        }
    }

    pub fn i16(&self, key: &str) -> Vec<i16> {
        self.arr(key).as_i16().unwrap_or_else(|| panic!("{key}: expected i16")).to_vec()
    }

    pub fn scalar(&self, key: &str) -> f32 {
        self.f32(key)[0]
    }

    pub fn int(&self, key: &str) -> i64 {
        self.i64(key)[0]
    }

    /// The array as a candle tensor on the CPU.
    pub fn tensor(&self, key: &str) -> Tensor {
        Tensor::from_vec(self.f32(key), self.shape(key), &Device::Cpu).unwrap()
    }

    /// Token ids (an `i64` array) as `u32`.
    pub fn ids(&self, key: &str) -> Vec<u32> {
        self.i64(key).into_iter().map(|v| v as u32).collect()
    }
}

pub fn host(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

/// `|got - want| <= atol + rtol * max|want|` element-wise, with a message naming the worst element.
pub fn close(what: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length {} vs {}", got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let bound = atol + rtol * scale;
    let (mut worst, mut at) = (0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst || !d.is_finite() {
            worst = if d.is_finite() { d } else { f32::INFINITY };
            at = i;
        }
    }
    assert!(
        worst <= bound,
        "{what}: worst difference {worst:e} at index {at} (got {}, want {}) exceeds {bound:e} (scale {scale:e})",
        got[at],
        want[at]
    );
}

/// Forward values: `atol = 1e-5 + 1e-4 * max`.
pub fn close_fwd(what: &str, got: &[f32], want: &[f32]) {
    close(what, got, want, 1e-5, 1e-4);
}

/// Gradients and updates: `atol = 1e-5 + 1e-3 * max`.
pub fn close_grad(what: &str, got: &[f32], want: &[f32]) {
    close(what, got, want, 1e-5, 1e-3);
}
