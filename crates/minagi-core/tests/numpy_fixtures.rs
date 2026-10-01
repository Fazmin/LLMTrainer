//! Interop regression tests against files written by real numpy (2.5, `np.savez` and
//! `np.savez_compressed`), see `tests/fixtures/`. The same arrays are rebuilt here from simple
//! formulas, so a failure means our reader/bf16 packing disagrees with numpy/ml_dtypes.

use minagi_core::store::bf16;
use minagi_core::store::npy::{NpyArray, NpyData};
use minagi_core::store::{NpzReader, NpzWriter};
use std::io::Cursor;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn expected() -> Vec<(&'static str, NpyArray)> {
    let router: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 3.0).collect();
    let bf_src: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 1.1).collect();
    vec![
        ("recur.0.mlp.router.weight", NpyArray::f32(vec![4, 6], router).unwrap()),
        ("bf16|w.0", NpyArray::from_bf16(vec![2, 3], &bf_src).unwrap()),
        ("step", NpyArray::scalar_f64(0.1)),
        (
            "special",
            NpyArray::f32(
                vec![8],
                vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-45, 3.4e38, f32::MIN_POSITIVE],
            )
            .unwrap(),
        ),
        ("ids", NpyArray::i64(vec![2, 2], vec![-1, 0, 9007199254740993, i64::MIN]).unwrap()),
        ("empty", NpyArray::f32(vec![0, 3], vec![]).unwrap()),
        ("bytes", NpyArray::new(vec![5], NpyData::U8(vec![0, 1, 127, 128, 255])).unwrap()),
        ("i4", NpyArray::new(vec![3], NpyData::I32(vec![i32::MIN, 0, i32::MAX])).unwrap()),
        (
            "adam|v|recur.1.attn.wq.weight",
            NpyArray::f32(vec![3, 4], (0..12).map(|i| i as f32 * 0.125 - 0.5).collect()).unwrap(),
        ),
    ]
}

fn check_file(name: &str) {
    let mut r = NpzReader::open(fixture(name)).unwrap();
    let want = expected();
    assert_eq!(r.keys().len(), want.len(), "{name}: key count");
    for (k, a) in &want {
        assert!(r.contains(k), "{name}: missing key {k}");
        let got = r.get(k).unwrap();
        assert!(got.bit_eq(a), "{name}: {k}: got {got:?}, want {a:?}");
    }
}

#[test]
fn reads_numpy_savez_stored() {
    check_file("numpy_stored.npz");
}

#[test]
fn reads_numpy_savez_compressed() {
    check_file("numpy_deflate.npz");
}

#[test]
fn rust_written_archive_has_the_same_arrays_as_numpy_fixture() {
    let mut w = NpzWriter::new(Cursor::new(Vec::new()));
    for (k, a) in expected() {
        w.add(k, &a).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    let mut ours = NpzReader::new(Cursor::new(bytes)).unwrap();
    let mut theirs = NpzReader::open(fixture("numpy_stored.npz")).unwrap();
    let mut a: Vec<String> = ours.keys().to_vec();
    let mut b: Vec<String> = theirs.keys().to_vec();
    a.sort();
    b.sort();
    assert_eq!(a, b);
    for k in &a {
        assert!(ours.get(k).unwrap().bit_eq(&theirs.get(k).unwrap()), "{k}");
    }
}

#[test]
fn bf16_pack_matches_ml_dtypes_for_fixture_values() {
    let mut r = NpzReader::open(fixture("numpy_stored.npz")).unwrap();
    let packed = r.get("bf16|w.0").unwrap();
    let bf_src: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 1.1).collect();
    assert_eq!(packed.as_i16().unwrap(), bf16::pack(&bf_src).as_slice());
}
