//! bfloat16 packing.
//!
//! bf16 is the upper 16 bits of an IEEE-754 binary32. Weights are stored in `.npz` files as
//! `<i2` arrays holding those 16 bits (numpy has no bf16 dtype), so [`pack`] returns `i16` and
//! [`unpack`] takes `i16`.
//!
//! This is bit-for-bit what the Python reference does with `t.to(torch.bfloat16).view(torch.int16)`
//! (`minagi/precision.py::pack_bf16`) and `a.view(torch.bfloat16).to(torch.float32)`
//! (`unpack_bf16`): round to nearest, ties to even; overflow becomes infinity; every NaN becomes the
//! positive quiet NaN `0x7FC0` (torch canonicalises NaN payloads and signs). The test-suite checks
//! this against torch on a fixture of 40,000 values, ties included
//! (`tests/store_golden_bf16.rs`).

/// Whether a saved array's name marks it as an Adam moment rather than a weight.
///
/// Mirrors `precision.is_moment`: expert files name their moments `w1_m` / `w1_v`, the trunk's
/// optimiser file uses `<param>|m` and `<param>|v`. `|t` is a step count and is *not* a moment.
pub fn is_moment(name: &str) -> bool {
    name.ends_with("_m") || name.ends_with("_v") || name.ends_with("|m") || name.ends_with("|v")
}

/// The bits every NaN is stored as (torch writes this for any NaN, whatever its payload or sign).
pub const BF16_QUIET_NAN: u16 = 0x7FC0;

/// Convert an `f32` to bf16 bits using round-to-nearest, ties-to-even.
///
/// Finite values that round beyond the bf16 range become infinity, infinities are preserved and
/// every NaN becomes [`BF16_QUIET_NAN`], like torch.
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return BF16_QUIET_NAN;
    }
    // Round to nearest even: add 0x7FFF plus the least significant kept bit, then truncate.
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

/// Convert bf16 bits to `f32` (exact).
#[inline]
pub fn bf16_bits_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Pack `f32` values into bf16 bit patterns stored as `i16` (the `<i2` npz representation).
pub fn pack(src: &[f32]) -> Vec<i16> {
    src.iter().map(|&x| f32_to_bf16_bits(x) as i16).collect()
}

/// Unpack `<i2` bf16 bit patterns into `f32`.
pub fn unpack(src: &[i16]) -> Vec<f32> {
    src.iter().map(|&x| bf16_bits_to_f32(x as u16)).collect()
}

/// Pack `src` into the existing buffer `dst` (same length), without allocating.
///
/// # Panics
/// If the slices differ in length.
pub fn pack_into(src: &[f32], dst: &mut [i16]) {
    assert_eq!(src.len(), dst.len(), "pack_into: length mismatch");
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = f32_to_bf16_bits(x) as i16;
    }
}

/// Unpack `src` into the existing buffer `dst` (same length), without allocating.
///
/// # Panics
/// If the slices differ in length.
pub fn unpack_into(src: &[i16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "unpack_into: length mismatch");
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = bf16_bits_to_f32(x as u16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn exhaustive_bf16_roundtrip() {
        for bits in 0..=u16::MAX {
            let f = bf16_bits_to_f32(bits);
            if f.is_nan() {
                assert_eq!(f32_to_bf16_bits(f), BF16_QUIET_NAN);
            } else {
                assert_eq!(f32_to_bf16_bits(f), bits, "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn matches_half_crate_on_random_and_special_values() {
        // xorshift for deterministic pseudo-random bit patterns covering every exponent
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let check = |x: f32| {
            let ours = f32_to_bf16_bits(x);
            let theirs = bf16::from_f32(x).to_bits();
            if x.is_nan() {
                assert_eq!(ours, BF16_QUIET_NAN);
            } else {
                assert_eq!(ours, theirs, "x = {x:e} ({:#010x})", x.to_bits());
            }
        };
        for _ in 0..2_000_000 {
            check(f32::from_bits(next() as u32));
        }
        for x in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-45,
            3.389_531_4e38, // just below the rounding boundary to inf
            3.4e38,
        ] {
            check(x);
        }
    }

    #[test]
    fn ties_round_to_even() {
        // 1.0 = 0x3F80_0000. Exactly halfway between 0x3F80 and 0x3F81 -> even (0x3F80).
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
        // Halfway between 0x3F81 and 0x3F82 -> even (0x3F82).
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
        // Just above / below halfway.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_7FFF)), 0x3F80);
    }

    #[test]
    fn overflow_goes_to_infinity_and_nan_stays_nan() {
        assert_eq!(f32_to_bf16_bits(f32::MAX), 0x7F80); // rounds up to +inf
        assert_eq!(f32_to_bf16_bits(f32::MIN), 0xFF80);
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7F80);
        // A NaN whose payload lives only in the low 16 bits must not truncate to infinity, and like
        // torch every NaN (either sign, any payload) comes out as the one canonical bit pattern.
        for bits in [0x7F80_0001u32, 0xFF80_0001, 0x7FFF_FFFF, 0xFFC0_0001, 0x7FC0_0000] {
            let nan = f32::from_bits(bits);
            assert!(nan.is_nan());
            assert_eq!(f32_to_bf16_bits(nan), BF16_QUIET_NAN, "{bits:#010x}");
        }
    }

    #[test]
    fn moment_names_follow_the_python_rule() {
        for yes in ["w1_m", "w2_v", "tok_emb.weight|m", "pool.gate|v"] {
            assert!(is_moment(yes), "{yes}");
        }
        for no in ["w1", "w3", "tok_emb.weight|t", "b0_w1", "recur.0.mlp.depth_emb", "step"] {
            assert!(!is_moment(no), "{no}");
        }
    }

    #[test]
    fn pack_into_and_unpack_into_match_the_allocating_versions() {
        let v = [1.0f32, -2.5, 0.0, 65504.0, 1e-3, f32::INFINITY];
        let mut p = vec![0i16; v.len()];
        pack_into(&v, &mut p);
        assert_eq!(p, pack(&v));
        let mut u = vec![0f32; v.len()];
        unpack_into(&p, &mut u);
        assert_eq!(u, unpack(&p));
    }

    #[test]
    fn pack_unpack_i16_representation() {
        let v = [1.0f32, -2.5, 0.0, 65504.0, 1e-3];
        let p = pack(&v);
        assert_eq!(p[0], 0x3F80u16 as i16);
        assert_eq!(p[1], 0xC020u16 as i16); // negative values map to negative i16
        assert!(p[1] < 0);
        let u = unpack(&p);
        for (a, b) in v.iter().zip(&u) {
            assert!((a - b).abs() <= a.abs() * 2f32.powi(-8), "{a} vs {b}");
        }
        // unpack(pack(unpack(x))) is bit exact
        assert_eq!(pack(&u), p);
    }
}
