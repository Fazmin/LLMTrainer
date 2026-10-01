"""Verify Rust-written npz files with numpy (run after `spike_s7 rewrite`).

usage: spike_s7_verify.py <numpy_dir> <rust_dir>
"""
import sys, zipfile, io, numpy as np, ml_dtypes

npd, rsd = sys.argv[1], sys.argv[2]
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# 1. numpy-written -> Rust-read -> Rust-written -> numpy-read must be bit identical
for src, dst in [("np_stored.npz", "rust_from_stored.npz"), ("np_deflate.npz", "rust_from_deflate.npz")]:
    a = np.load(f"{npd}/{src}", allow_pickle=False)
    b = np.load(f"{rsd}/{dst}", allow_pickle=False)
    check(f"{src}->rust->{dst}: same key set/order", list(a.files) == list(b.files), str(list(b.files)))
    all_equal = True
    for k in a.files:
        x, y = a[k], b[k]
        same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        all_equal &= same
        if not same:
            print("   mismatch", k, x.dtype, y.dtype, x.shape, y.shape)
    check(f"{src}->rust->{dst}: all arrays bit exact (dtype, shape, bytes incl. NaN payloads)", all_equal)
    # raw entry bytes: header + data must be byte-identical to numpy's own np.save output
    za = zipfile.ZipFile(f"{npd}/{src}")
    zb = zipfile.ZipFile(f"{rsd}/{dst}")
    same_bytes = all(za.read(n) == zb.read(n) for n in za.namelist())
    check(f"{dst}: every .npy entry byte-identical to numpy's (header padding, dict text)", same_bytes)
    check(f"{dst}: entries stored (ZIP_STORED) like np.savez", all(i.compress_type == zipfile.ZIP_STORED for i in zb.infolist()))
    check(f"{dst}: zip CRCs valid (testzip)", zb.testzip() is None)

# 2. bf16: Rust pack vs ml_dtypes (RNE), Rust unpack vs ml_dtypes widening, on 1.3M values
c = np.load(f"{npd}/bf16_corpus.npz")
src, ref_bits = c["src"], c["bits"]
r = np.load(f"{rsd}/rust_bf16.npz")
packed, unpacked = r["packed"], r["unpacked"]
check("bf16 packed dtype is <i2", packed.dtype == np.dtype("<i2"), str(packed.dtype))
nan_mask = np.isnan(src)
ref_f = ref_bits.view(ml_dtypes.bfloat16).astype(np.float32)
eq = packed == ref_bits
check(f"bf16 pack == ml_dtypes RNE for all {src.size} values (non-NaN bit exact)", bool(np.all(eq[~nan_mask])), f"{int((~eq[~nan_mask]).sum())} mismatches")
check("bf16 pack: NaN inputs stay NaN", bool(np.all(np.isnan(packed[nan_mask].view(ml_dtypes.bfloat16).astype(np.float32)))), f"{int(nan_mask.sum())} NaNs")
check("bf16 unpack == ml_dtypes widening, bit exact", bool(np.all(unpacked.view(np.uint32)[~np.isnan(ref_f)] == ref_f.view(np.uint32)[~np.isnan(ref_f)])))
check("bf16 unpack NaN stays NaN", bool(np.all(np.isnan(unpacked[np.isnan(ref_f)]))))

# 3. arrays constructed in Rust: values, dtypes, 0-d scalars, odd keys
m = np.load(f"{rsd}/rust_made.npz", allow_pickle=False)
router = ((np.arange(2048, dtype=np.uint32) * 37 % 101).astype(np.float32) - np.float32(50.0)) / np.float32(7.0)
exp = {
    "recur.0.mlp.router.weight": router.reshape(64, 32),
    "bf16|weights.0": router.astype(ml_dtypes.bfloat16).view(np.int16).reshape(64, 32),
    "train|step": np.float64(12345.0),
    "lr": np.float64(3e-4),
    "tokens|i2": ((np.arange(1000, dtype=np.int64) * 31) % 65536 - 32768).astype(np.int16),
    "big": ((np.arange(1 << 22, dtype=np.uint32) % 1000).astype(np.float32) * np.float32(1e-3)),
    "empty.rank2": np.zeros((0, 7), dtype=np.float32),
    "ünï key with spaces|and.dots": np.array([1, 2, 3], dtype=np.int32),
}
check("rust_made: key set", sorted(m.files) == sorted(exp.keys()), str(m.files))
for k, v in exp.items():
    got = m[k]
    good = got.dtype == np.asarray(v).dtype and got.shape == np.asarray(v).shape and got.tobytes() == np.asarray(v).tobytes()
    check(f"rust_made[{k!r}] dtype={got.dtype} shape={got.shape} bit exact", good)
check("rust_made: 0-d scalar has ndim 0", m["train|step"].ndim == 0 and float(m["train|step"]) == 12345.0)
zb = zipfile.ZipFile(f"{rsd}/rust_made.npz")
check("rust_made: stored entries, CRC ok", all(i.compress_type == zipfile.ZIP_STORED for i in zb.infolist()) and zb.testzip() is None)
# np.save of the same arrays must give byte-identical entries
same = True
for k, v in exp.items():
    buf = io.BytesIO(); np.save(buf, np.asarray(v), allow_pickle=False)
    same &= buf.getvalue() == zb.read(k + ".npy")
check("rust_made: each entry byte-identical to np.save()", same)
print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
