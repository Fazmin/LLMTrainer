"""Create numpy-written fixtures for the Rust npz/bf16 tests (stored + deflate + large bf16 corpus)."""
import sys, numpy as np, ml_dtypes

out = sys.argv[1]
fix = sys.argv[2]  # crate fixtures dir

def arrays():
    a = {}
    a["recur.0.mlp.router.weight"] = (np.arange(24, dtype=np.float32) * np.float32(0.25) - np.float32(3.0)).reshape(4, 6)
    src = ((np.arange(6, dtype=np.float32) - np.float32(2.5)) * np.float32(1.1)).astype(np.float32)
    a["bf16|w.0"] = src.astype(ml_dtypes.bfloat16).view(np.int16).reshape(2, 3)
    a["step"] = np.float64(0.1)  # 0-d
    a["special"] = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, 3.4e38, 1.17549435e-38], dtype=np.float32)
    a["ids"] = np.array([[-1, 0], [9007199254740993, -2**63]], dtype=np.int64)
    a["empty"] = np.zeros((0, 3), dtype=np.float32)
    a["bytes"] = np.array([0, 1, 127, 128, 255], dtype=np.uint8)
    a["i4"] = np.array([-2**31, 0, 2**31 - 1], dtype=np.int32)
    a["adam|v|recur.1.attn.wq.weight"] = (np.arange(12, dtype=np.float32) * np.float32(0.125) - np.float32(0.5)).reshape(3, 4)
    return a

a = arrays()
np.savez(f"{out}/np_stored.npz", **a)
np.savez_compressed(f"{out}/np_deflate.npz", **a)
np.savez(f"{fix}/numpy_stored.npz", **a)
np.savez_compressed(f"{fix}/numpy_deflate.npz", **a)

# bf16 corpus: random bit patterns (all exponents, incl. subnormals/inf/nan) + ties + normal-ish values
rng = np.random.default_rng(1234)
bits = rng.integers(0, 2**32, size=1 << 20, dtype=np.uint64).astype(np.uint32)
ties = np.array([0x3F808000, 0x3F818000, 0x3F808001, 0x3F807FFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000, 0x00008000, 0x80018000], dtype=np.uint32)
vals = np.concatenate([bits, ties]).view(np.float32)
vals = np.concatenate([vals, rng.standard_normal(1 << 18).astype(np.float32) * 3.0])
ref_bits = vals.astype(ml_dtypes.bfloat16).view(np.int16)
np.savez(f"{out}/bf16_corpus.npz", src=vals, bits=ref_bits)
print("numpy", np.__version__, "ml_dtypes", ml_dtypes.__version__, "corpus", vals.shape)
