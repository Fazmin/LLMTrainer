"""Shared plumbing for the golden-fixture scripts (dump_model.py, verify_golden.py).

Nothing in here touches tools/golden/ref/: the reference is imported read-only
(bytecode writing is disabled so no __pycache__ appears under it).
"""

import json
import os
import re
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REF = HERE / "ref"
REPO = HERE.parent.parent
OUT_DIR = REPO / "crates" / "minagi-core" / "tests" / "fixtures" / "golden"

if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

# ----------------------------------------------------------------------------
# the tiny model every model-level fixture uses
# ----------------------------------------------------------------------------

TINY = dict(
    vocab_size=265, d_model=32, n_head=2, d_ff=64,
    n_prelude=1, n_recur=1, n_coda=0,
    max_steps=4, min_steps=1,
    halt_prior=0.4, halt_thresh=0.9, halt_freeze=True,
    ponder_beta=0.01, bptt_window=4, train_steps_mean=0.0,
    block=64, rope_theta=10000.0, tie_embeddings=True,
)
TINY_POOL = dict(
    use_pool=True, pool_experts=8, pool_d_ff=16, pool_depth=1, pool_top_k=2,
    pool_capacity_factor=1.5, pool_max=8, pool_aux=0.01,
)
# run-time settings that live outside RecurConfig (config.yaml / PagedPool)
TINY_RUNTIME = dict(pool_resident=4, explore_bias=0.65, explore_steps=1000.0)


def tiny_cfg(**over):
    """RecurConfig for the tiny model. `over` overrides any field."""
    from minagi.recur import RecurConfig
    d = dict(TINY)
    d.update(over)
    return RecurConfig(**d)


def tiny_pool_cfg(**over):
    d = dict(TINY)
    d.update(TINY_POOL)
    d.update(over)
    from minagi.recur import RecurConfig
    return RecurConfig(**d)


# ----------------------------------------------------------------------------
# deterministic randomness
# ----------------------------------------------------------------------------

class Rng:
    """A seeded torch.Generator with convenience draws (CPU, float32)."""

    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(int(seed))

    def randn(self, *shape, std=1.0):
        return torch.randn(*shape, generator=self.g) * std

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g)

    def randint(self, lo, hi, *shape):
        return torch.randint(lo, hi, tuple(shape), generator=self.g)


# ----------------------------------------------------------------------------
# tensor -> numpy
# ----------------------------------------------------------------------------

def to_np(x):
    """float -> f32, int -> i64, bool -> u8. The Rust npy reader has no bool/f16."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype == np.bool_:
        return x.astype(np.uint8)
    if x.dtype.kind == "f":
        return np.ascontiguousarray(x.astype(np.float32))
    if x.dtype.kind in "iu":
        if x.dtype in (np.uint8, np.int16, np.int32, np.int64):
            return np.ascontiguousarray(x)
        return np.ascontiguousarray(x.astype(np.int64))
    raise TypeError(f"unsupported dtype {x.dtype}")


WRITTEN = []        # every fixture file written by this process (for SHA256SUMS)


class Npz:
    """An ordered dict of arrays written like numpy.savez (stored zip of .npy).

    Differences from numpy.savez, both for reproducibility only: entry
    timestamps are pinned to 1980-01-01 so two runs are byte-identical, and the
    file is written atomically. The on-disk format is otherwise exactly numpy's
    (ZIP_STORED, zip64 local headers, NPY 1.0/2.0 headers).
    """

    def __init__(self):
        self.arrays = {}

    def add(self, key, value):
        if key in self.arrays:
            raise KeyError(f"duplicate fixture key {key}")
        a = to_np(value)
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"non-finite values in {key}")
        self.arrays[key] = a
        return a

    def scalar(self, key, value):
        """Scalars are stored with shape [1] (f32 for floats, i64 for ints)."""
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        if isinstance(value, (int, np.integer)):
            return self.add(key, np.asarray([value], dtype=np.int64))
        return self.add(key, np.asarray([float(value)], dtype=np.float32))

    def add_dict(self, prefix, d):
        for k, v in d.items():
            self.add(f"{prefix}{k}", v)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for k, a in self.arrays.items():
                zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o600 << 16
                with zf.open(zi, "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, a, allow_pickle=False)
        os.replace(tmp, path)
        WRITTEN.append(path.name)
        return path.stat().st_size


# ----------------------------------------------------------------------------
# JSON (numeric lists collapsed onto one line so diffs stay readable)
# ----------------------------------------------------------------------------

_NUMLIST = re.compile(r"\[\s*([-0-9.eE+,\s]+?)\s*\]")


def dump_json(path, obj, indent=1):
    text = json.dumps(obj, indent=indent, ensure_ascii=True, allow_nan=False)
    text = _NUMLIST.sub(lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", text)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text + "\n")
    os.replace(tmp, path)
    WRITTEN.append(path.name)
    return path.stat().st_size


def fl(x):
    """Tensor/ndarray/scalar -> JSON-able python floats.

    float32 values are emitted as the shortest decimal that round-trips to the same float32
    (e.g. 0.3 for the f32 nearest 0.3), so a reader may parse as f32 or as f64-then-cast.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    conv = (lambda v: float(str(v))) if x.dtype == np.float32 else float
    if x.ndim == 0:
        return conv(x[()])
    if x.ndim == 1:
        return [conv(v) for v in x]
    return [fl(r) for r in x]


def il(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return [int(v) for v in np.asarray(x).reshape(-1)]


# ----------------------------------------------------------------------------
# line-level capture of locals inside a reference function (no re-implementation)
# ----------------------------------------------------------------------------

class LineProbe:
    """Record locals of a reference function at one source line, every time it runs.

    Used to read per-row quantities (lam, p_n, cum, halted, ...) out of
    RecurCoder.forward without re-implementing it. `marker` is the stripped text
    of the first source line equal to it; the probe fires just BEFORE that line
    executes.
    """

    def __init__(self, func, marker, names):
        self.code = func.__code__
        import inspect
        lines, first = inspect.getsourcelines(func)
        self.line = None
        for i, l in enumerate(lines):
            if l.strip() == marker:
                self.line = first + i
                break
        if self.line is None:
            raise RuntimeError(f"marker {marker!r} not found in {func}")
        self.names = names
        self.records = []

    def _local(self, frame, event, arg):
        if event == "line" and frame.f_lineno == self.line:
            loc = frame.f_locals
            rec = {}
            for n in self.names:
                v = loc.get(n, None)
                rec[n] = v.detach().clone() if torch.is_tensor(v) else v
            self.records.append(rec)
        return self._local

    def _global(self, frame, event, arg):
        return self._local if frame.f_code is self.code else None

    def __enter__(self):
        self._old = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._old)
        return False


class Probes:
    """Several LineProbes at once (sys.settrace admits only one tracer)."""

    def __init__(self, *probes):
        self.probes = probes

    def _global(self, frame, event, arg):
        for p in self.probes:
            if frame.f_code is p.code:
                return p._local
        return None

    def __enter__(self):
        self._old = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._old)
        return False
