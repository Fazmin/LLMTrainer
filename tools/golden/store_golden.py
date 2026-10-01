#!/usr/bin/env python3
"""
Golden data for the Rust store (crates/minagi-core/src/store) made with the REAL Python reference.

    python tools/golden/store_golden.py generate [--out DIR]
        Writes the fixtures under crates/minagi-core/tests/fixtures/store/ :

          py_weights/          a paged weights directory exactly as train.py saves it (d_model 32,
                               n_head 2, d_ff 64, 1 prelude, 1 recurrent block, no coda, max_steps 4,
                               8 experts of 16 units, top_k 2, resident 4, block 64) after a few real
                               AdamW steps, then two grown experts and a prune, so that expert uids
                               are NOT 0..n (files are named by uid), with Adam moments in optim.npz
                               and in the expert files (bf16 bit patterns in <i2).
          non_paged_weights/   the older, non-paged save: pool.gate in core.npz, no `paged` /
                               `telemetry` in the manifest, expert files with fp32 moments.
          expected.json        per file, per array: dtype, shape and crc32 of the raw bytes and of the
                               values widened to float32, as the Python loaders see them.
          bf16_golden.npz      36,020 float32 bit patterns and the bf16 bits torch makes of them.

    python tools/golden/store_golden.py verify DIR REFERENCE.npz
        Loads DIR (written by the Rust store) with the real Python loaders -- store.py's
        `_load_dir` / `train.build_paged` / `_load_optim` and the paged pool's own Tiers -- and compares
        everything with REFERENCE.npz, which the Rust test wrote from the values it started with.
        Exit status 0 means Python loads what Rust wrote and sees exactly those values.

    python tools/golden/store_golden.py resave SRC DST
        Copies SRC (written by Rust) to DST, loads DST with the real Python code (build_paged +
        _load_optim) and SAVES it again with store.save, in place. The Rust test then reads DST back:
        Rust -> Python load -> Python save -> Rust.

Run it with the repo's venv:  .venv/bin/python tools/golden/store_golden.py generate
"""

import argparse
import dataclasses
import json
import os
import shutil
import sys
import zlib

sys.dont_write_bytecode = True  # importing the reference must not leave .pyc files inside tools/golden/ref

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(REPO, "crates", "minagi-core", "tests", "fixtures", "store")

# the reference is imported as a package from its own directory
CALLER_CWD = os.getcwd()
sys.path.insert(0, REF)
os.chdir(REF)


def absp(p):
    """A command-line path, relative to where the user ran the script (we chdir into the reference)."""
    return os.path.abspath(os.path.join(CALLER_CWD, p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from minagi.precision import pack_bf16, unpack_bf16  # noqa: E402

# The tiny model from the task: everything create() would read from config.yaml, overridden.
TINY = {
    "model": {"d_model": 32, "n_head": 2, "d_ff": 64, "n_prelude": 1, "n_recur": 1, "n_coda": 0,
              "max_steps": 4, "context_end": 64},
    "pool": {"experts": 8, "resident": 4, "width": 16, "depth": 1, "top_k": 2},
}


def crc(b):
    return zlib.crc32(b) & 0xFFFFFFFF


def describe_npz(path):
    """dtype / shape / crc32 of every array in an npz, plus the float32 view the Python loaders use."""
    z = np.load(path)
    out = {}
    for k in z.files:
        a = z[k]
        e = {"dtype": a.dtype.str, "shape": list(a.shape), "crc_raw": crc(np.ascontiguousarray(a).tobytes())}
        if a.dtype == np.int16:
            f = unpack_bf16(a).numpy()
        else:
            f = a.astype(np.float32)
        e["crc_f32"] = crc(np.ascontiguousarray(f).tobytes())
        if a.ndim == 0:
            e["value"] = float(a)
        out[k] = e
    return out


def make_model(out, seed=3):
    """create() reads config.yaml; point it at TINY instead (create() looks load() up at call time)."""
    import minagi.config as mc
    from minagi.create import create
    mc.load = lambda path=None: TINY
    return create(out, seed=seed, verbose=False, force=True)


def adamw(model, lr=1e-2):
    import train as T
    trunk, pool_ps = T._split_trunk_pool(model)
    return torch.optim.AdamW([{"params": trunk, "name": "trunk", "weight_decay": 0.1, "lr": lr},
                              {"params": pool_ps, "name": "pool", "weight_decay": 0.1, "lr": lr}],
                             lr=lr, betas=(0.9, 0.95))


def resync(opt, model):
    """train.py::_resync_opt - growth and pruning replace parameters; the optimiser must follow."""
    live = {id(p) for p in model.parameters()}
    for g in opt.param_groups:
        for p in g["params"]:
            if id(p) not in live:
                opt.state.pop(p, None)
        g["params"] = [p for p in g["params"] if id(p) in live]
    known = {id(p) for g in opt.param_groups for p in g["params"]}
    fresh = [p for p in model.parameters() if id(p) not in known]
    if fresh:
        opt.add_param_group({"params": fresh, "name": "pool", "weight_decay": 0.1, "lr": 1e-2})


def train_steps(model, opt, n, gen):
    model.train()
    for _ in range(n):
        ids = torch.randint(0, 256, (1, 24), generator=gen)
        opt.zero_grad()
        _, loss = model(ids[:, :-1], ids[:, 1:])
        loss.backward()
        opt.step()


def generate_paged(out):
    import train as T
    from minagi import store
    cfgd = make_model(out)
    model, cfg, pool, man = T.build_paged(out, torch.device("cpu"), resident=4, ram_capacity=4)
    cfg.halt_freeze = True
    model.cfg.halt_freeze = True
    opt = adamw(model)
    pool.attach_optimiser(opt)
    pool.dying_at = 0.75
    gen = torch.Generator().manual_seed(11)
    train_steps(model, opt, 4, gen)

    # Grow by four (new files e00008..e00011) and train on, then prune three experts nothing has
    # asked for, with the real PagedPool.prune (the victims are chosen by making them the stalest, since
    # which experts a 24-character window happens to admit is otherwise arbitrary). The uids that
    # remain are not 0..n any more, which is the whole point of naming expert files by uid.
    pool.add_experts(4, step=4)
    resync(opt, model)
    train_steps(model, opt, 3, gen)
    resident = {e for e in pool.slots if e >= 0}
    victims = [i for i in range(8) if i not in resident][:3]
    victim_uids = [int(pool.uid[i]) for i in victims]
    for i in range(pool._n):
        pool.last_seen[i] = 0.0 if i in victims else float(pool.segments)
    gone = pool.prune(step=7, survival=4)
    assert gone == len(victims), (gone, victims)
    resync(opt, model)
    train_steps(model, opt, 2, gen)
    # one more growth after the prune: the new uid continues from next_uid, not from the count
    pool.add_experts(1, step=9)
    resync(opt, model)
    train_steps(model, opt, 1, gen)

    print(f"  paged: pool grew 8->12, pruned uids {victim_uids}, grew by one more: {pool.n_experts()} experts, "
          f"uid {pool.uid.tolist()}, next_uid {pool.next_uid}")
    store.save(model, out, step=9, val=1.234, opt=opt, cfg=dataclasses.asdict(cfg), verbose=True,
               extra={"read_chars": 4321, "read_nats": 1234.5, "plasticity": {"lr_mult": 0.75, "history": [1, 2]},
                      "context_now": 23})
    return pool


def generate_non_paged(out):
    """The pre-paging save: a SharedPool model, store.save's first branch."""
    from minagi import store
    from minagi.recur import RecurCoder, RecurConfig
    make_model(out + "_seed")
    man = json.load(open(os.path.join(out + "_seed", "manifest.json")))
    cfg = RecurConfig(**{k: v for k, v in man["cfg"].items() if k in RecurConfig.__dataclass_fields__})
    torch.manual_seed(5)
    model = RecurCoder(cfg)
    cfg.halt_freeze = True
    model.cfg.halt_freeze = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(12)
    model.train()
    for _ in range(2):
        ids = torch.randint(0, 256, (1, 24), generator=gen)
        opt.zero_grad()
        _, loss = model(ids[:, :-1], ids[:, 1:])
        loss.backward()
        opt.step()
    if os.path.exists(out):
        shutil.rmtree(out)
    store.save(model, out, step=2, val=2.5, opt=opt, cfg=dataclasses.asdict(cfg), verbose=True)
    shutil.rmtree(out + "_seed")


def bf16_golden(path):
    """float32 bit patterns -> bf16 bits, made by torch: random, exact ties, near-ties, subnormals, specials."""
    rng = np.random.default_rng(20260930)
    parts = []
    parts.append(rng.integers(0, 2 ** 32, size=20000, dtype=np.uint64).astype(np.uint32))
    # exact ties: low 16 bits == 0x8000, random sign/exponent/high mantissa (normal range)
    sign = rng.integers(0, 2, size=8000, dtype=np.uint32) << 31
    exp = rng.integers(1, 255, size=8000, dtype=np.uint32) << 23
    mant = rng.integers(0, 128, size=8000, dtype=np.uint32) << 16
    parts.append(sign | exp | mant | np.uint32(0x8000))
    # one ulp either side of a tie
    for low in (0x7FFF, 0x8001):
        sign = rng.integers(0, 2, size=2000, dtype=np.uint32) << 31
        exp = rng.integers(1, 255, size=2000, dtype=np.uint32) << 23
        mant = rng.integers(0, 128, size=2000, dtype=np.uint32) << 16
        parts.append(sign | exp | mant | np.uint32(low))
    # subnormals (exponent 0) with random mantissas, both signs
    sign = rng.integers(0, 2, size=3000, dtype=np.uint32) << 31
    parts.append(sign | rng.integers(0, 1 << 23, size=3000, dtype=np.uint32))
    # the overflow edge: exponent 254, mantissa near all ones
    sign = rng.integers(0, 2, size=1000, dtype=np.uint32) << 31
    parts.append(sign | (np.uint32(254) << 23) | rng.integers((1 << 23) - 40000, 1 << 23, size=1000, dtype=np.uint32))
    specials = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
                         0xFF800001, 0x7FBFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x00000001, 0x007FFFFF, 0x00800000,
                         0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x3F808000, 0x3F818000],
                        dtype=np.uint32)
    parts.append(specials)
    bits = np.concatenate(parts).astype(np.uint32)
    t = torch.from_numpy(bits.view(np.int32).copy()).view(torch.float32)
    packed = pack_bf16(t)
    np.savez(path, bits=bits.view(np.int32), bf16=packed)
    print(f"  bf16_golden.npz: {len(bits)} values")


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    py = os.path.join(out_dir, "py_weights")
    npg = os.path.join(out_dir, "non_paged_weights")
    for d in (py, npg):
        if os.path.exists(d):
            shutil.rmtree(d)
    torch.manual_seed(0)
    generate_paged(py)
    generate_non_paged(npg)
    bf16_golden(os.path.join(out_dir, "bf16_golden.npz"))

    expected = {}
    for name, d in (("py_weights", py), ("non_paged_weights", npg)):
        e = {"manifest": json.load(open(os.path.join(d, "manifest.json"))), "files": {}}
        for f in ("core.npz", "routers.npz", "optim.npz"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                e["files"][f] = describe_npz(p)
        for f in sorted(os.listdir(os.path.join(d, "experts"))):
            e["files"]["experts/" + f] = describe_npz(os.path.join(d, "experts", f))
        expected[name] = e
    with open(os.path.join(out_dir, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1, sort_keys=True)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out_dir) for f in fs)
    print(f"fixtures written to {out_dir} ({total / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------------------------
# verify: Python loads what Rust wrote
# ---------------------------------------------------------------------------------------------

def verify(path, ref_path):
    import train as T
    from minagi import store
    from minagi.recur import _load_dir

    ref = np.load(ref_path)
    refkeys = set(ref.files)
    bad, n_checks = [], 0

    def check(cond, msg):
        nonlocal n_checks
        n_checks += 1
        if not cond:
            bad.append(msg)

    def same(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    def same_flat(a, b):
        """Expert arrays are stored flat in the reference (their shape follows from d_ff, d_model)."""
        return same(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))

    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    uids = [int(u) for u in ref["meta/uids"]]
    n = len(uids)
    check(man["n_experts"] == n, f"manifest n_experts {man['n_experts']} != {n}")
    check(man.get("paged") is True, "manifest is not marked paged")
    check([int(u) for u in man["telemetry"]["uid"]] == uids, "telemetry.uid differs")
    check([e["id"] for e in man["experts"]] == uids, "experts list differs")
    check(int(man["telemetry"]["next_uid"]) == int(ref["meta/next_uid"]), "next_uid differs")
    check(int(man["telemetry"]["segments"]) == int(ref["meta/segments"]), "segments differs")

    # 1. the real loader: _load_dir -> build_paged (paged manifests), read_only so nothing is written
    model, info = _load_dir(path, torch.device("cpu"), read_only=True)
    pool = model.pool
    check(pool.n_experts() == n, f"pool has {pool.n_experts()} experts, expected {n}")
    check(pool.uid.tolist() == uids, f"pool.uid {pool.uid.tolist()} != {uids}")
    check(int(pool.next_uid) == int(ref["meta/next_uid"]), "pool.next_uid")
    check(int(pool.segments) == int(ref["meta/segments"]), "pool.segments")
    check(info["step"] == man["step"], "step")

    # 2. trunk, routers, gates: model.state_dict() must hold exactly what was written
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    for key in sorted(refkeys):
        if key.startswith("core/") or key.startswith("routers/"):
            name = key.split("/", 1)[1]
            check(name in sd, f"state_dict lacks {name}")
            if name in sd:
                check(same(sd[name], ref[key]), f"tensor {name} differs after load")

    # 3. every expert, through the paged pool's own Tiers (which reads the files the way training does)
    for u in uids:
        ent = pool.tiers.fetch(u)
        for k in ("w1", "w3", "w2"):
            check(same_flat(ent[k].numpy(), ref[f"expert/{u}/{k}"]), f"expert {u} {k} differs")
        for k in ("w1_m", "w1_v", "w3_m", "w3_v", "w2_m", "w2_v"):
            key = f"expert/{u}/{k}"
            if key in refkeys:
                # the reference holds the float32 values Rust started from; what Python must read is
                # Python's own bf16 rounding of them -- except from an old file that stores fp32
                # moments, where unpack_bf16 passes them through unchanged
                if "meta/expert_moments_fp32" in refkeys:
                    want = ref[key].astype(np.float32)
                else:
                    want = unpack_bf16(pack_bf16(torch.from_numpy(ref[key]))).numpy()
                check(k in ent and same_flat(ent[k].numpy(), want), f"expert {u} moment {k} differs")
            else:
                check(k not in ent, f"expert {u} has an unexpected moment {k}")

    # 4. optimiser state through the real _load_optim
    opt = adamw(model)
    pool.attach_optimiser(opt)
    store._load_optim(opt, model, path)
    name_of = {id(p): nm for nm, p in model.named_parameters()}
    restored = 0
    for g in opt.param_groups:
        for p in g["params"]:
            nm = name_of.get(id(p))
            km = f"optim/{nm}|m"
            if km not in refkeys:
                continue
            st = opt.state.get(p)
            check(bool(st) and "exp_avg" in st, f"optimiser state for {nm} was not restored")
            if not st or "exp_avg" not in st:
                continue
            restored += 1
            for nm_state, suffix in (("exp_avg", "m"), ("exp_avg_sq", "v")):
                want = unpack_bf16(pack_bf16(torch.from_numpy(ref[f"optim/{nm}|{suffix}"]))).numpy()
                check(same(st[nm_state].numpy(), want), f"optimiser {nm_state} of {nm} differs")
            kt = f"optim/{nm}|t"
            if kt in refkeys:
                check(float(st["step"]) == float(ref[kt]), f"optimiser step of {nm}")
    check(restored > 0, "no optimiser state was restored at all")

    # 5. the model actually runs on what was loaded
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, 256, (1, 16)))
    check(bool(torch.isfinite(logits).all()), "forward pass produced non-finite logits")
    store.summarise(path)

    if bad:
        print("MISMATCHES:", file=sys.stderr)
        for b in bad[:40]:
            print("  " + b, file=sys.stderr)
        print(json.dumps({"ok": False, "checks": n_checks, "failures": len(bad)}))
        return 1
    print(json.dumps({"ok": True, "checks": n_checks, "experts": n, "optimiser_params_restored": restored}))
    return 0


def resave(src, dst):
    """Rust-written directory -> loaded by Python -> saved by Python (store.save), then read by Rust."""
    import train as T
    from minagi import store
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    marker = os.path.join(dst, "COMPLETE")
    if os.path.exists(marker):
        os.remove(marker)       # Python's save knows nothing of it, and a Python-written folder has none
    with open(os.path.join(dst, "manifest.json")) as f:
        man = json.load(f)
    model, cfg, pool, _ = T.build_paged(dst, torch.device("cpu"), resident=None, ram_capacity=4)
    opt = adamw(model)
    pool.attach_optimiser(opt)
    store._load_optim(opt, model, dst)
    extra = {k: man[k] for k in ("read_chars", "read_nats", "plasticity", "context_now") if k in man}
    store.save(model, dst, step=man.get("step"), val=man.get("val"), opt=opt, cfg=dataclasses.asdict(cfg),
               extra=extra)
    # the Rust-only namespace and marker are not Python's to keep: save() rebuilt the manifest from scratch
    print(json.dumps({"ok": True, "experts": pool.n_experts(), "uid": pool.uid.tolist()}))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", default=FIXTURES)
    v = sub.add_parser("verify")
    v.add_argument("path")
    v.add_argument("reference")
    r = sub.add_parser("resave")
    r.add_argument("src")
    r.add_argument("dst")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(absp(a.out))
        return 0
    if a.cmd == "resave":
        return resave(absp(a.src), absp(a.dst))
    return verify(absp(a.path), absp(a.reference))


if __name__ == "__main__":
    sys.exit(main())
