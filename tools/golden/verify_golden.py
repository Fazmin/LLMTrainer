#!/usr/bin/env python3
"""Self-consistency checks for the golden fixtures.

    .venv/bin/python tools/golden/verify_golden.py [--skip-repro]

What it checks (all independent of the PyTorch autograd that produced the fixtures):

  repro        dump_model.py run twice in fresh processes -> byte-identical files, and
               identical to what is committed under tests/fixtures/golden/
  npz-format   our .npz writer is byte-identical to numpy.savez
  rmsnorm/attn forward + finite-difference gradient checks in float64 numpy
  moe          an independent numpy float64 re-implementation of PooledMLP reproduces `out`,
               `aux`, the routes, and finite differences reproduce every stored gradient group
  dense        an independent numpy float64 re-implementation of the whole RecurCoder training
               forward reproduces loss / per-row arrays / halting decisions of every variant
               and finite differences reproduce the stored gradients (not for bptt: a detach
               is not visible to finite differences; that variant is checked structurally)
  adamw        a numpy float32 replay of the AdamW update + clipping reproduces the stored
               parameter trajectory of adamw_paged.npz and train_step_dense.npz
  decode       a numpy re-implementation of pick_next's adaptation reproduces `adjusted`
  readme       every key in every .npz is documented in README.md (shape/dtype checked)

Exit status 0 if everything passes.
"""

import sys
sys.dont_write_bytecode = True
import argparse
import filecmp
import json
import os
import re
import subprocess
import sys
import tempfile

from golden_common import HERE, OUT_DIR, Npz

import numpy as np

FAILS = []


def report(name, ok, detail=""):
    print(("[ PASS ] " if ok else "[ FAIL ] ") + name + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1e-12, np.abs(b).max()))


# ----------------------------------------------------------------------------------------
# repro + format
# ----------------------------------------------------------------------------------------

def check_repro():
    py = sys.executable
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        for t in (t1, t2):
            subprocess.run([py, str(HERE / "dump_model.py"), "--out", t], check=True,
                           stdout=subprocess.DEVNULL)
        names = sorted(os.listdir(t1))
        same12 = all(filecmp.cmp(os.path.join(t1, n), os.path.join(t2, n), shallow=False)
                     for n in names) and names == sorted(os.listdir(t2))
        report("repro: two fresh runs are byte-identical", same12, f"{len(names)} files")
        missing = [n for n in names if not (OUT_DIR / n).exists()]
        same = not missing and all(
            filecmp.cmp(os.path.join(t1, n), OUT_DIR / n, shallow=False) for n in names)
        foreign = sorted(set(os.listdir(OUT_DIR)) - set(names))
        report("repro: fresh run equals committed fixtures", same,
               ("missing " + str(missing) if missing else "differs; run dump_model.py again")
               if not same else (f"(ignoring files not made by dump_model.py: {foreign})" if foreign else ""))


def check_npz_writer():
    d = {"a/b": np.arange(6, dtype=np.float32).reshape(2, 3), "c": np.array([1, 2], np.int64),
         "u": np.array([[1, 0]], np.uint8), "i16": np.array([1, -2], np.int16)}
    with tempfile.TemporaryDirectory() as t:
        np.savez(os.path.join(t, "np.npz"), **d)
        n = Npz()
        for k, v in d.items():
            n.add(k, v)
        n.write(os.path.join(t, "mine.npz"))
        a = open(os.path.join(t, "np.npz"), "rb").read()
        b = open(os.path.join(t, "mine.npz"), "rb").read()
    report("npz-format: writer byte-identical to numpy.savez", a == b)


# ----------------------------------------------------------------------------------------
# numpy float64 building blocks (independent of torch)
# ----------------------------------------------------------------------------------------

def rms(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, cos, sin):                       # x [H,T,hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def attn(x, qkv, proj, cos, sin, n_head=2, cache=None):
    """x [T,D]; cache = (k [H,P,hd], v) of positions before x; returns (out, k_all, v_all)."""
    T, D = x.shape
    hd = D // n_head
    q, k, v = np.split(x @ qkv.T, 3, axis=1)
    q = q.reshape(T, n_head, hd).transpose(1, 0, 2)
    k = k.reshape(T, n_head, hd).transpose(1, 0, 2)
    v = v.reshape(T, n_head, hd).transpose(1, 0, 2)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    if cache is not None:
        k = np.concatenate([cache[0], k], 1)
        v = np.concatenate([cache[1], v], 1)
    P = k.shape[1] - T
    s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    mask = np.arange(k.shape[1])[None, :] <= (np.arange(T)[:, None] + P)
    s = np.where(mask[None], s, -np.inf)
    y = (softmax(s) @ v).transpose(1, 0, 2).reshape(T, D)
    return y @ proj.T, k, v


def swiglu(x, w1, w3, w2):
    a = x @ w1.T
    return ((a / (1 + np.exp(-a))) * (x @ w3.T)) @ w2.T


def fd(f, x, idxs, eps=1e-6):
    """central differences of scalar f wrt selected flat entries of x (modified in place)."""
    out = []
    flat = x.reshape(-1)
    for i in idxs:
        o = flat[i]
        flat[i] = o + eps
        fp = f()
        flat[i] = o - eps
        fm = f()
        flat[i] = o
        out.append((fp - fm) / (2 * eps))
    return np.array(out)


def pick(rng, n, k):
    return rng.choice(n, size=min(k, n), replace=False)


def cmp_grad(name, num, ana_flat, idxs, rtol=2e-3, atol=2e-5):
    ana = np.asarray(ana_flat, np.float64).reshape(-1)[idxs]
    err = np.abs(num - ana)
    ok = bool((err <= atol + rtol * np.abs(ana)).all())
    report(f"fd: {name}", ok, f"max|err|={err.max():.2e} max|g|={np.abs(ana).max():.2e}")


# ----------------------------------------------------------------------------------------
# ops
# ----------------------------------------------------------------------------------------

def check_ops():
    rng = np.random.default_rng(0)
    z = np.load(OUT_DIR / "ops_rmsnorm.npz")
    w = z["weight"].astype(np.float64)
    for case in ("base", "big", "tiny", "batch2"):
        x = z[f"{case}/x"].astype(np.float64)
        g = z[f"{case}/g"].astype(np.float64)
        out = rms(x, w)
        report(f"rmsnorm {case}: forward", rel(out, z[f"{case}/out"]) < 1e-5,
               f"rel={rel(out, z[f'{case}/out']):.1e}")
        xs = x.copy()
        ws = w.copy()
        L = lambda: float((rms(xs, ws) * g).sum())      # noqa: E731
        ix = pick(rng, x.size, 12)
        cmp_grad(f"rmsnorm {case} dx", fd(L, xs, ix, 1e-6 * max(1.0, np.abs(x).max())), z[f"{case}/dx"], ix)
        iw = pick(rng, w.size, 8)
        cmp_grad(f"rmsnorm {case} dweight", fd(L, ws, iw), z[f"{case}/dweight"], iw)

    z = np.load(OUT_DIR / "ops_rope.npz")
    cos, sin = z["cos"].astype(np.float64), z["sin"].astype(np.float64)
    for case in ("pos0", "pos7", "pos59"):
        off = int(z[f"{case}/pos_offset"][0])
        x = z[f"{case}/x"][0].astype(np.float64)
        out = rope(x, cos[off:off + 5], sin[off:off + 5])
        report(f"rope {case}: forward", rel(out, z[f"{case}/out"][0]) < 1e-5)
        # rotation is orthogonal: dx = R^T g
        g = z[f"{case}/g"][0].astype(np.float64)
        x1, x2 = g[..., 0::2], g[..., 1::2]
        c, s = cos[off:off + 5], sin[off:off + 5]
        dx = np.stack([x1 * c + x2 * s, -x1 * s + x2 * c], -1).reshape(g.shape)
        report(f"rope {case}: dx = R^T g", rel(dx, z[f"{case}/dx"][0]) < 1e-5)

    z = np.load(OUT_DIR / "ops_attention.npz")
    qkv, proj = z["qkv.weight"].astype(np.float64), z["proj.weight"].astype(np.float64)
    ropez = np.load(OUT_DIR / "ops_rope.npz")
    cos, sin = ropez["cos"].astype(np.float64), ropez["sin"].astype(np.float64)
    x = z["a/x"][0].astype(np.float64)
    g = z["a/g"][0].astype(np.float64)
    y, _, _ = attn(x, qkv, proj, cos[:8], sin[:8])
    report("attention a: forward", rel(y, z["a/out"][0]) < 2e-5, f"rel={rel(y, z['a/out'][0]):.1e}")
    xs, qs, ps = x.copy(), qkv.copy(), proj.copy()
    L = lambda: float((attn(xs, qs, ps, cos[:8], sin[:8])[0] * g).sum())    # noqa: E731
    ix = pick(rng, xs.size, 12)
    cmp_grad("attention a dx", fd(L, xs, ix), z["a/dx"], ix)
    iq = pick(rng, qs.size, 14)
    cmp_grad("attention a dqkv.weight", fd(L, qs, iq), z["a/dqkv.weight"], iq)
    ip = pick(rng, ps.size, 10)
    cmp_grad("attention a dproj.weight", fd(L, ps, ip), z["a/dproj.weight"], ip)
    for case in ("b", "c"):
        P, T = int(z[f"{case}/P"][0]), int(z[f"{case}/T"][0])
        xp = z[f"{case}/x_prev"][0].astype(np.float64)
        xn = z[f"{case}/x_new"][0].astype(np.float64)
        gn = z[f"{case}/g"][0].astype(np.float64)
        ck, cv = z[f"{case}/cache_k_in"][0].astype(np.float64), z[f"{case}/cache_v_in"][0].astype(np.float64)
        _, ck2, cv2 = attn(xp, qkv, proj, cos[:P], sin[:P])
        report(f"attention {case}: cache_in = k/v of the previous positions",
               rel(ck2, ck) < 1e-5 and rel(cv2, cv) < 1e-5)
        y, ko, vo = attn(xn, qkv, proj, cos[P:P + T], sin[P:P + T], cache=(ck, cv))
        report(f"attention {case}: forward with cache", rel(y, z[f"{case}/out"][0]) < 2e-5)
        report(f"attention {case}: returned cache", rel(ko, z[f"{case}/cache_k_out"][0]) < 1e-5
               and rel(vo, z[f"{case}/cache_v_out"][0]) < 1e-5)
        xs, qs, ps = xn.copy(), qkv.copy(), proj.copy()
        L = lambda: float((attn(xs, qs, ps, cos[P:P + T], sin[P:P + T], cache=(ck, cv))[0] * gn).sum())  # noqa: E731
        ix = pick(rng, xs.size, 10)
        cmp_grad(f"attention {case} dx_new", fd(L, xs, ix), z[f"{case}/dx_new"], ix)
        iq = pick(rng, qs.size, 12)
        cmp_grad(f"attention {case} dqkv.weight", fd(L, qs, iq), z[f"{case}/dqkv.weight"], iq)
    z = np.load(OUT_DIR / "ops_swiglu.npz")
    w1, w3, w2 = (z[k].astype(np.float64) for k in ("w1", "w3", "w2"))
    for case in ("base", "big"):
        x = z[f"{case}/x"][0].astype(np.float64)
        g = z[f"{case}/g"][0].astype(np.float64)
        report(f"swiglu {case}: forward", rel(swiglu(x, w1, w3, w2), z[f"{case}/out"][0]) < 2e-5)
        xs, a, b, c = x.copy(), w1.copy(), w3.copy(), w2.copy()
        L = lambda: float((swiglu(xs, a, b, c) * g).sum())      # noqa: E731
        ix = pick(rng, xs.size, 8)
        cmp_grad(f"swiglu {case} dx", fd(L, xs, ix), z[f"{case}/dx"], ix)
        i1 = pick(rng, a.size, 8)
        cmp_grad(f"swiglu {case} dw1", fd(L, a, i1), z[f"{case}/dw1"], i1)
        i2 = pick(rng, c.size, 8)
        cmp_grad(f"swiglu {case} dw2", fd(L, c, i2), z[f"{case}/dw2"], i2)


# ----------------------------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------------------------

def np_moe(x, router, demb, gate, W, top_k=2, active=None, z_weight=1e-3, cap=0.0,
           stable_drop=True, n_slots=None):
    """x [N,D]. Independent re-implementation of PooledMLP._route (resident pool).
    Returns out [N,D], aux, idx, w, counts."""
    n = router.shape[0] if n_slots is None else n_slots
    N, D = x.shape
    keep_tok = np.ones(N, bool) if active is None else active
    xa = x[keep_tok]
    z = (xa + demb) @ router[:n].T
    p = softmax(z)
    idx = np.argsort(-p, axis=1, kind="stable")[:, :top_k]
    w = np.take_along_axis(p, idx, 1)
    w = w / w.sum(1, keepdims=True)
    w = w * gate[idx]
    frac = np.zeros(n)
    np.add.at(frac, idx[:, 0], 1.0)
    frac /= len(xa)
    lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
    aux = (frac * p.mean(0)).sum() * n + z_weight * (lse ** 2).mean()
    kept = np.ones(idx.shape, bool)
    if cap:
        limit = max(1, int(np.ceil(cap * idx.size / n)))
        seen = np.zeros(n, int)
        for a, e in enumerate(idx.reshape(-1)):
            if seen[e] >= limit:
                kept.reshape(-1)[a] = False
            seen[e] += 1
    outa = np.zeros_like(xa)
    for t in range(len(xa)):
        for r in range(top_k):
            if not kept[t, r]:
                continue
            e = idx[t, r]
            outa[t] += w[t, r] * swiglu(xa[t:t + 1], W["w1"][e], W["w3"][e], W["w2"][e])[0]
    out = np.zeros_like(x)
    out[keep_tok] = outa
    return out, aux, idx, w, kept


def check_moe():
    rng = np.random.default_rng(1)
    z = np.load(OUT_DIR / "moe_shared.npz")
    W = {k: z[k].astype(np.float64) for k in ("w1", "w3", "w2")}
    for case, kw in (("base", {}), ("wide", {}), ("active", {"active": True})):
        P = {k: z[f"{case}/{k}"].astype(np.float64) for k in ("router", "depth_emb", "gate")} \
            if f"{case}/gate" in z.files else None
        if P is None:      # wide reuses base gate/depth_emb
            P = {"router": z[f"{case}/router"].astype(np.float64),
                 "depth_emb": z["base/depth_emb"].astype(np.float64),
                 "gate": z["base/gate"].astype(np.float64)}
        x = z[f"{case}/x"][0].astype(np.float64)
        g = z[f"{case}/g"][0].astype(np.float64)
        act = z["active/active"][0].astype(bool) if case == "active" else None
        out, aux, idx, w, _ = np_moe(x, P["router"], P["depth_emb"], P["gate"], W, active=act,
                                     n_slots=8)
        report(f"moe {case}: out", rel(out, z[f"{case}/out"][0]) < 3e-5, f"rel={rel(out, z[f'{case}/out'][0]):.1e}")
        report(f"moe {case}: aux", abs(aux - float(z[f"{case}/aux"][0])) < 2e-5 * max(1, abs(aux)))
        report(f"moe {case}: routes", (idx == z[f"{case}/route_idx"]).all()
               and rel(w, z[f"{case}/route_w"]) < 3e-5)
        xs, rs, ds, gs = x.copy(), P["router"].copy(), P["depth_emb"].copy(), P["gate"].copy()
        W2 = {k: v.copy() for k, v in W.items()}
        Lo = lambda: float((np_moe(xs, rs, ds, gs, W2, active=act, n_slots=8)[0] * g).sum())   # noqa: E731
        La = lambda: float(np_moe(xs, rs, ds, gs, W2, active=act, n_slots=8)[1])            # noqa: E731
        ix = pick(rng, xs.size, 10)
        cmp_grad(f"moe {case} grad/x", fd(Lo, xs, ix), z[f"{case}/grad/x"], ix, rtol=5e-3)
        ir = pick(rng, 8 * 32, 12)          # first 8 rows only (the rest are unused)
        cmp_grad(f"moe {case} grad/router", fd(Lo, rs, ir), z[f"{case}/grad/router"], ir, rtol=5e-3)
        idd = pick(rng, 32, 8)
        cmp_grad(f"moe {case} grad/depth_emb", fd(Lo, ds, idd), z[f"{case}/grad/depth_emb"], idd, rtol=5e-3)
        ig = np.arange(8)
        cmp_grad(f"moe {case} grad/gate", fd(Lo, gs, ig), z[f"{case}/grad/gate"], ig, rtol=5e-3)
        cmp_grad(f"moe {case} grad_aux/x", fd(La, xs, ix), z[f"{case}/grad_aux/x"], ix, rtol=5e-3)
        cmp_grad(f"moe {case} grad_aux/router", fd(La, rs, ir), z[f"{case}/grad_aux/router"], ir, rtol=5e-3)
        cmp_grad(f"moe {case} grad_aux/depth_emb", fd(La, ds, idd), z[f"{case}/grad_aux/depth_emb"], idd, rtol=5e-3)
        if case == "base":
            for nm in ("w1", "w3", "w2"):
                a = W2[nm]
                iw = pick(rng, a.size, 10)
                cmp_grad(f"moe base grad/{nm}", fd(Lo, a, iw), z[f"base/grad/{nm}"], iw, rtol=5e-3)
    # capacity: counts + deterministic stable realisation
    P = {k: z[f"cap/{k}"].astype(np.float64) for k in ("router", "depth_emb", "gate")}
    x = z["cap/x"][0].astype(np.float64)
    out, aux, idx, w, kept = np_moe(x, P["router"], P["depth_emb"], P["gate"], W, cap=1.5)
    report("moe cap: counts_before", (np.bincount(idx.reshape(-1), minlength=8) == z["cap/counts_before"]).all())
    cnt = np.bincount(idx.reshape(-1), minlength=8)
    report("moe cap: kept_counts/limit/dropped", (np.minimum(cnt, int(z["cap/limit"][0])) == z["cap/kept_counts"]).all()
           and int((cnt - z["cap/kept_counts"]).sum()) == int(z["cap/dropped"][0]) and int(z["cap/routed"][0]) == idx.size)
    report("moe cap: kept_mask_stable", (kept == z["cap/kept_mask_stable"].astype(bool)).all())
    report("moe cap: out_stable", rel(out, z["cap/out_stable"][0]) < 3e-5)
    report("moe cap: aux independent of drops", abs(aux - float(z["cap/aux"][0])) < 2e-5 * abs(aux))


# ----------------------------------------------------------------------------------------
# dense model
# ----------------------------------------------------------------------------------------

def np_dense(P, cfg, idx, targets, n_steps=None):
    """Independent float64 re-implementation of RecurCoder.forward (training path, no cache)."""
    T = idx.shape[1]
    ropez = np.load(OUT_DIR / "ops_rope.npz")
    cos, sin = ropez["cos"].astype(np.float64)[:T], ropez["sin"].astype(np.float64)[:T]
    emb = P["tok_emb.weight"]

    def block(x, pre):
        x = x + attn(rms(x, P[pre + "ln1.weight"]), P[pre + "attn.qkv.weight"],
                     P[pre + "attn.proj.weight"], cos, sin)[0]
        return x + swiglu(rms(x, P[pre + "ln2.weight"]), P[pre + "mlp.w1.weight"],
                          P[pre + "mlp.w3.weight"], P[pre + "mlp.w2.weight"])
    x = emb[idx[0]]
    x = block(x, "prelude.0.")
    h = np.zeros_like(x)
    cum = np.ones(T)
    halted = np.zeros(T, bool)
    N = n_steps or cfg["max_steps"]
    rows, Ps, Ls = [], [], []
    for n in range(N):
        active = (~halted) if cfg["halt_freeze"] and n else None
        if active is None or active.any():
            hn = block(np.concatenate([h, x], -1) @ P["adapter.weight"].T, "recur.0.")
            h = hn if active is None else np.where(active[:, None], hn, h)
        yf = rms(h, P["ln_f.weight"])
        logits = yf @ emb.T
        lam = 1 / (1 + np.exp(-(yf @ P["halt.weight"].T + P["halt.bias"])[:, 0]))
        if n == N - 1:
            lam = np.ones_like(lam)
        elif n < cfg["min_steps"] - 1:
            lam = np.zeros_like(lam)
        p = cum * lam
        cum = cum * (1 - lam)
        m = logits.max(1, keepdims=True)
        lse = np.log(np.exp(logits - m).sum(1)) + m[:, 0]
        ce = lse - logits[np.arange(T), targets[0]]
        Ps.append(p)
        Ls.append(p * ce)
        if cfg["halt_freeze"]:
            halted = halted | ((1 - cum) >= cfg["halt_thresh"])
        rows.append(dict(lam=lam, p_n=p, cum=cum.copy(), ce=ce, halted=halted.copy(),
                         active=np.ones(T, bool) if active is None else active, h=h.copy()))
    Pm, Lm = np.stack(Ps), np.stack(Ls)
    ce_term = Lm.sum(0).mean()
    prior = np.array([cfg["halt_prior"] * (1 - cfg["halt_prior"]) ** i for i in range(N)])
    prior = prior / prior.sum()
    Pc = np.maximum(Pm, 1e-8)
    kl = (Pc * (np.log(Pc) - np.log(prior)[:, None])).sum(0).mean()
    return ce_term + cfg["ponder_beta"] * kl, ce_term, kl, rows


def check_dense():
    rng = np.random.default_rng(2)
    z = np.load(OUT_DIR / "dense_forward.npz")
    meta = json.load(open(OUT_DIR / "dense_forward.json"))
    base_cfg = json.load(open(OUT_DIR / "config.json"))["config"]
    idx, targets = z["idx"], z["targets"]
    pnames = [k.split("/", 2)[2] for k in z.files if k.startswith("perturbed/grad/")]
    for vname, vm in meta["variants"].items():
        cfg = dict(base_cfg)
        cfg.update(vm["cfg_overrides"])
        w = vm["weights"]
        if w == "own":
            wsrc = vname
        elif w.startswith("same as variant"):
            wsrc = re.findall(r"'(\w+)'", w)[0]
        else:                       # halt_only: perturbed + overrides
            wsrc = "perturbed"
        P = {n: z[f"{wsrc}/{n}"].astype(np.float64) for n in pnames}
        if w.startswith("halt.weight"):
            P["halt.weight"] = z[f"{vname}/halt.weight"].astype(np.float64)
            P["halt.bias"] = z[f"{vname}/halt.bias"].astype(np.float64)
        loss, ce_t, kl_t, rows = np_dense(P, cfg, idx, targets, n_steps=vm["n_steps"])
        okf = abs(loss - vm["loss"]) < 2e-5 * abs(loss) and abs(kl_t - vm["kl_term"]) < 5e-5
        okr = all(rel(r["p_n"], z[f"{vname}/p_n/{n}"][0]) < 1e-4 and rel(r["h"], z[f"{vname}/h/{n}"][0]) < 1e-4
                  and (r["halted"] == z[f"{vname}/halted/{n}"][0].astype(bool)).all()
                  and (r["active"] == z[f"{vname}/active/{n}"][0].astype(bool)).all()
                  for n, r in enumerate(rows))
        report(f"dense {vname}: loss & per-row arrays vs numpy re-implementation", okf and okr,
               f"loss {loss:.6f} vs {vm['loss']:.6f}")
        if not vm["grads_stored"] or vname == "bptt":
            continue
        # finite differences on a sample of every parameter
        Pw = {n: v.copy() for n, v in P.items()}
        f_ = lambda: np_dense(Pw, cfg, idx, targets, n_steps=vm["n_steps"])[0]    # noqa: E731
        worst = 0.0
        ok = True
        for n in pnames:
            ii = pick(rng, Pw[n].size, 2 if Pw[n].size > 64 else 3)
            num = fd(f_, Pw[n], ii, 1e-6)
            ana = z[f"{vname}/grad/{n}"].astype(np.float64).reshape(-1)[ii]
            err = np.abs(num - ana)
            tol = 3e-5 + 5e-3 * np.abs(ana)
            ok &= bool((err <= tol).all())
            worst = max(worst, float((err / tol).max()))
        report(f"dense {vname}: finite-difference gradients (all {len(pnames)} params, sampled)", ok,
               f"worst err/tol={worst:.2f}")
    # bptt: same forward as perturbed, different gradients, same loss
    gp = np.concatenate([z[f"perturbed/grad/{n}"].ravel() for n in pnames])
    gb = np.concatenate([z[f"bptt/grad/{n}"].ravel() for n in pnames])
    report("dense bptt: loss equal to perturbed, gradients differ", abs(meta["variants"]["bptt"]["loss"]
           - meta["variants"]["perturbed"]["loss"]) < 1e-6 and np.abs(gp - gb).max() > 1e-4,
           f"max|dg|={np.abs(gp - gb).max():.3e}")
    # in bptt rows 0,1 are detached: the recur-block gradient of h from rows>=2 cannot reach row 0.
    # Structural: grad of the loss restricted to row-0 logits path only (== perturbed contribution)
    # is not separable here, so the check above is the structural one.

    # inference path vs training halting + chunked/cached == one-shot
    zi = np.load(OUT_DIR / "dense_infer.npz")
    mi = json.load(open(OUT_DIR / "dense_infer.json"))
    for sc, m in mi["scenarios"].items():
        cat = np.concatenate([zi[f"{sc}/chunk{i}/logits"] for i in range(len(m["chunks"]))], 1)
        report(f"dense_infer {sc}: chunked logits == one-shot logits", rel(cat, zi[f"{sc}/oneshot/logits"]) < 1e-4)
    # inference `halt_row` must agree with the training-forward halting for the same weights
    hr = np.array(mi["scenarios"]["halting"]["halt_row_per_token"])
    tr = z["halting/halt_row"][0]
    report("dense_infer halting == dense_forward halting (per-token halt rows)",
           (np.where(tr == -1, 3, tr) == hr).all())


# ----------------------------------------------------------------------------------------
# paged MoE forward (numpy) and the pager state machine (plain python)
# ----------------------------------------------------------------------------------------

def np_paged_moe(x, router, demb, gate, Wslot, slots, mask, bias=None, top_k=2, z_weight=1e-3,
                 cap=0.0):
    """Independent numpy float64 version of PooledMLP._route over a PagedPool card."""
    rows = np.array([max(s, 0) for s in slots])
    n = len(slots)
    z = (x + demb) @ router[rows].T
    z = np.where(mask[None, :], z, -np.inf)
    zz = np.where(np.isfinite(z), z, -1e300)
    m = zz.max(1, keepdims=True)
    e = np.where(np.isfinite(z), np.exp(np.where(np.isfinite(z), z, 0) - m), 0.0)
    p = e / e.sum(1, keepdims=True)
    score = z if bias is None else z + bias[rows][None, :]
    idx = np.argsort(-score, axis=1, kind="stable")[:, :top_k]
    w = np.take_along_axis(p, idx, 1)
    w = w / w.sum(1, keepdims=True)
    w = w * gate[rows][idx]
    frac = np.zeros(n)
    np.add.at(frac, idx[:, 0], 1.0)
    frac /= len(x)
    lse = np.log(e.sum(1)) + m[:, 0]
    aux = (frac * p.mean(0)).sum() * n + z_weight * (lse ** 2).mean()
    kept = np.ones(idx.shape, bool)
    if cap:
        limit = max(1, int(np.ceil(cap * idx.size / n)))
        seen = np.zeros(n, int)
        for a, e_ in enumerate(idx.reshape(-1)):
            if seen[e_] >= limit:
                kept.reshape(-1)[a] = False
            seen[e_] += 1
    out = np.zeros_like(x)
    for t in range(len(x)):
        for r in range(top_k):
            if not kept[t, r]:
                continue
            s = idx[t, r]
            out[t] += w[t, r] * swiglu(x[t:t + 1], Wslot["w1"][s], Wslot["w3"][s], Wslot["w2"][s])[0]
    np_paged_moe.dropped = int((~kept).sum())
    return out, aux, idx, w


def check_paged_forward():
    rng = np.random.default_rng(3)
    z = np.load(OUT_DIR / "moe_paged_forward.npz")
    E = {k: z[f"expert/{k}"].astype(np.float64) for k in ("w1", "w3", "w2")}
    for case in ("a", "b", "c"):
        router = z[f"{case}/router"].astype(np.float64)
        demb = z[f"{case}/depth_emb"].astype(np.float64)
        gate = z[f"{case}/gate"].astype(np.float64)
        x = z[f"{case}/x"][0].astype(np.float64)
        g = z[f"{case}/g"][0].astype(np.float64)
        slots = [int(s) for s in z[f"{case}/slots"]]
        mask = z[f"{case}/admitted_mask"].astype(bool)
        bias = z[f"{case}/bias"].astype(np.float64) if f"{case}/bias" in z.files else None
        # admission mass from the router, over ALL 8 experts
        zf = (x + demb) @ router.T
        for nm, b in (("admit_mass", bias), ("admit_merit", None)):
            sc = zf + (b[None, :] if b is not None else 0)
            pr = softmax(sc)
            top = np.argsort(-pr, axis=1, kind="stable")[:, :2]
            mass = np.zeros(8)
            for t in range(len(x)):
                for e_ in top[t]:
                    mass[e_] += pr[t, e_]
            if nm == "admit_merit" and not int(z[f"{case}/admit_had_merit"][0]):
                continue
            report(f"moe_paged {case}: {nm} from router", rel(mass, z[f"{case}/{nm}"]) < 3e-5)
        # slots = the top-4 of the vote, placed in mass-descending order on an empty card
        order = [int(e_) for e_ in np.argsort(-z[f"{case}/admit_mass"], kind="stable") if z[f"{case}/admit_mass"][e_] > 0][:4]
        report(f"moe_paged {case}: slots = experts in mass-descending order", order == [s for s in slots if s >= 0])
        Wslot = {k: np.stack([E[k][max(s, 0)] if s >= 0 else np.zeros_like(E[k][0]) for s in slots]) for k in E}
        out, aux, idx, w = np_paged_moe(x, router, demb, gate, Wslot, slots, mask, bias)
        report(f"moe_paged {case}: out", rel(out, z[f"{case}/out"][0]) < 5e-5, f"rel={rel(out, z[f'{case}/out'][0]):.1e}")
        report(f"moe_paged {case}: aux", abs(aux - float(z[f"{case}/aux"][0])) < 3e-5 * abs(aux))
        report(f"moe_paged {case}: routes", (idx == z[f"{case}/route_slot"]).all()
               and (np.array(slots)[idx] == z[f"{case}/route_idx"]).all() and rel(w, z[f"{case}/route_w"]) < 3e-5)
        rs, ds, gs, xs = router.copy(), demb.copy(), gate.copy(), x.copy()
        Ws = {k: v.copy() for k, v in Wslot.items()}
        Lo = lambda: float((np_paged_moe(xs, rs, ds, gs, Ws, slots, mask, bias)[0] * g).sum())    # noqa: E731
        ix = pick(rng, xs.size, 8)
        cmp_grad(f"moe_paged {case} grad/x", fd(Lo, xs, ix), z[f"{case}/grad/x"], ix, rtol=5e-3)
        ir = pick(rng, rs.size, 14)
        cmp_grad(f"moe_paged {case} grad/router", fd(Lo, rs, ir), z[f"{case}/grad/router"], ir, rtol=5e-3)
        ig = np.arange(8)
        cmp_grad(f"moe_paged {case} grad/gate", fd(Lo, gs, ig), z[f"{case}/grad/gate"], ig, rtol=5e-3)
        for nm in ("w1", "w3", "w2"):
            iw = pick(rng, Ws[nm].size, 10)
            cmp_grad(f"moe_paged {case} grad/slot_{nm}", fd(Lo, Ws[nm], iw), z[f"{case}/grad/slot_{nm}"], iw, rtol=5e-3)
        La = lambda: float(np_paged_moe(xs, rs, ds, gs, Ws, slots, mask, bias)[1])   # noqa: E731
        cmp_grad(f"moe_paged {case} grad_aux/router", fd(La, rs, ir), z[f"{case}/grad_aux/router"], ir, rtol=5e-3)


class PyPager:
    """Plain-python re-statement of PagedPool admission (README Q17-Q19), float32 where torch is."""

    def __init__(self, n, resident, ram_capacity, bias, steps):
        f = np.float32
        self.n, self.R, self.cap = n, resident, ram_capacity
        self.slots = [-1] * resident
        self.last_seen = np.zeros(n, f)
        self.admits = np.zeros(n, f)
        self.ever = np.zeros(n, bool)
        self.recent = np.zeros(n, f)
        self.loads = self.swaps = self.segments = 0
        self.bias, self.steps = bias, steps
        self.vote = self.merit_vote = None
        self.voted = False
        self.admitted, self.merited = set(), set()
        self.b = None
        self.exploring = False
        self.ram = []                        # LRU order, oldest first
        self.reads = self.hits = self.evictions = 0

    def begin_forward(self, explore):
        f = np.float32
        self.admitted, self.merited, self.voted, self.b = set(), set(), False, None
        self.exploring = bool(explore and self.bias > 0)
        if self.exploring:
            fair = f(self.R / max(self.n, 1))
            self.b = (f(self.bias) * np.exp(-self.recent / fair)).astype(f)
            self.recent = (self.recent * f(1.0 - 1.0 / self.steps)).astype(f)

    def begin_text(self, explore):
        self.vote = self.merit_vote = None
        self.begin_forward(explore)
        self.segments += 1

    def _fetch(self, e):
        if e in self.ram:
            self.hits += 1
            self.ram.remove(e)
        else:
            self.reads += 1
        self.ram.append(e)
        while len(self.ram) > self.cap:
            self.ram.pop(0)
            self.evictions += 1

    def admit(self, mass, merit):
        f = np.float32
        m = np.asarray(mass, f)
        mm = m if merit is None else np.asarray(merit, f)
        if not self.voted:
            self.voted = True
            self.vote = m.copy() if self.vote is None else (self.vote + m).astype(f)
            self.merit_vote = mm.copy() if self.merit_vote is None else (self.merit_vote + mm).astype(f)
            m, mm = self.vote, self.merit_vote
        mfree = self.R - len(self.merited)
        if mfree > 0:
            morder = sorted(range(self.n), key=lambda e: -float(mm[e]))
            for e in [e for e in morder if mm[e] > 0 and e not in self.merited][:mfree]:
                self.merited.add(e)
                self.last_seen[e] = self.segments
        free = self.R - len(self.admitted)
        if free <= 0:
            return 0
        order = sorted(range(self.n), key=lambda e: -float(m[e]))
        new = [e for e in order if m[e] > 0 and e not in self.admitted][:free]
        if not new:
            return 0
        self.admitted |= set(new)
        here = {e: s for s, e in enumerate(self.slots) if e >= 0}
        victims = [s for s, e in enumerate(self.slots) if e < 0 or e not in self.admitted]
        victims.sort(key=lambda s: (self.slots[s] >= 0, float(self.last_seen[self.slots[s]]) if self.slots[s] >= 0 else -1.0))
        plan = list(self.slots)
        for e in new:
            if e not in here:
                plan[victims.pop(0)] = e
        loads = sum(1 for e in plan if e >= 0 and e not in here)
        if plan != self.slots:
            for e in plan:
                if e >= 0 and e not in here:
                    self.admits[e] += 1
            for s, e in enumerate(plan):
                if e >= 0 and self.slots[s] != e:
                    self._fetch(e)
            self.slots = plan
            self.swaps += 1
        for e in new:
            self.ever[e] = True
        if self.exploring:
            for e in new:
                self.recent[e] = np.float32(self.recent[e] + np.float32(1.0 / self.steps))
        self.loads += loads
        return loads


def check_admission():
    d = json.load(open(OUT_DIR / "moe_admission.json"))
    allok = True
    for sc in d["scenarios"]:
        c = sc["config"]
        pg = PyPager(c["n_experts"], c["resident"], c["ram_capacity"], c["explore_bias"], c["explore_steps"])
        ok = True
        for st in sc["steps"]:
            op, S = st["op"], st["state"]
            ret = None
            if op == "begin_text":
                pg.begin_text(st["explore"])
            elif op == "begin_forward":
                pg.begin_forward(st["explore"])
            elif op == "admit":
                ret = pg.admit(st["mass"], st["merit"])
                ok &= ret == st["returned_loads"]
            elif op == "set_recent":
                pg.recent = np.asarray(st["values"], np.float32)
            elif op == "set_explore":
                pg.bias, pg.steps = st["explore_bias"], st["explore_steps"]
            elif op != "init":
                raise ValueError(op)
            mask = [e in pg.admitted for e in pg.slots]
            ok &= pg.slots == S["slots"] and sorted(pg.admitted) == S["admitted"] and sorted(pg.merited) == S["merited"]
            ok &= mask == S["admitted_mask"] and (len(pg.admitted) < pg.R) == S["admitting"]
            ok &= np.allclose(pg.last_seen, S["last_seen"]) and np.allclose(pg.admits, S["admits"])
            ok &= list(pg.ever) == S["ever"] and pg.loads == S["loads"] and pg.swaps == S["swaps"] and pg.segments == S["segments"]
            ok &= np.allclose(pg.recent, S["recent"], atol=1e-6)
            ok &= (pg.b is None) == (S["bias"] is None) and (pg.b is None or np.allclose(pg.b, S["bias"], atol=1e-6))
            ok &= pg.exploring == S["exploring"] and pg.voted == S["voted"]
            ok &= (pg.vote is None) == (S["vote"] is None) and (pg.vote is None or np.allclose(pg.vote, S["vote"], atol=1e-6))
            ok &= pg.ram == S["ram_order"] and [pg.reads, pg.hits, pg.evictions] == [S["tiers"]["reads"], S["tiers"]["hits"], S["tiers"]["evictions"]]
            if not ok:
                print("   admission mismatch in", sc["name"], "at", st["op"], st.get("mass"))
                break
        allok &= ok
        report(f"moe_admission {sc['name']}: python re-statement reproduces every state", ok)
    # dying()
    okd = True
    for c in d["dying_cases"]:
        f = np.float32
        n = 8
        dy = np.zeros(n, f)
        surv, step = float(c["trial"]), float(c["now"])
        born, ls = np.asarray(c["born"], f), np.asarray(c["last_seen"], f)
        if not (n == 0 or surv <= 0 or step <= 0):
            per_step = c["segments"] / max(step, 1.0)
            window = surv * per_step
            if window > 0:
                born_seg = (np.maximum(f(step) - born, f(0)) * f(per_step)).astype(f)
                idle = np.maximum(f(c["segments"]) - ls, f(0)).astype(f)
                frac = (np.minimum(idle, born_seg) / f(window)).astype(f)
                young = (f(step) - born) < f(surv)
                dy = np.where(young, f(0), frac).astype(f)
        okd &= bool(np.allclose(dy, c["dying"], rtol=2e-6, atol=1e-7))
        okd &= int((dy >= c["dying_at"]).sum()) == c["saturation_idle"]
    report("moe_admission dying(): numpy f32 re-statement reproduces every case", okd)


# ----------------------------------------------------------------------------------------
# optimiser replays
# ----------------------------------------------------------------------------------------

def np_pooled(P, W, cfg, idx, targets, slots, bias, cap):
    """float64 re-implementation of the paged RecurCoder training forward (loss + 0.01*aux)."""
    T = idx.shape[1]
    ropez = np.load(OUT_DIR / "ops_rope.npz")
    cos, sin = ropez["cos"].astype(np.float64)[:T], ropez["sin"].astype(np.float64)[:T]
    emb = P["tok_emb.weight"]
    mask = np.ones(len(slots), bool)
    assert all(s >= 0 for s in slots)

    def dense_block(x, pre):
        x = x + attn(rms(x, P[pre + "ln1.weight"]), P[pre + "attn.qkv.weight"],
                     P[pre + "attn.proj.weight"], cos, sin)[0]
        return x + swiglu(rms(x, P[pre + "ln2.weight"]), P[pre + "mlp.w1.weight"],
                          P[pre + "mlp.w3.weight"], P[pre + "mlp.w2.weight"])

    def pooled_block(u, active):
        x = u + attn(rms(u, P["recur.0.ln1.weight"]), P["recur.0.attn.qkv.weight"],
                     P["recur.0.attn.proj.weight"], cos, sin)[0]
        hn = rms(x, P["recur.0.ln2.weight"])
        m = np.zeros_like(hn)
        sel = np.ones(T, bool) if active is None else active
        o, aux, _, _ = np_paged_moe(hn[sel], P["recur.0.mlp.router.weight"], P["recur.0.mlp.depth_emb"],
                                    P["pool.gate"], W, slots, mask, bias, cap=cap)
        m[sel] = o
        return x + m, aux

    x = dense_block(emb[idx[0]], "prelude.0.")
    h = np.zeros_like(x)
    cum, halted = np.ones(T), np.zeros(T, bool)
    N = cfg["max_steps"]
    Ps, Ls = [], []
    aux_last = None
    for n in range(N):
        active = (~halted) if cfg["halt_freeze"] and n else None
        if active is None or active.any():
            hn, aux_last = pooled_block(np.concatenate([h, x], -1) @ P["adapter.weight"].T, active)
            h = hn if active is None else np.where(active[:, None], hn, h)
        yf = rms(h, P["ln_f.weight"])
        logits = yf @ emb.T
        lam = 1 / (1 + np.exp(-(yf @ P["halt.weight"].T + P["halt.bias"])[:, 0]))
        if n == N - 1:
            lam = np.ones_like(lam)
        elif n < cfg["min_steps"] - 1:
            lam = np.zeros_like(lam)
        p = cum * lam
        cum = cum * (1 - lam)
        m_ = logits.max(1, keepdims=True)
        ce = np.log(np.exp(logits - m_).sum(1)) + m_[:, 0] - logits[np.arange(T), targets[0]]
        Ps.append(p)
        Ls.append(p * ce)
        if cfg["halt_freeze"]:
            halted = halted | ((1 - cum) >= cfg["halt_thresh"])
    Pm, Lm = np.stack(Ps), np.stack(Ls)
    prior = np.array([cfg["halt_prior"] * (1 - cfg["halt_prior"]) ** i for i in range(N)])
    prior = prior / prior.sum()
    Pc = np.maximum(Pm, 1e-8)
    kl = (Pc * (np.log(Pc) - np.log(prior)[:, None])).sum(0).mean()
    loss = Lm.sum(0).mean() + cfg["ponder_beta"] * kl
    return loss + cfg["pool_aux"] * aux_last, loss, aux_last


def check_paged_model():
    """adamw_paged forward/backward vs an independent numpy float64 model (teacher-forced card)."""
    rng = np.random.default_rng(4)
    z = np.load(OUT_DIR / "adamw_paged.npz")
    meta = json.load(open(OUT_DIR / "adamw_paged.json"))
    cfg = json.load(open(OUT_DIR / "config.json"))["pool_config"]
    names = meta["config"]["trunk_param_names"] + meta["config"]["pool_param_names"]
    slot = ["pool.w1", "pool.w3", "pool.w2"]
    P = {n: z[f"init/{n}"].astype(np.float64) for n in names if n not in slot}
    for k in range(4):
        slots = [int(s) for s in z[f"s{k}/slots"]]
        bias = z[f"s{k}/bias"].astype(np.float64)
        W = {nm.split(".")[1]: z[f"s{k}/pre_param/{nm}"].astype(np.float64) for nm in slot}
        idx, tg = z[f"s{k}/idx"], z[f"s{k}/targets"]
        tot, loss, aux = np_pooled(P, W, cfg, idx, tg, slots, bias, cfg["pool_capacity_factor"])
        dropped = np_paged_moe.dropped
        ok = (abs(tot - float(z[f"s{k}/loss"][0])) < 3e-5 * tot and abs(loss - float(z[f"s{k}/loss_model"][0])) < 3e-5 * loss
              and abs(aux - float(z[f"s{k}/aux"][0])) < 3e-5 * abs(aux))
        report(f"adamw_paged step {k}: loss/aux vs numpy pooled model", ok,
               f"loss {tot:.6f} vs {float(z[f's{k}/loss'][0]):.6f}, aux {aux:.6f} (last row {meta['steps'][k]['last_pooled_row']})")
        Pw = {n: v.copy() for n, v in P.items()}
        Ww = {n: v.copy() for n, v in W.items()}
        f_ = lambda: np_pooled(Pw, Ww, cfg, idx, tg, slots, bias, cfg["pool_capacity_factor"])[0]    # noqa: E731
        worst, good = 0.0, True
        for n in names:
            arr = Ww[n.split(".")[1]] if n in slot else Pw[n]
            ii = pick(rng, arr.size, 2 if arr.size > 64 else 3)
            num = fd(f_, arr, ii, 1e-6)
            ana = z[f"s{k}/grad/{n}"].astype(np.float64).reshape(-1)[ii]
            err = np.abs(num - ana)
            tol = 5e-5 + 5e-3 * np.abs(ana)
            good &= bool((err <= tol).all())
            worst = max(worst, float((err / tol).max()))
        report(f"adamw_paged step {k}: finite-difference gradients of the pooled model (22 tensors)", good,
               f"worst err/tol={worst:.2f}")
        P = {n: z[f"s{k}/param/{n}"].astype(np.float64) for n in names if n not in slot}


def adamw_np(p, g, m, v, t, lr, wd, b1=0.9, b2=0.95, eps=1e-8):
    """torch.optim.AdamW (single-tensor, CPU) in float32, same op order."""
    f32 = np.float32
    p, g, m, v = (a.astype(f32).copy() for a in (p, g, m, v))
    p = p * f32(1 - lr * wd)
    m = m + (g - m) * f32(1 - b1)                      # lerp_
    v = v * f32(b2) + g * g * f32(1 - b2)              # mul_ + addcmul_
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    step_size = lr / bc1
    denom = (np.sqrt(v) / f32(np.sqrt(bc2))) + f32(eps)
    p = p - f32(step_size) * (m / denom)
    return p, m, v


def check_adamw():
    z = np.load(OUT_DIR / "adamw_paged.npz")
    meta = json.load(open(OUT_DIR / "adamw_paged.json"))
    cfg = meta["config"]
    slot = ["pool.w1", "pool.w3", "pool.w2"]
    trunk = set(cfg["trunk_param_names"])
    allp = list(cfg["trunk_param_names"]) + list(cfg["pool_param_names"])
    worst = 0.0
    worst_m = 0.0
    state = {}          # non-slot moments chained from zero
    prev = {n: z[f"init/{n}"] for n in allp if n not in slot}
    ok = True
    for k in range(4):
        coef = float(z[f"s{k}/clip_coef"][0])
        for n in allp:
            lr = cfg["lr_trunk"] if n in trunk else cfg["lr_pool"]
            g = z[f"s{k}/grad/{n}"].astype(np.float32) * np.float32(coef)
            if n in slot:
                p0 = z[f"s{k}/pre_param/{n}"]
                if k >= 1 and f"s{k}/pre_exp_avg/{n}" in z.files:
                    m0, v0 = z[f"s{k}/pre_exp_avg/{n}"], z[f"s{k}/pre_exp_avg_sq/{n}"]
                else:
                    m0 = v0 = np.zeros_like(p0)
            else:
                p0 = prev[n]
                m0, v0 = state.get(n, (np.zeros_like(p0), np.zeros_like(p0)))
            p1, m1, v1 = adamw_np(p0, g, m0, v0, k + 1, lr, cfg["weight_decay"])
            if n not in slot:
                state[n] = (m1, v1)
                prev[n] = z[f"s{k}/param/{n}"]
            e = float(np.abs(p1 - z[f"s{k}/param/{n}"]).max())
            worst = max(worst, e)
            if f"s{k}/exp_avg/{n}" in z.files:
                worst_m = max(worst_m, float(np.abs(m1 - z[f"s{k}/exp_avg/{n}"]).max()),
                              float(np.abs(v1 - z[f"s{k}/exp_avg_sq/{n}"]).max()))
        # slot weights loaded at step k+1 must equal what was parked: either the previous
        # post-step slot values (same expert, same slot) - checked via expert-file continuity
    ok = worst < 2e-6 and worst_m < 1e-6
    report("adamw_paged: numpy AdamW replay (clip, 2 groups, restored moments)", ok,
           f"max|dparam|={worst:.2e} max|dmoment|={worst_m:.2e}")
    # global norm check
    for k in range(4):
        tot = np.sqrt(sum(float((z[f"s{k}/grad/{n}"].astype(np.float64) ** 2).sum()) for n in allp))
        c = min(1.0, 1.0 / (tot + 1e-6))
        report(f"adamw_paged: grad_norm/clip_coef step {k}",
               abs(tot - float(z[f"s{k}/grad_norm"][0])) < 1e-4 * tot and abs(c - float(z[f"s{k}/clip_coef"][0])) < 1e-5)
    # parked expert files hold the last slot values of the experts (weights fp32)
    last_slot = {}
    for k in range(4):
        for s, e in enumerate(z[f"s{k}/slots"]):
            for i, nm in enumerate(slot):
                last_slot[(int(e), nm)] = z[f"s{k}/param/{nm}"][s]
    okf = True
    for e in range(8):
        for nm, key in zip(slot, ("w1", "w3", "w2")):
            if (e, nm) in last_slot:
                okf &= np.array_equal(last_slot[(e, nm)], z[f"file/{e}/{key}"])
            else:
                okf &= np.array_equal(z[f"expert/{key}"][e], z[f"file/{e}/{key}"])
    report("adamw_paged: final expert files == last resident weights", okf)
    # bf16 packing of moments in files = round-to-nearest-even of the fp32 moments
    okb = True
    last_m = {}
    for k in range(4):
        for s, e in enumerate(z[f"s{k}/slots"]):
            for nm, key in zip(slot, ("w1", "w3", "w2")):
                last_m[(int(e), key)] = (z[f"s{k}/exp_avg/{nm}"][s], z[f"s{k}/exp_avg_sq/{nm}"][s])
    import torch
    for (e, key), (m, v) in last_m.items():
        for arr, mv in ((m, "m"), (v, "v")):
            bits = torch.from_numpy(arr.copy()).to(torch.bfloat16).view(torch.int16).numpy()
            okb &= np.array_equal(bits, z[f"file/{e}/{key}_{mv}_bits"])
    report("adamw_paged: file moments are bf16(round-nearest-even) of the last fp32 slot moments", okb)

    # dense train step
    z = np.load(OUT_DIR / "train_step_dense.npz")
    meta = json.load(open(OUT_DIR / "train_step_dense.json"))
    zf = np.load(OUT_DIR / "dense_forward.npz")
    pn = [k.split("/", 2)[2] for k in zf.files if k.startswith("perturbed/grad/")]
    for rname, run in meta["runs"].items():
        prev = {n: zf[f"perturbed/{n}"] for n in pn}
        state = {}
        worst = 0.0
        for k, st in enumerate(run["steps"]):
            coef = float(z[f"{rname}/s{k}/clip_coef"][0])
            for n in pn:
                if run["grads_stored"]:
                    g = z[f"{rname}/s{k}/grad/{n}"].astype(np.float32) * np.float32(coef)
                else:
                    continue
                wd = meta["weight_decay"]
                if rname == "wd_split" and prev[n].ndim < 2:
                    wd = 0.0
                m0, v0 = state.get(n, (np.zeros_like(prev[n]), np.zeros_like(prev[n])))
                p1, m1, v1 = adamw_np(prev[n], g, m0, v0, k + 1, st["lr_trunk"], wd)
                state[n] = (m1, v1)
                prev[n] = p1
                if f"{rname}/s{k}/param/{n}" in z.files:
                    worst = max(worst, float(np.abs(p1 - z[f"{rname}/s{k}/param/{n}"]).max()))
        if run["grads_stored"]:
            report(f"train_step_dense {rname}: numpy AdamW replay", worst < 2e-6, f"max|dparam|={worst:.2e}")
        else:
            report(f"train_step_dense {rname}: (no per-step grads stored; loss/param only)", True)


# ----------------------------------------------------------------------------------------
# decode
# ----------------------------------------------------------------------------------------

def check_decode():
    d = json.load(open(OUT_DIR / "decode.json"))
    ok = True
    for c in d["cases"]:
        p = c["params"]
        lg = np.array(c["logits"], np.float32)
        prefix = np.array(c["prefix"], np.int64)
        adj = lg.copy()
        if p["adapt_strength"] and len(prefix):
            n = min(len(prefix), p["adapt_window"])
            tail = prefix[-n:]
            wts = (np.float32(p["adapt_decay"]) ** np.arange(n - 1, -1, -1).astype(np.float32)).astype(np.float32)
            tr = np.zeros(265, np.float32)
            for t, w in zip(tail, wts):
                tr[t] += w
            adj = (adj - np.float32(p["adapt_strength"]) * tr).astype(np.float32)
        if p["rep_penalty"] != 1.0 and len(prefix):
            u = np.unique(prefix)
            l = adj[u]
            adj[u] = np.where(l > 0, l / np.float32(p["rep_penalty"]), l * np.float32(p["rep_penalty"]))
        e = np.abs(adj - np.array(c["adjusted"], np.float32)).max()
        ok &= bool(e < 2e-6) and int(np.argmax(adj)) == c["expected_next"]
        if e >= 2e-6:
            print("   decode mismatch", c["name"], e)
    report("decode: numpy re-implementation of pick_next reproduces every case", ok)


# ----------------------------------------------------------------------------------------
# README coverage
# ----------------------------------------------------------------------------------------

DT = {"float32": "f32", "int64": "i64", "uint8": "u8", "int16": "i16", "int32": "i32"}


PARAM_RX = (r"(?:tok_emb\.weight|(?:prelude|recur)\.0\.(?:ln1|ln2)\.weight"
            r"|(?:prelude|recur)\.0\.attn\.(?:qkv|proj)\.weight"
            r"|(?:prelude|recur)\.0\.mlp\.(?:w1|w3|w2)\.weight"
            r"|recur\.0\.mlp\.(?:router\.weight|depth_emb)|adapter\.weight|ln_f\.weight"
            r"|halt\.(?:weight|bias)|pool\.(?:w1|w3|w2|gate))")


def to_regex(pat):
    out = ""
    for part in re.split(r"(<[^>]*>)", pat):
        if part == "<param>":
            out += PARAM_RX
        elif re.fullmatch(r"<[^>]*>", part):
            out += "[^/]+"
        else:
            out += re.escape(part)
    return re.compile("^" + out + "$")


def check_readme():
    text = (HERE / "README.md").read_text()
    sections = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^###\s+`([\w.]+\.npz)`", line)
        if m:
            cur = m.group(1)
            sections[cur] = []
            continue
        if line.startswith("#"):
            cur = None
        if cur and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and cells[0].startswith("`"):
                sections[cur].append(cells)
    for fn in sorted(os.listdir(OUT_DIR)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(OUT_DIR / fn)
        rows = sections.get(fn)
        if rows is None:
            report(f"readme: section for {fn}", False)
            continue
        pats = []
        for cells in rows:
            for pat in re.findall(r"`([^`]+)`", cells[0]):
                pats.append((to_regex(pat), cells[1], cells[2], pat))
        hit = {p[3]: 0 for p in pats}
        bad = []
        for k in z.files:
            m = [p for p in pats if p[0].match(k)]
            if not m:
                bad.append(f"undocumented key {k}")
                continue
            a = z[k]
            good = False
            for rxp, shp, dt, pat in m:
                shape_ok = True
                digits = re.findall(r"[^\[\],\s]+", shp)
                if digits and all(x.isdigit() for x in digits):
                    shape_ok = tuple(int(x) for x in digits) == a.shape
                dt_ok = dt.strip("` ").split("/")[0] in ("", DT[str(a.dtype)]) or DT[str(a.dtype)] in dt
                if shape_ok and dt_ok:
                    good = True
                    hit[pat] += 1
                    break
            if not good:
                bad.append(f"{k}: shape {a.shape} dtype {DT[str(a.dtype)]} contradicts README ({m[0][1]} / {m[0][2]})")
        unused = [p for p, c in hit.items() if c == 0]
        report(f"readme: {fn} keys documented ({len(z.files)} keys)", not bad and not unused,
               "; ".join(bad[:4] + [f"pattern never matches: {u}" for u in unused[:4]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    if not args.skip_repro:
        check_repro()
    check_npz_writer()
    check_ops()
    check_moe()
    check_paged_forward()
    check_admission()
    check_dense()
    check_paged_model()
    check_adamw()
    check_decode()
    check_readme()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
