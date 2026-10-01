#!/usr/bin/env python3
"""Regenerate every golden fixture for the Rust port from the Python reference.

    .venv/bin/python tools/golden/dump_model.py            # all fixtures
    .venv/bin/python tools/golden/dump_model.py --only ops_rope,decode

Writes to crates/minagi-core/tests/fixtures/golden/ . Everything is seeded,
single-threaded, float32, CPU. Nothing under tools/golden/ref/ is modified; the
few places that need monkeypatching are marked `MONKEYPATCH` and listed in the
README. See tools/golden/README.md for the meaning of every key.
"""

import sys
sys.dont_write_bytecode = True      # keep __pycache__ out of tools/golden/ and ref/
import argparse
import warnings
import contextlib
import math
import os
import shutil
import sys
import tempfile
import time

from golden_common import (OUT_DIR, REF, TINY, TINY_POOL, TINY_RUNTIME, WRITTEN,
                           LineProbe, Probes, Npz, Rng, dump_json, fl, il, tiny_cfg,
                           tiny_pool_cfg)
import numpy as np
import torch
import torch.nn.functional as F

from minagi.model import (Attention, RMSNorm, SwiGLU, apply_rope, build_rope)
from minagi.recur import RecurCoder
from minagi.tokenizer import ByteTokenizer, SPECIALS, SPECIAL_ID
from minagi.decode import pick_next
from minagi.precision import unpack_bf16

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")

D = TINY["d_model"]


def say(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# 0. config.json
# ----------------------------------------------------------------------------

def fx_config(out):
    dense = dict(TINY)
    dense.update(TINY_POOL)
    dense["use_pool"] = False
    pooled = dict(TINY)
    pooled.update(TINY_POOL)
    meta = {
        "note": "Tiny config used by every model-level fixture. Keys are RecurConfig "
                "field names. `config` is the DENSE model (use_pool false; the pool_* "
                "fields are then unused); `pool_config` is the same model with the shared "
                "expert pool (use_pool true; pool_max 8 = router rows = n experts). "
                "`runtime` holds the settings that live outside RecurConfig. Individual "
                "fixtures may override fields (e.g. bptt_window) - they say so in their own "
                "json/README entry.",
        "config": dense,
        "pool_config": pooled,
        "runtime": dict(TINY_RUNTIME),
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "reference_commit": _ref_commit(),
        "derived": {"head_dim": TINY["d_model"] // TINY["n_head"],
                    "n_slots_cache": TINY["n_prelude"] + TINY["max_steps"] *
                    (TINY["n_recur"] + TINY["n_coda"])},
    }
    dump_json(out / "config.json", meta)
    return meta


def _ref_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(REF), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 1. tokenizer.json
# ----------------------------------------------------------------------------

def fx_tokenizer(out):
    tok = ByteTokenizer()
    texts = [
        "",
        "a",
        "hello world",
        "Hello, World! 123\n\ttab and newline",
        "<think>",
        "</think>",
        "<user>",
        "</user>",
        "<bot>",
        "</bot>",
        "<g>",
        "</g>",
        "<|endoftext|>",
        "<think>reason</think><bot>answer</bot>",
        "<user>hi</user><bot>yo</bot><|endoftext|>",
        "<g></g>",
        "<think><think></think></think>",
        "<g><g><g>",
        "<g>1850</g>",
        "<g 1850 1-0>",
        "<g 1850 1-0></g>",
        "x<g 1850 1-0>y<g>z",
        "<G>",
        "<THINK>",
        "< g>",
        "<g >",
        "<think",
        "think>",
        "</think",
        "<<think>>",
        "<</think>>",
        "<think>>",
        "<|endoftext|",
        "<|endoftext |>",
        "<|endoftext|><|endoftext|>",
        "<user><bot>",
        "</user></bot>",
        "<thinking>",
        "<think/>",
        "<t<think>hink>",
        "<bot><user></bot></user>",
        "<g> <g>",
        "text <think> middle </think> text",
        "héllo wörld",
        "naïve café",
        "日本語のテキスト",
        "مرحبا",
        "\U0001F600",
        "\U0001F600<think>\U0001F680</think>",
        "é<g>é",
        "á",
        "\u0000",
        "nul\u0000byte",
        "line1\r\nline2",
        "  ",
        "﻿BOM",
        "퟿￿",
        "x" * 70,
        "<think>" * 5,
        "a<g 1>b</g 2>c",
        "<|endoftext|>x<|endoftext|>",
    ]
    cases = []
    for t in texts:
        ids = tok.encode(t).ids
        rt = tok.decode(ids)
        assert rt == t, (t, rt)
        cases.append({"text": t, "ids": list(ids),
                      "utf8_hex": t.encode("utf-8").hex(),
                      "roundtrip_decode_equals_text": True})

    # decode expectations. `ids` may hold anything the decoder is given.
    S = SPECIAL_ID
    dec_in = [
        ("empty", []),
        ("ascii", [72, 105]),
        ("markers_only", [S["<think>"], S["</think>"]]),
        ("all_markers", [256 + i for i in range(9)]),
        ("lone_continuation_byte", [0x80]),
        ("lone_ff", [0xFF, 0x41]),
        ("ff_fe", [0xFF, 0xFE]),
        ("truncated_3byte_euro", [0xE2, 0x82]),
        ("truncated_4byte_emoji", [0xF0, 0x9F, 0x98]),
        ("truncated_then_ascii", [0xE2, 0x82, 0x41]),
        ("valid_euro", [0xE2, 0x82, 0xAC]),
        ("valid_emoji", [0xF0, 0x9F, 0x98, 0x80]),
        ("overlong_c0_80", [0xC0, 0x80]),
        ("overlong_e0_80_80", [0xE0, 0x80, 0x80]),
        ("surrogate_ed_a0_80", [0xED, 0xA0, 0x80]),
        ("above_max_f4_90_80_80", [0xF4, 0x90, 0x80, 0x80]),
        ("bad_continuation_e2_28_a1", [0xE2, 0x28, 0xA1]),
        ("marker_flushes_partial_utf8", [0xE2, 0x82, S["<think>"], 0xAC]),
        ("marker_between_valid", [0x68, S["<g>"], 0x69]),
        ("partial_before_endoftext", [0xF0, 0x9F, S["<|endoftext|>"]]),
        ("out_of_range_ignored", [72, 300, 105, 265, 1000, 99999]),
        ("negative_ignored", [72, -1, 105, -256]),
        ("nul_byte", [0, 65, 0]),
        ("trailing_partial", [65, 0xC3]),
        ("lead_c3_then_marker", [0xC3, S["</user>"], 0xA9]),
        ("two_byte_split_by_nothing", [0xC3, 0xA9, 0xC3, 0xA9]),
    ]
    decodes = []
    for name, ids in dec_in:
        s = tok.decode(ids)
        b = bytes(i for i in ids if 0 <= i < 256) if all(0 <= i < 256 for i in ids) else None
        decodes.append({"name": name, "ids": list(ids),
                        "input_bytes_hex": None if b is None else b.hex(),
                        "text": s, "text_utf8_hex": s.encode("utf-8").hex(),
                        "n_replacement_chars": s.count("�")})

    t2i = []
    for t in ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>", "<g>",
              "</g>", "<|endoftext|>", "a", "A", "0", " ", "\x00", "\x7f", "é",
              "ab", "", "<g 1>", "<THINK>", "think", "€"]:
        t2i.append({"token": t, "id": tok.token_to_id(t)})

    obj = {
        "note": "encode(text) -> ids over the reference ByteTokenizer. Markers are matched "
                "left-to-right by one alternation regex in SPECIALS order; everything else "
                "is UTF-8 bytes. decode drops ids outside 0..264, flushes the pending "
                "byte buffer (errors='replace') at every marker, and emits the marker text.",
        "vocab_size": tok.get_vocab_size(),
        "specials": [{"text": s, "id": SPECIAL_ID[s]} for s in SPECIALS],
        "encode_cases": cases,
        "decode_cases": decodes,
        "token_to_id_cases": t2i,
    }
    dump_json(out / "tokenizer.json", obj)
    return obj


# ----------------------------------------------------------------------------
# 2-5. ops
# ----------------------------------------------------------------------------

def fx_rmsnorm(out):
    rng = Rng(101)
    f = Npz()
    norm = RMSNorm(D)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.3 * rng.randn(D))
    f.scalar("eps", norm.eps)
    f.add("weight", norm.weight)
    cases = {
        "base": rng.randn(1, 5, D),
        "big": rng.randn(1, 5, D) * 30.0,
        "tiny": rng.randn(1, 5, D) * 1e-3,
    }
    for name, x in cases.items():
        x = x.clone().requires_grad_(True)
        norm.weight.grad = None
        g = rng.randn(1, 5, D)
        y = norm(x)
        (y * g).sum().backward()
        f.add(f"{name}/x", x)
        f.add(f"{name}/g", g)
        f.add(f"{name}/out", y)
        f.add(f"{name}/dx", x.grad)
        f.add(f"{name}/dweight", norm.weight.grad)
    # a batch>1 / odd-length case: normalisation is per position over the last dim
    x = rng.randn(2, 3, D).requires_grad_(True)
    norm.weight.grad = None
    g = rng.randn(2, 3, D)
    y = norm(x)
    (y * g).sum().backward()
    f.add("batch2/x", x)
    f.add("batch2/g", g)
    f.add("batch2/out", y)
    f.add("batch2/dx", x.grad)
    f.add("batch2/dweight", norm.weight.grad)
    return f.write(out / "ops_rmsnorm.npz")


def fx_rope(out):
    rng = Rng(102)
    f = Npz()
    cos, sin = build_rope(64, 16, 10000.0, torch.device("cpu"))
    f.add("cos", cos)
    f.add("sin", sin)
    # float64 tables, only to quantify how far a naive f64 implementation drifts
    inv = 1.0 / (10000.0 ** (np.arange(0, 16, 2, dtype=np.float64) / 16))
    fr = np.outer(np.arange(64, dtype=np.float64), inv)
    f.scalar("max_abs_diff_cos_f32_vs_f64", np.abs(cos.numpy() - np.cos(fr)).max())
    f.scalar("max_abs_diff_sin_f32_vs_f64", np.abs(sin.numpy() - np.sin(fr)).max())
    for name, off in (("pos0", 0), ("pos7", 7), ("pos59", 59)):
        T = 5
        x = rng.randn(1, 2, T, 16).requires_grad_(True)
        g = rng.randn(1, 2, T, 16)
        y = apply_rope(x, cos[off:off + T], sin[off:off + T])
        (y * g).sum().backward()
        f.scalar(f"{name}/pos_offset", off)
        f.add(f"{name}/x", x)
        f.add(f"{name}/g", g)
        f.add(f"{name}/out", y)
        f.add(f"{name}/dx", x.grad)
    return f.write(out / "ops_rope.npz")


def _attn_module(rng, std=0.25):
    cfg = tiny_cfg()
    attn = Attention(cfg)
    with torch.no_grad():
        attn.qkv.weight.copy_(rng.randn(3 * D, D, std=std))
        attn.proj.weight.copy_(rng.randn(D, D, std=std))
    return attn


def fx_attention(out):
    rng = Rng(103)
    f = Npz()
    attn = _attn_module(rng)
    cos, sin = build_rope(64, 16, 10000.0, torch.device("cpu"))
    f.add("qkv.weight", attn.qkv.weight)
    f.add("proj.weight", attn.proj.weight)

    def grads(y, g, x):
        attn.qkv.weight.grad = None
        attn.proj.weight.grad = None
        (y * g).sum().backward()
        return x.grad, attn.qkv.weight.grad.clone(), attn.proj.weight.grad.clone()

    # (a) no cache, T=8
    x = rng.randn(1, 8, D).requires_grad_(True)
    g = rng.randn(1, 8, D)
    y = attn(x, cos[:8], sin[:8])
    dx, dqkv, dproj = grads(y, g, x)
    # the reference's own intermediate steps, recomputed with its own ops so the Rust author
    # can localise a mismatch (q/k after rope, v, attention output before proj)
    with torch.no_grad():
        q, k, v = attn.qkv(x).split(D, dim=2)
        q = q.view(1, 8, 2, 16).transpose(1, 2)
        k = k.view(1, 8, 2, 16).transpose(1, 2)
        v = v.view(1, 8, 2, 16).transpose(1, 2)
        q = apply_rope(q, cos[:8], sin[:8])
        k = apply_rope(k, cos[:8], sin[:8])
        yy = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ypre = yy.transpose(1, 2).contiguous().view(1, 8, D)
        assert torch.allclose(attn.proj(ypre), y, atol=1e-6)
    f.add("a/x", x)
    f.add("a/g", g)
    f.add("a/out", y)
    f.add("a/dx", dx)
    f.add("a/dqkv.weight", dqkv)
    f.add("a/dproj.weight", dproj)
    f.add("a/q_rope", q)
    f.add("a/k_rope", k)
    f.add("a/v", v)
    f.add("a/y_pre_proj", ypre)

    # (b) cache: P=5 cached, T=3 new, offset causal mask. The cache is a constant input
    # (detached): gradients below are those of the 3 new positions only.
    def cache_case(name, P, T, seed_shift):
        xp = rng.randn(1, P, D)
        xn = rng.randn(1, T, D).requires_grad_(True)
        gn = rng.randn(1, T, D)
        c0 = {"k": None, "v": None}
        with torch.no_grad():
            attn(xp, cos[:P], sin[:P], c0)
        cache = {"k": c0["k"].clone(), "v": c0["v"].clone()}
        y = attn(xn, cos[P:P + T], sin[P:P + T], cache)
        dx, dqkv, dproj = grads(y, gn, xn)
        # sanity: identical to the last T rows of one uncached pass over all P+T positions
        with torch.no_grad():
            full = attn(torch.cat([xp, xn.detach()], 1), cos[:P + T], sin[:P + T])
        assert torch.allclose(full[:, P:], y.detach(), atol=1e-5), (full[:, P:] - y).abs().max()
        f.scalar(f"{name}/P", P)
        f.scalar(f"{name}/T", T)
        f.add(f"{name}/x_prev", xp)
        f.add(f"{name}/x_new", xn)
        f.add(f"{name}/g", gn)
        f.add(f"{name}/cache_k_in", c0["k"])
        f.add(f"{name}/cache_v_in", c0["v"])
        f.add(f"{name}/out", y)
        f.add(f"{name}/cache_k_out", cache["k"])
        f.add(f"{name}/cache_v_out", cache["v"])
        f.add(f"{name}/dx_new", dx)
        f.add(f"{name}/dqkv.weight", dqkv)
        f.add(f"{name}/dproj.weight", dproj)
        f.add(f"{name}/full_out", full)

    cache_case("b", 5, 3, 0)
    cache_case("c", 7, 1, 1)    # single-token decode step
    return f.write(out / "ops_attention.npz")


def fx_swiglu(out):
    rng = Rng(104)
    f = Npz()
    cfg = tiny_cfg()
    m = SwiGLU(cfg)
    with torch.no_grad():
        m.w1.weight.copy_(rng.randn(cfg.d_ff, D, std=0.3))
        m.w3.weight.copy_(rng.randn(cfg.d_ff, D, std=0.3))
        m.w2.weight.copy_(rng.randn(D, cfg.d_ff, std=0.3))
    for name, x0 in (("base", rng.randn(1, 5, D)), ("big", rng.randn(1, 5, D) * 6.0)):
        x = x0.clone().requires_grad_(True)
        g = rng.randn(1, 5, D)
        for p in m.parameters():
            p.grad = None
        y = m(x)
        (y * g).sum().backward()
        with torch.no_grad():
            h = F.silu(m.w1(x)) * m.w3(x)
        f.add(f"{name}/x", x)
        f.add(f"{name}/g", g)
        f.add(f"{name}/out", y)
        f.add(f"{name}/h", h)
        f.add(f"{name}/dx", x.grad)
        f.add(f"{name}/dw1", m.w1.weight.grad)
        f.add(f"{name}/dw3", m.w3.weight.grad)
        f.add(f"{name}/dw2", m.w2.weight.grad)
    f.add("w1", m.w1.weight)
    f.add("w3", m.w3.weight)
    f.add("w2", m.w2.weight)
    return f.write(out / "ops_swiglu.npz")


# ----------------------------------------------------------------------------
# 6. dense model: forward + backward with the PonderNet loss
# ----------------------------------------------------------------------------

def perturb_dense(model, rng, halt_w_std=0.3, halt_bias=-0.5):
    """Overwrite every parameter with larger random values so tests are informative.

    The reference default init leaves attention/MLP ~0.02, the halting weight x0.01 and the
    adapter at [I|I]; gradients through such a model are tiny and many bugs hide. Parameters
    are visited in named_parameters() order, so the draw sequence is fixed.
    """
    eye = torch.eye(D)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "tok_emb.weight":
                p.copy_(rng.randn(*p.shape, std=0.35))
            elif name.endswith(("ln1.weight", "ln2.weight", "ln_f.weight")):
                p.copy_(1.0 + 0.1 * rng.randn(*p.shape))
            elif name == "adapter.weight":
                p.copy_(torch.cat([eye, eye], 1) + 0.05 * rng.randn(*p.shape))
            elif name == "halt.weight":
                p.copy_(rng.randn(*p.shape, std=halt_w_std))
            elif name == "halt.bias":
                p.fill_(halt_bias)
            elif name.endswith("router.weight") or name.endswith("depth_emb"):
                continue            # pool models only: keep the reference default init
            elif name == "pool.gate":
                p.copy_(0.6 + 0.8 * rng.rand(*p.shape))
            elif name.startswith("pool.experts."):
                p.copy_(rng.randn(*p.shape, std=0.25))
            else:  # qkv/proj/w1/w3/w2
                p.copy_(rng.randn(*p.shape, std=0.15))


def params_of(model):
    """Unique parameters by name (the tied head.weight is tok_emb.weight)."""
    return {n: p for n, p in model.named_parameters()}


ROW_NAMES = ["n", "lam", "p_n", "cum", "halted", "active", "ce", "logits_n", "h", "x"]


def run_dense_variant(model, idx, targets, n_steps=None):
    """Training forward + backward on `model`, reading per-row locals through LineProbe."""
    model.train()
    for p in model.parameters():
        p.grad = None
    if n_steps is not None:
        # MONKEYPATCH: sample_depth() is random when train_steps_mean > 0; pin it.
        model.sample_depth = lambda: n_steps
    probe = LineProbe(RecurCoder.forward, "if collect:", ROW_NAMES)
    with probe:
        logits, loss = model(idx, targets)
    loss.backward()
    rows = probe.records
    cfg = model.cfg
    N = len(rows)
    P = torch.stack([r["p_n"].squeeze(-1) for r in rows], 0)
    Lt = torch.stack([r["p_n"].squeeze(-1) * r["ce"] for r in rows], 0)
    ce_term = Lt.sum(0).mean()
    prior = torch.tensor([cfg.halt_prior * (1 - cfg.halt_prior) ** n for n in range(N)])
    prior = (prior / prior.sum()).view(-1, 1, 1)
    kl = (P.clamp_min(1e-8) * (P.clamp_min(1e-8).log() - prior.log())).sum(0)
    kl_term = kl.mean()
    assert abs(float(loss) - float(ce_term + cfg.ponder_beta * kl_term)) < 1e-6
    steps = (P * torch.arange(1, N + 1).view(-1, 1, 1)).sum(0)
    assert abs(float(steps.mean()) - model.last_steps) < 1e-6
    # how close any still-running token comes to the halting threshold (discrete decision)
    margin = float("inf")
    was_halted = torch.zeros(1, rows[0]["halted"].shape[1], 1, dtype=torch.bool)
    for rr in rows[:-1]:
        d_ = ((1.0 - rr["cum"]) - cfg.halt_thresh).abs()
        margin = min(margin, float(d_[~was_halted].min()) if bool((~was_halted).any()) else margin)
        was_halted = was_halted | rr["halted"]
    return dict(logits=logits.detach(), loss=loss.detach(), rows=rows, ce_term=ce_term,
                min_halt_margin=margin,
                kl_term=kl_term, prior=prior.view(-1), steps_tok=steps, N=N,
                last_steps=model.last_steps,
                grads={n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                       for n, p in params_of(model).items()})


def halt_rows(rows):
    """Per token: index of the row at whose end it became halted, -1 if never (freeze only)."""
    T = rows[0]["halted"].shape[1]
    hr = torch.full((1, T), -1, dtype=torch.long)
    for r in rows:
        newly = r["halted"].squeeze(-1) & (hr < 0)
        hr[newly] = int(r["n"])
    return hr


def new_dense(cfg_over=None, state=None):
    cfg = tiny_cfg(**(cfg_over or {}))
    torch.manual_seed(0)
    m = RecurCoder(cfg)
    if state is not None:
        m.load_state_dict(state)
    return m


def dense_states():
    """The four weight sets shared by dense_forward / dense_infer / train_step_dense."""
    m_def = new_dense()                                   # reference default init (seed 0)
    sd_default = {k: v.clone() for k, v in m_def.state_dict().items()}

    m_pert = new_dense()
    perturb_dense(m_pert, Rng(11))
    sd_pert = {k: v.clone() for k, v in m_pert.state_dict().items()}

    m_halt = new_dense()
    perturb_dense(m_halt, Rng(11), halt_w_std=0.6, halt_bias=0.9)
    sd_halt = {k: v.clone() for k, v in m_halt.state_dict().items()}

    sd_halt_all = {k: v.clone() for k, v in sd_pert.items()}
    sd_halt_all["halt.bias"] = torch.full_like(sd_pert["halt.bias"], 6.0)
    sd_halt_all["halt.weight"] = sd_pert["halt.weight"] * 0.01
    return sd_default, sd_pert, sd_halt, sd_halt_all


def fx_dense_forward(out):
    f = Npz()
    meta = {"variants": {}}
    rng = Rng(7)
    idx = rng.randint(0, 265, 1, 16)
    idx[0, [0, 5, 9]] = torch.tensor([256, 262, 264])   # a few marker ids
    targets = rng.randint(0, 265, 1, 16)
    f.add("idx", idx)
    f.add("targets", targets)

    # ---- weight sets -------------------------------------------------------
    sd_default, sd_pert, sd_halt, sd_halt_all = dense_states()

    # name -> (state, cfg overrides, n_steps, store own weights?, description)
    variants = {
        "default_init": (sd_default, {}, None, "own",
                         "reference default init (seed 0): attn/mlp N(0,0.02), halt.weight*0.01, "
                         "halt.bias -2, adapter [I|I]; nothing halts"),
        "perturbed": (sd_pert, {}, None, "own",
                      "random larger weights (std 0.15, tok_emb 0.35), halt.weight std 0.3, "
                      "halt.bias -0.5: a few tokens halt on rows 1-2"),
        "halting": (sd_halt, {}, None, "own",
                    "perturbed + halt.weight std 0.6, halt.bias 0.9: tokens halt at different "
                    "rows, partial `active` masks"),
        "halting_all": (sd_halt_all, {}, None, "halt_only",
                        "perturbed weights but halt.bias 6, halt.weight*0.01: every token halts "
                        "after row 0, rows 1..3 have an empty active set"),
        "bptt": (sd_pert, {"bptt_window": 2}, None, "perturbed",
                 "perturbed weights, bptt_window 2: h is detached on rows 0 and 1"),
        "min_steps2": (sd_pert, {"min_steps": 2}, None, "perturbed",
                       "perturbed weights, min_steps 2: lam forced to 0 on row 0"),
        "n2": (sd_pert, {}, 2, "perturbed",
               "perturbed weights, only 2 rows run (sample_depth pinned to 2): last row forced "
               "lam=1, prior normalised over 2 rows, no detach"),
        "no_freeze": (sd_pert, {"halt_freeze": False}, None, "perturbed",
                      "perturbed weights, halt_freeze False: `active` is always None, `halted` "
                      "never updates"),
    }
    full_logits_for = {"default_init", "perturbed", "halting"}
    no_grads_for = {"min_steps2"}        # (size budget) forward arrays + loss only

    for vname, (state, over, n_steps, wmode, desc) in variants.items():
        model = new_dense(over, state)
        r = run_dense_variant(model, idx, targets, n_steps)
        rows = r["rows"]
        N = r["N"]
        # --- weights
        if wmode == "own":
            for n, p in params_of(model).items():
                f.add(f"{vname}/{n}", p)
        elif wmode == "halt_only":
            f.add(f"{vname}/halt.weight", model.halt.weight)
            f.add(f"{vname}/halt.bias", model.halt.bias)
        # --- grads
        if vname not in no_grads_for:
            for n, gtensor in r["grads"].items():
                f.add(f"{vname}/grad/{n}", gtensor)
        # --- per-row
        for rr in rows:
            n = int(rr["n"])
            f.add(f"{vname}/lam/{n}", rr["lam"].squeeze(-1))
            f.add(f"{vname}/p_n/{n}", rr["p_n"].squeeze(-1))
            f.add(f"{vname}/cum/{n}", rr["cum"].squeeze(-1))
            f.add(f"{vname}/ce/{n}", rr["ce"])
            f.add(f"{vname}/halted/{n}", rr["halted"].squeeze(-1))
            act = rr["active"]
            f.add(f"{vname}/active/{n}",
                  torch.ones(1, idx.shape[1], dtype=torch.bool) if act is None else act)
            f.add(f"{vname}/h/{n}", rr["h"])
            if vname in full_logits_for:
                f.add(f"{vname}/logits_n/{n}", rr["logits_n"])
        f.add(f"{vname}/x_prelude", rows[0]["x"])
        if vname in full_logits_for:
            f.add(f"{vname}/mixture_logits", r["logits"])
        hr = halt_rows(rows)
        f.add(f"{vname}/halt_row", hr)
        f.add(f"{vname}/steps_tok", r["steps_tok"])
        f.add(f"{vname}/prior", r["prior"])
        meta["variants"][vname] = {
            "description": desc,
            "cfg_overrides": over,
            "n_steps": N,
            "weights": ("own" if wmode == "own" else
                        "halt.weight/halt.bias stored here, all other weights from variant "
                        "'perturbed'" if wmode == "halt_only" else f"same as variant '{wmode}'"),
            "loss": float(r["loss"]),
            "ce_term": float(r["ce_term"]),
            "kl_term": float(r["kl_term"]),
            "ponder_beta": model.cfg.ponder_beta,
            "last_steps": r["last_steps"],
            "active_count_per_row": [int(rr["active"].sum()) if rr["active"] is not None
                                     else idx.shape[1] for rr in rows],
            "halted_count_after_row": [int(rr["halted"].sum()) for rr in rows],
            "halt_row_histogram": {str(k): int((hr == k).sum()) for k in range(-1, N)},
            "rows_with_empty_active_set": [int(rr["n"]) for rr in rows
                                           if rr["active"] is not None and not bool(rr["active"].any())],
            "min_abs_distance_of_1_minus_cum_to_halt_thresh": r["min_halt_margin"],
            "mixture_logits_stored": vname in full_logits_for,
            "grads_stored": vname not in no_grads_for,
        }
        if vname == "default_init":
            # (the last row forces lam=1, so everything is halted after it - ignore that row)
            assert max(meta["variants"][vname]["halted_count_after_row"][:-1]) == 0
    # sanity on the interesting variants
    mh = meta["variants"]["halting"]
    assert mh["halt_row_histogram"]["0"] > 0 and mh["halt_row_histogram"]["1"] > 0, mh
    assert any(0 < c < 16 for c in mh["active_count_per_row"]), mh
    ma = meta["variants"]["halting_all"]
    assert ma["rows_with_empty_active_set"] == [1, 2, 3], ma
    meta["idx"] = il(idx)
    meta["targets"] = il(targets)
    meta["note"] = ("loss = ce_term + ponder_beta*kl_term (+ nothing else; no pool aux in the "
                    "dense model). Per-row arrays are keyed by row index n and have shape "
                    "[1,16] (squeezed from [B,T,1]) unless noted.")
    dump_json(out / "dense_forward.json", meta)
    size = f.write(out / "dense_forward.npz")
    return size


# ----------------------------------------------------------------------------
# 6b. dense model: inference forward (targets=None), with and without KV caches
# ----------------------------------------------------------------------------

INFER_NAMES = ["n", "lam", "cum", "halted", "newly"]


def _infer_call(model, cur, caches=None, offset=0):
    probe = LineProbe(RecurCoder.forward, "if collect:", INFER_NAMES)
    with probe, torch.no_grad():
        lg, ex = model(cur, caches=caches, pos_offset=offset, collect=True)
    return lg, ex, probe.records


def fx_dense_infer(out):
    _, sd_pert, sd_halt, sd_halt_all = dense_states()
    rng = Rng(7)
    idx = rng.randint(0, 265, 1, 16)
    idx[0, [0, 5, 9]] = torch.tensor([256, 262, 264])
    f = Npz()
    f.add("idx", idx)
    meta = {"note": "Inference forward = model.eval(), targets=None, collect=True. `logits` is "
                    "the halted_logits the reference returns (each token keeps the logits of "
                    "the first row at which 1-cum >= halt_thresh); `steps` is steps_used "
                    "(1-based row count). A chunked run feeds the same tokens through "
                    "empty_caches()/pos_offset; it must reproduce the one-shot logits.",
            "scenarios": {}}
    scen = [("perturbed", sd_pert, "perturbed"), ("halting", sd_halt, "halting"),
            ("allhalt", sd_halt_all, "halting_all")]
    for sname, state, wsrc in scen:
        model = new_dense({}, state)
        model.eval()
        lg1, ex1, rows1 = _infer_call(model, idx)
        halted_rows = torch.stack([r["halted"].squeeze(-1) for r in rows1], 0)   # [R,1,T]
        T = idx.shape[1]
        hrow = torch.full((T,), -1, dtype=torch.long)
        for r in rows1:
            newly = r["halted"].reshape(-1) & (hrow < 0)
            hrow[newly] = int(r["n"])
        last = TINY["max_steps"] - 1
        if sname == "perturbed":
            plan = [(0, 8), (8, 16)]
        elif sname == "allhalt":
            plan = [(0, 10), (10, 15), (15, 16)]
        else:
            # single-token chunks at a token that halts on row 0 (rows 1..3 skipped) and at one
            # that is still running at the last row (full path with caches)
            a = next(i for i in range(1, T - 3) if hrow[i] == 0)
            b = next(i for i in range(a + 2, T - 1) if hrow[i] == last)
            plan = [(0, a), (a, a + 1), (a + 1, b), (b, b + 1), (b + 1, T)]
        f.add(f"{sname}/oneshot/logits", lg1)
        f.add(f"{sname}/oneshot/steps", ex1["steps"])
        for r in rows1:
            n = int(r["n"])
            f.add(f"{sname}/oneshot/lam/{n}", r["lam"].squeeze(-1))
            f.add(f"{sname}/oneshot/cum/{n}", r["cum"].squeeze(-1))
            f.add(f"{sname}/oneshot/halted/{n}", r["halted"].squeeze(-1))
        caches = model.empty_caches()
        off = 0
        chunk_meta = []
        cat_logits = []
        for ci, (a0, a1) in enumerate(plan):
            cur = idx[:, a0:a1]
            lg, ex, rows = _infer_call(model, cur, caches, off)
            reached = [int(r["n"]) for r in rows]
            f.add(f"{sname}/chunk{ci}/logits", lg)
            f.add(f"{sname}/chunk{ci}/steps", ex["steps"])
            f.scalar(f"{sname}/chunk{ci}/offset", off)
            for r in rows:
                n = int(r["n"])
                f.add(f"{sname}/chunk{ci}/halted/{n}", r["halted"].squeeze(-1))
            chunk_meta.append({"start": a0, "end": a1, "offset": off,
                               "rows_reaching_end_of_loop": reached,
                               "rows_skipped_all_halted": [n for n in range(TINY["max_steps"])
                                                           if n not in reached]})
            cat_logits.append(lg)
            off += a1 - a0
        cat = torch.cat(cat_logits, 1)
        diff = float((cat - lg1).abs().max())
        assert diff < 1e-4, (sname, diff)
        for slot, c in enumerate(caches):
            assert c["k"].shape[2] == T, c["k"].shape
            f.add(f"{sname}/cache/{slot}/k", c["k"])
            f.add(f"{sname}/cache/{slot}/v", c["v"])
        meta["scenarios"][sname] = {
            "weights": ("dense_forward.npz variant '%s'" % wsrc) if wsrc != "halting_all" else
                       "dense_forward.npz variant 'perturbed' with halt.weight/halt.bias from "
                       "variant 'halting_all'",
            "oneshot_steps": il(ex1["steps"].long()),
            "halt_row_per_token": il(hrow),
            "chunks": chunk_meta,
            "max_abs_diff_chunked_vs_oneshot_logits": diff,
        }
    dump_json(out / "dense_infer.json", meta)
    return f.write(out / "dense_infer.npz")


# ----------------------------------------------------------------------------
# 7. MoE: PooledMLP routing into a plain resident SharedPool
# ----------------------------------------------------------------------------

from minagi.pool import PooledMLP, SharedPool, capture_routes  # noqa: E402

E_, DFF_ = 8, 16


@contextlib.contextmanager
def stable_argsort():
    """MONKEYPATCH: make torch.argsort stable.

    PooledMLP sorts the (token, rank) assignments by expert with an *unstable* argsort; on CPU
    it is not stable once there are >= ~24 elements. Which assignments an over-capacity expert
    drops therefore depends on that unspecified order. Under this patch the order is
    ascending flat assignment index (token-major, rank-minor), a deterministic convention
    that the Rust port can adopt.
    """
    orig = torch.argsort
    torch.argsort = lambda t, *a, **k: orig(t, *a, stable=True, **k)
    try:
        yield
    finally:
        torch.argsort = orig


def moe_expert_weights(rng):
    return dict(w1=rng.randn(E_, DFF_, D, std=0.25),
                w3=rng.randn(E_, DFF_, D, std=0.25),
                w2=rng.randn(E_, D, DFF_, std=0.25))


def build_shared(wexp, router, depth_emb, gate, cap=0.0, top_k=2):
    """SharedPool + PooledMLP with the given weights. router may have > E_ rows."""
    pool = SharedPool(D, E_, DFF_, router.shape[0], depth=1)
    mlp = PooledMLP(pool, D, top_k=top_k, site=0, capacity_factor=cap, grad_checkpoint=False)
    with torch.no_grad():
        mlp.router.weight.copy_(router)
        mlp.depth_emb.copy_(depth_emb)
        pool.gate.copy_(gate)
        for i, e in enumerate(pool.experts):
            e.w1.weight.copy_(wexp["w1"][i])
            e.w3.weight.copy_(wexp["w3"][i])
            e.w2.weight.copy_(wexp["w2"][i])
    return pool, mlp


def topk_margins(probs, k=2):
    """Smallest gaps that decide discrete routing: (rank1 vs rank2, rank k vs rank k+1)."""
    srt = torch.sort(probs, dim=-1, descending=True).values
    m12 = float((srt[:, 0] - srt[:, 1]).min())
    mk = float((srt[:, k - 1] - srt[:, k]).min()) if srt.shape[1] > k else float("inf")
    return m12, min(mk, 1e9)


def _stack_grads(gl, shapes):
    return [torch.zeros(s) if g is None else g for g, s in zip(gl, shapes)]


def run_moe_case(f, prefix, mlp, pool, x0, g, active=None, expert_params=None, n_routable=E_,
                 order_free=True, expert_grads=True):
    """Forward through the reference PooledMLP and record everything under `prefix`.

    order_free=False skips `out` and the out-gradients: with a capacity drop they depend on
    torch's unstable argsort (aux, routes and counters never do).
    """
    x = x0.clone().requires_grad_(True)
    router, demb, gate = mlp.router.weight, mlp.depth_emb, pool.gate
    ew = expert_params
    inputs = [x, router, demb, gate] + ew
    with capture_routes() as picks:
        out = mlp(x, active) if active is not None else mlp(x)
    aux = mlp.aux
    go = torch.autograd.grad((out * g).sum(), inputs, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(aux, inputs, retain_graph=True, allow_unused=True)
    shapes = [t.shape for t in inputs]
    go = _stack_grads(go, shapes)
    ga = _stack_grads(ga, shapes)

    def put(kind, gl, full=True):
        f.add(f"{prefix}{kind}/x", gl[0])
        f.add(f"{prefix}{kind}/router", gl[1])
        f.add(f"{prefix}{kind}/depth_emb", gl[2])
        if not full:          # grad_aux: gate and expert grads are exactly zero, not stored
            return
        f.add(f"{prefix}{kind}/gate", gl[3])
        if not expert_grads:
            return
        for nm, k in (("w1", 0), ("w3", 1), ("w2", 2)):
            f.add(f"{prefix}{kind}/{nm}", torch.stack(gl[4 + k * E_: 4 + (k + 1) * E_]))
    f.add(f"{prefix}x", x0)
    f.add(f"{prefix}g", g)
    if order_free:
        f.add(f"{prefix}out", out)
        put("grad", go)
    f.scalar(f"{prefix}aux", aux)
    assert float(ga[3].abs().max()) == 0 and all(float(t.abs().max()) == 0 for t in ga[4:])
    put("grad_aux", ga, full=False)
    idx, w = picks[0]
    f.add(f"{prefix}route_idx", idx)
    f.add(f"{prefix}route_w", w)
    # router logits/probs the reference computed (recomputed with the same formula; the
    # top-k below must agree with route_idx)
    with torch.no_grad():
        flat = x0.reshape(-1, D)
        if active is not None:
            flat = flat[active.reshape(-1)]
        z = F.linear(flat + demb, router[:n_routable]).float()
        pr = F.softmax(z, -1)
        assert torch.equal(torch.topk(pr, 2, dim=-1).indices, idx), "route mismatch"
    f.add(f"{prefix}logits", z)
    f.add(f"{prefix}probs", pr)
    m12, mk = topk_margins(pr)
    f.scalar(f"{prefix}min_margin_top1_top2", m12)
    f.scalar(f"{prefix}min_margin_topk_boundary", mk)
    f.add(f"{prefix}use", pool.use)
    f.add(f"{prefix}age", pool.age)
    f.scalar(f"{prefix}pressure", pool.pressure)
    f.scalar(f"{prefix}want_k", pool.want_k)
    f.add(f"{prefix}last_route", mlp.last_route)
    f.scalar(f"{prefix}dropped", mlp.dropped)
    f.scalar(f"{prefix}routed", mlp.routed)
    return out, picks[0]


def expert_param_list(pool):
    return ([e.w1.weight for e in pool.experts] + [e.w3.weight for e in pool.experts]
            + [e.w2.weight for e in pool.experts])


def manual_moe(x_flat, idx, w, wexp):
    """Independent plain-loop MoE: sum_r w[t,r] * SwiGLU_{idx[t,r]}(x[t]) (for validation)."""
    out = torch.zeros_like(x_flat)
    for t in range(x_flat.shape[0]):
        for r in range(idx.shape[1]):
            e = int(idx[t, r])
            h = F.silu(wexp["w1"][e] @ x_flat[t]) * (wexp["w3"][e] @ x_flat[t])
            out[t] += w[t, r] * (wexp["w2"][e] @ h)
    return out


def fx_moe_shared(out):
    rng = Rng(201)
    f = Npz()
    wexp = moe_expert_weights(rng)
    for k, v in wexp.items():
        f.add(k, v)
    f.scalar("n_experts", E_)
    f.scalar("top_k", 2)
    f.scalar("z_weight", 1e-3)

    # ---- base: no drops (capacity_factor 0) --------------------------------------------
    router = rng.randn(E_, D, std=0.5)
    demb = rng.randn(D, std=0.3)
    gate = 0.5 + rng.rand(E_)
    x0 = rng.randn(1, 12, D)
    g = rng.randn(1, 12, D)
    pool, mlp = build_shared(wexp, router, demb, gate, cap=0.0)
    out_b, (idx_b, w_b) = run_moe_case(f, "base/", mlp, pool, x0, g, None, expert_param_list(pool))
    f.add("base/router", router)
    f.add("base/depth_emb", demb)
    f.add("base/gate", gate)
    man = manual_moe(x0.reshape(-1, D), idx_b, w_b, wexp)
    assert torch.allclose(man, out_b.reshape(-1, D), atol=1e-5), (man - out_b.reshape(-1, D)).abs().max()
    assert mlp.dropped == 0 and mlp.routed == 24

    # ---- wide router: router has 12 rows, only the first 8 are experts ---------------------
    router_w = torch.cat([router, rng.randn(4, D, std=0.5)], 0)
    pool, mlp = build_shared(wexp, router_w, demb, gate, cap=0.0)
    out_w, _ = run_moe_case(f, "wide/", mlp, pool, x0, g, None, expert_param_list(pool),
                            expert_grads=False)
    f.add("wide/router", router_w)
    assert torch.allclose(out_w, out_b, atol=1e-6)

    # ---- active mask: inactive tokens produce exact zeros, are excluded from aux -----------
    active = torch.ones(1, 12, dtype=torch.bool)
    active[0, [1, 4, 5, 9]] = False
    pool, mlp = build_shared(wexp, router, demb, gate, cap=0.0)
    out_a, _ = run_moe_case(f, "active/", mlp, pool, x0, g, active, expert_param_list(pool))
    f.add("active/active", active)
    f.add("active/router", router)
    f.add("active/depth_emb", demb)
    f.add("active/gate", gate)
    assert (out_a[0, ~active[0]] == 0).all()
    assert mlp.routed == 2 * int(active.sum())

    # ---- capacity: imbalanced router, capacity_factor 1.5 -----------------------------------
    r2 = Rng(218)
    pool = None
    m = r2.randn(D)
    m = m / m.norm()
    router_c = r2.randn(E_, D, std=0.4)
    router_c[0] += 1.0 * m
    router_c[3] += 0.6 * m
    demb_c = r2.randn(D, std=0.2)
    gate_c = 0.5 + r2.rand(E_)
    x_c = r2.randn(1, 12, D) * 0.8 + 1.2 * m
    g_c = r2.randn(1, 12, D)
    pool, mlp = build_shared(wexp, router_c, demb_c, gate_c, cap=1.5)
    out_py, (idx_c, w_c) = run_moe_case(f, "cap/", mlp, pool, x_c, g_c, None,
                                        expert_param_list(pool), order_free=False)
    f.add("cap/router", router_c)
    f.add("cap/depth_emb", demb_c)
    f.add("cap/gate", gate_c)
    # the same call under a stable sort, with gradients, as the deterministic convention
    pool, mlp = build_shared(wexp, router_c, demb_c, gate_c, cap=1.5)
    x = x_c.clone().requires_grad_(True)
    with stable_argsort():
        o_st = mlp(x)
    inputs = [x, mlp.router.weight, mlp.depth_emb, pool.gate] + expert_param_list(pool)
    gs = _stack_grads(torch.autograd.grad((o_st * g_c).sum(), inputs, allow_unused=True),
                      [t.shape for t in inputs])
    f.add("cap/out_stable", o_st)
    f.add("cap/grad_stable/x", gs[0])
    f.add("cap/grad_stable/router", gs[1])
    f.add("cap/grad_stable/depth_emb", gs[2])
    f.add("cap/grad_stable/gate", gs[3])
    for nm, k in (("w1", 0), ("w3", 1), ("w2", 2)):
        f.add(f"cap/grad_stable/{nm}", torch.stack(gs[4 + k * E_: 4 + (k + 1) * E_]))
    counts = torch.bincount(idx_c.reshape(-1), minlength=E_)
    n_assign = idx_c.numel()
    limit = max(1, int(math.ceil(1.5 * n_assign / E_)))
    kept_counts = torch.clamp(counts, max=limit)
    dropped = int((counts - kept_counts).sum())
    assert dropped == mlp.dropped and mlp.routed == n_assign, (dropped, mlp.dropped)
    assert mlp.dropped == int(sum(max(int(c) - limit, 0) for c in counts))
    # stable keep-mask over the flat assignment order (token-major, rank-minor)
    flat_e = idx_c.reshape(-1)
    seen = torch.zeros(E_, dtype=torch.long)
    keep = torch.zeros(n_assign, dtype=torch.bool)
    for a in range(n_assign):
        e = int(flat_e[a])
        if seen[e] < limit:
            keep[a] = True
        seen[e] += 1
    keepw = torch.where(keep.view_as(idx_c), w_c, torch.zeros_like(w_c))
    man_c = manual_moe(x_c.reshape(-1, D), idx_c, keepw, wexp)
    assert torch.allclose(man_c, o_st.detach().reshape(-1, D), atol=1e-5), \
        (man_c - o_st.reshape(-1, D)).abs().max()
    f.add("cap/counts_before", counts)
    f.add("cap/kept_counts", kept_counts)
    f.add("cap/kept_mask_stable", keep.view_as(idx_c))
    f.scalar("cap/limit", limit)
    f.scalar("cap/capacity_factor", 1.5)
    assert int((counts > limit).sum()) >= 2 and int((counts == 0).sum()) >= 1
    return f.write(out / "moe_shared.npz")


# ----------------------------------------------------------------------------
# 8. PagedPool admission / eviction / exploration bookkeeping
# ----------------------------------------------------------------------------

from minagi.paged import PagedPool, Tiers  # noqa: E402


def write_expert_files(path, rng, n_experts=E_, d=D, dff=DFF_):
    """Write e00000.npz.. with the reference's own writer (Tiers._to_disk): weights only,
    fp32, no moments - exactly what store.save/create.py leave behind."""
    os.makedirs(path, exist_ok=True)
    t = Tiers(str(path), d, dff)
    ws = []
    for i in range(n_experts):
        ent = {"w1": rng.randn(dff, d, std=0.25), "w3": rng.randn(dff, d, std=0.25),
               "w2": rng.randn(d, dff, std=0.25)}
        t._to_disk(i, ent)
        ws.append(ent)
    return ws


def pool_state(pool):
    n = pool._n
    return {
        "slots": [int(s) for s in pool.slots],
        "admitted": sorted(int(e) for e in pool._admitted),
        "merited": sorted(int(e) for e in pool._merited),
        "admitting": bool(pool.admitting()),
        "admitted_mask": [bool(v) for v in pool.admitted_mask().tolist()],
        "last_seen": fl(pool.last_seen),
        "admits": fl(pool.admits),
        "ever": [bool(v) for v in pool.ever.tolist()],
        "loads": int(pool.loads),
        "swaps": int(pool.swaps),
        "segments": int(pool.segments),
        "recent": fl(pool.recent),
        "bias": None if pool.selection_bias() is None else fl(pool.selection_bias()),
        "exploring": bool(pool._exploring),
        "vote": None if pool._vote is None else fl(pool._vote),
        "merit_vote": None if pool._merit_vote is None else fl(pool._merit_vote),
        "voted": bool(pool._voted),
        "ram_order": [int(i) for i in pool.tiers.ram.keys()],
        "tiers": {"reads": pool.tiers.reads, "hits": pool.tiers.hits,
                  "evictions": pool.tiers.evictions, "writebacks": pool.tiers.writebacks},
    }


def _assert_no_ties(pool, mass, merit, name):
    """Fixtures must not depend on torch.argsort's (unstable) tie order: the vectors the pool
    will actually rank (this pass's mass, or the accumulated vote on a forward's first pass)
    must have pairwise distinct positive entries."""
    from minagi.paged import _tally
    m = mass.float()
    mm = m if merit is None else merit.float()
    if not pool._voted:
        m = _tally(None if pool._vote is None else pool._vote.clone(), m)
        mm = _tally(None if pool._merit_vote is None else pool._merit_vote.clone(), mm)
    for v in (m, mm):
        # only the top `resident`+1 positions matter: order among the admitted (slot placement)
        # and the cut between admitted and not admitted
        pos = sorted((float(x) for x in v if float(x) > 0), reverse=True)[:pool.resident + 1]
        gaps = [a - b for a, b in zip(pos, pos[1:])]
        assert not gaps or min(gaps) > 1e-3, (name, "near-tie in ranking vector", pos)


def run_admission_scenario(name, desc, cfg, ops):
    tmp = tempfile.mkdtemp(prefix="golden_admit_")
    try:
        write_expert_files(os.path.join(tmp, "experts"), Rng(300), cfg["n_experts"])
        pool = PagedPool(os.path.join(tmp, "experts"), D, DFF_, cfg["n_experts"],
                         resident=cfg["resident"], ram_capacity=cfg["ram_capacity"])
        pool.explore_bias = float(cfg["explore_bias"])
        pool.explore_steps = float(cfg["explore_steps"])
        steps = [{"op": "init", "state": pool_state(pool)}]
        for op in ops:
            rec = dict(op)
            k = op["op"]
            if k == "begin_text":
                pool.begin_text(op["explore"])
            elif k == "begin_forward":
                pool.begin_forward(op["explore"])
            elif k == "admit":
                mass = torch.tensor(op["mass"], dtype=torch.float32)
                merit = None if op.get("merit") is None else torch.tensor(op["merit"],
                                                                        dtype=torch.float32)
                _assert_no_ties(pool, mass, merit, name)
                rec["returned_loads"] = int(pool.admit(mass, merit))
            elif k == "set_recent":
                with torch.no_grad():
                    pool.recent.copy_(torch.tensor(op["values"], dtype=torch.float32))
            elif k == "set_explore":
                pool.explore_bias = float(op["explore_bias"])
                pool.explore_steps = float(op["explore_steps"])
            else:
                raise ValueError(k)
            rec["state"] = pool_state(pool)
            steps.append(rec)
        return {"name": name, "description": desc, "config": cfg, "steps": steps}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _m(**kw):
    """mass vector over 8 experts from {index: value}; ties are avoided by the callers."""
    v = [0.0] * E_
    for k, x in kw.items():
        v[int(k[1:])] = float(x)
    return v


def fx_moe_admission(out):
    base = {"n_experts": E_, "resident": 4, "ram_capacity": 3, "explore_bias": 0.65,
            "explore_steps": 1000.0}
    T_, F_ = (lambda e: {"op": "begin_text", "explore": e}), (lambda e: {"op": "begin_forward", "explore": e})
    A_ = lambda mass, merit=None: {"op": "admit", "mass": mass, "merit": merit}  # noqa: E731
    scenarios = []

    scenarios.append(run_admission_scenario(
        "first_admission_and_lru_eviction",
        "Empty card filled by the first admission (slots in mass-descending order), then "
        "single-expert admissions that evict the occupant with the smallest last_seen "
        "(empty slots first; ties -> lowest slot index). No exploration.", base, [
            T_(False),
            A_([0.9, 0.5, 0.0, 1.2, 0.3, 0.0, 0.7, 0.1]),          # card empty -> [3,0,6,1]
            A_([0.0, 0.0, 2.0, 0.0, 1.5, 0.0, 0.0, 0.0]),          # 2nd pass, card full -> no-op
            T_(False), A_(_m(e0=1.0)),
            T_(False), A_(_m(e1=1.0)),
            T_(False), A_(_m(e2=1.0)),
            T_(False), A_(_m(e3=1.0)),
            T_(False), A_(_m(e4=1.0)),
            T_(False), A_(_m(e5=1.0)),
            T_(False), A_(_m(e1=1.0)),
            T_(False), A_([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.9, 0.0]),
        ]))

    scenarios.append(run_admission_scenario(
        "empty_card_one_at_a_time",
        "Fresh pool, one expert per text: shows empty slots are used first (in slot order), "
        "last_seen/ever/admits bookkeeping and that re-admitting a resident expert costs "
        "no load.", base, [
            T_(False), A_(_m(e0=1.0)),
            T_(False), A_(_m(e1=1.0)),
            T_(False), A_(_m(e2=1.0)),
            T_(False), A_(_m(e3=1.0)),
            T_(False), A_(_m(e2=1.0)),
            T_(False), A_(_m(e4=1.0)),
        ]))

    scenarios.append(run_admission_scenario(
        "later_pass_fills_room",
        "A sparse first pass leaves room; a later pass of the SAME forward admits more using "
        "only ITS OWN mass (the vote only includes each forward's first pass). A third pass "
        "on a full card does nothing. Then a new forward of the same text: its vote is "
        "first-pass-1 + first-pass-2.", base, [
            T_(False),
            A_([0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            A_([0.2, 0.0, 0.4, 0.0, 0.0, 0.9, 0.0, 0.0]),
            A_([0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0]),
            F_(False),
            A_([0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.8, 0.0]),
            A_([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]))

    scenarios.append(run_admission_scenario(
        "text_vote_accumulation",
        "Several forwards of one text: each forward is admitted by the accumulated vote of all "
        "first passes so far, not by its own mass. Forward 2 prefers e4..e7 but the vote still "
        "favours e0..e3 (no load); forward 3 tips the vote and swaps all four. A new text "
        "(begin_text) clears the vote.", base, [
            T_(False),
            A_([3.0, 2.9, 2.8, 2.7, 0.0, 0.0, 0.0, 0.0]),
            F_(False),
            A_([0.0, 0.0, 0.0, 0.0, 2.0, 1.9, 1.8, 1.7]),
            F_(False),
            A_([0.0, 0.0, 0.0, 0.0, 2.0, 1.9, 1.8, 1.7]),
            T_(False),
            A_([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3]),
        ]))

    scenarios.append(run_admission_scenario(
        "exploration_config_values",
        "explore_bias 0.65, explore_steps 1000 (config.yaml). bias = 0.65*exp(-recent/fair), "
        "fair = resident/n = 0.5, computed from `recent` BEFORE it decays by (1-1/1000). "
        "admit(mass, merit): the card follows `mass` (router + bonus), the prune clock "
        "(last_seen) follows `merit` (router alone). recent += 1/explore_steps for every expert "
        "added to the forward's admitted set (resident or not). explore=False leaves recent and bias alone; "
        "explore_bias 0 disables exploration even when explore=True.", base, [
            T_(True),
            A_([0.21, 0.11, 0.0, 0.63, 0.52, 0.41, 0.33, 0.0],
               [0.91, 0.82, 0.73, 0.14, 0.0, 0.0, 0.0, 0.0]),
            F_(True),
            A_([0.0, 0.0, 0.0, 0.32, 0.62, 0.53, 0.12, 0.24],
               [0.52, 0.0, 0.43, 0.34, 0.15, 0.0, 0.0, 0.0]),
            {"op": "set_recent", "values": [0.0, 0.3, 0.6, 0.1, 0.9, 0.0, 0.5, 0.2]},
            F_(True),
            A_([0.95, 0.0, 0.0, 0.0, 0.0, 0.63, 0.57, 0.41],
               [0.0, 0.0, 0.0, 0.93, 0.0, 0.11, 0.0, 0.22]),
            T_(False),
            A_([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9]),
            {"op": "set_explore", "explore_bias": 0.0, "explore_steps": 1000.0},
            T_(True),
            A_([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]),
        ]))

    fast = dict(base)
    fast["explore_steps"] = 8.0
    scenarios.append(run_admission_scenario(
        "exploration_fast_decay",
        "explore_bias 0.65 with explore_steps 8 so `recent` moves visibly: +0.125 per "
        "admission to the forward's admitted set, x0.875 per exploring forward.", fast, [
            T_(True), A_([0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0]),
            T_(True), A_([0.9, 0.0, 0.0, 0.6, 0.8, 0.7, 0.0, 0.0]),
            F_(True), A_([0.0, 0.0, 0.0, 0.0, 0.83, 0.71, 0.97, 0.52]),
            T_(True), A_([0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.7, 0.5]),
            T_(True), A_([0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]),
        ]))

    # --- dying() --------------------------------------------------------------------------
    dying_cases = []
    tmp = tempfile.mkdtemp(prefix="golden_dying_")
    try:
        pool = PagedPool(os.path.join(tmp, "experts"), D, DFF_, E_, resident=4, ram_capacity=3)

        def dcase(name, born, last_seen, segments, now, trial, dying_at=0.75, note=""):
            with torch.no_grad():
                pool.born.copy_(torch.tensor(born, dtype=torch.float32))
                pool.last_seen.copy_(torch.tensor(last_seen, dtype=torch.float32))
            pool.segments, pool.now, pool.trial, pool.dying_at = segments, now, trial, dying_at
            d = pool.dying()
            sat = pool.saturation()
            dying_cases.append({"name": name, "note": note, "born": born, "last_seen": last_seen,
                                "segments": segments, "now": now, "trial": trial,
                                "dying_at": dying_at, "dying": fl(d),
                                "saturation_idle": sat["idle"]})
        dcase("typical", [0, 0, 0, 0, 950, 990, 500, 0], [500, 480, 450, 400, 0, 0, 460, 300],
              500, 1000, 100, note="per_step 0.5, window 50; e4/e5 are inside their trial -> 0")
        dcase("trial_zero", [0] * 8, [1, 2, 3, 4, 5, 6, 7, 8], 500, 1000, 0,
              note="trial <= 0 -> all zeros")
        dcase("now_zero", [0] * 8, [1, 2, 3, 4, 5, 6, 7, 8], 500, 0, 100, note="now <= 0 -> zeros")
        dcase("segments_zero", [0] * 8, [0] * 8, 0, 1000, 100, note="window <= 0 -> zeros")
        dcase("last_seen_in_future_clamped", [0] * 8, [600, 501, 500, 499, 450, 400, 300, 0], 500,
              1000, 100, note="idle = clamp_min(segments - last_seen, 0)")
        dcase("born_in_future_is_young", [1200, 0, 0, 0, 0, 0, 0, 0], [0] * 8, 500, 1000, 100,
              note="now - born < trial -> young -> 0 even for negative age")
        dcase("newborn_clamped_to_own_age", [0, 0, 900, 940, 960, 880, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0], 500, 1000, 100,
              note="last_seen 0 for experts born at 900..: idle capped by age*per_step")
        dcase("odd_fractions", [10, 99, 250, 0, 700, 123, 5, 333],
              [12, 40, 90, 3, 118, 77, 1, 60], 123, 777, 333, dying_at=0.6,
              note="non-integer per_step 123/777; f32 arithmetic throughout")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    obj = {
        "note": "Scenarios drive a reference PagedPool directly. `state` after each op holds "
                "every bookkeeping field. Ties in `mass` among positive entries are avoided on "
                "purpose (torch.argsort(descending) is unstable on ties).",
        "d_model": D, "d_ff": DFF_,
        "scenarios": scenarios,
        "dying_cases": dying_cases,
    }
    return dump_json(out / "moe_admission.json", obj)


# ----------------------------------------------------------------------------
# 9. PooledMLP routing into a PagedPool (4 resident of 8)
# ----------------------------------------------------------------------------

def write_expert_weights(path, wexp):
    """Expert files for the given stacked weights, written with Tiers._to_disk."""
    os.makedirs(path, exist_ok=True)
    t = Tiers(str(path), D, DFF_)
    for i in range(wexp["w1"].shape[0]):
        t._to_disk(i, {"w1": wexp["w1"][i].clone(), "w3": wexp["w3"][i].clone(),
                       "w2": wexp["w2"][i].clone()})


@contextlib.contextmanager
def spy_admit(pool, check_name=None):
    """MONKEYPATCH (observation only): record every pool.admit(mass, merit) call and, if
    check_name is given, assert the ranking vectors have no near-ties."""
    calls = []
    orig = pool.admit

    def wrapped(mass, merit=None):
        if check_name:
            _assert_no_ties(pool, mass.detach().float().cpu(),
                            None if merit is None else merit.detach().float().cpu(), check_name)
        calls.append((mass.detach().float().cpu().clone(),
                      None if merit is None else merit.detach().float().cpu().clone()))
        return orig(mass, merit)
    pool.admit = wrapped
    try:
        yield calls
    finally:
        pool.admit = orig


def paged_case(f, prefix, wexp, router, demb, gate, x0, g, explore, recent=None, tmpdir=None,
               explore_bias=0.65, explore_steps=1000.0):
    d = os.path.join(tmpdir, prefix.strip("/").replace("/", "_"))
    write_expert_weights(os.path.join(d, "experts"), wexp)
    pool = PagedPool(os.path.join(d, "experts"), D, DFF_, E_, resident=4, ram_capacity=3)
    pool.explore_bias, pool.explore_steps = float(explore_bias), float(explore_steps)
    if recent is not None:
        with torch.no_grad():
            pool.recent.copy_(recent)
    mlp = PooledMLP(pool, D, top_k=2, site=0, capacity_factor=0.0, grad_checkpoint=False)
    with torch.no_grad():
        mlp.router.weight.copy_(router)
        mlp.depth_emb.copy_(demb)
        pool.gate.copy_(gate)
    recent_before = pool.recent.clone()
    pool.begin_text(explore)
    bias = None if pool.selection_bias() is None else pool.selection_bias().clone()
    x = x0.clone().requires_grad_(True)
    inputs = [x, mlp.router.weight, mlp.depth_emb, pool.gate, pool.w1, pool.w3, pool.w2]
    with spy_admit(pool, check_name=prefix) as calls, capture_routes() as picks:
        out = mlp(x)
    aux = mlp.aux
    go = torch.autograd.grad((out * g).sum(), inputs, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(aux, inputs, retain_graph=True, allow_unused=True)
    go = _stack_grads(go, [t.shape for t in inputs])
    ga = _stack_grads(ga, [t.shape for t in inputs])
    assert len(calls) == 1
    f.add(f"{prefix}router", router)
    f.add(f"{prefix}depth_emb", demb)
    f.add(f"{prefix}gate", gate)
    f.add(f"{prefix}x", x0)
    f.add(f"{prefix}g", g)
    f.add(f"{prefix}out", out)
    f.scalar(f"{prefix}aux", aux)
    f.add(f"{prefix}slots", torch.tensor(pool.slots))
    f.add(f"{prefix}admitted_mask", pool.admitted_mask())
    for k, nm in enumerate(["x", "router", "depth_emb", "gate", "slot_w1", "slot_w3", "slot_w2"]):
        f.add(f"{prefix}grad/{nm}", go[k])
    for k, nm in enumerate(["x", "router", "depth_emb"]):
        f.add(f"{prefix}grad_aux/{nm}", ga[k])
    for k, nm in ((5, "slot_w3"), (6, "slot_w2"), (4, "slot_w1"), (3, "gate")):
        assert float(ga[k].abs().max()) == 0     # aux does not depend on experts or gates
    idx, w = picks[0]
    slot_of = {int(e): s for s, e in enumerate(pool.slots) if e >= 0}
    f.add(f"{prefix}route_idx", idx)                       # EXPERT ids
    f.add(f"{prefix}route_slot", torch.tensor([[slot_of[int(e)] for e in row] for row in idx]))
    f.add(f"{prefix}route_w", w)
    f.add(f"{prefix}admit_mass", calls[0][0])
    f.add(f"{prefix}admit_merit", calls[0][1] if calls[0][1] is not None else calls[0][0])
    f.scalar(f"{prefix}admit_had_merit", calls[0][1] is not None)
    # slot weights after loading must equal the expert files
    for s, e in enumerate(pool.slots):
        if e >= 0:
            assert torch.equal(pool.w1.data[s], wexp["w1"][e])
            assert torch.equal(pool.w2.data[s], wexp["w2"][e])
    rows = pool.resident_rows()
    with torch.no_grad():
        z = F.linear(x0.reshape(-1, D) + demb, router[rows]).float()
        z = z.masked_fill(~pool.admitted_mask(), float("-inf"))
    pr_ = F.softmax(z, -1)
    f.add(f"{prefix}probs", pr_)
    m12, mk = topk_margins(pr_)
    f.scalar(f"{prefix}min_margin_top1_top2", m12)
    f.scalar(f"{prefix}min_margin_topk_boundary", mk)
    f.add(f"{prefix}logits_unmasked", F.linear(x0.reshape(-1, D) + demb, router[rows]))
    if bias is not None:
        f.add(f"{prefix}bias", bias)
    f.add(f"{prefix}recent_before", recent_before)
    f.add(f"{prefix}recent_after", pool.recent)
    f.add(f"{prefix}last_seen", pool.last_seen)
    f.add(f"{prefix}admits", pool.admits)
    f.add(f"{prefix}ever", pool.ever)
    f.add(f"{prefix}use", pool.use)
    f.add(f"{prefix}age", pool.age)
    f.add(f"{prefix}last_route", mlp.last_route)
    f.scalar(f"{prefix}pressure", pool.pressure)
    f.scalar(f"{prefix}want_k", pool.want_k)
    f.scalar(f"{prefix}loads", pool.loads)
    f.scalar(f"{prefix}swaps", pool.swaps)
    f.scalar(f"{prefix}segments", pool.segments)
    return pool


def fx_moe_paged_forward(out):
    rng = Rng(401)
    f = Npz()
    wexp = moe_expert_weights(rng)
    f.add("expert/w1", wexp["w1"])
    f.add("expert/w3", wexp["w3"])
    f.add("expert/w2", wexp["w2"])
    # small router so the exploration bonus (0.65 logits) can reorder the admitted set
    router = rng.randn(E_, D, std=0.15)
    demb = rng.randn(D, std=0.09)
    gate = 0.5 + rng.rand(E_)
    x0 = rng.randn(1, 12, D)
    g = rng.randn(1, 12, D)
    tmp = tempfile.mkdtemp(prefix="golden_paged_")
    try:
        pa = paged_case(f, "a/", wexp, router, demb, gate, x0, g, explore=False, tmpdir=tmp)
        assert sum(1 for s in pa.slots if s >= 0) == 4
        recent = torch.tensor([0.0, 0.3, 0.6, 0.1, 0.9, 0.0, 0.5, 0.2])
        pb = paged_case(f, "b/", wexp, router, demb, gate, x0, g, explore=True, recent=recent,
                        tmpdir=tmp)
        assert sorted(pa.slots) != sorted(pb.slots), (pa.slots, pb.slots)   # exploration changed the card
        # c: every token ranks the same two experts first, so only two get mass>0 and two
        # slots stay empty / masked (-inf) for the whole forward
        m = Rng(402).randn(D)
        m = m / m.norm()
        router_c = Rng(403).randn(E_, D, std=0.05)
        router_c[1] += 2.0 * m
        router_c[4] += 1.5 * m
        demb_c = 3.0 * m
        pc = paged_case(f, "c/", wexp, router_c, demb_c, gate, x0, g, explore=False, tmpdir=tmp)
        assert sorted(s for s in pc.slots if s >= 0) == [1, 4] and pc.slots.count(-1) == 2, pc.slots
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    f.scalar("resident", 4)
    return f.write(out / "moe_paged_forward.npz")


# ----------------------------------------------------------------------------
# 10. paged model + the real optimiser setup (AdamW, two groups, clip, attach_optimiser)
# ----------------------------------------------------------------------------

LR_POOL = 0.01          # expert-group lr (config.yaml uses 3e-4; larger here so updates show)
TRUNK_LR_MULT = 0.1
WEIGHT_DECAY = 0.1
CLIP = 1.0
SLOT_NAMES = ["pool.w1", "pool.w3", "pool.w2"]
ADAMW_DATA_SEED = 548     # batches of the adamw_paged scenario (chosen for decision margins)


def build_paged_tiny(workdir, ram_capacity=3):
    """Tiny pooled model -> weights directory (store.save, as create.py does) -> paged model
    (train.build_paged, the reference's own loader)."""
    import dataclasses as dc
    from minagi import store as ref_store
    import train as ref_train
    cfg = tiny_pool_cfg()
    torch.manual_seed(0)
    m0 = RecurCoder(cfg)
    perturb_dense(m0, Rng(21), halt_w_std=0.6, halt_bias=0.9)
    cfgd = dc.asdict(cfg)
    cfgd["pool_resident"] = TINY_RUNTIME["pool_resident"]
    ref_store.save(m0, workdir, step=-1, val=None, opt=None, cfg=cfgd, verbose=False)
    model, cfg2, pool, man = ref_train.build_paged(workdir, torch.device("cpu"),
                                                   ram_capacity=ram_capacity)
    assert pool.resident == 4 and pool.explore_bias == 0.65 and pool.explore_steps == 1000
    assert cfg2.pool_capacity_factor == 1.5 and cfg2.pool_max == 8
    return m0, model, cfg2, pool


def make_optimiser(model, pool, lr=LR_POOL):
    """Exactly the paged trainer's construction (train.py ~L745)."""
    import train as ref_train
    trunk, pool_ps = ref_train._split_trunk_pool(model)
    tg = {"params": trunk, "name": "trunk", "weight_decay": WEIGHT_DECAY,
          "lr": lr * TRUNK_LR_MULT, "base_lr": lr * TRUNK_LR_MULT}
    pg = {"params": pool_ps, "name": "pool", "weight_decay": WEIGHT_DECAY,
          "lr": lr, "base_lr": lr}
    opt = torch.optim.AdamW([tg, pg], lr=lr, betas=(0.9, 0.95), fused=False)
    pool.attach_optimiser(opt)
    return opt, trunk, pool_ps


@contextlib.contextmanager
def spy_tiers(pool, log):
    """MONKEYPATCH (observation only): log Tiers.fetch/put/_to_disk calls."""
    t = pool.tiers
    of, op, od = t.fetch, t.put, t._to_disk

    def fetch(i, count=True):
        log.append(["fetch", int(i), "ram" if i in t.ram else "disk", bool(count)])
        return of(i, count)

    def put(i, tensors, dirty=True):
        log.append(["put", int(i), any(k.endswith(("_m", "_v")) for k in tensors)])
        return op(i, tensors, dirty)

    def to_disk(i, ent):
        log.append(["write_back", int(i), any(k.endswith(("_m", "_v")) for k in ent)])
        return od(i, ent)
    t.fetch, t.put, t._to_disk = fetch, put, to_disk
    try:
        yield
    finally:
        t.fetch, t.put, t._to_disk = of, op, od


def adam_state_of(opt, p):
    st = opt.state.get(p)
    if not st or "exp_avg" not in st:
        return None
    return st


def fx_adamw_paged(out):
    f = Npz()
    meta = {"steps": []}
    work = tempfile.mkdtemp(prefix="golden_adamw_")
    try:
        m0, model, cfg, pool = build_paged_tiny(os.path.join(work, "w"))
        names = {id(p): n for n, p in model.named_parameters()}
        pnames = [n for n, _ in model.named_parameters()]
        opt, trunk, pool_ps = make_optimiser(model, pool)
        params = dict(model.named_parameters())
        model.train()

        # ---- initial state -----------------------------------------------------------------
        for n in pnames:
            if n not in SLOT_NAMES:
                f.add(f"init/{n}", params[n])
        init_files = {}
        for i in range(E_):
            z = np.load(os.path.join(work, "w", "experts", "e%05d.npz" % i))
            for k in ("w1", "w3", "w2"):
                init_files.setdefault(k, []).append(torch.from_numpy(z[k]))
        for k, v in init_files.items():
            f.add(f"expert/{k}", torch.stack(v))
        assert all(float(params[n].abs().max()) == 0 for n in SLOT_NAMES)   # empty card = zeros

        # ---- scenario ----------------------------------------------------------------------
        rng = Rng(ADAMW_DATA_SEED)
        BIG = 6.0
        favored = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [2, 5, 6, 7]]
        batches = [(rng.randint(0, 265, 1, 16), rng.randint(0, 265, 1, 16)) for _ in range(4)]
        evals = [(rng.randint(0, 265, 1, 16), rng.randint(0, 265, 1, 16)) for _ in range(2)]

        pre = {}

        def pre_hook(opt_, args, kwargs=None):
            # runs AFTER the pool's own _own_moments pre-hook (registered later)
            pre["own"] = [bool(v) for v in pool._own]
            pre["slots"] = list(pool.slots)
            for nm in SLOT_NAMES:
                st = adam_state_of(opt, params[nm])
                pre[nm] = None if st is None else (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
            pre["w"] = {nm: params[nm].detach().clone() for nm in SLOT_NAMES}
        opt.register_step_pre_hook(pre_hook)

        log = []
        # MONKEYPATCH: stable argsort, so a capacity drop keeps the first `limit` assignments
        # of each expert in (token, rank) order - see stable_argsort().
        with spy_tiers(pool, log), stable_argsort():
            for k in range(4):
                if k == 3:
                    # two held-out style forwards: no grad, no exploration, the router's own
                    # choice. The second one evicts experts the first loaded without ever
                    # stepping them (parked for free: no put).
                    meta["evals_between_step2_and_step3"] = []
                    for ei, (ex_, ey_) in enumerate(evals):
                        model.eval()
                        log_eval_start = len(log)
                        with torch.no_grad():
                            model(ex_, ey_)
                        model.train()
                        _sh, ev_dropped, ev_routed = model.pool_dropped(reset=True)
                        meta["evals_between_step2_and_step3"].append({
                            "dropped_routed": [int(ev_dropped), int(ev_routed)],
                            "slots_after": [int(s_) for s_ in pool.slots],
                            "admitted": sorted(int(e_) for e_ in pool._admitted),
                            "tier_log": [list(e_) for e_ in log[log_eval_start:]],
                            "ram_order": [int(i) for i in pool.tiers.ram.keys()],
                            "loads_total": int(pool.loads), "segments": int(pool.segments)})
                        f.add(f"eval{ei}/idx", ex_)
                        f.add(f"eval{ei}/targets", ey_)
                        f.add(f"eval{ei}/slots_after", torch.tensor(pool.slots))
                recent = torch.full((E_,), BIG)
                recent[favored[k]] = 0.0
                with torch.no_grad():
                    pool.recent.copy_(recent)
                slots_before = list(pool.slots)
                step_log_start = len(log)
                x, y = batches[k]
                opt.zero_grad(set_to_none=True)
                probe = LineProbe(RecurCoder.forward, "if collect:",
                                  ["n", "active", "cum", "halted"])
                rprobe = LineProbe(PooledMLP._route, "w = w / w.sum(-1, keepdim=True)",
                                   ["logits", "pick", "probs"])
                with spy_admit(pool, check_name=f"adamw step {k}") as calls, Probes(probe, rprobe):
                    _logits, loss_model = model(x, y, caches=None, pos_offset=0)
                # smallest gaps behind the discrete decisions of this step
                route_margin, top12_margin = float("inf"), float("inf")
                for rr in rprobe.records:
                    sc = rr["logits"] + (rr["pick"] if rr["pick"] is not None else 0.0)
                    srt = torch.sort(sc, dim=-1, descending=True).values
                    gap = srt[:, 1] - srt[:, 2]       # rank 2 vs rank 3 (top_k = 2)
                    gap = gap[torch.isfinite(gap)]
                    if gap.numel():
                        route_margin = min(route_margin, float(gap.min()))
                    g12 = (srt[:, 0] - srt[:, 1])
                    g12 = g12[torch.isfinite(g12)]
                    if g12.numel():
                        top12_margin = min(top12_margin, float(g12.min()))
                halt_margin, was_h = float("inf"), torch.zeros(1, 16, 1, dtype=torch.bool)
                for rr in probe.records[:-1]:
                    d_ = ((1.0 - rr["cum"]) - cfg.halt_thresh).abs()
                    if bool((~was_h).any()):
                        halt_margin = min(halt_margin, float(d_[~was_h].min()))
                    was_h = was_h | rr["halted"]
                act_counts = [16 if r["active"] is None else int(r["active"].sum())
                              for r in probe.records]
                aux = model.pool_aux()
                loss = loss_model + cfg.pool_aux * aux
                loss.backward()
                slots_used = list(pool.slots)
                bias = pool.selection_bias()
                grads = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                         for n, p in params.items()}
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                total_sq = sum(float((g.double() ** 2).sum()) for g in grads.values())
                coef = min(1.0, CLIP / (float(gn) + 1e-6))
                assert abs(math.sqrt(total_sq) - float(gn)) < 1e-4 * max(1.0, float(gn))
                opt.step()
                assert set(slots_used[s] for s in range(4)) == set(favored[k]), (k, slots_used)
                # ---- record
                p_ = f"s{k}/"
                f.add(p_ + "idx", x)
                f.add(p_ + "targets", y)
                f.add(p_ + "recent_set", recent)
                f.add(p_ + "slots_before", torch.tensor(slots_before))
                f.add(p_ + "slots", torch.tensor(slots_used))
                f.add(p_ + "admit_mass", calls[0][0])
                f.add(p_ + "admit_merit", calls[0][1] if calls[0][1] is not None else calls[0][0])
                f.add(p_ + "bias", bias)
                f.scalar(p_ + "loss", loss)
                f.scalar(p_ + "loss_model", loss_model)
                f.scalar(p_ + "aux", aux)
                f.scalar(p_ + "grad_norm", gn)
                f.scalar(p_ + "clip_coef", coef)
                for n in pnames:
                    f.add(p_ + f"grad/{n}", grads[n])
                    f.add(p_ + f"param/{n}", params[n])
                for n in pnames:
                    st = opt.state[params[n]]
                    if n in SLOT_NAMES or k == 3:
                        f.add(p_ + f"exp_avg/{n}", st["exp_avg"])
                        f.add(p_ + f"exp_avg_sq/{n}", st["exp_avg_sq"])
                if k >= 1:
                    for nm in SLOT_NAMES:
                        if pre[nm] is not None:
                            f.add(p_ + f"pre_exp_avg/{nm}", pre[nm][0])
                            f.add(p_ + f"pre_exp_avg_sq/{nm}", pre[nm][1])
                for nm in SLOT_NAMES:
                    f.add(p_ + f"pre_param/{nm}", pre["w"][nm])
                steps_adam = {float(opt.state[params[n]]["step"]) for n in pnames}
                assert steps_adam == {float(k + 1)}, steps_adam      # one global Adam step count
                _share, dropped, routed = model.pool_dropped(reset=True)
                meta["steps"].append({
                    "step": k,
                    "favored_by_recent": favored[k],
                    "slots_before": slots_before,
                    "slots_used": slots_used,
                    "own_flags_at_optimizer_step": pre["own"],
                    "own_flags_after_step": [bool(v) for v in pool._own],
                    "adam_step_all_params": k + 1,
                    "min_gap_topk_boundary_in_routing_scores": route_margin,
                    "min_gap_top1_top2_in_routing_scores": top12_margin,
                    "min_abs_distance_of_1_minus_cum_to_halt_thresh": halt_margin,
                    "active_count_per_row": act_counts,
                    "last_pooled_row": max(i for i, c in enumerate(act_counts) if c > 0),
                    "loss": float(loss), "loss_model": float(loss_model), "aux": float(aux),
                    "grad_norm": float(gn), "clip_coef": coef,
                    "tier_log": [list(e) for e in log[step_log_start:]],
                    "ram_order": [int(i) for i in pool.tiers.ram.keys()],
                    "dirty": sorted(int(i) for i in pool.tiers.dirty),
                    "loads_total": int(pool.loads), "swaps_total": int(pool.swaps),
                    "segments": int(pool.segments),
                    "recent_after": fl(pool.recent),
                    "last_seen": fl(pool.last_seen),
                    "admits": fl(pool.admits),
                    "pool_use": fl(pool.use),
                    "dropped_routed_this_step": [int(dropped), int(routed)],
                })
            fl_start = len(log)
            pool.flush()
            meta["flush_tier_log"] = [list(e) for e in log[fl_start:]]
        # ---- expert files after the last flush ---------------------------------------------
        for i in range(E_):
            z = np.load(os.path.join(work, "w", "experts", "e%05d.npz" % i))
            files = set(z.files)
            meta.setdefault("files", {})[str(i)] = sorted(files)
            for k in ("w1", "w3", "w2"):
                f.add(f"file/{i}/{k}", z[k])
            for k in ("w1", "w3", "w2"):
                for mv in ("m", "v"):
                    key = f"{k}_{mv}"
                    if key in files:
                        f.add(f"file/{i}/{key}_bits", z[key])                  # int16 bf16 bits
                        f.add(f"file/{i}/{key}", unpack_bf16(z[key]).numpy())  # as f32
                        assert z[key].dtype == np.int16
        meta["config"] = {"lr_pool": LR_POOL, "lr_trunk": LR_POOL * TRUNK_LR_MULT,
                          "trunk_lr_mult": TRUNK_LR_MULT, "weight_decay": WEIGHT_DECAY,
                          "betas": [0.9, 0.95], "eps": 1e-8, "clip": CLIP,
                          "ram_capacity": 3, "pool_aux_weight": cfg.pool_aux,
                          "fused": False, "foreach": False,
                          "param_groups": [{"name": g["name"], "n_tensors": len(g["params"]),
                                            "lr": g["lr"], "weight_decay": g["weight_decay"]}
                                           for g in opt.param_groups],
                          "trunk_param_names": [names[id(p)] for p in trunk],
                          "pool_param_names": [names[id(p)] for p in pool_ps]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    dump_json(out / "adamw_paged.json", meta)
    return f.write(out / "adamw_paged.npz")


# ----------------------------------------------------------------------------
# 11. decode.pick_next (greedy + adaptation trace)
# ----------------------------------------------------------------------------

def fx_decode(out):
    rng = Rng(701)
    cases = []

    def case(name, logits, prefix, note="", adapt_strength=2.5, adapt_decay=0.88,
             rep_penalty=1.0, tie_note=False):
        logits = logits.clone().float().view(1, -1)
        assert logits.shape[1] == 265
        prev = torch.tensor(prefix, dtype=torch.long).view(1, -1)
        probe = LineProbe(pick_next, "if temperature is None or temperature <= 0:",
                          ["logits", "trace"])
        with probe:
            nxt = pick_next(logits, prev, 0.0, 0, 1.0, rep_penalty, 0,
                            adapt_strength=adapt_strength, adapt_decay=adapt_decay)
        adj = probe.records[0]["logits"][0]
        assert int(adj.argmax()) == int(nxt)
        order = sorted(range(265), key=lambda i: (-float(adj[i]), i))[:8]
        cases.append({
            "name": name, "note": note,
            "params": {"temperature": 0.0, "adapt_strength": adapt_strength,
                       "adapt_decay": adapt_decay, "rep_penalty": rep_penalty,
                       "adapt_window": 64, "top_k": 0, "top_p": 1.0, "no_repeat_ngram": 0},
            "prefix_len": len(prefix),
            "prefix": list(prefix),
            "logits": fl(logits[0]),
            "expected_next": int(nxt),
            "adjusted_top8": [{"id": i, "value": float(str(adj[i].numpy()))} for i in order],
            "adjusted": fl(adj),
            "unadjusted_argmax": int(logits.argmax()),
            "argmax_is_a_tie": tie_note,
        })

    base = lambda std=1.0: rng.randn(265, std=std)  # noqa: E731
    text = lambda s: list(s.encode("utf-8"))      # noqa: E731

    case("no_prefix", base(), [], "empty prefix: adaptation skipped, plain argmax")
    lg = base(0.5)
    lg[108] = lg.max() + 3.0
    case("short_prefix_flip", lg, text("hello"),
         "'l' (108) twice in the last 4 positions: trace 0.88+0.88^2, x2.5 = 4.14 flips the pick")
    lg = base(0.5)
    lg[111] = lg.max() + 0.5
    case("last_token_suppressed", lg, text("hello"),
         "'o' is the newest token: trace weight 1.0, suppression 2.5")
    # window boundary: prefix length 70 -> only the last 64 count (positions 6..69)
    filler = [65 + (i % 20) for i in range(70)]
    for nm, pos in (("window_edge_inside", 6), ("window_edge_outside", 5)):
        lg = base(0.3).clamp(-1.0, 1.0)
        lg[200], lg[201] = 3.0000, 2.9995
        pre = list(filler)
        pre[pos] = 200
        case(nm, lg, pre,
             f"len 70, token 200 only at index {pos}. Window = last 64 = indices 6..69. weight "
             f"of index 6 is 0.88^63 = 3.2e-4 -> x2.5 = 8e-4 which flips 3.0000 vs 2.9995; "
             f"index 5 is outside the window")
    lg = base(0.5)
    lg[50] = 25.0
    case("loop_converges_to_20_8", lg, [50] * 90,
         "token 50 repeated: trace -> (1-0.88^64)/0.12 = 8.33, x2.5 = 20.83 logits")
    lg = torch.full((265,), -1.0)
    lg[10] = 4.0
    lg[20] = 4.0
    lg[30] = 3.9
    case("exact_tie_lowest_index", lg, text("abc"),
         "ids 10 and 20 tie exactly; torch argmax on CPU returns the lowest index",
         tie_note=True)
    lg = base(0.5)
    lg[262] = lg.max() + 1.0
    case("special_tokens_in_prefix", lg, [256, 72, 105, 262, 257, 262, 262, 264],
         "marker ids (>=256) take part in the trace like any other id")
    case("adapt_disabled", base(), text("hello world hello"),
         "adapt_strength 0 and rep_penalty 1: untouched logits", adapt_strength=0.0)
    for i in range(6):
        n = [3, 17, 40, 64, 65, 200][i]
        case(f"random_{n}", base(1.5), [int(v) for v in rng.randint(0, 265, n)],
             "random logits (std 1.5) and random prefix", )
    # a decayed-but-not-gone repeat: same token 10 and 30 chars back
    lg = base(0.4)
    lg[88] = lg.max() + 1.2
    pre = [88] + [65 + (i % 7) for i in range(9)] + [66] * 5
    case("old_repeat_decays", lg, pre, "token 88 14 chars back: weight 0.88^14 = 0.167")
    # extras: rep_penalty (not used by the chat path, included for completeness)
    lg = base(1.0)
    pre = [int(v) for v in rng.randint(0, 265, 30)]
    case("extra_rep_penalty_with_adapt", lg, pre, "rep_penalty 1.12 on top of adaptation",
         rep_penalty=1.12)
    case("extra_rep_penalty_only", lg, pre, "adapt_strength 0, rep_penalty 1.12 only",
         adapt_strength=0.0, rep_penalty=1.12)
    obj = {
        "note": "pick_next(logits[1,265], prev_ids[1,n], temperature=0) as called by "
                "RecurCoder.generate: adapt_strength 2.5, adapt_decay 0.88, rep_penalty 1.0, "
                "adapt_window 64. trace[v] = sum over the last min(n,64) prefix positions i "
                "with token v of decay**(age), age = 0 for the newest token (computed in "
                "float32, accumulated in position order); adjusted = logits - strength*trace; "
                "next = argmax(adjusted) (lowest index on exact ties). `adjusted` is the "
                "reference's own local after all adjustments.",
        "cases": cases,
    }
    return dump_json(out / "decode.json", obj)


# ----------------------------------------------------------------------------
# 12. one dense training step with the real AdamW setup
# ----------------------------------------------------------------------------

def fx_train_step_dense(out):
    _, sd_pert, _, _ = dense_states()
    LR_BASE, MULT, WD, WARMUP = 0.05, 0.1, 0.1, 100
    f = Npz()
    meta = {
        "weights_before": "dense_forward.npz variant 'perturbed' (same names, no prefix needed)",
        "lr_base": LR_BASE, "trunk_lr_mult": MULT, "weight_decay": WD, "warmup": WARMUP,
        "betas": [0.9, 0.95], "eps": 1e-8,
        "warmup_factor": "min(1, (train_step_index+1)/warmup)",
        "train_step_indices": [98, 100],
        "runs": {},
    }
    rng = Rng(601)
    batches = [(rng.randint(0, 265, 1, 16), rng.randint(0, 265, 1, 16)) for _ in range(2)]
    for k, (x, y) in enumerate(batches):
        f.add(f"idx/{k}", x)
        f.add(f"targets/{k}", y)
    runs = {
        # train.py ~L745 (the paged trainer): ONE trunk group, weight decay on everything
        "wd_all": dict(clip=1.0, split=False, store_all=True),
        # train.py ~L164 (batch trainer): matrices (dim>=2) decay, vectors/biases do not
        "wd_split": dict(clip=100.0, split=True, store_all=False),
    }
    for rname, rc in runs.items():
        model = new_dense({}, sd_pert)
        model.train()
        params = dict(model.named_parameters())
        if rc["split"]:
            mats = [p for p in model.parameters() if p.dim() >= 2]
            vecs = [p for p in model.parameters() if p.dim() < 2]
            groups = [{"params": mats, "name": "trunk", "weight_decay": WD},
                      {"params": vecs, "name": "trunk", "weight_decay": 0.0}]
        else:
            groups = [{"params": list(model.parameters()), "name": "trunk", "weight_decay": WD}]
        opt = torch.optim.AdamW(groups, lr=LR_BASE, betas=(0.9, 0.95), fused=False)
        recs = []
        for k, idx_step in enumerate(meta["train_step_indices"]):
            factor = min(1.0, (idx_step + 1) / WARMUP)
            lr = LR_BASE * factor
            for g in opt.param_groups:
                g["lr"] = lr * (MULT if g["name"] == "trunk" else 1.0)
            opt.zero_grad(set_to_none=True)
            x, y = batches[k]
            _lg, loss = model(x, y)
            loss.backward()
            grads = {n: p.grad.clone() for n, p in params.items()}
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), rc["clip"])
            coef = min(1.0, rc["clip"] / (float(gn) + 1e-6))
            opt.step()
            pre = f"{rname}/s{k}/"
            f.scalar(pre + "loss", loss)
            f.scalar(pre + "grad_norm", gn)
            f.scalar(pre + "clip_coef", coef)
            f.scalar(pre + "lr_trunk", lr * MULT)
            f.scalar(pre + "warmup_factor", factor)
            if rc["store_all"] or k == len(batches) - 1:
                for n, p in params.items():
                    f.add(pre + f"param/{n}", p)
            if rc["store_all"]:
                for n, gt in grads.items():
                    f.add(pre + f"grad/{n}", gt)
            recs.append({"train_step_index": idx_step, "lr_trunk": lr * MULT, "factor": factor,
                         "loss": float(loss), "grad_norm": float(gn), "clip_coef": coef})
        meta["runs"][rname] = {"clip": rc["clip"], "groups": [
            {"weight_decay": g["weight_decay"], "n_tensors": len(g["params"])}
            for g in opt.param_groups], "steps": recs,
            "params_stored": "every step" if rc["store_all"] else "last step only",
            "grads_stored": rc["store_all"]}
    dump_json(out / "train_step_dense.json", meta)
    return f.write(out / "train_step_dense.npz")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

ORDER = ["config", "tokenizer", "rmsnorm", "rope", "attention", "swiglu",
         "dense_forward", "dense_infer", "moe_shared", "moe_admission",
         "moe_paged_forward", "adamw_paged", "decode", "train_step_dense"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma separated subset of: " + ",".join(ORDER))
    ap.add_argument("--out", default=str(OUT_DIR), help="output directory")
    args = ap.parse_args()
    out = type(OUT_DIR)(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = [s for s in args.only.split(",") if s] or ORDER
    for w in want:
        if w not in ORDER:
            raise SystemExit(f"unknown fixture {w!r}; choose from {ORDER}")
    for name in ORDER:
        if name not in want:
            continue
        t0 = time.time()
        fn = globals().get("fx_" + name)
        if fn is None:
            say(f"[skip] {name}: not implemented")
            continue
        res = fn(out)
        extra = f" ({res} bytes)" if isinstance(res, int) else ""
        say(f"[ok] {name}{extra}  {time.time() - t0:.1f}s")
    if set(want) == set(ORDER):
        import hashlib
        lines = []
        for fn in sorted(set(WRITTEN)):
            lines.append(f"{hashlib.sha256((out / fn).read_bytes()).hexdigest()}  {fn}")
        (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        total = sum((out / fn).stat().st_size for fn in set(WRITTEN))
        say(f"[ok] SHA256SUMS ({len(lines)} files, {total / 1e6:.2f} MB total)")


if __name__ == "__main__":
    main()
