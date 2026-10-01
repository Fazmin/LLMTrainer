#!/usr/bin/env python3
"""
Golden fixtures for the Rust port of the context-window ramp (`train/context.rs`).

In the reference, the ramp lives inside the body of the reading command in
`train.py` (closures over loop variables), so it cannot be imported. Rather than
re-typing it, this script PARSES `train.py`, pulls out the real statements and
nested functions, and executes them:

    _in_steps          characters -> steps
    context_gain       mean loss early in the window minus mean loss deep in it
    visit_chunks       chunks per visit to a file
    the `if ctx_now < ctx_max and ctx_every_steps:` block (check + grow)
    the three statements that start and clamp `ctx_now` / `ctx_max` on resume

plus `ramp_context` from `minagi/stream.py` (the older, schedule-based ramp used
by the batch trainer), which IS importable. If train.py changes these, the
fixtures change with it.

Writes crates/minagi-core/tests/fixtures/pure/context.json.
Run from `tools/golden/ref`:

    cd tools/golden/ref && ../../../.venv/bin/python ../pure_context.py
"""

import ast
import collections
import json
import math
import os
import random
import sys
import types

import numpy as np

sys.dont_write_bytecode = True        # never write .pyc files into the read-only reference
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
sys.path.insert(0, REF)
OUT = os.path.join(HERE, "..", "..", "crates", "minagi-core", "tests", "fixtures", "pure")

TREE = ast.parse(open(os.path.join(REF, "train.py")).read())


def _compile(node, name):
    mod = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, f"train.py:{name}", "exec")


def _one(pred, what):
    found = [n for n in ast.walk(TREE) if pred(n)]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {what} in train.py, found {len(found)}")
    return found[0]


def _func(name):
    return _compile(_one(lambda n: isinstance(n, ast.FunctionDef) and n.name == name, f"def {name}"), name)


def _assign(target, value_src):
    return _compile(
        _one(lambda n: isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == target
             and ast.unparse(n.value) == value_src, f"{target} = {value_src}"), target)


CODE_IN_STEPS = _func("_in_steps")
CODE_GAIN = _func("context_gain")
CODE_VISIT = _func("visit_chunks")
CODE_RAMP = _compile(_one(lambda n: isinstance(n, ast.If)
                          and ast.unparse(n.test) == "ctx_now < ctx_max and ctx_every_steps",
                          "context ramp block"), "ramp")
CODE_START = _assign("ctx_now", "int(man.get('context_now') or args.context_start)")
CODE_MAX = _assign("ctx_max", "max(64, min(int(cfg.block), int(args.context)))")
CODE_CLAMP = _assign("ctx_now", "max(64, min(ctx_now, ctx_max))")

from minagi.stream import ramp_context  # noqa: E402


def namespace(**args):
    a = types.SimpleNamespace(**args)
    ns = {"np": np, "collections": collections, "args": a, "by_pos": collections.defaultdict(list),
          "wants_more": False}
    exec(CODE_IN_STEPS, ns)
    exec(CODE_GAIN, ns)
    exec(CODE_VISIT, ns)
    return ns


# --------------------------------------------------------------- scenarios
def run_scenario(name, args, block, saved, nsteps, amp, sample_every, seed, exact=False):
    """Drive the extracted reference over a synthetic reader and record what the ramp did."""
    rng = random.Random(seed)
    ns = namespace(**args)
    ns["cfg"] = types.SimpleNamespace(block=block)
    ns["man"] = {"context_now": saved} if saved is not None else {}
    exec(CODE_START, ns)
    exec(CODE_MAX, ns)
    exec(CODE_CLAMP, ns)
    ns["ctx_every_steps"] = ns["_in_steps"](args["context_every"])
    ns["ctx_grow_every_steps"] = ns["_in_steps"](args["context_grow_every"])
    ns["turn"] = ns["visit_chunks"](ns["ctx_now"])
    start_ctx, max_ctx = ns["ctx_now"], ns["ctx_max"]
    step = 0
    losses, changes, samples = [], [], []
    while step < nsteps:
        turn = ns["turn"]                       # a visit keeps the length it started with
        for j in range(turn):
            if step >= nsteps:
                break
            n_buckets = max(2, ns["ctx_now"] // args["chunk"])
            # loss falls with distance into the window by an amount that fades with the run
            a = amp(step)
            if exact:
                # no noise: early positions 2.5, late 2.0, the rest 2.25, so the gain is exactly 0.5
                loss = 2.5 if j < n_buckets * 0.5 else 2.0 if j >= n_buckets * 0.75 else 2.25
            else:
                loss = round(2.0 + a * math.exp(-j / max(n_buckets / 3, 1.0)) * 2 - a + rng.gauss(0, 0.02), 4)
            losses.append(loss)
            ns["by_pos"][j].append(loss)
            step += 1
            ns["step"] = step
            before = ns["ctx_now"]
            exec(CODE_RAMP, ns)
            if ns["ctx_now"] != before:
                changes.append([step, ns["ctx_now"]])
            if step % sample_every == 0:
                g = ns["context_gain"]()
                samples.append({"step": step, "gain": g, "wants_more": ns["wants_more"], "ctx": ns["ctx_now"]})
    return {"name": name, "args": args, "block": block, "saved": saved, "nsteps": nsteps, "losses": losses,
            "start_ctx": start_ctx, "max_ctx": max_ctx,
            "every_steps": ns["ctx_every_steps"], "grow_every_steps": ns["ctx_grow_every_steps"],
            "changes": changes, "samples": samples, "final_ctx": ns["ctx_now"], "final_turn": ns["turn"]}


def base_args(**kw):
    a = dict(chunk=64, passage=1024, context=640, context_start=256, context_step=16, context_gain_min=0.015,
             context_grow_every=640, context_every=1280)
    a.update(kw)
    return a


def main():
    scenarios = []
    # ramps up while there is gain deep into the window, then stalls once the gain fades
    scenarios.append(run_scenario(
        "ramps_then_stalls", base_args(context_grow_every=3200), 1024, None, 2400,
        lambda s: 0.12 if s < 500 else 0.0, 70, 1))
    # gain never fades: the window walks to the ceiling and stays there
    scenarios.append(run_scenario("reaches_ceiling", base_args(context_step=32), 1024, None, 1500,
                                  lambda s: 0.2, 70, 2))
    # gain starts low, then appears (the check re-gathers evidence every `every` steps)
    scenarios.append(run_scenario("gain_appears_late", base_args(context_step=8), 1024, None, 2200,
                                  lambda s: 0.0 if s < 700 else 0.1, 70, 3))
    # resumed from a checkpoint whose window had grown
    scenarios.append(run_scenario("resumed_from_saved", base_args(), 1024, 448, 1200, lambda s: 0.1, 70, 4))
    # resumed from a window above today's ceiling: held at the ceiling, never walks back up
    scenarios.append(run_scenario("saved_above_ceiling", base_args(context=512), 4096, 4000, 800,
                                  lambda s: 0.3, 70, 5))
    # block (rotary tables) below the configured ceiling wins
    scenarios.append(run_scenario("block_below_ceiling", base_args(context=2048), 512, None, 800,
                                  lambda s: 0.3, 70, 6))
    # a gain exactly equal to the minimum does not grow the window (strictly greater is required) ...
    scenarios.append(run_scenario("gain_equals_minimum", base_args(context_gain_min=0.5), 1024, None, 600,
                                  lambda s: 0, 70, 9, exact=True))
    # ... and one a hair under does
    scenarios.append(run_scenario("gain_just_above_minimum", base_args(context_gain_min=0.4999), 1024, None, 600,
                                  lambda s: 0, 70, 10, exact=True))
    # the shipped defaults: tiny steps, long cadences (a character per 195 steps)
    scenarios.append(run_scenario(
        "defaults_like",
        dict(chunk=512, passage=16384, context=2048, context_start=1024, context_step=1,
             context_gain_min=0.015, context_grow_every=100_000, context_every=65_536),
        2048, None, 2600, lambda s: 0.15, 200, 7))
    # a context_step that does not divide the room: the last step is clipped to the ceiling
    scenarios.append(run_scenario("clipped_last_step", base_args(context_step=100), 1024, 560, 800,
                                  lambda s: 0.3, 70, 8))

    # resume/clamp table, straight from the three extracted statements
    starts = []
    for block in (64, 512, 1024, 4096, 32768):
        for context in (32, 256, 1024, 2048, 24576):
            for ctx_start in (16, 1024):
                for saved in (None, 0, 100, 1500, 5000, 100000):
                    ns = {"cfg": types.SimpleNamespace(block=block),
                          "args": types.SimpleNamespace(context=context, context_start=ctx_start),
                          "man": {"context_now": saved} if saved is not None else {}}
                    exec(CODE_START, ns)
                    exec(CODE_MAX, ns)
                    exec(CODE_CLAMP, ns)
                    starts.append({"block": block, "context": context, "start": ctx_start, "saved": saved,
                                   "ctx_now": ns["ctx_now"], "ctx_max": ns["ctx_max"],
                                   "was_above": bool(ns["ctx_now"] < int(ns["man"].get("context_now") or 0))})

    # the gain rule, directly: random samples at random positions, several window sizes and chunk sizes
    rng = random.Random(99)
    gains = []
    for _ in range(80):
        chunk = rng.choice([1, 64, 512, 1000])
        ctx_now = rng.choice([chunk, chunk * 2, chunk * 3, chunk * 4 + 5, chunk * 8, chunk * 17 + 3, 1])
        if ctx_now < 64:                           # the ramp never runs below its floor of 64
            ctx_now += 64
        n_buckets_hi = max(2, ctx_now // chunk)
        count = rng.choice([0, 3, 10, 16, 17, 40, 120])
        samples = [[rng.randint(0, n_buckets_hi + 1), round(rng.uniform(0.5, 3.0), 4)] for _ in range(count)]
        ns = namespace(chunk=chunk)
        ns["ctx_now"] = ctx_now
        for p, l in samples:
            ns["by_pos"][p].append(l)
        gains.append({"chunk": chunk, "ctx_now": ctx_now, "samples": samples, "gain": ns["context_gain"]()})
    # the threshold on the sample count, exactly: 7 and 8 per side
    for per_side in (7, 8):
        ns = namespace(chunk=64)
        ns["ctx_now"] = 640                       # 10 buckets: early < 5, late >= 7.5
        samples = [[0, 3.0]] * per_side + [[9, 2.0]] * per_side
        for p, l in samples:
            ns["by_pos"][p].append(l)
        gains.append({"chunk": 64, "ctx_now": 640, "samples": samples, "gain": ns["context_gain"]()})
    # positions on the boundaries: 4 is early and 5 is not (10 buckets); 7 is not late and 8 is
    ns = namespace(chunk=64)
    ns["ctx_now"] = 640
    samples = [[4, 3.0]] * 8 + [[5, 9.0]] * 8 + [[7, 9.0]] * 8 + [[8, 1.0]] * 8
    for p, l in samples:
        ns["by_pos"][p].append(l)
    gains.append({"chunk": 64, "ctx_now": 640, "samples": samples, "gain": ns["context_gain"]()})

    # characters -> steps, including exact halves (Python rounds them to even)
    in_steps = []
    for chunk in (0, 1, 7, 64, 512, 4096):
        for chars in (0, 1, 100, 256, 320, 1000, 65_536, 100_000, 2_000_000, 3 * 512 // 2, 5 * 512 // 2):
            ns = namespace(chunk=chunk)
            in_steps.append({"chars": chars, "chunk": chunk, "steps": ns["_in_steps"](chars)})
    for chars, chunk in ((1, 2), (3, 2), (5, 2), (7, 2), (2, 4), (6, 4), (10, 4)):   # x.5 ratios
        ns = namespace(chunk=chunk)
        in_steps.append({"chars": chars, "chunk": chunk, "steps": ns["_in_steps"](chars)})

    visits = []
    for ctx in (1, 64, 100, 512, 513, 1024, 1025, 4096, 24576):
        for chunk in (1, 64, 512):
            for passage in (1, 64, 512, 16384, 65536):
                ns = namespace(chunk=chunk, passage=passage)
                visits.append({"ctx": ctx, "chunk": chunk, "passage": passage, "turn": ns["visit_chunks"](ctx)})

    ramps = []
    for start, end in ((256, 4096), (1024, 2048), (512, 512), (2048, 1024), (64, 100000)):
        for total in (1, 100, 1000, 100000):
            for step in sorted({0, 1, total // 4, total // 3, int(total * 0.35), total // 2, total - 1, total,
                                total * 2}):
                for warm, gran in ((0.35, 256), (0.35, 1), (1.0, 64), (0.0, 256)):
                    ramps.append({"step": step, "total": total, "start": start, "end": end, "warm": warm,
                                  "granularity": gran,
                                  "ctx": ramp_context(step, total, start, end, warm=warm, granularity=gran)})

    doc = {"scenarios": scenarios, "starts": starts, "gains": gains, "in_steps": in_steps, "visits": visits,
           "ramps": ramps}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "context.json")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)")
    for s in scenarios:
        print(f"  {s['name']:22s} ctx {s['start_ctx']} -> {s['final_ctx']} (max {s['max_ctx']}), "
              f"{len(s['changes'])} growths, {len(s['samples'])} gain samples, "
              f"{sum(1 for x in s['samples'] if x['gain'] is None)} undefined")


if __name__ == "__main__":
    main()
