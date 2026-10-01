#!/usr/bin/env python3
"""
Golden fixtures for the Rust port of the growth brakes (`optim/brakes.rs`).

Drives the REAL `minagi.pool.AutoGrow` (ROOM / USED / KEPT / FITS and the two
helpers `_in_flight` and `_earning`) against a minimal duck-typed pool, and the
reader's own `gap_ok` rule extracted from `train.py` (the honest brake lives in
the training loop, not in AutoGrow). Writes

    crates/minagi-core/tests/fixtures/pure/brakes.json

Run from `tools/golden/ref` so `import minagi` works:

    cd tools/golden/ref && ../../../.venv/bin/python ../pure_brakes.py

The fake pool carries exactly what AutoGrow reads: `saturation()` (experts and
idle), `dying()`, `dying_at`, `born`, `trial`, `disk_bytes(extra)`, `use`,
`add_experts()`, `grow_events` and `n_experts()`. `dying()` and `born` are
float32 tensors, as in the real pool. Values are chosen so nothing sits within
float32 rounding of a threshold (a tie would compare differently in float32 and
float64 and say nothing about the logic).

Each case records the inputs the Rust function takes and, from the reference:
  grew       how many experts were added (0 when blocked)
  blocked    which brake refused, in the order the reference reports them:
             honest (reader), fits, room, used, earning ("kept" in the Python);
             "none" when it grew
  in_flight, idle, keep_ratio   the numbers the brakes read, unrounded
"""

import ast
import json
import os
import random
import sys
import types

import torch

sys.dont_write_bytecode = True        # never write .pyc files into the read-only reference
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
sys.path.insert(0, REF)
OUT = os.path.join(HERE, "..", "..", "crates", "minagi-core", "tests", "fixtures", "pure")

import minagi.pool as pool_mod  # noqa: E402
from minagi.pool import AutoGrow  # noqa: E402


# ---------------------------------------------- the reader's honest brake
def _extract_gap_ok():
    """Pull the real `gap_ok = (...)` statement out of train.py and compile it."""
    tree = ast.parse(open(os.path.join(REF, "train.py")).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "gap_ok"):
            expr = ast.Expression(node.value)
            ast.fix_missing_locations(expr)
            return compile(expr, "train.py:gap_ok", "eval")
    raise SystemExit("gap_ok not found in train.py")


GAP_OK = _extract_gap_ok()


def gap_ok(last_gap, max_gap):
    return eval(GAP_OK, {"last_gap": last_gap, "args": types.SimpleNamespace(max_gap=max_gap)})


# ----------------------------------------------------------- the fake pool
class FakePool:
    def __init__(self, born, dying, dying_at, trial, per_expert_bytes, has_disk=True):
        self.born = torch.tensor(born, dtype=torch.float32)
        self._dying = torch.tensor(dying, dtype=torch.float32)
        self.dying_at = dying_at
        self.trial = trial
        self.per = per_expert_bytes
        self.has_disk = has_disk
        self.use = torch.zeros(max(len(dying), 1))
        self.grow_events = []
        self._n = len(dying)

    def n_experts(self):
        return self._n

    def dying(self):
        return self._dying

    def saturation(self):
        # as PagedPool.saturation(): idle is a staleness count, dying() >= dying_at
        return {"experts": self._n, "idle": int((self._dying >= self.dying_at).sum())}

    def add_experts(self, k, seed_from=None, step=0, birth_gate=0.001):
        self._n += k
        self.born = torch.cat([self.born, torch.full((k,), float(step))])
        self._dying = torch.cat([self._dying, torch.zeros(k)])

    def __getattr__(self, name):
        # `disk_bytes` must be ABSENT (not None) on a pool that cannot say, as getattr(pool, "disk_bytes", None)
        if name == "disk_bytes" and self.has_disk:
            return lambda extra=0: (self._n + int(extra)) * self.per
        raise AttributeError(name)


def classify(rec, ok_gap):
    if not ok_gap:
        return "honest"
    if rec["grew"]:
        return "none"
    r = rec["reason"]
    if "still inside their trial" in r:
        return "fits"
    if r.startswith("no room"):
        return "room"
    if "unaddressed" in r:
        return "used"
    return "earning"


def run_case(cfg, inp):
    """Run the real reference on one decision; returns the expected outputs."""
    pool_mod._mem_frac = lambda m=inp["mem"]: m
    pool = FakePool(inp["born"], inp["dying"], inp["dying_at"], inp["trial"], inp["per"], inp["has_disk"])
    g = AutoGrow(grow_k=cfg["k"], max_experts=cfg["max_experts"], mem_frac_max=cfg["mem_frac"],
                 dying_frac_max=cfg["dying_frac_max"], max_in_flight=cfg["max_in_flight"],
                 keep_ratio_min=cfg["keep_ratio_min"], max_disk_gb=cfg["max_disk_gb"],
                 recent_mult=cfg["recent_mult"])
    step = inp["step"]
    # the numbers the brakes read, from the reference's own helpers, before anything is added
    in_flight = g._in_flight(pool, step)
    keep_ratio = g._earning(pool, step)
    idle = pool.saturation()["idle"]
    ok = gap_ok(inp["gap"], cfg["max_gap"])
    rec = g.step(2.0, pool, step) if ok else {"grew": 0}
    if ok:
        assert abs(g.keep_ratio - keep_ratio) == 0.0
        assert rec["in_flight"] == in_flight and rec["idle"] == idle
    return {"grew": rec["grew"], "blocked": classify(rec, ok), "in_flight": in_flight,
            "idle": idle, "keep_ratio": keep_ratio}


# ------------------------------------------------------------ case builders
def base_cfg():
    return {"k": 1, "max_experts": 64, "dying_frac_max": 0.35, "keep_ratio_min": 0.35, "mem_frac": 0.95,
            "max_disk_gb": 10.0, "max_in_flight": 50, "recent_mult": 4.0, "max_gap": 0.4}


def make_inp(step, born, dying, dying_at=0.65, trial=1000.0, mem=0.5, per=25_000_000, gap=None, has_disk=True):
    return {"step": step, "born": born, "dying": dying, "dying_at": dying_at, "trial": trial, "mem": mem,
            "per": per, "gap": gap, "has_disk": has_disk}


def random_case(rng):
    cfg = base_cfg()
    cfg["k"] = rng.choice([1, 1, 1, 2, 4, 8])
    cfg["max_experts"] = rng.choice([8, 16, 24, 32, 64, 1024])
    cfg["dying_frac_max"] = rng.choice([0.0, 0.1, 0.25, 0.35, 0.5, 1.0])
    cfg["keep_ratio_min"] = rng.choice([0.0, 0.2, 0.35, 0.5, 0.8, 1.0])
    cfg["mem_frac"] = rng.choice([0.5, 0.85, 0.95])
    cfg["max_disk_gb"] = rng.choice([0.0, 0.5, 1.0, 4.0, 10.0])
    cfg["max_in_flight"] = rng.choice([0, 0, 3, 6, 12, 50])
    cfg["recent_mult"] = rng.choice([4.0, 2.5, 3.0, 1.5])
    cfg["max_gap"] = rng.choice([0.0, 0.2, 0.4, 0.4])
    n = rng.randint(1, 30)
    trial = rng.choice([0.0, 0.0, 100.0, 500.0, 1000.0, 1600.0, 8000.5])
    step = rng.randint(0, 60) * 1000 + rng.choice([0, 0, 500])
    born, dying = [], []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35:
            b = 0                                                # an original
        elif kind < 0.55 and step > 0:
            b = step - rng.randint(0, 3) * 100                   # very recent
        else:
            b = rng.randint(1, max(2, step + 500))               # anywhere (sometimes the future)
        born.append(float(b))
        dying.append(round(rng.random(), 3) + 0.00037)
    dying_at = rng.choice([0.5, 0.65, 0.75, 0.8])
    mem = rng.choice([0.0, 0.3, 0.7, 0.9, 0.96, 0.99])
    per = rng.choice([1_000_000, 25_000_000, 100_000_000, 400_000_000])
    gap = rng.choice([None, None, -0.1, 0.05, 0.3, 0.45, 1.0])
    has_disk = rng.random() > 0.1
    return cfg, make_inp(step, born, dying, dying_at, trial, mem, per, gap, has_disk)


def steered_cases(rng):
    """Cases built so that each brake is the one that refuses, and some that grow."""
    out = []
    for blocker in ("none", "fits", "room_count", "room_mem", "room_disk", "used", "earning", "honest"):
        for _ in range(8):
            cfg = base_cfg()
            cfg["k"] = rng.choice([1, 2])
            trial = 1000.0
            step = 20_000
            n = rng.randint(6, 16)
            # an old settled core, a middle band that has finished its trial, and newborns
            born = [0.0] * 3
            dying = [round(rng.random() * 0.3, 3) + 0.00037 for _ in range(3)]
            for _ in range(n - 3):
                band = rng.random()
                if band < 0.55:
                    b = step - rng.randint(1000, 3900)           # judged: past trial, within 4x
                elif band < 0.8:
                    b = step - rng.randint(0, 900)               # still on trial
                else:
                    b = step - rng.randint(4000, 9000)           # too old to judge
                born.append(float(b))
                dying.append(round(rng.random() * 0.5, 3) + 0.00037)
            mem, gap = 0.5, 0.1
            if blocker == "fits":
                cfg["max_in_flight"] = 1                         # fewer than the newborns present
            if blocker == "room_count":
                cfg["max_experts"] = n + cfg["k"] - 1
            if blocker == "room_mem":
                mem = 0.97
            if blocker == "room_disk":
                cfg["max_disk_gb"] = 0.05
            if blocker == "used":
                dying = [0.9 + 0.001 * i + 0.00037 for i in range(len(dying))]
            if blocker == "earning":
                dying = [0.9 + 0.001 * i + 0.00037 if born[i] > 0 else dying[i] for i in range(len(dying))]
                cfg["dying_frac_max"] = 1.0                      # so USED does not refuse first
            if blocker == "honest":
                gap = 0.5
            out.append((cfg, make_inp(step, born, dying, 0.75, trial, mem, 25_000_000, gap)))
    return out


def edge_cases():
    cases = []
    # empty pool: nothing to judge, nothing idle
    cases.append((base_cfg(), make_inp(5000, [], [], 0.65, 1000.0)))
    # pool of originals only: "has not yet failed at anything"
    cases.append((base_cfg(), make_inp(5000, [0.0] * 6, [0.9] * 6, 0.65, 1000.0)))
    # trial 0 disables the in-flight count and defaults the judging window to 1600
    cases.append((base_cfg(), make_inp(9000, [100.0, 7000.0, 8500.0, 8900.0], [0.1, 0.8, 0.2, 0.0], 0.65, 0.0)))
    # exact ties on every ceiling at once: count 6+2 == 8, memory, disk 8 * 125e6 == 1e9, in-flight 3+2 == 5,
    # idle fraction 2/6 == limit. Everything is "<=" in the reference, so this grows.
    c = base_cfg()
    c.update(mem_frac=0.85, max_disk_gb=1.0, max_in_flight=5, dying_frac_max=2 / 6, k=2, max_experts=8)
    born = [0.0, 0.0, 0.0, 9500.0, 9600.0, 9700.0]
    dying = [0.9 + 0.00037, 0.9 + 0.00037, 0.1, 0.0, 0.0, 0.0]
    cases.append((c, make_inp(10_000, born, dying, 0.75, 1000.0, 0.85, 125_000_000, None)))
    # one byte past each tie in turn: each must refuse
    for tweak in ({"max_experts": 7}, {"max_in_flight": 4}, {"dying_frac_max": 0.33333333333333},
                  {"max_disk_gb": 0.999999999}, {"mem_frac": 0.8499999}):
        c2 = dict(c)
        c2.update(tweak)
        cases.append((c2, make_inp(10_000, born, dying, 0.75, 1000.0, 0.85, 125_000_000, None)))
    # a pool whose born array is longer than its dying array (the Python slices the mask)
    cases.append((base_cfg(), make_inp(8000, [0.0, 3000.0, 3500.0, 4000.0, 7500.0], [0.1, 0.9, 0.1], 0.65, 1000.0)))
    # NaN-free but extreme gap
    cases.append((base_cfg(), make_inp(8000, [0.0, 3000.0], [0.1, 0.2], 0.65, 1000.0, gap=5.0)))
    cases.append((base_cfg(), make_inp(8000, [0.0, 3000.0], [0.1, 0.2], 0.65, 1000.0, gap=0.4)))      # gap == max_gap
    # born in the future counts as in flight
    cases.append((base_cfg(), make_inp(1000, [0.0, 2000.0], [0.1, 0.0], 0.65, 500.0)))
    return cases


def simulate(rng, rounds, n0, cfg, every=2000, trial=8000.0, dying_at=0.75, per=25_000_000, p_dead=0.03):
    """The pool ratchets: each round it is asked, may grow, and the fake pool is updated by AutoGrow itself."""
    pool = FakePool([0.0] * n0, [0.0] * n0, dying_at, trial, per)
    g = AutoGrow(grow_k=cfg["k"], max_experts=cfg["max_experts"], mem_frac_max=cfg["mem_frac"],
                 dying_frac_max=cfg["dying_frac_max"], max_in_flight=cfg["max_in_flight"],
                 keep_ratio_min=cfg["keep_ratio_min"], max_disk_gb=cfg["max_disk_gb"],
                 recent_mult=cfg["recent_mult"])
    pool_mod._mem_frac = lambda: 0.4
    dead = set()
    rows = []
    step = 20_000
    for _ in range(rounds):
        step += every
        # experts that finished their trial are either wanted or stale; stale ones stay stale
        d = []
        for i in range(pool.n_experts()):
            age = step - float(pool.born[i])
            if age < trial:
                d.append(0.0)
                continue
            if i not in dead and rng.random() < p_dead:
                dead.add(i)
            d.append(round(0.85 + 0.1 * rng.random(), 3) + 0.00037 if i in dead else round(rng.random() * 0.5, 3) + 0.00037)
        pool._dying = torch.tensor(d, dtype=torch.float32)
        inp = make_inp(step, [float(b) for b in pool.born], d, dying_at, trial, 0.4, per, None)
        in_flight = g._in_flight(pool, step)
        keep_ratio = g._earning(pool, step)
        idle = pool.saturation()["idle"]
        rec = g.step(2.0, pool, step)
        rows.append({"cfg": cfg, "in": inp,
                     "out": {"grew": rec["grew"], "blocked": classify(rec, True), "in_flight": in_flight,
                             "idle": idle, "keep_ratio": keep_ratio}})
    return rows


def main():
    rng = random.Random(2024)
    cases = []
    for cfg, inp in edge_cases() + steered_cases(rng) + [random_case(rng) for _ in range(180)]:
        cases.append({"cfg": cfg, "in": inp, "out": run_case(cfg, inp)})

    sims = []
    sim_cfg = base_cfg()
    sim_cfg.update(k=2, max_experts=40, max_in_flight=6, keep_ratio_min=0.5, dying_frac_max=0.5)
    sims.append({"name": "ratchet_k2_cap6", "rounds": simulate(random.Random(5), 60, 8, sim_cfg, p_dead=0.04)})
    sim_cfg2 = base_cfg()
    sim_cfg2.update(k=1, max_experts=60, max_in_flight=0, keep_ratio_min=0.75, dying_frac_max=1.0)
    sims.append({"name": "ratchet_k1_uncapped", "rounds": simulate(random.Random(6), 40, 6, sim_cfg2, p_dead=0.15)})

    # summary so the fixture can be sanity-checked at a glance
    blocked = {}
    for c in cases:
        blocked[c["out"]["blocked"]] = blocked.get(c["out"]["blocked"], 0) + 1
    for s in sims:
        for r in s["rounds"]:
            blocked["sim_" + r["out"]["blocked"]] = blocked.get("sim_" + r["out"]["blocked"], 0) + 1
    print("outcomes:", blocked)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "brakes.json")
    with open(path, "w") as f:
        json.dump({"cases": cases, "simulations": sims}, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB), {len(cases)} cases + "
          f"{sum(len(s['rounds']) for s in sims)} simulated rounds")


if __name__ == "__main__":
    main()
