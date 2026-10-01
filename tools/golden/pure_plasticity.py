#!/usr/bin/env python3
"""
Golden fixtures for the Rust port of the learning-rate controller
(`optim/plasticity.rs`) and the gradient-signal meter (`GradSnr`).

Drives the REAL reference code (`minagi/plasticity.py`, `minagi/optim.py`) over
deterministic scenarios and writes JSON the Rust tests replay:

    crates/minagi-core/tests/fixtures/pure/plasticity.json
    crates/minagi-core/tests/fixtures/pure/gradsnr.json

Run from `tools/golden/ref` so `import minagi` works:

    cd tools/golden/ref && ../../../.venv/bin/python ../pure_plasticity.py

Encoding: JSON has no NaN or infinity, so a non-finite number is written as the
string "nan" / "inf" / "-inf" and a missing value as null.
"""

import json
import math
import os
import random
import sys

sys.dont_write_bytecode = True        # never write .pyc files into the read-only reference
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ref"))
OUT = os.path.join(HERE, "..", "..", "crates", "minagi-core", "tests", "fixtures", "pure")

from minagi.plasticity import Plasticity  # noqa: E402


def enc(x):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


# ------------------------------------------------------------------ series
def improving(rng, n, start, rate, noise):
    return [start - rate * i + rng.gauss(0, noise) for i in range(n)]


def flat(rng, n, level, noise):
    return [level + rng.gauss(0, noise) for _ in range(n)]


def rising(rng, n, start, rate, noise):
    return [start + rate * i + rng.gauss(0, noise) for i in range(n)]


def se_series(rng, n, base=0.012):
    return [base * (1 + 0.2 * rng.random()) for _ in range(n)]


def make_obs(vals, ses, ticks=37):
    return [{"v": enc(v), "s": enc(s), "k": ticks} for v, s in zip(vals, ses)]


# ------------------------------------------------------------------ driver
def run(p, obs):
    """Replay observations on a Plasticity; return what the Rust test must match."""
    out = []
    for o in obs:
        for _ in range(o["k"]):
            p.tick()
        v = o["v"]
        s = o["s"]
        # decode
        dec = lambda x: (float(x) if isinstance(x, str) else x)  # noqa: E731
        v, s = dec(v), dec(s)
        n_events = len(p.events)
        note = p.observe(v, s)
        t, n_eff, e = p._verdict()
        jumped = len(p.events) > n_events and p.events[-1]["kind"] == "regime"
        counted = not (v is None or not math.isfinite(float(v)))
        row = {"scale": p.scale, "t": t, "e": e, "n": n_eff}
        # Compact: only the unusual outcomes are written out (fixtures stay small)
        if jumped:
            row["jumped"] = True
        if not counted:
            row["counted"] = False
        if note is not None:
            row["note"] = note
        if p.step < Plasticity.WARMUP:            # later it is scale * 1
            row["factor"] = p.factor()
        out.append(row)
    return out


def snapshot(p):
    """What the manifest would hold: state() through a JSON round trip."""
    return json.loads(json.dumps(p.state()))


def scenario(name, obs, initial_state=None, resume_after=None):
    p = Plasticity.restore(initial_state) if initial_state is not None else Plasticity()
    sc = {"name": name, "obs": obs, "initial_state": initial_state}
    if resume_after is None:
        sc["expect"] = run(p, obs)
        sc["final_state"] = snapshot(p)
        return sc
    # Run to the resume point, snapshot as a checkpoint would, restore, carry on.
    head = run(p, obs[:resume_after])
    snap = snapshot(p)
    q = Plasticity.restore(snap)
    tail = run(q, obs[resume_after:])
    sc["expect"] = head + tail
    sc["resume_after"] = resume_after
    sc["resume_state"] = snap
    sc["final_state"] = snapshot(q)
    return sc


def main():
    scenarios = []

    # 1. steady improvement: held-out falls well clear of the noise
    rng = random.Random(11)
    n = 100
    scenarios.append(scenario("steady_improvement",
                              make_obs(improving(rng, n, 3.0, 0.02, 0.01), se_series(rng, n))))

    # 2. plateau: nothing is improving, the rate should ease down to the floor
    rng = random.Random(12)
    n = 100
    scenarios.append(scenario("plateau",
                              make_obs(flat(rng, n, 2.0, 0.02), se_series(rng, n, 0.02))))

    # 3. noisy, slow improvement (the typical real-run posture)
    rng = random.Random(13)
    n = 120
    scenarios.append(scenario("noisy_slow_improvement",
                              make_obs(improving(rng, n, 2.5, 0.0014, 0.018), se_series(rng, n, 0.018))))

    # 4. deterioration: improves, then overfits and rises
    rng = random.Random(14)
    vals = improving(rng, 50, 3.0, 0.02, 0.01) + rising(rng, 60, 1.9, 0.03, 0.01)
    n = len(vals)
    scenarios.append(scenario("deterioration", make_obs(vals, se_series(rng, n))))

    # 5. a regime change that stays: new corpus, held-out jumps by >1 nat
    rng = random.Random(15)
    vals = improving(rng, 90, 3.0, 0.015, 0.01)
    vals += [v + 1.2 for v in improving(rng, 50, 1.6, 0.015, 0.01)]
    n = len(vals)
    scenarios.append(scenario("regime_change_confirmed", make_obs(vals, se_series(rng, n))))

    # 6. a one-evaluation spike that does not persist must NOT fire
    rng = random.Random(16)
    vals = improving(rng, 90, 3.0, 0.015, 0.01)
    vals[60] += 1.5
    n = len(vals)
    scenarios.append(scenario("transient_spike_discarded", make_obs(vals, se_series(rng, n))))

    # 7. a jump that only clears the FLOOR_JUMP bar because se is tiny / missing
    rng = random.Random(17)
    vals = flat(rng, 40, 2.0, 0.005) + flat(rng, 40, 2.4, 0.005)
    scenarios.append(scenario("regime_change_no_se", make_obs(vals, [None] * len(vals))))

    # 8. a long run: improve, plateau, regime change, improve, plateau (320 evaluations)
    rng = random.Random(18)
    vals = (improving(rng, 90, 3.2, 0.012, 0.012)
            + flat(rng, 60, 2.1, 0.015)
            + [v + 0.9 for v in improving(rng, 90, 2.0, 0.01, 0.012)]
            + flat(rng, 80, 1.6, 0.012))
    n = len(vals)
    scenarios.append(scenario("long_run_320", make_obs(vals, se_series(rng, n, 0.015), ticks=25)))

    # 9. non-finite and missing readings in the middle of a run are ignored
    rng = random.Random(19)
    vals = improving(rng, 50, 3.0, 0.02, 0.01)
    ses = se_series(rng, 50)
    obs = make_obs(vals, ses)
    obs[10]["v"] = "nan"
    obs[20]["v"] = "inf"
    obs[21]["v"] = None
    obs[30]["s"] = "nan"
    obs[31]["s"] = 0.0
    obs[32]["s"] = -1.0
    obs[33]["s"] = "inf"
    scenarios.append(scenario("nonfinite_readings", obs))

    # 10. a perfectly straight line has no scatter, so neither statistic is defined
    obs = make_obs([2.0 - 0.01 * i for i in range(40)], [0.01] * 40)
    scenarios.append(scenario("perfect_line", obs))
    obs = make_obs([1.5] * 40, [0.01] * 40)
    scenarios.append(scenario("constant", obs))

    # 11. warmup: several ticks per evaluation, evaluations arrive before step 100
    rng = random.Random(20)
    obs = make_obs(improving(rng, 30, 3.0, 0.02, 0.01), se_series(rng, 30), ticks=7)
    scenarios.append(scenario("early_warmup_ticks", obs))

    # 11b. starting from a lowered rate: it can climb back to the ceiling, drop to the
    # floor, and a regime change restores it by a factor of UP
    rng = random.Random(26)
    n = 270
    scenarios.append(scenario("recover_to_ceiling",
                              make_obs(improving(rng, n, 6.0, 0.02, 0.01), se_series(rng, n)),
                              initial_state={"scale": 0.3, "step": 500}))
    rng = random.Random(27)
    n = 110
    scenarios.append(scenario("deteriorate_to_floor",
                              make_obs(rising(rng, n, 2.0, 0.03, 0.01), se_series(rng, n)),
                              initial_state={"scale": 0.3, "step": 500}))
    rng = random.Random(28)
    vals = flat(rng, 40, 2.0, 0.01) + flat(rng, 40, 3.0, 0.01)
    scenarios.append(scenario("regime_from_low_rate", make_obs(vals, se_series(rng, 80, 0.01)),
                              initial_state={"scale": 0.2, "step": 500}))

    # 12. resume mid-way, for a few shapes of run
    rng = random.Random(21)
    vals = improving(rng, 110, 3.0, 0.015, 0.01)
    ob = make_obs(vals, se_series(rng, len(vals)))
    scenarios.append(scenario("resume_mid_improvement", ob, resume_after=60))
    rng = random.Random(25)
    vals = improving(rng, 60, 3.0, 0.015, 0.01) + [v + 1.0 for v in improving(rng, 80, 2.1, 0.012, 0.01)]
    ob = make_obs(vals, se_series(rng, len(vals)))
    scenarios.append(scenario("resume_after_regime", ob, resume_after=100))
    # resumed with a jump candidate pending: the candidate is not persisted, so the
    # confirmation is lost. Python is the oracle for what happens.
    rng = random.Random(22)
    vals = improving(rng, 60, 3.0, 0.015, 0.01)
    vals += [v + 1.5 for v in improving(rng, 40, 1.6, 0.015, 0.01)]
    ob = make_obs(vals, se_series(rng, len(vals)))
    scenarios.append(scenario("resume_with_pending_jump", ob, resume_after=60))
    rng = random.Random(23)
    vals = flat(rng, 120, 2.0, 0.02)
    ob = make_obs(vals, se_series(rng, 120, 0.02))
    scenarios.append(scenario("resume_on_plateau", ob, resume_after=80))

    # 13. restore from checkpoints written in other shapes (older or partial)
    rng = random.Random(24)
    tail = make_obs(improving(rng, 15, 2.0, 0.01, 0.01), se_series(rng, 15))
    base = Plasticity()
    for o in make_obs(improving(rng, 90, 3.0, 0.015, 0.01), se_series(rng, 90)):
        for _ in range(o["k"]):
            base.tick()
        base.observe(o["v"], o["s"])
    full = snapshot(base)
    # states are also compared as a whole: round trip of what Python wrote
    restore_cases = []

    def case(name, state):
        p = Plasticity.restore(state)
        restore_cases.append({
            "name": name, "state": state,
            "state_after_restore": snapshot(p),
            "factor": p.factor(),
            "tail_obs": tail,
            "tail": run(p, tail),
        })

    case("null", None)
    case("empty_dict", {})
    case("scale_only", {"scale": 0.4})
    case("scale_and_step", {"scale": 0.4, "step": 250})
    case("full_python_state", full)
    no_f = dict(full)
    no_f.pop("F")
    case("no_fast_fit", no_f)
    no_i = dict(full)
    no_i.pop("i")
    case("no_i_counter", no_i)
    case("legacy_hist_window", {"scale": 0.8, "prev": 2.0, "step": 40,
                                "hist": [2.9, 2.85, 2.82, 2.8, 2.75, 2.7, 2.68, 2.66, 2.6, 2.58,
                                         2.55, 2.5, 2.49, 2.45, 2.4]})
    case("empty_S_dict", {"scale": 0.6, "S": {}, "i": 5, "step": 3})
    case("unknown_keys_in_S", {"scale": 0.6, "S": {"w": 3.0, "bogus": 9.0}, "i": 3})
    many_se = dict(full)
    many_se["se"] = [0.01 + 0.0001 * k for k in range(100)]   # the deque keeps the last 64
    case("more_than_64_se", many_se)
    many_events = dict(full)
    many_events["events"] = [{"at": k, "kind": "settling", "val": 2.0, "scale": 1.0 - 0.01 * k, "t": 0.5}
                             for k in range(60)]
    case("events_more_than_40", many_events)
    case("step_past_warmup", {"scale": 1.0, "step": 5000})
    case("step_inside_warmup", {"scale": 0.7, "step": 30})
    case("null_step_and_i", {"scale": 0.9, "step": None, "i": None, "S": {"w": 4.0}})

    # factor(): the warmup table, straight from the class
    factors = []
    for scale in (1.0, 0.5, 0.05):
        for step in list(range(0, 105)) + [200, 100000]:
            p = Plasticity()
            p.scale = scale
            p.step = step
            factors.append({"scale": scale, "step": step, "factor": p.factor()})

    # a few direct _fit probes (empty, one point, two points, three points)
    fits = []
    for pts in ([], [2.0], [2.0, 1.9], [2.0, 1.9, 1.85], [3.0, 2.0, 1.5, 1.4, 1.38]):
        p = Plasticity()
        for v in pts:
            p._accumulate(v)
        t, n_eff, e = p._verdict()
        fits.append({"vals": pts, "t": t, "n": n_eff, "e": e})

    doc = {
        "constants": {k: getattr(Plasticity, k) for k in
                      ("FLOOR", "CEIL", "LAM", "LAM_FAST", "EFFECT", "MIN_EFF", "NUDGE", "NUDGE_DOWN",
                       "T_W", "T_W_DOWN", "T_MID", "JUMP_SE", "HOLD_SE", "FLOOR_JUMP", "UP", "WARMUP")},
        "scenarios": scenarios,
        "restore_cases": restore_cases,
        "factors": factors,
        "fits": fits,
        "describe": [
            {"scale": 1.0, "se": [], "text": Plasticity().describe()},
            {"scale": 0.5, "se": [0.012, 0.02, 0.015], "text": _describe(0.5, [0.012, 0.02, 0.015])},
        ],
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "plasticity.json")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB), "
          f"{sum(len(s['obs']) for s in scenarios)} evaluations in {len(scenarios)} scenarios")

    write_gradsnr()


def _describe(scale, ses):
    p = Plasticity()
    p.scale = scale
    for s in ses:
        p.se_hist.append(s)
    return p.describe()


# ------------------------------------------------------------------ GradSNR
def write_gradsnr():
    import torch
    from minagi.optim import GradSNR

    class P:
        def __init__(self, g):
            self.grad = g

    g = torch.Generator().manual_seed(7)
    shapes = [(4, 3), (5,), (3,)]
    scenarios = []
    for name, signal, noise, steps, beta, none_at in (
        ("pure_noise", 0.0, 1.0, 40, 0.98, ()),
        ("mostly_signal", 1.0, 0.2, 40, 0.98, ()),
        ("mixed_beta_0.9", 0.4, 0.8, 30, 0.9, ()),
        ("missing_grads", 0.5, 0.5, 24, 0.98, (3, 4, 12)),
    ):
        direction = [torch.randn(s, generator=g) for s in shapes]
        meter = GradSNR(beta=beta)
        rows = []
        for step in range(steps):
            if step in none_at:
                params = [P(None) for _ in shapes]
            else:
                params = [P((signal * d + noise * torch.randn(d.shape, generator=g)).float())
                          for d in direction]
            r = meter.observe(params)
            rows.append({
                "grads": [None if p.grad is None else [float(x) for x in p.grad.reshape(-1)] for p in params],
                "ratio": r,
            })
        scenarios.append({"name": name, "beta": beta, "rows": rows, "n": meter.n, "sq": meter.sq})
    path = os.path.join(OUT, "gradsnr.json")
    with open(path, "w") as f:
        json.dump({"scenarios": scenarios}, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
