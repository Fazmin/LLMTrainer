"""Held-out style loss of a checkpoint directory, computed by the ORIGINAL Python implementation.

usage: ref_loss.py CHECKPOINT_DIR IDS.json [CAPACITY_FACTOR]   (IDS.json: {"x": [...], "y": [...]})
prints one JSON object: {"loss": float, "steps": float, "slots": [...]}

Used by crates/minagi-core/tests/python_loss_parity.rs to check that a model trained and saved by the Rust engine scores
text the same way in the reference.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
sys.path.insert(0, REF)
os.chdir(REF)

import torch  # noqa: E402

torch.set_num_threads(1)
import train as T  # noqa: E402


def main():
    ckpt, ids_path = sys.argv[1], sys.argv[2]
    ids = json.load(open(ids_path))
    model, cfg, pool, man = T.build_paged(ckpt, torch.device("cpu"), read_only=True)
    if len(sys.argv) > 3:
        # the reference reads the capacity bound from config.yaml; a test may override it (0 = no bound)
        from minagi.pool import PooledMLP
        for m in model.modules():
            if isinstance(m, PooledMLP):
                m.capacity_factor = float(sys.argv[3])
    model.eval()
    x = torch.tensor([ids["x"]], dtype=torch.long)
    y = torch.tensor([ids["y"]], dtype=torch.long)
    with torch.no_grad():
        _, loss = model(x, y)
    print(json.dumps({"loss": float(loss), "slots": [int(s) for s in pool.slots],
                      "capacity_factor": cfg.pool_capacity_factor, "resident": pool.resident}))


if __name__ == "__main__":
    main()
