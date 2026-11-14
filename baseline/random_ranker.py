"""Random ranking baseline: seeded random entity lists (and random traces)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STD_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up", "2in", "3in", "inp", "pin", "pni"]
HPLUS_TYPES = STD_TYPES + ["4p", "4i"]
K, KT = 1000, 100


def random_lists(rng, n_rows, n_entities, k):
    out = np.empty((n_rows, k), dtype=np.int16)
    for i in range(0, n_rows, 256):
        block = rng.random((min(256, n_rows - i), n_entities))
        out[i : i + len(block)] = np.argpartition(block, k, axis=1)[:, :k]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="random")
    p.add_argument("--data", type=Path, default=Path("data/downloads/fb15k237"))
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    n_entities = json.loads((args.data / "meta.json").read_text())["num_entities"]
    rng = np.random.default_rng(args.seed)
    for split in ("valid", "test"):
        for bench, types in (("std", STD_TYPES), ("hplus", HPLUS_TYPES)):
            z = np.load(args.data / f"{split}_{bench}.npz", allow_pickle=False)
            out = {f"rank_{t}": random_lists(rng, len(z[f"q_{t}"]), n_entities, K) for t in types}
            if bench == "std":
                out["trace_2p_v1"] = random_lists(rng, len(z["q_2p"]), n_entities, KT)
                out["trace_3p_v1"] = random_lists(rng, len(z["q_3p"]), n_entities, KT)
                out["trace_3p_v2"] = random_lists(rng, len(z["q_3p"]), n_entities, KT)
            dst = args.outputs / args.name / f"{split}_{bench}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, **out)
            print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
