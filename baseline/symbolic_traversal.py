"""Exhaustive symbolic grounding over the observed graph, plus a popularity prior.

Every existential variable is grounded by enumerating all entities reachable on
the observed graph (training graph for validation queries, train+valid graph
for test queries). The exactly grounded answers are the ``obs_*`` sets of the
public query files; they are left out of the ranked list (the evaluator
ignores them anyway because they are not missing facts), which is the most
favourable choice for this baseline. Every other entity is ranked by how often it is
the tail of the relations that point at the answer variable, then by global
in-degree. Intermediate-variable traces rank the observed groundings of each
variable by how many outgoing edges of the next relation they have.

This is the brute-force reasoning that does not need any learned model; it
shows how much of the benchmark is answerable without inferring missing facts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STD_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up", "2in", "3in", "inp", "pin", "pni"]
HPLUS_TYPES = STD_TYPES + ["4p", "4i"]
# relation columns of the positive atoms that point at the answer variable
TARGET_RELS = {
    "1p": [0], "2p": [1], "3p": [2], "4p": [3], "2i": [0, 1], "3i": [0, 1, 2], "4i": [0, 1, 2, 3],
    "ip": [2], "pi": [1, 2], "2u": [0, 1], "up": [2], "2in": [0], "3in": [0, 1], "inp": [2], "pin": [1], "pni": [2],
}
NUM_ANCHORS = {"1p": 1, "2p": 1, "3p": 1, "4p": 1, "2i": 2, "3i": 3, "4i": 4, "ip": 2, "pi": 2, "2u": 2, "up": 2,
               "2in": 2, "3in": 3, "inp": 2, "pin": 2, "pni": 2}
K, KT = 1000, 100


class Graph:
    def __init__(self, triples: np.ndarray, n_entities: int, n_relations: int):
        self.n, self.r = n_entities, n_relations
        key = triples[:, 0].astype(np.int64) * n_relations + triples[:, 1]
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.tails = triples[order, 2]
        self.indeg = np.zeros((n_relations, n_entities), dtype=np.float64)
        np.add.at(self.indeg, (triples[:, 1], triples[:, 2]), 1.0)
        self.outdeg = np.zeros((n_relations, n_entities), dtype=np.float64)
        np.add.at(self.outdeg, (triples[:, 1], triples[:, 0]), 1.0)
        self.global_indeg = self.indeg.sum(0)

    def neighbors(self, heads, rel):
        heads = np.atleast_1d(heads).astype(np.int64)
        lo = np.searchsorted(self.key, heads * self.r + rel, side="left")
        hi = np.searchsorted(self.key, heads * self.r + rel, side="right")
        if len(heads) == 1:
            return self.tails[lo[0] : hi[0]]
        return np.unique(np.concatenate([self.tails[a:b] for a, b in zip(lo, hi)])) if len(heads) else np.empty(0, np.int32)


def top(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, k, axis=-1)[..., :k]
    part = np.take_along_axis(scores, idx, axis=-1)
    order = np.argsort(-part, axis=-1, kind="stable")
    return np.take_along_axis(idx, order, axis=-1).astype(np.int16)


def rank_split(g: Graph, z, types, with_traces):
    out = {}
    tie = 1e-6 * g.global_indeg / max(1.0, g.global_indeg.max())
    for t in types:
        q = z[f"q_{t}"]
        indptr, indices = z[f"obs_{t}_indptr"], z[f"obs_{t}_indices"]
        na = NUM_ANCHORS[t]
        rows = []
        for i in range(len(q)):
            rels = q[i, na:]
            s = tie.copy()
            for c in TARGET_RELS[t]:
                s += g.indeg[rels[c]]
            s[indices[indptr[i] : indptr[i + 1]]] = -1e9  # exactly grounded answers are not missing facts
            rows.append(top(s, K))
        out[f"rank_{t}"] = np.stack(rows)
        if with_traces and t in ("2p", "3p"):
            v1rows, v2rows = [], []
            for i in range(len(q)):
                a0, r = q[i, 0], q[i, 1:]
                first = g.neighbors(a0, r[0])
                s1 = tie.copy()
                s1[first] += 1.0 + g.outdeg[r[1], first]
                v1rows.append(top(s1, KT))
                if t == "3p":
                    second = g.neighbors(first, r[1])
                    s2 = tie.copy()
                    s2[second] += 1.0 + g.outdeg[r[2], second]
                    v2rows.append(top(s2, KT))
            out[f"trace_{t}_v1"] = np.stack(v1rows)
            if t == "3p":
                out["trace_3p_v2"] = np.stack(v2rows)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="symbolic_traversal")
    p.add_argument("--data", type=Path, default=Path("data/downloads/fb15k237"))
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = p.parse_args()
    meta = json.loads((args.data / "meta.json").read_text())
    train = np.load(args.data / "train_triples.npy", allow_pickle=False)
    valid = np.load(args.data / "valid_triples.npy", allow_pickle=False)
    graphs = {
        "valid": Graph(train, meta["num_entities"], meta["num_relations"]),
        "test": Graph(np.concatenate([train, valid]), meta["num_entities"], meta["num_relations"]),
    }
    for split in ("valid", "test"):
        for bench, types in (("std", STD_TYPES), ("hplus", HPLUS_TYPES)):
            z = np.load(args.data / f"{split}_{bench}.npz", allow_pickle=False)
            out = rank_split(graphs[split], z, types, bench == "std")
            dst = args.outputs / args.name / f"{split}_{bench}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, **out)
            print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
