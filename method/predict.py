"""Write ranked predictions for the validation and test queries.

Output: outputs/<name>/{valid,test}_{std,hplus}.npz with
  rank_<type>   [n, 1000] int16  entity ids, best first
  trace_2p_v1, trace_3p_v1, trace_3p_v2  [n, 100] int16 (std files only):
                ranking of each existential variable's Skolem embedding, with
                intermediates already reachable on the observed graph (which
                the evaluator ignores) left out

Answers already reachable on the observed graph (``obs_*`` in the public query
files) are removed from the ranked list before truncation; the evaluator
ignores them wherever they appear, so this only avoids wasting list slots.

``--mode vsa`` scores queries with training-free VSA composition of the
stage-1 embeddings (variables = bundled projections, negation = sign flip);
it is used as the simple composition baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from lvsa import HPLUS_TYPES, LVSA, STD_TYPES, num_anchors

K = 1000
KT = 100


class ObservedGraph:
    """Neighbour lookup on the graph observed for a split (for trace filtering only)."""

    def __init__(self, triples: np.ndarray, num_relations: int):
        key = triples[:, 0].astype(np.int64) * num_relations + triples[:, 1]
        order = np.argsort(key, kind="stable")
        self.key, self.tails, self.r = key[order], triples[order, 2].astype(np.int64), num_relations

    def neighbors(self, heads, rel) -> np.ndarray:
        heads = np.asarray(heads, dtype=np.int64).reshape(-1)
        if len(heads) == 0:
            return np.empty(0, dtype=np.int64)
        k = heads * self.r + int(rel)
        lo, hi = np.searchsorted(self.key, k, "left"), np.searchsorted(self.key, k, "right")
        return np.unique(np.concatenate([self.tails[a:b] for a, b in zip(lo, hi)]))


def predict_file(model, src: Path, dst: Path, types, device, pure_vsa: bool, traces: bool, batch: int, observed=None):
    z = np.load(src, allow_pickle=False)
    out = {}
    for t in types:
        q = z[f"q_{t}"]
        indptr, indices = z[f"obs_{t}_indptr"], z[f"obs_{t}_indices"]
        na = num_anchors(t)
        ranks, tr = [], {}
        for i in range(0, len(q), batch):
            rows = q[i : i + batch]
            anchors = torch.from_numpy(rows[:, :na].astype(np.int64)).to(device)
            rels = torch.from_numpy(rows[:, na:].astype(np.int64)).to(device)
            scores, nodes = model.query_scores(t, anchors, rels, pure_vsa=pure_vsa, return_nodes=True)
            lo, hi = indptr[i], indptr[min(i + batch, len(q))]
            counts = np.diff(indptr[i : min(i + batch, len(q)) + 1])
            row_ids = np.repeat(np.arange(len(rows)), counts)
            if hi > lo:
                scores[torch.from_numpy(row_ids).to(device), torch.from_numpy(indices[lo:hi].astype(np.int64)).to(device)] = float("-inf")
            ranks.append(torch.topk(scores, K, dim=1).indices.to(torch.int16).cpu().numpy())
            if traces and t in ("2p", "3p"):
                for v in (["v1"] if t == "2p" else ["v1", "v2"]):
                    vs = model.score(nodes[0][v])
                    if observed is not None:
                        for b, row in enumerate(rows):
                            seen = observed.neighbors([row[0]], row[1])
                            if v == "v2":
                                seen = observed.neighbors(seen, row[2])
                            if len(seen):
                                vs[b, torch.from_numpy(seen).to(device)] = float("-inf")
                    tr.setdefault(v, []).append(torch.topk(vs, KT, dim=1).indices.to(torch.int16).cpu().numpy())
        out[f"rank_{t}"] = np.concatenate(ranks)
        for v, parts in tr.items():
            out[f"trace_{t}_{v}"] = np.concatenate(parts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    print(f"wrote {dst}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path, default=Path("data/downloads/fb15k237"))
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--mode", choices=["lvsa", "vsa"], default="lvsa")
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=500)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = state["config"]
    meta = json.loads((args.data / "meta.json").read_text())
    model = LVSA(meta["num_entities"], meta["num_relations"], cfg["dim"], cfg["hidden"],
                 skolem_residual=bool(cfg.get("skolem_residual", False))).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    train = np.load(args.data / "train_triples.npy", allow_pickle=False)
    valid = np.load(args.data / "valid_triples.npy", allow_pickle=False)
    observed = {"valid": ObservedGraph(train, meta["num_relations"]),
                "test": ObservedGraph(np.concatenate([train, valid]), meta["num_relations"])}
    with torch.no_grad():
        for split in args.splits.split(","):
            predict_file(model, args.data / f"{split}_std.npz", args.outputs / args.name / f"{split}_std.npz",
                         STD_TYPES, device, args.mode == "vsa", True, args.batch, observed.get(split))
            predict_file(model, args.data / f"{split}_hplus.npz", args.outputs / args.name / f"{split}_hplus.npz",
                         HPLUS_TYPES, device, args.mode == "vsa", False, args.batch)


if __name__ == "__main__":
    main()
