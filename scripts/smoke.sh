#!/bin/bash
# Fast smoke test of the method code on a tiny CPU model: every query type,
# every training loss with a backward pass, and the prediction writer.

set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "method")
from lvsa import HPLUS_TYPES, LVSA, STD_TYPES, num_anchors  # noqa: E402
from predict import ObservedGraph, predict_file  # noqa: E402
from train import select_mrr  # noqa: E402

data = Path("data/downloads/fb15k237")
meta = json.loads((data / "meta.json").read_text())
n = meta["num_entities"]
torch.manual_seed(0)
model = LVSA(n, meta["num_relations"], dim=8, hidden=16)

for bench, types in (("std", STD_TYPES), ("hplus", HPLUS_TYPES)):
    z = np.load(data / f"valid_{bench}.npz", allow_pickle=False)
    for t in types:
        q = z[f"q_{t}"][:4]
        na = num_anchors(t)
        a, r = torch.from_numpy(q[:, :na].astype(np.int64)), torch.from_numpy(q[:, na:].astype(np.int64))
        for pure in (False, True):
            s, nodes = model.query_scores(t, a, r, pure_vsa=pure, return_nodes=True)
            assert s.shape == (4, n) and torch.isfinite(s).all(), (t, s.shape)

train = np.load(data / "train_queries.npz", allow_pickle=False)
tr = torch.from_numpy(np.load(data / "train_triples.npy")[:16].astype(np.int64))
loss = model.kge_loss(tr[:, 0], tr[:, 1], tr[:, 2], 0.25, 0.05)
loss.backward()
assert torch.isfinite(loss)
for t in ["2p", "3p", "2in", "3in", "inp", "pin", "pni"]:
    q = train[f"q_{t}"][:8]
    na = num_anchors(t)
    ans = train[f"ans_{t}_indices"][train[f"ans_{t}_indptr"][:8]]
    loss = model.query_loss(t, torch.from_numpy(q[:, :na].astype(np.int64)), torch.from_numpy(q[:, na:].astype(np.int64)),
                            torch.from_numpy(ans.astype(np.int64)), in_batch=True, alpha=1.0, beta=1.0)
    loss.backward()
    assert torch.isfinite(loss), t

tmp = Path("tmp/smoke")
tmp.mkdir(parents=True, exist_ok=True)
z = np.load(data / "valid_std.npz", allow_pickle=False)
small = {}
for t in STD_TYPES:
    ip = z[f"obs_{t}_indptr"]
    small[f"q_{t}"] = z[f"q_{t}"][:3]
    small[f"obs_{t}_indptr"] = ip[:4] - ip[0]
    small[f"obs_{t}_indices"] = z[f"obs_{t}_indices"][ip[0]:ip[3]]
np.savez(tmp / "valid_std.npz", **small)
obs = ObservedGraph(np.load(data / "train_triples.npy"), meta["num_relations"])
with torch.no_grad():
    predict_file(model, tmp / "valid_std.npz", tmp / "out.npz", STD_TYPES, torch.device("cpu"), False, True, 500, obs)
out = np.load(tmp / "out.npz")
q2 = small["q_2p"][0]
assert not (set(out["trace_2p_v1"][0].tolist()) & set(obs.neighbors([q2[0]], q2[1]).tolist())), "observed V1 kept in trace"

residual = LVSA(n, meta["num_relations"], dim=8, hidden=16, skolem_residual=True)
sel = np.load(data / "select_std.npz", allow_pickle=False)
small_sel = {}
for t in ("2p", "2in"):
    hp, fp = sel[f"hard_{t}_indptr"], sel[f"filt_{t}_indptr"]
    small_sel[f"q_{t}"] = sel[f"q_{t}"][:5]
    small_sel[f"hard_{t}_indptr"], small_sel[f"hard_{t}_indices"] = hp[:6] - hp[0], sel[f"hard_{t}_indices"][hp[0]:hp[5]]
    small_sel[f"filt_{t}_indptr"], small_sel[f"filt_{t}_indices"] = fp[:6] - fp[0], sel[f"filt_{t}_indices"][fp[0]:fp[5]]
score = select_mrr(residual, small_sel, ["2p", "2in"], torch.device("cpu"))
assert 0.0 <= score <= 100.0, score
assert out["rank_3p"].shape == (3, 1000) and out["trace_3p_v2"].shape == (3, 100)
for t in STD_TYPES:
    row = out[f"rank_{t}"][0]
    assert len(set(row.tolist())) == 1000
    obs = set(small[f"obs_{t}_indices"][small[f"obs_{t}_indptr"][0]:small[f"obs_{t}_indptr"][1]].tolist())
    assert not (set(row.tolist()) & obs), t
print("smoke test passed")
PY
