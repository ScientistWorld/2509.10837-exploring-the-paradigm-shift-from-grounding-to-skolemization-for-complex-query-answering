#!/bin/bash
# Fast compute-container smoke test for LVSA query execution.

set -e

cd /home/user

python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "method")

from lvsa import LVSA, load_jsonl

meta = json.loads(Path("data/synthetic/meta.json").read_text())
rows = load_jsonl(Path("data/synthetic/train.jsonl"))[:8]
model = LVSA(meta["num_entities"], meta["relations"], dim=8, hidden=16)

for row in rows:
    scores = model.scores(row["query"])
    if scores.shape[0] != meta["num_entities"]:
        raise RuntimeError(f"unexpected score shape: {tuple(scores.shape)}")
    model.trace_scores(row["query"])

loss = model.ce_loss(rows)
if not loss.isfinite():
    raise RuntimeError("non-finite smoke-test loss")

print(f"LVSA smoke test passed on {len(rows)} queries; loss={float(loss.detach()):.4f}")
PY
