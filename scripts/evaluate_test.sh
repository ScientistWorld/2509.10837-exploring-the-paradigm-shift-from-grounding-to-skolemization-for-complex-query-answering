#!/bin/bash
# Held-out test-slice evaluator for the research gym.
#
# Usage mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_test.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]
#
# With no arguments, the script first reuses the train-slice input manifest,
# then falls back to discovering outputs/*_predictions.jsonl. It writes
# scoring/scores_test.json and enforces the same method set as scores_train.json.

set -e

cd /home/user

mkdir -p scoring tmp

if [ "$#" -eq 0 ]; then
  discovered=()
  if [ -f tmp/eval_train_inputs.json ]; then
    mapfile -t discovered < <(python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("tmp/eval_train_inputs.json").read_text())
if payload.get("slice") != "train":
    raise SystemExit("tmp/eval_train_inputs.json does not describe the train slice")
for item in payload.get("methods", []):
    print(item["name"])
    print(item["predictions"])
PY
)
  elif [ -f scoring/scores_train.json ]; then
    mapfile -t discovered < <(python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("scoring/scores_train.json").read_text())
methods = sorted(payload.get("experiments", {}).get("scaled_synthetic_cqa", {}).get("results", {}))
for name in methods:
    path = Path("outputs") / f"{name}_predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"could not discover predictions for train method {name}: expected {path}")
    print(name)
    print(path)
PY
)
  else
    shopt -s nullglob
    for pred in outputs/*_predictions.jsonl; do
      base=$(basename "$pred")
      name=${base%_predictions.jsonl}
      discovered+=("$name" "$pred")
    done
    shopt -u nullglob
  fi

  if [ "${#discovered[@]}" -eq 0 ]; then
    echo "Usage: bash scripts/evaluate_test.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]" >&2
    echo "No train manifest, scores_train.json mapping, or outputs/*_predictions.jsonl files were found." >&2
    exit 2
  fi
  set -- "${discovered[@]}"
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/evaluate_test.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]" >&2
  exit 2
fi
if [ $(( $# % 2 )) -ne 0 ]; then
  echo "Usage: bash scripts/evaluate_test.sh expects method/prediction pairs; got an odd number of arguments." >&2
  exit 2
fi

names=()
preds=()
while [ "$#" -gt 0 ]; do
  names+=("$1")
  preds+=("$2")
  shift 2
done

python eval/test/evaluate.py \
  --names "${names[@]}" \
  --predictions "${preds[@]}" \
  --out scoring/scores_test.json \
  --require-train-methods scoring/scores_train.json \
  --manifest-out tmp/eval_test_inputs.json
