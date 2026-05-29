#!/bin/bash
# Visible train-slice evaluator for the research gym.
#
# Usage mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_train.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]
#
# With no arguments, the script discovers outputs/*_predictions.jsonl and
# evaluates every discovered method. It writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring tmp

if [ "$#" -eq 0 ]; then
  discovered=()
  shopt -s nullglob
  for pred in outputs/*_predictions.jsonl; do
    base=$(basename "$pred")
    name=${base%_predictions.jsonl}
    discovered+=("$name" "$pred")
  done
  shopt -u nullglob
  if [ "${#discovered[@]}" -eq 0 ]; then
    echo "Usage: bash scripts/evaluate_train.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]" >&2
    echo "No outputs/*_predictions.jsonl files were found for default discovery." >&2
    exit 2
  fi
  set -- "${discovered[@]}"
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/evaluate_train.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]" >&2
  exit 2
fi
if [ $(( $# % 2 )) -ne 0 ]; then
  echo "Usage: bash scripts/evaluate_train.sh expects method/prediction pairs; got an odd number of arguments." >&2
  exit 2
fi

names=()
preds=()
while [ "$#" -gt 0 ]; do
  names+=("$1")
  preds+=("$2")
  shift 2
done

python eval/train/evaluate.py \
  --names "${names[@]}" \
  --predictions "${preds[@]}" \
  --out scoring/scores_train.json \
  --manifest-out tmp/eval_train_inputs.json
