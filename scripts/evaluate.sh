#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   bash scripts/evaluate.sh                                        # discover outputs/*_predictions.jsonl
#   bash scripts/evaluate.sh lvsa outputs/lvsa_predictions.jsonl
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

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
    echo "Usage: bash scripts/evaluate.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]" >&2
    echo "No outputs/*_predictions.jsonl files were found for default discovery." >&2
    exit 2
  fi
  set -- "${discovered[@]}"
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/evaluate.sh <method_name> <predictions.jsonl> [<method_name> <predictions.jsonl> ...]" >&2
  exit 2
fi
if [ $(( $# % 2 )) -ne 0 ]; then
  echo "Usage: bash scripts/evaluate.sh expects method/prediction pairs; got an odd number of arguments." >&2
  exit 2
fi

names=()
preds=()
while [ "$#" -gt 0 ]; do
  names+=("$1")
  preds+=("$2")
  shift 2
done

python eval/evaluate_predictions.py --names "${names[@]}" --predictions "${preds[@]}" --out scoring/scores.json
