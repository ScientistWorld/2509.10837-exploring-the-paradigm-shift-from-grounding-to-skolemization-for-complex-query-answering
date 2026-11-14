#!/bin/bash
# Train-side (public validation) evaluation of every method on disk.
#
# Usage: bash scripts/evaluate_train.sh        (no arguments)
#
# Every directory under outputs/ is a method and is scored from its raw ranked
# lists outputs/<method>/valid_std.npz and outputs/<method>/valid_hplus.npz
# (format: see briefing/evaluation.md). A method whose files are missing or
# invalid is still listed, with worst values. Writes scoring/scores_train.json.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p scoring outputs
python3 eval/train/evaluate.py --outputs outputs --data data/downloads/fb15k237 --out scoring/scores_train.json
