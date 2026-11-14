#!/bin/bash
# Hidden test evaluation of every method on disk.
#
# Usage: bash scripts/evaluate_test.sh         (no arguments)
#
# Every directory under outputs/ is a method and is scored from its raw ranked
# lists outputs/<method>/test_std.npz and outputs/<method>/test_hplus.npz
# against the ground truth that lives only in eval/test/data/. A method whose
# files are missing or invalid is still listed, with worst values.
# Writes scoring/scores_test.json.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p scoring outputs
python3 eval/test/evaluate.py --outputs outputs --out scoring/scores_test.json
