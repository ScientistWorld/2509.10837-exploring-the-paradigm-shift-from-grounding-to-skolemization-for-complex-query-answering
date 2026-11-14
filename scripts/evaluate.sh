#!/bin/bash
# Standard evaluation entry point: scores every method under outputs/ on the
# public validation queries (scoring/scores_train.json) and on the hidden test
# queries (scoring/scores_test.json), and mirrors the test scores to
# scoring/scores.json. Uses only code in eval/; never imports method/.
#
# Usage: bash scripts/evaluate.sh              (no arguments)

set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
cp scoring/scores_test.json scoring/scores.json
