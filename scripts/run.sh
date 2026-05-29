#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

bash scripts/download.sh
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/ablation.sh

bash scripts/evaluate.sh \
  lvsa outputs/lvsa_predictions.jsonl \
  lvsa_no_logic outputs/lvsa_no_logic_predictions.jsonl \
  random outputs/random_predictions.jsonl

python validate.py --compare
