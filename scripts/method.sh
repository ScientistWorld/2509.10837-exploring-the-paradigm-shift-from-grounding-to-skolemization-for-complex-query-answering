#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

if [ ! -f data/synthetic/train.jsonl ]; then
  bash scripts/download.sh
fi

python method/train.py "$@"
python method/predict.py --checkpoint checkpoints/lvsa_synthetic.pt --out outputs/lvsa_predictions.jsonl

if [ -f outputs/random_predictions.jsonl ]; then
  bash scripts/evaluate.sh lvsa outputs/lvsa_predictions.jsonl random outputs/random_predictions.jsonl
else
  bash scripts/evaluate.sh lvsa outputs/lvsa_predictions.jsonl
fi
