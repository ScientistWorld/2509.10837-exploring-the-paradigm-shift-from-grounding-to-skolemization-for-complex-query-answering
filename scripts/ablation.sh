#!/bin/bash
# Run the LVSA negation ablation corresponding to the paper's w/o L_NL setting.

set -e

cd /home/user

if [ ! -f data/synthetic/train.jsonl ]; then
  bash scripts/download.sh
fi

python method/train.py \
  --out checkpoints/lvsa_no_logic_synthetic.pt \
  --neg-beta 0.0
python method/predict.py \
  --checkpoint checkpoints/lvsa_no_logic_synthetic.pt \
  --out outputs/lvsa_no_logic_predictions.jsonl

bash scripts/evaluate.sh \
  lvsa outputs/lvsa_predictions.jsonl \
  lvsa_no_logic outputs/lvsa_no_logic_predictions.jsonl \
  random outputs/random_predictions.jsonl
