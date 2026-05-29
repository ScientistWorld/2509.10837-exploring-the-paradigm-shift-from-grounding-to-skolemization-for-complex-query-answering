#!/bin/bash
# Final design-stage job: regenerate the scaled reproduction artifacts, then
# score both the standard reproduction metrics and the train/test gym slices.

set -e

cd /home/user

bash scripts/download.sh
bash scripts/smoke.sh
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/ablation.sh

bash scripts/evaluate.sh \
  lvsa outputs/lvsa_predictions.jsonl \
  lvsa_no_logic outputs/lvsa_no_logic_predictions.jsonl \
  random outputs/random_predictions.jsonl

bash scripts/evaluate_train.sh \
  lvsa outputs/lvsa_predictions.jsonl \
  lvsa_no_logic outputs/lvsa_no_logic_predictions.jsonl \
  random outputs/random_predictions.jsonl
bash scripts/evaluate_test.sh \
  lvsa outputs/lvsa_predictions.jsonl \
  lvsa_no_logic outputs/lvsa_no_logic_predictions.jsonl \
  random outputs/random_predictions.jsonl

python validate.py --compare
python validate.py --post-design
