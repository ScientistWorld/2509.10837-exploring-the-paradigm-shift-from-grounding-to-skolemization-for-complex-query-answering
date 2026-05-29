#!/bin/bash
# Final design-stage job: score the visible train slice and held-out test slice
# against every discovered prediction artifact under outputs/.

set -e

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

python validate.py --post-design
