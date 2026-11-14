#!/bin/bash
# Reproduce the gym's reference results end-to-end (inside the compute
# container, one GPU): data, reproduced method, paper ablation, baselines,
# train/test scoring and validation.

set -euo pipefail

cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8

bash scripts/download.sh
bash scripts/smoke.sh
bash scripts/method.sh
bash scripts/ablation.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
python3 validate.py --post-design
