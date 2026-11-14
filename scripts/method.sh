#!/bin/bash
# Train the reproduced method (three-stage curriculum) and write its ranked
# predictions to outputs/lvsa/. Extra arguments go to method/train.py.
#
# Usage: bash scripts/method.sh [train.py options]

set -euo pipefail

cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8

bash scripts/download.sh
python3 method/train.py --name lvsa --seed 1 "$@"
python3 method/predict.py --checkpoint checkpoints/lvsa.pt --name lvsa
