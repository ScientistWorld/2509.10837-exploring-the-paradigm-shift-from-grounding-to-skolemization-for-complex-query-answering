#!/bin/bash
# Paper ablation "w/o L_NL": rerun only the negator stage of the reference
# (same stage-1 embeddings and stage-2 Skolem modules) with the logical-axiom
# loss removed (beta = 0). Requires scripts/method.sh to have run.
#
# Usage: bash scripts/ablation.sh

set -euo pipefail

cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 method/train.py --name lvsa_no_logic --seed 1 --init checkpoints/lvsa_stage2.pt --start-stage 3 --beta 0
python3 method/predict.py --checkpoint checkpoints/lvsa_no_logic.pt --name lvsa_no_logic
