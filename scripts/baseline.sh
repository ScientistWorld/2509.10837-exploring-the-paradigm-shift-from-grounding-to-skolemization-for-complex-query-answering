#!/bin/bash
# Baselines, each written as a method directory under outputs/:
#   random              seeded random rankings
#   symbolic_traversal  exhaustive grounding on the observed graph + popularity
#   vsa_composition     training-free VSA composition of the reference's
#                       stage-1 embeddings (needs checkpoints/lvsa_stage1.pt)
#
# Usage: bash scripts/baseline.sh

set -euo pipefail

cd "$(dirname "$0")/.."

python3 baseline/random_ranker.py --name random
python3 baseline/symbolic_traversal.py --name symbolic_traversal
if [ -f checkpoints/lvsa_stage1.pt ]; then
  python3 method/predict.py --checkpoint checkpoints/lvsa_stage1.pt --name vsa_composition --mode vsa
else
  echo "checkpoints/lvsa_stage1.pt missing; run scripts/method.sh first for vsa_composition" >&2
fi
