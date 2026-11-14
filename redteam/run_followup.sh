#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
bash redteam/better_graph_pretraining/run.sh
bash redteam/weaker_constant/run.sh
python -m redteam.data_checks
python redteam/summarize.py
