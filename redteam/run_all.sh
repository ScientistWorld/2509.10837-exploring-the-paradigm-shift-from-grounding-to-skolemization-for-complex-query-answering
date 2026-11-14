#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python - <<'PY'
import subprocess
from redteam.common import CASES
for case in CASES:
    print('\nAUDIT CASE:', case, flush=True)
    subprocess.run(['bash', f'redteam/{case}/run.sh'], check=True)
PY
python -m redteam.data_checks
python redteam/summarize.py
