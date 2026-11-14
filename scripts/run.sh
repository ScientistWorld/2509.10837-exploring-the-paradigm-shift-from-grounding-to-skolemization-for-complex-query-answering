#!/bin/bash
# Compute job: reproduce the reference end to end, then run the hardening
# checks (seed replicate, better/weaker variants, shortcut attempts) through
# the hidden test. Reference outputs go to outputs/ (and therefore into
# scoring/scores_{train,test}.json); variant outputs go to tmp/hardening/.

set -euo pipefail

cd /home/user
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p tmp/hardening/logs

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i), flush=True)
PY
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
SECOND=cuda:0
if [ "$NGPU" -ge 2 ]; then SECOND=cuda:1; fi

bash scripts/download.sh
bash scripts/smoke.sh

# Independent seed replicate of the full reference pipeline (noise estimate).
seed_replicate() {
  python3 method/train.py --name lvsa_seed2 --seed 2 --device "$SECOND"
  python3 method/predict.py --checkpoint checkpoints/lvsa_seed2.pt --name lvsa_seed2 --outputs tmp/hardening/outputs --device "$SECOND"
}
if [ "$NGPU" -ge 2 ]; then
  seed_replicate > tmp/hardening/logs/lvsa_seed2.log 2>&1 &
  SEED_PID=$!
fi

# Reference method, paper ablation, baselines, scoring.
bash scripts/method.sh
bash scripts/ablation.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh

# Better-variant candidates (share the reference's stage 1 and stage 2):
#   lvsa_negall   negator trained on all five negation types instead of 2in only
#   lvsa_alltypes the same plus a joint fine-tuning stage of the Skolem modules
#                 and negator on every multi-hop / negated training type
python3 method/train.py --name lvsa_negall --seed 1 --init checkpoints/lvsa_stage2.pt --start-stage 3 \
  --stage3-types 2in,3in,inp,pin,pni --epochs-stage3 12
python3 method/predict.py --checkpoint checkpoints/lvsa_negall.pt --name lvsa_negall --outputs tmp/hardening/outputs
python3 method/train.py --name lvsa_alltypes --seed 1 --init checkpoints/lvsa_negall_stage3.pt --start-stage 4 \
  --epochs-stage4 2
python3 method/predict.py --checkpoint checkpoints/lvsa_alltypes.pt --name lvsa_alltypes --outputs tmp/hardening/outputs

if [ "$NGPU" -ge 2 ]; then
  wait "$SEED_PID" || echo "seed replicate failed; see tmp/hardening/logs/lvsa_seed2.log"
else
  seed_replicate > tmp/hardening/logs/lvsa_seed2.log 2>&1 || echo "seed replicate failed"
fi
tail -n 5 tmp/hardening/logs/lvsa_seed2.log || true

python3 redteam/hardening/fill_trace_reference.py
python3 redteam/hardening/cases.py
python3 redteam/hardening/summarize.py
python3 validate.py --post-design || echo "validate.py reported problems"
