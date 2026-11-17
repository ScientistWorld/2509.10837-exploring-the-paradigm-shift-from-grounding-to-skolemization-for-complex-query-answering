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

# Variants on the second GPU (run in the background):
#   lvsa_seed2              independent seed replicate of the full reference pipeline (noise estimate)
#   lvsa_residual           better-variant candidate: residual Skolem module V = forward + MLP_D(...)
#                           (not the paper's design), otherwise the reference curriculum
#   lvsa_residual_alltypes  lvsa_residual + negator on all negation types + joint fine-tuning
second_gpu() {
  python3 method/train.py --name lvsa_seed2 --seed 2 --device "$SECOND"
  python3 method/predict.py --checkpoint checkpoints/lvsa_seed2.pt --name lvsa_seed2 --outputs tmp/hardening/outputs --device "$SECOND"
  python3 method/train.py --name lvsa_residual --seed 1 --skolem-residual --init checkpoints/lvsa_stage1.pt --start-stage 2 --device "$SECOND"
  python3 method/predict.py --checkpoint checkpoints/lvsa_residual.pt --name lvsa_residual --outputs tmp/hardening/outputs --device "$SECOND"
  python3 method/train.py --name lvsa_residual_alltypes --seed 1 --skolem-residual --init checkpoints/lvsa_residual_stage2.pt \
    --start-stage 3 --stage3-types 2in,3in,inp,pin,pni --epochs-stage3 40 --eval-every-stage3 4 --epochs-stage4 6 --device "$SECOND"
  python3 method/predict.py --checkpoint checkpoints/lvsa_residual_alltypes.pt --name lvsa_residual_alltypes --outputs tmp/hardening/outputs --device "$SECOND"
}
if [ ! -f checkpoints/lvsa_stage1.pt ]; then
  # lvsa_residual starts from the reference's stage 1; train it first when absent
  python3 method/train.py --name lvsa --seed 1 --max-stage 1
fi
if [ "$NGPU" -ge 2 ]; then
  second_gpu > tmp/hardening/logs/second_gpu.log 2>&1 &
  SECOND_PID=$!
fi

# Reference method, paper ablation, baselines, scoring.
bash scripts/method.sh
bash scripts/ablation.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh

# Better-variant candidate with the paper's Skolem module (shares stages 1-2 with the reference):
#   lvsa_alltypes  negator trained on all five negation types, then joint fine-tuning of the
#                  Skolem modules and negator on every multi-hop / negated training type
python3 method/train.py --name lvsa_alltypes --seed 1 --init checkpoints/lvsa_stage2.pt --start-stage 3 \
  --stage3-types 2in,3in,inp,pin,pni --epochs-stage3 40 --eval-every-stage3 4 --epochs-stage4 6
python3 method/predict.py --checkpoint checkpoints/lvsa_alltypes.pt --name lvsa_alltypes --outputs tmp/hardening/outputs

if [ "$NGPU" -ge 2 ]; then
  wait "$SECOND_PID" || echo "second-GPU variants failed; see tmp/hardening/logs/second_gpu.log"
else
  second_gpu > tmp/hardening/logs/second_gpu.log 2>&1 || echo "second-GPU variants failed"
fi
grep -E "kept|final checkpoint|Error|error" tmp/hardening/logs/second_gpu.log || true

python3 redteam/hardening/fill_trace_reference.py
python3 redteam/hardening/cases.py
python3 redteam/hardening/summarize.py
python3 validate.py --post-design || echo "validate.py reported problems"
