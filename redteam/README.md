# Gym scoring audit

Run all cases on a compute node with `bash redteam/run_all.sh`. Every case also
has its own `run.sh`. These scripts install a real candidate artifact, invoke
the public train evaluator, and invoke `bash scripts/evaluate_test.sh` with no
arguments in a fresh subprocess. Each case saves the actual score files,
manifest, exit codes and diagnostic counts beside its script. Generated model
checkpoints, predictions, backups and verbose logs are ignored under this
directory. Cases restore their temporary data, output and score changes even
on failure. No evaluator source or Python environment is changed.

Candidate construction uses only scientist-visible data, generator, baseline,
and (in improve mode) method code. The initial reproduction and ablation
checks reuse the provided trained predictions, as the audit permits. Other
cases generate a seed-11 reference from visible method code and reuse its
predictions. Cases called `shortcut_*` deliberately demonstrate scoring
shortcuts; they are not proposed scientific improvements.

There are 23 executed cases: ten quality/reproduction checks and thirteen
shortcut trials. The two submitted CPU bundles completed successfully; one
deliberately invalid candidate (`shortcut_null`) failed its hidden test, as
recorded in its `run_result.json` and `failure_evidence.txt`. Each successful
case retains its actual hidden-test score file. `data_checks.json` records the
data-effect checks. The final interpretation is in `../REDTEAM.md`.

`summarize.py` is analysis code, separate from candidate construction. It uses
the hidden outputs and metric definitions only after each hidden-test run.
It reports signed rank differences relative to the existing test reference
using absolute coefficient magnitudes and metric directions. Since the future
ranking cohort is unknowable, both the full finite audit cohort and honest
training-only normalization scales are disclosed. Neither is a prediction of
the eventual leaderboard's numeric score.
