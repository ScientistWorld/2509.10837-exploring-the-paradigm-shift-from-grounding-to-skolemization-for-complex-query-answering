> **Superseded (iteration 11):** this file describes the retired synthetic gym. The current gym (FB15k-237 / FB15k-237+H, hidden test truth under `eval/test/data/`) is documented in `HARDENING.md`, `briefing/` and `scoring/`.

# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The test script calls `eval/test/evaluate.py`, writes `scoring/scores_test.json`, and carries a test manifest path; no train/test copy-paste target bug was found.
- Checked method discovery. Both train and test scripts enumerate prediction artifacts or reuse the train input manifest; neither has a hardcoded default method name.
- Checked `eval/train/` and `eval/test/` for cross-imports. The slice evaluators duplicate their logic and do not import each other.
- Checked whether the eval is attached to method outputs. Both slices read caller-supplied prediction JSONL files and optional prediction metadata; they do not regenerate the method inline.
- Checked score stubs. `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran syntax/static validation only: `bash -n` on both train/test scripts and JSON parsing for `scoring/reference.json`.

## Changes Made

- Replaced the exact even/odd train/test complement split with a stratified train/guard/test partition:
  - `eval/train/evaluate.py` now scores the first 40 percent of each query-type hash ordering.
  - `eval/test/evaluate.py` now scores the last 40 percent of each query-type hash ordering.
  - The middle 20 percent is unscored guard data. This makes the visible train split less directly informative about the hidden test slice and reduces complement-overfitting pressure while preserving similar query-type coverage.
- Reweighted `scoring/reference.json` so the runnable synthetic CQA split is the dominant scored gym target (`weight: 0.98`) and static paper-table context experiments have only minimal validator-compatible weight (`0.01` each). The previous weights let non-runnable reference tables dominate the train/test gym signal.
- Added required `description` fields to every metric entry in `scoring/reference.json`, including the static context metrics.

## Runtime Concerns For The Runner Iteration

- The held-out evaluator still slices `data/synthetic/test.jsonl`, whose full answers are present in the workspace. This is a residual leakage risk for agents that directly memorize visible answer files rather than producing predictions from a method. A stronger future design would place held-out truth only under the hidden `eval/test/` payload or evaluate a method interface on hidden queries.
- The test script enforces the same method set as `scores_train.json` when that file exists. If a runner invokes test first, it falls back to the manifest or output discovery path; this is intentional but should be watched in orchestration logs.
- The train/test split sizes are close but not identical after integer per-type partitioning: 278 train rows, 283 test rows, and 139 guard rows on the current synthetic test pool.
