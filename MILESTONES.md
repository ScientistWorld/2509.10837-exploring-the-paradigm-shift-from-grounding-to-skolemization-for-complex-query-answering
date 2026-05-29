# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-29 06:45] - secondary_claims
- CPU test job `a80ed4a6-a5d` completed successfully after the dependent-Skolemization fix.
- The job passed the LVSA smoke test, trained the graph-structured LVSA model, trained the no-logic ablation, wrote combined scores, and passed `python validate.py --compare`.
- Refreshed scaled benchmark results: LVSA MRR 0.4914 versus random MRR 0.0556; negated-query 2in MRR 0.6334 versus 0.5109 for the no-logic ablation.
- Traceability metrics were produced from method outputs: trace-variable MRR 0.2535, trace-variable P@1 0.1638, and trace-path P@1 0.0703.
- Restored `secondary_claims` because the verified run now covers the core CQA gain plus the paper's logical-negation and traceability secondary claims in the scaled package.

### [2026-05-29 06:35] - core_claim_plus
- Reviewed failed CPU test job `5c6ed090-2a1`; stage 1 completed, then stage 2 crashed on an undefined `forward` variable in dependent Skolemization.
- Fixed `method/lvsa.py` to concatenate `forward_vec` with the backward dependency vector.
- Added `scripts/smoke.sh` and wired it into `scripts/run.sh` and `scripts/reproduce.sh` so the compute job catches LVSA forward/scoring regressions before training.
- Resubmitting a CPU test job to retrain and refresh scores.

### [2026-05-29 06:20] - core_claim_plus
- Continued from the reset `core_claim_plus` milestone and audited method, evaluation, scripts, scoring docs, and environment.
- Fixed standalone evaluation against legacy prediction artifacts by comparing canonical query identifiers while keeping fixed test labels as the source of truth.
- Fixed `scripts/reproduce.sh` to run the baseline, LVSA method, no-logic ablation, combined evaluation, and validation.
- Did not restore `secondary_claims` yet because the current scores predate the graph-structured LVSA encoder change; submitting a CPU retraining job to refresh them.

### [2026-05-29 05:55] - core_claim_plus
- Hardened the evaluator so fixed test labels are the source of truth and predictions cannot carry their own answers.
- Reworked LVSA query execution around explicit branch atoms, dependency-order variable materialization, dependent Skolemization, and generic trace scoring.
- Prepared a compute-side retraining job to refresh the benchmark scores after the implementation audit fixes.

### [2026-05-29 05:32] - core_claim_plus
- Existing scaled reproduction trains LVSA end-to-end and beats the random baseline on the deterministic CQA benchmark.
- Existing no-logic ablation shows weaker negated-query performance than the full logical objective.

## Stop Justification
- Completed at `secondary_claims`. Higher milestones would require reproducing a larger share of the paper's full benchmark matrix, including public FB15k-237/NELL query sets and additional query families beyond the current scaled package.
