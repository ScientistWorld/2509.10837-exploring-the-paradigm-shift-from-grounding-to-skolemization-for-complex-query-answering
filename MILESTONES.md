# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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
- Not stopping this turn; submitting a CPU test job to refresh scores after the implementation and evaluation hardening.
