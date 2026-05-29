# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-29 04:55] - none
- Read the paper and extracted LVSA's required computation: complex vector binding, normalized bundling, differentiable Skolemization, neural negator, and curriculum losses.
- Implemented the scaled benchmark generator, LVSA method code, random baseline, artifact-based evaluator, scoring contract, and reproducibility scripts.
- Validation passes locally; the LVSA training run has not completed yet, so the milestone remains `none`.

### [2026-05-29 05:02] - none
- CPU test job trained all three LVSA curriculum stages and saved a checkpoint.
- Prediction failed due to PyTorch checkpoint loading defaults, so the method has not yet run end-to-end without errors.

### [2026-05-29 05:08] - core_claim
- CPU test job completed end-to-end training, prediction artifact generation, and evaluation.
- LVSA achieved MRR 0.493 on the scaled CQA benchmark versus 0.056 for random, with clear gains on 2p, 3p, 2i, and 2in queries.
- Added a planned negation-regularization ablation to pursue a secondary claim.

## Stop Justification
- Not stopped.
