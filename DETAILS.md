# Overview

- **Paper ID:** 2509.10837
- **Title:** Exploring the Paradigm Shift from Grounding to Skolemization for Complex Query Answering on Knowledge Graphs
- **Domain:** Knowledge graph reasoning
- **TL;DR:** LVSA answers complex logical KG queries by explicitly modeling Skolem functions in a vector-symbolic architecture.

## Short Summary

The paper studies complex query answering over incomplete knowledge graphs. It
argues that grounding-based methods are faithful but expensive, while previous
Skolemization-based methods are efficient but weakly constrained. LVSA combines
complex-valued VSA binding and bundling with differentiable Skolem functions
and a logic-regularized neural negator. The reported results show better
standard benchmark accuracy than Skolemization baselines and stronger
efficiency than grounding-based search.

## Key Results

- Reported FB15k-237 aggregate MRR is 22.4 and aggregate H@1 is 15.4.
- LVSA is reported as 2.6x faster than CLMPT and 1.7x faster than ConE.
- Negation regularization ablations show lower negation MRR when logical axiom loss is removed.

---

# Problem Definition

## Research Question

How can a system answer complex first-order logical queries over an incomplete
knowledge graph when the query contains existential variables, conjunction,
disjunction, and negation?

## Why It Matters

Knowledge graphs are incomplete in practice, so exact symbolic lookup misses
valid answers. A good CQA system should generalize from observed triples while
remaining efficient enough for multi-hop and multi-variable queries.

## Success Criteria

A successful solution ranks correct answer entities highly for non-trivial
complex queries, handles negation and multi-hop paths, and keeps inference cost
controlled as query depth grows.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper proposes LVSA, a logic-constrained vector-symbolic architecture for
complex query answering. Its core contribution is explicit differentiable
Skolemization for existential variables combined with simple VSA operations and
logic-regularized negation.

## Approach

Entities and relations are represented as complex vectors. Relation projection
uses complex Hadamard binding, conjunction uses normalized vector bundling, and
candidate answers are ranked by the real part of the Hermitian inner product
between query and entity embeddings. Queries are processed sequentially in a
topological order so variable dependencies are resolved before each variable is
embedded. Independent existential variables are produced by an MLP over the
relation embedding; dependent existential variables are produced by another MLP
over forward and inverse-relation dependency embeddings. Negation is handled by
a neural negator trained with satisfiability, double-negation, and contradiction
regularizers.

## Main Claims

The primary claim is that explicit Skolemization in LVSA improves CQA accuracy
while retaining low inference complexity. Secondary claims are that the approach
scales better than grounding-based enumeration, generalizes better to harder
unobservable queries, and yields more interpretable intermediate witnesses.

---

# Evaluation

## Metrics

Primary accuracy is filtered mean reciprocal rank (MRR): for each correct
answer, rank it against non-answer entities and average reciprocal ranks.
Hits@K reports the fraction of answers ranked in the top K. Higher MRR and
Hits@K are better. The reusable scaled benchmark also reports parameter count
as a resource constraint, where lower is better for comparable accuracy.

## Baselines and Targets

Reference systems for this task include both neural query-embedding approaches
and search-based logical reasoning approaches. The public-paper reference
values are recorded in `scoring/reference.json` for context. The runnable gym
uses a scaled synthetic CQA benchmark with the same output contract and metric
types, so future methods can be evaluated without requiring the full standard
benchmark download.

## Evaluation Protocol

Evaluation reads prediction artifacts produced by a method and scores them
against fixed test query labels. The evaluator does not import method code or
regenerate predictions. For each query, predictions are ranked entity IDs; the
first predicted entity that is a labeled answer determines MRR and Hits@K.
Prediction files are not trusted as a label source; if they include query
metadata, it must match the fixed test set.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-05-29 05:25] - secondary_claims
- CPU test job completed the full LVSA reproduction and the w/o logical-axiom negation ablation.
- Full LVSA retained MRR 0.493 overall and improved negated-intersection MRR from 0.505 without the logical axiom loss to 0.643 with it.
- `python validate.py --compare` passes, and `scoring/scores.json` now includes LVSA, the no-logic ablation, and the random baseline.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-29 05:11] - core_claim_plus
- Implemented the paper's LVSA computation on a scaled synthetic CQA benchmark: complex entity/relation embeddings, Hadamard binding projections, sequential Skolemization modules, normalized conjunction, neural negation, and logic-driven negator constraints.
- Ran the method, random baseline, and no-logic negation ablation end-to-end through the reusable prediction/evaluation interface.
- LVSA reached MRR 0.4931 versus random MRR 0.0556; the logic-constrained negator improved negated-query MRR from 0.5047 to 0.6429 at equal parameter count.

### [2026-05-29 05:18] - secondary_claims
- Added a reusable traceability evaluation for Skolem intermediate rankings on multi-hop queries.
- Regenerated LVSA and ablation predictions with trace fields on a CPU test job.
- Measured trace-variable MRR 0.2419, trace-variable P@1 0.1574, and trace-path P@1 0.0671, extending the scaled reproduction to the paper's interpretability claim.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-29 06:50] - secondary_claims
- Final verification preserved the `secondary_claims` milestone.
- Cleaned method-specific wording from from-scratch-visible documentation and clarified that scaled benchmark reference targets are local gym targets, not paper-published values.
- Re-ran `python validate.py --compare`, `python validate.py --post-design`, and shell syntax checks for the main scripts; all passed.

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-29 06:09] - secondary_claims
- Continued from the completed `secondary_claims` state and independently checked the paper method summary against the packaged implementation.
- Re-ran `python validate.py --compare` and `python validate.py --post-design`; both passed.
- Confirmed the final action remains `done` because the environment already contains a validated LVSA reproduction, baseline, ablation, trace scoring, and reusable scripts.

### [2026-05-29 06:50] - secondary_claims
- Final verification preserved the `secondary_claims` milestone.
- Cleaned method-specific wording from from-scratch-visible documentation and clarified that scaled benchmark reference targets are local gym targets, not paper-published values.
- Re-ran `python validate.py --compare`, `python validate.py --post-design`, and shell syntax checks for the main scripts; all passed.

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

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
