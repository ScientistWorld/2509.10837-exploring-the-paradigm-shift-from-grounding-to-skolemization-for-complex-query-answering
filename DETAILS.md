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

How can a system answer complex first-order logical queries over an
incomplete knowledge graph, when the queries contain existentially quantified
intermediate variables, conjunction, disjunction and negation, and the
answers depend on facts that are missing from the observed graph?

## Setting

The knowledge graph is FB15k-237 (14,505 entities, 237 relations, each with an
inverse relation). A query is an existential first-order formula with one free
variable, e.g. "which entities `x` are reached from anchor `a` by relation
`r1` followed by `r2`, but are not linked to anchor `b` by `r3`". Its answer
set is defined on the complete graph, but a solver only sees the observed part.
Answers that can already be read off the observed graph by exact traversal are
trivial and are not scored; what counts is ranking the answers that need at
least one missing fact.

The gym uses two query benchmarks over the same graph:
- the standard benchmark (14 query types: paths of length 1-3, 2- and 3-way
  intersections, their compositions, unions, and five negated forms), where a
  query's answers are typically reachable with a single missing fact;
- a harder benchmark of fully unobservable queries, where every reasoning path
  to a scored answer needs missing facts, including longer (4-hop) paths and
  4-way intersections.

Training queries of 10 types (with all their answers on the training graph) and
validation queries (with answers) are provided; test queries come without
answers.

## Why It Matters

Real knowledge graphs are incomplete, so exact symbolic lookup misses valid
answers, and grounding every existential variable over every entity grows
exponentially with the number of variables. A useful system must infer missing
facts, compose them according to the query's logical structure (including
negation), and do so at a cost that scales to large graphs and deeper queries.
Being able to point at the intermediate entities that support an answer makes
the reasoning checkable.

## Success Criteria

A successful solution ranks the non-trivial answers of held-out queries highly
across all query types, including negated ones and fully unobservable ones,
and exposes intermediate entities for multi-hop queries that genuinely lie on
a path to such an answer.

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

## Reproduction in This Gym

`method/` implements the paper's LVSA on the paper's own FB15k-237 data
(`scripts/method.sh`):
- complex embeddings with `d = 1000` (2,000 reals per entity/relation),
  Hadamard binding, NormAdd bundling, DNF max for unions, Hermitian-real
  scoring; Kahn-order sequential execution of every query DAG;
- `MLP_I(v(r))` for independent variables, `MLP_D(forward ; backward)` for
  dependent ones (backward = NormAdd of `r^-1 ⊗ MLP_I(v(r^-1))` over the
  variable's successors), `MLP_N(context ; atom)` for negation; LeakyReLU MLPs
  with two hidden layers of width 2,000;
- stage 1: ComplEx entity prediction with N3 (0.05) and the auxiliary relation
  prediction objective (weight 0.25), Adagrad lr 0.1, batch 1,000, 100 epochs
  (the paper initialises from pretrained ComplEx, d = 1000);
- stage 2: embeddings frozen, `MLP_I`/`MLP_D` on 2p+3p with cross entropy plus
  the in-batch candidate term (Eq. 25), Adam lr 5e-4, batch 512, up to 60
  epochs;
- stage 3: negator on 2in with `L_CE + L_NS + L_NL` (alpha = beta = 1,
  `MSELoss(x, y) = ||x - y||^2` as in Eq. 20), Adam lr 5e-4, up to 100 epochs;
- in stages 2-3 the checkpoint with the best filtered MRR on a disjoint
  validation sample (`select_std.npz`, the stage's own query types) is kept;
- ablation `lvsa_no_logic`: stage 3 rerun with beta = 0 (same stages 1-2).

Choices not fixed by the paper: epochs per stage and checkpoint selection,
batch sizes, MLP width and depth, relation-prediction weight. Trace lists leave
out intermediates already reachable on the observed graph. Grid search over alpha/beta/lr is not
repeated (the paper's selected values are used).

---

# Evaluation

## Data

`scripts/download.sh` fetches the official release of the FB15k-237 query
benchmarks and prepares `data/downloads/fb15k237/`:

| file | content |
|---|---|
| `meta.json` | entity / relation counts, query types, query templates, list lengths |
| `train_triples.npy`, `valid_triples.npy` | observed graph `[n, 3]` (head, relation, tail); relation ids come in inverse pairs, `inverse(r) = r ^ 1` |
| `train_queries.npz` | training queries of 10 types (`q_<type>`) with all answers (`ans_<type>_indptr/indices`, CSR) |
| `valid_std.npz`, `valid_hplus.npz` | validation queries (`q_<type>`), their scored answers (`hard_*`), the entities ignored when ranking (`filt_*`) and the entities reachable on the observed training graph (`obs_*`) |
| `select_std.npz` | further validation queries (disjoint from the scored ones) with the same fields, for model selection |
| `test_std.npz`, `test_hplus.npz` | test queries (`q_<type>`) and the entities reachable on the observed train+valid graph (`obs_*`); no answers |

A query row lists its anchor entities followed by its relation ids, in the
column order of the type's template in `meta.json` (`(head, relation column,
tail, negated)` atoms; anchors `a0..`, variables `v1..`, answer `x`; union types
have one atom list per branch). Test answers exist only in the hidden
evaluator.

## Experiments (scored)

| experiment | weight | what it measures |
|---|---:|---|
| `fb15k237_cqa` | 0.55 | answering the 14 standard query types (fixed sample of 2,000 test queries per type) |
| `fb15k237_unobservable` | 0.25 | answering fully unobservable queries: 14 standard types plus 4p and 4i (up to 1,000 test queries per type) |
| `fb15k237_traceability` | 0.20 | whether the intermediate entities a method names for 2p/3p queries lie on a real path to a non-trivial answer |

## Metrics

All values are percentages; higher is better.

- **Filtered MRR / Hits@k.** For each scored (non-trivial) answer, its rank is
  1 + the number of *non-answer* entities listed above it. Other answers,
  the benchmark's easy answers, and every entity already reachable on the
  observed graph are ignored wherever they appear. Answers missing from the
  submitted list count as reciprocal rank 0. Per query, scores are averaged
  over its scored answers; per type, over queries.
  `mrr_A` / `hits*_A` average the 14 standard types, `*_Ap` the 9 positive
  types (1p 2p 3p 2i 3i ip pi 2u up), `*_An` the 5 negated types (2in 3in inp
  pin pni). `mrr_<type>` reports each type.
- **Traceability.** For a 2p query `a -r1-> V1 -r2-> x`, an entity is a valid
  `V1` if `(a, r1, V1)` and `(V1, r2, t)` hold in the complete graph for some
  scored answer `t`; for 3p, `V1` and `V2` must lie on a complete-graph chain
  from `a` to a scored answer. Intermediates already reachable from the anchor
  on the observed graph (`V1`: observed `r1`-neighbours of `a`; `V2`: observed
  two-hop entities) are ignored wherever they appear in a trace list, exactly
  like ignored answers, so only inferred (missing-fact) intermediates count.
  `trace_var_mrr` / `trace_var_p1`: reciprocal rank of the first valid
  intermediate / precision at 1, over every 2p-V1, 3p-V1 and 3p-V2 slot that
  has an unobserved valid intermediate. `trace_path_p1`: fraction of queries
  whose top non-ignored intermediates and top non-ignored answer form a valid
  complete-graph chain ending at a scored answer (2p and 3p averaged).

## Submission format

Every directory `outputs/<method>/` is one method. It must contain integer
arrays (`np.savez`, no pickles), rows in the order of the public query files:

| file | arrays |
|---|---|
| `valid_std.npz`, `test_std.npz` | `rank_<type>` `[n_type, K]` for the 14 standard types; `trace_2p_v1` `[n_2p, KT]`, `trace_3p_v1`, `trace_3p_v2` `[n_3p, KT]` |
| `valid_hplus.npz`, `test_hplus.npz` | `rank_<type>` `[n_type, K]` for the 16 types |

`rank_*` rows list entity ids best first, without repeats (`1 <= K <= 14505`;
1000 is enough in practice, and the `obs_*` entities may be left out because
they are ignored). Trace rows rank candidate entities for each existential
variable (`KT = 100` recommended; observed intermediates may be left out).
For validation the observed graph is `train_triples.npy`; for test it is
train + valid.

## Scoring rules

- `bash scripts/evaluate_train.sh` scores every method on the validation
  queries → `scoring/scores_train.json`; `bash scripts/evaluate_test.sh`
  scores every method on the hidden test queries → `scoring/scores_test.json`.
  Both take no arguments.
- Every number is recomputed from the submitted rankings; nothing else a
  method writes (claimed scores, parameter counts, timings, flags) is read.
- A present array with a wrong shape or row count, a non-integer dtype, an
  out-of-range or repeated id fails the whole method: every metric of every
  experiment gets its worst value (0). A missing file or ranking array gives
  worst values for that experiment; missing trace arrays give worst trace
  values. Failed methods are still listed.

## Baselines

Runnable baselines (in `scores_*.json`): `random` (random lists),
`symbolic_traversal` (exhaustive grounding of all variables on the observed
graph plus a relation-popularity prior), `vsa_composition` (training-free
vector composition of pretrained link-prediction embeddings). Published
numbers of other systems are recorded in `scoring/reference.json`.

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

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 52m | **GPU**: 1.3h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-19] - secondary_claims (hardening)
- Replaced the synthetic substitute with the paper's FB15k-237 standard and FB15k-237+H benchmarks; test truth only under eval/test/data.
- Reproduced LVSA at the paper's scale: hidden-test mrr_A 18.72 (paper 22.4), FB15k-237+H 11.29 (paper 12.2); w/o L_NL lowers negation MRR 8.07 -> 5.75 (paper 10.9 -> 7.6).
- Seed noise 0.024 rank; better variants +0.14 to +0.40; all audited shortcuts closed or scored worst (HARDENING.md).

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
