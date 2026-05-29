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

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

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

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

Primary accuracy is filtered mean reciprocal rank (MRR): for each correct
answer, rank it against non-answer entities and average reciprocal ranks.
Hits@K reports the fraction of answers ranked in the top K. Higher MRR and
Hits@K are better. The reusable scaled benchmark also reports parameter count
as a resource constraint, where lower is better for comparable accuracy.

## Baselines and Targets

The paper compares against Skolemization-based CQA methods including CQD-CO,
BetaE, ConE, LMPNN, and CLMPT, and grounding-based methods including CQD-Beam,
GNN-QE, ULTRAQ, and QTO. On FB15k-237 the reported LVSA aggregate MRR is 22.4
and H@1 is 15.4, with particularly strong conjunction and negation results.
This workspace uses a scaled synthetic CQA benchmark to verify the same
evaluation behavior without requiring the full standard benchmark download.

## Evaluation Protocol

Evaluation reads prediction artifacts produced by a method and scores them
against fixed test query labels. The evaluator does not import method code or
regenerate predictions. For each query, predictions are ranked entity IDs; the
first predicted entity that is a labeled answer determines MRR and Hits@K.

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


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
