# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-05-29 05:11] - core_claim_plus
- Implemented the paper's LVSA computation on a scaled synthetic CQA benchmark: complex entity/relation embeddings, Hadamard binding projections, sequential Skolemization modules, normalized conjunction, neural negation, and logic-driven negator constraints.
- Ran the method, random baseline, and no-logic negation ablation end-to-end through the reusable prediction/evaluation interface.
- LVSA reached MRR 0.4931 versus random MRR 0.0556; the logic-constrained negator improved negated-query MRR from 0.5047 to 0.6429 at equal parameter count.

### [2026-05-29 05:18] - secondary_claims
- Added a reusable traceability evaluation for Skolem intermediate rankings on multi-hop queries.
- Regenerated LVSA and ablation predictions with trace fields on a CPU test job.
- Measured trace-variable MRR 0.2419, trace-variable P@1 0.1574, and trace-path P@1 0.0671, extending the scaled reproduction to the paper's interpretability claim.

## Stop Justification
- Completed at milestone `secondary_claims`.
