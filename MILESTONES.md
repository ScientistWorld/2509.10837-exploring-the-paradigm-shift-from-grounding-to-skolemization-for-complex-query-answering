# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-29 05:11] - core_claim_plus
- Implemented the paper's LVSA computation on a scaled synthetic CQA benchmark: complex entity/relation embeddings, Hadamard binding projections, sequential Skolemization modules, normalized conjunction, neural negation, and logic-driven negator constraints.
- Ran the method, random baseline, and no-logic negation ablation end-to-end through the reusable prediction/evaluation interface.
- LVSA reached MRR 0.4931 versus random MRR 0.0556; the logic-constrained negator improved negated-query MRR from 0.5047 to 0.6429 at equal parameter count.

## Stop Justification
- Completed at milestone `core_claim_plus` on the scaled reproducible benchmark. Full paper-scale FB15k/FB15k-237/NELL reproduction remains future expansion, but the packaged environment demonstrates the core claim and a negation constraint ablation with the paper's actual LVSA algorithm.
