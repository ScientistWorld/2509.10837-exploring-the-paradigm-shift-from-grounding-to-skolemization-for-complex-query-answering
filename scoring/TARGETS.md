# Evaluation Targets

All metrics are percentages computed by the evaluators from submitted rankings
(see `briefing/evaluation.md`). Paper values are in `scoring/reference.json`.

## Primary: fb15k237_cqa
Rank the non-trivial answers of standard FB15k-237 complex queries (14 types).
- **Metrics**: `mrr_A` (primary), `mrr_Ap`, `mrr_An`, `hits1_A`, `hits1_Ap`,
  `hits1_An`, `hits3_A`, `hits10_A`, per-type `mrr_<type>` (higher is better).
- **Reference**: the reproduced method's score in `scoring/scores_test.json`;
  the paper reports `mrr_A` 22.4 / `hits1_A` 15.4 on the full test set.
- **Baselines**: `random`, `symbolic_traversal`, `vsa_composition` must score
  clearly below the reference.

## Secondary: fb15k237_unobservable
Rank the full-inference answers of FB15k-237+H queries (14 standard types plus
4p and 4i).
- **Metrics**: `mrr_A` (primary, average of the 14 standard types), `mrr_Ap`,
  `mrr_An`, `hits1_A`, `hits10_A`, per-type `mrr_<type>`, `mrr_4p`, `mrr_4i`.
- **Reference**: paper `mrr_A` (A_H) 12.2.

## Secondary / constraint: fb15k237_traceability
Intermediate entities named for 2p/3p queries must lie on complete-graph paths
to non-trivial answers. This is the logic-fidelity requirement: answer accuracy
must not be bought with untraceable reasoning.
- **Metrics**: `trace_var_mrr` (primary), `trace_var_p1`, `trace_path_p1`.

## Constraint: valid submissions
Every ranking must be a duplicate-free list of valid entity ids for every
query; any violation scores the method as failed (worst value on every metric).

## Ablation (within fb15k237_cqa)
Removing the logical-axiom regulariser of the negation operator
(`lvsa_no_logic`) should lower `mrr_An` relative to the reference while
leaving the positive types unchanged (paper Table VII: An 10.9 → 7.6).

## Not scored
Inference throughput (paper Sec. VI-D) is not scored: metrics may not depend on
wall-clock time and cost cannot be recomputed from rankings. See
`scoring/DIRECTION.md` for the scope rule that keeps attempts within
efficient, structure-following inference.
