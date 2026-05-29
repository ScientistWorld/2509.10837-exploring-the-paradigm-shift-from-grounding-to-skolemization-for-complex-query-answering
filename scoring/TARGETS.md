# Evaluation Targets

## Primary: scaled_synthetic_cqa
Tests whether a method ranks correct entities highly for fixed complex KG
queries containing paths, intersections, and negation.
- **Metric**: `mrr`, `hits1`, `hits10` (higher is better)
- **Reference**: compare scaled results against the reproduced random baseline
  and the paper-reported direction of improvement. Paper table values are kept
  separately in `scoring/reference.json` because they are not numerically
  comparable to this substitute dataset.

## Constraint: scaled_synthetic_cqa
Tracks model size so improvements cannot come only from unbounded parameter
scaling.
- **Metric**: `param_count_k` (lower is better for comparable accuracy)

## Ablation: scaled_synthetic_cqa
Tests whether removing explicit logical axiom regularization harms negated
query answering.
- **Metric**: `mrr_2in` (higher is better)

## Paper References
`scoring/reference.json` records the paper's FB15k-237 main-result aggregates
and Table VII negation ablation values. The runnable reproduction in
`scoring/scores.json` uses `scaled_synthetic_cqa` to report the same kinds of
effects on an accessible fixed benchmark.
