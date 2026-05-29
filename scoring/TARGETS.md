# Evaluation Targets

## Primary: scaled_synthetic_cqa
Tests whether a method ranks correct entities highly for fixed complex KG
queries containing paths, intersections, and negation.
- **Metric**: `mrr`, `hits1`, `hits10` (higher is better)
- **Reference**: deterministic scaled-benchmark calibration values, because the
  paper's FB15k/FB15k-237/NELL table values are not numerically comparable to
  this substitute dataset.

## Constraint: scaled_synthetic_cqa
Tracks model size so improvements cannot come only from unbounded parameter
scaling.
- **Metric**: `param_count_k` (lower is better for comparable accuracy)

## Ablation: scaled_synthetic_cqa
Tests whether removing explicit logical axiom regularization harms negated
query answering.
- **Metric**: `mrr_2in` (higher is better)
