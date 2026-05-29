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
Prediction files are not trusted as a label source; if they include query
metadata, it must match the fixed test set.
