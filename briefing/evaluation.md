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
