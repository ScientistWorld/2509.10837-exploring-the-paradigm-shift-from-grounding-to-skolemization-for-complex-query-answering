# Experiment Overview

All three experiments are measured on held-out test queries over FB15k-237,
from the ranked entity lists a method submits; the same measurements on the
public validation queries appear in `scores_train.json`.

## fb15k237_cqa (weight 0.55)
Establishes whether a method can answer complex logical queries whose answers
depend on facts missing from the observed graph. It covers 14 query shapes:
1- to 3-hop paths, 2- and 3-way intersections, their compositions, unions and
five forms with negation. Only answers that cannot be obtained by exact lookup
on the observed graph are scored, so success requires inferring missing facts
and composing them correctly. Aggregate and per-shape MRR and Hits@k are
reported; the negated shapes are reported separately because handling negation
is a distinct difficulty.

## fb15k237_unobservable (weight 0.25)
Establishes whether that ability generalises to queries in which every
reasoning path to a scored answer needs missing facts, so nothing can be
shortcut through observed partial paths. It adds longer (4-hop) paths and
4-way intersections. A method that leans on memorised observed facts, or on
grounding intermediate variables over known neighbours, scores much lower here
than on the standard benchmark.

## fb15k237_traceability (weight 0.20)
Establishes whether a method's intermediate reasoning is faithful and
inferential: for 2- and 3-hop path queries, the method names candidate
entities for each intermediate variable, and the experiment checks whether
they are missing (not observed) intermediate entities that lie on a real path
to a scored answer, and whether the top intermediate choices and the top answer
form a valid chain. Intermediates that can be read off the observed graph are
ignored, so this cannot be satisfied by graph lookup.
