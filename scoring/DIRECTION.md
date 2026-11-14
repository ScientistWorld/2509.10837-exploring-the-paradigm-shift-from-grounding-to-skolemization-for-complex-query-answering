# Research Direction

## Research Problem
Answer complex first-order queries (existential variables, conjunction,
disjunction, negation) over the incomplete FB15k-237 knowledge graph, ranking
answers that require missing facts, including fully unobservable queries and
deeper query shapes, while keeping intermediate reasoning traceable.

## Core Contribution
Improvements should come from better reasoning over the query structure:
how entities, relations and existentially quantified variables are
represented, how operators for projection, conjunction, disjunction and
negation are modelled and constrained, and how they are trained.

## Approach Scope
In scope: representation spaces, operators for each logical connective,
handling of existential variables, logical-consistency objectives,
curricula and training data selection from the training split, multi-type
training, regularisation, model selection on validation queries.

## Out of Bounds
- Exhaustive grounding: enumerating all entities (or all pairs/tuples) for the
  existential variables of a query, e.g. dense |V| x |V| score-matrix
  propagation, whose cost grows exponentially with query depth.
- Using test files of the public benchmark, external knowledge, or models
  pretrained on data that includes FB15k-237 test facts.
- Producing intermediate-variable rankings with a separate lookup that is not
  part of the answering computation.
- Hand-written rules over the observed graph in place of learned inference.
