# Research Direction

## Research Problem
Improve complex query answering over incomplete knowledge graphs, especially
queries with existential variables, conjunction, paths, and negation.

## Core Contribution
Improvements should stay within structured complex-query answering: composing
entity and relation representations according to the query's logical structure
while preserving efficient inference and support for negation.

## Approach Scope
Work may change the representation space, query-composition operators,
consistency objectives, curriculum, or optimization protocol while preserving
structured query execution.

## Out of Bounds
- Replacing complex-query reasoning with brute-force symbolic grounding over
  all existential assignments.
- Training a generic classifier that ignores query structure.
- Adding external facts or labels to solve the fixed benchmark by data scaling.
