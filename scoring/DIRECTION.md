# Research Direction

## Research Problem
Improve complex query answering over incomplete knowledge graphs, especially
queries with existential variables, conjunction, paths, and negation.

## Core Contribution
The paper's contribution is in efficient Skolemization-style query embedding
and logically constrained neural-symbolic operations. Improvements should stay
within this family: composing entity and relation representations according to
the query's logical structure.

## Approach Scope
Work may change the representation space, Skolem function parameterization,
logical regularization, curriculum, or optimization protocol while preserving
structured query execution.

## Out of Bounds
- Replacing complex-query reasoning with brute-force symbolic grounding over
  all existential assignments.
- Training a generic classifier that ignores query structure.
- Adding external facts or labels to solve the fixed benchmark by data scaling.
