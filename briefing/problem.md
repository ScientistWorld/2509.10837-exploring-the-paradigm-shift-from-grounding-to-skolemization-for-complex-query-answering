# Problem Definition

## Research Question

How can a system answer complex first-order logical queries over an
incomplete knowledge graph, when the queries contain existentially quantified
intermediate variables, conjunction, disjunction and negation, and the
answers depend on facts that are missing from the observed graph?

## Setting

The knowledge graph is FB15k-237 (14,505 entities, 237 relations, each with an
inverse relation). A query is an existential first-order formula with one free
variable, e.g. "which entities `x` are reached from anchor `a` by relation
`r1` followed by `r2`, but are not linked to anchor `b` by `r3`". Its answer
set is defined on the complete graph, but a solver only sees the observed part.
Answers that can already be read off the observed graph by exact traversal are
trivial and are not scored; what counts is ranking the answers that need at
least one missing fact.

The gym uses two query benchmarks over the same graph:
- the standard benchmark (14 query types: paths of length 1-3, 2- and 3-way
  intersections, their compositions, unions, and five negated forms), where a
  query's answers are typically reachable with a single missing fact;
- a harder benchmark of fully unobservable queries, where every reasoning path
  to a scored answer needs missing facts, including longer (4-hop) paths and
  4-way intersections.

Training queries of 10 types (with all their answers on the training graph) and
validation queries (with answers) are provided; test queries come without
answers.

## Why It Matters

Real knowledge graphs are incomplete, so exact symbolic lookup misses valid
answers, and grounding every existential variable over every entity grows
exponentially with the number of variables. A useful system must infer missing
facts, compose them according to the query's logical structure (including
negation), and do so at a cost that scales to large graphs and deeper queries.
Being able to point at the intermediate entities that support an answer makes
the reasoning checkable.

## Success Criteria

A successful solution ranks the non-trivial answers of held-out queries highly
across all query types, including negated ones and fully unobservable ones,
and exposes intermediate entities for multi-hop queries that genuinely lie on
a path to such an answer.
