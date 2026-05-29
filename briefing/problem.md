# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

How can a system answer complex first-order logical queries over an incomplete
knowledge graph when the query contains existential variables, conjunction,
disjunction, and negation?

## Why It Matters

Knowledge graphs are incomplete in practice, so exact symbolic lookup misses
valid answers. A good CQA system should generalize from observed triples while
remaining efficient enough for multi-hop and multi-variable queries.

## Success Criteria

A successful solution ranks correct answer entities highly for non-trivial
complex queries, handles negation and multi-hop paths, and keeps inference cost
controlled as query depth grows.
