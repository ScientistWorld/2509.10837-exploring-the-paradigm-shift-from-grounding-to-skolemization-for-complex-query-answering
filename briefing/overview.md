# Overview

- **Paper ID:** 2509.10837
- **Title:** Exploring the Paradigm Shift from Grounding to Skolemization for Complex Query Answering on Knowledge Graphs
- **Domain:** Knowledge graph reasoning
- **TL;DR:** LVSA answers complex logical KG queries by explicitly modeling Skolem functions in a vector-symbolic architecture.

## Short Summary

The paper studies complex query answering over incomplete knowledge graphs. It
argues that grounding-based methods are faithful but expensive, while previous
Skolemization-based methods are efficient but weakly constrained. LVSA combines
complex-valued VSA binding and bundling with differentiable Skolem functions
and a logic-regularized neural negator. The reported results show better
standard benchmark accuracy than Skolemization baselines and stronger
efficiency than grounding-based search.

## Key Results

- Reported FB15k-237 aggregate MRR is 22.4 and aggregate H@1 is 15.4.
- LVSA is reported as 2.6x faster than CLMPT and 1.7x faster than ConE.
- Negation regularization ablations show lower negation MRR when logical axiom loss is removed.
