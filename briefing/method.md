# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper proposes LVSA, a logic-constrained vector-symbolic architecture for
complex query answering. Its core contribution is explicit differentiable
Skolemization for existential variables combined with simple VSA operations and
logic-regularized negation.

## Approach

Entities and relations are represented as complex vectors. Relation projection
uses complex Hadamard binding, conjunction uses normalized vector bundling, and
candidate answers are ranked by the real part of the Hermitian inner product
between query and entity embeddings. Queries are processed sequentially in a
topological order so variable dependencies are resolved before each variable is
embedded. Independent existential variables are produced by an MLP over the
relation embedding; dependent existential variables are produced by another MLP
over forward and inverse-relation dependency embeddings. Negation is handled by
a neural negator trained with satisfiability, double-negation, and contradiction
regularizers.

## Main Claims

The primary claim is that explicit Skolemization in LVSA improves CQA accuracy
while retaining low inference complexity. Secondary claims are that the approach
scales better than grounding-based enumeration, generalizes better to harder
unobservable queries, and yields more interpretable intermediate witnesses.
