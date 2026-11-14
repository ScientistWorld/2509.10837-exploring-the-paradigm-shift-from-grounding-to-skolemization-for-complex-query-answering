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

## Reproduction in This Gym

`method/` implements the paper's LVSA on the paper's own FB15k-237 data
(`scripts/method.sh`):
- complex embeddings with `d = 1000` (2,000 reals per entity/relation),
  Hadamard binding, NormAdd bundling, DNF max for unions, Hermitian-real
  scoring; Kahn-order sequential execution of every query DAG;
- `MLP_I(v(r))` for independent variables, `MLP_D(forward ; backward)` for
  dependent ones (backward = NormAdd of `r^-1 ⊗ MLP_I(v(r^-1))` over the
  variable's successors), `MLP_N(context ; atom)` for negation; LeakyReLU MLPs
  with two hidden layers of width 2,000;
- stage 1: ComplEx entity prediction with N3 (0.05) and the auxiliary relation
  prediction objective (weight 0.25), Adagrad lr 0.1, batch 1,000, 100 epochs
  (the paper initialises from pretrained ComplEx, d = 1000);
- stage 2: embeddings frozen, `MLP_I`/`MLP_D` on 2p+3p with cross entropy plus
  the in-batch candidate term (Eq. 25), Adam lr 5e-4, batch 512, 20 epochs;
- stage 3: negator on 2in with `L_CE + L_NS + L_NL` (alpha = beta = 1,
  `MSELoss(x, y) = ||x - y||^2` as in Eq. 20), Adam lr 5e-4, 30 epochs;
- ablation `lvsa_no_logic`: stage 3 rerun with beta = 0 (same stages 1-2).

Choices not fixed by the paper: epochs per stage, batch sizes, MLP width and
depth, relation-prediction weight. Grid search over alpha/beta/lr is not
repeated (the paper's selected values are used).
