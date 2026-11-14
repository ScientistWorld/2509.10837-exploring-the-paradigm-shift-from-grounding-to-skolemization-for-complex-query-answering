"""Logic-constrained Vector Symbolic Architecture (LVSA), batched by query type.

Implements Section IV of the paper:
  * complex entity / relation embeddings (Eq. 8), stored as [Re ; Im],
  * relational projection by complex Hadamard binding (Eq. 9),
  * conjunction by normalised bundling NormAdd (Eqs. 10-12), disjunction by DNF
    (max over branch scores),
  * similarity = real part of the Hermitian inner product (Eq. 13),
  * neural negator MLP_N on [conjunctive context ; negated atom] (Eq. 14),
  * differentiable Skolemization: independent variables MLP_I(v(r)) (Eq. 15) and
    dependent variables MLP_D(forward ; inverse-relation backward) (Eqs. 16-17, 22),
  * sequential reasoning over the query DAG in Kahn topological order,
  * losses: cross entropy (Eq. 18) plus the in-batch candidate term (Eq. 25),
    satisfiability (Eq. 19) and logical-axiom (Eq. 20) negator regularisers,
  * stage-1 embedding training as ComplEx with N3 regularisation and the
    auxiliary relation-prediction objective (Chen et al., 2021).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

STD_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up", "2in", "3in", "inp", "pin", "pni"]
HPLUS_TYPES = STD_TYPES + ["4p", "4i"]

# (head, relation column, tail, negated); one list per DNF branch.
TEMPLATES = {
    "1p": [[("a0", 0, "x", False)]],
    "2p": [[("a0", 0, "v1", False), ("v1", 1, "x", False)]],
    "3p": [[("a0", 0, "v1", False), ("v1", 1, "v2", False), ("v2", 2, "x", False)]],
    "4p": [[("a0", 0, "v1", False), ("v1", 1, "v2", False), ("v2", 2, "v3", False), ("v3", 3, "x", False)]],
    "2i": [[("a0", 0, "x", False), ("a1", 1, "x", False)]],
    "3i": [[("a0", 0, "x", False), ("a1", 1, "x", False), ("a2", 2, "x", False)]],
    "4i": [[("a0", 0, "x", False), ("a1", 1, "x", False), ("a2", 2, "x", False), ("a3", 3, "x", False)]],
    "ip": [[("a0", 0, "v1", False), ("a1", 1, "v1", False), ("v1", 2, "x", False)]],
    "pi": [[("a0", 0, "v1", False), ("v1", 1, "x", False), ("a1", 2, "x", False)]],
    "2u": [[("a0", 0, "x", False)], [("a1", 1, "x", False)]],
    "up": [[("a0", 0, "v1", False), ("v1", 2, "x", False)], [("a1", 1, "v1", False), ("v1", 2, "x", False)]],
    "2in": [[("a0", 0, "x", False), ("a1", 1, "x", True)]],
    "3in": [[("a0", 0, "x", False), ("a1", 1, "x", False), ("a2", 2, "x", True)]],
    "inp": [[("a0", 0, "v1", False), ("a1", 1, "v1", True), ("v1", 2, "x", False)]],
    "pin": [[("a0", 0, "v1", False), ("v1", 1, "x", False), ("a1", 2, "x", True)]],
    "pni": [[("a0", 0, "v1", False), ("v1", 1, "x", True), ("a1", 2, "x", False)]],
}


def num_anchors(qtype: str) -> int:
    return len({a[0] for branch in TEMPLATES[qtype] for a in branch if a[0].startswith("a")})


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LVSA(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int = 1000, hidden: int = 2000, init_scale: float = 1e-3):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.entity = nn.Parameter(torch.randn(num_entities, 2 * dim) * init_scale)
        self.relation = nn.Parameter(torch.randn(num_relations, 2 * dim) * init_scale)
        self.mlp_i = MLP(2 * dim, 2 * dim, hidden)
        self.mlp_d = MLP(4 * dim, 2 * dim, hidden)
        self.mlp_n = MLP(4 * dim, 2 * dim, hidden)

    # ------------------------------------------------------------ VSA algebra
    def split(self, x):
        return x[..., : self.dim], x[..., self.dim :]

    def bind(self, a, b):
        ar, ai = self.split(a)
        br, bi = self.split(b)
        return torch.cat([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

    @staticmethod
    def normadd(xs):
        if len(xs) == 1:
            return xs[0]
        stacked = torch.stack(xs, dim=0)
        mean = stacked.mean(dim=0)
        target = stacked.abs().mean(dim=-1).mean(dim=0)
        current = mean.abs().mean(dim=-1).clamp_min(1e-12)
        return mean * (target / current).unsqueeze(-1)

    def project(self, head, rel_ids):
        return self.bind(self.relation[rel_ids], head)

    def score(self, q):
        return q @ self.entity.t()

    # ------------------------------------------------------------ Skolemization
    def independent_variable(self, rel_ids):
        """Skolem constant of an independent variable V_I in r(V_I, .) (Eq. 15)."""
        return self.mlp_i(self.relation[rel_ids])

    def backward_term(self, successor_rel_ids):
        """Superposition of r^-1 (x) V_I over the successors of a dependent variable (Eqs. 16, 22)."""
        terms = []
        for rel in successor_rel_ids:
            inv = torch.bitwise_xor(rel, torch.ones_like(rel))
            terms.append(self.project(self.independent_variable(inv), inv))
        return self.normadd(terms)

    def negate(self, context, atom):
        return self.mlp_n(torch.cat([context, atom], dim=-1))

    # ------------------------------------------------------------ query execution
    def _combine(self, incoming, node, rels, neg_records, at_target, pure_vsa):
        positive = [self.project(node[h], rels[:, j]) for (h, j, _t, neg) in incoming if not neg]
        context = self.normadd(positive)
        negated = [(h, j) for (h, j, _t, neg) in incoming if neg]
        if not negated:
            return context
        terms = [context]
        for h, j in negated:
            atom = self.project(node[h], rels[:, j])
            neg_vec = -atom if pure_vsa else self.negate(context, atom)
            terms.append(neg_vec)
            if neg_records is not None:
                neg_records.append((context, atom, neg_vec, at_target))
        return self.normadd(terms)

    def encode_branch(self, branch, anchors, rels, neg_records=None, pure_vsa=False):
        """Sequential (Kahn-order) execution of one conjunctive branch.

        Returns the target embedding and the embeddings of every node."""
        node = {f"a{i}": self.entity[anchors[:, i]] for i in range(anchors.shape[1])}
        pending = sorted({t for (_h, _j, t, _n) in branch if t != "x"})
        while pending:
            ready = [v for v in pending if all(a[0] in node for a in branch if a[2] == v)]
            if not ready:
                raise ValueError(f"cyclic or unresolved query branch: {branch}")
            for v in ready:
                incoming = [a for a in branch if a[2] == v]
                forward = self._combine(incoming, node, rels, neg_records, False, pure_vsa)
                if pure_vsa:
                    node[v] = forward
                else:
                    successors = [rels[:, j] for (h, j, _t, _n) in branch if h == v]
                    node[v] = self.mlp_d(torch.cat([forward, self.backward_term(successors)], dim=-1))
                pending.remove(v)
        target = self._combine([a for a in branch if a[2] == "x"], node, rels, neg_records, True, pure_vsa)
        return target, node

    def query_scores(self, qtype, anchors, rels, neg_records=None, pure_vsa=False, return_nodes=False):
        branch_scores, nodes = [], []
        for branch in TEMPLATES[qtype]:
            q, node = self.encode_branch(branch, anchors, rels, neg_records, pure_vsa)
            branch_scores.append(self.score(q))
            nodes.append(node)
        scores = branch_scores[0] if len(branch_scores) == 1 else torch.stack(branch_scores, 0).max(0).values
        return (scores, nodes) if return_nodes else scores

    # ------------------------------------------------------------ losses
    def kge_loss(self, h, r, t, rel_weight: float, n3: float):
        """Stage 1: ComplEx entity prediction + relation prediction + N3."""
        eh, er, et = self.entity[h], self.relation[r], self.entity[t]
        loss = F.cross_entropy(self.score(self.bind(eh, er)), t)
        if rel_weight:
            hr, hi = self.split(eh)
            tr, ti = self.split(et)
            u = torch.cat([hr * tr + hi * ti, -(hi * tr - hr * ti)], dim=-1)
            loss = loss + rel_weight * F.cross_entropy(u @ self.relation.t(), r)
        if n3:
            reg = 0.0
            for x in (eh, er, et):
                xr, xi = self.split(x)
                reg = reg + torch.sqrt(xr * xr + xi * xi + 1e-12).pow(3).sum()
            loss = loss + n3 * reg / h.shape[0]
        return loss

    def query_loss(self, qtype, anchors, rels, answers, in_batch: bool, alpha: float = 0.0, beta: float = 0.0):
        """Eq. 18 (+ Eq. 25 in-batch term) and, for negated queries, Eqs. 19-20."""
        records = [] if (alpha or beta) else None
        scores = self.query_scores(qtype, anchors, rels, neg_records=records)
        loss = F.cross_entropy(scores, answers)
        if in_batch:
            cand, idx = torch.unique(answers, return_inverse=True)
            loss = loss + F.cross_entropy(scores[:, cand], idx)
        if records:
            ns, nl = [], []
            for context, atom, neg_vec, at_target in records:
                double = self.negate(context, neg_vec)
                # Eq. 20 with the paper's MSELoss(x, y) = ||x - y||_2^2, averaged over the batch
                nl.append(((double - atom) ** 2).sum(-1).mean() + ((neg_vec + atom) ** 2).sum(-1).mean())
                if at_target:
                    ns.append(-F.logsigmoid((neg_vec * self.entity[answers]).sum(-1)).mean())
            if alpha and ns:
                loss = loss + alpha * torch.stack(ns).mean()
            if beta and nl:
                loss = loss + beta * torch.stack(nl).mean()
        return loss
