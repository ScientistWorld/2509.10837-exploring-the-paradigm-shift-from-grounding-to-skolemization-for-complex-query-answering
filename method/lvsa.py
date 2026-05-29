"""Logic-constrained Vector Symbolic Architecture implementation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LVSA(nn.Module):
    def __init__(self, num_entities: int, relations: list[str], dim: int = 64, hidden: int = 128):
        super().__init__()
        self.num_entities = num_entities
        relations = list(relations)
        for rel in list(relations):
            inv = self.inverse_relation(rel)
            if inv not in relations:
                relations.append(inv)
        self.relations = relations
        self.rel_to_id = {r: i for i, r in enumerate(relations)}
        self.dim = dim
        self.entity = nn.Embedding(num_entities, 2 * dim)
        self.relation = nn.Embedding(len(relations), 2 * dim)
        self.mlp_i = MLP(2 * dim, 2 * dim, hidden)
        self.mlp_d = MLP(4 * dim, 2 * dim, hidden)
        self.mlp_n = MLP(4 * dim, 2 * dim, hidden)
        self.mlp_n_atom = MLP(2 * dim, 2 * dim, hidden)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    @staticmethod
    def inverse_relation(relation_name: str) -> str:
        if relation_name.endswith("_inv"):
            return relation_name[: -len("_inv")]
        return f"{relation_name}_inv"

    def split(self, x):
        return x[..., : self.dim], x[..., self.dim :]

    def bind(self, a, b):
        ar, ai = self.split(a)
        br, bi = self.split(b)
        return torch.cat([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

    def normadd(self, xs):
        stacked = torch.stack(xs, dim=0)
        mean = stacked.mean(dim=0)
        target = stacked.abs().mean(dim=-1).mean(dim=0).clamp_min(1e-6)
        current = mean.abs().mean(dim=-1).clamp_min(1e-6)
        return mean * (target / current).unsqueeze(-1)

    def project(self, head_vec, relation_name: str):
        rid = torch.tensor(self.rel_to_id[relation_name], device=head_vec.device)
        rel_vec = self.relation(rid)
        return self.bind(rel_vec, head_vec)

    def skolem_independent(self, relation_name: str):
        rid = torch.tensor(self.rel_to_id[relation_name], device=self.entity.weight.device)
        return self.mlp_i(self.relation(rid))

    def skolem_dependent(self, forward_vec, successor_relations: list[str]):
        backward_terms = []
        for relation_name in successor_relations:
            inv_rel = self.inverse_relation(relation_name)
            vi = self.skolem_independent(inv_rel)
            backward_terms.append(self.project(vi, inv_rel))
        backward = self.normadd(backward_terms) if backward_terms else torch.zeros_like(forward_vec)
        return self.mlp_d(torch.cat([forward_vec, backward], dim=-1))

    def legacy_query_to_branches(self, query: dict):
        qtype = query["type"]
        rels = query["relations"]
        if qtype == "1p":
            atoms = [{"head": "a0", "rel": rels[0], "tail": "x", "negated": False}]
        elif qtype == "2p":
            atoms = [
                {"head": "a0", "rel": rels[0], "tail": "v1", "negated": False},
                {"head": "v1", "rel": rels[1], "tail": "x", "negated": False},
            ]
        elif qtype == "3p":
            atoms = [
                {"head": "a0", "rel": rels[0], "tail": "v1", "negated": False},
                {"head": "v1", "rel": rels[1], "tail": "v2", "negated": False},
                {"head": "v2", "rel": rels[2], "tail": "x", "negated": False},
            ]
        elif qtype == "2i":
            atoms = [
                {"head": "a0", "rel": rels[0], "tail": "x", "negated": False},
                {"head": "a1", "rel": rels[1], "tail": "x", "negated": False},
            ]
        elif qtype == "2in":
            atoms = [
                {"head": "a0", "rel": rels[0], "tail": "x", "negated": False},
                {"head": "a1", "rel": rels[1], "tail": "x", "negated": False},
                {"head": "a2", "rel": rels[2], "tail": "x", "negated": True},
            ]
        else:
            raise ValueError(f"unsupported query type: {qtype}")
        return [{"target": "x", "atoms": atoms}]

    def query_branches(self, query: dict):
        return query.get("branches") or self.legacy_query_to_branches(query)

    def encode_branch(self, query: dict, branch: dict, return_trace: bool = False):
        device = self.entity.weight.device
        anchors = torch.tensor(query["anchors"], device=device, dtype=torch.long)
        node_vecs = {f"a{i}": self.entity(anchor) for i, anchor in enumerate(anchors)}
        target = branch.get("target", "x")
        atoms = branch["atoms"]
        positive_atoms = [a for a in atoms if not a.get("negated", False)]
        trace = {}

        # Kahn-style sequential execution: repeatedly materialize variables
        # whose positive predecessors have already been resolved.
        while True:
            progressed = False
            tails = {
                atom["tail"]
                for atom in positive_atoms
                if atom["tail"] != target and atom["tail"] not in node_vecs
            }
            for tail in sorted(tails):
                incoming = [
                    atom for atom in positive_atoms
                    if atom["tail"] == tail and atom["head"] in node_vecs
                ]
                unresolved_incoming = [
                    atom for atom in positive_atoms
                    if atom["tail"] == tail and atom["head"] not in node_vecs
                ]
                if not incoming or unresolved_incoming:
                    continue
                forward = self.normadd([self.project(node_vecs[a["head"]], a["rel"]) for a in incoming])
                successors = [
                    atom["rel"]
                    for atom in positive_atoms
                    if atom["head"] == tail and atom["tail"] not in node_vecs
                ]
                node_vecs[tail] = self.skolem_dependent(forward, successors)
                if return_trace and tail.startswith("v"):
                    trace[f"{tail}_scores"] = self.score_vec(node_vecs[tail])
                progressed = True
            if not progressed:
                break

        target_positive = [
            atom for atom in positive_atoms
            if atom["tail"] == target and atom["head"] in node_vecs
        ]
        if not target_positive:
            unresolved = [
                atom for atom in positive_atoms
                if atom["tail"] == target or atom["head"] not in node_vecs
            ]
            raise ValueError(f"could not resolve query branch; unresolved atoms: {unresolved}")
        positive_vec = self.normadd([self.project(node_vecs[a["head"]], a["rel"]) for a in target_positive])

        negated = [
            atom for atom in atoms
            if atom.get("negated", False) and atom["tail"] == target and atom["head"] in node_vecs
        ]
        if negated:
            terms = [positive_vec]
            for atom in negated:
                neg_atom = self.project(node_vecs[atom["head"]], atom["rel"])
                terms.append(self.mlp_n(torch.cat([positive_vec, neg_atom], dim=-1)))
            return self.normadd(terms), trace
        return positive_vec, trace

    def encode_query(self, query: dict):
        branch_vecs = [self.encode_branch(query, branch)[0] for branch in self.query_branches(query)]
        if len(branch_vecs) == 1:
            return branch_vecs[0]
        return self.normadd(branch_vecs)

    def score_vec(self, qvec):
        er, ei = self.split(self.entity.weight)
        qr, qi = self.split(qvec)
        return er @ qr + ei @ qi

    def scores(self, query: dict):
        branch_scores = [
            self.score_vec(self.encode_branch(query, branch)[0])
            for branch in self.query_branches(query)
        ]
        if len(branch_scores) == 1:
            return branch_scores[0]
        return torch.stack(branch_scores, dim=0).max(dim=0).values

    def trace_scores(self, query: dict):
        traces = {}
        for branch in self.query_branches(query):
            _, trace = self.encode_branch(query, branch, return_trace=True)
            traces.update(trace)
        return traces

    def ce_loss(self, rows):
        losses = []
        for row in rows:
            scores = self.scores(row["query"])
            answer = int(row["answers"][0])
            losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([answer], device=scores.device)))
        return torch.stack(losses).mean()

    def negator_loss(self, rows, alpha: float = 1.0, beta: float = 1.0):
        neg_rows = [r for r in rows if r["query"]["type"] == "2in"]
        if not neg_rows:
            return torch.tensor(0.0, device=self.entity.weight.device)
        sat_terms = []
        ax_terms = []
        for row in neg_rows:
            q = row["query"]
            branch = self.query_branches(q)[0]
            pos_branch = {
                "target": branch.get("target", "x"),
                "atoms": [a for a in branch["atoms"] if not a.get("negated", False)],
            }
            pos, _ = self.encode_branch(q, pos_branch)
            target = int(row["answers"][0])
            for atom_def in [a for a in branch["atoms"] if a.get("negated", False)]:
                anchor_id = int(atom_def["head"][1:]) if atom_def["head"].startswith("a") else None
                if anchor_id is None:
                    continue
                anchor = torch.tensor(q["anchors"][anchor_id], device=self.entity.weight.device)
                atom = self.project(self.entity(anchor), atom_def["rel"])
                neg = self.mlp_n(torch.cat([pos, atom], dim=-1))
                negneg = self.mlp_n_atom(neg)
                sat_terms.append(-F.logsigmoid(self.score_vec(neg)[target]))
                ax_terms.append(F.mse_loss(negneg, atom) + F.mse_loss(neg, -atom))
        loss = torch.tensor(0.0, device=self.entity.weight.device)
        if alpha:
            loss = loss + alpha * torch.stack(sat_terms).mean()
        if beta:
            loss = loss + beta * torch.stack(ax_terms).mean()
        return loss
