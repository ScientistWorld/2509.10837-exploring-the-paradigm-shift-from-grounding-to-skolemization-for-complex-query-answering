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

    def skolem_dependent(self, prev_vec, in_rel: str, out_rel: str):
        rel_out = self.relation(torch.tensor(self.rel_to_id[out_rel], device=prev_vec.device))
        vi = self.mlp_i(rel_out)
        forward = self.project(prev_vec, in_rel)
        backward = self.project(vi, out_rel)
        return self.mlp_d(torch.cat([forward, backward], dim=-1))

    def encode_query(self, query: dict):
        device = self.entity.weight.device
        anchors = torch.tensor(query["anchors"], device=device, dtype=torch.long)
        rels = query["relations"]
        qtype = query["type"]
        if qtype == "1p":
            return self.project(self.entity(anchors[0]), rels[0])
        if qtype == "2p":
            h = self.entity(anchors[0])
            v1 = self.skolem_dependent(h, rels[0], rels[1])
            return self.project(v1, rels[1])
        if qtype == "3p":
            h = self.entity(anchors[0])
            v1 = self.skolem_dependent(h, rels[0], rels[1])
            v2 = self.skolem_dependent(v1, rels[1], rels[2])
            return self.project(v2, rels[2])
        if qtype == "2i":
            return self.normadd([
                self.project(self.entity(anchors[0]), rels[0]),
                self.project(self.entity(anchors[1]), rels[1]),
            ])
        if qtype == "2in":
            pos = self.normadd([
                self.project(self.entity(anchors[0]), rels[0]),
                self.project(self.entity(anchors[1]), rels[1]),
            ])
            neg_atom = self.project(self.entity(anchors[2]), rels[2])
            neg = self.mlp_n(torch.cat([pos, neg_atom], dim=-1))
            return self.normadd([pos, neg])
        raise ValueError(f"unsupported query type: {qtype}")

    def score_vec(self, qvec):
        er, ei = self.split(self.entity.weight)
        qr, qi = self.split(qvec)
        return er @ qr + ei @ qi

    def scores(self, query: dict):
        return self.score_vec(self.encode_query(query))

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
            anchors = torch.tensor(q["anchors"], device=self.entity.weight.device, dtype=torch.long)
            rels = q["relations"]
            pos = self.normadd([
                self.project(self.entity(anchors[0]), rels[0]),
                self.project(self.entity(anchors[1]), rels[1]),
            ])
            atom = self.project(self.entity(anchors[2]), rels[2])
            neg = self.mlp_n(torch.cat([pos, atom], dim=-1))
            negneg = self.mlp_n_atom(neg)
            target = int(row["answers"][0])
            sat_terms.append(-F.logsigmoid(self.score_vec(neg)[target]))
            ax_terms.append(F.mse_loss(negneg, atom) + F.mse_loss(neg, -atom))
        return alpha * torch.stack(sat_terms).mean() + beta * torch.stack(ax_terms).mean()
