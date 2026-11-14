"""Stronger LVSA training using all supplied observed triples for warm start.

No validation or test answer labels are opened. The original structured model,
losses and freeze/unfreeze curriculum remain in use. Stage 1 learns the inverse
relation embeddings as well, which the reference uses without supervision.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from method.lvsa import LVSA, load_jsonl


def train_queries(model, rows, parameters, epochs, seed, neg=False):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=0.005)
    for epoch in range(epochs):
        ordered = list(rows)
        random.Random(seed+epoch).shuffle(ordered)
        total = 0
        for offset in range(0, len(ordered), 64):
            batch = ordered[offset:offset+64]
            optimizer.zero_grad(set_to_none=True)
            loss = model.ce_loss(batch)
            if neg:
                loss = loss+model.negator_loss(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if epoch % 10 == 0 or epoch == epochs-1:
            print('query epoch', epoch+1, 'loss', total, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(11)
    data = Path('data/synthetic')
    meta = json.loads((data/'meta.json').read_text())
    model = LVSA(meta['num_entities'], meta['relations'], 64, 128)
    train = load_jsonl(data/'train.jsonl')
    targets = defaultdict(set)
    for line in (data/'triples.tsv').read_text().splitlines():
        h, rel, t = line.split('\t')
        targets[int(h), rel].add(int(t))
        targets[int(t), model.inverse_relation(rel)].add(int(h))
    keys = sorted(targets)
    heads = torch.tensor([key[0] for key in keys])
    relations = torch.tensor([model.rel_to_id[key[1]] for key in keys])
    # Uniform targets over all true tails, rather than always the smallest ID.
    target_probabilities = torch.zeros(len(keys), meta['num_entities'])
    for i, key in enumerate(keys):
        for tail in targets[key]:
            target_probabilities[i, tail] = 1/len(targets[key])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in [model.entity.weight, model.relation.weight]:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([model.entity.weight, model.relation.weight], lr=0.005)
    for epoch in range(200):
        permutation = torch.randperm(len(keys))
        for batch in permutation.split(64):
            optimizer.zero_grad(set_to_none=True)
            query = model.bind(model.entity(heads[batch]), model.relation(relations[batch]))
            scores = query @ model.entity.weight.T
            loss = -(target_probabilities[batch]*scores.log_softmax(-1)).sum(-1).mean()
            loss.backward()
            optimizer.step()
        if epoch % 25 == 0 or epoch == 199:
            print('graph warm start epoch', epoch+1, 'loss', float(loss.detach()), flush=True)
    paths = [row for row in train if row['query']['type'] in ('2p', '3p')]
    train_queries(model, paths, list(model.mlp_i.parameters())+list(model.mlp_d.parameters()), 60, 111)
    negated = [row for row in train if row['query']['type'] == '2in']
    train_queries(model, negated, list(model.mlp_n.parameters())+list(model.mlp_n_atom.parameters()), 30, 211, neg=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'meta': meta, 'args': {'dim': 64, 'hidden': 128}}, args.out)


if __name__ == '__main__':
    main()
