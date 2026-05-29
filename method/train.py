from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from lvsa import LVSA, load_jsonl


def batches(rows, batch_size, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def train_stage(model, rows, params, lr, epochs, batch_size, seed, neg=False):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in batches(rows, batch_size, seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss = model.ce_loss(batch)
            if neg:
                loss = loss + model.negator_loss(batch)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        print(f"epoch={epoch+1} loss={total/max(n,1):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/lvsa_synthetic.pt"))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs-stage1", type=int, default=18)
    parser.add_argument("--epochs-stage2", type=int, default=18)
    parser.add_argument("--epochs-stage3", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    meta = json.loads((args.data / "meta.json").read_text())
    train = load_jsonl(args.data / "train.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LVSA(meta["num_entities"], meta["relations"], args.dim, args.hidden).to(device)

    one_hop = [r for r in train if r["query"]["type"] == "1p"]
    paths = [r for r in train if r["query"]["type"] in {"2p", "3p"}]
    neg = [r for r in train if r["query"]["type"] == "2in"]
    print(f"stage1 rows={len(one_hop)}", flush=True)
    train_stage(model, one_hop, [model.entity.weight, model.relation.weight], args.lr, args.epochs_stage1, args.batch_size, args.seed)
    print(f"stage2 rows={len(paths)}", flush=True)
    train_stage(model, paths, list(model.mlp_i.parameters()) + list(model.mlp_d.parameters()), args.lr, args.epochs_stage2, args.batch_size, args.seed + 100)
    print(f"stage3 rows={len(neg)}", flush=True)
    neg_params = list(model.mlp_n.parameters()) + list(model.mlp_n_atom.parameters())
    train_stage(model, neg, neg_params, args.lr, args.epochs_stage3, args.batch_size, args.seed + 200, neg=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta, "args": vars(args)}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
