"""Three-stage LVSA curriculum (paper Sec. IV-C and VI-A).

  stage 1  entity/relation embeddings on atomic (1p) facts: ComplEx with N3 and
           the auxiliary relation-prediction objective (the paper initialises
           from pretrained ComplEx, d = 1000)
  stage 2  embeddings frozen; Skolemization MLP_I / MLP_D on 2p and 3p queries
           with L_CE over all entities plus the in-batch term (Eq. 25)
  stage 3  everything else frozen; negator MLP_N on 2in queries with
           L_CE + alpha L_NS + beta L_NL (alpha = beta = 1)

Each stage saves a checkpoint and is skipped when its checkpoint exists, so
runs resume after interruption and variants can share earlier stages
(``--init`` loads another run's stage checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from lvsa import LVSA, num_anchors

DATA = Path("data/downloads/fb15k237")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train(data: Path):
    z = np.load(data / "train_queries.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def sample_answers(train, qtype, rows, rng):
    indptr = train[f"ans_{qtype}_indptr"]
    indices = train[f"ans_{qtype}_indices"]
    start = indptr[rows]
    count = indptr[rows + 1] - start
    return indices[start + (rng.random(len(rows)) * count).astype(np.int64)]


def freeze_except(model, params):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)


def stage1(model, args, device):
    triples = np.load(args.data / "train_triples.npy", allow_pickle=False).astype(np.int64)
    freeze_except(model, [model.entity, model.relation])
    opt = torch.optim.Adagrad([model.entity, model.relation], lr=args.lr_stage1)
    rng = np.random.default_rng(args.seed)
    trip = torch.from_numpy(triples).to(device)
    for epoch in range(args.epochs_stage1):
        perm = torch.from_numpy(rng.permutation(len(triples))).to(device)
        total, steps = 0.0, 0
        for i in range(0, len(perm), args.batch_stage1):
            b = trip[perm[i : i + args.batch_stage1]]
            loss = model.kge_loss(b[:, 0], b[:, 1], b[:, 2], args.rel_weight, args.n3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        log(f"stage1 epoch {epoch + 1}/{args.epochs_stage1} loss {total / steps:.4f}")


def query_stage(model, args, device, train, types, params, epochs, lr, alpha, beta, in_batch, seed, name):
    freeze_except(model, params)
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        batches = []
        for t in types:
            n = len(train[f"q_{t}"])
            perm = rng.permutation(n)
            batches += [(t, perm[i : i + args.batch]) for i in range(0, n, args.batch)]
        order = rng.permutation(len(batches))
        total, steps = 0.0, 0
        for bi in order:
            t, rows = batches[bi]
            q = train[f"q_{t}"][rows]
            na = num_anchors(t)
            anchors = torch.from_numpy(q[:, :na].astype(np.int64)).to(device)
            rels = torch.from_numpy(q[:, na:].astype(np.int64)).to(device)
            ans = torch.from_numpy(sample_answers(train, t, rows, rng).astype(np.int64)).to(device)
            loss = model.query_loss(t, anchors, rels, ans, in_batch=in_batch, alpha=alpha, beta=beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        log(f"{name} epoch {epoch + 1}/{epochs} loss {total / steps:.4f}")


def save(model, args, path: Path, stage: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    torch.save({"model": model.state_dict(), "config": cfg, "stage": stage}, path)
    log(f"saved {path}")


def load_into(model, path: Path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    log(f"loaded {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name; checkpoints/<name>_stage<k>.pt")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--init", type=Path, default=None, help="start from this checkpoint (e.g. another run's stage 2)")
    p.add_argument("--start-stage", type=int, default=1)
    p.add_argument("--dim", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    # stage 1 (ComplEx-N3 + relation prediction)
    p.add_argument("--epochs-stage1", type=int, default=100)
    p.add_argument("--batch-stage1", type=int, default=1000)
    p.add_argument("--lr-stage1", type=float, default=0.1)
    p.add_argument("--n3", type=float, default=0.05)
    p.add_argument("--rel-weight", type=float, default=0.25)
    # stages 2/3 (paper: Adam, lr 5e-4, alpha = beta = 1)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs-stage2", type=int, default=20)
    p.add_argument("--epochs-stage3", type=int, default=30)
    p.add_argument("--stage2-types", default="2p,3p")
    p.add_argument("--stage3-types", default="2in")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--no-in-batch", action="store_true", help="drop the Eq. 25 in-batch term")
    # optional stage 4: joint fine-tuning of the Skolem modules and negator
    p.add_argument("--epochs-stage4", type=int, default=0)
    p.add_argument("--stage4-types", default="2p,3p,2in,3in,inp,pin,pni")
    p.add_argument("--lr-stage4", type=float, default=2.5e-4)
    p.add_argument("--max-stage", type=int, default=4)
    p.add_argument("--allow-cpu", action="store_true", help="permit the full-size run without a GPU")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu and args.dim >= 256:
        raise SystemExit("no CUDA device visible; the full-size run needs a GPU (pass --allow-cpu to force)")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(args.seed)
    meta = json.loads((args.data / "meta.json").read_text())
    model = LVSA(meta["num_entities"], meta["num_relations"], args.dim, args.hidden).to(device)
    log(f"device {device}; parameters {sum(p.numel() for p in model.parameters())}")
    if args.init is not None:
        load_into(model, args.init, device)
    ck = lambda k: args.ckpt_dir / f"{args.name}_stage{k}.pt"  # noqa: E731
    # resume from the latest finished stage of this run
    for k in (4, 3, 2, 1):
        if k >= args.start_stage and ck(k).exists():
            load_into(model, ck(k), device)
            args.start_stage = k + 1
            break

    train = None
    if args.start_stage <= 1 <= args.max_stage:
        stage1(model, args, device)
        save(model, args, ck(1), 1)
    if args.start_stage <= 2 <= args.max_stage:
        train = train or load_train(args.data)
        params = list(model.mlp_i.parameters()) + list(model.mlp_d.parameters())
        query_stage(model, args, device, train, args.stage2_types.split(","), params, args.epochs_stage2, args.lr,
                    0.0, 0.0, not args.no_in_batch, args.seed + 2, "stage2")
        save(model, args, ck(2), 2)
    if args.start_stage <= 3 <= args.max_stage:
        train = train or load_train(args.data)
        params = list(model.mlp_n.parameters())
        query_stage(model, args, device, train, args.stage3_types.split(","), params, args.epochs_stage3, args.lr,
                    args.alpha, args.beta, False, args.seed + 3, "stage3")
        save(model, args, ck(3), 3)
    if args.epochs_stage4 and args.start_stage <= 4 <= args.max_stage:
        train = train or load_train(args.data)
        params = list(model.mlp_i.parameters()) + list(model.mlp_d.parameters()) + list(model.mlp_n.parameters())
        query_stage(model, args, device, train, args.stage4_types.split(","), params, args.epochs_stage4, args.lr_stage4,
                    args.alpha, args.beta, not args.no_in_batch, args.seed + 4, "stage4")
        save(model, args, ck(4), 4)
    final = ck(4) if args.epochs_stage4 else ck(3)
    if final.exists():
        (args.ckpt_dir / f"{args.name}.pt").write_bytes(final.read_bytes())
        log(f"final checkpoint {args.ckpt_dir / (args.name + '.pt')}")


if __name__ == "__main__":
    main()
