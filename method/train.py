"""Three-stage LVSA curriculum (paper Sec. IV-C and VI-A).

  stage 1  entity/relation embeddings on atomic (1p) facts: ComplEx with N3 and
           the auxiliary relation-prediction objective (the paper initialises
           from pretrained ComplEx, d = 1000)
  stage 2  embeddings frozen; Skolemization MLP_I / MLP_D on 2p and 3p queries
           with L_CE over all entities plus the in-batch term (Eq. 25)
  stage 3  everything else frozen; negator MLP_N on 2in queries with
           L_CE + alpha L_NS + beta L_NL (alpha = beta = 1)

Within stages 2-4 the checkpoint with the best filtered MRR on a disjoint
sample of validation queries (select_std.npz, the stage's own query types) is
kept, as model selection on validation data (the scored validation sample is
not used for this).

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


@torch.no_grad()
def select_mrr(model, sel, types, device, batch: int = 250) -> float:
    """Exact filtered MRR (%) on the model-selection queries, averaged over types."""
    model.eval()
    per_type = []
    for t in types:
        q = sel[f"q_{t}"]
        hip, hix = sel[f"hard_{t}_indptr"], sel[f"hard_{t}_indices"]
        fip, fix = sel[f"filt_{t}_indptr"], sel[f"filt_{t}_indices"]
        na = num_anchors(t)
        total = 0.0
        for i in range(0, len(q), batch):
            j = min(i + batch, len(q))
            rows = q[i:j]
            scores = model.query_scores(t, torch.from_numpy(rows[:, :na].astype(np.int64)).to(device),
                                        torch.from_numpy(rows[:, na:].astype(np.int64)).to(device))
            b = j - i
            hc = np.diff(hip[i : j + 1])
            hr = torch.from_numpy(np.repeat(np.arange(b), hc)).to(device)
            hcol = torch.from_numpy(hix[hip[i] : hip[j]].astype(np.int64)).to(device)
            fr = torch.from_numpy(np.repeat(np.arange(b), np.diff(fip[i : j + 1]))).to(device)
            fcol = torch.from_numpy(fix[fip[i] : fip[j]].astype(np.int64)).to(device)
            ignore = torch.zeros_like(scores, dtype=torch.bool)
            ignore[hr, hcol] = True
            ignore[fr, fcol] = True
            negatives = scores.masked_fill(ignore, float("-inf"))
            target = scores[hr, hcol]
            above = torch.empty_like(target)
            for c in range(0, len(target), 2048):
                above[c : c + 2048] = (negatives[hr[c : c + 2048]] > target[c : c + 2048, None]).sum(1).float()
            per_query = torch.zeros(b, device=device).index_add_(0, hr, 1.0 / (1.0 + above))
            total += float((per_query / torch.from_numpy(hc).to(device).float()).sum())
        per_type.append(total / len(q))
    model.train()
    return 100.0 * sum(per_type) / len(per_type)


def query_stage(model, args, device, train, types, params, epochs, lr, alpha, beta, in_batch, seed, name,
                select=None, select_types=None, every=0):
    freeze_except(model, params)
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    best_score, best_state = -1.0, None
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
        if select is not None and every and ((epoch + 1) % every == 0 or epoch + 1 == epochs):
            score = select_mrr(model, select, select_types, device)
            log(f"{name} epoch {epoch + 1} selection MRR {score:.3f} on {','.join(select_types)}")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"{name} kept the checkpoint with selection MRR {best_score:.3f}")


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
    p.add_argument("--epochs-stage2", type=int, default=60)
    p.add_argument("--epochs-stage3", type=int, default=100)
    p.add_argument("--select-data", type=Path, default=DATA / "select_std.npz",
                   help="disjoint validation queries for checkpoint selection")
    p.add_argument("--eval-every-stage2", type=int, default=5)
    p.add_argument("--eval-every-stage3", type=int, default=10)
    p.add_argument("--eval-every-stage4", type=int, default=1)
    p.add_argument("--skolem-residual", action="store_true", help="variant only: V = forward + MLP_D(...)")
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
    model = LVSA(meta["num_entities"], meta["num_relations"], args.dim, args.hidden,
                 skolem_residual=args.skolem_residual).to(device)
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
    select = None
    if args.select_data is not None and Path(args.select_data).exists():
        z = np.load(args.select_data, allow_pickle=False)
        select = {k: z[k] for k in z.files}
    if args.start_stage <= 1 <= args.max_stage:
        stage1(model, args, device)
        save(model, args, ck(1), 1)
    if args.start_stage <= 2 <= args.max_stage:
        train = train or load_train(args.data)
        params = list(model.mlp_i.parameters()) + list(model.mlp_d.parameters())
        types = args.stage2_types.split(",")
        query_stage(model, args, device, train, types, params, args.epochs_stage2, args.lr,
                    0.0, 0.0, not args.no_in_batch, args.seed + 2, "stage2", select, types, args.eval_every_stage2)
        save(model, args, ck(2), 2)
    if args.start_stage <= 3 <= args.max_stage:
        train = train or load_train(args.data)
        params = list(model.mlp_n.parameters())
        types = args.stage3_types.split(",")
        query_stage(model, args, device, train, types, params, args.epochs_stage3, args.lr,
                    args.alpha, args.beta, False, args.seed + 3, "stage3", select, types, args.eval_every_stage3)
        save(model, args, ck(3), 3)
    if args.epochs_stage4 and args.start_stage <= 4 <= args.max_stage:
        train = train or load_train(args.data)
        params = list(model.mlp_i.parameters()) + list(model.mlp_d.parameters()) + list(model.mlp_n.parameters())
        types = args.stage4_types.split(",")
        query_stage(model, args, device, train, types, params, args.epochs_stage4, args.lr_stage4,
                    args.alpha, args.beta, not args.no_in_batch, args.seed + 4, "stage4", select, types, args.eval_every_stage4)
        save(model, args, ck(4), 4)
    final = ck(4) if args.epochs_stage4 else ck(3)
    if final.exists():
        (args.ckpt_dir / f"{args.name}.pt").write_bytes(final.read_bytes())
        log(f"final checkpoint {args.ckpt_dir / (args.name + '.pt')}")


if __name__ == "__main__":
    main()
