from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lvsa import LVSA, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lvsa_synthetic.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/lvsa_predictions.jsonl"))
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LVSA(meta["num_entities"], meta["relations"], ckpt["args"]["dim"], ckpt["args"]["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows = load_jsonl(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.out.open("w") as f, torch.no_grad():
        for row in rows:
            scores = model.scores(row["query"])
            top = torch.topk(scores, k=min(args.topk, scores.numel())).indices.cpu().tolist()
            f.write(json.dumps({"query": row["query"], "answers": row["answers"], "ranking": top}) + "\n")
    elapsed = time.perf_counter() - started
    param_count = sum(p.numel() for p in model.parameters())
    meta_out = {"method": "lvsa", "elapsed_seconds": elapsed, "num_queries": len(rows), "param_count": param_count}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")


if __name__ == "__main__":
    main()
