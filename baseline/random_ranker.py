from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, default=Path("data/synthetic/meta.json"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/random_predictions.jsonl"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    meta = json.loads(args.meta.read_text())
    rng = random.Random(args.seed)
    entities = list(range(meta["num_entities"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.data.open() as src, args.out.open("w") as dst:
        for line in src:
            row = json.loads(line)
            ranking = rng.sample(entities, k=min(args.topk, len(entities)))
            dst.write(json.dumps({"query": row["query"], "answers": row["answers"], "ranking": ranking}) + "\n")
    args.out.with_suffix(".meta.json").write_text(json.dumps({"method": "random", "param_count": 0}, indent=2) + "\n")


if __name__ == "__main__":
    main()
