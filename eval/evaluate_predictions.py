from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def score_file(path: Path, full_entity_count: int):
    rows = read_jsonl(path)
    reciprocal = []
    hits1 = []
    hits10 = []
    by_type = {}
    for row in rows:
        answers = set(row["answers"])
        ranking = list(row["ranking"])
        rank = None
        for idx, ent in enumerate(ranking, start=1):
            if ent in answers:
                rank = idx
                break
        if rank is None:
            rank = full_entity_count + 1
        reciprocal.append(1.0 / rank)
        hits1.append(1.0 if rank <= 1 else 0.0)
        hits10.append(1.0 if rank <= 10 else 0.0)
        qtype = row["query"]["type"]
        by_type.setdefault(qtype, []).append(1.0 / rank)
    metrics = {
        "mrr": sum(reciprocal) / max(1, len(reciprocal)),
        "hits1": sum(hits1) / max(1, len(hits1)),
        "hits10": sum(hits10) / max(1, len(hits10)),
    }
    for qtype, vals in sorted(by_type.items()):
        metrics[f"mrr_{qtype}"] = sum(vals) / len(vals)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/synthetic/meta.json"))
    parser.add_argument("--out", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    if len(args.predictions) != len(args.names):
        raise SystemExit("--predictions and --names must have equal length")

    meta = json.loads(args.meta.read_text())
    results = {}
    for name, pred in zip(args.names, args.predictions):
        path = Path(pred)
        metrics = score_file(path, int(meta["num_entities"]))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            run_meta = json.loads(meta_path.read_text())
            metrics["param_count_k"] = run_meta.get("param_count", 0) / 1000.0
        results[name] = metrics
    payload = {"experiments": {"scaled_synthetic_cqa": {"results": results}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
