from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def load_adjacency(path: Path):
    adj = defaultdict(lambda: defaultdict(set))
    with path.open() as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            adj[r][int(h)].add(int(t))
    return adj


def path_answers(adj, heads, relation):
    out = set()
    for h in heads:
        out.update(adj[relation].get(h, set()))
    return out


def rank_of_any(ranking, valid, full_entity_count):
    for idx, entity in enumerate(ranking, start=1):
        if entity in valid:
            return idx
    return full_entity_count + 1


def add_trace_metrics(metrics: dict, rows: list[dict], full_entity_count: int, triples_path: Path):
    if not triples_path.exists():
        return
    adj = load_adjacency(triples_path)
    variable_rr = []
    variable_p1 = []
    path_p1 = []
    for row in rows:
        trace = row.get("trace") or {}
        q = row["query"]
        qtype = q["type"]
        rels = q["relations"]
        if qtype == "2p" and "v1_ranking" in trace:
            h = q["anchors"][0]
            valid_v1 = {
                v1 for v1 in adj[rels[0]].get(h, set())
                if adj[rels[1]].get(v1, set()) & set(row["answers"])
            }
            rank = rank_of_any(trace["v1_ranking"], valid_v1, full_entity_count)
            variable_rr.append(1.0 / rank)
            variable_p1.append(1.0 if rank == 1 else 0.0)
            pred_v1 = trace["v1_ranking"][0]
            pred_answer = row["ranking"][0] if row["ranking"] else None
            path_p1.append(1.0 if pred_v1 in valid_v1 and pred_answer in adj[rels[1]].get(pred_v1, set()) else 0.0)
        elif qtype == "3p" and "v1_ranking" in trace and "v2_ranking" in trace:
            h = q["anchors"][0]
            answer_set = set(row["answers"])
            valid_v2 = {
                v2 for v1 in adj[rels[0]].get(h, set())
                for v2 in adj[rels[1]].get(v1, set())
                if adj[rels[2]].get(v2, set()) & answer_set
            }
            valid_v1 = {
                v1 for v1 in adj[rels[0]].get(h, set())
                if path_answers(adj, adj[rels[1]].get(v1, set()), rels[2]) & answer_set
            }
            for key, valid in [("v1_ranking", valid_v1), ("v2_ranking", valid_v2)]:
                rank = rank_of_any(trace[key], valid, full_entity_count)
                variable_rr.append(1.0 / rank)
                variable_p1.append(1.0 if rank == 1 else 0.0)
            pred_v1 = trace["v1_ranking"][0]
            pred_v2 = trace["v2_ranking"][0]
            pred_answer = row["ranking"][0] if row["ranking"] else None
            ok = (
                pred_v1 in adj[rels[0]].get(h, set())
                and pred_v2 in adj[rels[1]].get(pred_v1, set())
                and pred_answer in adj[rels[2]].get(pred_v2, set())
            )
            path_p1.append(1.0 if ok else 0.0)
    if variable_rr:
        metrics["trace_variable_mrr"] = sum(variable_rr) / len(variable_rr)
        metrics["trace_variable_p1"] = sum(variable_p1) / len(variable_p1)
    if path_p1:
        metrics["trace_path_p1"] = sum(path_p1) / len(path_p1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/synthetic/meta.json"))
    parser.add_argument("--triples", type=Path, default=Path("data/synthetic/triples.tsv"))
    parser.add_argument("--out", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    if len(args.predictions) != len(args.names):
        raise SystemExit("--predictions and --names must have equal length")

    meta = json.loads(args.meta.read_text())
    results = {}
    for name, pred in zip(args.names, args.predictions):
        path = Path(pred)
        rows = read_jsonl(path)
        metrics = score_file(path, int(meta["num_entities"]))
        add_trace_metrics(metrics, rows, int(meta["num_entities"]), args.triples)
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
