from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SLICE = "train"
EXPERIMENT = "scaled_synthetic_cqa"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def comparable_query(query: dict) -> dict:
    return {
        "type": query.get("type"),
        "anchors": query.get("anchors"),
        "relations": query.get("relations"),
    }


def split_indices(rows: list[dict]) -> list[int]:
    """Visible stratified slice with a guard band before the held-out pool."""
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        qtype = row["query"]["type"]
        key = json.dumps(
            {"query": comparable_query(row["query"]), "answers": row.get("answers", [])},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        groups[qtype].append((digest, idx))

    selected = []
    for qtype in sorted(groups):
        ordered = sorted(groups[qtype])
        cutoff = max(1, (len(ordered) * 2) // 5)
        selected.extend(idx for _, idx in ordered[:cutoff])
    return sorted(selected)


def subset_predictions(pred_rows: list[dict], truth_rows: list[dict], indices: list[int]) -> list[dict]:
    if len(pred_rows) == len(truth_rows):
        return [pred_rows[idx] for idx in indices]
    if len(pred_rows) == len(indices):
        return pred_rows
    raise ValueError(
        f"prediction row count {len(pred_rows)} matches neither full truth count "
        f"{len(truth_rows)} nor {SLICE} slice count {len(indices)}"
    )


def score_rows(pred_rows: list[dict], truth_rows: list[dict], full_entity_count: int) -> dict:
    if len(pred_rows) != len(truth_rows):
        raise ValueError(f"prediction row count {len(pred_rows)} does not match truth row count {len(truth_rows)}")
    reciprocal = []
    hits1 = []
    hits10 = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for idx, (pred, truth) in enumerate(zip(pred_rows, truth_rows)):
        if "query" in pred and comparable_query(pred["query"]) != comparable_query(truth["query"]):
            raise ValueError(f"prediction query mismatch at {SLICE} row {idx}")
        answers = set(truth["answers"])
        ranking = list(pred.get("ranking", []))
        rank = None
        for rank_idx, ent in enumerate(ranking, start=1):
            if ent in answers:
                rank = rank_idx
                break
        if rank is None:
            rank = full_entity_count + 1
        reciprocal.append(1.0 / rank)
        hits1.append(1.0 if rank <= 1 else 0.0)
        hits10.append(1.0 if rank <= 10 else 0.0)
        by_type[truth["query"]["type"]].append(1.0 / rank)

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


def rank_of_any(ranking, valid, full_entity_count: int) -> int:
    for idx, entity in enumerate(ranking, start=1):
        if entity in valid:
            return idx
    return full_entity_count + 1


def add_trace_metrics(metrics: dict, pred_rows: list[dict], truth_rows: list[dict], full_entity_count: int, triples_path: Path) -> None:
    if not triples_path.exists():
        return
    adj = load_adjacency(triples_path)
    variable_rr = []
    variable_p1 = []
    path_p1 = []
    for pred, truth in zip(pred_rows, truth_rows):
        trace = pred.get("trace") or {}
        q = truth["query"]
        qtype = q["type"]
        rels = q["relations"]
        if qtype == "2p" and "v1_ranking" in trace:
            h = q["anchors"][0]
            valid_v1 = {
                v1
                for v1 in adj[rels[0]].get(h, set())
                if adj[rels[1]].get(v1, set()) & set(truth["answers"])
            }
            rank = rank_of_any(trace["v1_ranking"], valid_v1, full_entity_count)
            variable_rr.append(1.0 / rank)
            variable_p1.append(1.0 if rank == 1 else 0.0)
            pred_v1 = trace["v1_ranking"][0]
            pred_answer = pred.get("ranking", [None])[0] if pred.get("ranking") else None
            path_p1.append(1.0 if pred_v1 in valid_v1 and pred_answer in adj[rels[1]].get(pred_v1, set()) else 0.0)
        elif qtype == "3p" and "v1_ranking" in trace and "v2_ranking" in trace:
            h = q["anchors"][0]
            answer_set = set(truth["answers"])
            valid_v2 = {
                v2
                for v1 in adj[rels[0]].get(h, set())
                for v2 in adj[rels[1]].get(v1, set())
                if adj[rels[2]].get(v2, set()) & answer_set
            }
            valid_v1 = {
                v1
                for v1 in adj[rels[0]].get(h, set())
                if path_answers(adj, adj[rels[1]].get(v1, set()), rels[2]) & answer_set
            }
            for key, valid in [("v1_ranking", valid_v1), ("v2_ranking", valid_v2)]:
                rank = rank_of_any(trace[key], valid, full_entity_count)
                variable_rr.append(1.0 / rank)
                variable_p1.append(1.0 if rank == 1 else 0.0)
            pred_v1 = trace["v1_ranking"][0]
            pred_v2 = trace["v2_ranking"][0]
            pred_answer = pred.get("ranking", [None])[0] if pred.get("ranking") else None
            ok = (
                pred_v1 in adj[rels[0]].get(h, set())
                and pred_v2 in adj[rels[1]].get(pred_v1, set())
                and pred_answer in adj[rels[2]].get(pred_v2, set())
            )
            path_p1.append(1.0 if ok else 0.0)

    has_path_queries = any(row["query"]["type"] in {"2p", "3p"} for row in truth_rows)
    if variable_rr:
        metrics["trace_variable_mrr"] = sum(variable_rr) / len(variable_rr)
        metrics["trace_variable_p1"] = sum(variable_p1) / len(variable_p1)
    elif has_path_queries:
        metrics["trace_variable_mrr"] = 0.0
        metrics["trace_variable_p1"] = 0.0
    if path_p1:
        metrics["trace_path_p1"] = sum(path_p1) / len(path_p1)
    elif has_path_queries:
        metrics["trace_path_p1"] = 0.0


def write_manifest(path: Path | None, names: list[str], predictions: list[str], truth: Path, indices: list[int], rows: list[dict]) -> None:
    if path is None:
        return
    selected_counts = Counter(rows[idx]["query"]["type"] for idx in indices)
    payload = {
        "slice": SLICE,
        "experiment": EXPERIMENT,
        "truth": str(truth),
        "num_examples": len(indices),
        "query_type_counts": dict(sorted(selected_counts.items())),
        "methods": [{"name": name, "predictions": pred} for name, pred in zip(names, predictions)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/synthetic/meta.json"))
    parser.add_argument("--truth", type=Path, default=Path("data/synthetic/test.jsonl"))
    parser.add_argument("--triples", type=Path, default=Path("data/synthetic/triples.tsv"))
    parser.add_argument("--out", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--manifest-out", type=Path, default=Path("tmp/eval_train_inputs.json"))
    args = parser.parse_args()
    if len(args.predictions) != len(args.names):
        raise SystemExit("--predictions and --names must have equal length")

    meta = json.loads(args.meta.read_text())
    truth_rows = read_jsonl(args.truth)
    indices = split_indices(truth_rows)
    truth_slice = [truth_rows[idx] for idx in indices]
    results = {}
    for name, pred in zip(args.names, args.predictions):
        path = Path(pred)
        rows = subset_predictions(read_jsonl(path), truth_rows, indices)
        metrics = score_rows(rows, truth_slice, int(meta["num_entities"]))
        add_trace_metrics(metrics, rows, truth_slice, int(meta["num_entities"]), args.triples)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            run_meta = json.loads(meta_path.read_text())
            metrics["param_count_k"] = run_meta.get("param_count", 0) / 1000.0
        results[name] = metrics

    payload = {"experiments": {EXPERIMENT: {"results": results}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    write_manifest(args.manifest_out, args.names, args.predictions, args.truth, indices, truth_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
