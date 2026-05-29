"""Synthetic complex-query benchmark for knowledge-graph CQA.

The generator creates a typed directed KG with learnable but non-trivial
relations. Query files are small JSONL artifacts that can be tracked in git or
regenerated deterministically by scripts/download.sh.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


RELATIONS = ["r0", "r1", "r2", "r3"]
QUERY_TYPES = ["1p", "2p", "3p", "2i", "2in"]


def add_query_graph(query: dict):
    qtype = query["type"]
    rels = query["relations"]
    if qtype == "1p":
        atoms = [{"head": "a0", "rel": rels[0], "tail": "x", "negated": False}]
    elif qtype == "2p":
        atoms = [
            {"head": "a0", "rel": rels[0], "tail": "v1", "negated": False},
            {"head": "v1", "rel": rels[1], "tail": "x", "negated": False},
        ]
    elif qtype == "3p":
        atoms = [
            {"head": "a0", "rel": rels[0], "tail": "v1", "negated": False},
            {"head": "v1", "rel": rels[1], "tail": "v2", "negated": False},
            {"head": "v2", "rel": rels[2], "tail": "x", "negated": False},
        ]
    elif qtype == "2i":
        atoms = [
            {"head": "a0", "rel": rels[0], "tail": "x", "negated": False},
            {"head": "a1", "rel": rels[1], "tail": "x", "negated": False},
        ]
    elif qtype == "2in":
        atoms = [
            {"head": "a0", "rel": rels[0], "tail": "x", "negated": False},
            {"head": "a1", "rel": rels[1], "tail": "x", "negated": False},
            {"head": "a2", "rel": rels[2], "tail": "x", "negated": True},
        ]
    else:
        raise ValueError(f"unsupported query type: {qtype}")
    query["branches"] = [{"target": "x", "atoms": atoms}]
    return query


def _add_edge(adj, triples, h, r, t):
    adj[r][h].add(t)
    triples.add((h, r, t))


def build_kg(num_entities: int, seed: int):
    rng = random.Random(seed)
    triples = set()
    adj = {r: defaultdict(set) for r in RELATIONS}
    block = max(4, num_entities // 8)
    for h in range(num_entities):
        group = h // block
        for rid, r in enumerate(RELATIONS):
            # One deterministic structural edge plus one noisy same-band edge.
            t = (h + (rid + 1) * (group + 3)) % num_entities
            _add_edge(adj, triples, h, r, t)
            if rng.random() < 0.72:
                lo = (group * block) % num_entities
                t2 = (lo + rng.randrange(block)) % num_entities
                _add_edge(adj, triples, h, r, t2)
    return sorted(triples), adj


def _answers_1p(adj, h, r):
    return set(adj[r].get(h, set()))


def _answers_path(adj, h, rels):
    cur = {h}
    for r in rels:
        nxt = set()
        for e in cur:
            nxt.update(adj[r].get(e, set()))
        cur = nxt
        if not cur:
            break
    return cur


def _sample_with_answers(rng, num_entities, fn, max_tries=500):
    for _ in range(max_tries):
        q, ans, aux = fn()
        if ans:
            return q, sorted(ans), aux
    raise RuntimeError("could not sample query with answers")


def sample_queries(adj, num_entities: int, count: int, seed: int):
    rng = random.Random(seed)
    all_entities = set(range(num_entities))
    queries = []

    for _ in range(count):
        qtype = rng.choices(QUERY_TYPES, weights=[2, 2, 2, 2, 1], k=1)[0]

        if qtype == "1p":
            def make():
                h, r = rng.randrange(num_entities), rng.choice(RELATIONS)
                return {"type": "1p", "anchors": [h], "relations": [r]}, _answers_1p(adj, h, r), {}
        elif qtype == "2p":
            def make():
                h = rng.randrange(num_entities)
                rels = [rng.choice(RELATIONS), rng.choice(RELATIONS)]
                return {"type": "2p", "anchors": [h], "relations": rels}, _answers_path(adj, h, rels), {}
        elif qtype == "3p":
            def make():
                h = rng.randrange(num_entities)
                rels = [rng.choice(RELATIONS), rng.choice(RELATIONS), rng.choice(RELATIONS)]
                return {"type": "3p", "anchors": [h], "relations": rels}, _answers_path(adj, h, rels), {}
        elif qtype == "2i":
            def make():
                r1, r2 = rng.choice(RELATIONS), rng.choice(RELATIONS)
                h1, h2 = rng.randrange(num_entities), rng.randrange(num_entities)
                ans = _answers_1p(adj, h1, r1) & _answers_1p(adj, h2, r2)
                q = {"type": "2i", "anchors": [h1, h2], "relations": [r1, r2]}
                return q, ans, {}
        else:
            def make():
                r1, r2, rn = rng.choice(RELATIONS), rng.choice(RELATIONS), rng.choice(RELATIONS)
                h1, h2, hn = rng.randrange(num_entities), rng.randrange(num_entities), rng.randrange(num_entities)
                pos = _answers_1p(adj, h1, r1) & _answers_1p(adj, h2, r2)
                neg = _answers_1p(adj, hn, rn)
                ans = pos - neg
                if not ans or len(ans) > max(2, num_entities // 2):
                    ans = set()
                q = {"type": "2in", "anchors": [h1, h2, hn], "relations": [r1, r2, rn]}
                return q, ans, {}

        query, answers, aux = _sample_with_answers(rng, num_entities, make)
        queries.append({"query": add_query_graph(query), "answers": answers, "aux": aux})
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--entities", type=int, default=160)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--valid", type=int, default=500)
    parser.add_argument("--test", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    triples, adj = build_kg(args.entities, args.seed)
    meta = {"num_entities": args.entities, "relations": RELATIONS, "query_types": QUERY_TYPES}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (args.out / "triples.tsv").open("w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    for split, count, offset in [("train", args.train, 0), ("valid", args.valid, 1), ("test", args.test, 2)]:
        rows = sample_queries(adj, args.entities, count, args.seed + 100 + offset)
        with (args.out / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
