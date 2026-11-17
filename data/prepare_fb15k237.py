"""Prepare the public FB15k-237 complex-query data from the official release zip.

Source: https://github.com/april-tools/is-cqa-complex/releases/tag/benchs-1.0
(``iscqa-compl-benchmarks-v1.1.zip``), which bundles the standard BetaE
FB15k-237 query benchmark (Ren & Leskovec, 2020) and the harder FB15k-237+H
benchmark (Gregucci et al., 2025) together with the FB15k-237 triple splits.

Only public material is written here:
  * the training graph and the validation graph (triples, inverse edges included),
  * training queries with all their answers,
  * validation queries with their answers (both benchmarks), plus a disjoint
    sample of standard validation queries for model selection (select_std.npz),
  * test *queries* (no answers) for both benchmarks, plus, for every validation
    and test query, the answers that are already reachable by exact traversal of
    the graph observed at that time (``obs_*``). These are derived from public
    triples only, so they reveal nothing about held-out facts.

Test answers and the test triples are never read by this script.

Pure standard library: the release pickles are read with a restricted unpickler
that only reconstructs builtin containers, and arrays are written in the .npy /
.npz format so that numpy can load them with ``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import pickle
import random
import struct
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = "iscqa-compl-benchmarks/"
STD = ROOT + "old_benchmarks/FB15k-237-betae/"
HPLUS = ROOT + "new_benchmarks/FB15k-237+H/"
NUM_ENTITIES = 14505
NUM_RELATIONS = 474

# name -> BetaE structure. Order defines the canonical type order.
STRUCTURES = {
    "1p": ("e", ("r",)),
    "2p": ("e", ("r", "r")),
    "3p": ("e", ("r", "r", "r")),
    "2i": (("e", ("r",)), ("e", ("r",))),
    "3i": (("e", ("r",)), ("e", ("r",)), ("e", ("r",))),
    "ip": ((("e", ("r",)), ("e", ("r",))), ("r",)),
    "pi": (("e", ("r", "r")), ("e", ("r",))),
    "2u": (("e", ("r",)), ("e", ("r",)), ("u",)),
    "up": ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)),
    "2in": (("e", ("r",)), ("e", ("r", "n"))),
    "3in": (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))),
    "inp": ((("e", ("r",)), ("e", ("r", "n"))), ("r",)),
    "pin": (("e", ("r", "r")), ("e", ("r", "n"))),
    "pni": (("e", ("r", "r", "n")), ("e", ("r",))),
    "4p": ("e", ("r", "r", "r", "r")),
    "4i": (("e", ("r",)), ("e", ("r",)), ("e", ("r",)), ("e", ("r",))),
}
STD_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up", "2in", "3in", "inp", "pin", "pni"]
HPLUS_TYPES = STD_TYPES + ["4p", "4i"]
TRAIN_TYPES = ["1p", "2p", "3p", "2i", "3i", "2in", "3in", "inp", "pin", "pni"]
# Fixed evaluation sample sizes per query type (all queries when fewer exist).
CAP = {"std": 2000, "hplus": 1000}
# Separate validation queries for model selection (disjoint from the scored sample).
SELECT_CAP = 1000

# Node/atom template of every query type: (head, relation column, tail, negated).
# Anchors are a0, a1, ...; existential variables v1, v2, ...; the target is x.
# Union types list one atom group per DNF branch.
TEMPLATES = {
    "1p": [[("a0", 0, "x", False)]],
    "2p": [[("a0", 0, "v1", False), ("v1", 1, "x", False)]],
    "3p": [[("a0", 0, "v1", False), ("v1", 1, "v2", False), ("v2", 2, "x", False)]],
    "4p": [[("a0", 0, "v1", False), ("v1", 1, "v2", False), ("v2", 2, "v3", False), ("v3", 3, "x", False)]],
    "2i": [[("a0", 0, "x", False), ("a1", 1, "x", False)]],
    "3i": [[("a0", 0, "x", False), ("a1", 1, "x", False), ("a2", 2, "x", False)]],
    "4i": [[("a0", 0, "x", False), ("a1", 1, "x", False), ("a2", 2, "x", False), ("a3", 3, "x", False)]],
    "ip": [[("a0", 0, "v1", False), ("a1", 1, "v1", False), ("v1", 2, "x", False)]],
    "pi": [[("a0", 0, "v1", False), ("v1", 1, "x", False), ("a1", 2, "x", False)]],
    "2u": [[("a0", 0, "x", False)], [("a1", 1, "x", False)]],
    "up": [[("a0", 0, "v1", False), ("v1", 2, "x", False)], [("a1", 1, "v1", False), ("v1", 2, "x", False)]],
    "2in": [[("a0", 0, "x", False), ("a1", 1, "x", True)]],
    "3in": [[("a0", 0, "x", False), ("a1", 1, "x", False), ("a2", 2, "x", True)]],
    "inp": [[("a0", 0, "v1", False), ("a1", 1, "v1", True), ("v1", 2, "x", False)]],
    "pin": [[("a0", 0, "v1", False), ("v1", 1, "x", False), ("a1", 2, "x", True)]],
    "pni": [[("a0", 0, "v1", False), ("v1", 1, "x", True), ("a1", 2, "x", False)]],
}


class SafeUnpickler(pickle.Unpickler):
    """Only rebuild builtin containers; refuse every other global."""

    ALLOWED = {("builtins", "set"), ("builtins", "frozenset"), ("collections", "defaultdict")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def load_pickle(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as f:
        return SafeUnpickler(io.BufferedReader(f)).load()


# ---------------------------------------------------------------- npy writer
_DESCR = {"i": "<i4", "h": "<i2"}


def npy_bytes(arr: array.array, shape: tuple) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("little-endian host required")
    n = 1
    for s in shape:
        n *= s
    if n != len(arr):
        raise ValueError(f"shape {shape} does not match {len(arr)} elements")
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (_DESCR[arr.typecode], repr(tuple(shape)))
    pad = 64 - ((10 + len(header) + 1) % 64)
    header = header + " " * pad + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + arr.tobytes()


def write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for key, (arr, shape) in arrays.items():
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, npy_bytes(arr, shape), compresslevel=1)
    tmp.replace(path)


def write_npy(path: Path, arr: array.array, shape: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(npy_bytes(arr, shape))


# ---------------------------------------------------------------- queries
def flatten(structure, query, anchors: list, rels: list) -> None:
    if structure == "e":
        anchors.append(int(query))
    elif structure == "r":
        rels.append(int(query))
    elif structure in ("n", "u"):
        return
    else:
        for s, q in zip(structure, query):
            flatten(s, q, anchors, rels)


def query_row(structure, query) -> list:
    anchors, rels = [], []
    flatten(structure, query, anchors, rels)
    return anchors + rels


def is_chain(structure) -> bool:
    return isinstance(structure, tuple) and len(structure) >= 1 and all(isinstance(x, str) for x in structure) and structure[0] in ("r", "n")


def execute(structure, query, adj, all_entities) -> set:
    """Exact set semantics of a BetaE query on a graph (dict (h, r) -> set(t))."""
    if structure[0] == "e":
        current = {int(query[0])}
        for s, r in zip(structure[1], query[1]):
            if s == "r":
                nxt = set()
                for e in current:
                    nxt |= adj.get((e, r), set())
                current = nxt
            elif s == "n":
                current = all_entities - current
        return current
    if len(structure) == 2 and is_chain(structure[1]):
        current = execute(structure[0], query[0], adj, all_entities)
        for s, r in zip(structure[1], query[1]):
            if s == "r":
                nxt = set()
                for e in current:
                    nxt |= adj.get((e, r), set())
                current = nxt
            elif s == "n":
                current = all_entities - current
        return current
    if structure[-1] == ("u",):
        out = set()
        for s, q in zip(structure[:-1], query[:-1]):
            out |= execute(s, q, adj, all_entities)
        return out
    parts = [execute(s, q, adj, all_entities) for s, q in zip(structure, query)]
    out = parts[0]
    for p in parts[1:]:
        out = out & p
    return out


def sample_queries(queries: set, bench: str, split: str, qtype: str, cap: int) -> list:
    ordered = sorted(queries)
    rng = random.Random(f"lvsa-gym|{bench}|{split}|{qtype}|v1")
    chosen = ordered if len(ordered) <= cap else rng.sample(ordered, cap)
    return sorted(chosen)


def csr(rows: list) -> tuple:
    indptr = array.array("i", [0])
    indices = array.array("i")
    for row in rows:
        indices.extend(sorted(int(x) for x in row))
        indptr.append(len(indices))
    return (indptr, (len(indptr),)), (indices, (len(indices),))


def read_triples(zf: zipfile.ZipFile, member: str) -> list:
    out = []
    for line in zf.read(member).decode().splitlines():
        if line.strip():
            h, r, t = line.split()
            out.append((int(h), int(r), int(t)))
    return out


def adjacency(triples) -> dict:
    adj = defaultdict(set)
    for h, r, t in triples:
        adj[(h, r)].add(t)
    return dict(adj)


def triples_array(triples) -> tuple:
    flat = array.array("i")
    for h, r, t in triples:
        flat.extend((h, r, t))
    return flat, (len(triples), 3)


def structure_names(queries: dict) -> dict:
    lookup = {v: k for k, v in STRUCTURES.items()}
    return {lookup[s]: s for s in queries if s in lookup}


def select_queries(queries: set, qtype: str) -> list:
    """Model-selection sample: validation queries outside the scored validation sample."""
    scored = set(sample_queries(queries, "std", "valid", qtype, CAP["std"]))
    rest = sorted(set(queries) - scored)
    rng = random.Random(f"lvsa-gym|std|select|{qtype}|v1")
    return rest if len(rest) <= SELECT_CAP else sorted(rng.sample(rest, SELECT_CAP))


def build_eval_split(zf, bench_dir, bench, split, types, adj_obs, all_entities, with_answers, selection=False):
    queries = load_pickle(zf, bench_dir + f"{split}-queries.pkl")
    names = structure_names(queries)
    easy = load_pickle(zf, bench_dir + f"{split}-easy-answers.pkl") if with_answers else None
    hard = load_pickle(zf, bench_dir + f"{split}-hard-answers.pkl") if with_answers else None
    arrays = {}
    counts = {}
    for qtype in types:
        structure = names[qtype]
        if selection:
            chosen = select_queries(queries[structure], qtype)
        else:
            chosen = sample_queries(queries[structure], bench, split, qtype, CAP[bench])
        rows = array.array("i")
        obs_rows, hard_rows, filt_rows = [], [], []
        for q in chosen:
            rows.extend(query_row(structure, q))
            obs = execute(structure, q, adj_obs, all_entities)
            obs_rows.append(obs)
            if with_answers:
                h = set(hard[q])
                e = set(easy.get(q, ()))
                if h & (e | obs):
                    raise RuntimeError(f"hard answers overlap filtered answers for {bench}/{split}/{qtype}")
                hard_rows.append(h)
                filt_rows.append(e | obs)
        width = len(rows) // len(chosen)
        arrays[f"q_{qtype}"] = (rows, (len(chosen), width))
        (ip, ips), (ix, ixs) = csr(obs_rows)
        arrays[f"obs_{qtype}_indptr"], arrays[f"obs_{qtype}_indices"] = (ip, ips), (ix, ixs)
        if with_answers:
            (ip, ips), (ix, ixs) = csr(hard_rows)
            arrays[f"hard_{qtype}_indptr"], arrays[f"hard_{qtype}_indices"] = (ip, ips), (ix, ixs)
            (ip, ips), (ix, ixs) = csr(filt_rows)
            arrays[f"filt_{qtype}_indptr"], arrays[f"filt_{qtype}_indices"] = (ip, ips), (ix, ixs)
        counts[qtype] = len(chosen)
        print(f"  {bench}/{split}/{qtype}: {len(chosen)} queries", flush=True)
    return arrays, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    zf = zipfile.ZipFile(args.zip)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    all_entities = frozenset(range(NUM_ENTITIES))

    train = read_triples(zf, HPLUS + "train.txt")
    valid = read_triples(zf, HPLUS + "valid.txt")
    train_set = set(train)
    if not all((t, r ^ 1, h) in train_set for h, r, t in train):
        raise RuntimeError("expected inverse relation r^1 for every training triple")
    write_npy(out / "train_triples.npy", *triples_array(train))
    write_npy(out / "valid_triples.npy", *triples_array(valid))
    adj_train = adjacency(train)
    adj_trainvalid = adjacency(train + valid)
    print(f"graph: {len(train)} train / {len(valid)} valid triples (inverses included)", flush=True)

    # Training queries with all answers (shared by both benchmarks).
    tq = load_pickle(zf, STD + "train-queries.pkl")
    ta = load_pickle(zf, STD + "train-answers.pkl")
    names = structure_names(tq)
    arrays = {}
    train_counts = {}
    for qtype in TRAIN_TYPES:
        structure = names[qtype]
        ordered = sorted(tq[structure])
        rows = array.array("i")
        answers = []
        for q in ordered:
            rows.extend(query_row(structure, q))
            answers.append(ta[q])
        width = len(rows) // len(ordered)
        arrays[f"q_{qtype}"] = (rows, (len(ordered), width))
        (ip, ips), (ix, ixs) = csr(answers)
        arrays[f"ans_{qtype}_indptr"], arrays[f"ans_{qtype}_indices"] = (ip, ips), (ix, ixs)
        train_counts[qtype] = len(ordered)
        print(f"  train/{qtype}: {len(ordered)} queries, {len(ix)} answers", flush=True)
    del tq, ta
    write_npz(out / "train_queries.npz", arrays)
    del arrays

    counts = {"train": train_counts}
    for bench, bench_dir, types in [("std", STD, STD_TYPES), ("hplus", HPLUS, HPLUS_TYPES)]:
        arrays, c = build_eval_split(zf, bench_dir, bench, "valid", types, adj_train, all_entities, True)
        write_npz(out / f"valid_{bench}.npz", arrays)
        counts[f"valid_{bench}"] = c
        arrays, c = build_eval_split(zf, bench_dir, bench, "test", types, adj_trainvalid, all_entities, False)
        write_npz(out / f"test_{bench}.npz", arrays)
        counts[f"test_{bench}"] = c

    arrays, c = build_eval_split(zf, STD, "std", "valid", STD_TYPES, adj_train, all_entities, True, selection=True)
    write_npz(out / "select_std.npz", arrays)
    counts["select_std"] = c

    meta = {
        "dataset": "FB15k-237 (BetaE query splits) and FB15k-237+H",
        "source": "https://github.com/april-tools/is-cqa-complex/releases/download/benchs-1.0/iscqa-compl-benchmarks-v1.1.zip",
        "num_entities": NUM_ENTITIES,
        "num_relations": NUM_RELATIONS,
        "inverse_relation": "relation ids come in inverse pairs: inverse(r) = r ^ 1",
        "query_types": {"std": STD_TYPES, "hplus": HPLUS_TYPES, "train": TRAIN_TYPES},
        "templates": {k: [[list(a) for a in branch] for branch in v] for k, v in TEMPLATES.items()},
        "query_columns": "anchors (a0, a1, ...) followed by relation ids in template column order",
        "counts": counts,
        "prediction_list_length": 1000,
        "trace_list_length": 100,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    digest = hashlib.sha256()
    for name in sorted(p.name for p in out.iterdir() if p.suffix in (".npz", ".npy", ".json")):
        digest.update(name.encode())
        digest.update((out / name).read_bytes())
    (out / ".complete").write_text(digest.hexdigest() + "\n")
    print("public data written to", out)


if __name__ == "__main__":
    main()
