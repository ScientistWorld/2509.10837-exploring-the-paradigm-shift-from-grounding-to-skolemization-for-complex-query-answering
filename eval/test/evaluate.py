"""Hidden test evaluator for the FB15k-237 complex-query gym.

Scores every method directory under ``outputs/`` from its raw ranked entity
lists alone. Nothing a method writes besides those integer arrays is read:
no claimed metric, flag, parameter count or timing.

Inputs per method ``outputs/<method>/``:
  test_std.npz    rank_<type>  [n_type, K] entity ids, best first (14 types)
                  trace_2p_v1, trace_3p_v1, trace_3p_v2  [n, KT] entity ids
  test_hplus.npz  rank_<type>  [n_type, K] (16 types)

Ground truth lives only in ``eval/test/data/`` (built by prepare_hidden.py):
the scored answers, the ignored entities, the complete graph and the graph
observed at test time.
Ranks are filtered: an answer's rank counts only the non-answer entities
listed above it; entities that are benchmark easy answers or that are already
reachable on the observed graph are ignored wherever they appear. Answers not
listed count as reciprocal rank 0.

Failure rules: a file that is present but unreadable, or any array with a
wrong shape, non-integer dtype, out-of-range or repeated id marks the method
as failed, and every metric of every experiment gets its worst value (0). A
missing file or missing ranking array gives worst values for that experiment;
missing trace arrays give worst trace values.

Standard library only.
"""

from __future__ import annotations

import argparse
import array
import ast
import json
import math
import struct
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STD_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up", "2in", "3in", "inp", "pin", "pni"]
EPFO_TYPES = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
NEG_TYPES = ["2in", "3in", "inp", "pin", "pni"]
HPLUS_TYPES = STD_TYPES + ["4p", "4i"]
EXP_STD = "fb15k237_cqa"
EXP_HPLUS = "fb15k237_unobservable"
EXP_TRACE = "fb15k237_traceability"
STD_METRICS = ["mrr_A", "mrr_Ap", "mrr_An", "hits1_A", "hits1_Ap", "hits1_An", "hits3_A", "hits10_A"] + [f"mrr_{t}" for t in STD_TYPES]
HPLUS_METRICS = ["mrr_A", "mrr_Ap", "mrr_An", "hits1_A", "hits10_A"] + [f"mrr_{t}" for t in HPLUS_TYPES]
TRACE_METRICS = ["trace_var_mrr", "trace_var_p1", "trace_path_p1"]
TRACE_KEYS = {"2p": ["trace_2p_v1"], "3p": ["trace_3p_v1", "trace_3p_v2"]}
SPLIT_FILES = {"std": "test_std.npz", "hplus": "test_hplus.npz"}


class InvalidPrediction(Exception):
    pass


# ----------------------------------------------------------------- npy reading
_TYPECODES = {
    "|i1": ("b", 1), "<i1": ("b", 1), "|u1": ("B", 1), "<u1": ("B", 1),
    "<i2": ("h", 2), "<u2": ("H", 2), "<i4": ("i", 4), "<u4": ("I", 4),
    "<i8": ("q", 8), "<u8": ("Q", 8),
}


def parse_npy(raw: bytes, name: str):
    if raw[:6] != b"\x93NUMPY":
        raise InvalidPrediction(f"{name}: not a .npy array")
    major = raw[6]
    if major == 1:
        hlen, off = struct.unpack("<H", raw[8:10])[0], 10
    elif major in (2, 3):
        hlen, off = struct.unpack("<I", raw[8:12])[0], 12
    else:
        raise InvalidPrediction(f"{name}: unsupported npy version")
    try:
        header = ast.literal_eval(raw[off : off + hlen].decode("latin1"))
        descr, fortran, shape = header["descr"], bool(header["fortran_order"]), tuple(int(s) for s in header["shape"])
    except Exception as exc:  # noqa: BLE001
        raise InvalidPrediction(f"{name}: bad npy header ({exc})")
    if descr not in _TYPECODES:
        raise InvalidPrediction(f"{name}: dtype {descr!r} is not an integer type")
    code, size = _TYPECODES[descr]
    count = 1
    for s in shape:
        count *= s
    body = raw[off + hlen :]
    if len(body) != count * size:
        raise InvalidPrediction(f"{name}: payload size does not match shape {shape}")
    arr = array.array(code)
    if arr.itemsize != size:
        raise InvalidPrediction(f"{name}: platform integer size mismatch")
    arr.frombytes(body)
    if fortran and len(shape) == 2:
        n, k = shape
        arr = array.array(code, (arr[j * n + i] for i in range(n) for j in range(k)))
    return arr, shape


def read_npz(path: Path) -> dict:
    """Return {key: (raw_bytes_loader)}; raises InvalidPrediction on corrupt archives."""
    try:
        zf = zipfile.ZipFile(path)
        names = zf.namelist()
    except Exception as exc:  # noqa: BLE001
        raise InvalidPrediction(f"{path.name}: unreadable archive ({exc})")
    out = {}
    for member in names:
        if not member.endswith(".npy"):
            continue
        key = member[:-4]
        try:
            raw = zf.read(member)
        except Exception as exc:  # noqa: BLE001
            raise InvalidPrediction(f"{path.name}/{member}: unreadable ({exc})")
        out[key] = parse_npy(raw, f"{path.name}/{member}")
    return out


def read_npy_file(path: Path):
    return parse_npy(path.read_bytes(), path.name)


def rows_of(arr, shape, n_entities: int, expected_rows: int, name: str):
    if len(shape) != 2:
        raise InvalidPrediction(f"{name}: expected a 2-D array, got shape {shape}")
    n, k = shape
    if n != expected_rows:
        raise InvalidPrediction(f"{name}: expected {expected_rows} rows, got {n}")
    if k < 1 or k > n_entities:
        raise InvalidPrediction(f"{name}: list length {k} outside [1, {n_entities}]")
    rows = []
    for i in range(n):
        row = arr[i * k : (i + 1) * k]
        if min(row) < 0 or max(row) >= n_entities:
            raise InvalidPrediction(f"{name}: entity id out of range in row {i}")
        if len(set(row)) != k:
            raise InvalidPrediction(f"{name}: repeated entity id in row {i}")
        rows.append(row)
    return rows


# ----------------------------------------------------------------- truth
def csr_sets(arrays: dict, prefix: str) -> list:
    indptr, _ = arrays[prefix + "_indptr"]
    indices, _ = arrays[prefix + "_indices"]
    return [set(indices[indptr[i] : indptr[i + 1]]) for i in range(len(indptr) - 1)]


def load_truth(data_dir: Path, bench: str, types: list) -> dict:
    arrays = read_npz(data_dir / f"test_{bench}_truth.npz")
    truth = {}
    for t in types:
        q, shape = arrays[f"q_{t}"]
        width = shape[1]
        truth[t] = {
            "queries": [list(q[i * width : (i + 1) * width]) for i in range(shape[0])],
            "hard": csr_sets(arrays, f"hard_{t}"),
            "filt": csr_sets(arrays, f"filt_{t}"),
        }
    return truth


def load_graph(data_dir: Path, name: str = "graph_full.npy") -> dict:
    arr, shape = read_npy_file(data_dir / name)
    adj = defaultdict(set)
    for i in range(shape[0]):
        adj[(arr[3 * i], arr[3 * i + 1])].add(arr[3 * i + 2])
    return dict(adj)


# ----------------------------------------------------------------- metrics
def rank_metrics(rows: list, hard: list, filt: list) -> dict:
    mrr = h1 = h3 = h10 = 0.0
    for row, answers, ignored in zip(rows, hard, filt):
        remaining = len(answers)
        negatives = 0
        rr = c1 = c3 = c10 = 0.0
        for e in row:
            if e in answers:
                rank = negatives + 1
                rr += 1.0 / rank
                c1 += rank <= 1
                c3 += rank <= 3
                c10 += rank <= 10
                remaining -= 1
                if remaining == 0:
                    break
            elif e not in ignored:
                negatives += 1
        n = len(answers)
        mrr += rr / n
        h1 += c1 / n
        h3 += c3 / n
        h10 += c10 / n
    q = max(1, len(rows))
    return {"mrr": 100 * mrr / q, "hits1": 100 * h1 / q, "hits3": 100 * h3 / q, "hits10": 100 * h10 / q}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def first_unfiltered(row, ignored):
    for e in row:
        if e not in ignored:
            return e
    return None


def trace_metrics(truth: dict, preds: dict, adj: dict, observed: dict, n_entities: int) -> dict:
    """Intermediate-witness metrics on 2p/3p queries.

    Candidates already reachable from the anchor on the observed graph (V1:
    observed r1-neighbours; V2: observed two-hop entities) are ignored wherever
    they appear in a trace list. A variable slot is scored when at least one
    unobserved entity lies on a complete-graph chain from the anchor to a
    non-trivial answer; the path metric uses the top non-ignored intermediates
    and the top non-ignored answer.
    """
    empty = set()
    slot_rr, slot_p1, path2, path3 = [], [], [], []

    def rr_skip(row, valid, skip):
        pos = 0
        for e in row:
            if e in skip:
                continue
            pos += 1
            if e in valid:
                return 1.0 / pos
        return 0.0

    t2 = truth["2p"]
    trace = rows_of(*preds["trace_2p_v1"], n_entities, len(t2["queries"]), "trace_2p_v1")
    answers = rows_of(*preds["rank_2p"], n_entities, len(t2["queries"]), "rank_2p")
    for q, hard, ignored, tr, ans in zip(t2["queries"], t2["hard"], t2["filt"], trace, answers):
        a0, r0, r1 = q
        first = adj.get((a0, r0), empty)
        seen1 = observed.get((a0, r0), empty)
        valid = {e for e in first if e not in seen1 and adj.get((e, r1), empty) & hard}
        top1 = first_unfiltered(tr, seen1)
        if valid:
            slot_rr.append(rr_skip(tr, valid, seen1))
            slot_p1.append(1.0 if top1 in valid else 0.0)
        top = first_unfiltered(ans, ignored)
        path2.append(1.0 if top in hard and top1 in first and top in adj.get((top1, r1), empty) else 0.0)

    t3 = truth["3p"]
    tr1 = rows_of(*preds["trace_3p_v1"], n_entities, len(t3["queries"]), "trace_3p_v1")
    tr2 = rows_of(*preds["trace_3p_v2"], n_entities, len(t3["queries"]), "trace_3p_v2")
    answers = rows_of(*preds["rank_3p"], n_entities, len(t3["queries"]), "rank_3p")
    for q, hard, ignored, v1, v2, ans in zip(t3["queries"], t3["hard"], t3["filt"], tr1, tr2, answers):
        a0, r0, r1, r2 = q
        first = adj.get((a0, r0), empty)
        seen1 = observed.get((a0, r0), empty)
        second, seen2 = set(), set()
        for e in first:
            second |= adj.get((e, r1), empty)
        for e in seen1:
            seen2 |= observed.get((e, r1), empty)
        valid2 = {e for e in second if adj.get((e, r2), empty) & hard}
        valid1 = {e for e in first if adj.get((e, r1), empty) & valid2}
        for row, val, seen in ((v1, valid1, seen1), (v2, valid2, seen2)):
            unobserved = val - seen
            if unobserved:
                slot_rr.append(rr_skip(row, unobserved, seen))
                slot_p1.append(1.0 if first_unfiltered(row, seen) in unobserved else 0.0)
        top = first_unfiltered(ans, ignored)
        top1, top2 = first_unfiltered(v1, seen1), first_unfiltered(v2, seen2)
        ok = top in hard and top1 in first and top2 in adj.get((top1, r1), empty) and top in adj.get((top2, r2), empty)
        path3.append(1.0 if ok else 0.0)
    return {
        "trace_var_mrr": 100 * mean(slot_rr),
        "trace_var_p1": 100 * mean(slot_p1),
        "trace_path_p1": 100 * mean([mean(path2), mean(path3)]),
    }

def worst(metrics: list) -> dict:
    return {m: 0.0 for m in metrics}


def score_benchmark(truth: dict, preds: dict, types: list, n_entities: int) -> dict:
    per_type = {}
    for t in types:
        key = f"rank_{t}"
        if key not in preds:
            raise KeyError(key)
        rows = rows_of(*preds[key], n_entities, len(truth[t]["queries"]), key)
        per_type[t] = rank_metrics(rows, truth[t]["hard"], truth[t]["filt"])
    return per_type


def validate_all(preds: dict, truth: dict, n_entities: int, file_name: str) -> None:
    """Every array that is present must be a valid ranking table."""
    for key, (arr, shape) in preds.items():
        if key.startswith("rank_"):
            t = key[len("rank_") :]
            if t not in truth:
                raise InvalidPrediction(f"{file_name}: unknown array {key}")
            rows_of(arr, shape, n_entities, len(truth[t]["queries"]), f"{file_name}/{key}")
        elif key.startswith("trace_"):
            t = key.split("_")[1]
            if key not in TRACE_KEYS.get(t, []):
                raise InvalidPrediction(f"{file_name}: unknown array {key}")
            rows_of(arr, shape, n_entities, len(truth[t]["queries"]), f"{file_name}/{key}")


def score_method(method_dir: Path, truth: dict, adj: dict, observed: dict, n_entities: int):
    status = []
    preds = {}
    for bench, fname in SPLIT_FILES.items():
        path = method_dir / fname
        if path.is_file():
            preds[bench] = read_npz(path)
            validate_all(preds[bench], truth[bench], n_entities, fname)
        else:
            status.append(f"missing {fname}")

    results = {EXP_STD: worst(STD_METRICS), EXP_HPLUS: worst(HPLUS_METRICS), EXP_TRACE: worst(TRACE_METRICS)}
    if "std" in preds:
        try:
            pt = score_benchmark(truth["std"], preds["std"], STD_TYPES, n_entities)
            r = {
                "mrr_A": mean(pt[t]["mrr"] for t in STD_TYPES),
                "mrr_Ap": mean(pt[t]["mrr"] for t in EPFO_TYPES),
                "mrr_An": mean(pt[t]["mrr"] for t in NEG_TYPES),
                "hits1_A": mean(pt[t]["hits1"] for t in STD_TYPES),
                "hits1_Ap": mean(pt[t]["hits1"] for t in EPFO_TYPES),
                "hits1_An": mean(pt[t]["hits1"] for t in NEG_TYPES),
                "hits3_A": mean(pt[t]["hits3"] for t in STD_TYPES),
                "hits10_A": mean(pt[t]["hits10"] for t in STD_TYPES),
            }
            r.update({f"mrr_{t}": pt[t]["mrr"] for t in STD_TYPES})
            results[EXP_STD] = r
        except KeyError as exc:
            status.append(f"test_std.npz lacks {exc.args[0]}")
        if all(k in preds["std"] for k in ("trace_2p_v1", "trace_3p_v1", "trace_3p_v2", "rank_2p", "rank_3p")):
            results[EXP_TRACE] = trace_metrics(truth["std"], preds["std"], adj, observed, n_entities)
        else:
            status.append("test_std.npz lacks trace arrays")
    if "hplus" in preds:
        try:
            pt = score_benchmark(truth["hplus"], preds["hplus"], HPLUS_TYPES, n_entities)
            r = {
                "mrr_A": mean(pt[t]["mrr"] for t in STD_TYPES),
                "mrr_Ap": mean(pt[t]["mrr"] for t in EPFO_TYPES),
                "mrr_An": mean(pt[t]["mrr"] for t in NEG_TYPES),
                "hits1_A": mean(pt[t]["hits1"] for t in STD_TYPES),
                "hits10_A": mean(pt[t]["hits10"] for t in STD_TYPES),
            }
            r.update({f"mrr_{t}": pt[t]["mrr"] for t in HPLUS_TYPES})
            results[EXP_HPLUS] = r
        except KeyError as exc:
            status.append(f"test_hplus.npz lacks {exc.args[0]}")
    return results, ("ok" if not status else "partial: " + "; ".join(status))


def clean(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 100:
        raise InvalidPrediction(f"impossible metric value {value!r}")
    return round(float(value), 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=Path("scoring/scores_test.json"))
    args = parser.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    n_entities = int(meta["num_entities"])
    truth = {"std": load_truth(args.data, "std", STD_TYPES), "hplus": load_truth(args.data, "hplus", HPLUS_TYPES)}
    adj = load_graph(args.data, "graph_full.npy")
    observed = load_graph(args.data, "graph_observed.npy")

    methods = sorted(p.name for p in args.outputs.iterdir() if p.is_dir() and not p.name.startswith(".")) if args.outputs.is_dir() else []
    payload = {"experiments": {EXP_STD: {"results": {}}, EXP_HPLUS: {"results": {}}, EXP_TRACE: {"results": {}}}, "_status": {}}
    for name in methods:
        try:
            results, status = score_method(args.outputs / name, truth, adj, observed, n_entities)
            results = {exp: {m: clean(v) for m, v in vals.items()} for exp, vals in results.items()}
        except (InvalidPrediction, Exception) as exc:  # noqa: BLE001 - any failure scores as failed
            results = {EXP_STD: worst(STD_METRICS), EXP_HPLUS: worst(HPLUS_METRICS), EXP_TRACE: worst(TRACE_METRICS)}
            status = f"failed: {type(exc).__name__}: {exc}"
        for exp, vals in results.items():
            payload["experiments"][exp]["results"][name] = vals
        payload["_status"][name] = status
        print(f"{name}: {status}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({exp: {m: {k: v for k, v in vals.items() if k in ('mrr_A', 'trace_var_mrr', 'trace_path_p1')} for m, vals in d['results'].items()} for exp, d in payload['experiments'].items()}, indent=1))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
