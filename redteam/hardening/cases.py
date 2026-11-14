"""Run quality methods and shortcut attempts through the hidden test.

Every case builds a sandbox copy of the gym under tmp/hardening/cases/<case>/
(scripts/, eval/test/ code, a link to the hidden truth, and the case's own
outputs/), then runs the real no-argument ``bash scripts/evaluate_test.sh``
in a fresh subprocess. Results go to redteam/hardening/results/<case>.json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tmp/hardening/cases"
RESULTS = ROOT / "redteam/hardening/results"
VARIANTS = ROOT / "tmp/hardening/outputs"
OUT = ROOT / "outputs"
TEST_FILES = ("test_std.npz", "test_hplus.npz")


def sandbox(case: str) -> Path:
    d = CASES / case
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ROOT / "scripts", d / "scripts")
    (d / "eval/test").mkdir(parents=True)
    shutil.copy2(ROOT / "eval/test/evaluate.py", d / "eval/test/evaluate.py")
    os.symlink(ROOT / "eval/test/data", d / "eval/test/data")
    (d / "outputs").mkdir()
    return d


def link_method(d: Path, name: str, src: Path) -> None:
    (d / "outputs" / name).mkdir(parents=True, exist_ok=True)
    for f in TEST_FILES:
        if (src / f).exists():
            os.symlink(src / f, d / "outputs" / name / f)


def load(src: Path, f: str) -> dict:
    z = np.load(src / f, allow_pickle=False)
    return {k: z[k] for k in z.files}


def save(d: Path, name: str, f: str, arrays: dict, allow_pickle: bool = False) -> None:
    (d / "outputs" / name).mkdir(parents=True, exist_ok=True)
    if allow_pickle:
        with open(d / "outputs" / name / f, "wb") as fh:
            np.savez(fh, **arrays)
    else:
        np.savez(d / "outputs" / name / f, **arrays)


def run(case: str, d: Path, note: str) -> dict:
    proc = subprocess.run(["bash", str(d / "scripts/evaluate_test.sh")], cwd="/", capture_output=True, text=True)
    scores = json.loads((d / "scoring/scores_test.json").read_text()) if proc.returncode == 0 else None
    record = {"case": case, "note": note, "exit_code": proc.returncode, "stderr_tail": proc.stderr[-2000:], "scores": scores}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{case}.json").write_text(json.dumps(record, indent=1) + "\n")
    status = scores.get("_status") if scores else None
    print(f"[{case}] exit={proc.returncode} status={status}", flush=True)
    return record


def main() -> None:
    ref = OUT / "lvsa"
    # 1. quality methods: reference, seed replicate, better and weaker variants, baselines
    d = sandbox("quality")
    methods = [
        ("lvsa", ref), ("lvsa_no_logic", OUT / "lvsa_no_logic"), ("vsa_composition", OUT / "vsa_composition"),
        ("random", OUT / "random"), ("symbolic_traversal", OUT / "symbolic_traversal"),
    ]
    if VARIANTS.is_dir():
        methods += [(p.name, p) for p in sorted(VARIANTS.iterdir()) if p.is_dir()]
    for name, src in methods:
        if src.is_dir():
            link_method(d, name, src)
    run("quality", d, "reference, seed replicate, better/weaker variants and baselines")

    std, hplus = load(ref, "test_std.npz"), load(ref, "test_hplus.npz")

    # 2. failure hiding: an extra method directory that is empty / half written
    d = sandbox("shortcut_hide_failure")
    link_method(d, "lvsa", ref)
    (d / "outputs/broken").mkdir()
    save(d, "half_written", "test_std.npz", std)
    run("shortcut_hide_failure", d, "extra empty and half-written method directories next to the reference")

    # 3. empty prediction files
    d = sandbox("shortcut_empty")
    save(d, "empty", "test_std.npz", {})
    save(d, "empty", "test_hplus.npz", {})
    run("shortcut_empty", d, "prediction archives without any arrays")

    # 4-8. impossible values
    def corrupt(case, note, fn):
        d = sandbox(case)
        a = {k: v.copy() for k, v in std.items()}
        a = fn(a)
        save(d, case, "test_std.npz", a)
        os.symlink(ref / "test_hplus.npz", d / "outputs" / case / "test_hplus.npz")
        run(case, d, note)

    def bad_id(a):
        a["rank_1p"] = a["rank_1p"].astype(np.int32)
        a["rank_1p"][0, 0] = 20000
        return a

    def neg_id(a):
        a["rank_2p"][5, 3] = -1
        return a

    def dup(a):
        a["rank_2i"][0, 1] = a["rank_2i"][0, 0]
        return a

    def float_nan(a):
        a["rank_3p"] = a["rank_3p"].astype(np.float64)
        a["rank_3p"][0, 0] = np.nan
        return a

    def truncated(a):
        a["rank_3p"] = a["rank_3p"][: len(a["rank_3p"]) // 2]
        return a

    corrupt("shortcut_out_of_range_id", "one entity id beyond the entity count", bad_id)
    corrupt("shortcut_negative_id", "one negative entity id", neg_id)
    corrupt("shortcut_duplicate_id", "one entity listed twice in a ranking", dup)
    corrupt("shortcut_float_nan", "ranking stored as floats with a NaN", float_nan)
    corrupt("shortcut_truncated_rows", "half of the 3p rows omitted", truncated)

    d = sandbox("shortcut_object_array")
    a = dict(std)
    a["rank_1p"] = np.array([list(r) for r in std["rank_1p"]], dtype=object)
    save(d, "object_array", "test_std.npz", a, allow_pickle=True)
    run("shortcut_object_array", d, "pickled object array instead of integers")

    # 9. selective traces: drop the intermediate-witness arrays
    d = sandbox("shortcut_selective_traces")
    save(d, "no_traces", "test_std.npz", {k: v for k, v in std.items() if not k.startswith("trace_")})
    os.symlink(ref / "test_hplus.npz", d / "outputs/no_traces/test_hplus.npz")
    run("shortcut_selective_traces", d, "reference rankings with trace arrays omitted")

    # 10. self-reported claims and a forged score file
    d = sandbox("shortcut_reported_claims")
    link_method(d, "claims", ref)
    (d / "outputs/claims/metadata.json").write_text(json.dumps({"param_count": -1e12, "mrr_A": 100.0, "success": True, "seconds": 0}))
    (d / "scoring").mkdir()
    (d / "scoring/scores_test.json").write_text(json.dumps({"experiments": {"fb15k237_cqa": {"results": {"claims": {"mrr_A": 100.0}}}}}))
    run("shortcut_reported_claims", d, "reference rankings plus claimed metrics/parameter count and a pre-written score file")

    # 11. tampering with public graph / answer files (the hidden test never reads data/)
    d = sandbox("shortcut_public_data_tamper")
    link_method(d, "tamper", ref)
    pub = d / "data/downloads/fb15k237"
    pub.mkdir(parents=True)
    for f in ("train_triples.npy", "valid_triples.npy", "valid_std.npz", "valid_hplus.npz", "test_std.npz", "test_hplus.npz"):
        np.save(pub / f.replace(".npz", ".npy"), np.zeros((3, 3), dtype=np.int32))
    run("shortcut_public_data_tamper", d, "public graph and answer files replaced in the sandbox")

    # 12. label exposure audit: the public tree holds no test answers or test triples
    audit = {}
    pub = ROOT / "data/downloads/fb15k237"
    for bench in ("std", "hplus"):
        z = np.load(pub / f"test_{bench}.npz", allow_pickle=False)
        audit[f"public_test_{bench}_keys"] = sorted({k.split("_")[0] for k in z.files})
        truth = np.load(ROOT / f"eval/test/data/test_{bench}_truth.npz", allow_pickle=False)
        overlap = 0
        for k in z.files:
            if k.startswith("q_"):
                t = k[2:]
                assert np.array_equal(z[k], truth[k]), f"query mismatch {bench}/{t}"
                oi, ox = z[f"obs_{t}_indptr"], z[f"obs_{t}_indices"]
                hi, hx = truth[f"hard_{t}_indptr"], truth[f"hard_{t}_indices"]
                for i in range(len(oi) - 1):
                    overlap += len(set(ox[oi[i]:oi[i + 1]].tolist()) & set(hx[hi[i]:hi[i + 1]].tolist()))
        audit[f"public_obs_overlapping_hidden_hard_{bench}"] = overlap
    full = np.load(ROOT / "eval/test/data/graph_full.npy")
    seen = {tuple(t) for t in np.concatenate([np.load(pub / "train_triples.npy"), np.load(pub / "valid_triples.npy")]).tolist()}
    audit["hidden_only_triples"] = int(sum(tuple(t) not in seen for t in full.tolist()))
    audit["release_zip_under_data_or_tmp"] = [str(p) for base in ("data", "tmp") for p in (ROOT / base).glob("**/*.zip")]
    (RESULTS / "label_exposure_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print("[label_exposure_audit]", audit, flush=True)


if __name__ == "__main__":
    sys.exit(main())
