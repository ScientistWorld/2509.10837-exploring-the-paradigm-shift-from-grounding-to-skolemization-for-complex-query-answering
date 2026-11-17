"""Summarise hidden-test results of the hardening cases.

Rank change of a method relative to the reference ``lvsa``:
    sum_exp weight_exp * sum_metric |coef| * dir * (value - value_ref) / spread_metric
with dir from ``higher_is_better`` and spread = population standard deviation of
the metric across the honest cohort (reference, seed replicate, variants and
baselines). This mirrors the ranking rule; the future cohort is unknown, so the
numbers are illustrative.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "redteam/hardening/results"
REF = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
REFERENCE = "lvsa"
COHORT = ["lvsa", "lvsa_seed2", "lvsa_alltypes", "lvsa_residual", "lvsa_residual_alltypes", "lvsa_no_logic",
          "vsa_composition", "random", "symbolic_traversal"]


def flat(scores: dict, method: str) -> dict:
    out = {}
    for exp, d in scores["experiments"].items():
        for m, v in d["results"].get(method, {}).items():
            out[(exp, m)] = v
    return out


def main() -> None:
    quality = json.loads((RESULTS / "quality.json").read_text())["scores"]
    methods = sorted(quality["experiments"]["fb15k237_cqa"]["results"])
    table = {m: flat(quality, m) for m in methods}
    cohort = [m for m in COHORT if m in table]
    spreads = {}
    for exp, e in REF.items():
        for metric in e["metrics"]:
            vals = [table[m].get((exp, metric)) for m in cohort]
            vals = [v for v in vals if v is not None]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            spreads[(exp, metric)] = sd

    def rank_delta(values: dict) -> float:
        total = 0.0
        ref = table[REFERENCE]
        for exp, e in REF.items():
            for metric, spec in e["metrics"].items():
                sd = spreads[(exp, metric)]
                if sd <= 0 or (exp, metric) not in values:
                    continue
                sign = 1.0 if spec["higher_is_better"] else -1.0
                total += e["weight"] * abs(spec["coefficient"]) * sign * (values[(exp, metric)] - ref[(exp, metric)]) / sd
        return total

    summary = {"rank_rule": __doc__.strip(), "cohort": cohort, "methods": {}, "cases": {}}
    keys = [("fb15k237_cqa", "mrr_A"), ("fb15k237_cqa", "mrr_An"), ("fb15k237_cqa", "hits1_A"),
            ("fb15k237_unobservable", "mrr_A"), ("fb15k237_traceability", "trace_var_mrr"),
            ("fb15k237_traceability", "trace_path_p1")]
    for m in methods:
        summary["methods"][m] = {"rank_change": round(rank_delta(table[m]), 4), **{f"{e}.{k}": table[m][(e, k)] for e, k in keys}}
    if "lvsa_seed2" in table:
        summary["seed_noise"] = {
            "rank_change_seed2_vs_reference": summary["methods"]["lvsa_seed2"]["rank_change"],
            "abs_metric_differences": {f"{e}.{k}": round(abs(table["lvsa_seed2"][(e, k)] - table[REFERENCE][(e, k)]), 4) for (e, k) in table[REFERENCE]},
        }
    for path in sorted(list(RESULTS.glob("shortcut_*.json")) + list(RESULTS.glob("alt_*.json"))):
        rec = json.loads(path.read_text())
        entry = {"note": rec["note"], "exit_code": rec["exit_code"]}
        if rec["scores"]:
            entry["status"] = rec["scores"].get("_status")
            entry["methods"] = {}
            for m in rec["scores"]["experiments"]["fb15k237_cqa"]["results"]:
                vals = flat(rec["scores"], m)
                entry["methods"][m] = {"rank_change": round(rank_delta(vals), 4), **{f"{e}.{k}": vals[(e, k)] for e, k in keys}}
        summary["cases"][rec["case"]] = entry
    audit = RESULTS / "label_exposure_audit.json"
    if audit.exists():
        summary["label_exposure_audit"] = json.loads(audit.read_text())
    (ROOT / "redteam/hardening/summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    lines = ["| method | rank change | std mrr_A | std mrr_An | std hits1_A | +H mrr_A | trace var MRR | trace path P@1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, v in sorted(summary["methods"].items(), key=lambda kv: -kv[1]["rank_change"]):
        lines.append(f"| {m} | {v['rank_change']:+.3f} | " + " | ".join(f"{v[f'{e}.{k}']:.2f}" for e, k in keys) + " |")
    lines += ["", "| shortcut case | method | status | rank change | std mrr_A | trace var MRR |", "|---|---|---|---:|---:|---:|"]
    for case, entry in summary["cases"].items():
        for m, v in (entry.get("methods") or {}).items():
            lines.append(f"| {case} | {m} | {entry['status'].get(m, '')} | {v['rank_change']:+.3f} | {v['fb15k237_cqa.mrr_A']:.2f} | {v['fb15k237_traceability.trace_var_mrr']:.2f} |")
    (ROOT / "redteam/hardening/summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
