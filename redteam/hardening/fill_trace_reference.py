"""Fill the traceability reference values in scoring/reference.json.

The paper's Table VIII uses a satisfiability criterion that a graph lookup
saturates, so the gym scores intermediate witnesses against non-trivial
answers instead (see the experiment's notes). No paper number exists on that
scale; the reference entry therefore records this gym's reproduced value of
the reference method from scoring/scores_test.json. Only placeholders (null)
are filled, so reruns never overwrite recorded values.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ref_path = ROOT / "scoring/reference.json"
ref = json.loads(ref_path.read_text())
scores = json.loads((ROOT / "scoring/scores_test.json").read_text())
entry = ref["experiments"]["fb15k237_traceability"]["results"]["lvsa"]
got = scores["experiments"]["fb15k237_traceability"]["results"]["lvsa"]
changed = False
for k in ("trace_var_mrr", "trace_var_p1", "trace_path_p1"):
    if entry.get(k) is None:
        entry[k] = round(got[k], 2)
        changed = True
if changed:
    ref_path.write_text(json.dumps(ref, indent=1) + "\n")
print("traceability reference:", entry)
