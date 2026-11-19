# Hardening report (iteration 11)

## Summary

The audited gym could not be repaired in place. Every hidden answer was
readable from the supplied synthetic graph. Negation changed none of the
hidden answer sets. Brute-force grounding scored perfectly, and the test
answers sat in `data/`. The gym now uses **the paper's own data**:

| Data | Role |
|---|---|
| Standard FB15k-237 complex-query benchmark (BetaE splits, 14 query types) | main test |
| FB15k-237+H (Gregucci et al., ICML 2025) | fully unobservable queries, plus 4p/4i |

Both come from the official public release. Only answers that need a fact
missing from the observed graph are scored.

The reference is rebuilt at the paper's scale and now follows the paper's
method:
- d = 1000;
- ComplEx stage 1 with the relation-prediction objective;
- the Eq. 25 in-batch term;
- LeakyReLU;
- one negator shared by the double-negation law;
- the squared-norm axiom loss.

Checkpoints are selected on a disjoint validation sample. Two compute jobs
were used:

| Job | ID | What it did |
|---|---|---|
| Run 1 | `cf5793a0-4ef` | first end-to-end reproduction; exposed the problems fixed in run 2 |
| Run 2 | `fb773907-08f` | final; produced every number in the "now" columns below |

Every number below comes from the real no-argument
`bash scripts/evaluate_test.sh`, each case in its own sandbox copy
(`redteam/hardening/cases.py`, results in `redteam/hardening/results/`,
summary in `redteam/hardening/summary.{json,md}`).

**Rank change** follows the ranking rule:

    Σ_exp weight × Σ_metric |coef| × direction × (value − reference) / spread

The spread is the population standard deviation of each metric across the
nine honest runs:
- the reference;
- its seed replicate;
- three better variants;
- the ablation;
- three baselines.

The future cohort is unknown, so the rank values are illustrative; their sign
and order are what matter.

## Do better results score higher?

Yes, and now by more than seed noise.

| Hidden test | std `mrr_A` | std `hits1_A` | std `mrr_An` | +H `mrr_A` | trace var MRR | Rank change |
|---|---:|---:|---:|---:|---:|---:|
| `lvsa_residual_alltypes` (better: residual Skolem module + all negation types + joint fine-tune) | 19.37 | 12.80 | 8.36 | 11.88 | 26.54 | **+0.403** |
| `lvsa_alltypes` (better: paper modules, negator on all negation types + joint fine-tune) | 19.13 | 12.56 | 8.22 | 11.50 | 25.77 | **+0.237** |
| `lvsa_residual` (better: residual Skolem module, paper curriculum) | 18.88 | 12.43 | 8.08 | 11.53 | 24.51 | **+0.140** |
| **`lvsa` (reference)** | **18.72** | **12.28** | **8.07** | **11.29** | **23.83** | 0 |
| `lvsa_seed2` (independent full-pipeline seed) | 18.80 | 12.42 | 8.09 | 11.23 | 23.01 | −0.024 |
| `lvsa_no_logic` (weaker: paper ablation, no L_NL) | 17.89 | 11.66 | 5.75 | 10.74 | 23.83 | **−0.588** |
| `vsa_composition` (baseline: training-free composition) | 17.23 | 11.68 | 4.47 | 11.50 | 25.43 | −0.664 |
| `symbolic_traversal` (baseline: brute-force grounding) | 7.15 | 4.28 | 3.16 | 6.20 | 0.40 | −4.596 |
| `random` | 0.06 | 0.01 | 0.04 | 0.07 | 0.20 | −7.458 |

**Seed noise.** Seed noise is 0.024 in rank (|Δ std `mrr_A`| = 0.08). Every
better variant beats it by 6–17×, and the weaker ablation falls 25× below it.
All runs score the same fixed queries, so these are paired comparisons.

**Before, in the old gym.** The redteam's audit cohort and normalization give:

| Variant | Rank change |
|---|---:|
| better (graph pretraining) | +2.24 |
| weaker (no path training) | −1.25 |
| more training | +0.10, below seed spread |
| seed spread | 0.13 |

The "better" variant there gained mostly because the old stage 1 did not
pretrain embeddings on the graph as the paper does; that is now restored in
the reference.

**Run 1 of this iteration** had no checkpoint selection. There the seed
replicate moved rank by +0.145 and the best variant by only +0.22. Selecting
stage checkpoints on a disjoint validation sample (`select_std.npz`) removed
most of that noise.

**The paper's claims in the paper's direction:**
- **Ablation.** Removing the logical-axiom loss lowers negated-query MRR:
  8.07 → 5.75 here, against paper Table VII's 10.9 → 7.6. Positive query
  types are unchanged.
- **Learned modules over training-free composition.** The learned Skolem
  module and negator beat training-free composition on the standard
  benchmark (+1.5 `mrr_A`, +3.6 `mrr_An`).
- **Not reproduced:** the comparison with the paper's trained Skolemization
  baselines (BetaE, ConE, CLMPT, …). Those are kept only as the paper's
  numbers in `reference.json`.

## Major shortcuts: fix, then score before → after

"Before" is REDTEAM.md (old gym, rank change A); "after" is this
iteration's hidden test.

| Shortcut | Fix | Before | After |
|---|---|---:|---:|
| Copy visible test labels | Test answers and test triples exist only in `eval/test/data/`. That folder is built by `eval/test/prepare_hidden.py` from the release zip, which `download.sh` deletes after preparation. Public test files contain only queries and observed-graph answers, which overlap hidden answers 0 times (`results/label_exposure_audit.json`). | +6.45 (perfect) | impossible in-tree |
| Rebuild answers from the public generator and seed | No generator any more. Answers depend on 40,876 held-out FB15k-237 triples. | +6.45 | impossible in-tree |
| Brute-force grounding of every variable | Real incomplete graph. Only answers needing a missing fact are scored, and entities reachable on the observed graph are ignored. | +6.45 (perfect, 0.6 s) | **−4.60** (std MRR 7.2) |
| Claim zero / negative parameter count | Parameter count is no longer read or scored; only rankings are read. | +0.45 / +1e−7 | **0.00** |
| Selective trace omission | Fixed-shape trace arrays. Missing traces give worst trace values. Trace slots and denominators are fixed by the evaluator. | +1.25 | **−0.81** |
| Replace the graph the evaluator trusts | The hidden evaluator reads only `eval/test/data/` and never `data/`. | +1.25 | **0.00** (tamper case) |
| Replace ground truth | Same as above. | +4.93 | **0.00** |
| Hide a failed method from discovery | Both scripts score every directory under `outputs/`, with no manifest. Missing files give worst values. | 0 (tie) | **−7.49** for the empty dir, −1.94 for a half-written one |
| Empty / null / non-finite outputs | Every metric of every experiment always gets a value. Invalid arrays (float, NaN, object, out-of-range, negative or repeated ids, wrong row count) fail the whole method with 0 everywhere. Output JSON is strict (`allow_nan=False`). | size metric absent; crash; `-Infinity` | **−7.49** each |
| Forged scores, flags or timing | Ignored; scores are recomputed from rankings. | fails | **0.00** |
| **New:** observed-graph lookup for intermediate traces | Found in run 1: an observed-neighbour degree lookup scored 68.5 trace MRR vs 47.8 for the reference. Pairing its traces with the reference's answers also raised path P@1 (study in `redteam/hardening/run1/trace_definition_study.*`). Traceability now ignores intermediates reachable on the observed graph and credits only inferred intermediates on a complete-graph chain to a non-trivial answer. | +(trace MRR 68.5 vs 47.8) | **−0.79** (lookup traces), **−0.03** (lookup entity forced to rank 1) |
| **New:** drop graph-reachable entities from negated-query rankings | Exploits the gap between the benchmark's easy answers and observed answers on negated queries. The evaluator now ignores every observed answer, so removing them cannot change a score. | not tested | neutral by construction |

## Shortcuts and alternatives left open

- **Composition traces next to the reference answers: +0.055.** This
  replaces only the intermediate rankings with those of training-free
  composition (`results/shortcut_composition_traces_offline.json`).
  - It beats seed noise but sits far below every honest improvement
    (+0.14 to +0.40).
  - It is a learned model's output, not a hand-written rule, so the metric
    cannot tell it apart without tying traces to the answer computation.
    Answer-conditioned definitions were themselves gameable by a lookup in
    run 1's study.
  - `CONSTRAINTS.md` requires traces to come from the answering model, and
    the grader can check this.
  - The underlying fact is real: on this protocol, the paper's Skolem MLP
    yields slightly worse intermediate rankings than plain binding
    (23.8 vs 25.4).
- **Per-type switch to training-free composition (ip/pi/up/4p): +0.192.**
  This is an honest alternative, not a loophole. The paper itself reports
  LVSA weak on ip/inp when its Skolem module gets a superposed input. A
  better Skolem module (the residual variant) captures the same gain.
- **Neural grounding (QTO / CQD-Beam style).** The paper reports QTO above
  LVSA on standard FB15k-237 (27.1 vs 22.4) but below it on FB15k-237+H
  (9.7 vs 12.2). I did not implement it, so I could not measure whether it
  outranks the reference under the 0.55 / 0.25 weights. `DIRECTION.md` and
  `CONSTRAINTS.md` rule out exhaustive grounding of existential variables
  (the cost the paper exists to avoid); that rule is left to the grader.
- **Public benchmark files online.** The test answers of the public benchmark
  are downloadable from the internet. Fetching them is forbidden in
  `CONSTRAINTS.md` and left to the grader. Training on validation triples is
  likewise forbidden there and left to the grader (expected gain is small:
  ~6% more facts).

## Reference score before and after

| | Benchmark | Reference | Paper |
|---|---|---:|---:|
| **Before** | synthetic | MRR 0.506 (fraction) | — |
| **After** | FB15k-237 standard | `mrr_A` 18.72, `hits1_A` 12.28 | 22.4 / 15.4 |
| **After** | FB15k-237+H | `mrr_A` 11.29 | 12.2 |
| **After** | traceability (this gym's protocol) | 23.83 / 14.27 / 1.43 | not on this scale |

Before, MRR scored only the first correct answer, and parameters were
self-reported.

**Train side (public validation):** std 17.12, +H 10.27, trace 22.49.
Selection uses a separate validation sample, so these are not tuned on.

**Remaining gap to the paper (−16% on standard).** Where it sits:
- **Training-free types.** 2i / 3i / 2u depend only on the stage-1
  embeddings: 37.0 / 50.1 / 17.7 vs the paper's 41.7 / 55.1 / 22.5. 1p is
  close (48.3 vs 49.2), so the stage-1 embeddings are the likely source; the
  paper used CQD's pretrained ComplEx.
- **Skolem-dependent types** (2p 11.6 vs 15.8, ip 13.6 vs 19.0).
- **Negated types.**

Epochs, MLP width and the relation-prediction weight are not given in the
paper (see `briefing/method.md`). The paper's numbers are unchanged in
`reference.json`.

## Headroom

No scored metric is at its bound: the reference sits at 9–50% MRR per type
and ~24% trace MRR, and three variants already score above it.

Two scored experiments were added from where the paper's method falls short:
- **`fb15k237_unobservable`** (weight 0.25): the paper's fully unobservable
  benchmark, including 4p and 4i. The reference reaches 11.3 MRR here, and
  only 2.7 on 4p.
- **Filtered traceability** (weight 0.20): only inferred intermediates count.

The baselines behave as expected:
- Brute-force grounding (−4.60) and random (−7.46) are far below.
- Training-free composition is below overall (−0.66), but not on every
  experiment: it edges the reference on +H (11.50 vs 11.29) and on traces.

## Weights and coefficients

| Experiment | Weight | Why |
|---|---:|---|
| `fb15k237_cqa` | 0.55 | paper Tables IV/V and VII, the main claim |
| `fb15k237_unobservable` | 0.25 | Table VI and Fig. 6, the paradigm-shift claim |
| `fb15k237_traceability` | 0.20 | Table VIII, logic fidelity |

- All coefficients are positive.
- The path-level trace metric has coefficient 0.25 because it is
  low-valued and noisy.
- The parameter-count metric is removed: it can't be verified from outputs.
- Inference throughput (Sec. VI-D) is recorded under `_paper_efficiency` but
  not scored: it is wall-clock based and cannot be recomputed from rankings.

## Not fixed / stated plainly

- **Protocol deviations from the paper's evaluation:**
  - a fixed sample of 2000 (standard) and ≤1000 (+H) test queries per type;
  - 1000-entity lists, where unlisted answers count 0; truncation is
    negligible because observed answers may be omitted;
  - observed answers of negated queries are ignored;
  - the traceability definition differs (reasons above).
- **The reproduction stays 16% below the paper on the standard benchmark,**
  and the paper's trained Skolemization baselines were not rerun.
- **Hidden truth handling.** The hidden test truth is gitignored and
  regenerated by `scripts/download.sh` (internet required). The hidden-test
  job uses the on-disk copy.
- **Historical redteam cases.** The `redteam/<case>/` folders from the
  previous audit target the retired synthetic gym and no longer run against
  this one. This iteration's cases are in `redteam/hardening/`.
