# Evaluation

## Data

`scripts/download.sh` fetches the official release of the FB15k-237 query
benchmarks and prepares `data/downloads/fb15k237/`:

| file | content |
|---|---|
| `meta.json` | entity / relation counts, query types, query templates, list lengths |
| `train_triples.npy`, `valid_triples.npy` | observed graph `[n, 3]` (head, relation, tail); relation ids come in inverse pairs, `inverse(r) = r ^ 1` |
| `train_queries.npz` | training queries of 10 types (`q_<type>`) with all answers (`ans_<type>_indptr/indices`, CSR) |
| `valid_std.npz`, `valid_hplus.npz` | validation queries (`q_<type>`), their scored answers (`hard_*`), the entities ignored when ranking (`filt_*`) and the entities reachable on the observed training graph (`obs_*`) |
| `test_std.npz`, `test_hplus.npz` | test queries (`q_<type>`) and the entities reachable on the observed train+valid graph (`obs_*`); no answers |

A query row lists its anchor entities followed by its relation ids, in the
column order of the type's template in `meta.json` (`(head, relation column,
tail, negated)` atoms; anchors `a0..`, variables `v1..`, answer `x`; union types
have one atom list per branch). Test answers exist only in the hidden
evaluator.

## Experiments (scored)

| experiment | weight | what it measures |
|---|---:|---|
| `fb15k237_cqa` | 0.55 | answering the 14 standard query types (fixed sample of 2,000 test queries per type) |
| `fb15k237_unobservable` | 0.25 | answering fully unobservable queries: 14 standard types plus 4p and 4i (up to 1,000 test queries per type) |
| `fb15k237_traceability` | 0.20 | whether the intermediate entities a method names for 2p/3p queries lie on a real path to a non-trivial answer |

## Metrics

All values are percentages; higher is better.

- **Filtered MRR / Hits@k.** For each scored (non-trivial) answer, its rank is
  1 + the number of *non-answer* entities listed above it. Other answers,
  the benchmark's easy answers, and every entity already reachable on the
  observed graph are ignored wherever they appear. Answers missing from the
  submitted list count as reciprocal rank 0. Per query, scores are averaged
  over its scored answers; per type, over queries.
  `mrr_A` / `hits*_A` average the 14 standard types, `*_Ap` the 9 positive
  types (1p 2p 3p 2i 3i ip pi 2u up), `*_An` the 5 negated types (2in 3in inp
  pin pni). `mrr_<type>` reports each type.
- **Traceability.** For a 2p query `a -r1-> V1 -r2-> x` an entity is a valid
  `V1` if `(a, r1, V1)` and `(V1, r2, t)` hold in the complete graph for some
  scored answer `t` (3p: `V1`, `V2` must lie on a complete-graph chain from `a`
  to a scored answer). `trace_var_mrr` / `trace_var_p1`: reciprocal rank of the
  first valid entity / precision at 1 in each variable's ranking, averaged over
  2p-V1, 3p-V1 and 3p-V2. `trace_path_p1`: fraction of queries whose top-1
  intermediate entities and top-1 non-ignored answer form a valid
  complete-graph chain ending at a scored answer (2p and 3p averaged).

## Submission format

Every directory `outputs/<method>/` is one method. It must contain integer
arrays (`np.savez`, no pickles), rows in the order of the public query files:

| file | arrays |
|---|---|
| `valid_std.npz`, `test_std.npz` | `rank_<type>` `[n_type, K]` for the 14 standard types; `trace_2p_v1` `[n_2p, KT]`, `trace_3p_v1`, `trace_3p_v2` `[n_3p, KT]` |
| `valid_hplus.npz`, `test_hplus.npz` | `rank_<type>` `[n_type, K]` for the 16 types |

`rank_*` rows list entity ids best first, without repeats (`1 <= K <= 14505`;
1000 is enough in practice, and the `obs_*` entities may be left out because
they are ignored). Trace rows rank candidate entities for each existential
variable (`KT = 100` recommended).

## Scoring rules

- `bash scripts/evaluate_train.sh` scores every method on the validation
  queries → `scoring/scores_train.json`; `bash scripts/evaluate_test.sh`
  scores every method on the hidden test queries → `scoring/scores_test.json`.
  Both take no arguments.
- Every number is recomputed from the submitted rankings; nothing else a
  method writes (claimed scores, parameter counts, timings, flags) is read.
- A present array with a wrong shape or row count, a non-integer dtype, an
  out-of-range or repeated id fails the whole method: every metric of every
  experiment gets its worst value (0). A missing file or ranking array gives
  worst values for that experiment; missing trace arrays give worst trace
  values. Failed methods are still listed.

## Baselines

Runnable baselines (in `scores_*.json`): `random` (random lists),
`symbolic_traversal` (exhaustive grounding of all variables on the observed
graph plus a relation-popularity prior), `vsa_composition` (training-free
vector composition of pretrained link-prediction embeddings). Published
numbers of other systems are recorded in `scoring/reference.json`.
