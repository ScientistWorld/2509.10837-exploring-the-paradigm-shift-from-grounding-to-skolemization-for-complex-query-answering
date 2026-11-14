# Scientist Constraints

## Data
- Train only on the training split: `train_triples.npy` and
  `train_queries.npz` (standard benchmark protocol). Validation queries,
  answers and `valid_triples.npy` are for model selection and the train-side
  score only; never fit parameters on them. (The `obs_*` lists, which mark
  entities the evaluator ignores, may be used to drop those entities from
  submitted lists.)
- Do not add external facts, entity descriptions, pretrained models trained on
  Freebase/FB15k-237 test triples, or the benchmark's test files (they are
  public on the internet; fetching them is a violation).
- Test answers are only available to `eval/test/`.

## Method
- Answer each query from its logical structure with learned representations;
  do not enumerate every entity for every existential variable
  (exhaustive grounding), which the problem rules out for cost reasons.
- Intermediate-variable rankings (`trace_*`) must come from the same reasoning
  that produces the answer ranking, not from a separate lookup.

## Evaluation
- Submit the integer ranking files described in `briefing/evaluation.md` under
  `outputs/<method>/`. Do not modify `eval/`, the hidden truth, or the scoring
  scripts.

## Compliance Checklist
- [ ] Trained only on the training split (validation used for selection only).
- [ ] No test files, external facts or Freebase-derived pretrained models.
- [ ] No exhaustive grounding of existential variables.
- [ ] Traces produced by the answering model itself.
- [ ] Scored with `scripts/evaluate_train.sh` / `scripts/evaluate_test.sh`.
