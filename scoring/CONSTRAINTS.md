# Scientist Constraints

## Data
Use the fixed generated dataset under `data/synthetic` with its train, valid,
and test splits. Do not add external KG triples, labels, or query instances for
training or evaluation. The test labels may only be used by `eval/`.

## Resources
Keep parameter count reported through `param_count_k`. Methods may improve the
algorithm, but hiding parameters or using an unreported external model violates
the benchmark.

## Evaluation
Submit prediction artifacts in the same JSONL format consumed by
`scripts/evaluate.sh`. Do not modify `eval/` to privilege a particular method.

## Compliance Checklist
- [ ] Uses only the fixed `data/synthetic` splits.
- [ ] Does not train on test answers.
- [ ] Reports all trainable parameters through `param_count_k`.
- [ ] Evaluation runs through `scripts/evaluate.sh` without importing `method/`.
