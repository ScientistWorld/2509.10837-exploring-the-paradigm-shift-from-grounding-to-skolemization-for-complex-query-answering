# Progress

## What Works

- Paper analysis, benchmark scaffolding, scoring metadata, and scripts are in place.
- `data/synthetic_cqa.py` generates a deterministic complex-query benchmark with 1p, 2p, 3p, 2i, and 2in queries.
- `method/lvsa.py` implements LVSA's complex embeddings, binding projection, normalized conjunction, differentiable Skolemization modules, and neural negator losses.
- `eval/evaluate_predictions.py` scores prediction artifacts without importing method code.
- `python validate.py` passes.

## Results

- Random baseline on the scaled benchmark: MRR 0.0556, Hits@1 0.0129, Hits@10 0.1414.
- LVSA results are pending the submitted compute job.

## Remaining

- Run LVSA training end-to-end through `scripts/reproduce.sh`.
- Inspect `scoring/scores.json` after the job and compare LVSA against the random baseline.
- If LVSA does not clearly beat the baseline, tune the scaled curriculum or model size while preserving the paper's computation.

## Issues

- The paper markdown and web search did not reveal an LVSA code release.
- Full FB15k-237/NELL CQA reproduction would require larger dataset download and pretrained ComplEx initialization; the first job uses a scaled generated KG to verify the core algorithm.

## Deviations from Paper

- Dataset scale: the paper evaluates FB15k, FB15k-237, NELL995, and harder variants; this first reproduction uses a deterministic synthetic KG with the same CQA query families to keep the initial job small.
- Initialization: the paper initializes from pretrained ComplEx embeddings with dimension 1000; this scaled run trains smaller complex embeddings directly before freezing them for Skolemization and negation stages.
- Training scale: epochs, dimension, and entity count are reduced for the first end-to-end verification job.
