# Progress

## What Works

- Paper analysis, benchmark scaffolding, scoring metadata, and scripts are in place.
- `data/synthetic_cqa.py` generates a deterministic complex-query benchmark with 1p, 2p, 3p, 2i, and 2in queries.
- `method/lvsa.py` implements LVSA's complex embeddings, binding projection, normalized conjunction, differentiable Skolemization modules, and neural negator losses.
- `eval/evaluate_predictions.py` scores prediction artifacts without importing method code.
- `python validate.py` passes.

## Results

- Random baseline on the scaled benchmark: MRR 0.0556, Hits@1 0.0129, Hits@10 0.1414.
- LVSA on the scaled benchmark: MRR 0.4931, Hits@1 0.4286, Hits@10 0.6129.
- Query breakdown: 1p MRR 0.7100, 2p 0.3894, 3p 0.2521, 2i 0.5583, 2in 0.6429.
- First CPU job completed all three LVSA training stages and saved a checkpoint,
  but failed during prediction because PyTorch loaded the checkpoint in
  `weights_only` mode.
- LVSA substantially exceeds the random baseline across all query families, so
  the scaled reproduction supports the paper's core claim.

## Remaining

- Run the w/o logical-axiom negation ablation and compare `mrr_2in` against full LVSA.
- If time allows, add another synthetic seed or a larger entity count for core_claim_plus.

## Issues

- The paper markdown and web search did not reveal an LVSA code release.
- Full FB15k-237/NELL CQA reproduction would require larger dataset download and pretrained ComplEx initialization; the first job uses a scaled generated KG to verify the core algorithm.
- Fixed checkpoint loading by saving path-free metadata and passing
  `weights_only=False` for locally generated checkpoints.

## Deviations from Paper

- Dataset scale: the paper evaluates FB15k, FB15k-237, NELL995, and harder variants; this first reproduction uses a deterministic synthetic KG with the same CQA query families to keep the initial job small.
- Initialization: the paper initializes from pretrained ComplEx embeddings with dimension 1000; this scaled run trains smaller complex embeddings directly before freezing them for Skolemization and negation stages.
- Training scale: epochs, dimension, and entity count are reduced for the first end-to-end verification job.
