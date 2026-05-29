# Progress

## What Works

- Paper analysis, benchmark scaffolding, scoring metadata, and scripts are in place.
- `data/synthetic_cqa.py` generates a deterministic complex-query benchmark with 1p, 2p, 3p, 2i, and 2in queries.
- `method/lvsa.py` implements LVSA's complex embeddings, binding projection, normalized conjunction, differentiable Skolemization modules, and neural negator losses.
- LVSA now executes explicit query-graph branches: it materializes intermediate variables in dependency order, applies independent/dependent Skolemization, and emits witness traces for multi-hop variables.
- `eval/evaluate_predictions.py` scores prediction artifacts without importing method code and reads canonical labels from the fixed test set instead of trusting labels embedded in prediction files.
- `python validate.py` passes.
- Continuation audit found `scripts/evaluate.sh` is standalone and import-separated, but existing LVSA prediction artifacts used legacy query metadata while the fixed test data now stores derived query branches. The evaluator now compares only semantic row identifiers (`type`, `anchors`, `relations`) and still reads answers from the canonical test set.
- CPU test job `a80ed4a6-a5d` completed `scripts/run.sh` end-to-end in the container: download guard, LVSA smoke test, random baseline, LVSA training, no-logic ablation, combined evaluation, and `validate.py --compare`.

## Results

- Random baseline on the scaled benchmark: MRR 0.0556, Hits@1 0.0129, Hits@10 0.1414.
- LVSA on the scaled benchmark after the graph-structured encoder retrain: MRR 0.4914, Hits@1 0.4414, Hits@10 0.5929.
- Query breakdown: 1p MRR 0.7164, 2p 0.3859, 3p 0.2323, 2i 0.5740, 2in 0.6334.
- Negation ablation without the logical axiom loss: MRR 0.4751, Hits@1 0.4229, Hits@10 0.5843, and 2in MRR 0.5109.
- The full negation objective improves 2in MRR by 0.1225 absolute over the no-logic ablation while keeping the same 252.416k parameter count, supporting the paper's secondary claim that logical constraints improve negated-query behavior.
- Added Skolem trace evaluation for multi-hop queries. The completed CPU job produced LVSA trace-variable MRR 0.2535, trace-variable P@1 0.1638, and trace-path P@1 0.0703 on the scaled synthetic benchmark, giving a runnable version of the paper's interpretability/causal-traceability analysis.
- First CPU job completed all three LVSA training stages and saved a checkpoint,
  but failed during prediction because PyTorch loaded the checkpoint in
  `weights_only` mode.
- LVSA substantially exceeds the random baseline across all query families, so
  the scaled reproduction supports the paper's core claim.

## Remaining

- A future expansion could add another synthetic seed, a larger entity count, or public FB15k-237/NELL query files.
- The current packaged environment is complete for the scaled core claim and the negation ablation.
- Further milestones would require broadening beyond this scaled package, such as adding the paper's public benchmark query files and more of the reported query families.

## Issues

- The paper markdown and web search did not reveal an LVSA code release.
- Full FB15k-237/NELL CQA reproduction would require larger dataset download and pretrained ComplEx initialization; the first job uses a scaled generated KG to verify the core algorithm.
- Fixed checkpoint loading by saving path-free metadata and passing
  `weights_only=False` for locally generated checkpoints.
- Audited the scoring metadata and fixed `scoring/reference.json` to contain paper-reported FB15k-237 main-result and negation-ablation values. The scaled synthetic reproduction remains in `scoring/scores.json` under a separate experiment name because it is not numerically comparable to paper-scale FB15k/FB15k-237/NELL results.
- Fixed the trace-evaluation validation failure by registering the trace metrics in the scaled benchmark's metric schema.
- Hardened evaluation after audit: prediction files no longer need to contain answers, query labels come from `data/synthetic/test.jsonl`, and methods without traces receive zero traceability credit.
- Reworked the LVSA encoder after audit so the implementation is identifiable as the paper's sequential Skolemization procedure rather than only a small set of hard-coded query-type branches.
- Local Python does not include `torch`, so syntax and metadata checks were run locally and the full forward/training verification is delegated to the submitted compute-container job.
- Continuation audit fixed a standalone evaluation failure caused by comparing derived `branches` metadata exactly, and fixed `scripts/reproduce.sh` so the end-to-end path includes the ablation, final combined evaluation, and `validate.py --compare`.
- CPU test job `5c6ed090-2a1` failed in stage 2 because `skolem_dependent()` referenced an undefined local variable after the graph-encoder refactor. Fixed the variable name and added `scripts/smoke.sh` to exercise LVSA scoring, tracing, and CE loss before training starts.
- Retry CPU test job `a80ed4a6-a5d` passed after the dependent-Skolemization fix and refreshed the final scores.

## Deviations from Paper

- Dataset scale: the paper evaluates FB15k, FB15k-237, NELL995, and harder variants; this first reproduction uses a deterministic synthetic KG with the same CQA query families to keep the initial job small.
- Query coverage: the packaged benchmark covers 1p, 2p, 3p, 2i, and 2in. The encoder accepts explicit branch atoms, but the benchmark does not yet instantiate all 14 query families reported in the paper.
- Scoring reference: paper-reported FB15k/FB15k-237/NELL MRR percentages are documented in `briefing/evaluation.md` and `paper/`; `scoring/reference.json` is calibrated to the deterministic scaled benchmark so automated validation compares like with like.
- Initialization: the paper initializes from pretrained ComplEx embeddings with dimension 1000; this scaled run trains smaller complex embeddings directly before freezing them for Skolemization and negation stages.
- Training scale: epochs, dimension, and entity count are reduced for the first end-to-end verification job.
