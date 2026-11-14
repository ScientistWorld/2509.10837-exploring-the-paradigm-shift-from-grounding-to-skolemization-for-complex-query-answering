# Submitted experiment bundle

The original checkout is recorded in `base_commit.txt`; `initial_status.txt`
records pre-existing harness changes. Audit source and evidence stay here.
Only the scheduler entrypoint and action file are temporarily changed outside
this directory. Each case restores the original data and scoring artifacts.

The bundle reruns the original predictions, trains the reference at seeds
11/12/13, trains for three times as many epochs at seed 11, and removes stage-2
training as a deliberately weaker method. It also reruns the random baseline
and saved negation ablation. Every candidate reaches the hidden test via its
own no-argument shell invocation, with no evaluator or environment tampering.

Shortcut trials: visible-label copying; deterministic graph reconstruction;
exhaustive existential grounding; zero/negative/nonfinite size metadata;
selectively emitting only successful traces; replacing the evaluator's graph;
replacing truth labels with a constant; claimed scores and success flags with
invalid entity rankings; empty outputs and missing metadata; null metadata;
and omitting a failed on-disk method from the public evaluation manifest.

All reported leaderboard differences will disclose the cohort used to
normalize each metric. The future ranking population is not available, so no
single absolute audit score can represent every possible future leaderboard.
