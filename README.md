# Exploring the Paradigm Shift from Grounding to Skolemization for Complex Query Answering on Knowledge Graphs

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.10837"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.10837-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2509.10837](https://arxiv.org/abs/2509.10837) |
|:---|:---|
| **Domain** | Knowledge graph reasoning |
| **TL;DR** | LVSA answers complex logical KG queries by explicitly modeling Skolem functions in a vector-symbolic architecture. |

## Summary

The paper studies complex query answering over incomplete knowledge graphs. It
argues that grounding-based methods are faithful but expensive, while previous
Skolemization-based methods are efficient but weakly constrained. LVSA combines
complex-valued VSA binding and bundling with differentiable Skolem functions
and a logic-regularized neural negator. The reported results show better
standard benchmark accuracy than Skolemization baselines and stronger
efficiency than grounding-based search.

## Problem

How can a system answer complex first-order logical queries over an incomplete
knowledge graph when the query contains existential variables, conjunction,
disjunction, and negation?

## The paper's method

The paper proposes LVSA, a logic-constrained vector-symbolic architecture for
complex query answering. Its core contribution is explicit differentiable
Skolemization for existential variables combined with simple VSA operations and
logic-regularized negation.

## Key results of the paper

- Reported FB15k-237 aggregate MRR is 22.4 and aggregate H@1 is 15.4.
- LVSA is reported as 2.6x faster than CLMPT and 1.7x faster than ConE.
- Negation regularization ablations show lower negation MRR when logical axiom loss is removed.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +107.3 |
| `gpt-5.6-terra-xhigh` | +104.6 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.10837 --mode innovate --agent claude-code
sw evaluate 2509.10837
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

