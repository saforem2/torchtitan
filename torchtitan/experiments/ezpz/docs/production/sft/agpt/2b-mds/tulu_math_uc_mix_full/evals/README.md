# Evals: full-mix 8N SFT (gs138650 x tulu_math_uc_mix_full)

**Date:** 2026-07-17 (updated at run completion; supersedes the 2026-07-12
partial-sweep version).
**Machine:** Sunspot.
**Jobs:** base-LM sweep 12470365 (steps 300/600/900) + 12470886 (steps
1500-8672); IFEval + GRPO 12470889 (step 8672) + 12470896 (step 900).
**Models:** `global_step138650` baseline vs
`checkpoint-{N}-hf` of the completed full-mix 8N SFT (~54B-token, 1-epoch,
step 8672 final). See [run README](../README.md).

## TL;DR -- the full mix works EARLY and is DESTROYED by full-epoch training

The headline the run was built to answer -- "does ~12x more SFT tokens beat
the metamathqa-729 deliverable?" -- is **no at the final checkpoint, but YES at
~step 900**:

- **checkpoint-8672 (final, 1 epoch) catastrophically forgot.** Base-LM
  benchmarks collapsed to ~random chance (hellaswag 0.59 -> 0.27, arc_easy
  0.69 -> 0.30) AND IFEval is flat-to-DOWN vs baseline (prompt-strict 0.168 vs
  base 0.179). It got very good at exactly one thing -- emitting
  OpenMathInstruct-2-style math CoT (train loss 0.357, token-acc 0.90) -- and
  lost everything else. **Worse than the metamathqa deliverable on every axis.**
- **checkpoint-900 (~5.7B tokens) is the real deliverable.** IFEval
  matches-or-beats metamathqa-729 (prompt-strict 0.253 vs 0.244; inst-loose
  0.417 vs 0.411) WHILE retaining base-LM capability (pre-collapse: hellaswag
  0.59, arc_easy 0.64). Best of all four models evaluated.
- **Root cause of the collapse: LR x tokens.** LR 2e-5 held above 1e-5 through
  step ~4350 (cosine decay only bites the second half). ~4000 steps at high LR
  on a narrow math distribution overfit + forgot. The metamathqa SFT survived
  only because it STOPPED at 729 steps -- before the same damage. Running the
  full mix to 1 epoch was the mistake, not the mix itself.

## Base-LM sweep (0-shot, acc_norm where present else acc)

Full trajectory, baseline + steps 300 -> 8672. The collapse is between
~step 1500 and ~4500:

| Task | base | 300 | 600 | 900 | 1500 | 3000 | 4500 | 6000 | 7500 | 8672 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| hellaswag | 0.593 | 0.591 | 0.594 | 0.592 | 0.592 | 0.502 | 0.321 | 0.287 | 0.275 | 0.273 |
| arc_easy | 0.694 | 0.662 | 0.655 | 0.640 | 0.632 | 0.516 | 0.372 | 0.331 | 0.298 | 0.298 |
| arc_challenge | 0.390 | 0.385 | 0.369 | 0.362 | 0.367 | 0.309 | 0.253 | 0.241 | 0.248 | 0.236 |
| winogrande | 0.582 | 0.607 | 0.608 | 0.604 | 0.590 | 0.519 | 0.496 | 0.523 | 0.511 | 0.530 |
| piqa | 0.743 | 0.736 | 0.735 | 0.730 | 0.722 | 0.648 | 0.576 | 0.562 | 0.526 | 0.537 |
| openbookqa | 0.366 | 0.388 | 0.382 | 0.376 | 0.368 | 0.332 | 0.310 | 0.302 | 0.292 | 0.264 |
| boolq | 0.599 | 0.596 | 0.623 | 0.626 | 0.623 | 0.598 | 0.508 | 0.509 | 0.475 | 0.495 |

Steps 300/600/900 are from the earlier sweep (job 12470365); 1500-8672 from
12470886. Random-chance references: hellaswag/arc/openbookqa ~0.25,
piqa/boolq/winogrande ~0.50. By step 8672 every task is at or near chance.

## IFEval (instruction following -- the actual SFT target)

The higher-signal metric. checkpoint-900 is the standout:

| metric | baseline | full-mix 900 | full-mix 8672 | metamathqa 729 |
|---|--:|--:|--:|--:|
| prompt_level_strict | 0.179 | **0.253** | 0.168 | 0.244 |
| inst_level_strict | 0.289 | **0.384** | 0.287 | 0.386 |
| prompt_level_loose | 0.183 | **0.283** | 0.183 | 0.274 |
| inst_level_loose | 0.294 | **0.417** | 0.300 | 0.411 |

- **full-mix 900**: +7.4pp prompt-strict, +9.8pp prompt-loose, +12.4pp
  inst-loose over baseline -- and matches/edges the metamathqa-729 deliverable.
- **full-mix 8672**: flat-to-down vs baseline. Overtraining destroyed
  instruction-following along with base-LM capability.

## GRPO (sum_digits) -- checkpoint-900, vLLM server-mode

**checkpoint-900 is a strong GRPO starting point.** Run on the blessed
vLLM-server path (`rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh`,
unified `venvs/rl-vllm/`, job 12470959): `accuracy_reward/mean` climbed
**0.31 -> ~0.74** over the first ~20 steps -- a clean, fast upward RL curve
(format_reward 0.35 -> 0.43 alongside). The run hit its 6h walltime at ~step 20
(not a crash), so this is an early-but-decisive trend, not a converged number;
the completed metamathqa GRPO climbed 0.4 -> 0.9 over a full 1000 steps for
comparison. Re-run with a higher walltime / lower max_steps for a converged
figure if needed.

Trajectory: `0.31, 0.34, 0.39, 0.59, 0.54, ..., 0.66, 0.75, 0.79, 0.63, 0.74`.

NOTE on tooling: the earlier attempt via the combined IFEval+GRPO wrapper AND a
standalone `.venv` hf.generate GRPO both failed -- the former on walltime
starvation, the latter because the `.venv` (torch 2.13 nightly) per-rank
`hf.generate()` FSDP path HANGS multi-rank (regressed since June; single-rank
works). The working path is the vLLM server-mode (`--use_vllm --vllm_mode
server`) from `venvs/rl-vllm/`; see `docs/production/rl/grpo-on-xpu-status.md`. The old
FSDP-generate + vllm-test-split GRPO scripts were removed 2026-07-17.

## Recommendation

- **Deliverable = full-mix `checkpoint-900-hf`** (not 8672). It has the
  instruction-following gains of the metamathqa deliverable AND intact base-LM
  capability. checkpoint-900-hf is consolidated and preserved.
- **Do NOT use checkpoint-8672** for anything downstream -- it is a
  math-CoT-only overfit, near-random on general tasks.
- **Recipe lesson (recorded):** for full-mix (narrow-distribution) SFT at
  LR 2e-5, cap training at O(1000) steps or lower the LR substantially; a full
  epoch (~8672 steps) at peak LR causes catastrophic forgetting. The metamathqa
  recipe's 729-step stop was load-bearing.

Raw results: `outputs/evals/fullmix-8n-sweep-{12470365,12470886}/` (base-LM);
`outputs/evals/aurora2b-ifeval-20260717-*/` (IFEval).
