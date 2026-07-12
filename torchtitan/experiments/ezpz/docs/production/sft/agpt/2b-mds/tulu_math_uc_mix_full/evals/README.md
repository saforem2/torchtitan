# lm-eval: full-mix SFT'd AuroraGPT-2B vs pretrained baseline (step sweep)

**Date:** 2026-07-12
**Machine:** Sunspot
**Job:** 12470365 (1N eval sweep, `rl/scripts/sft/eval_sweep_fullmix_8n.sh`)
**Models compared:**
- **baseline**: `global_step138650` (AuroraGPT-2B MDS stage-3, raw pretrained)
- **sft-step{300,600,900}**: `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-{N}-hf`
  (the ongoing 8N SFT on the FULL `tulu_math_uc_mix` -- OpenMathInstruct-2 mix,
  ~54B-token / 1-epoch target; these are early checkpoints, ~2-6B tokens in of
  ~54B). See [run README](../README.md).

## TL;DR

Same **alignment-tax** pattern as the completed metamathqa-swap SFT
([evals](../../tulu_math_uc_mix/evals/README.md)): base-LM multiple-choice
benchmarks are flat-to-slightly-down under instruction SFT, not up. The one
notable *trend* is arc_easy / arc_challenge declining monotonically with more
SFT steps (arc_easy 0.694 -> 0.640 over 0 -> 900 steps), i.e. more big-mix
tokens = a bit more base-LM tax; boolq and winogrande tick *up*. This is
expected and NOT a red flag -- SFT on chat/instruction data doesn't add base
knowledge. The real SFT payoff is measured by **IFEval** (instruction
following) and **downstream GRPO**, not these tasks (the completed SFT showed
base-LM flat but IFEval +8pp and GRPO 8x -- see the sibling recipe's evals).

## Results (0-shot, acc_norm where present else acc)

| Task | Baseline | step 300 | step 600 | step 900 | step900 - base |
|---|---:|---:|---:|---:|---:|
| hellaswag | 0.5926 | 0.5908 | 0.5939 | 0.5919 | -0.001 |
| arc_easy | 0.6944 | 0.6620 | 0.6553 | 0.6402 | **-0.054** |
| arc_challenge | 0.3899 | 0.3848 | 0.3686 | 0.3618 | **-0.028** |
| winogrande | 0.5817 | 0.6069 | 0.6077 | 0.6038 | **+0.022** |
| piqa | 0.7432 | 0.7361 | 0.7350 | 0.7296 | -0.014 |
| openbookqa | 0.3660 | 0.3880 | 0.3820 | 0.3760 | +0.010 |
| boolq | 0.5985 | 0.5963 | 0.6229 | 0.6263 | **+0.028** |

Metric is `acc_norm,none` where present (hellaswag, arc_*, piqa, openbookqa)
and `acc,none` otherwise (winogrande, boolq). 0-shot; identical task specs
across all models. Baseline numbers match the completed-SFT eval's baseline
(arc_easy 0.694, hellaswag 0.592, ...) -- pipeline-consistent.

## Notes

- **arc regression is the clearest signal**: arc_easy loses ~5pp and
  arc_challenge ~3pp monotonically with SFT steps. The completed metamathqa
  SFT lost ~2pp on arc at step 729 (~4.5B tokens); the bigger mix taxes arc
  more, plausibly because OpenMathInstruct-2 heavily narrows the output
  distribution toward math CoT.
- **boolq / winogrande gain** (+0.028 / +0.022) -- both benefit from the
  instruction/QA framing in tulu-3 + ultrachat.
- This is a *base-LM* eval only. **TODO** (higher-signal, not yet run for this
  run): IFEval (`eval_ifeval_sft_vs_baseline.sh`) and a GRPO smoke
  (`grpo_smoke_sft_vs_baseline.sh`) -- those are where the completed SFT's real
  wins showed up.
- More sweep points land as the run advances:
  `qsub -v SWEEP_CKPTS="1500 3000" rl/scripts/sft/eval_sweep_fullmix_8n.sh`.

Raw results: `outputs/evals/fullmix-8n-sweep-12470365/<label>/<task>/`.
