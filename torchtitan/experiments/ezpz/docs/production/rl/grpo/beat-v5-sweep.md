# agpt-2b GRPO+LoRA: "beat v5" tuning sweep (2026-07-19/20)

Follow-up to the [3-way study](../history/grpo-lora-agpt2b-repro.md) where **v5**
(easy task, lr 2e-5, LoRA rank 8) won at mean reward ~0.25. This sweep tries to
beat it, all on the same easy alphabet_sort task (1 turn, <=3 names), each pulling
a distinct lever.

![beat-v5 sweep](aurora2b/charts/beat-v5-sweep.svg)

## Configs

| Run | Lever vs v5 | config fn |
|-----|-------------|-----------|
| v5 (baseline) | -- | `rl_grpo_lora_agpt_2b_easy` |
| w1 | lr 2e-5 -> **5e-5** | `rl_grpo_lora_agpt_2b_w1` |
| w2 | LoRA rank 8 -> **32** (alpha 64) | `rl_grpo_lora_agpt_2b_w2` |
| w3 | lr **5e-5** + groups/step 8 -> **16** | `rl_grpo_lora_agpt_2b_w3` |

## Result (all four runs complete, 100 steps)

| Run | mean reward (plateau) | **peak bin** |
|-----|-----------------------|--------------|
| v5 | ~0.25 | 0.277 |
| w1 (lr5e-5) | ~0.25 | **0.362** |
| w2 (rank32) | ~0.24 | **0.386** |
| w3 (5e-5,16grp) | ~0.24 | 0.279 |

**Two readings, both true:**
1. **Mean reward plateaus at ~0.24-0.25 for all four** -- the running-mean ceiling is
   a task/reward-shape property, not a config limit. The tuned levers reach it
   faster/more reliably but the *average* rollout tops out around there.
2. **Peak per-batch reward IS raised by capacity + LR:** w2 (rank 32) peaks 0.386
   and w1 (lr 5e-5) peaks 0.362, vs v5's 0.277 -- ~+40%. So more adapter capacity
   (w2) and stronger updates (w1) DO push the top of the achievable range up; the
   mean is dragged down by the many partial/zero rollouts (the reward-shape issue).
   w3 (bigger batch) tracks v5 -- batch size was not the lever.

**Takeaway:** capacity (LoRA rank) is the most promising lever, but the *mean*
ceiling is set by the reward function -- lifting it needs reward-shaping, not more
tuning. That is confirmed by the [ceiling-attack](./ceiling-attack.md), which
breaks the ~0.25 wall (+85% on the identical metric) by componentizing the reward.

## Repro

```bash
for W in w1 w2 w3; do
  CONFIG=rl_grpo_lora_agpt_2b_$W DUMP=outputs/rl_lora_agpt2b_$W \
    bash torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_grpo.sh
done
# chart: PYTHONPATH=<repo> .venv/bin/python (plot script in this dir's history)
```
Configs live in `rl/alphabet_sort_agpt/config_registry.py`.
