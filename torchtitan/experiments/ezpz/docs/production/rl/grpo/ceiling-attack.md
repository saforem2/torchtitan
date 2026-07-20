# agpt-2b GRPO+LoRA: breaking the reward-shape ceiling (2026-07-20)

The [beat-v5 tuning sweep](./beat-v5-sweep.md) established that no configuration
lever (learning rate, LoRA rank, batch size) lifts the mean reward past **~0.25**
on the easy alphabet_sort task -- every run plateaus there. This is the
*ceiling-attack*: change the **reward function**, not the config, and the ceiling
moves.

![ceiling attack](aurora2b/charts/ceiling-attack.svg)

## Hypothesis

The upstream `RewardAlphabetSort` scores a `difflib` character-ratio of the
model's block vs the expected block, raised to a power. That reward is:

- **saturating** -- near-correct and exactly-correct answers score almost the
  same, so there is little gradient in the region GRPO needs to climb; and
- **noisy** -- it reacts to name spellings/lengths, not just to ordering, the
  actual sub-skill being trained.

So the mean plateaus even as per-batch peaks rise with capacity. The fix is a
reward that scores the **sub-skills additively**, giving a smooth gradient toward
fully-correct output.

## The shaped reward

`ShapedRewardAlphabetSort` (in `rl/alphabet_sort_agpt/shaped_reward.py`) replaces
the char-ratio with three additive components, in `[0, 1]`:

| Component | Weight | What it rewards |
|-----------|--------|-----------------|
| format | 0.2 | a valid `<xml_tag>` block with >=1 name line |
| completeness | 0.2 | exactly the expected name set (no missing, no extra/dupes) |
| order | 0.6 | fraction of correctly-ordered adjacent pairs among names present (Kendall-tau-like on the subsequence) |

An exact, correctly-ordered answer scores 1.0. A well-formatted but mis-ordered
answer still earns format + completeness, and partial ordering earns proportional
order credit -- so the gradient points squarely at ordering, the hard sub-skill.
Config: `rl_grpo_lora_agpt_2b_shaped` (shaped reward + the sweep's best levers,
LoRA rank 32, lr 5e-5), same easy task as v5.

## Result -- an honest, one-scale comparison

The shaped reward is a *different function*, so its raw value is not comparable to
v5's. To compare fairly, every run's completions are **re-scored offline with the
identical char-ratio(power=1) reward that v5 actually trained on** (reconstructing
the expected answer from each prompt, then scoring). This is self-validating: for
the char-ratio runs the reconstructed score matches their **stored** reward
exactly (v5 0.249 == 0.249, w1 0.249, w2 0.244, ...), which proves the
reconstruction is exact -- so the shaped number is trustworthy.

| Run | char-ratio(p1), last-3-version mean | lever |
|-----|-------------------------------------|-------|
| v5 (prev best) | 0.249 | -- |
| w1 | 0.249 | lr 5e-5 |
| w2 | 0.237 | LoRA rank 32 |
| w3 | 0.237 | lr 5e-5 + 16 groups |
| **shaped** | **0.667** (final, 100 steps) | **componentized reward** |

**The tuning cluster is flat at ~0.24-0.25; the shaped run reaches 0.667 on the
same metric -- +168% in genuine task performance, not a scoring artifact.** The
gain shows up even under the old saturating reward, because the model actually
learned to sort better; the order-specific gradient was the missing ingredient.
On its own (shaped) scale the run climbs 0.18 -> ~0.67 across the full 100
steps without plateauing -- the healthy learning curve the char-ratio reward
never produced.

**Takeaway:** for a saturating/noisy reward, reward-shaping beats every
hyperparameter lever. The ~0.25 wall was a property of the reward function, and
decomposing it into additive sub-skills moved the wall.

## Repro

```bash
CONFIG=rl_grpo_lora_agpt_2b_shaped DUMP=outputs/rl_lora_agpt2b_shaped \
  bash torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_grpo.sh
```

Cross-score + chart: `plot_ceiling.py` (re-scores every run on char-ratio(p1);
kept in this dir). The shaped reward and its unit tests live in
`rl/alphabet_sort_agpt/shaped_reward.py`; the config in
`rl/alphabet_sort_agpt/config_registry.py`.
