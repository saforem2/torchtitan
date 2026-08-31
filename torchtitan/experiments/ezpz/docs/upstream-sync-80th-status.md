# 80th upstream sync: what works, what is deferred, what it costs

**Worktree:** `/lus/tegu/.../tt-sync80` (branch `worktree-ezpz-80th-sync`)
**Merged:** 48 commits past `c537bca68`. 1 textual conflict (torchft/trainer.py,
upstream-owned, took theirs).

> [!IMPORTANT]
> **Status updated 2026-08-31: the #4121 blocker is cleared.** The
> "Remaining blocker" section at the bottom is a record of what the port
> cost, not an open item. `agpt_2b`, `agpt_20b` and `agpt_30b_olmo2tok` all
> build on this tree at `num_tokens_per_microbatch_per_dp_rank = 8192 ==
> max_context_length`, and syncs 82 and 83 have landed on top of the 80th
> (see [`upstream-sync.md`](./upstream-sync.md)).
>
> The **data-mix arms are still deferred** -- the `NotImplementedError` in
> `blendcorpus_builder.py:201` and the `c4_test` raise in
> `agpt/config_registry.py:414` are both still in the tree, for the reasons
> given below. That part of this page stands.

## Status

| | |
|---|---|
| imports clean | trainer, agpt/moe config_registry, blendcorpus |
| production configs BUILD | **yes, as of 2026-08-31** -- the `local_batch_size` unit change is ported |
| data-mix arms | **BROKEN by design** (see below) -- still true |

## What the deferral actually costs

An earlier commit note said "every production config names blendcorpus and
never reaches [the deferred paths]". That is **wrong** and worth correcting
precisely, because it understates the cost.

TRUE: `agpt()` sets `dataset = "blendcorpus"` (config_registry.py:274), and
`agpt_2b` / `agpt_20b` / `agpt_30b_olmo2tok` / `agpt_30b_olmo2tok_mano` all
inherit it. They take the blendcorpus branch, never construct a
`HuggingFaceTextDataLoader`, and are unaffected by the stubs.

FALSE: that this covers everything. **Seven data-mix arms reach the deferred
paths and now raise `NotImplementedError`:**

    agpt_2b_mds_mix_owm            agpt_2b_mds_mix_owm_edu_9010
    agpt_2b_mds_mix_edu            agpt_2b_mds_mix_owm_cosmo_7525
    agpt_2b_mds_mix_owm_edu_7525   agpt_2b_mds_mix_owm_nemotron_7525
    agpt_2b_mds_mix_owm_edu_5050

These are the Wave-1 owm/edu diversity experiment, not hypothetical configs.
`_base_config` also uses `c4_test`, so the bundled debug config is affected.

## Why raising is still the right call for the mix arms

Not laziness -- porting them naively would be WORSE than the raise. Upstream's
replacement for `InterleavedHuggingFaceTextDataLoader` is `DatasetMixConfig`,
whose weights select **emitted elements**, not tokens, with an explicit
`TODO(data-token-weighted-mix)` at `components/data/dataset.py:249`. Our arms
are commented as token-mixture ratios. A 75/25 token blend does not survive as
75/25 under the new semantics, so a silent port would quietly change what the
experiment measures. A loud `NotImplementedError` naming the replacement is
the honest failure.

`stopping_strategy="all_exhausted"` also has no analogue, and
`loader.py:92-97` raises outright when `dp_world_size > 1 and not repeat`.

## The #4121 port (was "Remaining blocker"; DONE as of 2026-08-31)

`TrainingConfig` no longer accepts `local_batch_size`. This is #4121's
sequences -> tokens change, NOT a rename:

    num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
    num_tokens_per_train_step             = global_batch_size * seq_len

Every numeric literal must be multiplied through. 53 `local_batch_size` sites
in 8 files, 30 `global_batch_size` in 6, plus 20 JSON run configs.
`pipeline_parallel_microbatch_size -> num_pp_microbatches` is a further
semantic inversion (size -> count), not a rename either.
