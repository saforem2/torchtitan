# Umbrella seat audit -- 2026-08-21

A full audit of all five umbrella seats found **two defects that would have
damaged production**, retired a seat whose seed was never a real checkpoint,
and solved two long-standing "corrupt checkpoint" mysteries that turned out to
be the same thing.

Method: one agent per seat plus three cross-cutting agents, each finding then
adversarially re-verified against the artifacts on disk. Every seat audit came
back `audit-needs-correction` -- the verifier caught real errors in all five,
including the most important finding below.

## Headline: t3's RoPE flavor was wrong

Seat t3 (2B-512 constant-LR fork) carried `complex` in field 12. It must be
`_real`.

`complex` launches `--config=agpt_2b`. But every t3 run in history launched
`agpt_2b_real`, and the fork's own `step-21300` was written 2026-08-17 -- long
after the 2026-06-25 cos_sin switch (5ffb850a1). Launching `complex` would have
loaded cos_sin-trained weights under complex RoPE and written **corrupted
checkpoints into a live 122-checkpoint production chain**. It would not have
crashed: a flavor mismatch loads cleanly and only shows up as loss.

The bad value came from resolving the flavor against the PARENT chain at the
SEED step. That is the wrong rule for a fork that plain-resumes.

> **THE RULE.** For a fork with an EMPTY field 8, resolve RoPE against the
> flavor that wrote **the fork's OWN latest checkpoint** -- not the parent's at
> the branch point. The seed's flavor governs only the very first launch into
> an empty dir; after that the fork has its own history. Disambiguate by
> checkpoint mtime vs the 2026-06-25 switch, or read `--config` off the last
> successful run's launch line.

Verified: `grep -oE 'agpt_2b[a-z_]*'` over t3's console logs in umbrellas
8756957 and 8764675 returns `agpt_2b_real` in both.

## Two mysteries, one cause

**"step-9200 resumes at ~6.5"** -- recorded as a corrupt checkpoint. It is not.
Job 8744247 logged `step 9201 loss 6.51705 grad_norm 18.4066`, re-converging to
2.93 by step 9300. That is the one-time complex-under-cos_sin transition cost at
the fork's first resume. step-9200 is fp32 with cosine **0.999** to its parent.

**The 2026-07-18 stage-2 NaN (job 8663177)** -- blamed on the olmo50/dolmino50
mix or the 2e-6 LR. Neither. It launched the complex-trained `step-92859` seed
under `agpt_2b_real` and resumed at `loss 7.23840 / grad_norm 46.4271`, where
the correct-flavor 512N twin resumes at `2.60254 / 0.2505`. It spent 3,800 steps
re-learning the scrambled Q/K pairing back to 2.63, then NaN'd at 3,801 on a
model driven through a large recovery excursion. The 512N seat runs PURE dolmino
at a **10x higher** LR and is clean past step 7,700.

Both were RoPE mismatches. Confirmed empirically today (see below).

## t4 retired: the seed was never a trained checkpoint

`.../n256-gbs6144-constlr-from9500/step-9500` is near-random-init weights
mislabeled step-9500. Seven attempts, zero real training steps.

| probe | seed | every healthy 2B ckpt |
|---|---|---|
| dtype | **bfloat16** | float32 only |
| size | 13G | 24G (exactly 2x) |
| `layers.0.attention_norm.weight` | exactly 1.0, **std 0.0** | 0.921 +/- 0.0177 |
| `tok_embeddings` std | 0.99999 (unit normal) | trained |
| absmax on projections | exactly 2.0000 (truncation) | -- |
| cosine to parent | wq +0.000663, wk -0.000178, wv +0.001649, head +0.003091 | ~1.0 |

All six off-diagonal wq/wk/wv pairings were also tested -- nothing above 0.9, so
it is not a permuted or mis-mapped tensor either. Its own `train_state` reports
`ntokens_seen = 155,648,000` where step 9500 at gbs 6144 x 8192 implies ~478e9 --
**325x too few**. The checkpoint contradicts its own label.

It resumed at loss 5.97 (not ln(256128) = 12.45 only because `output.weight`
retains a weak unigram prior) and went flat over 8 steps: that run was
pretraining from scratch at a fine-tuning learning rate.

No genuine 256N step-9500 exists anywhere -- the surviving `agpt-2b-v2` n256
chain starts at step 35600 -- so the seat could not be repaired, only redefined.

**This also exonerates `ckpt_key_compat`.** The remap was the leading suspect. It
is not the cause: a key rename cannot change a dtype, and step-9200 -- which does
NOT go through the remap -- is healthy. Exactly one of the two suspect
checkpoints was a remap consumer, not both.

## t4 replaced: 2B-256 stage-2 dolmino, and it is validated

The 256N twin of t0, seeded from the COMPLETED 256N stage-1 `step-92859`. That
chain finished: it exited cleanly 2026-07-03 with `Training starts at step 92860
/ Training completed`, having hit its budget (92859 * 6144 * 8192 = 4.6737T vs
the 4.673780159710T target). Earlier reports of "88 steps remaining" came from
reading W&B's last LOGGED step (92,772) rather than the last CHECKPOINTED step.

`rope=complex` verified, not assumed: `rope_flavor_for_step.py --chain
2b_v2_256 --step 92859` returns `2b`, and all 22 runs of that chain report `2b`
-- it never crossed the switch. (Contrast `2b_v2_512 @46429` -> `2b_real`, which
is why t0 correctly carries `_real`.)

**Smoke 8773381 (4 nodes, 8 steps) confirms it:**

```
config=agpt_2b            <- complex, correct
step 1  loss 2.59128  grad_norm 0.6761
validate step 1  loss 2.5527
step 8  loss 2.59126  grad_norm 1.3674
```

Against the pre-registered criteria: ~2.6-2.7 = correct flavor and good seed;
~7.2 = flavor still wrong. It landed at **2.591**, matching the 512N twin's
2.60254. The RoPE-mismatch diagnosis is now empirical, not circumstantial.

**Token budget is deliberately the same `2390375382006` as t0, not halved.**
Field 11 is tokens and steps = tok/(gbs*seq_len), so the same value gives 47,492
steps at gbs 6144 vs 23,746 at 12288 -- identical token exposure (291,790,848
samples both ways), which is what makes the two arms comparable.

Port 30000: 29700 belongs to t3, so the dead commented row could not be revived
verbatim.

## t1: the fork had never run, and its budget is wrong

The healthy `step 10600` previously attributed to t1 was the **canonical** chain,
not the fork -- `8764675` predates the fork row, and the fork dir did not exist
at all.

Its field 11 is `4673780159710`, the CUMULATIVE stage-1 target. Bootstrapped by
pre-copy the counter continues from 9000, so this is correct as written; the
overshoot only appears if the seat is seeded model-only via field 8 (counter
restarts at 0 -> 46,429 steps ON TOP of the 9,000 already done, +19.4%). Same bug
class as the t0 fix in cc4e22cfa.

**Resolution: pre-copy** (matches how t3/t4 were bootstrapped, and preserves the
SophiaG optimizer state that a model-only seed discards). Copying `step-9000`
(6144 shards, 244G) into the fork dir.

`step-9000` is the right source, not the current tip: decay onset for this chain
is `46429 + 1 - 200 - round(46429*0.8) = 9087`, so step-9000 is the last
pre-decay checkpoint and the 1,513 steps after it are exactly what the fork
exists to discard.

Copy mechanics: `.metadata` is written **LAST**. A resume scan treats its
presence as "this step is usable", so writing it first would expose a
half-copied 244G checkpoint.

## t0: blocked on a cold index cache

t0 is at step-7700, loss 2.519, 32.4% of its stage-2 budget, last progress
2026-08-17. Its row is correct; it is blocked operationally.

The blendcorpus cache key is an md5 over a descriptor including `Number of
samples`, derived from `training.steps`. Commit cc4e22cfa correctly fixed t0's
budget from the cumulative 7.064T to the stage-2 increment 2.390T, changing
steps 70,176 -> 23,746 (2.9553x). Every dataset hash changed with it, so the 84G
of Aug-16 cache is permanently unreachable. Do NOT "fix" this by reverting the
budget -- the cache is the disposable artifact.

Current state: 325 new-budget per-dataset descriptors exist, but only **1 of 7**
corpus-level entries. Pre-warming with `prewarm_blendcorpus_singlerank.sh`
(job 8773399), which is the right tool since the corpus build is serialized on
rank 0 anyway.

Note on a false alarm: the audit flagged the Aug-18 corpus entry as "torn" for
missing `_doc_idx/_sample_idx/_shuffle_idx`. The verifier refuted this -- the
corpus layer's artifact set is `{.dsc, _index.npy, _sample_index.npy}`; those
three siblings exist only at the per-dataset layer. All 15 corpus entries in the
tree look identical.

## Cache reuse for the new 256N seat (verified arithmetic, not assumed)

The 648 TRAIN descriptors are shared with the 512N seat because
`gbs * steps` is identical at both node counts:

```
512N  gbs=12288  steps=23746  -> 291,790,848 samples
256N  gbs= 6144  steps=47492  -> 291,790,848 samples
ceil(291790848 * 0.0032093698 * 1.005) = 941148  == on-disk value  MATCH
```

The 324 VALID descriptors will MISS -- `eval_samples = gbs * eval_iters` does
not cancel (614,400 vs 1,228,800). That is a one-time ~0.25G rebuild at startup,
expected, not a misconfiguration.

Symlinked 3,933 entries into the 256N seat's cache dir; 648 train `.dsc`
reachable, 0 broken links.

## Final seat table

| # | seat | N | dfl | rope | tokens | state |
|---|---|---|---|---|---|---|
| t0 | 2b-512 stage-2 dolmino | 512 | dolmino-mix-1124 | `_real` | 2390375382006 | step 7700; cache pre-warm queued |
| t1 | 20b-512 constlr from9000 | 512 | olmo-mix-1124 | `_real` | 4673780159710 | seed pre-copy in progress |
| t2 | 20b-256 canonical | 256 | olmo-mix-1124 | `_real` | 4673780159710 | healthy, step 11800 |
| t3 | 2b-512 constlr from9200 | 512 | olmo-mix-1124 | `_real` **(FIXED)** | 4673780159710 | step 21300 |
| t4 | 2b-256 stage-2 dolmino | 256 | dolmino-mix-1124 | `complex` | 2390375382006 | **NEW**, smoke-validated 2.591 |

Node need unchanged at 2098 (both the retired and new t4 are 256N).

## Still open

- **8769730 carries the pre-fix script.** PBS snapshots at qsub; it was
  submitted 2026-08-20 21:55:44, ~61 s after the RoPE commit efcae5419, so it
  captured the `complex` t3 row. It must be replaced before it runs, or t3 will
  train under the wrong flavor. Note PBS does not expose the snapshotted bytes,
  so this is inferred from timing, not read directly.
- **t2 decay onset.** The table comment says 9087; commit b08fccfd1's own W&B
  cross-check (predicted 2.25496e-05 vs observed 2.25502e-05 at step 9694) is
  reproducible only with **9287**. The chain is at ~11,800 either way, so this
  does not change urgency for t2 -- but the comment should be corrected.
- **t1's monotonic loss rise** across three consecutive jobs is pre-existing and
  independent of the LR bug. Unexplained.
- **`ckpt_key_compat` has never been validated end-to-end on a known-good
  checkpoint.** It is exonerated for this failure but not positively certified.
- **Disk.** 47,492 steps at CKPT_INTERVAL=100 with keep-latest-k=0 is ~475
  checkpoints x 24G = ~11.4T for the new seat alone.
- **66 NaN-poisoned checkpoints** (steps 3801-6600) from job 8663177 remain in
  `agpt-2b-stage2-olmo50dolmino50-const2e6-n256-gbs6144`. Known garbage; flagged
  for the user, not deleted.
