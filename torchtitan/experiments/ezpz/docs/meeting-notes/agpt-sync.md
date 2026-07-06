# AuroraGPT Sync — Meeting Notes

> Sam Foreman. Most recent first.

---

## 2026-07-06

### Headline: 2B base closed out (eval) and three new fronts opened -- CPT sweep, multi-node GRPO solved, first SFT on the completed v2 base

The week since 2026-06-29 resolved the entire post-PM action list and added
three new deliverables. Full detail in the
[~10-day summary](../summaries/2026-06-26_to_2026-07-06.md).

### 1. 2B base eval closeout (resolves 2026-06-29 action #1)

The final 2B checkpoint (**step-92,859 = 4.674T tokens**) was converted and
evaluated. Tail backfill (job `8638581`, 14 ckpts step-86,500..92,859):
**HellaSwag_norm 0.561, ARC-Easy 0.651, PIQA 0.733** at the final step, flat
over the last ~635B tokens (also resolved the old ARC-Easy ~0.59 artifact -- a
fresh eval reads ~0.65). The **dead-flat tail is the motivation for CPT** (item
2), not more base tokens. Table:
[`evals/agpt/2b`](../evals/agpt/2b/README.md).

### 2. 2B continued-pretraining (CPT) mixing-ratio sweep launched (resolves action #2)

With the base plateaued, launched a CPT pilot that forks the completed base and
continues on new data blends.

- **Fork mechanism:** `--checkpoint.initial-load-path` (weights-only) off the
  read-only step-92,859 base in a separate clone; distinct CKPT_DIRs so the base
  is never overwritten.
- **3 renormalized mixes:** `dolmino-mix-1124` (100), `olmo50-dolmino50`,
  `olmo25-dolmino75`. **256N pilots** `8638977` (dolmino-100) + `8638978`
  (olmo50/50) + afterany conts; GBS held 6144 (LR calibrated), LR 2.28e-5
  re-warm 200 + decay 0.8, ~300B tokens.
- **Fork verified** (smoke `8638933`): loads step-92,859 clean (0 mismatch),
  loss 7.2 -> 5.9 -- dolmino is a real distribution shift, so the CPT signal is
  live. Report:
  [`20260701-2b-cpt-olmo-dolmino-sweep`](../experiments/agpt/aurora/20260701-2b-cpt-olmo-dolmino-sweep.md).
- **Bug flagged:** the 2B autoretry script defaults async checkpointing, which
  is XPU-broken on this torch (`new_group(gloo)` -> `No backend type for xpu`).
  Workaround `CHECKPOINT_ASYNC_MODE=disabled` (20B already defaults disabled); a
  2B-default fix is TODO.

### 3. 80B: convergence NaN + 2048N init crash (resolves actions #2/#4)

- **Convergence run @ GBS=6144: all 3 optimizers NaN** at finder LRs past
  warmup -- a corner instability, not a tuning problem (only 80B cliffs; 2B/20B
  never do). Report:
  [`2026-06-30-80b-convergence`](../experiments/agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md).
- **Post-PM production launch:** 512N + 1024N ran; **2048N SIGSEGVs in
  `set_determinism`** at 24,864 ranks (init-time scaling wall, same class as the
  known 1024N crash). [`80b/README`](../production/agpt/80b/README.md).
- **Still open (team decision):** 80B optimizer -- SophiaG @1e-6 (launched) vs
  **mano @3e-6** (wider stability margin for a long unattended run); and whether
  to keep pushing scale above 512N given the 2048N init crash.

### 4. Multi-trainer-node GRPO on XPU -- SOLVED (new)

Multi-*trainer*-node GRPO (FSDP across 2+ nodes) now works: job `12470083`
(3N = 1 server + 2 trainer, 24 ranks) ran **8/8 steps**, loss -0.028,
**accuracy_reward 0.375**, real cross-node FSDP grad reduce-scatter. The
2026-07-01 overnight "desync hang" was **two ordinary bugs, not a desync**:
(a) the `--no-oneccl-tcp-kvs` runs left `CCL_ATL_TRANSPORT` at oneCCL's `mpi`
default, which SIGSEGVs forming the cross-world weight-sync PG; and (b) the
AVG->SUM FSDP patch rebound the wrong `from`-import name so it never ran. Both
fixed. Root cause + all-rank stack evidence:
[`2026-07-06_multinode-grpo-root-cause`](../rl/2026-07-06_multinode-grpo-root-cause.md);
status flipped to works in
[`grpo-on-xpu-status`](../rl/grpo-on-xpu-status.md).

### 5. First SFT on the completed v2 2B base (new)

Every prior production SFT (and thus all GRPO) used the older
`AuroraGPT-2B-sophiag-gs138650` base. Launched the first SFT on the **completed
v2 256N base** (step-92,859): converted DCP->HF on Aurora, transferred to
Sunspot, ran the proven `tulu_math_uc_mix` recipe. 2N smoke green; 32N full run
`12470088` in progress (loss 1.35 -> 0.95, mean_token_accuracy 0.68 -> 0.757,
checkpoint-100 saved, fresh CKPT_DIR confirmed). Trajectory:
[`sft/agpt-2b-v2-256n`](../production/sft/agpt-2b-v2-256n/tulu_math_uc_mix/README.md).
**Sets up an eval question:** does SFT on the completed base beat SFT on the
older lineage? (head-to-head once it finishes.)

### 6. Upstream syncs 60th-64th absorbed

Four syncs in the summary window (60th-63rd) plus the **64th** (2026-07-06,
6 commits: graph_trainer + a triton-upgrade loss asset, no replay). The 64th's
`sync_smoke.sh` caught a **latent bug unrelated to the merge**: the NaN-abort
guard read `config.training.nan_abort_consecutive` but the field lives on the
top-level trainer `Config`, so `trainer.train()` `AttributeError`'d on EVERY
agpt/moe run (SFT/GRPO unaffected -> unnoticed). Fixed. See
[`upstream-sync`](../upstream-sync.md).

### 7. Infra

Native `ezpz launch --auto-retry` umbrella replaces the legacy `failover_lib.sh`
wrapper; 20B 512N chain recovered; blendcorpus index-race fixed at source
(atomic-rename). Detail in the summary.

### Decisions / asks for the team

1. **80B optimizer:** keep SophiaG @1e-6, or switch the base to mano @3e-6 for
   the wider margin? (Easy to switch before the brackets run.)
2. **80B scale ceiling:** 2048N crashes at init; cap production at <=1024N until
   the `set_determinism` init-scaling issue is understood?
3. **2B CPT direction:** once the dolmino/olmo mixes report, which wins ->
   promote to the CPT base?
4. **SFT base:** eval v2-base SFT (`12470088`) head-to-head vs the gs138650 SFT
   to decide the canonical instruct checkpoint.

---

## 2026-06-29

### Headline: 2B v2 256N pre-training is COMPLETE (4.674T tokens, 100% of target)

The 2B 256N v2 chain reached its full **4.67T-token budget**: final
checkpoint **step-92,859 = 4.674T tokens (100.0%)**, final loss **2.652**,
grad_norm ~0.056, ~14% MFU. Chain head `8558531` (cont12) finished a clean
exit-0 ~10.2h run on 2026-06-29 03:03 UTC. This is the first agpt model to
finish the full v2 (fp32-master) base pre-training run end to end.

**Discussion / decisions for the team:**
- **Eval the final 2B checkpoint.** Convert step-92,859 DCP -> HF and run the
  full lm-eval suite (ARC-E/C, HellaSwag, Winogrande, etc.) so we have the
  end-of-pretraining scorecard. Blocked until Aurora returns from PM (lm-eval
  needs compute). Queue it first thing.
- **What's next for 2B?** Options: (a) start CPT (continued pre-training) on
  additional tokens with the constant-LR config we just built, (b) SFT, (c)
  freeze it as the reference base. The constant-LR (decay_ratio=0) plumbing is
  ready -- see the 80B launch item below.

### 80B production launched at SophiaG / constant-LR / scale brackets

Submitted the first real 80B v2 production runs (queued, will start post-PM):
**512N + 1024N + 2048N simultaneously**, SophiaG @ LR=1e-6, **constant LR after
warmup (no decay)** for the planned CPT regime, validator ON (95/5 split).

- **Optimizer/LR is the discussion point.** Per the 2026-06-27 production-batch
  LR-finder, AdamW @ GBS~6144 is on a NaN cliff (LR=1e-6 past it); the finder's
  safest pick is **mano @ ~3e-6**, with SophiaG @ ~1e-6 as the lowest-loss but
  narrow-band alternative. We launched **SophiaG @ 1e-6**. Worth a team
  decision: stick with SophiaG, or switch the 80B base to mano for the wider
  stability margin on an unattended multi-T-token run?
- **Scale is unvalidated above ~512N.** 1024N (12,288 ranks) is documented to
  crash at init (`set_determinism`); 2048N (24,576 ranks, dp_degree~6138) is 2x
  that and untested. The 512N bracket is the safety net; submitting all three
  is itself the scaling experiment. Expect possible init crashes at 1024/2048N.
- A 4N pre-check confirmed the config wiring + clean SophiaG descent before
  committing the big allocations.

### Infra fixes this week (all landed + pushed)

- **blendcorpus cold-cache index race FIXED** (was blocking any fresh-CKPT_DIR
  80B run). Root cause: at TP>1 the non-rank-0 readers raced rank-0's index
  write. Fixed at source in `saforem2/blendcorpus` -- atomic writes + poll
  (`041d015f`) + a TOCTOU follow-up (`1f7e9c0`). Confirmed end-to-end: cold 4N
  TP=4 80B now builds the index and trains with 0 EOFError. **80B no longer
  needs a manual cache prewarm.**
- **Validator at 80B TP=4: the "CCL deadlock" was a phantom** -- 3 unrelated
  bugs (cold-cache mmap race, a training-side barrier stall, a `loss_fn`
  tuple-unpack crash), all fixed. Validation CONFIRMED working at TP=4
  (job 12469784). `VALIDATOR_ENABLE=1` is now safe; 95/5 split is the default.
- **Checkpoint-resume incident (recovered).** A clone `git pull` advanced the
  prod clones past an optimizer state-dict format migration (#3623/#3269,
  nested->flat), breaking DCP resume. Recovered by pinning clones pre-#3623;
  resume verified. The clones stay pinned -- a nested->flat migration shim is
  hard/risky and not worth it (chains resume fine pinned).
- **walltime-aware checkpointing** added to the ezpz trainer (force a final
  ckpt before walltime) + a job-absolute-deadline fix so it survives failover
  retries.

### 20B 256N status

Running through the PM boundary; finished clean at **step-2,100 = 105.7B tokens
(2.3%)**, loss **2.85**, ~21.8% MFU. Resumes post-PM via `8558549`.

### Going into the PM maintenance (2026-06-29 06:00 -> 07-01 03:30 UTC)

Both 256N chains exited with valid checkpoints (2B step-92,859, 20B step-2,100);
all continuations + the 6 80B jobs are queued to resume/start post-maintenance.
Nothing at risk.

---

## 2026-05-04

### bf16-master RMSNorm-freeze fix is producing real downstream gains

Quick recap for context. All v1 production runs (2B / 20B / 80B) had
`training.dtype = bfloat16`, which kept the master parameter copy in
bf16. RMSNorm.weight initializes to 1.0; the bf16 ULP at 1.0 is
~7.8e-3 and per-step optimizer updates for those parameters are ~1.6e-5,
so every update rounded to zero and **norm weights never moved from
1.0 for the entire run**. Other parameters (linears, embeddings)
initialize at much smaller scales and updated fine, so training loss
curves looked plausible — the bug only became visible at eval time.

Default flipped to `float32` on 2026-04-30 and **v2 production was
restarted from scratch** rather than continuing from the bf16-tainted
checkpoints. The v1-vs-v2 lm-eval comparison is the smoking gun:

- 2B ARC-Easy climbed **0.277 → 0.429** over 100B tokens on v2,
  vs v1's flat ~0.27 across **450B** tokens.
- **+19.8pp ARC-Easy / +15.4pp HellaSwag at 503B tokens** vs v1's
  flat baseline.
- v1's flat trajectory across 450B+ tokens is the qualitative
  signature of the bug — a model with frozen normalization cannot
  improve on what lm-eval measures, no matter how much data it sees.
- 2B eval comparison:
  [`docs/evals/agpt/2b/`](../evals/agpt/2b/README.md);
  20B eval comparison:
  [`docs/evals/agpt/20b/`](../evals/agpt/20b/README.md);
  full diagnosis + cross-linked evidence:
  [`docs/guides/training-dtype-bf16-norm-freeze.md`](../guides/training-dtype-bf16-norm-freeze.md).

### Production status

- **2B 512N canonical chain** (`8460301 → 8463626 → 8463627`):
  step **5,073**, loss **2.97**, **510B tokens / 10.9% of 4.67T
  target**. Continuation 8463627 queued, follow-up 8466847 held on
  `afterany:8463627`. Trajectory page:
  [`docs/production/agpt/2b/n512/`](../production/agpt/2b/n512/README.md).
- **20B 512N canonical chain** (`8460302 → 8463628`): step **~862**,
  loss **~3.47**, MFU ~17.8%. 8463628 currently running near
  walltime; 8466848 held on `afterany:8463628`. Trajectory page:
  [`docs/production/agpt/20b/n512/`](../production/agpt/20b/n512/README.md).
- **20B 256N (8463659):** ran 9h walltime then **NODE_FAIL after
  step 364** (loss 4.61, 18.3B tokens). `shepherd died from signal 9`
  on `x4406c6s7b0n0`, PBS exit -20 — same recurring Aurora bad-node
  failure mode as 8459818 / 8460301. **step-300 ckpt saved cleanly,
  resumable.** Throughput on this run was bouncing 21-410 TPS
  depending on flare contention (1-20% MFU). Trajectory page:
  [`docs/production/agpt/20b/n256/`](../production/agpt/20b/n256/README.md).
  Are these recurring `signal 9` crashes being tracked anywhere?
  They've now killed three long-walltime jobs across three different
  nodes — worth raising with ALCF support if not.

### 80B blocker

- 80B `compile + AC + TP=2` still hits the
  `tensors_saved_with_vc_check` AOT autograd assertion (`DeviceMesh`
  leaks into saved-for-backward tensors). Toy minimal repro doesn't
  fire — bug needs the real `Module.parallelize` + `LocalMapConfig`
  path that torchtitan uses.
- **2026-05-03:** added `agpt_50b_wide` (dim=9216, 48 layers, ~48B
  params,
  [`98a02d04`](https://github.com/saforem2/torchtitan/commit/98a02d04))
  as a smaller bisect target. 2N + torch 2.10 smoke ran 10/10 steps
  cleanly. Concluded "bug is depth-sensitive — does NOT reproduce at
  48 layers." That conclusion turned out to be wrong (see below).
- **2026-05-05 correction:** ran a proper three-config bisect on torch
  2.13 (job 12465952 4N + job 12465962 2N). All three configs
  (`agpt_50b_wide` 48L, `agpt_70b_wide` 72L, `agpt_80b` 84L) **crash
  identically** with the same assertion. Smallest tested:
  `agpt_50b_wide` on 2N takes ~30s to crash. **Bug is
  torch-version-sensitive, not depth-sensitive.** The May 3 result was
  a torch-2.10 artifact (the failing assertion in
  `_AutogradSavedState.save_from_forward` likely doesn't exist or
  isn't reached on the older AOT-autograd code path). Working repro
  bracket: torch 2.10 (any depth) ✓ → torch 2.13 (every depth tested) ✗.
- Workaround in the meantime: `compile=OFF` for any 80B-family config
  on torch 2.13, OR pin to torch 2.10 for those configs.
- **Working v2 80B path validated 2026-05-05** (job 12466025, 4N
  smoke): `agpt_80b @ TP=2, AC=full, compile=OFF, AdamW LR=1e-6,
  fp32-master` on torch 2.13. Loss descended **12.98 → 10.46** over
  20 steps, MFU steady at **~17.8%** (matches v1 compile-on baseline),
  memory peak **88.94%** with ~7 GiB headroom. Production setup just
  needs to add the 200-step linear warmup the 2B/20B v2 configs use,
  then it's ready to launch.
- Initial toy repro (legacy `parallelize_module` — does NOT fire,
  needs the new sharding API):
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../upstream-issues/repro_devicemesh_in_saved_tensors.py).

### Open work I'm holding

- **Validation loss wiring:** blendcorpus's existing val split (5%
  slice) is now plumbed through `EzpzValidator` (subclass that also
  fixes the TP loss-reporting bug — see "Other notes" below). Default
  `enable=False` so production isn't disturbed. Smoke test not yet
  done. Do we want held-out NLL on production runs, or are downstream
  lm-eval scores at checkpoint cadence the right signal?
- **80B production** — still has open issues from prior sessions
  (LR=1e-6 stable but bad-node Gloo timeout crash at step 51).
  Worth flagging if 80B production is on the agenda.

### Other notes

- **TP loss-reporting bug found, fix filed upstream + locally
  workaround.** Upstream `_dist_reduce` change
  ([`pytorch/torchtitan@1786292d`](https://github.com/pytorch/torchtitan/commit/1786292d),
  2026-04-27) skips the cross-batch `all_reduce` when `loss` is a
  DTensor on a mesh orthogonal to `loss_mesh`, so reported loss on
  any TP > 1 run is `true / dp_world_size`. **No current production
  runs are TP > 1**, so no live dashboards are affected — but if/when
  we restart 80B production at TP=2, the historical 80B v1 W&B traces
  show `loss / 1536`. Gradients/optimizer steps were unaffected; only
  the printed value was wrong. Fix filed at
  [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204)
  (mergeable, awaiting maintainer review). Local ezpz workaround in
  [`a24ed2e1`](https://github.com/saforem2/torchtitan/commit/a24ed2e1)
  + [`a0b9b13d`](https://github.com/saforem2/torchtitan/commit/a0b9b13d).
  Full diagnosis:
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../guides/loss-reporting-tp-dist-reduce.md).

### Action items

- [ ] (Sam) Smoke-test `EzpzValidator` end-to-end on `agpt_2b` once
      consensus on whether to enable val.
- [ ] (Sam, blocked on review) Push for review on
      [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).
- [ ] (?) Volunteer to build LocalMapConfig-based minimal repro for
      the 80B compile + AC + TP=2 crash.
- [ ] (?) Decide cadence for held-out validation loss on production.
- [ ] (?) Decide whether to file an ALCF support ticket for the
      recurring `shepherd died from signal 9` NODE_FAIL pattern
      (jobs 8459818, 8460301, 8463659 — three crashes, three
      different nodes). Resume 8463659 from step-300 in the meantime?
