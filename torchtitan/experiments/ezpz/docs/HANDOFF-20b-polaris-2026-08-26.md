# Handoff -- Polaris 20B chain, 2026-08-26

Snapshot of session `b09e4b10-ba26-4278-92b7-9cdc17634795` at handoff to
`mbph`. Written so the resumed session does not have to reconstruct this
from the transcript.

**Status: handoff COMPLETE.** The session is running on `mbph` as of
2026-08-27 03:29 UTC -- transcript slug
`-Users-sam-projects-saforem2-torchtitan`, embedded `cwd` rewritten, memory
dir carried across (133 files). Both far-side blockers named at the bottom
of `scripts/handoff_to_mbph.sh` are cleared: ALCF ssh works
non-interactively from here (`polaris-login-04`, rc=0), and the checkout is
at `3bf235f52` with the grad-norm fix `40de81570` as an ancestor. Do not
re-run the script from `mbph` -- it targets `mbph` as the remote and there
is no `mbph` ssh identity in this account.

## Live state

| Job | State | Notes |
|---|---|---|
| `7560196` `agpt20b-p1` | Q in `backfill-large` | resumes step-5600 |
| `7560197` `agpt20b-p2` | H, `afterany:7560196` | continuation |

Both `-A AuroraGPT`, `select=130` (128 train + 2 spare), 12 h, submitted
2026-08-25 20:45 UTC from
`/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan`.

Queued ~26 h as of this writing. PBS published an estimate of 21:39 UTC on
08-26, then withdrew it (`--`). Not a priority problem: `Priority = 0`,
`base_score = 0`, `project_priority = 25`, same as every competing job.
The cluster is simply full -- 537/562 job-exclusive, 12 offline. Routed to
`backfill-large` rather than `large` because the AuroraGPT allocation is
negative (`burn_ratio 1.68`); `resources_min.burn_ratio = 1` on that queue.

## What to check the moment it starts

1. **Resume step is 5,600**, not 0. A step-0 start means it did not find
   the checkpoint -- kill it rather than let it overwrite the chain.
2. **Step-1 loss ~2.2**, not ~12.9. A fresh-init loss is the same failure.
3. **The grad-norm guard survives the first post-warmup step.** That path
   was broken until 08-26 (see below) and has never run green on hardware.

## Two clones, and which one is live

- `/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan` -- **the live
  chain.** `agpt-20b-sophiag-dolma-n128-gbs1024`, 58 ckpt dirs, latest
  `step-5600` (2026-08-16), 512 shards + `.metadata`.
- `/eagle/datascience/foremans/projects/saforem2/torchtitan` -- where the
  debug/gate smokes ran. Only holds `agpt-20b-sophiag-dolma-n4-gbs32`.

They have SEPARATE `outputs/checkpoints/`. Checking one and concluding "no
chain exists" is how a resume becomes a step-0 restart under the same
ckpt-dir name; jobs 7559749/7559750 were submitted that way and deleted
while queued. The account is the tell: `-A AuroraGPT` goes with the
AuroraGPT checkout.

## Resume was verified cold, before launch

Against `step-5600`, on the clone's own venv:

| Check | Result |
|---|---|
| model keys, code vs ckpt | 579 = 579, exact, zero drift either way |
| head spelling | `lm_head.weight` -- post-rename, no shim needed |
| optim layout | 579 `.fused` keys -- post-#3623 flat |
| SophiaG state | `exp_avg`/`hessian`/`step`, 579 each |
| param groups | all 9 entries, 579 each |
| `train_state.step` | 5,600 |
| `lr_scheduler.last_epoch` | 5,600 -- LR resumes in phase |

Neither of the two known resume-killers applies here: this checkpoint was
written by code already past the `output` -> `lm_head` rename and past
\#3623.

`train_state.ntokens_seen = 91,750,400` looks 512x low but is correct:
`ntokens_seen` accumulates PER-RANK (`trainer.py:738`), the `dist_sum` at
`:984` is for logging only, and `state_dict()` at `:1077` persists the raw
per-rank value. `5600 x (1024/512) x 8192` to the token. The dashboard
reads the reduced value from W&B, so nothing consumes it wrongly.

`dataloader.consumed_samples = 0` -- BlendCorpus does not persist a cursor,
so each leg re-draws from the shuffle start. Pre-existing for every leg.

## Two bugs caught in this session

**1. `[B, L*N, H]` attention is a silent no-op.** Commit `5b4a81803`
(2026-08-24, on `origin/ezpz`) "fixed" the post-#4121 error by reading dim 0
as the batch dim. That makes SDPA see sequence length 1, so causal attention
over one token is softmax of a single score = 1.0 and the output is EXACTLY
`v` -- `max|out - v| = 0.000e+00`. The model degenerates into a
position-wise MLP; it trains, loss descends, nothing raises. The GQA ratio
check PASSES under it (`131072/32768 == 4 == n_q/n_kv`, both sides scale by
L), so that check is not evidence. `42f4edfaa` (fold the dataloader to flat
`[T]`) is the correct fix and is what HEAD does. Guarded by
`tests/test_attention_layout_no_op_guard.py` (9 CPU tests). No production
impact -- step-5600 predates the bad commit by 8 days.

**2. The grad-norm guard NameError.** `40de81570` fixes a bare `grad_norm`
read in `train()` where the name is local to `train_step()`. Any run with
`--grad-norm-abort > 0` died on its FIRST post-warmup step. The submit
script defaults `GRAD_NORM_ABORT=20.0` and always passes the flag, so both
queued legs were aimed straight at it. The clone was pulled to `1bb11aac5`
to pick this up -- `_last_grad_norm` published at `trainer.py:1186`, guard
reads it at 1400-1406. PBS froze the submit SCRIPT at qsub but `trainer.py`
is imported at launch, so the queued jobs get the fix with no resubmit.

## Clone hygiene

The AuroraGPT clone was pinned 69 commits back and was deliberately pulled
to HEAD (the fold fix lives upstream). It is therefore NO LONGER PINNED:
anything pushed to `origin/ezpz` is one `git pull` away from landing under
a live chain. Three sessions ({polaris,aurora,sunspot}-tt-ezpz) are working
in these trees concurrently -- HEAD moved under this session twice. Consider
re-pinning once leg 1 starts.

`ezpz` in that venv was upgraded 0.25.0 -> 0.27.3 (`uv pip install --no-deps`,
torch verified unchanged at 2.13.0+cu129). 0.25.0 had NO `failover/` package
at all, so Polaris bad-node failover was blind. All 5 patterns now register.

## Not transferred by the handoff

Background Monitors do not survive. This session had two: job state on
`7560196`, and a watcher on the prod clone's HEAD + grad-norm-guard
regression. Re-arm if wanted.

`mbph` cannot ssh to Polaris non-interactively (ALCF MFA). Authenticate by
hand there before expecting `qstat` to work.
