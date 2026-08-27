# Summaries

> **Canonical record:** this page is authoritative for the index of periodic retrospectives.
> Other pages cross-reference it; do not duplicate status here.

Periodic retrospectives covering the project at a higher level than
[`journal.md`](../../journal.md) (which is a per-session running log).
**Naming:** each summary is named for the **last day it covers**
(`YYYY-MM-DD.md`), so the files sort chronologically and the newest is always
last. The full date range stays in the document title and in the Period column
below. (Renamed 2026-08-14 from the older `START_to_END.md` form. End-date was
chosen over "Friday of that week" because two pairs of periods share a Friday --
`06-26_to_07-06` and `07-06_to_07-10` both land on 07-10 -- which would have
collided.)

## Quarterly / program reports

Higher-level rollups for external (INCITE / program) reporting,
synthesized from the two-week summaries below.

| Period | Report | Headline |
|---|---|---|
| 2026 Q2 (Apr 1 -> Jun 30) | [INCITE Quarterly Report](2026-Q2-incite.md) | 2B base pre-training COMPLETE (4.674T tokens); 20B leading per token; 80B launched; RL (SFT+GRPO) end-to-end on XPU; ~29% Year-2 Aurora burn |

## Index

| Period | Summary | Headline |
|---|---|---|
| 2026-04-12 → 2026-04-27 | [Two-Week Summary](2026-04-27.md) | 291 commits — built LR finder + scaling study + production training + optimizer competition platform |
| 2026-05-08 → 2026-05-22 | [Two-Week Summary](2026-05-22.md) | 51 commits — filed first upstream PyTorch PR (xccl `_set_pg_timeout` dispatch), 4 upstream syncs, 80B bad-node failover wrapper, silent-hang detection fix |
| 2026-05-22 → 2026-05-29 | [One-Week Summary](2026-05-29.md) | 80 commits — 2 upstream PRs (1 closed, 1 superseded by draft #3450), 5 upstream syncs absorbed, 20B 512N HSn 0.579→0.635, failover wrapper hardening, eval/chart infrastructure overhaul |
| 2026-06-05 → 2026-06-12 | [One-Week Summary](2026-06-12.md) | ~140 commits — SFT 2B × tulu_math_uc_mix completed (729 steps), GRPO 8N production run done, 4 upstream syncs (50-53), PR #14 review with full A/B + 1.94× speedup confirmed, vLLM-XPU sibling venv, olmo-mix-1124 8N path documented, ambivalent + Iosevka chart restyle, os._exit hang fix |
| 2026-06-12 to 2026-06-26 | [Two-Week Summary](2026-06-26.md) | 151 commits -- 80B grad-path NaN root-caused (2 triggers: LBS>1 + dp_degree>186), TP=4/LBS=1/bf16 stable corner confirmed + stress-tested to 4-16x batch (512N/1024N/2048N global-batch sims), GRPO RL end-to-end on XPU via TRL vllm-serve, native ezpz launch --auto-retry scripts, 4 upstream syncs (56-59), PR #14 merged, blendcorpus index-race + barrier lesson |
| 2026-06-26 to 2026-07-06 | [~10-Day Summary](2026-07-06.md) | 168 commits -- 2B 256N base COMPLETE (4.674T tokens, 100%) + eval closeout, 2B CPT olmo/dolmino sweep launched, production-batch LR finders (only 80B cliffs; 2B/20B never do), 80B convergence run (all 3 optimizers NaN past warmup), 80B post-PM launch (2048N SIGSEGVs at init), multi-node vLLM GRPO cross-node gen works / multi-trainer-node walled + root-caused, 4 upstream syncs (60-63), native auto-retry umbrella + 20B chain recovery, blendcorpus race fixed at source |
| 2026-07-06 to 2026-07-10 | [~4-Day Summary](2026-07-10.md) | ~40 commits -- big-mix SFT data pipeline: vectorized interleave (~90min -> ~5s, bit-identical, upstream PR #8318) + offline pre-tokenize (SFTTrainer's ~8.5h runtime tokenize moved to a 1N job), big-mix SFT relaunched on gs138650 base (seq=2048, ~54B tokens), v2-base 32N oneCCL scale crash root-caused + sidestepped, 32N attempt-1 "crash" was the tokenize watchdog (not oneCCL), HF 429 root-caused (streaming bypasses rank-0 guard), 3 upstream syncs (65-67, all RL/CI-only) |
| 2026-07-10 → 2026-07-26 | [~16-Day Summary](2026-07-26.md) | ~376 commits -- RL/CoT campaign (Stages 0-2: cold-start CoT-SFT is the accuracy lever, gated GRPO drift-proof but not the 2B accuracy lever; ceiling-attack reward +168%; GRPO+LoRA reproduced+vendored+ported to agpt-2b on XPU), full-mix SFT COMPLETE but deliverable=checkpoint-900 (8672 overfit-forgot), 80B NaN root-caused (bf16 residual-stream overflow, optimizer-independent; fp32-residual insufficient), modern eval-suite review + mmlu/gsm8k backfills, DCP optim-format converter PROVEN + 2B-512 HEAD-migration rehearsal, production loss stack consolidated (prod_dash board + shared wandb_fetch + MDS stage lines), synthetic-summary POC, 20B->constant-LR + capacity-queue bridges vs starvation, 3 syncs (68-70), macOS local training |
| 2026-07-26 to 2026-08-10 | [~15-Day Summary](2026-08-10.md) | ~176 commits -- data MIX (not LR schedule) is the mid-training lever; MMLU flatline traced to the data with a validated harness; ARC-C "decline" retracted as a shot-count collision; umbrella seated twice; 4 silent doc-refresh failures fixed |
| 2026-08-11 → 2026-08-14 | [Week Summary](2026-08-14.md) | 44 commits -- 2B-512 canonical chain COMPLETE at 4.674T; Sunspot oneCCL broke ALL reducing collectives (reduce_scatter + all_reduce segv, all_gather fine) and the frameworks RC fixes it; RC re-validation 6/6 PASS incl. compile TP=4 and MoE EP=1 above 2N; 80B bf16 NaN reproduces on the RC at step 30 (stack-independent) while fp32-acts reveals bf16 computes a genuinely DIFFERENT gradient (grad_norm ~8 vs 17K-91K); config.job.dump_folder regression broke every run for ~5h; **§10 (08-16)**: the 30B goes from a config-less proposal to a measured model -- 27.89% MFU at 2N and 25.54% at 64N (1.42%/doubling vs the 2B's 13.96%), TP hurts monotonically, HSDP works on the 20B but not the 30B (size is the axis; not an OOM despite 18 GiB free), and agpt_30b_llama3tok -- which had never run once -- reaches LBS=4 / 28.99% MFU on freed HBM |
| 2026-08-14 → 2026-08-21 | [Week Summary](2026-08-21.md) | ~293 commits -- BOTH 2B stage-1 chains COMPLETE (512N 4.667T, 256N 4.669T) and the dolmino stage-2 arm 32% in; checkpoint export was permuting Q/K against the wrong RoPE convention, so every 20B eval after the cos_sin switch was measured on a scrambled model (the apparent ARC-C decline); a seat audit caught a t3 RoPE defect one commit before it would have corrupted a live 122-checkpoint chain and retired t4 whose "step-9500" seed was never a trained checkpoint; a latent NaN-gradient gap closed (never fired in prod); 30B converged (2000/2000, 12.028 -> 2.115, zero NaN) |
