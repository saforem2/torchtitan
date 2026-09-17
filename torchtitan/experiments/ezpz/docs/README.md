---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

> Living documentation for the `experiments/ezpz/` work. Sections
> ordered by importance (live → reference → outbound). Within each
> section, rows are sorted newest first by last-commit date.
>
> **Looking for something specific?** See [`TREE.md`](./TREE.md)
> for a single-page annotated tree of every directory and file
> under `docs/`, with descriptions of what goes where.

## Recently Updated

The 25 most-recently-changed docs by git commit date (across all 264
docs, not just the curated tables below). Auto-generated -- do not edit
by hand; run `utils/refresh_docs_readme_table.py` (or `refresh_all.sh`).

<!-- BEGIN recently-updated (auto-generated) -->
| Modified | Doc |
|---------:|-----|
| 2026-09-17 | [Production Training Runs — Aurora](./production/README.md) |
| 2026-09-16 | [Session state before Claude restart -- 2026-09-15](./session-state/2026-09-15-handoff.md) |
| 2026-09-16 | [Development Journal](./journal.md) |
| 2026-09-16 | [Claude Session Log](./claude-sessions.md) |
| 2026-09-15 | [Monitor snapshot -- 2026-09-15 19:10 UTC (ahead of a Claude Code update)](./MONITORS-restart-20260915.md) |
| 2026-09-09 | [12474810: a gradient instability developing at a defensible LR](./experiments/80b-gradient-growth-12474810.md) |
| 2026-09-08 | [Umbrella std::bad_alloc at init -- intermittent, not yet root-caused](./guides/known-bugs/umbrella-bad-alloc-init.md) |
| 2026-09-08 | [The 80B NaN: what is known, what is refuted, what is open](./guides/known-bugs/80b-nan-what-we-know.md) |
| 2026-09-08 | [Does gradient concentration track dp? A prediction and its falsifier](./experiments/80b-skew-vs-dp-prediction.md) |
| 2026-09-08 | [Adversarial review of the dp experiment: what it caught, what it missed](./experiments/80b-dp-design-review-2026-09-08.md) |
| 2026-09-08 | [dp=192 vs dp=96: the cleanest single-variable test in this investigation](./experiments/80b-dp-bracket-12474802.md) |
| 2026-09-07 | [12474761: the first valid reproduction of 8574385](./experiments/80b-rescaled-reproduction-12474761.md) |
| 2026-09-07 | [Separating batch size from parallelism: GBS is not the variable](./experiments/80b-gbs-vs-dp-separation.md) |
| 2026-09-06 | [Upstream Sync Log](./upstream-sync.md) |
| 2026-09-06 | [Summaries](./summaries/README.md) |
| 2026-09-06 | [Week ending 2026-09-06](./summaries/2026-09-06.md) |
| 2026-09-06 | [A short run silently rewrites your LR schedule](./guides/known-bugs/warmup-clamp-silently-voids-short-reproductions.md) |
| 2026-09-06 | [ALCF ticket draft: sunspot compute nodes failing home mount check](./guides/known-bugs/sunspot-home-mount-ticket-draft.md) |
| 2026-09-06 | [Sunspot: rack x1921 fails the home mount check; 64N jobs cannot run](./guides/known-bugs/sunspot-home-mount-check-offlines-nodes.md) |
| 2026-09-06 | [Gradient clipping runs on every 80B step, and it changes what a NaN means](./guides/known-bugs/clipping-hides-and-then-reveals-the-overflow.md) |
| 2026-09-06 | [12474740: an accidental LR-ceiling sweep](./experiments/80b-lr-ceiling-sweep-12474740.md) |
| 2026-09-06 | [12474733: five clean steps that are not as clean as they look](./experiments/80b-capture-12474733-findings.md) |
| 2026-09-02 | [Two days ending 2026-08-31](./summaries/2026-08-31.md) |
| 2026-09-02 | [SophiaG: a RECURRENT grad-norm blow-up at 30B](./guides/known-bugs/sophiag-stochastic-divergence-30b.md) |
| 2026-09-02 | [Fixed-batch optimizer comparison: AdamW vs Mano vs SophiaG](./experiments/optimizer-comparison/README.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-09-01 | [AuroraGPT Sync — Meeting Notes](./meeting-notes/agpt-sync.md) |
| 2026-09-01 | [MOVED -- and the title was wrong](./guides/known-bugs/80b-nan-rate-not-overflow.md) |
| 2026-09-01 | [AuroraGPT MMLU sits at chance because the models answer with a letter prior](./evals/mmlu-letter-prior-at-chance.md) |
| 2026-08-31 | [80th upstream sync: what works, what is deferred, what it costs](./upstream-sync-80th-status.md) |
| 2026-08-31 | [79th upstream sync -- 26 commits, four stacked defects, all from one PR](./upstream-sync-79.md) |
| 2026-08-31 | [INCITE Quarterly Report — Q2 2026 (Apr 1 – Jun 30)](./summaries/2026-Q2-incite.md) |
| 2026-08-31 | [Week ending 2026-08-29](./summaries/2026-08-29.md) |
| 2026-08-31 | [2026-06-26 to 2026-07-06 -- ~10-Day Summary](./summaries/2026-07-06.md) |
| 2026-08-31 | [2026-06-12 to 2026-06-26 -- Two-Week Summary](./summaries/2026-06-26.md) |
| 2026-08-31 | [2026-06-05 → 2026-06-12 — One-Week Summary](./summaries/2026-06-12.md) |
| 2026-08-31 | [Two-week summary — 2026-05-08 → 2026-05-22](./summaries/2026-05-22.md) |
| 2026-08-31 | [Two-Week Summary: 2026-04-12 → 2026-04-27](./summaries/2026-04-27.md) |
| 2026-08-31 | [AuroraGPT-2B Scaling](./scaling/agpt-2b.md) |
| 2026-08-31 | [AuroraGPT-20B Scaling](./scaling/agpt-20b.md) |
| 2026-08-31 | [GRPO+LoRA on Intel XPU: Sunspot reproduction (Monarch + TorchStore + vLLM)](./production/rl/history/grpo-lora-xpu-repro.md) |
| 2026-08-31 | [Production Training Runs -- Polaris (A100)](./production/polaris/README.md) |
| 2026-08-31 | [Production Training Metrics -- Ground-Truth Store](./production/metrics/README.md) |
| 2026-08-31 | [Production dispatch log](./production/dispatch-log.md) |
| 2026-08-31 | [Continued Pre-Training (CPT) — 2B olmo x dolmino mixing-ratio sweep](./production/cpt/README.md) |
| 2026-08-31 | [Production Training — Dense (agpt) Models](./production/agpt/README.md) |
| 2026-08-31 | [TPC26 MAPE Talk — Working Outline (2026-05-21)](./notes/slides-2026-05-21.md) |
| 2026-08-31 | [Data Strategy After 4.67T olmo-mix-1124 Tokens](./notes/data-strategy-after-olmo-mix-2026-07.md) |
| 2026-08-31 | [XPU Attention Issues](./guides/xpu-attention-issues.md) |
| 2026-08-31 | [Training agpt_80b on Aurora](./guides/training/agpt_80b.md) |
| 2026-08-31 | [SPMD backends on XPU: what works, what does not, and why](./guides/spmd-backend-status.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-09-17 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-08-31 |
| [2B 256N](./production/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6524. | 2026-08-30 |
| [2B 512N](./production/agpt/2b/n512/README.md) | step-**46429** (4.67T tokens, 100.0% of 4.67T), loss 2.68687. | 2026-08-30 |
| [20B 512N](./production/agpt/20b/n512/README.md) | step-**10,600** (10,600 x 12,288 x 8,192 = **1,067.0B tokens**), loss 2.41076. | 2026-08-30 |
| [20B 256N](./production/agpt/20b/n256/README.md) | step-**12,000** (604.0B tokens, 12.9% of 4.67T), loss 2.24577. | 2026-08-30 |
| [agpt 80B](./production/agpt/80b/README.md) | **Blocked at scale by a bf16 forward-activation overflow** (root-caused 2026-07-14, task #21): NOT an optimizer bug -- SophiaG (512N) and mano (62N) NaN with the *identical* flat-grad_norm signature, so it is optimizer-independent (the deep bf16 residual stream overflows at 80B's dim=9216 x 84L). fp32-residual prototype trains clean at 4N but STILL NaNs at dp=192 (necessary-but-insufficient); no live 80B production, fp32-residual work dormant. Wall 2 (256N init segfault) separate + open. | 2026-08-14 |
| [80B 512N NaN incident (2026-07-03)](./experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) | Incident record of the 512N NaN + NaN-abort guard. NOTE: the SophiaG-Hessian attribution was later disproven (2026-07-14, task #21) -- the NaN is an optimizer-independent bf16 residual-stream overflow; see the 80B README. | 2026-07-24 |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-06-24 |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-06-24 |
| [agpt 2B](./production/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-08-30 |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-05-03 |
| [2B CPT (olmo x dolmino)](./production/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-08-31 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-06-28 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-08-31 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-08-31 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-07-09 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-08-31 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](./guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](./experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-06-30 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-08-31 |
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-08-14 |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-07-24 |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-08-31 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-09-16 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-09-01 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-05-04 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-27.md) | 2-week retrospective | 2026-08-31 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-09-06 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Aurora quickstart: frameworks/2026.1.0](./guides/aurora-quickstart-frameworks-rc.md) | **Start here for new setups.** Validation-queue recipe on the RC module -- no venv tarball, no relocation step. The four required exports (libglog on `LD_LIBRARY_PATH`, both proxies, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`) each cost a failed job to find. Carries the validated 5-corner matrix from job `8789506`: compiled agpt TP=2 works here (the June `.venv` cannot), moe TP>1 needs `6e4e1996f`. | 2026-08-30 |
| [Aurora quickstart: shared torch 2.13 tarball](./guides/aurora-quickstart-tarball.md) | The pre-RC path -- debug-scaling queue, shared venv tarball, `relocate-venv.sh`. Correct until the RC is the default module on your nodes. | 2026-08-28 |
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-08-31 |
| [Reference Baselines](./baselines/README.md) | Training curves and benchmarks | 2026-04-29 |
| [Dense Model Configs](./configs/dense.md) | 2B / 20B / 50B / 80B | 2026-04-26 |
| [MoE Variants](./configs/moe.md) | 500M-10B | 2026-04-26 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](./scaling/README.md) | Top-level scaling landing page | 2026-06-06 |
| [agpt 2B scaling](./scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-08-31 |
| [agpt 20B scaling](./scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-08-31 |
| [agpt 80B scaling](./scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-04-26 |
| [MoE scaling](./scaling/moe.md) | Per-N TPS / MFU | 2026-06-13 |
| [Per-run Experiment Reports](./experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-07-11 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](./competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-04-28 |
| [RL (GRPO) Experiment](./production/rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-08-30 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe. Syncs 82 and 83 have both LANDED (`09b4ef235`, `609777a0e`). | 2026-09-06 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-06-12 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-05-01 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items. Audited 2026-08-31: the docs restructure and the 80B TP=2 OOM are done, the Aurora scaling study largely so; the 80B compile item and the production plan are superseded; MoE throughput is partly answered. The blendcorpus/Megatron aliasing item (the second `## 6.`) is the only fully open one. | 2026-08-31 |
