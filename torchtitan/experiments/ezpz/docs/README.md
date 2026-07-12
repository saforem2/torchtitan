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

The 25 most-recently-changed docs by git commit date (across all 136
docs, not just the curated tables below). Auto-generated -- do not edit
by hand; run `utils/refresh_docs_readme_table.py` (or `refresh_all.sh`).

<!-- BEGIN recently-updated (auto-generated) -->
| Modified | Doc |
|---------:|-----|
| 2026-07-12 | [Upstream Sync Log](./upstream-sync.md) |
| 2026-07-12 | [SFT recipe: gs138650 x tulu_math_uc_mix (FULL big mix, ~54B tokens)](./production/sft/gs138650/tulu_math_uc_mix_full/README.md) |
| 2026-07-12 | [Production SFT](./production/sft/README.md) |
| 2026-07-12 | [Production Training — agpt 2B @ 512 nodes](./production/agpt/2b/n512/README.md) |
| 2026-07-12 | [Production Training — agpt 2B @ 256 nodes](./production/agpt/2b/n256/README.md) |
| 2026-07-12 | [Production Training — agpt 2B](./production/agpt/2b/README.md) |
| 2026-07-12 | [Production Training — agpt 20B @ 512 nodes](./production/agpt/20b/n512/README.md) |
| 2026-07-12 | [Production Training — agpt 20B @ 256 nodes](./production/agpt/20b/n256/README.md) |
| 2026-07-12 | [Production Training — agpt 20B](./production/agpt/20b/README.md) |
| 2026-07-12 | [Production Training Runs — Aurora](./production/README.md) |
| 2026-07-12 | [Development Journal](./journal.md) |
| 2026-07-12 | [Pre-Training AuroraGPT with TorchTitan + 🍋 ezpz](./README.md) |
| 2026-07-11 | [Production Training Runs -- Polaris (A100)](./production/polaris/README.md) |
| 2026-07-11 | [Synthetic-summary data generation POC (olmo-mix-1124)](./experiments/synthetic/aurora/2026-07-11-summarize-olmo-mix-poc.md) |
| 2026-07-11 | [SFT on gs138650 with the FULL tulu_math_uc_mix (big OpenMathInstruct-2)](./experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md) |
| 2026-07-11 | [Experiment Benchmark Reports](./experiments/README.md) |
| 2026-07-10 | [Summaries](./summaries/README.md) |
| 2026-07-10 | [2026-07-06 to 2026-07-10 -- ~4-Day Summary](./summaries/2026-07-06_to_2026-07-10.md) |
| 2026-07-10 | [Continued Pre-Training (CPT) — 2B olmo x dolmino mixing-ratio sweep](./production/cpt/README.md) |
| 2026-07-10 | [Evaluation Results — agpt 20B](./evals/agpt/20b/README.md) |
| 2026-07-09 | [SFT recipe: AuroraGPT-2B-sophiag-138650 × tulu_math_uc_mix](./production/sft/aurora2b/tulu_math_uc_mix/README.md) |
| 2026-07-09 | [Evaluation Results — agpt 2B (Megatron-DeepSpeed SophiaG)](./evals/agpt/2b-mds/README.md) |
| 2026-07-06 | [2026-06-26 to 2026-07-06 -- ~10-Day Summary](./summaries/2026-06-26_to_2026-07-06.md) |
| 2026-07-06 | [2026-06-12 to 2026-06-26 -- Two-Week Summary](./summaries/2026-06-12_to_2026-06-26.md) |
| 2026-07-06 | [Wiring vLLM-XPU into ezpz/rl — architecture + sequencing plan](./rl/history/vllm-xpu-wiring-plan.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-07-06 | [vLLM-XPU on torch 2.13 — investigation findings](./rl/history/vllm-xpu-investigation.md) |
| 2026-07-06 | [vLLM-XPU + Monarch RL infra status (as of 2026-06-13 PM)](./rl/history/vllm-xpu-current-status.md) |
| 2026-07-06 | [Upstream torchtitan.experiments.rl.train port status (2026-06-13)](./rl/history/upstream-rl-port-status.md) |
| 2026-07-06 | [RL bring-up history](./rl/history/README.md) |
| 2026-07-06 | [Monarch + torch 2.13 deep-dive (2026-06-14)](./rl/history/2026-06-14_monarch-torch213-deep-dive.md) |
| 2026-07-06 | [RL bring-up + 2026-07-01 multi-node investigation (historical narrative)](./rl/history/2026-06-13-bringup-and-2026-07-01-desync.md) |
| 2026-07-06 | [GRPO on Intel XPU — status](./rl/grpo-on-xpu-status.md) |
| 2026-07-06 | [RL (GRPO) Experiment](./rl/README.md) |
| 2026-07-06 | [Multi-trainer-node GRPO on XPU: root cause (2026-07-06)](./rl/2026-07-06_multinode-grpo-root-cause.md) |
| 2026-07-06 | [SFT recipe: agpt-2b-v2-256n-step92859 x tulu_math_uc_mix](./production/sft/agpt-2b-v2-256n/tulu_math_uc_mix/README.md) |
| 2026-07-06 | [Production GRPO](./production/grpo/README.md) |
| 2026-07-06 | [Production Training — agpt 80B](./production/agpt/80b/README.md) |
| 2026-07-06 | [AuroraGPT Sync — Meeting Notes](./meeting-notes/agpt-sync.md) |
| 2026-07-06 | [SFT on the completed v2 2B base (step-92,859): conversion + transfer + smoke](./experiments/agpt/sunspot/2026-07-06-sft-2b-v2-256n-base-prep.md) |
| 2026-07-06 | [80B 512N production run NaN'd: SophiaG unstable at production batch](./experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) |
| 2026-07-06 | [Multi-chain umbrella on native auto-retry: diagnosis + smoke](./experiments/agpt/aurora/20260702-multi-autoretry-umbrella-smoke.md) |
| 2026-07-01 | [Production Training — Dense (agpt) Models](./production/agpt/README.md) |
| 2026-07-01 | [Fresh, Self-Contained .venv on Polaris (no conda inheritance)](./guides/polaris-fresh-venv.md) |
| 2026-07-01 | [LR Finder -- agpt 80B](./experiments/lr-finder/agpt/80b/README.md) |
| 2026-07-01 | [80B head-to-head convergence at the production batch (GBS=6144) -- Sunspot, 2026-06-30](./experiments/agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md) |
| 2026-07-01 | [2B continued-pretraining (CPT): olmo x dolmino mixing-ratio sweep](./experiments/agpt/aurora/20260701-2b-cpt-olmo-dolmino-sweep.md) |
| 2026-07-01 | [20B 512N canonical chain: relaunch on native auto-retry (resume step-4400)](./experiments/agpt/aurora/20260701-20b-512n-relaunch-autoretry.md) |
| 2026-07-01 | [Failover / auto-retry "restart economics" -- log-mining analysis](./experiments/agpt/aurora/20260630-failover-restart-economics.md) |
| 2026-07-01 | [80B v2 production launch: SophiaG, constant-LR, scale brackets 512N/1024N/2048N](./experiments/agpt/aurora/20260628-80b-sophiag-constant-lr-512-1024-2048.md) |
| 2026-07-01 | [Evaluation Results — agpt 2B](./evals/agpt/2b/README.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-07-12 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-07-01 |
| [2B 256N](./production/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6511. | 2026-07-12 |
| [2B 512N](./production/agpt/2b/n512/README.md) | step-**35000** (3.52T tokens, 75.4% of 4.67T). | 2026-07-12 |
| [20B 512N](./production/agpt/20b/n512/README.md) | step-**6,000** (604.0B tokens, 12.9% of 4.67T). | 2026-07-12 |
| [20B 256N](./production/agpt/20b/n256/README.md) | step-**4,200** (211.4B tokens, 4.5% of 4.67T), loss 3.2631. | 2026-07-12 |
| [agpt 80B](./production/agpt/80b/README.md) | **SophiaG @ 1e-6 NaN'd** the 512N prod run 2026-07-03 (grad_norm->inf step-14, Hessian overflow; ~12h wasted). Testing **mano @ 1e-6** as the replacement: mechanism probe PASSED (30 steps clean), 256N production-batch verdict (8647521) pending. | 2026-07-06 |
| [80B optimizer NaN report](./experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) | Full SophiaG-NaN diagnosis + why longer-warmup/grad-clip don't fix it + NaN-abort guard. | 2026-07-06 |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-06-24 |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-06-24 |
| [agpt 2B](./production/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-07-12 |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-05-03 |
| [2B CPT (olmo x dolmino)](./production/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-07-10 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-06-28 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-07-01 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-07-10 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-07-09 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-06-26 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](./guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](./experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-06-30 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-05-23 |
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-06-10 |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-06-12 |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-04-26 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-07-12 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-07-06 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-05-04 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-12_to_2026-04-27.md) | 2-week retrospective | 2026-06-28 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-07-10 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-06-09 |
| [Reference Baselines](./baselines/README.md) | Training curves and benchmarks | 2026-04-29 |
| [Dense Model Configs](./configs/dense.md) | 2B / 20B / 50B / 80B | 2026-04-26 |
| [MoE Variants](./configs/moe.md) | 500M-10B | 2026-04-26 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](./scaling/README.md) | Top-level scaling landing page | 2026-06-06 |
| [agpt 2B scaling](./scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-06-13 |
| [agpt 20B scaling](./scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-06-13 |
| [agpt 80B scaling](./scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-04-26 |
| [MoE scaling](./scaling/moe.md) | Per-N TPS / MFU | 2026-06-13 |
| [Per-run Experiment Reports](./experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-07-11 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](./competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-04-28 |
| [RL (GRPO) Experiment](./rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-07-06 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe | 2026-07-12 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-06-12 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-05-01 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items | 2026-05-05 |
