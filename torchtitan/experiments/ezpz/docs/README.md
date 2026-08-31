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
| 2026-08-31 | [Summaries](./summaries/README.md) |
| 2026-08-31 | [Two days ending 2026-08-31](./summaries/2026-08-31.md) |
| 2026-08-31 | [Data Strategy After 4.67T olmo-mix-1124 Tokens](./notes/data-strategy-after-olmo-mix-2026-07.md) |
| 2026-08-31 | [Development Journal](./journal.md) |
| 2026-08-31 | [muP for agpt: audit, design, and staged plan](./experiments/mup/README.md) |
| 2026-08-31 | [2026-08-16 -- umbrella 8756070: 9h13m, first real stage-2 dolmino steps, killed by an unexplained PBS -14](./experiments/agpt/aurora/20260816-umbrella-8756070.md) |
| 2026-08-31 | [AuroraGPT evaluation strategy: modern-suite review (2026-07)](./evals/eval-landscape-2026-07.md) |
| 2026-08-31 | [Evaluation Results — agpt 20B](./evals/agpt/20b/README.md) |
| 2026-08-31 | [Pre-Training AuroraGPT with TorchTitan + 🍋 ezpz](./README.md) |
| 2026-08-30 | [Week ending 2026-08-29](./summaries/2026-08-29.md) |
| 2026-08-30 | [SFT recipe: agpt-2b-v2-256n-step92859 x tulu_math_uc_mix](./production/sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md) |
| 2026-08-30 | [Evals: full-mix 8N SFT (gs138650 x tulu_math_uc_mix_full)](./production/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md) |
| 2026-08-30 | [SFT recipe: gs138650 x tulu_math_uc_mix (FULL big mix, ~54B tokens)](./production/sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md) |
| 2026-08-30 | [GRPO smoke — SFT-step729 vs baseline as RL starting point](./production/sft/agpt/2b-mds/tulu_math_uc_mix/evals/grpo-smoke.md) |
| 2026-08-30 | [lm-eval: SFT'd AuroraGPT-2B (tulu_math_uc_mix) vs pretrained baseline](./production/sft/agpt/2b-mds/tulu_math_uc_mix/evals/README.md) |
| 2026-08-30 | [SFT recipe: AuroraGPT-2B-sophiag-138650 × tulu_math_uc_mix](./production/sft/agpt/2b-mds/tulu_math_uc_mix/README.md) |
| 2026-08-30 | [B4 Cold-Start SFT Fix (finishing-stage + reweighted-mix) Implementation Plan](./production/sft/agpt/2b-mds/b4-finish-and-reweight/implementation-plan.md) |
| 2026-08-30 | [B4 cold-start SFT fix: recover + beat B2 after the B3 dilution regression](./production/sft/agpt/2b-mds/b4-finish-and-reweight/design.md) |
| 2026-08-30 | [B3 Cold-Start SFT (instruct + CoT rebuild) Implementation Plan](./production/sft/agpt/2b-mds/b3-instruct-cot-mix/implementation-plan.md) |
| 2026-08-30 | [B3 cold-start SFT design: combined instruct + CoT rebuild from gs138650](./production/sft/agpt/2b-mds/b3-instruct-cot-mix/design.md) |
| 2026-08-30 | [Production SFT](./production/sft/README.md) |
| 2026-08-30 | [GRPO on Intel XPU: TRL GRPOTrainer](./production/rl/trl.md) |
| 2026-08-30 | [Teaching agpt-2b Chain-of-Thought Reasoning (plan)](./production/rl/plans/cot.md) |
| 2026-08-30 | [GRPO+LoRA on XPU: Monarch + TorchStore + vLLM](./production/rl/monarch.md) |
| 2026-08-30 | [RL bring-up + 2026-07-01 multi-node investigation (historical narrative)](./production/rl/history/2026-06-13-bringup-and-2026-07-01-desync.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-08-30 | [GRPO recipe: AuroraGPT-2B-sft-tulu-mix x sum_digits arithmetic (8N)](./production/rl/grpo/aurora2b/sft_arithmetic/README.md) |
| 2026-08-30 | [Production GRPO](./production/rl/grpo/README.md) |
| 2026-08-30 | [RL (GRPO) on Intel XPU](./production/rl/README.md) |
| 2026-08-30 | [Production Training Runs -- Polaris (A100)](./production/polaris/README.md) |
| 2026-08-30 | [Production Training — MoE 10B_2B_sdpa EP=12](./production/moe/10b_2b_sdpa_ep/README.md) |
| 2026-08-30 | [Production Training Metrics -- Ground-Truth Store](./production/metrics/README.md) |
| 2026-08-30 | [Continued Pre-Training (CPT) — 2B olmo x dolmino mixing-ratio sweep](./production/cpt/README.md) |
| 2026-08-30 | [Production Training — Dense (agpt) Models](./production/agpt/README.md) |
| 2026-08-30 | [agpt 30B-exp -- a proposed next flagship](./production/agpt/30b-exp/README.md) |
| 2026-08-30 | [30B-exp experiment log](./production/agpt/30b-exp/EXPERIMENTS.md) |
| 2026-08-30 | [Production Training — agpt 2B @ 512 nodes](./production/agpt/2b/n512/README.md) |
| 2026-08-30 | [Production Training — agpt 2B @ 256 nodes](./production/agpt/2b/n256/README.md) |
| 2026-08-30 | [Production Training — agpt 2B](./production/agpt/2b/README.md) |
| 2026-08-30 | [Production Training — agpt 20B @ 512 nodes](./production/agpt/20b/n512/README.md) |
| 2026-08-30 | [Production Training — agpt 20B @ 256 nodes](./production/agpt/20b/n256/README.md) |
| 2026-08-30 | [Production Training — agpt 20B](./production/agpt/20b/README.md) |
| 2026-08-30 | [Production Training Runs — Aurora](./production/README.md) |
| 2026-08-30 | [ALCF ticket -- Polaris nodes with a stuck GPU](./ops/alcf-ticket-zombie-gpu-nodes-20260827.md) |
| 2026-08-30 | [ALCF ticket: job 8784460 has not scheduled in 101 hours](./ops/alcf-ticket-8784460-not-scheduling-20260830.md) |
| 2026-08-30 | [SophiaG: a RECURRENT grad-norm blow-up at 30B](./guides/known-bugs/sophiag-stochastic-divergence-30b.md) |
| 2026-08-30 | [MoE at TP>1: wo gets Shard(0) where row-parallel wants Partial(sum)](./guides/known-bugs/moe-tp2-wo-placement.md) |
| 2026-08-30 | [frameworks/2026.1.0 on Aurora: import torch dies on libglog.so.0](./guides/known-bugs/fw-rc-libglog-not-on-loader-path.md) |
| 2026-08-30 | [Dead CLI flags: 99 PBS scripts at the repo root would abort today](./guides/known-bugs/dead-cli-flags-in-repo-root-pbs.md) |
| 2026-08-30 | [TorchTitan on Aurora -- quickstart (validation queue + frameworks/2026.1.0)](./guides/aurora-quickstart-frameworks-rc.md) |
| 2026-08-30 | [Fixed-batch optimizer comparison: AdamW vs Mano vs SophiaG](./experiments/optimizer-comparison/README.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-08-30 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-08-30 |
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
| [2B CPT (olmo x dolmino)](./production/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-08-30 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-06-28 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-07-24 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-08-31 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-07-09 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-08-17 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](./guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](./experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-06-30 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-07-24 |
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-08-14 |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-07-24 |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-04-26 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-08-31 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-08-26 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-05-04 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-27.md) | 2-week retrospective | 2026-08-14 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-08-31 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Aurora quickstart: frameworks/2026.1.0](./guides/aurora-quickstart-frameworks-rc.md) | **Start here for new setups.** Validation-queue recipe on the RC module -- no venv tarball, no relocation step. The four required exports (libglog on `LD_LIBRARY_PATH`, both proxies, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`) each cost a failed job to find. Carries the validated 5-corner matrix from job `8789506`: compiled agpt TP=2 works here (the June `.venv` cannot), moe TP>1 needs `6e4e1996f`. | 2026-08-30 |
| [Aurora quickstart: shared torch 2.13 tarball](./guides/aurora-quickstart-tarball.md) | The pre-RC path -- debug-scaling queue, shared venv tarball, `relocate-venv.sh`. Correct until the RC is the default module on your nodes. | 2026-08-28 |
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
| [RL (GRPO) Experiment](./production/rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-08-30 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe. Syncs 82 and 83 have both LANDED (`09b4ef235`, `609777a0e`). | 2026-08-31 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-06-12 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-05-01 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items. Audited 2026-08-31: the docs restructure and the 80B TP=2 OOM are done, the Aurora scaling study largely so; the 80B compile item and the production plan are superseded; MoE throughput is partly answered. The blendcorpus/Megatron aliasing item (the second `## 6.`) is the only fully open one. | 2026-08-31 |
