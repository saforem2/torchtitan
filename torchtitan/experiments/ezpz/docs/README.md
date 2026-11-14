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

The 25 most-recently-changed docs by git commit date (across all 330
docs, not just the curated tables below). Auto-generated -- do not edit
by hand; run `utils/refresh_docs_readme_table.py` (or `refresh_all.sh`).

<!-- BEGIN recently-updated (auto-generated) -->
| Modified | Doc |
|---------:|-----|
| 2026-11-14 | [Upstream Sync Log](./upstream-sync.md) |
| 2026-11-14 | [Continued Pre-Training (CPT) — 2B olmo x dolmino mixing-ratio sweep](./production/cpt/README.md) |
| 2026-11-14 | [Production Training Runs — Aurora](./production/README.md) |
| 2026-11-14 | [AuroraGPT-2B post-training: complete status](./production/POST-TRAINING-2B.md) |
| 2026-11-14 | [Development Journal](./journal.md) |
| 2026-11-12 | [FineWeb-Edu 60/40 kind-shift: the next pre-training lever for agpt-2B](./production/sft/fineweb-edu-60-40-kind-shift.md) |
| 2026-11-12 | [Production SFT](./production/sft/README.md) |
| 2026-11-12 | [Production RL with Monarch, TorchStore, and vLLM on XPU](./production/rl/monarch.md) |
| 2026-11-12 | [Aurora 2088N umbrella 9053470 and per-seat resubmission — 2026-11-09 to 2026-11-11](./production/2026-11-11-umbrella-2088n.md) |
| 2026-11-12 | [XPU: intermittent FSDP all-gather hang after step 7 (2026-11-11)](./guides/known-bugs/xpu-fsdp-allgather-hang-after-step-7.md) |
| 2026-11-12 | [MoE at TP>1: wo gets Shard(0) where row-parallel wants Partial(sum)](./guides/known-bugs/moe-tp2-wo-placement.md) |
| 2026-11-12 | [CUDA: default CUDA graph capture breaks ezpz MoE and native DDP (2026-11-12)](./guides/known-bugs/cuda-graphs-moe-native-ddp.md) |
| 2026-11-12 | [AuroraGPT-2B: XPU throughput recovery on Sunspot](./experiments/agpt/sunspot/2026-11-02-throughput-recovery.md) |
| 2026-11-12 | [MMLU letter-prior gating note: constant-D beats every agpt-2B model](./evals/mmlu-letter-prior-gating.md) |
| 2026-11-12 | [AuroraGPT MMLU sits at chance because the models answer with a letter prior](./evals/mmlu-letter-prior-at-chance.md) |
| 2026-11-12 | [Evaluation Results](./evals/README.md) |
| 2026-11-05 | [SFT recipe: agpt-2b-v2-256n-step92859 x tulu_math_uc_mix](./production/sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md) |
| 2026-11-02 | [AuroraGPT-2B: recovered Sunspot MDS/TT comparison](./experiments/agpt/sunspot/2026-11-02-recovery-comparison.md) |
| 2026-11-02 | [agpt (Dense AuroraGPT) Benchmarks](./experiments/agpt/README.md) |
| 2026-11-01 | [Sunspot startup isolation: MDS/TT continuation](./experiments/agpt/sunspot/2026-11-01-startup-incident.md) |
| 2026-11-01 | [AuroraGPT-2B: same-allocation MDS and TorchTitan comparison](./experiments/agpt/sunspot/2026-11-01-paired-performance.md) |
| 2026-11-01 | [AuroraGPT-2B MDS/TT: recovered comparison and handoff audit](./experiments/agpt/sunspot/2026-11-01-handoff-audit.md) |
| 2026-11-01 | [AuroraGPT-2B: replicated TT versus MDS ZeRO-0](./experiments/agpt/sunspot/2026-10-31-replicated-comparison.md) |
| 2026-10-31 | [Production Training Runs -- Polaris (A100)](./production/polaris/README.md) |
| 2026-10-31 | [Sunspot AuroraGPT-2B: Megatron-DeepSpeed versus TorchTitan](./experiments/agpt/sunspot/2026-10-29-native-checkpoint-closure.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-10-31 | [Why the MDS ZeRO1 run outperformed TorchTitan HSDP](./experiments/agpt/sunspot/2026-10-24-zero1-hsdp-diagnosis.md) |
| 2026-10-31 | [Sunspot FLAT runs: timing complete, checkpoint weights differ](./experiments/agpt/sunspot/2026-10-24-matched-flat-execution.md) |
| 2026-10-31 | [AuroraGPT-2B: TorchTitan and Megatron-DeepSpeed on Sunspot](./experiments/agpt/sunspot/2026-10-23-compile-sharding.md) |
| 2026-10-29 | [AuroraGPT-80B Scaling & Benchmarks](./scaling/agpt-80b.md) |
| 2026-10-29 | [RL (GRPO) on Intel XPU](./production/rl/README.md) |
| 2026-10-29 | [Production Training — agpt 80B](./production/agpt/80b/README.md) |
| 2026-10-29 | [Aurora 20B production restart — 2026-10-29](./production/2026-10-29-20b-restart.md) |
| 2026-10-29 | [Training agpt_80b on Aurora](./guides/training/agpt_80b.md) |
| 2026-10-29 | [Dead CLI flags in archived historical PBS scripts](./guides/known-bugs/dead-cli-flags-in-repo-root-pbs.md) |
| 2026-10-29 | [AGPT MoE Hugging Face export](./guides/agpt-moe-hf-eval.md) |
| 2026-10-29 | [MoE HF export review fixes and pipeline validation](./experiments/moe/aurora/2026-10-06-hf-export-review-fixes.md) |
| 2026-10-29 | [HF export PR feedback fixes](./experiments/moe/aurora/2026-10-06-hf-export-pr-feedback.md) |
| 2026-10-29 | [Matched dense 2B and MoE 12B2A HF evaluation](./experiments/moe/aurora/2026-10-06-hf-dense-moe-paired.md) |
| 2026-10-29 | [MoE (Mixture of Experts) Benchmarks](./experiments/moe/README.md) |
| 2026-10-29 | [Mano LR finder at 30B (Sunspot, 2026-08-23)](./experiments/lr-finder/agpt/2026-08-23-30b-mano-sunspot.md) |
| 2026-10-29 | [Pre-Training AuroraGPT with TorchTitan + 🍋 ezpz](./README.md) |
| 2026-10-19 | [Follow-up umbrella: Monday boost (>2048N) — 2026-10-20](./production/followup-umbrella-boost-20261020.md) |
| 2026-10-16 | [Ten days ending 2026-09-26](./summaries/2026-09-26.md) |
| 2026-10-16 | [Production Training — agpt 2B @ 256 nodes](./production/agpt/2b/n256/README.md) |
| 2026-10-16 | [Production Training — agpt 20B @ 512 nodes](./production/agpt/20b/n512/README.md) |
| 2026-10-16 | [Production Training — agpt 20B @ 256 nodes](./production/agpt/20b/n256/README.md) |
| 2026-10-16 | [Evaluation Results — agpt 20B](./evals/agpt/20b/README.md) |
| 2026-10-15 | [BlendCorpus restarts replay the data stream](./guides/blendcorpus-resume-position.md) |
| 2026-10-01 | [Sunspot multi-host Monarch, TorchStore, and vLLM validation](./experiments/2026-09-25-sunspot-multihost-rl-validation.md) |
| 2026-09-30 | [Summaries](./summaries/README.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-11-14 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-08-31 |
| [2B 256N](./production/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6524. | 2026-10-16 |
| [2B 512N](./production/agpt/2b/n512/README.md) | step-**46429** (4.67T tokens, 100.0% of 4.67T), loss 2.6869. | 2026-09-30 |
| [20B 512N](./production/agpt/20b/n512/README.md) | step-**11,100** (11,100 x 12,288 x 8,192 = **1,117.4B tokens**), loss 2.31773. | 2026-10-16 |
| [20B 256N](./production/agpt/20b/n256/README.md) | step-**17,500** (880.8B tokens, 18.8% of 4.67T), loss 2.41783. | 2026-10-16 |
| [agpt 80B](./production/agpt/80b/README.md) | **Blocked at scale by a high-dp NaN; root cause unknown.** Optimizer-independent (SophiaG 512N, mano 62N); the earlier "bf16 residual-stream overflow" diagnosis is refuted ([what we know](./guides/known-bugs/80b-nan-what-we-know.md)). fp32 activations prevent it at ~3.4x cost; fp32-residual alone still NaNs at dp=192; no live 80B production, fp32-residual work dormant. Wall 2 (256N init segfault) separate + open. | 2026-10-29 |
| [80B 512N NaN incident (2026-07-03)](./experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) | Incident record of the 512N NaN + NaN-abort guard. NOTE: the SophiaG-Hessian attribution was later disproven (2026-07-14, task #21) -- the NaN is optimizer-independent; the follow-on residual-overflow diagnosis is also refuted, root cause unknown (see `guides/known-bugs/80b-nan-what-we-know.md`). | 2026-07-24 |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-06-24 |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-06-24 |
| [agpt 2B](./production/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-09-30 |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-05-03 |
| [2B CPT (olmo x dolmino)](./production/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-11-14 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-06-28 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-08-31 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-10-16 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-07-09 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-11-12 |

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
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-11-14 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-09-01 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-05-04 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-27.md) | 2-week retrospective | 2026-08-31 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-09-30 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Aurora quickstart: frameworks/2026.1.0](./guides/aurora-quickstart-frameworks-rc.md) | **Start here for new setups.** Validation-queue recipe on the RC module -- no venv tarball, no relocation step. The four required exports (libglog on `LD_LIBRARY_PATH`, both proxies, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`) each cost a failed job to find. Carries the validated 5-corner matrix from job `8789506`: compiled agpt TP=2 works here (the June `.venv` cannot), moe TP>1 needs `6e4e1996f`. | 2026-08-30 |
| [Aurora quickstart: shared torch 2.13 tarball](./guides/aurora-quickstart-tarball.md) | The pre-RC path -- debug-scaling queue, shared venv tarball, `relocate-venv.sh`. Correct until the RC is the default module on your nodes. | 2026-08-28 |
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-09-30 |
| [Reference Baselines](./baselines/README.md) | Training curves and benchmarks | 2026-04-29 |
| [Dense Model Configs](./configs/dense.md) | 2B / 20B / 50B / 80B | 2026-04-26 |
| [MoE Variants](./configs/moe.md) | 500M-10B | 2026-04-26 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](./scaling/README.md) | Top-level scaling landing page | 2026-06-06 |
| [agpt 2B scaling](./scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-08-31 |
| [agpt 20B scaling](./scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-08-31 |
| [agpt 80B scaling](./scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-10-29 |
| [MoE scaling](./scaling/moe.md) | Per-N TPS / MFU | 2026-06-13 |
| [Per-run Experiment Reports](./experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-09-29 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](./competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-04-28 |
| [RL (GRPO) Experiment](./production/rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-10-29 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe. Syncs 82 and 83 have both LANDED (`09b4ef235`, `609777a0e`). | 2026-11-14 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-06-12 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-05-01 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items. Audited 2026-08-31: the docs restructure and the 80B TP=2 OOM are done, the Aurora scaling study largely so; the 80B compile item and the production plan are superseded; MoE throughput is partly answered. The blendcorpus/Megatron aliasing item (the second `## 6.`) is the only fully open one. | 2026-08-31 |
