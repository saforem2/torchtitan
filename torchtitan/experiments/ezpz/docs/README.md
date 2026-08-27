---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

> Living documentation for the `experiments/ezpz/` work. Sections
> ordered by importance (live → reference → outbound). Within each
> section, rows are sorted newest first by last-commit date.
>
> **Looking for something specific?** See [`TREE.md`](./reference/TREE.md)
> for a single-page annotated tree of every directory and file
> under `docs/`, with descriptions of what goes where.

## Recently Updated

The 25 most-recently-changed docs by git commit date (across all 136
docs, not just the curated tables below). Auto-generated -- do not edit
by hand; run `utils/refresh_docs_readme_table.py` (or `refresh_all.sh`).

<!-- BEGIN recently-updated (auto-generated) -->
| Modified | Doc |
|---------:|-----|
| 2026-08-27 | [XPU graphs cannot capture oneCCL collectives (2026-08-16)](./reference/known-bugs/xpu-graphs-block-oneccl-collectives.md) |
| 2026-08-27 | [SophiaG: a RECURRENT grad-norm blow-up at 30B](./reference/known-bugs/sophiag-stochastic-divergence-30b.md) |
| 2026-08-27 | [Polaris failover patterns were correct and UNREACHABLE](./reference/known-bugs/polaris-failover-detect-machine-fqdn.md) |
| 2026-08-27 | [Polaris failover was always blind (no bad-node patterns registered)](./reference/known-bugs/polaris-failover-blind-rotation.md) |
| 2026-08-27 | [MoE at TP>1: wo gets Shard(0) where row-parallel wants Partial(sum)](./reference/known-bugs/moe-tp2-wo-placement.md) |
| 2026-08-27 | [moe's MLA attention was never ported to the #4121 fold](./reference/known-bugs/moe-mla-not-ported-to-4121-fold.md) |
| 2026-08-27 | [BlendCorpus yielded [B, L] after #4121 moved the stack to flat [T]](./reference/known-bugs/blendcorpus-fold-batch-dim.md) |
| 2026-08-27 | [Installing PyTorch in a Fresh, Self-Contained .venv on Polaris](./reference/guides/polaris-fresh-venv.md) |
| 2026-08-27 | [Perlmutter as a debug/verification host](./reference/guides/perlmutter-debug-host.md) |
| 2026-08-27 | [Frameworks RC (oneAPI 2026.1.0) -- validation status](./reference/guides/frameworks-rc-validation.md) |
| 2026-08-27 | [AuroraGPT Sync — Meeting Notes](./records/meeting-notes/agpt-sync.md) |
| 2026-08-27 | [Journal -- 2026-08](./records/journal/2026-08.md) |
| 2026-08-27 | [Fixed-batch optimizer comparison: AdamW vs Mano vs SophiaG](./records/experiments/optimizer-comparison/README.md) |
| 2026-08-27 | [Production dispatch log](./live/dispatch-log.md) |
| 2026-08-27 | [Production Training Runs -- Polaris (A100)](./live/chains/polaris/README.md) |
| 2026-08-27 | [Pre-Training AuroraGPT with TorchTitan + 🍋 ezpz](./README.md) |
| 2026-08-27 | [Handoff -- Aurora, 2026-08-26](./HANDOFF-aurora-2026-08-26.md) |
| 2026-08-26 | [Handoff -- Polaris 20B chain, 2026-08-26](./HANDOFF-20b-polaris-2026-08-26.md) |
| 2026-08-24 | [yeet-env Tarball Broadcast Scaling](./reference/scaling/yeet_env/README.md) |
| 2026-08-24 | [Scaling Tests & Production Runs — Aurora (2026-04-18 to 2026-04-21)](./reference/scaling/performance.md) |
| 2026-08-24 | [MoE Model Scaling](./reference/scaling/moe.md) |
| 2026-08-24 | [ENOSPC on /lus/tegu while df reports 1.1P free](./reference/known-bugs/sunspot-enospc-full-ost.md) |
| 2026-08-24 | [Why spmd_types leaves parameters unconverted](./reference/known-bugs/spmd-types-plain-tensor.md) |
| 2026-08-24 | [RoPE flavor mismatch: a mid-flight convention switch, and the exports it broke](./reference/known-bugs/rope-flavor-mismatch.md) |
| 2026-08-24 | [BlendCorpusDataLoader aliases torchtitan parallelism axes onto Megatron knobs](./reference/known-bugs/blendcorpus-megatron-aliasing.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-08-24 | [Blendcorpus EOFError Race in _build_index_mappings](./reference/known-bugs/blendcorpus-eoferror-race.md) |
| 2026-08-24 | [Aurora: 2098-node job killed at 9h13m of 24h with Exit_status = -14](./reference/known-bugs/aurora-job-8756070-exit-14.md) |
| 2026-08-24 | [agpt on full_dtensor: vc_check/DeviceMesh, and a pin that was justified uncompiled](./reference/known-bugs/agpt-full-dtensor-vc-check.md) |
| 2026-08-24 | [Training agpt_80b on Aurora](./reference/guides/training/agpt_80b.md) |
| 2026-08-24 | [training.dtype = bfloat16 silently freezes RMSNorm weights](./reference/guides/training-dtype-bf16-norm-freeze.md) |
| 2026-08-24 | [SPMD backends on XPU: what works, what does not, and why](./reference/guides/spmd-backend-status.md) |
| 2026-08-24 | [Running with Newer PyTorch (>= 2.10)](./reference/guides/running-with-newer-pytorch.md) |
| 2026-08-24 | [Known Issues and Operational Notes](./reference/guides/known-issues.md) |
| 2026-08-24 | [Checkpointing on SIGTERM/SIGINT](./reference/guides/checkpoint-on-signal.md) |
| 2026-08-24 | [Bad-node failover for production training](./reference/guides/bad-node-failover.md) |
| 2026-08-24 | [Upstream Sync — Loss Baselines](./reference/baselines/README.md) |
| 2026-08-24 | [docs/ tree map](./reference/TREE.md) |
| 2026-08-24 | [Upstream sync -- 2026-08](./records/upstream-sync/2026-08.md) |
| 2026-08-24 | [Upstream sync -- 2026-06](./records/upstream-sync/2026-06.md) |
| 2026-08-24 | [Upstream sync -- 2026-05](./records/upstream-sync/2026-05.md) |
| 2026-08-24 | [INCITE Quarterly Report — Q2 2026 (Apr 1 – Jun 30)](./records/summaries/2026-Q2-incite.md) |
| 2026-08-24 | [Week ending 2026-08-21](./records/summaries/2026-08-21.md) |
| 2026-08-24 | [Week ending 2026-08-14](./records/summaries/2026-08-14.md) |
| 2026-08-24 | [2026-07-26 to 2026-08-10 -- ~15-Day Summary](./records/summaries/2026-08-10.md) |
| 2026-08-24 | [2026-07-10 to 2026-07-26 -- ~16-Day Summary](./records/summaries/2026-07-26.md) |
| 2026-08-24 | [2026-07-06 to 2026-07-10 -- ~4-Day Summary](./records/summaries/2026-07-10.md) |
| 2026-08-24 | [2026-06-26 to 2026-07-06 -- ~10-Day Summary](./records/summaries/2026-07-06.md) |
| 2026-08-24 | [2026-06-12 to 2026-06-26 -- Two-Week Summary](./records/summaries/2026-06-26.md) |
| 2026-08-24 | [2026-06-05 → 2026-06-12 — One-Week Summary](./records/summaries/2026-06-12.md) |
| 2026-08-24 | [One-week summary — 2026-05-22 → 2026-05-29](./records/summaries/2026-05-29.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](live/dashboard.md) | Top-level snapshot of every active trajectory | 2026-08-24 |
| [Dense (agpt) Production](live/chains/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-08-24 |
| [2B 256N](live/chains/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6524. | 2026-08-24 |
| [2B 512N](live/chains/agpt/2b/n512/README.md) | step-**46429** (4.67T tokens, 100.0% of 4.67T), loss 2.68687. | 2026-08-24 |
| [20B 512N](live/chains/agpt/20b/n512/README.md) | step-**8,700** (875.8B tokens, 18.7% of 4.67T), loss 2.4635. | 2026-08-24 |
| [20B 256N](live/chains/agpt/20b/n256/README.md) | step-**10,300** (518.4B tokens, 11.1% of 4.67T). | 2026-08-24 |
| [agpt 80B](live/chains/agpt/80b/README.md) | **Blocked at scale by a bf16 forward-activation overflow** (root-caused 2026-07-14, task #21): NOT an optimizer bug -- SophiaG (512N) and mano (62N) NaN with the *identical* flat-grad_norm signature, so it is optimizer-independent (the deep bf16 residual stream overflows at 80B's dim=9216 x 84L). fp32-residual prototype trains clean at 4N but STILL NaNs at dp=192 (necessary-but-insufficient); no live 80B production, fp32-residual work dormant. Wall 2 (256N init segfault) separate + open. | 2026-08-24 |
| [80B 512N NaN incident (2026-07-03)](records/experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) | Incident record of the 512N NaN + NaN-abort guard. NOTE: the SophiaG-Hessian attribution was later disproven (2026-07-14, task #21) -- the NaN is an optimizer-independent bf16 residual-stream overflow; see the 80B README. | 2026-08-24 |
| [20B 1024N](live/chains/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-08-24 |
| [2B 1024N](live/chains/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-08-24 |
| [agpt 2B](live/chains/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-08-24 |
| [agpt 2B-MDS](live/chains/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-08-24 |
| [2B CPT (olmo x dolmino)](live/chains/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-08-24 |
| [Polaris 20B 128N](live/chains/polaris/README.md) | **The only non-Aurora live chain.** step-**5,600** persisted (47.0B tokens), val loss 2.3621, 56/56 ckpts complete. Leg 7560197 running (resumes step-5600); leg 7560196 burned 3h03m to a failover-detection bug, now fixed. NOT wired into `trajectories.py` or `update_all_charts.sh`, so `refresh_all.sh` cannot see it -- figures must be regenerated by hand on Polaris. [Handoff notes](HANDOFF-20b-polaris-2026-08-26.md) | 2026-08-27 |
| [Production Scaling Report](reference/scaling/performance.md) | Apr 18-21 experiments (historical) | 2026-08-24 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](records/evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-08-24 |
| [agpt 20B evals](records/evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-08-24 |
| [agpt 2B-MDS evals](records/evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-08-24 |
| [Eval Index](records/evals/README.md) | Top-level eval landing page | 2026-08-24 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](reference/guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](records/experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-08-24 |
| [Known Issues / Operational Notes](reference/guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-08-24 |
| [bf16-master RMSNorm freeze](reference/guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-08-24 |
| [TP > 1 loss reporting off by `dp_world_size`](reference/guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-08-23 |
| [XPU Attention Issues](reference/guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-08-23 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-08-24 |
| [AuroraGPT Sync Notes](records/meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-08-27 |
| [Meeting Notes Index](records/meeting-notes/README.md) | Top-level meeting index | 2026-08-23 |
| [Summary 2026-04-12 → 2026-04-27](records/summaries/2026-04-27.md) | 2-week retrospective | 2026-08-24 |
| [Periodic Summaries Index](records/summaries/README.md) | Index of 2-week / monthly retros | 2026-08-23 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Running with Newer PyTorch (≥ 2.10)](reference/guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-08-24 |
| [Reference Baselines](reference/baselines/README.md) | Training curves and benchmarks | 2026-08-24 |
| [Dense Model Configs](reference/configs/dense.md) | 2B / 20B / 50B / 80B | 2026-08-23 |
| [MoE Variants](reference/configs/moe.md) | 500M-10B | 2026-08-23 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](reference/scaling/README.md) | Top-level scaling landing page | 2026-08-23 |
| [agpt 2B scaling](reference/scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-08-23 |
| [agpt 20B scaling](reference/scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-08-23 |
| [agpt 80B scaling](reference/scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-08-23 |
| [MoE scaling](reference/scaling/moe.md) | Per-N TPS / MFU | 2026-08-24 |
| [Per-run Experiment Reports](records/experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-08-23 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](records/competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-08-23 |
| [RL (GRPO) Experiment](live/chains/rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-08-24 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe | 2026-08-23 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](outbound/upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-08-23 |
| [`StateDictStager` bug](outbound/upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-08-23 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](live/TODO.md) | Open work items | 2026-08-24 |
