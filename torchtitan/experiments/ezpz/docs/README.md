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
| 2026-08-23 | [Upstream Sync Log](./upstream-sync.md) |
| 2026-08-23 | [yeet-env Tarball Broadcast Scaling](./reference/scaling/yeet_env/README.md) |
| 2026-08-23 | [MoE Model Scaling](./reference/scaling/moe.md) |
| 2026-08-23 | [AuroraGPT-80B Scaling & Benchmarks](./reference/scaling/agpt-80b.md) |
| 2026-08-23 | [AuroraGPT-2B Scaling](./reference/scaling/agpt-2b.md) |
| 2026-08-23 | [AuroraGPT-20B Scaling](./reference/scaling/agpt-20b.md) |
| 2026-08-23 | [Scaling Results](./reference/scaling/README.md) |
| 2026-08-23 | [XPU graphs cannot capture oneCCL collectives (2026-08-16)](./reference/known-bugs/xpu-graphs-block-oneccl-collectives.md) |
| 2026-08-23 | [--debug.deterministic is not bit-reproducible on XPU (2026-08-16)](./reference/known-bugs/xpu-determinism-rank-seqlen-interaction.md) |
| 2026-08-23 | [The "validator CCL deadlock at 80B TP=4" was a phantom -- two unrelated bugs](./reference/known-bugs/validator-tp4-at-80b.md) |
| 2026-08-23 | [Unregistered W&B runs: the failure that never announces itself](./reference/known-bugs/unregistered-wandb-runs.md) |
| 2026-08-23 | [Umbrella std::bad_alloc at init -- intermittent, not yet root-caused](./reference/known-bugs/umbrella-bad-alloc-init.md) |
| 2026-08-23 | [sunspot-x1921c4s3b0n0-bad-mount](./reference/known-bugs/sunspot-x1921c4s3b0n0-bad-mount.md) |
| 2026-08-23 | [Sunspot: bare reduce_scatter_tensor SIGSEGVs at 48 ranks (2026-08-14)](./reference/known-bugs/sunspot-reduce-scatter-segv-20260814.md) |
| 2026-08-23 | [RETRACTED -- this was NOT a cluster fault](./reference/known-bugs/sunspot-ccl-allgatherv-outage-20260810.md) |
| 2026-08-23 | [Why spmd_types leaves parameters unconverted](./reference/known-bugs/spmd-types-plain-tensor.md) |
| 2026-08-23 | [Trying a newer XPU torch against spmd_types (2026-08-20)](./reference/known-bugs/spmd-types-newer-torch-attempt.md) |
| 2026-08-23 | [RoPE flavor mismatch: a mid-flight convention switch, and the exports it broke](./reference/known-bugs/rope-flavor-mismatch.md) |
| 2026-08-23 | [Pre-#3623 checkpoints can't resume on current code: optimizer state-dict format migration](./reference/known-bugs/pre3623-optim-statedict-resume.md) |
| 2026-08-23 | [Polaris 20B "eval gibberish" root cause: training-data / model tokenizer mismatch](./reference/known-bugs/polaris-20b-tokenizer-mismatch.md) |
| 2026-08-23 | [Flex attention on MoE: two stacked bugs, both fixed](./reference/known-bugs/moe-flex-attention-blockmask.md) |
| 2026-08-23 | [MoE under EP aborts in all_to_all_single, worse with model size (2026-08-19)](./reference/known-bugs/moe-ep-a2a-degrades-with-size.md) |
| 2026-08-23 | [--debug.deterministic costs ~27% device memory on the MoE path (2026-08-19)](./reference/known-bugs/moe-deterministic-memory.md) |
| 2026-08-23 | [hybridep on XPU: not a version floor, not portable](./reference/known-bugs/hybridep-is-nvidia-only.md) |
| 2026-08-23 | [frameworks-RC torch: torch.compile at TP=4 crashes in the SDPA flash-backward](./reference/known-bugs/fw-rc-compile-sdpa-backward-tp4.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-08-23 | ['Config' object has no attribute 'job' (ezpz branch, 2026-08-13)](./reference/known-bugs/config-job-dump-folder-attributeerror.md) |
| 2026-08-23 | [Concurrent-job checkpoint collision on 20b_v2_256](./reference/known-bugs/concurrent-job-ckpt-collision.md) |
| 2026-08-23 | [BlendCorpusDataLoader aliases torchtitan parallelism axes onto Megatron knobs](./reference/known-bugs/blendcorpus-megatron-aliasing.md) |
| 2026-08-23 | [Blendcorpus EOFError Race in _build_index_mappings](./reference/known-bugs/blendcorpus-eoferror-race.md) |
| 2026-08-23 | [Aurora: 2098-node job killed at 9h13m of 24h with Exit_status = -14](./reference/known-bugs/aurora-job-8756070-exit-14.md) |
| 2026-08-23 | [agpt on full_dtensor: vc_check/DeviceMesh, and a pin that was justified uncompiled](./reference/known-bugs/agpt-full-dtensor-vc-check.md) |
| 2026-08-23 | [Known bugs](./reference/known-bugs/README.md) |
| 2026-08-23 | [XPU Attention Issues](./reference/guides/xpu-attention-issues.md) |
| 2026-08-23 | [Training agpt_80b on Aurora](./reference/guides/training/agpt_80b.md) |
| 2026-08-23 | [training.dtype = bfloat16 silently freezes RMSNorm weights](./reference/guides/training-dtype-bf16-norm-freeze.md) |
| 2026-08-23 | [SPMD backends on XPU: what works, what does not, and why](./reference/guides/spmd-backend-status.md) |
| 2026-08-23 | [Running with Newer PyTorch (>= 2.10)](./reference/guides/running-with-newer-pytorch.md) |
| 2026-08-23 | [Installing PyTorch in a Fresh, Self-Contained .venv on Polaris](./reference/guides/polaris-fresh-venv.md) |
| 2026-08-23 | [Loss reporting was off by dp_world_size on TP > 1 — fixed upstream 2026-05-18](./reference/guides/loss-reporting-tp-dist-reduce.md) |
| 2026-08-23 | [Known Issues and Operational Notes](./reference/guides/known-issues.md) |
| 2026-08-23 | [Local HF Dataset Cache for Distributed Training](./reference/guides/hf-dataset-offline-cache.md) |
| 2026-08-23 | [Frameworks RC (oneAPI 2026.1.0) -- validation status](./reference/guides/frameworks-rc-validation.md) |
| 2026-08-23 | [Bad-node failover for production training](./reference/guides/bad-node-failover.md) |
| 2026-08-23 | [MoE Training Configs](./reference/configs/moe.md) |
| 2026-08-23 | [Dense Models: AuroraGPT-{2,20}B](./reference/configs/dense.md) |
| 2026-08-23 | [Upstream Sync — Loss Baselines](./reference/baselines/README.md) |
| 2026-08-23 | [docs/ tree map](./reference/TREE.md) |
| 2026-08-23 | [Upstream sync -- 2026-08](./records/upstream-sync/2026-08.md) |
| 2026-08-23 | [Upstream sync -- 2026-07](./records/upstream-sync/2026-07.md) |
| 2026-08-23 | [Upstream sync -- 2026-06](./records/upstream-sync/2026-06.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](live/dashboard.md) | Top-level snapshot of every active trajectory | 2026-08-23 |
| [Dense (agpt) Production](live/chains/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-08-23 |
| [2B 256N](live/chains/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6524. | 2026-08-23 |
| [2B 512N](live/chains/agpt/2b/n512/README.md) | step-**46429** (4.67T tokens, 100.0% of 4.67T), loss 2.68687. | 2026-08-23 |
| [20B 512N](live/chains/agpt/20b/n512/README.md) | step-**8,700** (875.8B tokens, 18.7% of 4.67T), loss 2.4635. | 2026-08-23 |
| [20B 256N](live/chains/agpt/20b/n256/README.md) | step-**10,300** (518.4B tokens, 11.1% of 4.67T). | 2026-08-23 |
| [agpt 80B](live/chains/agpt/80b/README.md) | **Blocked at scale by a bf16 forward-activation overflow** (root-caused 2026-07-14, task #21): NOT an optimizer bug -- SophiaG (512N) and mano (62N) NaN with the *identical* flat-grad_norm signature, so it is optimizer-independent (the deep bf16 residual stream overflows at 80B's dim=9216 x 84L). fp32-residual prototype trains clean at 4N but STILL NaNs at dp=192 (necessary-but-insufficient); no live 80B production, fp32-residual work dormant. Wall 2 (256N init segfault) separate + open. | 2026-08-23 |
| [80B 512N NaN incident (2026-07-03)](records/experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) | Incident record of the 512N NaN + NaN-abort guard. NOTE: the SophiaG-Hessian attribution was later disproven (2026-07-14, task #21) -- the NaN is an optimizer-independent bf16 residual-stream overflow; see the 80B README. | 2026-08-23 |
| [20B 1024N](live/chains/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-06-24 |
| [2B 1024N](live/chains/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-06-24 |
| [agpt 2B](live/chains/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-08-23 |
| [agpt 2B-MDS](live/chains/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-08-23 |
| [2B CPT (olmo x dolmino)](live/chains/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-08-23 |
| [Production Scaling Report](reference/scaling/performance.md) | Apr 18-21 experiments (historical) | 2026-08-23 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](records/evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-08-23 |
| [agpt 20B evals](records/evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-08-23 |
| [agpt 2B-MDS evals](records/evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-08-23 |
| [Eval Index](records/evals/README.md) | Top-level eval landing page | 2026-08-23 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](reference/guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](records/experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-08-23 |
| [Known Issues / Operational Notes](reference/guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-08-23 |
| [bf16-master RMSNorm freeze](reference/guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-08-23 |
| [TP > 1 loss reporting off by `dp_world_size`](reference/guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-08-23 |
| [XPU Attention Issues](reference/guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-08-23 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-08-23 |
| [AuroraGPT Sync Notes](records/meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-08-23 |
| [Meeting Notes Index](records/meeting-notes/README.md) | Top-level meeting index | 2026-08-23 |
| [Summary 2026-04-12 → 2026-04-27](records/summaries/2026-04-27.md) | 2-week retrospective | 2026-08-23 |
| [Periodic Summaries Index](records/summaries/README.md) | Index of 2-week / monthly retros | 2026-08-23 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Running with Newer PyTorch (≥ 2.10)](reference/guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-08-23 |
| [Reference Baselines](reference/baselines/README.md) | Training curves and benchmarks | 2026-08-23 |
| [Dense Model Configs](reference/configs/dense.md) | 2B / 20B / 50B / 80B | 2026-08-23 |
| [MoE Variants](reference/configs/moe.md) | 500M-10B | 2026-08-23 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](reference/scaling/README.md) | Top-level scaling landing page | 2026-08-23 |
| [agpt 2B scaling](reference/scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-08-23 |
| [agpt 20B scaling](reference/scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-08-23 |
| [agpt 80B scaling](reference/scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-08-23 |
| [MoE scaling](reference/scaling/moe.md) | Per-N TPS / MFU | 2026-08-23 |
| [Per-run Experiment Reports](records/experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-08-23 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](records/competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-08-23 |
| [RL (GRPO) Experiment](live/chains/rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-08-14 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe | 2026-08-23 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](outbound/upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-08-23 |
| [`StateDictStager` bug](outbound/upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-08-23 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](live/TODO.md) | Open work items | 2026-08-23 |
