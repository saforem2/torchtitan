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

The 25 most-recently-changed docs by git commit date (across all 367
docs, not just the curated tables below). Auto-generated -- do not edit
by hand; run `utils/refresh_docs_readme_table.py` (or `refresh_all.sh`).

<!-- BEGIN recently-updated (auto-generated) -->
| Modified | Doc |
|---------:|-----|
| 2026-11-19 | [13 — Aurora Grain CPT Anneal Smoke (2N PVC)](./run-ledger/13-aurora-smokes.md) |
| 2026-11-19 | [12 — Grain-Native CPT Curriculum (olmo / FineWeb-Edu / Cosmopedia Anneal)](./run-ledger/12-cpt-grain-curriculum.md) |
| 2026-11-19 | [Production Training — agpt 4B (4.64B), OLMo-3 tokenizer, 4K](./production/agpt/4b/README.md) |
| 2026-11-19 | [Production Training — agpt 2B @ 512 nodes](./production/agpt/2b/n512/README.md) |
| 2026-11-19 | [Production Training — agpt 20B @ 512 nodes](./production/agpt/20b/n512/README.md) |
| 2026-11-19 | [Production Training — agpt 20B @ 256 nodes](./production/agpt/20b/n256/README.md) |
| 2026-11-19 | [Production Training — agpt 10B (9.48B), OLMo-3 tokenizer, 4K](./production/agpt/10b/README.md) |
| 2026-11-19 | [Production Training Runs — Aurora](./production/README.md) |
| 2026-11-18 | [Upstream Sync Log](./upstream-sync.md) |
| 2026-11-18 | [13 - SFT MCQ Template Probe Smoke (2N, 20 Steps, Grain Streaming)](./run-ledger/13-sft-mcq-smoke.md) |
| 2026-11-18 | [13 — Launches 2026-11-18 (Grain CPT Anneal + SFT MCQ Probe)](./run-ledger/13-launches-20261118.md) |
| 2026-11-18 | [13 — Grain CPT anneal smoke (2N)](./run-ledger/13-grain-cpt-smoke.md) |
| 2026-11-18 | [13 — Eval Harness Closeout (CoT 136800 Final + MMLU Flat + Live CPT)](./run-ledger/13-eval-harness-closeout.md) |
| 2026-11-18 | [12 — SFT MCQ Template Probe (Template Adherence Before Correctness)](./run-ledger/12-sft-mcq-template-probe.md) |
| 2026-11-18 | [12 — CoT Eval at CPT Step 136800 (CPT Cannot Solve CoT)](./run-ledger/12-cot-136800-eval.md) |
| 2026-11-18 | [11 — Eval Matrix (MMLU + Commonsense + CoT) — updated 2026-11-18 with CPT 136800](./run-ledger/11-eval-matrix.md) |
| 2026-11-18 | [05 — Infra / RL / Storage Ledger](./run-ledger/05-infra-rl-storage.md) |
| 2026-11-18 | [03 — Eval Harness Ledger (GSM8K-CoT + MMLU + Commonsense)](./run-ledger/03-eval-harness.md) |
| 2026-11-18 | [agpt-2B post-training plan: best general instruct model](./production/POST-TRAINING-PLAN.md) |
| 2026-11-18 | [Development Journal](./journal.md) |
| 2026-11-18 | [Eval Matrix Plan — AGPT-2B across MDS / TT-256N / TT-512N / CPT-FWE + post-training](./evals/eval-matrix/eval-matrix-2026-11-16.md) |
| 2026-11-18 | [Evaluation Results](./evals/README.md) |
| 2026-11-18 | [Pre-Training AuroraGPT with TorchTitan + 🍋 ezpz](./README.md) |
| 2026-11-17 | [10 — Infra Readiness — VENV, LD-first, PBS, Host Routing, Preflight](./run-ledger/10-infra-readiness.md) |
| 2026-11-17 | [09 — Gate & RL Probe Ledger (150k Decision Gate + GRPO Probe + Ranked Next-Course Matrix)](./run-ledger/09-gate-rl-probe.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-11-17 | [08 — SFT broad distill unpark (E7 Thread B §§4-5, SYNTHESIS E7, POST-TRAINING-PLAN phases 1-3)](./run-ledger/08-sft-broad-distill-unpark.md) |
| 2026-11-17 | [07 — Eval Sweeps Unpark (E2 + E4-E6)](./run-ledger/07-eval-sweeps-unpark.md) |
| 2026-11-17 | [06 — HF Ladder Unpark (E1 Scoring Ladder P0-P3, 14 Converts)](./run-ledger/06-hf-ladder-unpark.md) |
| 2026-11-17 | [04 — SFT broad distill (distill_cot_mix @ 8192, pretok-gated)](./run-ledger/04-sft-broad-distill.md) |
| 2026-11-17 | [02 — HF Conversion Batch Ledger (CPT-FWE DCP -> HF)](./run-ledger/02-hf-conversion.md) |
| 2026-11-17 | [01 — CPT-FWE 4N Frontier Ledger (60/40 FineWeb-Edu Kind-Shift)](./run-ledger/01-cpt-frontier.md) |
| 2026-11-17 | [Run-Ledger Index — agpt-2B Post-Training (ezpz-cpt111k-eval-sft @ a6135a79)](./run-ledger/00-index.md) |
| 2026-11-17 | [Thread E — SFT/RL Hyperparameter and Recipe Optimization](./exploration/thread-e-recipe-optimization.md) |
| 2026-11-17 | [B2 Autopsy: Why 0.205 Worked and Everything Else Did Not](./exploration/thread-d-b2-autopsy.md) |
| 2026-11-17 | [Eval Harness Robustness and Scoring Strategy for agpt-2B Post-Training](./exploration/thread-c-eval-harness.md) |
| 2026-11-17 | [Broad-Distill SFT Path: Deep Analysis](./exploration/thread-b-broad-distill.md) |
| 2026-11-17 | [CPT Trajectory Analysis: Is 60/40 FineWeb-Edu Learning, Plateauing, or Drifting?](./exploration/thread-a-cpt-trajectory.md) |
| 2026-11-17 | [agpt-2B Post-Training Synthesis: Where We Stand and What to Do Next](./exploration/SYNTHESIS.md) |
| 2026-11-17 | [LR Finder — agpt 5B OLMo-tokenizer](./experiments/lr-finder/agpt/5b/README.md) |
| 2026-11-17 | [OLMo-3-vocab LR finder: 5B / 10B at GBS=128 on Sirius (OLMo-3-1K)](./experiments/lr-finder/agpt/2026-11-12-olmo3-gbs128-sirius.md) |
| 2026-11-17 | [LR Finder -- agpt 10B OLMo-tokenizer](./experiments/lr-finder/agpt/10b/README.md) |
| 2026-11-16 | [Production Training — Dense (agpt) Models](./production/agpt/README.md) |
| 2026-11-16 | [Production Training — agpt 2B](./production/agpt/2b/README.md) |
| 2026-11-16 | [Production Training — agpt 20B](./production/agpt/20b/README.md) |
| 2026-11-16 | [AuroraGPT-2B post-training: complete status](./production/POST-TRAINING-2B.md) |
| 2026-11-16 | [Umbrella 9094039 -- 2346N, 24h, five seats (2026-11-16)](./production/2026-11-16-umbrella-9094039.md) |
| 2026-11-15 | [sft_distill_cot_4n — Phase 2 broad CoT lever verification](./production/sft_distill_cot_4n_verification.md) |
| 2026-11-15 | [RL guardrail — CoT-gated GRPO on agpt-2B](./production/rl_guardrail_note.md) |
| 2026-11-15 | [Verified OLMo-3-1K GBS=128 Sirius evidence -- 2026-11-11 to 2026-11-15](./experiments/lr-finder/agpt/data/2026-11-12-olmo3-gbs128-sirius/README.md) |
| 2026-11-14 | [Production Training Runs -- Polaris (A100)](./production/polaris/README.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-11-19 |
| [Umbrella `9094039`](./production/2026-11-16-umbrella-9094039.md) | 2346N/24h five-seat umbrella: 4B/10B reload vs idle watchdog, 20B-512 `step-10900` save segfault, 2B fork from `step-39900` | 2026-11-16 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-11-16 |
| [2B 256N](./production/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6524. | 2026-10-16 |
| [2B 512N](./production/agpt/2b/n512/README.md) | step-**46429** (4.67T tokens, 100.0% of 4.67T), loss 2.6869. | 2026-11-19 |
| [20B 512N](./production/agpt/20b/n512/README.md) | step-**10,800** (10,800 x 12,288 x 8,192 = **1,087.2B tokens**), loss 2.4606. | 2026-11-19 |
| [20B 256N](./production/agpt/20b/n256/README.md) | step-**22,000** (22,000 x 6,144 x 8,192 = **1,107.3B tokens**), loss 2.4219. | 2026-11-19 |
| [agpt 4B, OLMo-3, 4K](./production/agpt/4b/README.md) | Aurora 512N at step-11,700 (294.4B tokens), running in `9094039`; Sirius 2N finished; Polaris queued. | 2026-11-19 |
| [agpt 10B, OLMo-3, 4K](./production/agpt/10b/README.md) | Aurora 512N at step-4,900 (123.3B tokens); umbrella seat stopped (reloads exceeded the idle watchdog). 768N diverged at step 1. Sirius 8N finished; Polaris queued. | 2026-11-19 |
| [agpt 80B](./production/agpt/80b/README.md) | **Blocked at scale by a high-dp NaN; root cause unknown.** Optimizer-independent (SophiaG 512N, mano 62N); the earlier "bf16 residual-stream overflow" diagnosis is refuted ([what we know](./guides/known-bugs/80b-nan-what-we-know.md)). fp32 activations prevent it at ~3.4x cost; fp32-residual alone still NaNs at dp=192; no live 80B production, fp32-residual work dormant. Wall 2 (256N init segfault) separate + open. | 2026-10-29 |
| [80B 512N NaN incident (2026-07-03)](./experiments/agpt/aurora/20260703-80b-512n-sophiag-nan.md) | Incident record of the 512N NaN + NaN-abort guard. NOTE: the SophiaG-Hessian attribution was later disproven (2026-07-14, task #21) -- the NaN is optimizer-independent; the follow-on residual-overflow diagnosis is also refuted, root cause unknown (see `guides/known-bugs/80b-nan-what-we-know.md`). | 2026-10-16 |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-10-16 |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-10-16 |
| [agpt 2B](./production/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-11-16 |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-10-16 |
| [2B CPT (olmo x dolmino)](./production/cpt/README.md) | Continued-pretraining ratio sweep forked from the completed 2B base (step-92,859). 300B pilots done (dolmino-100 val 2.49, olmo50-50 val 2.60, both beat the olmo-100 plateau ~2.80); eval screen queued (8647850), winner scales to ~2.4T (MDS stage-2 match). | 2026-11-14 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-10-16 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-10-16 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-10-16 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-10-16 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-11-18 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](./guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](./experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-10-16 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-10-16 |
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-10-16 |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-10-16 |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-10-16 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-11-18 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-10-16 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-10-16 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-27.md) | 2-week retrospective | 2026-10-16 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-10-16 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Aurora quickstart: frameworks/2026.1.0](./guides/aurora-quickstart-frameworks-rc.md) | **Start here for new setups.** Validation-queue recipe on the RC module -- no venv tarball, no relocation step. The four required exports (libglog on `LD_LIBRARY_PATH`, both proxies, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`) each cost a failed job to find. Carries the validated 5-corner matrix from job `8789506`: compiled agpt TP=2 works here (the June `.venv` cannot), moe TP>1 needs `6e4e1996f`. | 2026-10-16 |
| [Aurora quickstart: shared torch 2.13 tarball](./guides/aurora-quickstart-tarball.md) | The pre-RC path -- debug-scaling queue, shared venv tarball, `relocate-venv.sh`. Correct until the RC is the default module on your nodes. | 2026-10-16 |
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-10-16 |
| [Reference Baselines](./baselines/README.md) | Training curves and benchmarks | 2026-10-16 |
| [Dense Model Configs](./configs/dense.md) | 2B / 20B / 50B / 80B | 2026-10-16 |
| [MoE Variants](./configs/moe.md) | 500M-10B | 2026-10-16 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](./scaling/README.md) | Top-level scaling landing page | 2026-10-16 |
| [agpt 2B scaling](./scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-10-16 |
| [agpt 20B scaling](./scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-10-16 |
| [agpt 80B scaling](./scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-10-29 |
| [MoE scaling](./scaling/moe.md) | Per-N TPS / MFU | 2026-10-16 |
| [Per-run Experiment Reports](./experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-10-16 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](./competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-10-16 |
| [RL (GRPO) Experiment](./production/rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-10-29 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe. Syncs 82 and 83 have both LANDED (`09b4ef235`, `609777a0e`). | 2026-11-18 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-10-16 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-10-16 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items. Audited 2026-08-31: the docs restructure and the 80B TP=2 OOM are done, the Aurora scaling study largely so; the 80B compile item and the production plan are superseded; MoE throughput is partly answered. The blendcorpus/Megatron aliasing item (the second `## 6.`) is the only fully open one. | 2026-10-16 |
