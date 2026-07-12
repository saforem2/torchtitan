# Production SFT

> SFT'd checkpoints derived from pre-trained AuroraGPT models. These
> are the inputs to downstream alignment work (GRPO, DPO, …) and the
> deliverables for instruction-tuned model releases.
>
> Pre-training trajectories live under
> [`docs/production/agpt/`](../agpt/README.md). This page tracks the
> recipes that run on top of those pre-trained checkpoints.

## What a "production SFT" deliverable is

Each entry below is a *recipe + checkpoint pair*: a specific
data-mix-and-hyperparameter combination applied to a specific base
model, producing a usable HF-format checkpoint and a comparison
against the base on a standardized eval suite.

The layout mirrors the pre-training layout (`docs/production/agpt/<size-base>/`):
base directories use the same `agpt/<size>-<base>` names as pre-training
(e.g. `agpt/2b-mds` is the AuroraGPT-2B MDS stage-3 base = `global_step138650`;
`agpt/2b-v2-256n` is the completed v2 256N base), and each base holds one or
more `<recipe>/` subdirs:

```
docs/production/sft/agpt/<size>-<base>/<recipe>/
├── README.md               trajectory overview (run table, loss, checkpoints)
├── failover-story.md       per-job operational writeup (when interesting)
└── evals/
    ├── README.md           base-LM benchmark comparison vs the base model
    ├── ifeval.md           instruction-following metric (when run)
    └── grpo-smoke.md       downstream RL signal (when run)
```

Recipes are named for their dataset mix when concise (e.g.
`tulu_math_uc_mix`), or by their distinguishing hyperparameter
(`metamathqa_only`, `dpo_v1`, etc.) when they share a mix with
another recipe.

## Index

| Base | Recipe | Status | Final loss | Tokens | Checkpoint | Trajectory |
|---|---|---|---:|---:|---|---|
| `agpt/2b-mds` (`global_step138650`) | [tulu_math_uc_mix](agpt/2b-mds/tulu_math_uc_mix/README.md) | **complete** | 0.77 | 4.5B | `outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf/` | 3 epochs, 32N, GBS=6144, metamathqa-swap mix, 4 autoretry attempts, 3 SIGABRTs survived (2026-06-10) |
| `agpt/2b-mds` (`global_step138650`) | [tulu_math_uc_mix_full (FULL big mix, ~54B tok)](agpt/2b-mds/tulu_math_uc_mix_full/README.md) | **in progress (8N, job 12470350+chain)** | -- (1.34->0.95 @ step 305) | ~54B target | `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/` | More-tokens SFT on the SAME 2b-mds base: FULL OpenMathInstruct-2 mix (53.3M packed seqs), 1 epoch, seq=1024. Runs at **8N** -- 32N GPU-page-faults (384-rank scale fault; bisect: 8N clean / 12N fault). Full saga: [launch report](../../experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md) |
| `agpt/2b-v2-256n` (`step92859`, completed 4.674T base) | [tulu_math_uc_mix](agpt/2b-v2-256n/tulu_math_uc_mix/README.md) | **blocked (v2-base oneCCL scale crash at 32N)** | -- | -- | `outputs/sft/agpt-2b-v2-256n-tulu-mix-32n-gbs6144/` | First SFT on the completed v2 base. 2N smoke green (loss 2.08->1.7, job 12470086); 32N deterministically hits a scale-only oneCCL GPU fault (jobs 12470254/258/262, unsolved). Prep: [report](../../experiments/agpt/sunspot/2026-07-06-sft-2b-v2-256n-base-prep.md) |

## Why SFT is a separate production stage (and not just an `experiment/`)

SFT'd checkpoints get re-used. The
`agpt/2b-mds/tulu_math_uc_mix` (checkpoint-729-hf) artifact is the input
to every subsequent GRPO experiment on this model — every alignment
experiment, every chat-mode eval, every red-team test. That makes it
a first-class production asset, not a one-off date-stamped report.

If you spin up a *new* recipe (different mix, different
hyperparameters, or a different base model), it gets its own folder
under `docs/production/sft/agpt/<size>-<base>/<recipe>/`. The
date-stamped `docs/experiments/agpt/sunspot/<date>-sft-…md` reports
are kept as the rolling debug log; once a recipe stabilizes (its
checkpoints are being consumed by downstream work), it should be
promoted to a folder here.
