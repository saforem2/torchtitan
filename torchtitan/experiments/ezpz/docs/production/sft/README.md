# Production SFT

> **2026-11-09 — Corpus mix note:** FineWeb-Edu 60/40 kind-shift as the next pre-training lever vs 94.8% DCLM olmo-mix — see [fineweb-edu-60-40-kind-shift.md](fineweb-edu-60-40-kind-shift.md) (SmolLM2 33.6% @38B, W&B `xakno7kg`/`ur6pqqhp` narrow-SFT halving 0.205->0.09/0.105 cited). MMLU gating companion: [mmlu-letter-prior-gating.md](../../evals/mmlu-letter-prior-gating.md) (constant-D 0.2689 > all 2B 0.2413-0.2655 — harness prior, not model bug).

> **Last updated: 2026-11-05.** B2v2 stage-2 SFT training + eval complete (jobs 12482423/12482485 training Exit 0, 12482572/73 eval Exit 0 on frameworks 2026.1.0 vLLM 0.26 XPU): **0.09/0.105 CoT vs B2 0.205** — more narrow epochs hurt, same as B3/B4. Training: loss 0.8069, acc 0.8198, W&B `xakno7kg`/`ur6pqqhp`. See [POST-TRAINING-2B](../POST-TRAINING-2B.md#b2v2----stronger-stage-2-sft-on-ckpt-900-2026-11-04) and [journal](../../journal.md). 16N fanout wired, not yet executed. Everything else below is finished/blocked/abandoned as of 2026-08-30.

> [!IMPORTANT]
> **Start here: [AuroraGPT-2B post-training status](../POST-TRAINING-2B.md)** --
> the complete SFT + RL picture in one page.
>
> Headline: **accuracy lives in SFT structure, not mix tuning and not RL.**
> Two-stage SFT (general math -> then GSM8K CoT) scores **0.205** GSM8K-CoT;
> every single-stage rebuild lands **0.02-0.065**. 100 steps of GRPO on top
> moved accuracy **0.205 -> 0.215** -- noise. ~0.2 is near this base's ceiling.
>
> **Deliverables:** `checkpoint-93` (B2) for CoT, `checkpoint-900` for
> general instruct.
>
> **Disk check, 2026-08-30:** `checkpoint-93` is present
> (`outputs/sft/agpt2b-gsm8k-r1cot-2n/`) and so is `checkpoint-900`'s
> sharded form, but the consolidated **`checkpoint-900-hf` export no
> longer exists** -- the only surviving `-hf` exports under
> `outputs/sft/` are `checkpoint-729-hf`, `checkpoint-100-hf` and
> `aurora2b-sophiag-metamathqa-32n/checkpoint-400-hf`. Re-consolidate
> from `checkpoint-900/` (96 `.distcp` shards, intact) before pointing
> any tool at a `-hf` path.
>
> **Do NOT use `checkpoint-8672`** -- it catastrophically forgot (hellaswag
> 0.593 -> 0.273, arc_easy 0.694 -> 0.298, at/near chance). The
> `tulu_math_uc_mix_full` header used to call it "the deliverable"; **fixed
> 2026-08-16** -- that header now says NOT the deliverable and points at
> `checkpoint-900`, matching its own body.
>
> **B3/B4 are still not in the table below.** They are single-base CoT
> experiments whose results live in
> [POST-TRAINING-2B](../POST-TRAINING-2B.md): B3 (instruct+CoT mix) and B4
> (finish-and-reweight) both FAILED to recover B2, which is why B2's
> `checkpoint-93` remains the CoT deliverable. Listing them here would need a
> row shape this table does not have (they are ablations off one base, not
> base->recipe pairs), so the status page stays their home.

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
| `agpt/2b-mds` (`global_step138650`) | [tulu_math_uc_mix](agpt/2b-mds/tulu_math_uc_mix/README.md) | **complete** | 0.77 | 4.5B | `outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf/` (present on disk, 7.5 GB) | 3 epochs, 32N, GBS=6144, metamathqa-swap mix, 4 autoretry attempts, 3 SIGABRTs survived (2026-06-10) |
| `agpt/2b-mds` (`global_step138650`) | [tulu_math_uc_mix_full (FULL big mix, ~54B tok)](agpt/2b-mds/tulu_math_uc_mix_full/README.md) | **COMPLETE (8N)** | 1.34 -> **0.357** @ step 8672/8672 (epoch 1.0) | ~54B target | `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900/` -- **sharded only; the `-hf` export is gone** (see below) | More-tokens SFT on the SAME 2b-mds base: FULL OpenMathInstruct-2 mix (53.3M packed seqs), 1 epoch, seq=1024. Ran at **8N** -- 32N GPU-page-faults (384-rank scale fault; bisect: 8N clean / 12N fault). Full saga: [launch report](../../experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md) |
| `agpt/2b-v2-256n` (`step92859`, completed 4.674T base) | [tulu_math_uc_mix](agpt/2b-v2-256n/tulu_math_uc_mix/README.md) | **blocked (v2-base oneCCL scale crash at 32N)** | -- | -- | `outputs/sft/agpt-2b-v2-256n-tulu-mix-32n-gbs6144/` | First SFT on the completed v2 base. 2N smoke green (loss 2.08->1.7, job 12470086); 32N deterministically hits a scale-only oneCCL GPU fault (jobs 12470254/258/262, unsolved). Prep: [report](../../experiments/agpt/sunspot/2026-07-06-sft-2b-v2-256n-base-prep.md) |
| `agpt/2b-mds` (`global_step138650`) via `checkpoint-900-hf` | **B2v2 gsm8k-r1cot 3ep/6ep** — see [POST-TRAINING-2B](../POST-TRAINING-2B.md#b2v2----stronger-stage-2-sft-on-ckpt-900-2026-11-04) | **complete, eval 0.09/0.105 vs B2 0.205 — worse** | 0.8069 / 0.77 | ~11k seqs x 3/6ep | `outputs/sft/agpt2b-b2v2-distill-cot-2n-ep6/final` (ckpts 2/4/6+final ep3, 2/4/6/8/10/12+final ep6) | Stage-2 SFT extending B2 lineage (ckpt-900 -> gsm8k-r1cot). 2N Sunspot jobs **12482423** (`xakno7kg`) / **12482485** (`ur6pqqhp`) Exit 0; eval **12482572/73** (`frameworks/2026.1.0 vLLM 0.26 XPU`, Exit 0) **0.09 (0.86/416/22)** / **0.105 (0.91/378/15)** vs B2 0.205 (0.985/282). More narrow epochs hurt — same as B3/B4. Distill mix shelved (HF cache race). Next: pre-tokenized OpenThoughts distill via [fineweb-edu-60-40-kind-shift.md](fineweb-edu-60-40-kind-shift.md) (CPT lever) |
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

## Further reading (2026-11-09)

- [FineWeb-Edu 60/40 kind-shift](fineweb-edu-60-40-kind-shift.md) — why 94.8% DCLM -> 60% FineWeb-Edu / 40% DCLM is a change of kind, with B2v2/B3/B4 narrow-SFT evidence (W&B `xakno7kg`/`ur6pqqhp`) and CPT availability on Aurora/Sunspot.
- [MMLU letter-prior gating](../../evals/mmlu-letter-prior-gating.md) — constant-D 0.2689 > all 2B (0.2413-0.2655) is a harness prior; do not gate promotion on MCF MMLU.
