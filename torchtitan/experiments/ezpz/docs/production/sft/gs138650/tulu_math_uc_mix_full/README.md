# SFT recipe: gs138650 x tulu_math_uc_mix (FULL big mix, ~54B tokens)

> **Last updated: 2026-07-12.**
> **Status: in progress at 8N.** Job **12470350** + afterany chain (Sunspot,
> launched 2026-07-12). This is the "more tokens" SFT: the SAME gs138650 base as
> the completed 729-step SFT, but over the FULL OpenMathInstruct-2
> `tulu_math_uc_mix` (~53.3M packed sequences, ~54B tokens, 1 epoch) instead of
> the small metamathqa-swap mix (~4.5B tokens).
>
> Runs at **8N, not 32N**: the 32N run GPU-page-faults at step 1-2 (a 384-rank
> scale fault; see "The 8N-vs-32N scale fault" below). 8N (96 ranks) is the
> largest scale proven safe by a bisect.

## Quick reference

| Field | Value |
|-------|-------|
| Base model | `global_step138650` (vocab 256000, 1.99B params, llama arch) -- the checkpoint the completed 729-step SFT trained on |
| Recipe | registered `tulu_math_uc_mix` = tulu-3 0.65 + **OpenMathInstruct-2** 0.15 + ultrachat-200k 0.20 (`all_exhausted`) -- the FULL 14M-row OpenMathInstruct-2, not the metamathqa swap |
| Materialized mix | ~93.1M interleaved rows -> **53,276,203 packed sequences** at len 1024 |
| Trainer | TRL `SFTTrainer` (HF Trainer base, FSDP full_shard) |
| Sequence length | 1024, packing pre-applied offline, `assistant_only_loss` via baked-in `assistant_masks` |
| Hyperparameters | LR 2e-5 (cosine to 0), bf16, AdamW (TRL default), 1 epoch |
| Scale | **8 nodes x 12 = 96 ranks**, GBS = 6144 (bsz 2 x gas 32), FSDP full_shard |
| Submit script | [`rl/scripts/sft/agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh`](../../../../../rl/scripts/sft/agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh) |
| Output dir | `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/` |
| Expected steps | ~8,672 (1 epoch at GBS=6144); ~24s/step -> ~58h -> ~5 x 12h windows |
| Fault tolerance | afterany chain + `--resume_from_checkpoint`, save_steps=50, NO ezpz auto-retry |

## The data pipeline (why this took a while)

The full OpenMathInstruct-2 mix needed an offline build pipeline; each stage hit
a distinct failure, all now fixed. Full blow-by-blow in the
[launch report](../../../../experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md).
Short version:

1. **Interleave build** was ~90 min single-threaded -> vectorized to ~5s
   (bit-identical; also upstream PR huggingface/datasets#8318).
2. **Runtime tokenize** of 93M rows (~8.5h) blew the job watchdog -> moved
   offline via `--pretokenize_to` / `--pretokenized_dataset` (TRL skips prep
   when `input_ids` present).
3. **save_to_disk** OOM/slowness -> parallel `save_to_disk(num_proc)` with no
   pre-flatten; **packing map** OOM at high num_proc -> walked down to
   `dataset_num_proc=8`.
4. **seq_len 2048** OOM'd the XPU tile -> reverted to the proven **1024** corner
   (bsz 2 / gas 32 here to hold GBS=6144 at 8N).
5. **TRL re-packs pre-packed data** -> `packing=False` auto-set when the dataset
   has `seq_lengths`.
6. **ezpz `--auto-retry`** false-positives on TRL (`stuck_pre_training`, since
   it looks for `step=` markers TRL never emits) -> dropped; fault tolerance is
   now the afterany chain.

## The 8N-vs-32N scale fault

The 32N run (384 ranks) GPU-page-faults at step 1-2:

```
Segmentation fault from GPU at 0x... ctx_id: 5 (CCS) type: 0 (NotPresent), access: 1 (Write), banned: 1, aborting
-> rank 221 died from signal 6
```

Ruled out (evidence-based): **bad node** (reproduces on rank 221 across
different nodes), **OOV token id** (full 53M-row scan: max 255998 < base vocab
256000). Confirmed a **384-rank scale fault** (same class as
`project_sft_v2_base_oom_badnode`, base-independent). Scale bisect (jobs
12470343/346/347/348/349):

| N | ranks | result |
|---:|---:|---|
| 2 / 4 / 8 | 24 / 48 / 96 | **clean** (20 steps, loss descends) |
| 12 / 16 / 32 | 144 / 192 / 384 | GPU segfault at step 1-2 |

-> **8N = 96 ranks is the largest safe scale.** The onset (~96-144 ranks)
is near the recurring ~186-192 dp_degree boundary on this XPU/torch stack.
The scale fault itself remains an open infra issue; running at 8N sidesteps it.

## Run status

| Job | Date | Steps | Loss | Notes |
|-----|------|------:|-----:|-------|
| 12470350 (+chain 351-355) | 2026-07-12 -> | ~128/8672 (in progress) | 1.343 -> 1.005 | 8N head; save_steps=50, checkpoint-50/100/150 saved. mean_token_accuracy 0.685 -> 0.735, grad_norm settled ~0.27. No GPU fault at 8N. |

<!-- RUN-PROGRESS -->

### Early trajectory (job 12470350)

First ~128 steps at 8N (GBS=6144): loss **1.343 -> 1.005**, mean_token_accuracy
**0.685 -> 0.735**, grad_norm settled from 9.2 to ~0.27. Clean textbook SFT
descent, consistent with the completed gs138650 SFT's start (~1.16 -> 0.77 over
729 steps on the small mix). Checkpoints accumulate every 50 steps.

## Comparison to the completed 729-step SFT

The prior production SFT
([tulu_math_uc_mix on gs138650, metamathqa swap](../../aurora2b/tulu_math_uc_mix/README.md))
reached final loss 0.77 over 729 steps / ~4.5B tokens. This run keeps the same
base + recipe weights but uses the FULL OpenMathInstruct-2 mix (~54B tokens,
~12x the tokens) for 1 epoch. The eval comparison (base-LM benchmarks, IFEval,
downstream GRPO signal) will show whether the ~12x-larger math+instruction SFT
corpus improves on the metamathqa-swap deliverable.
