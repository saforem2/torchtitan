# SFT recipe: gs138650 x tulu_math_uc_mix (FULL big mix, ~54B tokens)

> **Last updated: 2026-07-17.**
> **Status: COMPLETE at 8N -- step 8672/8672, epoch 1.0, final loss 0.357,
> mean_token_accuracy 0.902** (finished 2026-07-16). Head job **12470350**
> (2026-07-12) + a 10-link afterany chain (... -> 12470478 -> 12470479 final);
> `checkpoint-8672` saved, consolidated to `checkpoint-8672-hf` (the deliverable).
> Eval sweep in progress -- see [evals/](evals/README.md). This is the
> "more tokens" SFT: the SAME gs138650 base as
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
| Submit script | [`rl/scripts/sft/agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh`](../../../../../../rl/scripts/sft/agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh) |
| Output dir | `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/` |
| Expected steps | ~8,672 (1 epoch at GBS=6144); ~24s/step -> ~58h -> ~5 x 12h windows |
| Fault tolerance | afterany chain + `--resume_from_checkpoint`, save_steps=50, NO ezpz auto-retry |

## The data pipeline (why this took a while)

The full OpenMathInstruct-2 mix needed an offline build pipeline; each stage hit
a distinct failure, all now fixed. Full blow-by-blow in the
[launch report](../../../../../experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md).
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

## Loss trajectory

[![SFT training curves](charts/sft-curves.svg)](charts/sft-curves.svg)

Loss + grad_norm + LR + mean-token-accuracy + entropy + cumulative-tokens over
the run so far (regenerated from the latest checkpoint's `trainer_state.json` by
[`scripts/plot_sft_curves.py`](scripts/plot_sft_curves.py), wired into the
[refresh catch-all](../../../../../../scripts/update_all_charts.sh)). The token axis
stitches TRL's per-chain-link `num_tokens` counter into a monotonic total (each
afterany continuation resumes and re-inits the counter). Chart is absent until
the first `update_all_charts.sh` run lands it on Sunspot.

## Run status

| Job | Date | Steps | Loss | Notes |
|-----|------|------:|-----:|-------|
| 12470350 (head) | 2026-07-12 | 0 -> ~790/8672 | 1.343 -> 0.864 | 8N head; save_steps=50, checkpoints through step-850. mean_token_accuracy 0.685 -> 0.77, no GPU fault. **Idle-watchdog SIGTERM (rc=124) at step ~790**: a genuine ~30-min mid-training hang (not a crash/walltime; checkpoint saves are ~9s), afterany chain recovered. wandb `summer-mountain-82` / [vkwpxnqh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/vkwpxnqh). |
| 12470351 (cont 1) | 2026-07-12 | ~850 -> ~1535 | 0.87 -> 0.75 | Resumed from checkpoint-850. **Idle-watchdog SIGTERM (rc=124) at step ~1535** -- HANG #2 (same ~30-min-silence signature as the head; checkpointed to 1600 before dying). afterany recovered. wandb `peachy-morning-...` / [hrbfiwk7](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/hrbfiwk7). |
| 12470352 (cont 2) | 2026-07-12 | ~1600 -> ~2050 | 0.739 -> ~0.70 | Resumed from checkpoint-1600. Ended in ezpz#163 UnicodeDecodeError; chain then hit the DISK-FULL incident below (12470353+ all died with `Disk quota exceeded`). wandb `dashing-water` / [ghltmvgf](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/ghltmvgf). |
| **DISK-FULL incident** | 2026-07-13 | -- | -- | **datascience project quota hit 10T/10T on /lus/tegu** -> every job died in ~3s with `Disk quota exceeded` (empty logs), draining the whole chain. Root cause: save_steps=50 keep-all (~44 ckpts x 23G) + old SFT runs + 104 core dumps. Freed ~2T (this run's intermediate ckpts, old-SFT-run ckpts, core dumps; all `-hf` deliverables + checkpoint-729-hf preserved) -> project back to ~9.06T. |
| 12470375/376 (cont 3) | 2026-07-13 | resume from **900** | ~0.865 | checkpoint-2050 was an INCOMPLETE disk-full casualty (FSDP shards but no `trainer_state.json`/rng) -> unusable for HF resume; moved aside as `checkpoint-2050.incomplete-diskfull`. Highest COMPLETE kept ckpt was 900 (thinning had removed 950-2000), so resumed from 900 -- **~1150 steps recompute lost**. 375 died; 376 resumed clean (loss 0.865, lr continuous). wandb [7hps3eu1](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/7hps3eu1) (376, 900->1650). |
| 12470377-380 (cont 4-7) | 2026-07-13/14 | ~1750 -> ~2400 | 0.739 -> 0.656 | Four short chain links, each ended by the same ~30-min idle-watchdog SIGTERM (HANGs #4-#7, ~150-250 steps/link); afterany recovered every time. wandb [xjkrtv4y](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/xjkrtv4y) (377) / [fdxw19mr](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/fdxw19mr) (378) / [i44624kh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/i44624kh) (379) / [5kpf4glh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/5kpf4glh) (380, ->2400). |
| 12470436 (cont 8) | 2026-07-14 | 2450 -> ~4350 | 0.660 -> 0.496 | Resumed from checkpoint-2450 (relaunched this session, disk freed 2.65T first). **First long clean link (~1900 steps, no idle-hang)** -- ran to its 12h walltime. wandb [fhjznrn4](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/fhjznrn4). |
| 12470437 (cont 9) | 2026-07-14/15 | 4350 -> ~6200 | 0.511 -> 0.409 | Auto-took-over via afterany when 436 hit walltime. Clean ~1850 steps to its own 12h walltime (checkpoint-6200). wandb [30ntcag7](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/30ntcag7). |
| **12470478 (cont 10)** | 2026-07-15 | 6200 -> **~7050 (running)** | 0.409 -> **0.378** | Resumed from checkpoint-6200 (zero steps lost). Currently running; epoch **0.82**, accuracy **0.893**, LR cosine-decaying (cont 11 **12470479** queued). wandb [8gbmuj37](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/8gbmuj37). |

<!-- RUN-PROGRESS -->

> **DISK-FULL + lost-recompute lesson (2026-07-13):** the /lus/tegu `datascience`
> project quota filled (save_steps=50 keep-all is ~1T/run); jobs 3s-died on
> `Disk quota exceeded`. After freeing ~2T, the resume revealed
> checkpoint-2050 was written incompletely when the disk filled (shards but no
> `trainer_state.json`). Because thinning had kept ONLY 2050 + 300/600/900, the
> best *complete* fallback was 900 -> ~1150 steps recompute. **Fixes for next
> time:** (1) keep the latest 2-3 COMPLETE checkpoints, never just the single
> latest (the newest is the one most likely mid-write/corrupt); (2) add a
> keep-latest-N thinning policy so the disk never fills; (3) monitor project
> quota, not just user quota.

### wandb runs (one per chain link)

This run uses the afterany chain (NOT ezpz auto-retry), so each chain link
relaunches a fresh `train_sft` process and starts its OWN wandb run under
project [`aurora_gpt/torchtitan.ezpz.sft`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft).
Each continuation resumes from the latest checkpoint (not step 0). Run ids:

| Chain link | wandb run | Steps covered |
|---|---|---|
| 12470350 (head) | [vkwpxnqh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/vkwpxnqh) (`summer-mountain-82`) | 0 -> ~790 (idle-watchdog HANG #1) |
| 12470351 (cont 1) | [hrbfiwk7](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/hrbfiwk7) (`peachy-morning-...`) | ~850 -> ~1535 (idle-watchdog HANG #2) |
| 12470352 (cont 2) | [ghltmvgf](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/ghltmvgf) (`dashing-water-...`) | ~1600 -> ~2050 (ezpz#163, then disk-full) |
| 12470376 (cont 3) | [rx5p8ifz](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/rx5p8ifz) + resumed | resume from **900** (2050 was incomplete disk-full casualty; ~1150 steps recompute) |
| 12470376 (cont 3) | [7hps3eu1](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/7hps3eu1) | 900 -> ~1650 |
| 12470377-380 (cont 4-7) | [xjkrtv4y](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/xjkrtv4y) / [fdxw19mr](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/fdxw19mr) / [i44624kh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/i44624kh) / [5kpf4glh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/5kpf4glh) | ~1750 -> ~2400 (four idle-hang links) |
| 12470436 (cont 8) | [fhjznrn4](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/fhjznrn4) | 2450 -> ~4350 (clean, walltime) |
| 12470437 (cont 9) | [30ntcag7](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/30ntcag7) | 4350 -> **5950 (running)** |

> **Idle-hang pattern (update 2026-07-15):** chain links through step ~2400
> (head + cont 1-7) each ended in a ~30-min-silence idle-watchdog SIGTERM
> (rc=124) mid-training -- not a crash or walltime -- ~every 150-750 steps, so
> the run advanced but burned ~1 chain link per hang. **The pattern then
> abated**: cont 8 (12470436) ran ~1900 steps clean to walltime and cont 9
> (12470437) has run 4350 -> 5950 without an idle-hang. The afterany chain +
> save_steps=50 recovered every hang cleanly (<=50 steps lost). Root cause of
> the early hangs still unattributed (possible node/IO transients that eased);
> no longer the active blocker it was through step ~2400.

### Early trajectory (job 12470350)

First ~482 steps at 8N (GBS=6144): loss **1.343 -> 0.912**, mean_token_accuracy
**0.685 -> ~0.75**, grad_norm settled from 9.2 to ~0.29. Clean textbook SFT
descent, consistent with the completed gs138650 SFT's start (~1.16 -> 0.77 over
729 steps on the small mix). Checkpoints accumulate every 50 steps.

## Evals -- full picture (2026-07-17): the mix works EARLY, dies at full epoch

Full base-LM sweep (0 -> 8672) + IFEval in [`evals/README.md`](evals/README.md).
**Key finding:** the final checkpoint-8672 **catastrophically forgot** -- base-LM
benchmarks collapsed to ~random chance (hellaswag 0.59 -> 0.27, arc_easy
0.69 -> 0.30, collapse between step ~1500-4500) AND IFEval went flat-to-down vs
baseline. It overfit the OpenMathInstruct-2 math-CoT distribution (train loss
0.357) and lost everything else.

**BUT checkpoint-900 (~5.7B tokens) is the real deliverable:** IFEval
matches/beats the metamathqa-729 SFT (prompt-strict 0.253 vs 0.244; inst-loose
0.417 vs 0.411) WHILE retaining base-LM capability (pre-collapse). It is the
best of all four models evaluated.

**Root cause:** LR 2e-5 stayed >1e-5 through step ~4350 (cosine decay only bites
the 2nd half); ~4000 steps at high LR on a narrow distribution = catastrophic
forgetting. The metamathqa recipe survived only by STOPPING at 729 steps.
**Recipe lesson:** cap full-mix SFT at O(1000) steps (or drop the LR); a full
epoch at peak LR is the mistake, not the mix. GRPO (vLLM server-mode,
checkpoint-900): accuracy_reward climbed **0.31 -> ~0.74** in ~20 steps -- a
strong RL starting point (hit 6h walltime, not converged; see evals).

**Deliverable: `checkpoint-900-hf`** (NOT 8672).

## Comparison to the completed 729-step SFT

The prior production SFT
([tulu_math_uc_mix, metamathqa swap](../tulu_math_uc_mix/README.md))
reached final loss 0.77 over 729 steps / ~4.5B tokens. This run keeps the same
base + recipe weights but uses the FULL OpenMathInstruct-2 mix (~54B tokens,
~12x the tokens) for 1 epoch. The eval comparison (base-LM benchmarks, IFEval,
downstream GRPO signal) will show whether the ~12x-larger math+instruction SFT
corpus improves on the metamathqa-swap deliverable.
