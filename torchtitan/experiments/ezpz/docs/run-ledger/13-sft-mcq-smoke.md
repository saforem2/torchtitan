# 13 - SFT MCQ Template Probe Smoke (2N, 20 Steps, Grain Streaming)

> **Branch:** `ezpz-cpt111k-eval-sft` - covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** 2N XPU smoke of `agpt_2b_mds_sft_mcq_template_probe_streaming` (40/30/20/10 CoT/tulu/MCQ/math) on CPT-FWE 136800 HF (3.7G 9/9, job `12484888`); Grain streaming, NO pre-tokenize; queued, pending `o` file.

---

## 1. What we run - 20-step SFT Grain template-learning probe

| Field | Value | Source |
|---|---|---|
| Config | `agpt_2b_mds_sft_mcq_template_probe_streaming` | `agpt/config_registry.py:910-921` |
| Builder | `sft_mcq_template_probe_streaming_dataset()` | `agpt/sft_data.py:551-642` |
| Base HF | `/lus/tegu/projects/datasets/datasets/outputs/evals/cpt-fwe/step-136800/hf` (also `.../foremans/outputs/...` via Tegu routing; 3.7G 9/9, job `12484888`) | `12-cot-136800-eval.md:17` `12-sft-mcq-template-probe.md:59` |
| Base DCP | `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800` | same |
| Overrides | `MDS154391_SFT_BASE` / `MDS154391_SFT_DCP` env so next Grain-CPT ckpt swaps in without edit | `config_registry.py:695-703` |
| Trainer | `FaultTolerantTrainer + GrainDataLoader + EZPZTokenizer` (~4.5k tps/tile) | `sft_data.py:1-12` |
| Dataset | `DatasetMixConfig` 40 CoT-distill / 30 tulu / 20 MCQ (cais/mmlu auxiliary_train) / 10 math | `sft_data.py:551-642` |
| Steps | `20` via `--training.steps=20` (short template-learning probe) | script `sft_mcq_probe_smoke_2n.sh:83` |
| Pre-tokenize | **None.** Grain tokenizes on-the-fly; no `--pretokenize_to` / `save_to_disk` | contract + `sft_data.py` |

What it teaches: `format_hit_rate 0.0 -> 0.95` template adherence before correctness (the `P(template) * P(knowledge|template)` decomposition per `12-sft-mcq-template-probe.md S 1`).

Submit:

```bash
qsub torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_2n.sh
# record: qstat -u foremans ; grep -oP "step:\s+\K\d+" ...o<JOBID> | tail -1
```

## 2. Where logs / outputs land

| Artifact | Path | When |
|---|---|---|
| PBS o file | `/home/foremans/torchtitan/sft-mcq-probe-smoke-2n.o<JOBID>` + `logs/sft-mcq-probe-smoke-<JOBID>/run.log` | At `qsub` |
| Train log | `logs/sft-mcq-probe-smoke-<JOBID>/train.log` (losses via `grep -E "step:.*loss:"`) | During run |
| Outputs | `outputs/sft/mcq-probe/` (if checkpoint enabled; smoke disables `--checkpoint.no-enable`) | N/A for 20-step smoke |
| HF base | `/lus/tegu/projects/datasets/datasets/outputs/evals/cpt-fwe/step-136800/hf` | Already 3.7G |

Tegu note: `/home/foremans` and `/lus/tegu` share one Lustre (`172.22.8.115:/tegu/su/`) so Tegu vs HOME paths hit the same pool.

Pending job ID: queued via `qsub`; `o` file appears under `PBS_O_WORKDIR` once scheduled.

## 3. Why - format before correctness

CPT-FWE 136800 scores **CoT `format_hit_rate 0.0 / n_unclosed 200/200 / mean_gen_len 1527`** (`12-cot-136800-eval.md:40` job `12484889`, six probes 1k->136800 all identical) - CPT cannot emit the `<think>/<answer>\boxed{}` or `A/B/C/D` envelope by construction, so MMLU is flat `0.302 -> 0.317` inside chance band `0.2295-0.2689` (`03-eval-harness.md:72-77`).

SFT is the only lever that installs the envelope. This probe isolates **template adherence** (`format_hit_rate`) from **correctness** (`cot_accuracy` / `MMLU acc = P(template)*P(knowledge|template)`). Reading correctness before `format >= 0.90` conflates two bottlenecks - same category error as gating CPT on CoT per `SYNTHESIS.md:219` `12-sft-mcq-template-probe.md S 5`.

## 4. Expected results and next steps

### Expected (20 steps, per B2v2 ep3 precedent 900 steps -> format 0.985)

| Signal | Expect | Falsifier |
|---|---|---|
| `format_hit_rate` lifts within ~10 steps toward `>= 0.95` | Envelope learnable in O(1k) steps; B2v2 `tulu->ckpt-900->gsm8k-r1cot` did `0.985` in 900 | Stays `0.0` -> base swap or mix weight bump CoT `0.40->0.50` |
| Loss drops (training) | SFT loss `2.x -> 1.x` within 20 steps | Flat/rising -> LR or VENV issue |
| `mean_gen_len <= 550`, `n_unclosed <= 10` | Guards verbosity blow-up (B4a `1378/133`) per `12-sft-mcq-template-probe.md S 6` | `1527/200` again -> still degenerate |

### Next steps

- **If `format >= 0.95`:** scale to full SFT (1N `900` steps, same `40/30/20/10` mix) then re-eval CoT + MMLU MCF; only then gate RL (GRPO).
- **If `0.70 <= format < 0.95`:** bump CoT-distill `0.40->0.50` or extend `1N->2N` at same LR `2e-5` before widening mix.
- **If `format < 0.70`:** try Grain-CPT base upgrade (new `agpt_2b_grain_cpt_anneal` ckpt >136800) or cold-start `global_step138650` control; do not proceed to correctness.

### Submit record

```bash
qsub /home/foremans/torchtitan/torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_2n.sh
# qstat -u foremans  # capture JOBID
# tail -f /home/foremans/torchtitan/sft-mcq-probe-smoke-2n.o<JOBID>
```

---

## Provenance

- Base: HF `step-136800` `3.7G 9/9` jobs `12484888/12484889` `12-cot-136800-eval.md`.
- CoT `0.0 / 200 / 1527` at six probes: `12-sft-mcq-template-probe.md S 1` `03-eval-harness.md`.
- Mix + Grain contract: `agpt/sft_data.py:551-642` `agpt/config_registry.py:910-921` `components/data/{loader,dataset,packing}` `build_tokenizer.py:EZPZTokenizer`.
- Script pattern: `cpt_gentle_smoke_sunspot.sh` + `rl/scripts/sft/_smoke_agpt2b_v2_256n_tulu_mix_n2.sh` (VENV `xpu-torch216-sync-5a7a622f16` LD-first, `load_pinned_ezpz_utils`, `tegu:home A=datascience select=2 workq 00:45:00`).
