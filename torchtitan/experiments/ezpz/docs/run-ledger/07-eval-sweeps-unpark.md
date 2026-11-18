# 07 — Eval Sweeps Unpark (E2 + E4-E6)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` — ledger covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** E2 (commonsense+MMLU on 11 HFs) + E4 (B2 repro anchor) + E5/E6/E6b (LR x steps forgetting boundary) are submit-ready; E4 gates the sweep, E2 runs in parallel with live CPT chain.
> Every claim cites source file:line or job ID; inferences labeled `[INFERENCE]`.

---

## 1. Commonsense+MMLU sweep — E2 on every HF (P0-P2, 11 HFs)

### Purpose

E2 tests the dolmino question per `SYNTHESIS.md:68-77` E2: did CPT-FWE 60/40 regress commonsense (HS/ARC-Easy/PIQA) and did knowledge move off the letter prior (MMLU `max(CF,MCF)` + prior collapse)? Depends on E1 — HF must exist at `TEGU_ROOT/outputs/evals/cpt-fwe/step-$s/hf` per `SYNTHESIS.md:60-66`.

### HF ladder — 11 HFs scored (P0-P2)

| Phase | Steps | Count | Why | Source |
|-------|-------|------:|-----|--------|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | Dolmino shock hit by step 1000; constant LR 2e-6 should not shock — verify hypothesis, not loss | `SYNTHESIS.md:202-204` |
| **P1 — Early plateau** | `25000, 50000, 75000` (+ `74000/74200` already `3.7G 9/9` verified) | 3 new + 2 existing | 74k/74.2k already HF (9/9 files), fill gap to 50k where broad-distill was parked; 50k was `POST-TRAINING-PLAN.md` phase gate | `SYNTHESIS.md:205`, `03-eval-harness.md:292`, `02-hf-conversion.md` |
| **P2 — Frontier** | `100000, 111200, 125000, 143400` (+ `150000` gate deferred to E9) | 4 | 111.2k documented (`R step-111200 23G 556 dirs`), 143.4k snapshot = 75.2% of 190734 budget | `SYNTHESIS.md:206` |
| **Total P0-P2** | | **11** (including 74000 as anchor) | 11th is `74000` anchor; [INFERENCE: 3+3+4+1 anchor = 11; SYNTHESIS P0-P2 lists 10 converts + 1 anchor = 11 scored] | `SYNTHESIS.md:68-77` |

PBS for HF conversion: `select=1` `workq` `1N 02:00` `VENV=xpu-torch216-sync-5a7a622f16` LD-first, `flavor=2b-mds`, `PBS filesystems=tegu:home`, Tegu output `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-$s/hf` per `SYNTHESIS.md:61-64`, `03-eval-harness.md:370`.

### Benchmarks per HF — 5 tasks, 2 MMLU formats

| Benchmark | Shots | Metric | Base plateau | Why it gates | Source |
|-----------|-------|--------|-------------:|--------------|--------|
| HellaSwag | 0 | `acc_norm` | **0.560** | Dolmino-100 fell to **0.486 (-7.4pp)** at job `8647850` | `03-eval-harness.md:144-148`, `SYNTHESIS.md:231` |
| ARC-Easy | 0 | `acc` | **0.651** | Dolmino-100 **0.547 (-10.4pp)** monotonic collapse | same |
| PIQA | 0 | `acc_norm` | **0.733** | Dolmino -4.6pp — weak but cheap | `SYNTHESIS.md:233` |
| ARC-Challenge | 25 | `acc_norm` | **~0.31** | Catastrophic forgetting guard | `SYNTHESIS.md:234` |
| **MMLU MCF** (`mmlu`) | 5 | loglik letter `A-D` | band `0.2295-0.2689` | MCF at chance below ~1T; OLMES `max(CF,MCF)` | `03-eval-harness.md:66-82`, `SYNTHESIS.md:75` |
| **MMLU CF** (`mmlu_continuation`) | 5 | loglik answer text | same band ref | `max(CF,MCF)` monotonic in either; CF has signal at 400B+ | `SYNTHESIS.md:75-76` |

### PBS / lm-eval tasks per SYNTHESIS E2

**Commonsense thermometer (HS/ARC-Easy/PIQA/ARC-C):**

- **System:** Sunspot **or** Aurora per `03-eval-harness.md:368`, `SYNTHESIS.md:74`
- **Queue:** `workq`, cheap loglikelihood (0-shot, no generation)
- **Backend:** `lm-eval 0.4.10` HF backend, `frameworks/2026.1.0` MKL+CCL+compiler+PTI LD as in `submit_cot_eval.sh:15-20` per `03-eval-harness.md:294`
- **VENV:** `frameworks/2026.1.0` (module) with PTI lib `LD_LIBRARY_PATH=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-62kxpbr/lib64` per `submit_cot_eval.sh:16-17`
- **Env:** `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=0` per `submit_cot_eval.sh:18-20`
- **Do NOT use** `ezpz_setup_env` venv (transformers 5.6.2 breaks lm-eval) per `03-eval-harness.md:368`

Example commonsense sweep (1 HF; loop over 11):

```bash
# Sunspot workq — HellaSwag + ARC-Easy + PIQA per SHE, ARC-C per peer table
# Wrap: convert already done per E1 -> lm-eval HF backend
# SHOTS_SPEC loop per thread-c-eval-harness.md:579,664
qsub -l select=1 -l walltime=01:00:00 -l filesystems=tegu:home -q workq -A datascience \
  torchtitan/experiments/ezpz/scripts/eval/submit_eval.sh
# Or direct lm-eval on already-converted HF:
# lm_eval --model hf --model_args pretrained=/lus/tegu/.../step-75000/hf \
#   --tasks hellaswag,arc_easy,piqa,arc_challenge --device xpu:0 --batch_size auto
```

Source: `03-eval-harness.md:350-355`, `SYNTHESIS.md:74`.

**MMLU both formats — Aurora-only:**

- **System:** **Aurora only** per `SYNTHESIS.md:74`, `03-eval-harness.md:369`, `POST-TRAINING-2B.md:25`
- **Why:** Sunspot OOM `level_zero:0 40 63G OOM (compute_topk_scores)` unfixable on Sunspot XPU per `03-eval-harness.md:234-237`, `SYNTHESIS.md:74`
- **Queue:** `workq` `48h` per `03-eval-harness.md:241,369` (prior 12h pass died Exit -29 mid-MMLU)
- **Cost:** 57 subjects ~56k lm-eval requests per pass per `03-eval-harness.md:241`
- **Full 14k only:** Do NOT set `--limit` — sampling drove 0.028 noise (limit-10 `0.2281` vs limit-40 `0.2561` on same 30B ckpt) per `03-eval-harness.md:252-254`

Example MMLU sweep (Aurora, per-HF):

```bash
# Aurora workq 48h — both formats, max(CF,MCF), full 14k
qsub -l select=1 -l walltime=48:00:00 -l filesystems=tegu:home -q workq -A datascience \
  torchtitan/experiments/ezpz/scripts/eval/eval_mds_sweep.sh
# Inside: lm_eval --model hf --model_args pretrained=<hf-dir> \
#   --tasks mmlu,mmlu_continuation --num_fewshot 5 --device xpu:0
```

Source: `03-eval-harness.md:352-355`, `SYNTHESIS.md:74-76`.

### max(CF,MCF) band rule — mandatory scoring discipline

Every MMLU number reported as `max(CF,MCF)` with band + prior per `03-eval-harness.md:101-110` §1.5 seven rules, `SYNTHESIS.md:76`:

1. Read **group key** `mmlu` from `results.json`, not hand-rolled macro over `mmlu_*` (macro +0.005 high at step-46429: `0.2631` vs `0.2579`) per `03-eval-harness.md:101`.
2. State **format** (`MCF` vs `CF = mmlu_continuation`) for every number; report **both + `max(CF,MCF)`** per OLMES per `03-eval-harness.md:102`.
3. State **shot count** `n-shot=5` from `results.json` (persisted since `5ab9e50f6`) per `03-eval-harness.md:103`.
4. State **constant-letter band** `0.2295-0.2689` (always-A `0.2295`, B `0.2465`, C `0.2551`, D `0.2689`) per `03-eval-harness.md:72-77`.
5. Do not compare to SmolLM3 `44.1` — that is MMLU-CF contamination-free variant per `03-eval-harness.md:105`.
6. Score **every CPT checkpoint** in both formats (eval-only on existing ckpts) per `03-eval-harness.md:106`.
7. When new base clears band, **re-fit letter prior** `acc(subject) = sum_L p_L * keyfreq_L(subject)` and check residual dispersion `chi2/df -> 1` per `03-eval-harness.md:107`, `SYNTHESIS.md:235`.

Band anchor: **always-D `0.2689`** beats every measured agpt-2B MMLU (`0.2413-0.2655`) including flagship 2B step-46429 at 4.674T (`0.2579` group key) per `03-eval-harness.md:77-82`, `mmlu-letter-prior-at-chance.md:46-63`.

Proven 570q pilot on 30B step-2000 (3.93B tok, job `12474382`): MCF `0.2281` vs CF `0.2386` delta `+0.0105` SE `0.0256` not significant per `03-eval-harness.md:88-94`.

### E2 routing summary

| Workload | System | Queue | Walltime | Backend | Source |
|----------|--------|-------|----------|---------|--------|
| HS/ARC-Easy/PIQA/ARC-C | Sunspot or Aurora | `workq` | `01:00:00` per HF | `lm-eval 0.4.10` HF `frameworks/2026.1.0` | `03-eval-harness.md:368` |
| MMLU MCF+CF -> max | **Aurora only** | `workq` | `48:00:00` per HF | same | `03-eval-harness.md:369` |
| HF convert (E1 prereq) | Sunspot | `workq` | `02:00:00` | `xpu-torch216-sync-5a7a622f16` LD-first `flavor=2b-mds` | `SYNTHESIS.md:60-64` |

Minimum viable E2: convert `74000/74200+112k` (2-3 converts) + commonsense thermometer on each (1N cheap) + Aurora 48h MMLU both formats — <1 day walltime, zero new CPT compute, parallel with live CPT chain per `03-eval-harness.md:373`, `SYNTHESIS.md:70-77`.

---

## 2. CoT repro anchor — E4 (2N GBS384 seq2048 93 steps 3ep LR2e-5 cosine)

### Purpose — load-bearing sanity check

E4 reproduces B2 exactly before varying anything per `SYNTHESIS.md:89-96` E4, `thread-e-recipe-optimization.md:340-352` Config 1. Tests: does B2's `0.205` reproduce on current stack (`frameworks/2026.1.0` + TRL on XPU)? Source: `SYNTHESIS.md:93`.

**Why load-bearing:** B2 is the only CoT deliverable on this base per `03-eval-harness.md:33,159-165` (B2 = `checkpoint-93` `0.205/0.985/282` vs every alternative `0.02-0.105` per `POST-TRAINING-2B.md:77-82,100-102`). If E4 does not reproduce, stack or data drifted — root-cause before E5-E7 per `SYNTHESIS.md:96` decision rule, `thread-e-recipe-optimization.md:352` success criterion.

### Config

| Field | Value | Source |
|-------|-------|--------|
| **Base** | `checkpoint-900-hf` (IFEval 0.253, `global_step138650 -> tulu_math_uc_mix_full 8N`) | `SYNTHESIS.md:95`, `thread-e:342`, `agpt2b_b2v2_distill_cot_2n.sh:67` |
| **Data** | `gsm8k-r1cot` 7.4k ex, narrow — same as B2 stage-2 | `SYNTHESIS.md:95`, `thread-e:345` |
| **LR** | **2e-5 cosine** warmup 200 -> 0 over 3 epochs (93 steps total) | `SYNTHESIS.md:95`, `thread-e:345` |
| **GBS** | **384** (`LBS 2` x `GAS 8` x 24 ranks @ 2N) | `SYNTHESIS.md:95`, `thread-e:346`, `agpt2b_b2v2_distill_cot_2n.sh:107-109` |
| **Seq len** | **2048** (2048 mandatory — 1024 silently truncates `<answer>` tail via packing) | `SYNTHESIS.md:95`, `thread-e:347`, `agpt2b_b2v2_distill_cot_2n.sh:109` |
| **Steps / epochs** | **93 steps = 3 epochs** at GBS 384 on 7.4k ex | `SYNTHESIS.md:95` |
| **Scale** | **2N** `select=2` `PPN=12` 24 ranks, `workq` `02:00:00` | `SYNTHESIS.md:95`, `thread-e:349`, `agpt2b_b2v2_distill_cot_2n.sh:5` |
| **VENV** | `rl-monarch-torch214-locality-20261001` LD-first (TRL path, not `xpu-torch216`) | contract VENV mapping, `agpt2b_b2v2_distill_cot_2n.sh:47` |
| **PBS filesystems** | `tegu:home` (reads `/home` DCP via symlink, writes Tegu `outputs/sft`) | contract, `agpt2b_b2v2_distill_cot_2n.sh:4` |
| **Precision** | `bf16` FSDP `full_shard`, `assistant_only_loss`, `packing=True` | `agpt2b_b2v2_distill_cot_2n.sh:106-113` |
| **Eval** | `submit_cot_eval.sh` 1 tile vLLM fp32 `LIMIT=200` temp `0.0` `max_tokens 768` on `final/` | `SYNTHESIS.md:96`, `thread-e:352` |

### Halt-sweep gate

| Condition | Interpretation | Action | Source |
|-----------|---------------|--------|--------|
| `0.18-0.23` + `format >=0.95` + `gen_len 250-350` | **PASS** — reproduces B2's `0.205/0.985/282` within `+/-0.03` noise at `N=200` | Proceed to E5/E6 in parallel with E2 | `SYNTHESIS.md:96`, `thread-e:352` |
| Outside `0.18-0.23` or `format <0.95` or `gen_len` outside `250-350` | **FAIL** — stack/data drifted | **Halt sweep**, root-cause before E5-E7 | `SYNTHESIS.md:96` ("If outside, halt sweep — stack or data drifted") |

B2v2 timing anchor: 2N train `~2 min` + CoT eval `00:01:38` Exit 0 per `03-eval-harness.md:34-35` jobs `12482572/73` (`x1922c4s2b0n0/5b0n0`) at `LIMIT=200` on Sunspot `frameworks/2026.1.0` vLLM fp32.

### qsub command

```bash
# E4: Reproduce B2 exactly from checkpoint-900-hf (2N, GBS384, seq2048, 93 steps, LR2e-5 cosine)
# Base: $SUBMIT_DIR/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf
# VENV: rl-monarch-torch214-locality-20261001 + LD-first (TRL path)
qsub -v BASE_MODEL=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf,NUM_EPOCHS=3,SFT_DATASET=gsm8k-r1cot \
  torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh

# Eval the final checkpoint (Sunspot 1 tile, 40 min walltime, ~98s actual)
qsub -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-b2v2-distill-cot-2n-ep3/checkpoint-93,LIMIT=200 \
  torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
# Or eval the consolidated HF if produced:
# qsub -v CKPT=<final-hf-dir>,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
```

Source: `agpt2b_b2v2_distill_cot_2n.sh:6,67,75,100-114`, `submit_cot_eval.sh:10,24-34`, `SYNTHESIS.md:89-96`, `thread-e:340-352`.

Alternative Tegu-root form (PBS_O_WORKDIR defaults to `/lus/tegu/.../torchtitan`):

```bash
qsub -v BASE_MODEL=$HOME/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf \
  torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh
```

---

## 3. Forgetting boundary — E5 2e-5 constant to 600 (save 50) + E6 1e-5 to 1000 + E6b cosine-vs-constant

### What is unmeasured

No run on this branch has swept LR x steps on narrow data at any LR other than 2e-5 per `thread-e:58-60`, `SYNTHESIS.md:98-114` E5-E6. Known points: safe `~900` steps at 2e-5 on broad mix (tulu 0.65, GBS 6144), unsafe `~3000` steps at LR>1e-5 on same mix (hellaswag `0.593->0.273` per `thread-e:22-35`), and 6 extra steps at 2e-5 on pure gsm8k-r1cot halved CoT `0.205->0.09/0.105` per `thread-e:25`, `03-eval-harness.md:34-35`. This sweep maps the boundary.

All configs are **fresh SFT from ckpt-900 on gsm8k-r1cot** (not continuation of B2 — avoids "already near narrow optimum" confound) per `thread-e:336-338`.

### 7-config sweep table

Each row is one fresh SFT; eval at `ckpt-{50,100,300,600,1000}` + `final/` via `submit_cot_eval.sh` `LIMIT=200`. Hold fixed: base `checkpoint-900-hf`, data `gsm8k-r1cot` 7.4k ex, warmup 200, seq 2048, GBS 384 (or 768 for config 6), bf16 FSDP `full_shard`, `packing=True` per `thread-e:314-326`.

| # | Config | LR | Schedule | Max steps | GBS | Seq len | Scale | qsub line | Source |
|---|--------|----|----------|----------:|----:|--------:|-------|-----------|--------|
| **C1** | **E4 repro anchor** | `2e-5` | **cosine** warmup 200 ->0 over 3ep | **93** (3ep) | 384 | 2048 | 2N `02:00` ~2 min | `qsub -v BASE_MODEL=<ckpt-900-hf>,NUM_EPOCHS=3,SFT_DATASET=gsm8k-r1cot torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` | `SYNTHESIS.md:89-96`, `thread-e:340-352` |
| **C2** | **E5 forgetting boundary** | `2e-5` | **constant** | **600** (`save_steps=50`, eval 100/300/600) | 384 | 2048 | 2N `02:00` ~5 min @0.5s/step | `qsub -v BASE_MODEL=<ckpt-900-hf>,MAX_STEPS=600,LR=2e-5,SCHED=constant torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` [INFERENCE: script flag pattern `learning_rate`/`max_steps`; PBS header `select=2 filesystems=tegu:home`] | `SYNTHESIS.md:98-105` E5, `thread-e:354-364` Config 2 |
| **C3** | **E6 lower LR** | `1e-5` | **constant** | **1000** (eval 100/300/600/1000) | 384 | 2048 | 2N `02:00` ~8 min | `qsub -v BASE_MODEL=<ckpt-900-hf>,MAX_STEPS=1000,LR=1e-5,SCHED=constant torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` | `SYNTHESIS.md:107-114` E6, `thread-e:366-376` Config 3 |
| **C4** | **E6b cosine arm** | `2e-5` | **cosine** warmup 200 ->0 over 600 | **600** (at 300 LR~1e-5, at 600 LR=0) | 384 | 2048 | 2N `02:00` ~5 min | `qsub -v BASE_MODEL=<ckpt-900-hf>,MAX_STEPS=600,LR=2e-5,SCHED=cosine torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` | `SYNTHESIS.md:107-114` E6b, `thread-e:378-388` Config 4 |
| **C5** | **Floor 5e-6** | `5e-6` | **constant** | **1000** (eval 300/600/1000) | 384 | 2048 | 2N `02:00` ~8 min | `qsub -v BASE_MODEL=<ckpt-900-hf>,MAX_STEPS=1000,LR=5e-6,SCHED=constant torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` | `thread-e:390-400` Config 5 |
| **C6** | **Broad mix** | `1e-5` | **constant** (conservative) | **1000** (eval 300/600/1000) | 768 [INFERENCE: 4N LBS2 GAS8 48ranks; 2N smoke GBS384 first per `thread-e:408`] | 8192 (OpenThoughts traces 2k-10k) | 2N smoke `02:00` ~5 min then 4N `12:00` | `qsub -v BASE_MODEL=<ckpt-900-hf>,PRETOK_DIR=/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192,MAX_STEPS=1000,LR=1e-5 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh` | `thread-e:402-412` Config 6, `SYNTHESIS.md:116-145` E7 |
| **C7** | **RL probe** | `5e-6` | GRPO | **50,100** | 384 equiv (Monarch 2N) | 512-768 `max_completion_length` | 2N Monarch `~30 min` | `qsub torchtitan/experiments/ezpz/rl/scripts/rl_grpo_b2_2n.sh` [INFERENCE: base B2 ckpt-93, `group_size=8`, `beta=0.0->0.01`, `generator.model_dtype=float32` per `thread-e:418-426`] | `thread-e:414-427` Config 7, `SYNTHESIS.md:168-174` E8 |

Notes on table:

- `save_steps=50` + `save_strategy=steps` + `logging_steps=5` fixes the B2v2 `save_strategy=epoch` single-entry bug per `thread-e:338`, `SYNTHESIS.md:116`.
- All 2N jobs: `select=2` `PPN=12` 24 ranks `workq 02:00:00` `VENV=rl-monarch-torch214-locality-20261001` LD-first `filesystems=tegu:home` per `agpt2b_b2v2_distill_cot_2n.sh:3-5,47` and contract VENV mapping.
- C6 gated on pretok `PRETOK_DIR=/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json` + `[decontam]` counts per `thread-e:410`, `sft_distill_cot_4n_verification.md:§5` [INFERENCE: pretok artifact `91k->36.9k @8192 3.4G cba5c64` per `SYNTHESIS.md:22`].
- C6 seq 8192 required for OpenThoughts long traces (2k-10k); 2048 sufficient for pure gsm8k-r1cot (~300 tok avg) per `thread-e:268-283`.

### Representative qsub — E5 (C2) full example

```bash
# E5/C2: Forgetting boundary at 2e-5 constant to 600, save 50, eval 100/300/600
# 2N 02:00:00 walltime covers ~5 min train + 3x 98s eval
qsub -l select=2 -l walltime=02:00:00 -l filesystems=tegu:home -q workq -A datascience \
  -v BASE_MODEL=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf,MAX_STEPS=600,SAVE_STEPS=50,LR=2e-5,SCHED=constant \
  torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh

# Eval each intermediate checkpoint (Sunspot 1 tile, 40 min queue, ~98s actual per 03-eval-harness.md:34)
qsub -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-e5-2e5-const/checkpoint-100,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
qsub -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-e5-2e5-const/checkpoint-300,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
qsub -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-e5-2e5-const/checkpoint-600,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
```

### Decision tree after E5/E6 per `SYNTHESIS.md:105,114`, `thread-e:478-489`

| Outcome | Interpretation | Next action | Source |
|---------|---------------|-------------|--------|
| C2 holds `>=0.15` through 600 at 2e-5 | B2v2 halving was continuation-specific, not step-count | Run broad mix C6 at 2e-5 with `max_steps 1000` | `SYNTHESIS.md:105`, `thread-e:480` |
| C2 collapses at 100-300 at 2e-5, C3 holds at 1e-5 to 1000 | **LR is lever** — 1e-5 extends budget 3x | Broad mix C6 at 1e-5, CPT-FWE general SFT at 1e-5 if >1000 steps needed | `SYNTHESIS.md:114`, `thread-e:482` |
| Both 2e-5 and 1e-5 collapse, 5e-6 holds | Need very low LR for long narrow SFT | Broad mix at 5e-6 (slow but safe) | `thread-e:484` |
| All three collapse at similar steps | Step count / repetition dominates, not LR | **Hard ceiling ~100 steps** regardless of LR — Muennighoff 4-epoch limit; lever is data breadth (C6) or base model (CPT-FWE) | `SYNTHESIS.md:114`, `thread-e:486` |
| C6 (broad mix) beats 0.15 where narrow collapsed | Distribution breadth extends budget | Promote broad distill to 4N full run | `SYNTHESIS.md:114`, `thread-e:487` |
| C6 also collapses | Even broad CoT has ceiling at 2B | Accept B2 as CoT deliverable; invest in CPT-FWE base lift | `thread-e:488` |

Cosine vs constant (C4/E6b) informs schedule choice for C6: at step 300 cosine LR~1e-5, at 600 LR=0 per `thread-e:386`; pick whichever holds longer per `SYNTHESIS.md:114`.

Total sweep walltime sequential worst case `~2+5+5+8+8+5+30 = ~63 min` on 2N each + `~2 min` eval per checkpoint; parallel with C2/C4/C7 concurrent <30 min active SFT per `thread-e:458-459`.

---

## 4. Eval protocol — 200 fixed split, metrics 4-tuple, promotion rule weighted

### 200 fixed split — the N=200 decision

**Dataset:** `openai/gsm8k` `main` `test` 1,319 problems per `eval_cot_gsm8k.py:152-154`; `--limit 200` = first-200 slice for headline per `submit_cot_eval.sh:25`, `03-eval-harness.md:20`.

**Why fixed 200:** N=200 gives `SE = sqrt(p(1-p)/N)` and `95% half-width = +/-1.96*SE ~ +/-2*SE` sufficient for shipping decisions (B2 vs every alternative gap `>=0.10` detectable at `>2sigma` on 200) per `03-eval-harness.md:116-139` §2.3. Larger N matters only for near-tied variants `Δ=0.03-0.05` — add second benchmark (MATH) rather than 500/1000 on same task per `03-eval-harness.md:260`.

**Power context:** MMLU at fixed band `0.2295-0.2689` (always-D `0.2689` beats all agpt-2B) — MMLU not CoT; subsampling MMLU drove `0.028` noise (limit-10 `0.2281` vs limit-40 `0.2561` on 30B step-2000 job `12474382`) per `03-eval-harness.md:88-94`.

### Metrics 4-tuple — every eval reports all four

Scoring per `eval_cot_gsm8k.py:30-84,118-131,186-213`, `03-eval-harness.md:22-23`:

| Metric | Definition | B2 reference | Source |
|--------|-----------|-------------|--------|
| `cot_accuracy` | Exact normalized match vs `#### N` gold; inside `<answer>` span prefers `\boxed{...}` else first number; `_norm_num` drops commas/$/`5.0->5`; strict fallback NO last-number | **0.205** (41/200, 95% CI 0.149-0.260) | `eval_cot_gsm8k.py:30-84`, `03-eval-harness.md:33` |
| `format_hit_rate` | `_FORMAT_RE = r"<think>.*?</think>\s*<answer>(.*?)</answer>"` DOTALL+IGNORECASE | **0.985** | `eval_cot_gsm8k.py:33-35`, `03-eval-harness.md:23` |
| `mean_gen_len` | Mean token length of generation per `summarize()` | **282** | `eval_cot_gsm8k.py:123`, `03-eval-harness.md:33` |
| `n_unclosed` | `finish_reason=="length"` count (max_tokens truncated) | **~0** | `eval_cot_gsm8k.py:122`, `03-eval-harness.md:33` |

Failure modes distinguished by 4-tuple per `03-eval-harness.md:43-46`:

- **Envelope collapse:** `format 0.26` `133/200` unclosed (B4a) — different failure than reasoning per `03-eval-harness.md:37`, `SYNTHESIS.md:238`.
- **Verbosity blow-up:** B2v2 `282->416` + `0->22` unclosed before format fully collapsed (`0.86`) per `03-eval-harness.md:34`.
- **Healthy loss hiding halved eval:** B2v2 loss `0.8069` acc `0.8198` while CoT `0.205->0.09/0.105` per `03-eval-harness.md:183-186`.

Raw CPT scores `0/200` `0.0 fmt` degenerate loops `n_unclosed=200` **expected pre-SFT** (no `<think>/<answer>\boxed{}` contract) per `03-eval-harness.md:30-32,192-197` §3.1 — not a failure.

### Generation stack — identical for every eval

Per `submit_cot_eval.sh:1-34`, `eval_cot_gsm8k.py:161-175`, `03-eval-harness.md:12-24`:

- **PBS:** `#PBS -l select=1` `walltime=00:40:00` `filesystems=tegu:home` `q=workq` `A=datascience` per `submit_cot_eval.sh:2-6`
- **Module/VENV:** `frameworks/2026.1.0` + `pti-gpu`, `LD_LIBRARY_PATH` PTI `/pti-gpu-0.17.0-62kxpbr/lib64`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=0` per `submit_cot_eval.sh:15-20`
- **Python:** `/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python` per `submit_cot_eval.sh:28`
- **Eval script:** `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` per `submit_cot_eval.sh:29-30`
- **Invocation:** `qsub -v CKPT=<hf-dir>,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh` per `submit_cot_eval.sh:10`
- **Env overrides:** `CKPT/LIMIT=200/DTYPE=float32/OUT=.../cot/gsm8k-cot-baseline.jsonl/PY/EVAL_SCRIPT` per `submit_cot_eval.sh:24-30`
- **Prompt:** Hand-serialized Gemma turns `<start_of_turn>user\n{question}{_PROMPT_SUFFIX}<end_of_turn>\n<start_of_turn>model\n` where `_PROMPT_SUFFIX = "\nReason step by step inside <think></think>, then give the final answer inside <answer>\boxed{}</answer>."` per `eval_cot_gsm8k.py:86-115`
- **Generation:** `vLLM LLM(dtype=float32, gpu_memory_utilization=0.70, enforce_eager=True, max_model_len=2048)` `SamplingParams(temperature=0.0, max_tokens=768, stop_token_ids=[1,107])` per `eval_cot_gsm8k.py:161-175` — textual `</answer>` NOT a stop
- **Output:** stdout JSON summary `{"n":200, "format_hit_rate":..., "cot_accuracy":..., "mean_gen_len":..., "n_unclosed":..., "model/dtype/temperature/max_tokens":...}` + per-example JSONL at `args.out` per `eval_cot_gsm8k.py:215-232`

### SE table — what 200 vs 500 vs 1000 buys

Per `03-eval-harness.md:118-123` (exact table from `thread-c-eval-harness.md:300-304`):

| N | SE at p=0.05 | SE at p=0.10 | SE at p=0.20 | 95% half-width at 0.20 | Source |
|---:|-------------:|-------------:|-------------:|-----------------------:|--------|
| **200** | 0.015 | 0.021 | **0.028** | **+/-0.055** (0.145-0.255) | `03-eval-harness.md:118-123` |
| **500** | 0.010 | 0.013 | **0.018** | **+/-0.035** (0.165-0.235) | same |
| **1000** | 0.007 | 0.010 | **0.013** | **+/-0.025** (0.175-0.225) | same |

Comparison delta `SE_delta = sqrt(SE_A^2 + SE_B^2)`; significant at ~2sigma if `Δ > 2*SE_delta` per `03-eval-harness.md:126-134`:

| Comparison | N=200 per arm | N=500 | N=1000 | Source |
|-----------|:---:|:---:|:---:|--------|
| B2 0.205 vs B2v2-ep6 0.105 (Δ=0.10) | 2.5σ **detectable** | 4.0σ | 5.5σ | `03-eval-harness.md:130` |
| B2 0.205 vs B3 0.05 (Δ=0.155) | 4.1σ **detectable** at 200 | 6+σ | 8+σ | same |
| B4b 0.065 vs B4a 0.02 (Δ=0.045) | 1.5σ borderline at 200 | 2.3σ | 3.3σ | same |
| B2 0.205 vs 0.215 (+0.01 RL gain) | 0.26σ **no** | 0.41σ **no** | 0.56σ **no** | same |
| Any two separated by <0.04 | **no** at 200 | **no** at 500 | borderline | same |

Consequence: 200 sufficient for gate/shipping; 500/1000 only matters for near-tied variants `Δ=0.03-0.05` where second benchmark has higher information per walltime per `03-eval-harness.md:138-139`.

Caveat: GSM8K item difficulty correlated (not i.i.d. Bernoulli) — SE optimistic per `03-eval-harness.md:136`; subsample spread `0.028` is lived warning.

### Promotion rule — weighted, not accuracy-only (G1-G4)

Gate is conjunction — **all must hold to continue to 190k** per `SYNTHESIS.md:225-239` §5, `03-eval-harness.md:302-311`:

| Gate | Metric | Base | Threshold to **CONTINUE** to 190k | SE / rationale | Source |
|------|--------|------|-----------------------------------|----------------|--------|
| **G1** — No commonsense regression | HellaSwag `acc_norm` 0-shot | 0.560 (2B-256 plateau) | **>= 0.540** (-2pp) | SE ~0.004; -2pp = 5σ; dolmino-100 fell 0.486 | `SYNTHESIS.md:231`, `03-eval-harness.md:304` |
| **G1b** | ARC-Easy `acc` 0-shot | 0.651 | **>= 0.621** (-3pp) | SE ~0.006; dolmino 50/50 held 0.610 | `SYNTHESIS.md:232`, `03-eval-harness.md:305` |
| **G1c** | PIQA `acc_norm` 0-shot | 0.733 | **>= 0.713** (-2pp) | SE ~0.005; dolmino -4.6pp weak guard | `SYNTHESIS.md:233`, `03-eval-harness.md:306` |
| **G2** — No catastrophic forgetting | ARC-Challenge `acc_norm` 25-shot | ~0.31 | **>= 0.290** (-2pp) | peer table in `evals/` | `SYNTHESIS.md:234`, `03-eval-harness.md:307` |
| **G3** — MMLU must not regress | MMLU `max(CF,MCF)` 5-shot | 0.241-0.266 chance band | **>= 0.240** (no regression) | Band 0.2295-0.2689 (best constant-D), SE 0.0036. Clearing band requires prior collapse `R2->0` | `SYNTHESIS.md:235`, `03-eval-harness.md:308` |
| **G4** — SFT probe lifts ceiling | GSM8K-CoT `acc` @200 post-SFT (B2 recipe) | B2 = **0.205** (41/200, 95% CI 0.149-0.260) | **>= 0.265** (B2 +0.06 at 2σ; 0.06 gap ~2.0σ on 200) | SE ~0.028 at 0.20. RL move 0.205->0.215 is 0.26σ — noise | `SYNTHESIS.md:236`, `03-eval-harness.md:309` |
| **G4b** — Format held | GSM8K-CoT `format_hit_rate` | B2 = 0.985 | **>= 0.90** | B2v2 fell 0.86/0.91 while degrading; B3 0.86 masking 0.05 | `SYNTHESIS.md:237`, `03-eval-harness.md:310` |
| **G4c** — No verbosity blow-up | `mean_gen_len` / `n_unclosed` | B2 = 282 / ~0 | **<= 550** / **<= 10** | B2v2 416/378 with 22/15 unclosed; B4a 1378/133 flagged | `SYNTHESIS.md:238`, `03-eval-harness.md:311` |

**SFT-level promotion** (post-E3 probe): promote new base iff `GSM8K-CoT >=0.265` at 2σ + `format >=0.90` + `unclosed <=10` vs B2 `0.205` per `SYNTHESIS.md:252`.

**Envelope collapse guard:** if `format <0.80` or `unclosed >30`, **stop — do not interpret acc as reasoning** per `03-eval-harness.md:268-271`. Treat `gen_len > ~1.5x` B2 baseline (282) as drift flag even when `format >0.80` per `thread-c-eval-harness.md:280-281`. Do NOT down-weight format to fix — RL hacking `0.2->0.05` drove `95%` no `<think>`, `20%` echoed `\boxed{5}` per `03-eval-harness.md:265-268`.

**Near-miss interpretation** per `SYNTHESIS.md:240-249`, `03-eval-harness.md:313-317`:

- G1-G2 fail by >2pp but G4 passes: **continue to 190k** — expected FWE HS trade (FWE 37.5 vs DCLM 35.5 MMLU, DCLM 62.3 vs 60.1 HS); re-check at 190k; monotonic bleed = ratio wrong -> `config_registry.py:1024-1043`.
- G1-G2 pass but G4 fails (`0.08-0.12` like B2v2/B3/B4): **stop CPT**, fork SFT from best intermediate (max over G1-G4), pivot to broad-distill on MDS base (E7).
- Everything fails: **stop CPT, revert to MDS base** — FWE kind-shift wrong or LR 2e-6 needs re-tuning.
- G1-G3 pass but SFT probe not yet run: **continue CPT** — do not stop without ceiling test; CPT-direct MMLU/ARC are thermometers, SFT probe is thermostat.

**Provenance contract** for every artifact per `03-eval-harness.md:358-361`: persist `n_shot + task_key + format (MCF/CF/max) + band 0.2295-0.2689 + letter-prior p(A,B,C,D) + residual dispersion` in `results.json`; CoT JSONL header `model/dtype/temperature/max_tokens/finish_reason` + 95% CI — byte-comparable re-scores. Do NOT merge old 2B-v2 artifacts lacking `n-shot`/`@5shot` keys — retro-rescore flagship `step-46429` + `step-92859` in both MMLU formats as first MMLU job per `03-eval-harness.md:361`.

---

## 5. Submit order — E4 first, then E2 in parallel, E5/E6 after E4 passes

### DAG — dependencies and parallelism

```
E1 HF converts (P0-P2, 11 HFs) ───────────────────── lives on Sunspot 1N 02:00 each
        │                                                    (already: 74000/74200 3.7G 9/9)
        │ E1 already done or in-flight; does not block E4
        │
        ├─► E4 repro anchor (2N, 93 steps, 2 min) ──► PASS (0.18-0.23) ──┬─► E5 (C2) 2e-5 const 600 ──► E6 (C3) 1e-5 1000
        │         │ halt if FAIL                                    │         └─► E6b (C4) cosine 600 (parallel with E5)
        │         │ (load-bearing gate)                             │
        │         │                                                 └─► C5 5e-6 floor 1000 (if E5+E6 collapse)
        │         │                                                     C6 broad 1e-5 4N (if winning LR known)
        │         │                                                     C7 RL probe 50/100 (independent, parallel with E5)
        │         │
        └─► E2 commonsense thermometer ─────────────────────────── can start once E1 HF ready
                │  (1N cheap, does not need E4)
                │  Sunspot HS/ARC/PIQA/ARC-C per HF — ~1h per batch
                │
                └─► E2 MMLU MCF+CF (Aurora 48h) ─────────────────── Aurora-only, queued in parallel
                        │  max(CF,MCF) + prior fit per HF
                        │
                        └─► E3 SFT probe on best CPT HF (E2 winner) ──► G4 gate 0.265
                                    │  (gated on E2 G1-G3 + E4 pass)
                                    │
                                    └─► 150k/190k gates G1-G4 ──► continue vs stop vs fork
```

Sources: `SYNTHESIS.md:89-96` E4 dep (nothing — run immediately), `SYNTHESIS.md:98-114` E5 (depends on E4 pass) + E6 (depends on E5 result), `SYNTHESIS.md:68-77` E2 (depends on E1 HF), `thread-e:430-456` cheapest validation order, `SYNTHESIS.md:194` walltime summary.

**Ordering rules per `thread-e:456` parallelization:**

- **E4 must run first** — validates stack. If outside `0.18-0.23`, root-cause before any sweep per `SYNTHESIS.md:96`.
- **E2 can start as soon as E1 HFs exist** — does not need E4; runs in parallel with E4 and with live CPT 4N chain per `SYNTHESIS.md:194`.
- **E5/E6 halt on E4 FAIL** — E5 depends on E4 pass per `SYNTHESIS.md:103`.
- After E4 pass, **E5 (C2), E6b (C4), and RL probe (C7) are independent** — run in parallel on 2N each (6N total, within `workq`) per `thread-e:456`.
- **E6 (C3) depends on E5 result** (which LR to test next) — follows sequentially per `thread-e:456`.
- **C6 broad mix depends on winning LR** from C2-C5 + pretok gate (`/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json` + `[decontam]` counts) — start 2N smoke as soon as LR known, promote to 4N on positive signal per `thread-e:456`.

### Walltime estimates — per-job and batched

| Work | Scale | Train | Eval (per ckpt) | Total per work | Source |
|------|-------|-------|-----------------|----------------|--------|
| **E4** repro (93 steps) | 2N 24 ranks | **~2 min** @0.5s/step | **~98s** (`00:01:38` Exit 0 at LIMIT=200) | **~4 min** end-to-end | `SYNTHESIS.md:95` ("~2 min train, ~98s eval"), `03-eval-harness.md:34` jobs `12482572/73` |
| **E5** C2 (600 steps) | 2N 24 ranks | **~5 min** (600*0.5s) | **~2 min x3** (ckpt-100/300/600) | **~11 min** | `thread-e:363` (600 steps ~5 min well within 2h) |
| **E6** C3 (1000 steps) | 2N 24 ranks | **~8 min** (1000*0.5s) | **~2 min x4** (100/300/600/1000) | **~16 min** | `thread-e:375` (~8 min at 0.5s/step) |
| **E6b** C4 (600 cosine) | 2N 24 ranks | **~5 min** | **~2 min x3** | **~11 min** | `thread-e:387` |
| **C5** floor 5e-6 (1000) | 2N | ~8 min | ~2 min x3 | ~14 min | `thread-e:398-400` |
| **C6** broad smoke -> full | 2N -> 4N 48 ranks | ~5 min smoke -> ~1h 4N (6.3M tok/step @4N) | ~2 min per ckpt | **~1-1.5h** full | `SYNTHESIS.md:194` ("E1 4N 1h; E2 4N 1.5h") |
| **C7** RL probe | 2N Monarch | **~30 min** for 50+100 steps | ~2 min per point | **~34 min** | `SYNTHESIS.md:174`, `thread-e:425` |
| **E2** commonsense per HF | 1N Sunspot/Aurora | — | **~2 min/eval** (0-shot loglik) x 4 benchmarks x 11 HFs = **~88 min** if serial | **batched <30 min** when fanned per `SYNTHESIS.md:194` | `03-eval-harness.md:372`, `SYNTHESIS.md:194` |
| **E2** MMLU per HF | Aurora 1N | — | **48h `workq`** per full MMLU pass (57 tasks ~56k requests) — **walltime-dominated** | **48h** per HF pass (queue, not compute) | `03-eval-harness.md:241,369`, `SYNTHESIS.md:74` |
| **HF converts P0-P2** | 1N Sunspot | **02:00:00** `workq` each | 9/9 verify | **8 converts** = 8x2h queue, <1h compute | `SYNTHESIS.md:61-64` |

**Sequential worst case** (all 7 configs + E2): `~63 min` SFT on 2N + `~2 min` eval per checkpoint per `thread-e:458-459`. Under 2h even sequentially.

**Parallelized reality per `thread-e:459`, `SYNTHESIS.md:194`:**

- **Time 0:** E4 (2N) + E1 converts (1N) + E2 commonsense on already-ready HFs (1N) all launch in parallel — no dependency conflict.
- **E4 pass at ~4 min:** fan out E5 (C2) + E6b (C4) + C7 RL probe in parallel on 3x2N = 6N (`workq` allows).
- **E5 result at ~15 min:** branch to E6 (C3) or C5 or C6 smoke depending on outcome — sequential tail <20 min.
- **E2 MMLU Aurora 48h** overlaps entire sweep — queued at time 0, results informational first, gating second.

**Full sweep walltime when fanned:** **<30 min active SFT** + **MMLU walltime 48h (background)** + **HF convert queue overlap** per `SYNTHESIS.md:194` ("E1 1N 2h and E4 2N 2m start immediately, in parallel. E2 Aurora 48h or Sunspot commonsense 1N overlaps with E5 2N 5m and E8 2N 30m. Sequential worst case <2h of 2N time plus MMLU walltime; parallelized <30 min active SFT — all on workq, none blocks 4N CPT chain").

### PBS headers — identical across sweep

All Sunspot SFT sweep jobs:

```bash
#PBS -A datascience
#PBS -l walltime=02:00:00
#PBS -l select=2          # 2N = 24 ranks for GBS384; select=5 for C6 4N GBS768
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
```

All Sunspot CoT eval jobs:

```bash
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
```

All Aurora MMLU jobs:

```bash
#PBS -A datascience
#PBS -l walltime=48:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
```

Sources: `submit_cot_eval.sh:1-7`, `agpt2b_b2v2_distill_cot_2n.sh:2-6`, `03-eval-harness.md:17,369`.

### VENV and routing — every job

| Workload | VENV / Module | LD | Device | Source |
|----------|---------------|----|--------|--------|
| **SFT / RL** (E4-E6, C5-C7) | `rl-monarch-torch214-locality-20261001` + LD-first `.cache/xpu-ld-first/` | `LD_LIBRARY_PATH=$VENV/lib:$LD` + PTI/CCL/compiler/MKL via `.cache/xpu-ld-first` symlink farm | `ZE_FLAT FLAT` `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | contract VENV mapping, `agpt2b_b2v2_distill_cot_2n.sh:47,60` |
| **CoT eval** (E4 final + E5/E6 checkpoints) | `frameworks/2026.1.0` module + `pti-gpu` | `LD_LIBRARY_PATH=/opt/aurora/.../pti-gpu-0.17.0-62kxpbr/lib64` | `ZE_FLAT FLAT` `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` `ZE_AFFINITY_MASK=0` | `submit_cot_eval.sh:15-20` |
| **E2 commonsense** (HS/ARC/PIQA) | same `frameworks/2026.1.0` | same MKL+CCL+compiler+PTI LD | `xpu:0` loglik backend | `03-eval-harness.md:18,368` |
| **E2 MMLU** | same `frameworks/2026.1.0` on **Aurora** | same | `xpu:0` | `03-eval-harness.md:369` |
| **E1 HF converts** | `xpu-torch216-sync-5a7a622f16` + LD-first | `.cache/xpu-ld-first/` | `flavor=2b-mds` Tegu output | `SYNTHESIS.md:61` |

`PBS filesystems=tegu:home` for any job reading `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` (15T DCP) and writing Tegu HF/evals per contract, `03-eval-harness.md:372`.

### Cost to decide

Per `03-eval-harness.md:373`, `SYNTHESIS.md:32-33`:

- **~8 HF converts** (1N 2h each, Tegu `3.7G 9/9` each, ~52G full P0-P2 ladder — trivial vs 15T DCP / 23G frontier) + **1 E2 eval sweep** (commonsense + MMLU) + **3 SFT probes** (E4 + E5 + E6x) — all 1N-2N `workq`, **zero new CPT compute**, parallel with live 4N chain advancing on 12h cadence (`qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 cpt_fineweb_edu_4n.sh` auto-resumes from `CKPT_DIR` per `SYNTHESIS.md:178`).

---

### Cross-refs

- **Stack + scoring + power + gates:** `torchtitan/experiments/ezpz/docs/exploration/thread-c-eval-harness.md` — 7 scoring rules §1.5, power table §2.3, 4-tuple diagnosis §2.2, gating protocol §5.2
- **MMLU letter-prior audit:** `torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-at-chance.md` — band 0.2295-0.2689 p.27-49, Llama 0.312/0.653 p.128-131, CF-MCF 0.2281/0.2386 p.272-276
- **SFT deliverables + B2/B2v2/B3/B4/RL + CPT infra + VENV map:** `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` — CoT table p.73-82 (jobs 12482572/73 00:01:38/39 Exit 0), walltime 99e918b342
- **Decision gates G1-G4 + E2 sweep + thresholds + walltime:** `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md` §§3-5 — E2 sweep p.68-77, E4-E6 p.89-114, G1-G4 table p.229-238
- **LR x steps budget + parallelism + RL + sweep configs 1-7:** `torchtitan/experiments/ezpz/docs/exploration/thread-e-recipe-optimization.md` — §1 LR analysis, §2 GBS 384/6144, §5.2 7-config sweep p.336-427
- **Prior ledger index:** `torchtitan/experiments/ezpz/docs/run-ledger/00-index.md` + `03-eval-harness.md` (262 lines, harness stack + job table + SE + routing)
- **Eval scripts:** `torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh:1-34` (PBS, VENV, qsub contract), `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py:1-236` (dataset/prompting/generation/scoring), `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh:1-119` (SFT PBS + VENV + train flags)
