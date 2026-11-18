# 08 — SFT broad distill unpark (E7 Thread B §§4-5, SYNTHESIS E7, POST-TRAINING-PLAN phases 1-3)

> **Branch:** `ezpz-cpt111k-eval-sft` HEAD `a6135a79` (2026-11-16).
> **Scope:** Submit-ready launch spec that turns parked broad-distill (Thread B §§4-5, SYNTHESIS E7, `sft_distill_cot_4n_verification.md` gates) into executable qsub commands with PBS headers, VENVs, CKPT/LOG dirs, and Go/No-go gates. This doc does not re-derive the B-series diagnosis — it wires the landed pretok artifact to the training scripts.
> **Every claim cites `script:line`, `doc:section`, or PBS `job ID`;** [INFERENCE] marks estimates.

**Parked-to-unpark map:** `04-sft-broad-distill.md:9-10` (pretok landed, consumer gated/parked to 50k) + `sft_distill_cot_4n_verification.md:5-6` (gate check only, no training) + `SYNTHESIS.md:116-165` (E7 A0-E7 queue) are the three prior parked anchors. This ledger is the unpark: same artifact, same scripts, now with smoke-first submit lines and PBS selects.

---

## 1. Pretok gate — landed 12483443 (91k→36.9k @8192 3.4G cba5c64)

**What landed:** Job `12483443` (`Exit`) on `_pretokenize_distill_cot_mix_1n.sh` — the dedicated 1N pre-tokenize that materializes `distill_cot_mix` once and writes a TRL-consumable `save_to_disk` arrow set (`POST-TRAINING-2B.md:36-37`, `04-sft-broad-distill.md:9-27`, `SYNTHESIS.md:116-118`). The prior 6 attempts segfaulted walking `NPROC 32→16→8→4→1` (`SIGBUS`/`3553055 Segmentation fault` on decontam/packing `map` subprocesses) until `NPROC=1` + VENV-path fix (`_pretokenize_distill_cot_mix_1n.sh:103-106`, `sft_distill_cot_4n_verification.md:11-13`).

| Property | Value | Source |
|---|---|---|
| **Job** | `12483443` `Exit` (final), 6 prior segfaults | `04-sft-broad-distill.md:15` `POST-TRAINING-2B.md:37` `thread-b-broad-distill.md:95` |
| **Script** | `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh` (`PBS -l select=1 -l walltime=12:00:00 -l filesystems=tegu:home -q workq`, single-rank `NPROC=1` direct `${VENV}/bin/python -m torchtitan.experiments.ezpz.rl.train_sft --pretokenize_to`, no `ezpz_setup_job` MPI wrapping) | `_pretokenize_distill_cot_mix_1n.sh:1-8,77-83,103-123` |
| **VENV** | `rl-monarch-torch214-locality-20261001` (`trl 1.13` probe at `:101`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, LD-first cache `.cache/xpu-ld-first/`) — the only VENV with `trl` (`xpu-torch214` lacks it) | `_pretokenize_distill_cot_mix_1n.sh:89-99` `sft_distill_cot_4n.sh:99-112` `sft_distill_cot_4n_verification.md:48-52` |
| **Output** | `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (`dataset_info.json` + 8 x `data-*.arrow`, `3.4G`) | `_pretokenize_distill_cot_mix_1n.sh:108-109,149` `sft_distill_cot_4n.sh:152` `POST-TRAINING-2B.md:37` |
| **Rows** | **91,481 raw rows** (after envelope formatting + per-component GSM8K-test decontam, before packing) → **36,934 packed sequences @ `max_length 8192`** (`assistant_only_loss` packing, `max_length 8192`) | `POST-TRAINING-2B.md:37` `04-sft-broad-distill.md:21` `thread-b-broad-distill.md:98-99` `SYNTHESIS.md:118` `sft_distill_cot_4n.sh:53-54` |
| **Mix** | `distill_cot_mix` = **0.50 OpenThoughts-114k-decontam + 0.50 OpenR1-Math-220k-decontam** (`all_exhausted` interleave, seed 42) — `datasets_sft.py:1262` via `sft_distill_cot_4n_verification.md:18-27`; `_pretokenize_distill_cot_mix_1n.sh:21-22` `sft_distill_cot_4n.sh:19-24` header comment | `datasets_sft.py:1262` `sft_distill_cot_4n_verification.md:18-23` `SYNTHESIS.md:116-118` |
| **Mix-cache** | Hit **`cba5c64`** (materialized-mix cache under `/tegu/datasets/datasets/ezpz_sft_mixes/cba5c64`, recipe hash captures decontam + `DISTILL_MAX_THINK_CHARS`) | `POST-TRAINING-2B.md:37` `thread-b-broad-distill.md:95` `SYNTHESIS.md:118` |
| **Shared staging** | `HF_HOME=/tegu/datasets/datasets/hf` + `EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes` set identically in producer and consumer | `_pretokenize_distill_cot_mix_1n.sh:66-71` `sft_distill_cot_4n.sh:108-110` `sft_distill_cot_4n_verification.md:14-15` |

### Verify on Sunspot login before any training qsub

Per `sft_distill_cot_4n_verification.md:72-91` (§5 Gate) and `sft_distill_cot_4n.sh:156-167` (`_wait_for_pretok()` gating on `dataset_info.json` + fallback `outputs/sft-pretok/distill-cot-mix-len8192/dataset_info.json`), **all three** must pass:

```bash
# 1. dataset_info.json exists and is non-empty (primary + fallback the script checks)
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
ls -lh outputs/sft-pretok/distill-cot-mix-len8192/dataset_info.json 2>&1 | head

# 2. Size matches landed artifact (3.4G, 8 shards)
du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/data-*.arrow 2>&1 | wc -l
# Expected: 3.4G and 8 arrow files

# 3. Decontam counts are non-trivial (confirms 13-gram + normalized-exact ran, not stale cache)
grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log | tail -20
# Expected: per-component drop lines from decontam_traces.GSM8KDecontaminator (13-gram + normalized-exact)
# If counts are zero/near-zero on 114k+220k corpora with GSM8K overlap, flag as suspicious — see §5.

# 4. Recipe hash tag in producer log (proves cache hit was decontaminated build)
grep -F "cba5c64" logs/pretokenize-distill-cot-mix-*/run.log 2>&1 | head
```

If missing, rebuild stage 1 only (1N, proxy ON, shared `/tegu`, VENV `locality-20261001`, `NPROC=1`):

```bash
qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh
# Poll the same 4 checks above; do NOT re-trigger HF download from the 4N consumer
# (sft_distill_cot_4n.sh:150-164 _wait_for_pretok() sleep-loops rather than rebuilding).
```

**BASE_MODEL gate (also required before stage 2):**

```bash
# A — primary base for E1/E2: checkpoint-900-hf (IFEval 0.253, pre-collapse, only base with >0.10 CoT trajectory)
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900{,-hf}/ 2>&1 | head
# If -hf missing but sharded checkpoint-900/ exists (96 .distcp shards), re-consolidate:
#   accelerate merge-weights + copy config.json/tokenizer.* per docs/production/sft/README.md
#   Consumer fallback chain: checkpoint-900-hf -> checkpoint-900/ -> legacy agpt2b-gsm8k-r1cot-2n (sft_distill_cot_4n.sh:120-146, sft_distill_cot_4n_verification.md:32-40)

# B — CPT-FWE frontier (gated, not yet deliverable — see §4 CPT-FWE follow-on)
ls -lh /lus/tegu/projects/datascience/foremans/projects/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/ 2>&1 | head

# C — raw pretrain (only for E5 control)
ls -lh ~/global_step138650/config.json ~/global_step138650/tokenizer.json 2>&1
```

Consumer defaults to `BASE_MODEL=$TEGU_SFT_ROOT/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf` (`sft_distill_cot_4n.sh:120-121`); override via `BASE_MODEL=... qsub ...`. Producer uses `BASE_MODEL="${HOME}/global_step138650"` for tokenizer only (`_pretokenize_distill_cot_mix_1n.sh:107`, `sft_distill_cot_4n_verification.md:41-45` — no mismatch, vendored Gemma tokenizer identical).

---

## 2. A0-E6 queue — cheap ablations first (A0 audit 0 GPU 10min, A1 base probe 1 tile 2min, A3 decontam n=8 login, A4 len 4096 vs 8192) then main E1-E7

**Order:** Cheap ablations `A0/A1/A3/A4` run first (all <2N, under 30 min wall) before any 4N training — they gate whether `E1` should launch at all and which hparams it should use (`thread-b-broad-distill.md:285-397`, `SYNTHESIS.md:120-130`). `A2` (component-only mix) is conditional on `E1` ambiguity (see queue table note). Common hparams per `SYNTHESIS.md:133-134`: `VENV=rl-monarch-torch214-locality-20261001` LD-first, `assistant_only_loss=True`, `packing=True`, `bf16 fsdp full_shard dataset_num_proc 8 logging_steps 5 save_strategy epoch HF_DATASETS_OFFLINE=1`. All evals: 200-problem GSM8K-CoT `temperature 0.0 max_tokens 768` `frameworks/2026.1.0 vLLM 0.26 XPU fp32` `workq`, reporting `cot_accuracy / format_hit_rate / mean_gen_len / n_unclosed` vs **B2 `0.205 / 0.985 / 282 / ~0`** (`POST-TRAINING-2B.md:45`, `thread-b-broad-distill.md:38`, `SYNTHESIS.md:17`).

### Cheap ablations — run in parallel, 0-1 tile, block or inform E1

| ID | Question (SYNTHESIS E7) | Setup (scale / script) | PBS select | Gate on E1 | Source |
|---|---|---|---|---|---|
| **A0 — Artifact audit** | Is the landed 36.9k packed dataset well-formed? | **0 GPUs, 10 min, login node.** `load_from_disk("/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192")` → verify rows ~91k raw / ~36.9k packed @8192, `input_ids`+`labels` present, `seq_lengths` p50/p90, decode 3 samples for `<think>/<answer>\boxed{}` envelope, `grep [decontam]` non-zero | **select=1 login** (or Sunspot login shell; no qsub needed) | **Blocks all if failed** — wrong artifact (missing `input_ids`, zero decontam, broken envelope) → re-stage pretok before any training. If p50 >> 4096 then 8192 justified; if p50 ~1024 then half-empty. | `SYNTHESIS.md:124` `thread-b-broad-distill.md:289-327` |
| **A1 — Base-model probe** | How does `checkpoint-900-hf` score on CoT before stage-2? | **1 XPU tile, ~2 min.** `eval_cot_gsm8k.py` via `submit_cot_eval.sh` on `checkpoint-900-hf` itself (no SFT) | **select=1 `qsub -l select=1 -l walltime=00:20:00 -q workq` eval tile** | [INFERENCE] Expect ~0.0 acc (never saw `<think>/\boxed{}`) — stage-2 must teach format+reasoning; if 0.05-0.10 with fmt 0.3-0.5 then stage-1 already has CoT-adjacent behavior (warmer start, informs E1 delta). | `SYNTHESIS.md:125` `thread-b-broad-distill.md:330-341` |
| **A3 — Decontam sensitivity** | How many GSM8K-test near-duplicates survive 13-gram? | **0 GPUs, 30 min login node.** Re-run `decontam_traces.py` at `n=8` (more sensitive) on landed artifact vs production `n=13`; compare drop counts | **select=1 login** (CPU-only `decontam_traces --n 8 --report-only`) | Delta <0.5% negligible; 0.5-2% note in report; >2% re-stage at `n=8` before gain claim (see §5 prohibition 13). | `SYNTHESIS.md:127` `thread-b-broad-distill.md:360-381` `SYNTHESIS.md:285` |
| **A4 — Length 4096 vs 8192** | Does 4096 (half attention cost) hurt vs 8192? | **1N pretok `NPROC=4` ~1-2h + 2N SFT ~1h.** Stage 4096 pretok of `distill_cot_mix` → `AGPT2B_DISTILL_COT_MIX_len4096` → SFT at `max_length 4096 LBS4 GAS4 GBS384` (tokens/step matched to 8192@GBS768 by halving seq_len, doubling micro-batch) | **Pretok `select=1` + SFT `select=3 NHOSTS_TRAIN=2 GBS384 workq 01:00:00`** (or `select=2` if PPN=12 → 24 GPUs) | **Only run if E1 `gen_len > 500`** (verbosity-sensitive). Go: within 0.02 acc, fmt ≥0.85 → 4096 is efficient default (64x less attention `O(L^2)` without flash-attn). No-go: fmt <0.70 or acc -0.05 vs E1 → keep 8192. | `SYNTHESIS.md:128` `thread-b-broad-distill.md:383-395` |

Parallel block: `A0 + A1 + A3` finish in <30 min with no allocation contention (`SYNTHESIS.md:150-153`).

### Main queue — ranked E1-E7 (SYNTHESIS E7, Thread B §5)

Ordered by information value per GPU-hour; early exits save downstream allocation. `A2` (component-only) is conditional — run only if E1 ambiguous (0.15 = which half did the work).

| # | Experiment | Priority | Base | Mix | Hparams | Scale (PBS) | Gate (Go / No-go) | Source |
|---|---|---|---|---|---|---|---|
| **E1** | **Broad-distill stage-2 on checkpoint-900, 1ep — THE GATE** | **P0 run first** | `checkpoint-900-hf` (re-consolidate from sharded `checkpoint-900/` if `-hf` missing; verify `config.json eos [1,107]` + `tokenizer.json` + `model.safetensors`) — `sft_distill_cot_4n.sh:120-146` | `distill_cot_mix` = OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam, pre-tokenized `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (36.9k packed @8192, 91k raw, 3.4G, `cba5c64`) — `sft_distill_cot_4n.sh:152-153` | LR **2e-5 cosine**, **1ep ~48 steps** @GBS768, `LBS2 GAS8 max_length 8192 bf16 fsdp full_shard dataset_num_proc 8 logging_steps 5 save_strategy epoch` — `sft_distill_cot_4n.sh:236-275` | **4N `select=5 NHOSTS_TRAIN=4 PPN=12 GBS768 workq 12h` actual ~1h; script `sft_distill_cot_4n.sh` with `NUM_EPOCHS=1`** | **Go:** acc ≥0.15 fmt ≥0.80 gen_len <600 → E2. **Pivot:** 0.05-0.15 good fmt → E2b or MATH/science eval (broad gain invisible on GSM8K). **No-go:** <0.05 or fmt <0.50 → stop, run A2. | `SYNTHESIS.md:138` `thread-b-broad-distill.md:398-420` |
| **E2** | **Same 3ep (B2 duration)** | **P1 if E1 Go/Pivot** | `checkpoint-900-hf` | `distill_cot_mix` same | LR 2e-5 cosine **3ep ~144 steps** otherwise E1; `save_strategy epoch` per-epoch eval | **4N `select=5 NHOSTS_TRAIN=4 GBS768 12h` actual ~1.5h** | **Go:** ep3 > ep1 and ≥0.18 → queue E3 or E6. **Flat:** ep3 ~ ep1 ±0.02 → cap at 1ep. **No-go:** ep3 < ep1 or fmt degrades → broad also collapses; cap future at 1ep. | `SYNTHESIS.md:139` `thread-b-broad-distill.md:421-430` |
| **E2b** | **Stage-3 narrow polish on E1/E2 best-epoch HF** | **P1 if E1 Pivot or E2 Flat** | **E1 or E2 best-epoch HF** (not `checkpoint-900`) | `gsm8k-r1cot` 7.4k ex `<think>/\boxed{}` inline tokenize (no pretok) | LR 2e-5 cosine (or 1e-5 if format-sensitive) **3ep 93 steps @GBS384 `max_length 2048`** match B2 proven seq_len; `dataset_num_proc 8` | **2N `select=3 NHOSTS_TRAIN=2 GBS384 workq 01:00:00` ~30m; pattern `agpt2b_b2v2_distill_cot_2n.sh:90-114`** | **Go:** >E1/E2 best and ≥0.15 fmt ≥0.95 → 3-stage curriculum works; deliverable candidate. **No-go:** flat/collapse → narrow polish destabilizes even broad base (B4a mechanism). | `SYNTHESIS.md:140` `thread-b-broad-distill.md:431-440` |
| **E3** | **LR ablation 1e-5 cosine** | **P2 if E1/E2 format-degraded or E1 No-go** | `checkpoint-900-hf` | `distill_cot_mix` | LR **1e-5 cosine** 3ep ~144 steps otherwise E2 | **4N `select=5 NHOSTS_TRAIN=4 GBS768 12h`** | **Go:** ≥E2 best or fmt ≥0.90 gen_len <400 → promote 1e-5. **No-go:** flat vs E2 → LR not lever. | `SYNTHESIS.md:141` |
| **E4** | **Seq-len 4096 vs 8192** (same as A4 but as ranked experiment) | **P2 cheap alongside E3 or if E1 gen_len>500** | `checkpoint-900-hf` | `distill_cot_mix` re-tokenized @4096 (one-off 1N pretok `NPROC=4` ~1-2h → `/tegu/.../len4096`) | LR 2e-5 cos 1ep `LBS4 GAS4 max_length 4096 GBS384` tokens/step matched to 8192@GAS8 | **2N `select=3 NHOSTS_TRAIN=2 GBS384 01:00:00` (+1N pretok `select=1`)** | Within 0.02 acc fmt ≥0.85 → 4096 efficient default; fmt <0.70 or -0.05 vs E1 → keep 8192. | `SYNTHESIS.md:142` `thread-b-broad-distill.md:383-395` |
| **E5** | **Single-stage control from gs138650 — null hypothesis** | **P3 one control, not main line** | **`global_step138650` raw MDS** `~/global_step138650` NOT `checkpoint-900` | `distill_cot_mix` 36.9k @8192 (tokenizer-compatible — vocab 256000 Gemma) | LR 2e-5 cos 1ep ~48 steps `LBS2 GAS8 max_length 8192` as E1 | **4N `select=5 NHOSTS_TRAIN=4 GBS768 12h` ~1h** | **Surprise Go:** >0.10 fmt ≥0.80 → revisit structural hypothesis. **Expected No-go:** 0.03-0.07 → confirms structure is lever (B3 class). | `SYNTHESIS.md:143` `thread-b-broad-distill.md:252-281` |
| **E6** | **CPT-FWE frontier base (gated follow-on)** | **P4 gated on CPT 112k+ HF** | **CPT-FWE 112k+ HF** `step-112000` via `hf_batch_convert_cpt_fwe.sh` flavor `2b-mds` — gate verified `config.json`+`tokenizer.json`+`model.safetensors` non-gibberish on 5-problem smoke | `distill_cot_mix` | LR **1e-5 cosine conservative** (shifted base; 2e-5 re-warm may repeat CPT-17000 0.015 collapse) 1ep ~48 steps `max_length 8192` | **4N `select=5 NHOSTS_TRAIN=4 GBS768 12h`; precondition: narrow probe on CPT base (gsm8k-r1cot 3ep LR 2e-5 2N ~30m) scored first** | **Go:** >E1 best or >0.15 if E1 No-go → schedule CPT-to-190k re-train. **Pivot:** ~E1 within noise → CPT did not lift CoT ceiling (may still help MMLU). **No-go:** collapse <0.05 → needs general SFT first (mirror B2 on new base). | `SYNTHESIS.md:144-145` `04-sft-broad-distill.md:247-250` |
| **E7** | **Scale sensitivity 2N vs 4N vs 8N (optional)** | **P5 only if GBS questioned** | `checkpoint-900-hf` | `distill_cot_mix` | LR 2e-5 cos 1ep `LBS2 GAS8` hold GBS per scale (2N GBS384 / 8N GBS1536 `select=9 NHOSTS_TRAIN=8`) | **2N `select=3 NHOSTS_TRAIN=2` + 8N `select=9 NHOSTS_TRAIN=8 workq 12h`** | Informational — if 8N degrades vs 4N future production stays 4N. | `SYNTHESIS.md:145` |

**Minimum viable queue (tight allocation):** `A0 + E1 + E2` = 2 SFT runs (48 + 144 steps) + 4 evals, ~3h compute, answers "does broad-distill on proven base beat narrow-distill?" (`SYNTHESIS.md:164`, `04-sft-broad-distill.md:234`).

**Queue timeline (SYNTHESIS §3 E7):**

```
Time 0h ─────────────────────────────── Time ~1h ───── Time ~3h ── Time ~6h
A0 (0 GPU audit, 10m) ─┐
A1 (1 tile probe, 2m)  ─┤ All three in parallel, <30 min
A3 (0 GPU n=8, 30m)    ─┘
                        v
E1 (4N 1ep ~1h) ─────> eval (~2 min) ─┐
                                      ├─> branch on gate:
E2 (4N 3ep ~1.5h)     ◄── Go/Pivot ────┤──> E2b (2N polish ~30m)
E3 (4N 1e-5)          ◄── format-sens ─┤
E4 (1N pretok+2N)     ◄── gen_len>500 ─┤
E5 (4N cold-start)    ◄── control ────┤──> (informational)
                                      └─> E6 (CPT-FWE gated on 112k+ HF)
```

**B-series anchor for gates (all `temperature 0.0 max_tokens 768 frameworks/2026.1.0 vLLM 0.26 XPU fp32 workq` vs B2 `0.205/0.985/282`):** B3 0.05/0.86/601/26 (dilution), B4a 0.02/0.26/1378/133 (verbose-poisoned), B4b 0.065/0.86/611/27 (reweight only +0.015), B2v2 0.09/0.86/416/22 (ep3) and 0.105/0.91/378/15 (ep6, loss 0.8069 acc 0.8198 clean — eval halved silently), CPT-17000 SFT 0.015 collapsed (`04-sft-broad-distill.md:82-91`, `thread-b-broad-distill.md:36-45`, `POST-TRAINING-2B.md:77-78,98-101`).

---

## 3. Smoke-first guardrail — NUM_EPOCHS=1 or --max_steps 1000 before 3ep

**Rule:** Do not run `NUM_EPOCHS=3` (144 steps @ GBS768 @8192 on 36.9k packed) without a prior 1-epoch smoke and per-epoch eval. The guardrail is `sft_distill_cot_4n_verification.md:58-71` (§4) — **cap full-mix SFT at `O(1000)` steps or drop LR**.

**Why the cap exists — ckpt-8672 forgot at ~4000 steps with LR >1e-5:**

| Artifact | LR schedule | What happened | Source |
|---|---|---|---|
| `checkpoint-8672` (`tulu_math_uc_mix_full` full epoch, 8672 steps ~54B tok, LR 2e-5 cosine) | LR 2e-5 → 0 over 8672; LR >1e-5 for **~4000 steps** (at 8672, LR(4350)=1e-5 = half-peak at S/2) | **Catastrophically forgot:** hellaswag 0.593→0.273, arc_easy 0.694→0.298 while training loss 0.357 looked superb — cosine's flat top kept LR high long enough to erase capability | `POST-TRAINING-2B.md:50,126` `thread-d-b2-autopsy.md:132` `SYNTHESIS.md:19,275` `sft_distill_cot_4n_verification.md:60` |
| B2v2 continuation | LR 2e-5 cosine, 93 steps → +6 / +12 steps on gsm8k-r1cot | **Halved 0.205→0.09/0.105** at training loss 0.8069 acc 0.8198 (clean, descending) — eval is only detector | `POST-TRAINING-2B.md:77-78` `thread-b-broad-distill.md:62-63` |
| Current lever | `NUM_EPOCHS=3` LR 2e-5, 36.9k packed → **144 steps @4N GBS768 / 72 @8N GBS1536** (well under 1000 at this scale, but future mix/re-tokenize could grow steps) | Relying on epochs alone is unsafe — guardrail is about *any* future scale that grows steps; use `max_steps` for hard cap | `sft_distill_cot_4n_verification.md:65` `sft_distill_cot_4n.sh:230,242-243` |

**How to enforce (pick one):**

```bash
# Preferred — smoke 1ep, inspect eval, then extend (this doc's §4 submit lines do this):
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
# inspect ${CKPT_DIR}/final eval (format + gen_len + n_unclosed) before:
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=3 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# Hard cap — add --max_steps 1000 to the train_sft invocation if editing sft_distill_cot_4n.sh:261-275
# (sft_distill_cot_4n_verification.md:67-68 — not in current default; add at launch if needed)

# Lower LR — E3 arm at 1e-5 if 2e-5 shows format degradation (SYNTHESIS E6 LR arm)
```

**Cap is proven safe at 900-1000 steps:** `checkpoint-900` (900 steps at LR 2e-5) held IFEval 0.253 and base-LM intact (hellaswag 0.592 vs base 0.593); `POST-TRAINING-2B.md:107-112` + `tulu_math_uc_mix_full/README.md` document the 900-vs-8672 split. B2 stage-2 at 93 steps also proves the low-step regime is safe. `SYNTHESIS.md:268-275` prohibitions 2-3 codify this: do not exceed 1000 steps at 2e-5 without the E5-E6 curve.

**logging_steps 5 fix:** `sft_distill_cot_4n.sh:272` sets `logging_steps 5`; `sft_distill_cot_4n_verification.md:70` notes this fixes B2v2's `logging_steps` vs `save_strategy=epoch` single-entry bug where loss trajectory was invisible before commit. `save_strategy epoch` (`:273`) gives per-epoch ckpts for the cap check (`04-sft-broad-distill.md:277`).

---

## 4. Submit commands — smoke 4N 1ep (select=5 NHOSTS_TRAIN=4), full 3ep, CPT-FWE follow-on gated

**Prereqs:** §1 gates passed (pretok `dataset_info.json` + `du 3.4G` + `[decontam]` non-trivial + `BASE_MODEL` checkpoint-900-hf present). All jobs share `PBS -l filesystems=tegu:home` (PBS lacks `/eagle` on Sunspot), `PBS -q workq`, `PBS -A datascience`, `VENV=rl-monarch-torch214-locality-20261001` + LD-first cache (see §1 VENV row), `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` (offline reads via `--pretokenized_dataset`, no Hub — `sft_distill_cot_4n.sh:257-264`).

### Copy-paste launches (submit from `torchtitan` checkout — `sft_distill_cot_4n.sh:85-93` preflight `SUBMIT_DIR` + `PBS_O_WORKDIR` canonicalization)

```bash
# ── 0. Gate check (Sunspot login) — both must pass before any qsub below ─────────
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192
grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log | tail -20
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900{,-hf}/ 2>&1 | head

# ── 1. Smoke: 1 epoch on 4N pack (48 steps GBS768 @8192, ~1h actual, 12h walltime headroom) ─
# Header default is already select=5 (4N active +1 spare per sft_distill_cot_4n.sh:6) — explicit select=5 shown for clarity.
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# ── 2. After smoke eval (consolidate + eval_cot_gsm8k on final — see Post-training below) ─
#    If E1 Go/Pivot, extend to 3ep (144 steps @4N, per-epoch eval via save_strategy epoch):
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=3 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# ── 3. Scale variant — standalone 8N (96 GPUs, 24 steps/ep at 3ep = 72 steps, GBS1536) ──
#    Per sft_distill_cot_4n.sh:58-65 header note; override select=9 (8N active +1 spare).
qsub -l select=9 -v NHOSTS_TRAIN=8,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
qsub -l select=9 -v NHOSTS_TRAIN=8,NUM_EPOCHS=3 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# ── 4. Cheap 2N scale (E7 GBS sensitivity, optional P5) ─────────────────────────────
qsub -l select=3 -v NHOSTS_TRAIN=2,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# ── 5. CPT-FWE follow-on (E6) — GATED on CPT 112k+ HF (do not run until gate passes) ─
# Gate: CPT-FWE 112k+ HF verified (config.json + tokenizer.json + model.safetensors non-gibberish)
#   via hf_batch_convert_cpt_fwe.sh flavor 2b-mds (wrong flavor = gibberish — SYNTHESIS.md:212)
#   at /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-112000/hf
#   + narrow probe on that base scored (gsm8k-r1cot 3ep LR2e-5 2N ~30m) to calibrate format.
#   Per POST-TRAINING-2B.md:37 (parked to CPT 50k) and SYNTHESIS.md:144 — E6 blocked until that HF exists.
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=1,BASE_MODEL=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-112000/hf torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
# Note: CPT-FWE base uses LR 1e-5 conservative via E3 arm (shifted manifold; 2e-5 re-warm may repeat 0.015 shock — SYNTHESIS.md:144).

# ── 6. If pretok missing — rebuild 1N stage 1 (do not launch 4N consumer until its gate passes) ─
qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh
# Poll: ls /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
#       grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log
```

### Dirs the scripts produce (per `sft_distill_cot_4n.sh:237-240,261-275`)

| Var | Default (via script) | Example at 4N 1ep | Source |
|---|---|---|---|
| **`CKPT_DIR`** | `$_SFT_ROOT/agpt2b-distill-cot-pretok-${NNODES}n-ep${NUM_EPOCHS}` where `_SFT_ROOT=$TEGU_SFT_ROOT` if present else `${SUBMIT_DIR}/outputs/sft` | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-distill-cot-pretok-4n-ep1` (1ep) or `…-ep3` (3ep) | `sft_distill_cot_4n.sh:237-239` |
| **`LOG_DIR`** | `${SUBMIT_DIR}/logs/sft-distill-pretok-${NHOSTS_TRAIN}n-${PBS_JOBID%%.*}` | `logs/sft-distill-pretok-4n-<jobid>/run.log` (contains `DISTILL_MAX_THINK_CHARS`, `git log -1 --oneline`, GBS/PBS_SELECT, `[decontam]` gate note) | `sft_distill_cot_4n.sh:240` `sft_distill_cot_4n_verification.md:124-126` |
| **Pretok fallback** | `${SUBMIT_DIR}/outputs/sft-pretok/distill-cot-mix-len8192` | Only used if `PRETOK_DIR` (`/tegu/.../len8192`) not ready — `_wait_for_pretok()` sleep-loops there | `sft_distill_cot_4n.sh:152-164` |
| **Launch** | `ezpz launch --nproc ${NGPUS_ACTIVE} --nproc_per_node ${PPN} --auto-retry --spare-nodes auto --timeout 1800 -- python3 -m torchtitan.experiments.ezpz.rl.train_sft ... --pretokenized_dataset ${USE_PRETOK} ... --bf16 --fsdp full_shard --max_length 8192 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --dataset_num_proc 8 --logging_steps 5 --save_strategy epoch` | GBS `768@4N (48 GPUs LBS2 GAS8)` / `1536@8N (96 GPUs)` / `384@2N (24 GPUs)` — tokens/step ~6.3M @8192 @4N | `sft_distill_cot_4n.sh:242-275` `05-infra-rl-storage.md` VENV map |

**Pack budget note (4N default):** `select=5` = 4N active +1 spare; 4N active + peer `cpt_fineweb 4N` + `cpt_olmo2575 4N` + `rl_grpo 2N` + `test.sh 2N` = 16N active (budget 16N, `workq` max 128N) — `sft_distill_cot_4n.sh:57-60,246`.

### Next step after each SFT job — consolidate + eval

```bash
# Per sft_distill_cot_4n.sh:280-281 and 04-sft-broad-distill.md:272-273 — run for each checkpoint (per-epoch ckpts at save_strategy epoch):
./torchtitan/experiments/ezpz/scripts/eval/consolidate_and_eval_cot.sh ${CKPT_DIR}/final
# Also: ${CKPT_DIR}/checkpoint-*/ for per-epoch trajectory (identify collapse epoch like B4a per-epoch 900/1800/3605 at 0.035/0.070/0.050)

# Evals land as /lus/tegu outputs/evals/cot/gsm8k-cot-*.jsonl per POST-TRAINING-2B.md:80
# Harness: 200-problem GSM8K-CoT frameworks/2026.1.0 vLLM 0.26 XPU fp32 temperature 0.0 max_tokens 768
# Report: cot_accuracy / format_hit_rate / mean_gen_len / n_unclosed vs B2 0.205/0.985/282/~0
# Then apply §2 gate (Go/Pivot/No-go) before launching the next PBS job.
```

---

## 5. Mix envelope + decontam invariants

### Envelope — byte-identical to gsm8k-r1cot and eval_cot_gsm8k.py

```
<think>{trace}</think>
<answer>\boxed{N}</answer>
```

Wrapped in `_openthoughts_format_row` + `_openr1_format_row` (`datasets_sft.py` component builders) — same envelope `agpt2b_b2v2_distill_cot_2n.sh:88-90` used for B2v2 `gsm8k-r1cot` baseline (`g<1k` vs `boxed`). `eval_cot_gsm8k.py` extracts via `LAST \boxed{}` and expects this exact envelope; format is gated at ≥0.80 (E1 Go) and ≥0.90 (promotion) — `SYNTHESIS.md:230-235`.

Rules (per `thread-b-broad-distill.md:101-106`, `datasets_sft.py:_build_distill_cot_mix`):

* Code-generation rows without `\boxed{}` are **intentionally dropped** (they would teach a malformed tail).
* Envelope is `packed` at `max_length 8192` with `assistant_only_loss=True` — only `<think>`+`<answer>` tokens carry loss; prompt tokens masked.
* A0 (§2) decodes 3 samples to verify well-formedness before E1; unclosed `</answer>` rate is the format-collapse detector (B4a 133/200 vs B2 ~0).

### Decontam — 13-gram + normalized-exact, baked into `cba5c64` recipe hash

| Invariant | Value | Source |
|---|---|---|
| **Detector** | `decontam_traces.GSM8KDecontaminator` — **13-gram token overlap** + **normalized-exact question match** (question lowercased, punctuation/whitespace stripped, then exact string match — `decontam_traces.py:GSM8KDecontaminator` docstring) | `_pretokenize_distill_cot_mix_1n.sh:27-35` `sft_distill_cot_4n_verification.md:25-26` `datasets_sft.py:_decontaminate_against_gsm8k()` |
| **Scope** | **Per-component before interleave** — both OpenThoughts-114k and OpenR1-Math-220k pass through `_decontaminate_against_gsm8k()` before `_materialized_mix_load_or_build()` interleaves them | `_pretokenize_distill_cot_mix_1n.sh:32-35` `sft_distill_cot_4n_verification.md:25` |
| **Hash capture** | Decontam is part of each component's registered build → captured in **materialized-mix recipe hash** `cba5c64`; a cache hit at `cba5c64` is **guaranteed decontaminated per that detector** | `sft_distill_cot_4n_verification.md:26` `thread-b-broad-distill.md:172-175` `SYNTHESIS.md:118` |
| **Logging** | Per-source `[decontam]` drop counts in `logs/pretokenize-distill-cot-mix-*/run.log` — verify non-trivial before trusting eval delta | `_pretokenize_distill_cot_mix_1n.sh:144-146` `sft_distill_cot_4n.sh:42-45` |
| **Sensitivity note** | 13-gram is conservative (false positives negligible); reworded GSM8K-test with <13 shared tokens passes — `A3 n=8` (SYNTHESIS prohibition 13) audits residual; delta >2% → re-stage at `n=8` | `SYNTHESIS.md:285` `thread-b-broad-distill.md:168-175` |

**Audit the detector:**

```bash
grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log | tail -20
# Expected: two lines (one per component) with drop counts in the dozens (114k+220k corpora
# with GSM8K overlap should drop non-trivial rows). Zero drops on this size is suspicious.
# For n=8 sensitivity gate: A3 re-run decontam_traces --n 8 --report-only on the landed artifact.
```

### DISTILL_MAX_THINK_CHARS — 16000 trims run-ons, part of recipe

| Var | Default | Role | Source |
|---|---|---|---|
| **`DISTILL_MAX_THINK_CHARS`** | `16000` (chars) | Trims pathological frontier traces at format time — keeps normal long traces intact; unlike B4b's `OPENR1_MAX_THINK_CHARS=1200` clamp that filtered 100% of OpenR1 at median 13k chars, 16000 keeps >80% (median OpenR1 ~13k) | `_pretokenize_distill_cot_mix_1n.sh:72-75,112` `sft_distill_cot_4n.sh:38-41` `sft_distill_cot_4n_verification.md:27-28` `thread-b-broad-distill.md:138-139` |
| **Recipe linkage** | Part of component builds → **changing it forces a fresh mix cache** (new recipe hash, not `cba5c64`) — `EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes` | `_pretokenize_distill_cot_mix_1n.sh:72-75` `sft_distill_cot_4n_verification.md:27` |
| **Tighten knob** | If B4 verbose-dilution reappears (gen_len >600 with format intact but acc stuck — B4b 0.065 pattern), tighten to **4000** and re-run pretok (new hash) | `_pretokenize_distill_cot_mix_1n.sh:38-41` `sft_distill_cot_4n.sh:38-41` |

---

*Ledger 08 produced 2026-11-16. No training or qsub was issued in this unpark — submit-ready specs only. Every numeric claim cites its source file:line or PBS job ID; [INFERENCE] marks the only extrapolations (expected gate values, R/Q status). Verify §1 gates on Sunspot login, run A0/A1/A3 in parallel, then launch E1 smoke `NUM_EPOCHS=1` before `NUM_EPOCHS=3`.*
