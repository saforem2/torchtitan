# 04 — SFT broad distill (distill_cot_mix @ 8192, pretok-gated)

**Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a7992` (prior doc anchor `bcbca5bd90` in `thread-b-broad-distill.md:6`, `thread-d-b2-autopsy.md:9`).
**Scope:** Pre-tokenized reasoning-distillation SFT lever (`distill_cot_mix`) on the proven two-stage base. No training launched in this verification window; this ledger records what landed, why it exists, what was audited, what broke, and how to launch.
**Sources:** `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh`, `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh`, `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`, `torchtitan/experiments/ezpz/docs/production/sft_distill_cot_4n_verification.md`, `torchtitan/experiments/ezpz/docs/exploration/thread-b-broad-distill.md`, `torchtitan/experiments/ezpz/docs/exploration/thread-d-b2-autopsy.md`, `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md`.

---

## What is running / landed — pretok 12483443 (91k rows→36.9k packed @8192 3.4G cba5c64, 6 segfaults 3553055 SIGBUS), B2v2 jobs, SFT scripts ready but gated/parked to 50k

### Pretok producer: landed and verified

| Field | Value | Source |
|---|---|---|
| **Job** | `12483443` `Exit` (final attempt) | `POST-TRAINING-2B.md:36` (2026-11-12 entry), `thread-b-broad-distill.md:95` |
| **Script** | `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh` | script header `PBS -l select=1 -l walltime=12:00:00 -q workq` |
| **Mode** | 1N single-rank, `NPROC=1`, direct `${VENV}/bin/python -m torchtitan.experiments.ezpz.rl.train_sft --pretokenize_to`, no `ezpz_setup_job` MPI wrapping | `_pretokenize_distill_cot_mix_1n.sh:105-123` |
| **VENV** | `rl-monarch-torch214-locality-20261001` (with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, LD-first cache) | `_pretokenize_distill_cot_mix_1n.sh:89-99` |
| **Tokenizer base** | `global_step138650` (`~/global_step138650`, vocab 256000, Gemma `assets/hf/gemma-7b` compatible) — tokenizer-only; training BASE_MODEL is independent | `_pretokenize_distill_cot_mix_1n.sh:107` |
| **Output** | `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` — `dataset_info.json` + 8 x `data-*.arrow`, `3.4G` | `POST-TRAINING-2B.md:37`, `thread-b-broad-distill.md:97`, `_pretokenize_distill_cot_mix_1n.sh:108` |
| **Rows** | **91,481 raw rows** (after envelope formatting + per-component GSM8K-test decontamination, before packing) → **36,934 packed sequences @ `max_length 8192`** | `POST-TRAINING-2B.md:37`, `thread-b-broad-distill.md:98-99` |
| **Mix** | `distill_cot_mix` = `0.50 OpenThoughts-114k-decontam + 0.50 OpenR1-Math-220k-decontam`, `all_exhausted` interleave, seed 42, `DISTILL_MAX_THINK_CHARS=16000` | `datasets_sft.py:1262` via `sft_distill_cot_4n_verification.md:18-27`, `_pretokenize_distill_cot_mix_1n.sh:21-22` |
| **Mix-cache** | Hit `cba5c64` in **5.8s** on final attempt (materialized-mix cache under `/tegu/datasets/datasets/ezpz_sft_mixes/cba5c64`, recipe hash captures decontam + `DISTILL_MAX_THINK_CHARS`) | `POST-TRAINING-2B.md:37`, `thread-b-broad-distill.md:95` |
| **Envelope** | `<think>{trace}</think>\n<answer>\boxed{N}</answer>` — byte-identical to `gsm8k-r1cot` and `eval_cot_gsm8k.py` | `thread-b-broad-distill.md:101` |
| **Decontam** | Per-component `decontam_traces.GSM8KDecontaminator` (13-gram + normalized-exact) before interleave, logged `[decontam]` drop counts in `logs/pretokenize-distill-cot-mix-*/run.log`, captured in recipe hash | `_pretokenize_distill_cot_mix_1n.sh:27-35`, `sft_distill_cot_4n_verification.md:25-26` |
| **Shared cache** | `HF_HOME=/tegu/datasets/datasets/hf`, `EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes` | `_pretokenize_distill_cot_mix_1n.sh:66-71`, `sft_distill_cot_4n.sh:108-110` |
| **Prior segfaults** | **6 jobs segfaulted** before landing, walking `NPROC 32→16→8→4→1` (e.g. `3553055 Segmentation fault` / `SIGBUS` on decontam/packing `map` subprocesses) until VENV-path fix + `NPROC=1` | `POST-TRAINING-2B.md:37` (`6 prior jobs e.g. 12483441 segfaulted at NPROC=1 via 3553055 Segmentation fault until VENV-path fix`), `thread-b-broad-distill.md:95` (`6 segfaults NPROC 32->16->8->4->1`) |

**Current state:** No pretok job is `R`/`Q`. The artifact is on disk and is the gate for all downstream SFT. Verify with:

```bash
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192
grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log
```

Per `sft_distill_cot_4n_verification.md:10-14` and `sft_distill_cot_4n.sh:156-167`, the consumer gates on `dataset_info.json` via `_wait_for_pretok()` with fallback `outputs/sft-pretok/distill-cot-mix-len8192/dataset_info.json`.

### B2v2 narrow SFT (the refutation that motivates broad distill)

B2v2 extended the **proven B2 stage-2 recipe** on the same base with more epochs on the same narrow data to test whether duration lifts the ~0.2 ceiling. Both jobs are done and evaluated; they refute that lever cleanly.

| Job | Role | VENV | Hparams | W&B | Outcome |
|---|---|---|---|---|---|
| `12482423` (train, 2N) / `12482572` (eval, `x1922c4s2b0n0 00:01:38`) | B2v2 ep3 (3→6ep, 6 steps) | `rl-monarch-torch214-locality-20261001` | `ckpt-900-hf → gsm8k-r1cot`, LR 2e-5 cosine, GBS 384, seq 2048, `dataset_num_proc 8` | `xakno7kg` (loss 0.8069 acc 0.8198) | **0.09 acc / 0.86 fmt / 416 gen_len / 22 unclosed** |
| `12482485` (train, 2N) / `12482573` (eval, `x1922c4s5b0n0 00:01:39`) | B2v2 ep6 (6ep, 12 steps) | same | same, 12 steps | `ur6pqqhp` | **0.105 acc / 0.91 fmt / 378 gen_len / 15 unclosed** |

Sources: `POST-TRAINING-2B.md:68-84` (table + eval harness `temperature 0.0 max_tokens 768 frameworks/2026.1.0 vLLM 0.26 XPU fp32 Sunspot workq`), `thread-b-broad-distill.md:42`, `thread-d-b2-autopsy.md:285-286`, script `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh:90-114` (note: that script defaults to `SFT_DATASET=gsm8k-r1cot` alone, not `distill_cot_mix` — the broad mix was shelved for B2v2 due to the HF cache race).

Training was healthy (loss descending, token acc 0.8198); eval halved vs B2 `0.205`. Takeaway per `POST-TRAINING-2B.md:88`: do not extend narrow stage-2 further. [INFERENCE] B2v2 jobs are complete (`Exit`/`Exit_status=0`); no B2v2 training is currently `R`.

### SFT consumer scripts: ready but gated/parked to 50k

| Script | PBS | Scale | GBS @ seq | Epochs | VENV | Purpose |
|---|---|---|---|---|---|---|
| `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh` | `select=5 workq 12h` (pack default 4N active +1 spare; override `select=9 NHOSTS_TRAIN=8` for 8N standalone) | 4N: `48*2*8=768` @ 8192, 48 steps/ep; 8N: `96*2*8=1536` @ 8192, 24 steps/ep | `max_length 8192 packing=True` | `NUM_EPOCHS=3` default, smoke `NUM_EPOCHS=1` | `rl-monarch-torch214-locality-20261001` | Stage-2 broad-distill SFT from `checkpoint-900-hf` on pretokenized `distill_cot_mix` (consumes via `--pretokenized_dataset`, `HF_DATASETS_OFFLINE=1`) |
| `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh` | `select=1 workq 12h` | 1N single-rank | `max_length 8192 dataset_num_proc 1` | 1 pre-tokenize → 36.9k packed | same VENV | Producer that landed the artifact above |

Both scripts set `HF_HOME` + `EZPZ_SFT_MIX_CACHE_DIR` to `/tegu/datasets` shared staging (`sft_distill_cot_4n.sh:106-110`, `_pretokenize_distill_cot_mix_1n.sh:66-71`). The consumer defaults to `BASE_MODEL=$TEGU_SFT_ROOT/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf` with fallback chain to sharded `checkpoint-900/` and legacy path (`sft_distill_cot_4n.sh:120-146`).

**Parked to CPT 50k:** Per `POST-TRAINING-2B.md:37` (2026-11-12 entry) and `POST-TRAINING-PLAN.md` phase ordering cited there, broad SFT via `sft_distill_cot_4n.sh --pretokenized_dataset /tegu/datasets/agpt2b-distill-cot-mix-len8192` is **ready but parked** to CPT 50k — no training until the 50k checkpoint converts to HF. `sft_distill_cot_4n_verification.md:6` explicitly states "No training launched. This is a gate check only."

---

## Why — two-stage tulu_math→ckpt-900→gsm8k-r1cot only 0.205 recipe, every single-stage collapsed, broad distill only open path adding new reasoning data

### The only recipe that worked (B2)

Per `thread-d-b2-autopsy.md:25-31` and `POST-TRAINING-2B.md:107-112`:

```
global_step138650 --[stage 1: tulu_math_uc_mix_full, 53.3M seqs, 1ep, seq 1024, 8N, 900 steps ~5.7B tok, LR 2e-5 cosine]--> checkpoint-900 --[stage 2: gsm8k-r1cot, 7.4k ex, 3ep, 93 steps, seq 2048, LR 2e-5 cosine]--> checkpoint-93 = B2
                  IFEval 0.179->0.253, base-LM intact (hellaswag 0.592 vs base 0.593)                                          0.205 CoT, format 0.985, gen_len 282
```

Stage 1 (`tulu_math_uc_mix` = tulu-3-sft-mixture 0.65 + OpenMathInstruct-2 0.15 + ultrachat-200k 0.20, `all_exhausted` interleave) builds a general instruction-following + shallow math manifold with no `<think>` contamination and no envelope conflict (`thread-d-b2-autopsy.md:62-116`). Stage 2 wraps that competence in the `<think>/<answer>\boxed{}` envelope on in-distribution GSM8K data. See `tulu_math_uc_mix_full/README.md` stage-1 chain `12470350` 11-link afterany.

### Every single-stage and narrow-extension collapsed

All eval numbers share the same harness: **200-problem GSM8K-CoT, `temperature 0.0 max_tokens 768`, `frameworks/2026.1.0 vLLM 0.26 XPU fp32`, Sunspot `workq`** (`thread-b-broad-distill.md:36`, `POST-TRAINING-2B.md:80`).

| Model | Base | Data lever | GSM8K-CoT acc | Format | Mean gen_len | n_unclosed /200 | Verdict | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| **B2 = checkpoint-93** | `checkpoint-900` (tulu_math_uc_mix) | `gsm8k-r1cot` 7.4k ex, 3ep, 93 steps, LR 2e-5 cos | **0.205** | **0.985** | **282** | **~0** | **deliverable** | `POST-TRAINING-2B.md:45`, `thread-b-broad-distill.md:38`, `thread-d-b2-autopsy.md:281` |
| **B3** | `gs138650` | tulu 0.30 + OpenR1 0.25 + gsm8k-r1cot 0.15 + ultrachat 0.15 + OMI2 0.15, seq 8192, 1ep | **0.05** | **0.86** | **601** | **26** | long-CoT dilution — flat 0.035/0.070/0.050 across ckpt 900/1800/3605, not late-collapse | `POST-TRAINING-2B.md:98`, `thread-b-broad-distill.md:39`, `thread-d-b2-autopsy.md:282` |
| **B4a** | B3-3605 (`b3-3605-hf-diag`) | gsm8k-r1cot finish, 3ep, per-epoch ckpt | **0.02** | **0.26** | **1378** | **133** | verbose bias baked in — finish stage cannot un-teach it, destabilizes envelope | `POST-TRAINING-2B.md:100`, `thread-b-broad-distill.md:40`, `thread-d-b2-autopsy.md:283` (jobs `12471671` train, `12471676/77/78` per-epoch) |
| **B4b** | `gs138650` | gsm8k-r1cot 0.40 + OpenR1-filtered 0.15 + tulu 0.30 + ultrachat 0.15, seq 4096 | **0.065** | **0.86** | **611** | **27** | symptom fixed, accuracy stuck — 0.05→0.065 only | `POST-TRAINING-2B.md:101`, `thread-b-broad-distill.md:41`, `thread-d-b2-autopsy.md:284` (job `12471672`, eval `12471685`) |
| **B2v2 ep3** | `checkpoint-900-hf` | gsm8k-r1cot, 3→6ep, 6 steps, GBS 384 | **0.09** | **0.86** | **416** | **22** | more narrow epochs hurt — loss 0.8069 acc 0.8198 clean, eval halved | `POST-TRAINING-2B.md:77`, `thread-b-broad-distill.md:42` (jobs `12482423`/`12482572` W&B `xakno7kg`) |
| **B2v2 ep6** | `checkpoint-900-hf` | gsm8k-r1cot, 3→6ep, 12 steps, GBS 384 | **0.105** | **0.91** | **378** | **15** | still -0.10 vs B2 — non-monotonic, no sweet spot beyond 3ep | `POST-TRAINING-2B.md:78`, `thread-b-broad-distill.md:43` (jobs `12482485`/`12482573` W&B `ur6pqqhp`) |
| **CPT-17000 SFT** | CPT 17000 HF (FineWeb-Edu 60/40) | narrow gsm8k-r1cot, 3ep, LR 2e-5, GBS 384 | **0.015** | **collapsed** | — | — | format collapse on shifted manifold — SFT probe, not CPT evidence | `POST-TRAINING-2B.md:35` (`cpt34k-gsm8k-3ep 0.015`), `thread-b-broad-distill.md:44` |
| **ckpt-8672** | `gs138650` | `tulu_math_uc_mix_full` full epoch 8672 steps ~54B tok, LR 2e-5 cos | — | — | — | — | catastrophically forgot: hellaswag 0.593→0.273, arc_easy 0.694→0.298 (LR >1e-5 for ~4000 steps) — rule: cap at O(1000) steps or drop LR | `POST-TRAINING-2B.md:50,126`, `thread-d-b2-autopsy.md:132` |

Key finding per `thread-b-broad-distill.md:82` and `POST-TRAINING-2B.md:166`: **structure beats mix.** B4b restored format/verbosity to B3-like health and only moved 0.05→0.065 (+0.015). No tested knob (weight, length filter `OPENR1_MAX_THINK_CHARS` 1200→16000, seq_len 8192→4096, epoch count) closed more than 0.015 of the 0.155 gap to B2. The structural hypothesis (`thread-b-broad-distill.md:68-83`): two-stage is a sequential curriculum; single-stage forces the 2B to learn breadth + envelope simultaneously, and envelope wins.

### Broad distill is the only open path adding new reasoning data

Per `POST-TRAINING-2B.md:174` (Next steps #1) and `thread-b-broad-distill.md:120-123`: prior mixes were breadth in *task format* but narrow in *reasoning provenance* (all GSM8K/OpenR1 math). The distill mix adds **frontier R1 traces spanning math + science + code** (OpenThoughts via `< |begin_of_thought|>` / `< |begin_of_solution|>` re-wrapped, code-gen rows without `\boxed{}` intentionally dropped) that are longer and more diverse than gsm8k-r1cot (B2 gen_len 282 vs `DISTILL_MAX_THINK_CHARS 16000`). Framed as a **stage-2 replacement** on the proven `checkpoint-900` base (and later on CPT-FWE once available), it respects the two-stage structure that is load-bearing.

> **Mix provenance discrepancy:** The assignment/task prompt describes the mix as `50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot`. The implemented `distill_cot_mix` in `torchtitan/experiments/ezpz/rl/datasets_sft.py:1262` and `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh:21-22` is `50% OpenThoughts-114k-decontam + 50% OpenR1-Math-220k-decontam` (both frontier-distilled R1 corpora, no gsm8k-r1cot). `sft_distill_cot_4n.sh:19-24` comment block still references `gsm8k-r1cot` from an earlier spec, adding to the label confusion. The 91,481-row / 3.4G landed artifact matches the `OpenThoughts + OpenR1` definition (both ~100k+ raw rows → ~91k after filtering/decontam; a 50/50 with 7.4k gsm8k rows would be much smaller or heavily upsampled). See `sft_distill_cot_4n_verification.md:29` for the same note. [INFERENCE] If a gsm8k-r1cot blend is desired, it must be staged as a separate mix-spec and pre-tokenize; do not assume the landed 36.9k-packed artifact contains GSM8K problems (`thread-b-broad-distill.md:107`).

---

## Outcome — pretok landed/verified, consumer scripts audited (BASE_MODEL fallback chain, VENV correct, LD-first advisory, guardrail O(1000) steps needs max_steps)

### Pretok: landed and gated

- **Gate passes** per `sft_distill_cot_4n_verification.md:8-15`: `PRETOK_DIR=/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` default matches in both consumer and producer; producer built it via `train_sft.py --pretokenize_to → save_to_disk` (1N, `NPROC=1`, direct `${VENV}/bin/python`, no MPI); consumer gates on `dataset_info.json` via `_wait_for_pretok()` with fallback `outputs/sft-pretok/distill-cot-mix-len8192/dataset_info.json`.
- **Shared-cache parity:** `HF_HOME=/tegu/datasets/datasets/hf` + `EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes` set identically in producer and consumer (`sft_distill_cot_4n.sh:108-110`, `_pretokenize_distill_cot_mix_1n.sh:66-71`).
- **Decontam invariant:** Both components via `_decontaminate_against_gsm8k()` → `decontam_traces.GSM8KDecontaminator` before interleave through `_materialized_mix_load_or_build()`; decontam baked into component builds and captured in recipe hash — a cache hit at `cba5c64` is guaranteed decontaminated; verify via `grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log` (`sft_distill_cot_4n_verification.md:26`, `_pretokenize_distill_cot_mix_1n.sh:27-35`).
- **Offline training:** Consumer sets `HF_DATASETS_OFFLINE=1` + `HF_HUB_OFFLINE=1`; `train_sft.py` sees `input_ids` present and skips prep (`skip_prepare_dataset`), so no HF streaming cache race / 429 / 8.5h re-tokenize (`sft_distill_cot_4n.sh:257-264`).
- **No training launched** — lever is parked per `POST-TRAINING-2B.md:37` (parked to CPT 50k) and `sft_distill_cot_4n_verification.md:6` (gate check only).

### Consumer scripts audited — `sft_distill_cot_4n_verification.md` §2-4

| Check | Expected | Found | Verdict | Source |
|---|---|---|---|---|
| **BASE_MODEL chain** | `checkpoint-900-hf` (IFEval 0.253, pre-collapse) → sharded `checkpoint-900/` (96 `.distcp`, TRL accepts DCP) → legacy `agpt2b-gsm8k-r1cot-2n` fallback; missing all three → fatal with `find outputs/sft -maxdepth 1` hint + re-consolidate pointer | `TEGU_SFT_ROOT/.../checkpoint-900-hf` default with two fallback `if [[ ! -d ]]` blocks; `accelerate merge-weights` + copy `config.json`/`tokenizer.*` re-consolidate path documented | **Pass** | `sft_distill_cot_4n.sh:120-146`, `sft_distill_cot_4n_verification.md:32-40` |
| **BASE_MODEL tokenizer note** | Producer uses `~/global_step138650` for tokenizer only; consumer uses `checkpoint-900-hf` for weights — no mismatch (vendored Gemma tokenizer identical across lineage) | `BASE_MODEL="${HOME}/global_step138650"` in pretok vs `BASE_MODEL=.../checkpoint-900-hf` in consumer | **Noted, no fix needed** | `sft_distill_cot_4n_verification.md:41-45` |
| **VENV** | `rl-monarch-torch214-locality-20261001` (only VENV with `trl 1.13`, not `xpu-torch214` or `xpu-torch216`) | Both scripts default to `locality-20261001`; consumer probes `python3 -c "import trl; print(trl.__version__)"` at `:112` | **Pass** | `sft_distill_cot_4n.sh:99-112`, `_pretokenize_distill_cot_mix_1n.sh:89-101`, `sft_distill_cot_4n_verification.md:48-52` |
| **LD-first** | Full LD-first cache `.cache/xpu-ld-first/<venv>` symlink farm (`libur_* libsycl.so* libumf* libtcm*`) + `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` + `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | Pretok has full block (`:90-94`); consumer only `LD_LIBRARY_PATH="${VENV}/lib:..."` | **Partial — advisory, not blocking** (training likely still runs via `VENV/lib`; add LD-first block for §5 parity per skill, no behavior change for verification) | `sft_distill_cot_4n_verification.md:48-56` |
| **GBS / seq / packing** | `per_device_train_batch_size 2 GAS 8 seq 8192 packing=True max_length 8192` → GBS 768 @ 4N (48 GPUs) / 1536 @ 8N (96 GPUs), `bf16 fsdp full_shard logging_steps 5 save_strategy epoch dataset_num_proc 8` | `GBS=$((NGPUS_ACTIVE*2*8))` at `sft_distill_cot_4n.sh:242`, `ezpz launch --nproc ... --auto-retry --spare-nodes auto` with `HF_DATASETS_OFFLINE=1` | **Pass** (B2v2 fix `dataset_num_proc 32→8` honored) | `sft_distill_cot_4n.sh:236-275` |
| **Guardrail O(1000) steps** | `POST-TRAINING-PLAN.md` guardrail: cap full-mix SFT at O(1000) steps or drop LR — `ckpt-8672` forgot hellaswag 0.593→0.273 at LR >1e-5 for ~4000 steps | Consumer defaults `NUM_EPOCHS=3` LR 2e-5, no `max_steps`; at 36.9k packed @ GBS 768, 3ep = ~144 steps @ 4N (well under 1000), but the guardrail is about *any* future scale/re-tokenize that grows steps — rely on epochs alone is unsafe | **Requires `max_steps` at launch** — prefer `NUM_EPOCHS=1` smoke then per-epoch eval, or add `--max_steps 1000` to `train_sft` invocation, or drop LR to 1e-5 if 3ep justified | `sft_distill_cot_4n_verification.md:58-71`, `POST-TRAINING-2B.md:126` |

**Log artifacts for this outcome:**
- Pretok run log: `logs/pretokenize-distill-cot-mix-${PBS_JOBID}/run.log` (contains `DISTILL_MAX_THINK_CHARS=16000`, `git log -1 --oneline` for `train_sft.py`/`datasets_sft.py`/`decontam_traces.py`, `[decontam]` drop counts, `du -sh` of output) — `_pretokenize_distill_cot_mix_1n.sh:109-149`.
- Consumer run log (when launched): `${SUBMIT_DIR}/logs/sft-distill-pretok-${NHOSTS_TRAIN}n-${PBS_JOBID}/run.log` (`sft_distill_cot_4n.sh:240-244`).

---

## What went wrong — NPROC 32->1 segfaults, VENV xpu-torch214 no trl (fixed to locality), HF cache race + 8.5h re-tokenize 12470281, CPT-17000 SFT 0.015 collapse

Four independent failure modes blocked this lever before it landed. All are now mitigated; the narrative below preserves the causal chain so future jobs do not regress.

### 1. NPROC 32→1 segfaults on decontam/packing `map` (SIGBUS / SIGSEGV)

**Symptom:** `3553055 Segmentation fault` / `SIGBUS` / `One of the subprocesses has abruptly died` during the decontamination `.filter()/.map()` and the TRL tokenize+packing `map` inside the component builds (`_pretokenize_distill_cot_mix_1n.sh:48-53` header warns that `dataset_num_proc` controls both tokenize and packing maps; high `num_proc` is unsafe on XPU).

**Jobs:** 6 consecutive `Exit`/`F` before landing, e.g. `12483441` segfaulted at `NPROC=1` via `3553055 Segmentation fault` until the VENV-path fix unblocked it; earlier attempts at `NPROC 32→16→8→4` were needed to walk down from the `EzpzSFTConfig` default 32 (96/64 SIGBUS'd the tulu packing map at scale per `thread-b-broad-distill.md:97` note).

**Fix landed:** `NPROC="${NPROC:-1}"` default in `_pretokenize_distill_cot_mix_1n.sh:106` (job `12483443` landed at `NPROC=1` with mix-cache hit `cba5c64` in 5.8s — `POST-TRAINING-2B.md:37`). Prior mitigation steps `15f3e19`/`3665e3a`/`11df0ee3a` fixed `SIGBUS` on distill packing by walking `NPROC` down. The consumer honors the same fix with `dataset_num_proc 8` (B2v2 fix `32→8` after OOM in map subprocesses — `agpt2b_b2v2_distill_cot_2n.sh:110`).

**Source:** `POST-TRAINING-2B.md:37` (6 prior jobs, `3553055 Segmentation fault`), `thread-b-broad-distill.md:95` (`NPROC 32->16->8->4->1`), `_pretokenize_distill_cot_mix_1n.sh:103-106`.

### 2. VENV `xpu-torch214` missing `trl` (fixed to `locality-20261001`)

**Symptom:** `ModuleNotFoundError: No module named 'trl'` / `ezpz_get_job_env` clobbering `VIRTUAL_ENV` to stale `xpu-torch214` (no `trl`) and `xpu-torch216`/`xpu-torch214` lacking `trl 1.13` for `SFTTrainer`.

**Jobs:** `12483441` and `12483253 R` pretok attempts died on `python3 -c "import trl"` probe before reaching tokenization; `agpt2b_b2v2_distill_cot_2n.sh:47-62` documents the `ezpz_get_job_env` clobber and the `PATH` + `LD_LIBRARY_PATH` force-fix.

**Fix landed:** Both scripts now pin `VENV=/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001` with `PATH="${VENV}/bin:..."` + `LD_LIBRARY_PATH="${VENV}/lib:..."` + `import trl` probe (`sft_distill_cot_4n.sh:99-112`, `_pretokenize_distill_cot_mix_1n.sh:89-101`). `POST-TRAINING-2B.md:64-65` wiring repairs `40b29ca`/`d551d04c`/`927dd1e4f` fixed the VENV mapping.

**Source:** `_pretokenize_distill_cot_mix_1n.sh:89-101` (trl probe), `agpt2b_b2v2_distill_cot_2n.sh:54-62` (Force correct venv via PATH), `sft_distill_cot_4n_verification.md:48-52` (VENV identity pass).

### 3. HF streaming cache race + 8.5h re-tokenize (job 12470281, ~93M-row full mix)

**Symptom:** Two coupled blockers on the TRL `SFTTrainer` / HF `datasets` path (SFT does **not** use Grain; Grain streaming is verified for CPT only — `POST-TRAINING-2B.md:64`):
- **Multi-rank HF cache coherency race** at 2N: rank 0 writes a new arrow cache, other ranks look for old shards; plus 429 storms from Hub streaming. `agpt2b_b2v2_distill_cot_2n.sh:88-90` explicitly shelved `distill_cot_mix` for B2v2 for this reason, falling back to `gsm8k-r1cot` alone.
- **~8.5h re-tokenize walltime** walltime at 93M rows (full `tulu_math_uc_mix_full` mix) — longer than the PBS walltime, triggering idle-watchdog `SIGTERM` (job `12470281` killed by watchdog; `sft_distill_cot_4n.sh:27-30` documents the `93M-row mix ~8.5h` cost).

**Fix landed:** Pre-tokenization pattern (`_pretokenize_distill_cot_mix_1n.sh` via `train_sft.py --pretokenize_to → save_to_disk num_proc=16`, then consumer via `--pretokenized_dataset` with `HF_DATASETS_OFFLINE=1` where `train_sft.py` skips prep because `input_ids` is present — `sft_distill_cot_4n.sh:30-32`, `POST-TRAINING-2B.md:64` "moves that cost to a dedicated one-time job"). Also fixed `SFT 1N labels bottleneck` (`np.where`-batched `add_column` not `datasets.map Manager` `EOFError` on `PVC/FLAT` — `POST-TRAINING-2B.md:21`, commits `d351f8c204`/`36cfd634a5`).

**Source:** `POST-TRAINING-2B.md:64` (SFT HF cache race + 8.5h re-tokenize), `sft_distill_cot_4n.sh:26-33` (Why pre-tokenized), `agpt2b_b2v2_distill_cot_2n.sh:88-90` (shelved distill_cot_mix), `POST-TRAINING-2B.md:21` (labels bottleneck 12483770).

### 4. CPT-17000 SFT 0.015 collapse — format collapse on shifted manifold (SFT probe, not CPT evidence)

**Symptom:** Narrow SFT-on-CPT (`agpt2b_b2v2_distill_cot_2n.sh` on CPT-17000 HF, LR 2e-5 3ep GBS 384) scored **0.015 acc with format collapse** vs B2 `0.205` (`POST-TRAINING-2B.md:35`, `thread-b-broad-distill.md:44`, table row CPT-17000 SFT).

**Diagnosis:** A FineWeb-Edu-shifted pre-training manifold plus narrow SFT is strictly harder than SFT on the original `gs138650` manifold at that CPT maturity (~8.9% of 100B budget). The probe used a raw CPT checkpoint with no `chat_template` and no general instruction SFT — a larger distribution gap than `checkpoint-900` ever was. The 0.015 is an **SFT hparam failure on a shifted base**, not evidence that CPT-FWE cannot help — the SFT recipe must be re-tuned to the new base (lower LR 1e-5, or mirror B2's two-stage: general SFT first, then CoT). Raw CPT CoT `0/200` pre-SFT is expected (degenerate loops, `n_unclosed=200` — `POST-TRAINING-2B.md:23`).

**Implication for broad distill:** Do not run broad-distill SFT blindly on CPT-FWE; calibrate with a narrow probe first and gate E6 on 112k+ HF availability (see Next steps §5).

**Source:** `POST-TRAINING-2B.md:35` (`cpt34k-gsm8k-3ep 0.025 vs cpt34k-distill-broad-4n 0.000`), `thread-b-broad-distill.md:64,132-134` (CPT-17000 probe generalizes the point), `thread-d-b2-autopsy.md` §1.3.

---

## Next steps — gate checks (PRETOK_DIR dataset_info.json + [decontam] + BASE_MODEL), E7 queue A0 audit→E1 smoke 1ep 4N→E2 3ep→E6 CPT-FWE gated, exact qsub with NUM_EPOCHS=1 smoke first

### Gate checks — run on Sunspot login before any qsub

Per `sft_distill_cot_4n_verification.md:72-91` (§5 Gate — do NOT launch training) and `thread-b-broad-distill.md:199-211` (Base availability check), **both** must pass:

**1. Pretok ready (producer gate):**

```bash
# Must exist and be non-empty
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192
# Must show non-trivial [decontam] drop counts (confirms 13-gram + normalized-exact ran, not a stale cache)
grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log | tail -20
# Expected: 91,481 raw → 36,934 packed @ 8192, 3.4G, 8 x data-*.arrow, recipe hash cba5c64
```

If missing, rebuild stage 1 only (1N, proxy ON, shared `/tegu`, VENV `locality-20261001`, `NPROC=1`):

```bash
qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh
# Do NOT re-trigger HF download from the 4N consumer — it waits via _wait_for_pretok() sleep loop
```

**2. BASE_MODEL exists (consumer gate):**

```bash
# A — primary base for E1/E2 (must be first; only base with known-good >0.10 CoT trajectory)
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900{,-hf}/ 2>&1 | head
# If -hf missing but sharded checkpoint-900/ exists (96 .distcp shards), re-consolidate:
#   accelerate merge-weights + copy config.json/tokenizer.* per docs/production/sft/README.md
# B — CPT-FWE frontier (gated, not yet deliverable at 111.2k/190.7k 58.3% — next convert at 112k)
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/hf* 2>&1; ls ~/torchtitan/outputs/hf* 2>&1
# C — raw pretrain (only for E5 control)
ls -lh ~/global_step138650/config.json ~/global_step138650/tokenizer.json 2>&1
```

If either pretok or BASE_MODEL is missing, **do not launch stage 2**. Re-run the missing stage-1 job, then re-check gates.

### E7 queue — ordered by information value per GPU-hour

Per `thread-b-broad-distill.md:398-534` (§5 Ranked experiment queue), all evals are 200-problem GSM8K-CoT `temperature 0.0 max_tokens 768` fp32 vLLM, reporting `cot_accuracy / format_hit_rate / mean_gen_len / n_unclosed` vs B2 `0.205 / 0.985 / 282 / ~0`. Common hparams: `rl-monarch-torch214-locality-20261001`, `assistant_only_loss=True`, `packing=True`, `bf16`, `fsdp full_shard`, `dataset_num_proc 8`, `save_strategy epoch`, `HF_DATASETS_OFFLINE=1`.

```
Time 0h ─────────────────────────────── Time ~6h ── Time ~12h
A0 (0 GPU, artifact audit) ─┐
A1 (1 tile, base probe)     ─┤  All three in parallel, <30 min total
A3 (0 GPU, decontam n=8)    ─┤
                            v
E1 (4N, 1ep, ~1h) ──────────> eval (~2 min) ─┐
                                              ├─> branch:
E2 (4N, 3ep, ~1.5h)  ◄──────── Go/Pivot ──────┤───────────> E2b (2N polish, ~30 min)
E3 (4N, LR 1e-5)     ◄──────── format-sensitive┤
E4 (1N pretok + 2N SFT) ◄───── gen_len > 500 ─┤
E5 (4N, cold-start)  ◄──────── control ───────┤───────────> (informational)
                                              └───────────> E6 (CPT-FWE, gated on 112k+ HF)
```

**Minimum viable queue (allocation tight):** `A0 + E1 + E2` — 2 SFT runs (48 + 144 steps) + 4 evals, ~3h compute, answers "does broad-distill on the proven base beat narrow-distill?"

| ID | Base | Mix | Hparams | Scale | Gates | Artifacts |
|---|---|---|---|---|---|---|
| **A0** | — | distill_cot_mix landed artifact | 0 GPUs, login `load_from_disk` audit: rows ~91k, packed ~36.9k, `input_ids`/`labels` present, `seq_lengths` p50/p90, 3 decoded `<think>/<answer>\boxed{}` samples, `[decontam]` non-zero | 0 GPUs, 10 min | Blocks all if failed (wrong artifact, broken envelope, zero decontam) | `thread-b-broad-distill.md:289-327` |
| **A1** | `checkpoint-900-hf` | — | 1 XPU tile `eval_cot_gsm8k.py` on base itself | 1 tile, ~2 min | [INFERENCE] Expect ~0.0 acc (never trained on envelope); informs E1 delta | `thread-b-broad-distill.md:330-341` |
| **A3** | — | distill_cot_mix | `decontam_traces --n 8 --report-only` re-run, compare drop-rate delta vs n=13 | 0 GPUs, 30 min | <0.5% delta = negligible; >2% = re-stage at n=8 before gain claim | `thread-b-broad-distill.md:360-381` |
| **E1 — THE GATE (P0)** | `checkpoint-900-hf` | `distill_cot_mix` @ 8192, 36.9k packed, 3.4G | LR 2e-5 cosine, **1 epoch** = ~48 steps @ GBS 768, LBS 2 GAS 8, `logging_steps 5` | **4N** `select=5 NHOSTS_TRAIN=4 PPN=12 GBS=768 workq 12h` (~1h actual) | **Go:** acc ≥0.15 fmt ≥0.80 gen_len <600 → E2. **Pivot:** 0.05-0.15 good fmt → E2b or MATH/science eval. **No-go:** <0.05 or fmt <0.50 → stop, run A2 component ablation | `outputs/sft/agpt2b-distill-cot-pretok-4n-ep1/`, W&B, `consolidate_and_eval_cot.sh` JSONL |
| **E2 (P1)** | `checkpoint-900-hf` | same | LR 2e-5 cosine, **3 epochs** = ~144 steps @ 4N, `save_strategy epoch` per-epoch eval | 4N GBS 768 (~1.5h) | **Go:** ep3 > ep1 and ≥0.18 → E3/E6. **Flat:** ep3 ~ ep1 ±0.02 → cap at 1ep. **No-go:** ep3 < ep1 or fmt degrades → cap future at 1ep | `outputs/sft/agpt2b-distill-cot-pretok-4n-ep3/` (3 per-epoch ckpts) |
| **E2b (P1)** | E1/E2 best-epoch HF | `gsm8k-r1cot` 7.4k ex | LR 2e-5 (or 1e-5 if format-sensitive), 3ep ~93 steps @ GBS 384, seq 2048 | **2N** `select=3 NHOSTS_TRAIN=2 GBS=384` (~30 min) | **Go:** >E1/E2 best and ≥0.15 fmt ≥0.95 → deliverable candidate. **No-go:** flat/collapse → broad manifold does not support narrow polish | `outputs/sft/agpt2b-broad-distill-gsm8k-polish-2n-ep3/` |
| **E3 (P2)** | `checkpoint-900-hf` | same | LR **1e-5 cosine**, 3ep ~144 steps @ 4N, else as E2 | 4N GBS 768 | **Go:** ≥E2 best or fmt ≥0.90 gen_len <400 → promote 1e-5. **No-go:** flat vs E2 → LR not the lever | `outputs/sft/agpt2b-distill-cot-pretok-4n-ep3-lr1e5/` |
| **E4 (P2)** | `checkpoint-900-hf` | re-tokenized `distill_cot_mix` @ **4096** (1N pretok `NPROC=4` ~1-2h → `/tegu/.../agpt2b-distill-cot-mix-len4096`) | LR 2e-5 cos, 1ep, LBS 4 GAS 4 seq 4096 GBS 384 @ 2N | 2N GBS 384 (~1h) + 1N pretok | **Go:** within 0.02 of E1 fmt ≥0.85 → 4096 is efficient default. **No-go:** fmt <0.70 or -0.05 vs E1 → keep 8192 | `thread-b-broad-distill.md:383-395` |
| **E5 (P3 control)** | `global_step138650` (raw MDS, not ckpt-900) | `distill_cot_mix` @ 8192 | LR 2e-5 cos, 1ep ~48 steps @ 4N | 4N GBS 768 (~1h) | **Surprise Go:** >0.10 fmt ≥0.80 → revisit structural hypothesis. **Expected No-go:** 0.03-0.07 → confirms structure is the lever | `outputs/sft/agpt2b-distill-cot-coldstart-4n-ep1/` |
| **E6 (P4, gated)** | **CPT-FWE 112k+ HF** (via `hf_batch_convert_cpt_fwe.sh` flavor `2b-mds`, verified `config.json`+`tokenizer.json`+`model.safetensors`, 5-problem GSM8K smoke not gibberish) | same | LR **1e-5 cosine** conservative, 1ep ~48 steps @ 4N, seq 8192 | 4N GBS 768, gated on CPT 112k+ HF + narrow probe on that base scored | **Go:** >E1 best or >0.15 if E1 No-go → schedule CPT-to-190k re-train. **No-go:** collapse/<0.05 → needs general SFT first (mirror B2 on new base) | `outputs/sft/agpt2b-cptfwe-distill-cot-4n-ep1-lr1e5/` |
| **E7 (P5, optional)** | `checkpoint-900-hf` | same | LR 2e-5 cos, 1ep, GBS 384 @ 2N / 1536 @ 8N (`select=9 NHOSTS_TRAIN=8`) | 2N+8N | Informational GBS sensitivity | `outputs/sft/agpt2b-distill-cot-pretok-{2,8}n-ep1/` |

Per `POST-TRAINING-2B.md:174` and `sft_distill_cot_4n_verification.md:72-78`, **E6 is blocked** until CPT-FWE 111.2k/190.7k (58.3%) reaches 112k and converts; CPT chain `12483856 R MFU 15.8% tps 4625` is healthy at `step-111200` and will hop at `~13:29 UTC` expiry.

### Exact qsub — smoke first (NUM_EPOCHS=1)

Per `sft_distill_cot_4n_verification.md:86-91` and `sft_distill_cot_4n.sh:62-67`:

```bash
# 1. Verify gates (see Gate checks above) — both must pass.

# 2. Smoke: 1 epoch on 4N pack (48 steps, ~1h actual, guardrail-safe)
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# 3. After smoke eval (consolidate + eval_cot_gsm8k on final/):
#    If Go/Pivot, extend to 3ep (144 steps, per-epoch eval):
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=3 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
#    Or standalone 8N (96 GPUs, 24 steps/ep at 3ep = 72 steps):
qsub -l select=9 -v NHOSTS_TRAIN=8,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
qsub -l select=9 -v NHOSTS_TRAIN=8,NUM_EPOCHS=3 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

# 4. Hard guardrail cap (alternative to epochs-only) — add to train_sft invocation if editing script:
#    --max_steps 1000  (and/or lower LR to 1e-5 per E3)

# 5. Post-training: consolidate + eval (each checkpoint)
./torchtitan/experiments/ezpz/scripts/eval/consolidate_and_eval_cot.sh ${CKPT_DIR}/final
#    Evals land as /lus/tegu outputs/evals/cot/gsm8k-cot-*.jsonl per POST-TRAINING-2B.md:80
```

**Why smoke first:** B2v2 at 3→6ep (6→12 steps) halved accuracy with healthy training curves (loss 0.8069 acc 0.8198) — eval is the only detector (`thread-b-broad-distill.md:62-63`). Per-epoch `save_strategy epoch` + `logging_steps 5` (fixes prior `logging_steps` vs `save_strategy=epoch` single-entry bug — `sft_distill_cot_4n_verification.md:70`) makes the trajectory visible before committing to longer runs. The 12h walltime is headroom for queuing + I/O; actual compute is ~1-1.5h.

**If pretok missing (rebuild 1N):**

```bash
qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh
# Poll: ls /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
#       grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log
```

Do not launch the 4N consumer until that gate passes — it avoids re-triggering the multi-rank HF cache race.

---

*Ledger produced 2026-11-16. No training or qsub was issued in this verification. Every numeric claim cites its source file or PBS job ID; [INFERENCE] marks the only extrapolations (current R/Q status, expected gate values).*
