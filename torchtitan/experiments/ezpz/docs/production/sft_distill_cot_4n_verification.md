# sft_distill_cot_4n — Phase 2 broad CoT lever verification

**Date:** 2026-11-16 · **Plan ref:** `POST-TRAINING-PLAN.md` Phase 2 — Broad CoT data (`distill_cot_mix`).
**Scripts audited:** `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh` (identical to `sft_distill_cot_pretok_8n.sh` per `diff` exit 0) + `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh`.

> **No training launched.** This is a gate check only. Lever is gated on pretok + BASE_MODEL existence (see §5 below).

## 1. Pretokenized gate

| Item | Expected | Found |
|---|---|---|
| **Pretok path** | `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (shared `/lus/tegu`, W1 preflight `dataset_info.json`) | `PRETOK_DIR` default in both SFT scripts + `_pretokenize_*_1n.sh` `OUT_DIR` match exactly. Consumer gates on `dataset_info.json` via `_wait_for_pretok()`; fallback `outputs/sft-pretok/distill-cot-mix-len8192/dataset_info.json`. |
| **Producer** | `_pretokenize_distill_cot_mix_1n.sh` — `train_sft.py --pretokenize_to` → `save_to_disk` | Matches. 1N, `NPROC=1` (12483443 landing), direct `${VENV}/bin/python` (no `ezpz launch` MPI wrapping). Consumer uses `--pretokenized_dataset` with `HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1`; TRL skips prep when `input_ids` present — no HF streaming cache race / 429 / 8h re-tokenize. |
| **Shared cache** | `HF_HOME=/tegu/datasets/datasets/hf`, `EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes` | Set identically in producer and consumer. |

### Recipe hash + decontamination

`torchtitan/experiments/ezpz/rl/datasets_sft.py:1262` defines `distill_cot_mix` as:

```
0.50 OpenThoughts-114k-decontam
0.50 OpenR1-Math-220k-decontam
```

- Both components are built via `_decontaminate_against_gsm8k()` → `decontam_traces.GSM8KDecontaminator` (13-gram + normalized-exact question match) **before** interleave through `_materialized_mix_load_or_build()`.
- Decontam is baked into each component's registered build, therefore captured in the **materialized-mix recipe hash** (component names + `DISTILL_MAX_THINK_CHARS`). A cache hit is guaranteed decontaminated. Per-source `[decontam]` drop counts are logged — verify in pretok run log via `grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log` before trusting eval delta.
- `DISTILL_MAX_THINK_CHARS` (default `16000`, part of recipe) trims frontier traces at format time; changing it forces a fresh mix cache.

> **Assignment wording vs code:** task target cites `50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot`; implemented `distill_cot_mix` uses `OpenR1-Math-220k-decontam` as the second half (not `gsm8k-r1cot`). Intent matches — broad frontier R1 traces replacing the narrow math-only cold-start — and the envelope `<think>/<answer>\boxed{}` and decontam invariants are identical. If `gsm8k-r1cot` weighting is required, define a separate mix-spec; do not mutate this gate.

## 2. BASE_MODEL

**Consumer (`sft_distill_cot_4n.sh:120`):**
```
BASE_MODEL=${BASE_MODEL:-${SUBMIT_DIR}/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf}
```
- Correct deliverable: `checkpoint-900-hf` (IFEval 0.253, pre-collapse), not a fresh CPT-FWE HF. Matches `POST-TRAINING-PLAN.md` deliverables and `docs/production/sft/README.md` ("checkpoint-900-hf is the deliverable").
- Fallback chain is intact: `checkpoint-900-hf` → sharded `checkpoint-900/` (96 `.distcp`, TRL accepts DCP) → legacy `agpt2b-gsm8k-r1cot-2n` (B2 path warning). Missing all three → fatal with `find outputs/sft -maxdepth 1` hint + re-consolidate pointer.
- Override respected: `BASE_MODEL=... qsub ...` (consistent with skill §5 `BASE_MODEL=${BASE_MODEL:-...}` pattern).

**Producer (`_pretokenize_distill_cot_mix_1n.sh:107`):**
```
BASE_MODEL="${HOME}/global_step138650"
```
- Uses `global_step138650` (MDS base) **for tokenizer only** during `pretokenize_to`. Tokenizer is vendored Gemma (`assets/hf/gemma-7b`) and identical across `ckpt-900` lineage, so this does not affect training BASE_MODEL. No mismatch to fix; noted for auditability.

## 3. VENV + LD-first check

| Script | VENV | LD handling | Verdict |
|---|---|---|---|
| `sft_distill_cot_4n.sh:99-103` | `rl-monarch-torch214-locality-20261001` ✓ (`trl 1.13` probe at :112) | `LD_LIBRARY_PATH="${VENV}/lib:..."` only | **Partial** — locality VENV correct (per `sunspot-cpt-hf-pipeline` §5 + `rl/README.md`), but missing the **LD-first cache** (`.cache/xpu-ld-first/<venv>` symlink farm for `libur_*`, `libsycl.so*`, `libumf*`, `libtcm*`) that the pretok job and CPT HF pipeline use. Training will likely still run (XPU libs resolve via `VENV/lib`), but for §5 compliance and parity with `cpt_fineweb_edu_4n.sh` the block at `_pretokenize_*_1n.sh:90-94` should be adopted (no behavior change for this verification; flagged only). |
| `_pretokenize_distill_cot_mix_1n.sh:89-99` | same locality VENV ✓ | Full LD-first cache + `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | **Pass**. Single-rank, no `ezpz_setup_job` MPI wrapping (avoids `torchinfo` circular import segfault). |

**Skill §5 compliance:** VENV identity passes; LD-first in the 4N consumer is advisory, not blocking for this gate.

> Note: both scripts still hardcode `module load oneapi/release/2025.3.1` / `ZE_FLAT` / `CCL_OP_SYNC` headers. Machine-agnostic `ezpz_setup` migration is out of scope for this verification (Golden Rule #0 tracks it separately).

## 4. Guardrail cap

`POST-TRAINING-PLAN.md` guardrail: *cap full-mix SFT at O(1000) steps or drop LR — ckpt-8672 forgot hellaswag 0.593→0.273 / arc_easy 0.694→0.298 when cosine kept LR >1e-5 for ~4000 steps; no single-stage rebuilds (B3/B4/B2v2 0.02–0.105 vs B2 0.205).*

Current consumer defaults (`sft_distill_cot_4n.sh:230,260-267`):
- `NUM_EPOCHS=3`, `learning_rate=2e-5`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `max_length=8192`, `GBS=768@4N / 1536@8N`, no `max_steps`.

**Assessment:** LR matches the two-stage SFT that produced B2, but 3 epochs over the distill pretok (size unknown until `dataset_info.json` reports `num_examples`) can exceed the ~1000-step guardrail. Do **not** rely on epochs alone.

**Required at launch time (not now):**
- Prefer `qsub -v NUM_EPOCHS=1` for a smoke, then inspect eval before resuming; or pass `--max_steps 1000` (add to the `train_sft` invocation) for a hard O(1000)-step cap.
- Alternative: lower LR (e.g. `1e-5`) if the full 3-epoch run is justified by eval headroom.
- Logging is correctly set to `logging_steps 5` (fixes the prior `logging_steps` vs `save_strategy=epoch` single-entry bug cited in Phase 1 audit).

## 5. Gate — do NOT launch training

This lever is **gated** on both:

1. **Pretok ready:** `ls /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json` exists (size via `du -sh`) **and** `[decontam]` counts in the pretok log are non-trivial (confirms decontam ran, not a stale cache).
2. **BASE_MODEL exists:** `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf` present (re-consolidate from `checkpoint-900/` via `accelerate merge` + copy `config.json`/`tokenizer.*` if only sharded exists — see `docs/production/sft/README.md` disk-check note).

If either is missing:
```bash
# Stage 1 — build pretok (1N, proxy ON, shared /tegu)
qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh
# Poll until gated:
ls /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log
# Stage 2 — only after both gates pass:
qsub -l select=5 -v NHOSTS_TRAIN=4 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh        # 4N pack
qsub -l select=9 -v NHOSTS_TRAIN=8 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh        # 8N standalone
# Smoke:
qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
```

**No qsub was issued in this verification. No job scripts were created or modified.**

## 6. Summary of mismatches (informational only)

- **LD-first in 4N consumer:** add the `.cache/xpu-ld-first` symlink-farm block for §5 parity (non-blocking).
- **Mix label:** assignment cites `gsm8k-r1cot` half; code implements `OpenR1-Math-220k-decontam` half — intentional breadth lever, not a bug. Reconcile naming in `POST-TRAINING-PLAN.md` if `gsm8k-r1cot` weighting is desired.
- **Machine-agnostic headers:** still hardcoded oneAPI/CCL/ZE_FLAT; tracked under Golden Rule #0, not this gate.
