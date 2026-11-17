# 10 — Infra Readiness — VENV, LD-first, PBS, Host Routing, Preflight

> **Branch:** `ezpz-cpt111k-eval-sft` HEAD `a6135a79` (2026-11-16).
> **Scope:** Fix infra so Broad-distill / CPT / Eval all launch correctly. One file: VENV map, LD-first fix, PBS filesystems, host routing, preflight gates.
> **Every claim cites `script:line` or job ID.** Inferences labeled `[INFERENCE]`.

---

## 1. VENV map — CPT vs SFT/RL vs Eval (workload → VENV → LD handling → failure if wrong)

Three VENVs, one per workload class. Using the wrong VENV is a deterministic failure — the table below is the contract every script must satisfy.

| # | Workload | VENV path | LD handling | Failure if wrong |
|---|---|---|---|---|
| 1 | **CPT** (`cpt_fineweb_edu_4n.sh` `cpt_olmo25_dolmino75_4n.sh` — Grain `0.2.18` + `torch_remat` streaming on XPU) | `/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16` | **LD-first cache**: `libur_*` `libsycl.so*` `libumf*` `libtcm*` from `${VENV}/lib` are symlinked into `${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")/` and that dir is prepended to `LD_LIBRARY_PATH` so the venv's SYCL/UR wins over the system oneAPI `2025.3.1` / stale `2026.0` on `x1921` nodes. See `cpt_fineweb_edu_4n.sh:54-65`. | If pointed at `frameworks/2026.1.0` — `ModuleNotFoundError: No module named 'torch_remat'` (system framework lacks the fused `w13` remat kernel that `agpt/parallelize.py` needs to avoid padded `Shard(0)`). If LD-first omitted — `undefined symbol: urGraphGetIdExp` or stale UR loader wins on `x1921` nodes. Source: `05-infra-rl-storage.md:49` `cpt_fineweb_edu_4n.sh:52` header "fw-2026.1 lacks remat". |
| 2 | **SFT / RL + pretokenization** (`sft_distill_cot_4n.sh` `_pretokenize_distill_cot_mix_1n.sh` `rl_grpo_b2_2n.sh` `multihost_grpo_e2e.pbs` — TRL + Monarch + TorchStore + vLLM-XPU) | `/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001` | **LD-first cache** (now): same symlink-farm pattern as CPT — `${VENV}/lib/libur_*` etc -> `.cache/xpu-ld-first/rl-monarch-torch214-locality-20261001/` prepended to `LD_LIBRARY_PATH`. See `sft_distill_cot_4n.sh:98-114` (fixed this doc) and `_pretokenize_distill_cot_mix_1n.sh:89-94` (reference). Prior plain `LD_LIBRARY_PATH="${VENV}/lib:..."` also worked on most nodes but lost to stale `2026.0` on `x1921` — LD-first is the canonical hardening. | If replaced by `xpu-torch216-sync-5a7a622f16` — `No module named 'trl'` (CPT venv has Grain but no TRL/Monarch). If bare `xpu-torch214` (no `-locality-20261001` suffix) — UR loader mismatch `libsycl.so.9: UR loader 0.12` `ImportError` and missing `trl` per `POST-TRAINING-2B.md:64-65` wiring fix `40b29ca/d551d04c/927dd1e4f`. Preflight probe `python3 -c "import trl"` at `sft_distill_cot_4n.sh:112` catches this before `ezpz launch`. |
| 3 | **Eval only** (`eval/submit_cot_eval.sh` `eval/eval_cot_cpt_fwe_batch.sh` `eval/consolidate_and_eval_cot.sh` — vLLM `0.26` XPU `float32` CoT harness) | `frameworks/2026.1.0` environment module + `pti-gpu` | `module load frameworks/2026.1.0 pti-gpu` then PTI GPU lib path appended: `/opt/aurora/26.181.0/.../pti-gpu-0.17.0-62kxpbr/lib64` in `LD_LIBRARY_PATH`, plus `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` `ZE_FLAT_DEVICE_HIERARCHY=FLAT` `ZE_AFFINITY_MASK=0`. See `eval/submit_cot_eval.sh:16-21`. | If CPT/SFT/RL pointed at `frameworks/2026.1.0` — no `torch_remat`/`grain`/`trl`/`monarch` (training imports absent). If eval pointed at a training VENV — vLLM harness misconfigured (wrong PTI/CCL stack). Eval itself OOMs on MMLU `compute_topk_scores 63G` on XPU — Aurora-only per `POST-TRAINING-2B.md:25`. Source: `eval/submit_cot_eval.sh:16` `05-infra-rl-storage.md:51`. |

**Invariant:** `frameworks/2026.1.0` is **never a training VENV** — the 2026-11-04 five-commit LD fix established it lacks `torch_remat` and the TRL/Monarch stack; it exists only for the vLLM eval harness on XPU. Source: `05-infra-rl-storage.md:53`.

**Pretokenization shares SFT's VENV** (`rl-monarch-torch214-locality-20261001`) and bypasses `ezpz_setup_job` — it is single-rank, not Monarch-launched, to avoid `ezpz.doctor -> torchinfo -> torch` circular-import segfault (jobs `12483367/393/403` died there, `12483743` + `12483443` landed after bypass). Source: `_pretokenize_distill_cot_mix_1n.sh:77-84,98-101`.

---

## 2. LD-first fix applied — sft_distill_cot_4n.sh now mirrors cpt_fineweb_edu_4n.sh

### What was wrong

`sft_distill_cot_4n.sh:98-103` (pre-fix) had correct VENV identity but weak LD handling:

```bash
# Env bootstrap (Monarch / TRL)
VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001}"
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${VENV}/lib:${LD_LIBRARY_PATH:-}"
```

Single-line `LD_LIBRARY_PATH="${VENV}/lib:..."` prepends the venv lib dir directly. This works on most Sunspot nodes where the oneAPI `2025.3.1` system stack is current, but on `x1921` nodes (stale oneAPI `2026.0` image with older SYCL/UR) the system `LD_LIBRARY_PATH` can shadow `libur_*`/`libsycl.so*` after PBS injects its own paths — the venv's UR loader loses. CPT and the pretokenize producer already hardened against this with the LD-first symlink farm; SFT did not.

`torchtitan/experiments/ezpz/docs/production/sft_distill_cot_4n_verification.md:48-54` (verification gate report, 2026-11-16) flagged this:

> | `sft_distill_cot_4n.sh:99-103` | `rl-monarch-torch214-locality-20261001` OK | `LD_LIBRARY_PATH="${VENV}/lib:..."` only | **Partial — advisory, not blocking** (add LD-first block for section 5 parity) |
> | `_pretokenize_distill_cot_mix_1n.sh:89-99` | same VENV OK | Full LD-first cache + `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` `ZE_FLAT` | **Pass** |

Source: `sft_distill_cot_4n_verification.md:48-54` section "3. VENV + LD-first check" verdict `Partial -> Pass` after this fix.

### Reference pattern (CPT)

`cpt_fineweb_edu_4n.sh:52-65`:

```bash
# CPT VENV: xpu-torch216 with Grain 0.2.18 + torch_remat (verified; fw-2026.1 lacks remat)
VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16}"
# LD-first: venv libs win over system for UR/SYCL (same as data_resume job)
if [[ -d "${VENV}/lib" ]]; then
  _ld_first="${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")"
  if [[ ! -e "${_ld_first}/.complete" ]]; then
    mkdir -p "${_ld_first}"
    for _lib in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
      [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"
    done
    touch "${_ld_first}/.complete"
  fi
  export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
fi
```

The pretokenize producer `_pretokenize_distill_cot_mix_1n.sh:89-94` carries the identical block (single-line form but same four globs + `.complete` sentinel).

### Diff hunk applied

Single file, single hunk, no other logic touched. `bash -n` passes; `_ld_first` grep confirms presence.

```diff
--- a/torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
+++ b/torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
@@ sft_distill_cot_4n.sh:98-103 @@
-# Env bootstrap (Monarch / TRL)
-VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001}"
-export VIRTUAL_ENV="${VENV}"
-export PATH="${VENV}/bin:${PATH}"
-export PYTHONPATH="${SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
-export LD_LIBRARY_PATH="${VENV}/lib:${LD_LIBRARY_PATH:-}"
+# Env bootstrap (Monarch / TRL) — LD-first so venv SYCL/UR wins over stale 2026.0 on x1921 nodes
+VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001}"
+# LD-first: venv libs win over system for UR/SYCL (mirrors cpt_fineweb_edu_4n.sh:54-65)
+if [[ -d "${VENV}/lib" ]]; then
+  _ld_first="${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")"
+  if [[ ! -e "${_ld_first}/.complete" ]]; then
+    mkdir -p "${_ld_first}"
+    for _lib in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
+      [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"
+    done
+    touch "${_ld_first}/.complete"
+  fi
+  export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
+fi
+export VIRTUAL_ENV="${VENV}"
+export PATH="${VENV}/bin:${PATH}"
+export PYTHONPATH="${SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
```

**Verification:**

```bash
bash -n torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh && echo OK
# -> OK (no syntax error; fi/then/done balanced, quotes closed)

grep -n _ld_first torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh
# -> sft_distill_cot_4n.sh:102 _ld_first="${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")"
#    sft_distill_cot_4n.sh:103 if [[ ! -e "${_ld_first}/.complete" ]]; then
#    sft_distill_cot_4n.sh:108 touch "${_ld_first}/.complete"
#    sft_distill_cot_4n.sh:110 export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
```

`VIRTUAL_ENV`/`PATH`/`PYTHONPATH` and all `import trl`/`torch` probes below are unchanged. Checkpoint logic, batch sizing, and `ezpz launch` invocation untouched. Source: `sft_distill_cot_4n.sh:98-114` post-fix.

---

## 3. PBS filesystems=tegu:home everywhere it matters

### Why this value

Sunspot's `/home` and `/lus/tegu` live on the same `1.2P` Lustre pool at `17%` full (`~1P` free), but accounting differs: `/home` holds the **live 15T CPT DCP chain** (`outputs/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64`, `556` dirs `23G` at `111.2k`, `~717` dirs at `143.4k` [INFERENCE: `143400/200`]) and SFT/eval artifacts accumulate there unless routed off. Tegu project space has a separate quota via `/lus/tegu/projects/datascience/foremans` (previously hit `11T` hard cap mid-session per `journal.md:4546-4548`). New HFs and new SFT outputs must default to Tegu to avoid `/home` fill — commit `c519de37a5` repointed every new-output path behind a `[[ -d "$TEGU_ROOT" ]]` guard so local dev still falls back to relative `outputs/`. Source: `05-infra-rl-storage.md:39-46` `02-hf-conversion.md:42-62` `c519de37a5` diff.

PBS on Sunspot **has no `/eagle`** — `filesystems=home:eagle` or `home:flare` is wrong here and will fail to mount Tegu. Source: `SYNTHESIS.md:214` `POST-TRAINING-2B.md:13` sync note, `05-infra-rl-storage.md:47` "PBS filesystems=tegu:home (PBS lacks /eagle on Sunspot)".

`tegu:home` mounts both: read DCP from `/home` (live 15T never moved — "moving mid-train breaks resume" per `hf_batch_convert_cpt_fwe.sh:31` and `cpt_fineweb_edu_4n.sh:39-44` symlink fix), write HF/SFT/evals to Tegu.

### Scripts where tegu:home is required

Every job that reads `/home` DCP or writes Tegu HF/evals must carry it. Audit via `grep -n filesystems torchtitan/experiments/ezpz/scripts/**/*.sh`:

| Script | PBS line | Why tegu:home |
|---|---|---|
| `scripts/cpt_fineweb_edu_4n.sh:5` | `#PBS -l filesystems=tegu:home` | Reads `/home` DCP for resume (`cpt_fineweb_edu_4n.sh:39-44` symlink) + writes Tegu checkpoints when `TEGU_SFT_ROOT` exists (`cpt_fineweb_edu_4n.sh:83-89` `CKPT_DIR` prefers Tegu). Live chain itself stays on `/home` until `190k` then `rsync` to Tegu. Source: `01-cpt-frontier.md:25` `c519de37a5`. |
| `scripts/sft_distill_cot_4n.sh:5` | `#PBS -l filesystems=tegu:home` | Reads `/home` fallback BASE_MODEL (`checkpoint-900-hf` Tegu-first, sharded fallback) + reads `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` pretokenized dataset + writes Tegu SFT checkpoints (`sft_distill_cot_4n.sh:237-239` `CKPT_DIR` prefers `TEGU_SFT_ROOT`). Source: `sft_distill_cot_4n.sh:106-110` shared cache + `228-239` Tegu CKPT_DIR. |
| `scripts/eval/hf_batch_convert_cpt_fwe.sh:6` | `#PBS -l filesystems=tegu:home` | Reads DCP from `/home` (15T live chain, `DCP_ROOT=$REPO/outputs/checkpoints/...` at `hf_batch_convert_cpt_fwe.sh:32`) and writes HF to Tegu (`HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe` at `hf_batch_convert_cpt_fwe.sh:35-36`). PBS `tegu:home` covers both mounts. Source: `02-hf-conversion.md:21` `hf_batch_convert_cpt_fwe.sh:30-36`. |
| `scripts/eval/submit_cot_eval.sh:5` | `#PBS -l filesystems=tegu:home` | Reads HF from Tegu (`CKPT=$REPO/outputs/evals/cpt-fwe/...` or `$TEGU_ROOT/...`) and writes eval JSONL to Tegu (`OUT=$REPO/outputs/evals/cot/...`); also needs `/home` for `assets/hf/gemma-7b` tokenizer fallback on convert path. Source: `submit_cot_eval.sh:12-27`. |
| `scripts/eval/consolidate_and_eval_cot.sh:5` | `#PBS -l filesystems=tegu:home` | Consolidates DCP -> HF (reads `/home` DCP, writes Tegu HF) then evals in one job. Same mount requirement as `hf_batch_convert`. |
| `scripts/eval/eval_cot_cpt_fwe_batch.sh:5` | `#PBS -l filesystems=tegu:home` | Batch CoT eval sweep over Tegu HFs — reads Tegu HFs, writes Tegu eval JSONLs. |
| `rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh:4` | `#PBS -l filesystems=tegu:home` | Reads raw HF from `/tegu/datasets/datasets/hf` staging + writes pretokenized output to `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (shared datasets quota, not `~/.cache`). Needs both mounts for `HF_HOME` + `EZPZ_SFT_MIX_CACHE_DIR` under `/tegu/datasets`. Source: `_pretokenize_distill_cot_mix_1n.sh:64-71`. |

**Not required:** Aurora production jobs (`submit_agpt_20b_aurora_venv_failover.sh` etc.) use `filesystems=home:flare` (`/flare/AuroraGPT/foremans/` not Tegu — Tegu is Sunspot-only). Polaris uses `filesystems=home:eagle`. On Sunspot, never use those — Sunspot's PBS rejects `flare`/`eagle`. Source: `05-infra-rl-storage.md:47-48` + header comments in `cpt_olmo25_dolmino75_4n.sh:29`.

---

## 4. Host routing — Sunspot (XPU 12 tiles/node) vs Aurora (20B prod) vs local

### Sunspot — XPU, 12 tiles per node

Sunspot nodes expose **12 XPU tiles per node** (6 PVC GPUs x 2 tiles). PBS `select` counts nodes; `nproc` counts tiles. All Sunspot jobs must translate correctly:

| Script | PBS select | PPN (tiles/node) | Active GPUs | GBS | Walltime | Queue |
|---|---|---|---|---|---|---|
| `cpt_fineweb_edu_4n.sh:6,72-76` | `select=4` (`4` nodes) | `PPN=8` (proven at `12483025`, smaller than 12 to keep `hidden_dim 11008 % PPN == 0` and avoid padded-shard OOM `w13 (2,14336,5120)` with `14336=2^11*7` no factor of 3) | `4*8=32` | `64` (`32*2/1`, `LBS=2 GAS=1 seq 8192 -> 524288 tok/step -> 190734 steps`) | `12:00:00` (`workq` up to `23h`; was `02:00` churn at `~4k/hop`, now `~24k/hop` after `99e918b342`) | `workq` `A=datascience` |
| `sft_distill_cot_4n.sh:6,232-242` | `select=5` (`4N active + 1 spare` via `--spare-nodes auto`) | `PPN=12` (`NGPU_PER_HOST` default `12`, full node) | `4*12=48` | `768` (`48*2*8`, `LBS=2 GAS=8 seq 8192 packing`) | `12:00:00` (headroom; actual compute `~1-1.5h`) | `workq` `A=datascience` |
| `sft_distill_cot_4n.sh` standalone 8N override | `select=9` (`8N active + 1 spare`, `qsub -l select=9 -v NHOSTS_TRAIN=8`) | `12` | `8*12=96` | `1536` (`96*2*8`) | `12:00:00` | `workq` |
| `_pretokenize_distill_cot_mix_1n.sh:5` | `select=1` (`1` node) | `1` rank (single-rank, no `ezpz_setup_job` MPI) | `1` | N/A (data job, `NPROC=1` tokenize) | `12:00:00` | `workq` |
| `eval/submit_cot_eval.sh:4` | `select=1` (`1` node, single tile) | `1` tile (`ZE_AFFINITY_MASK=0`) | `1` | N/A (vLLM `float32` single-GPU) | `00:40:00` | `workq` |
| `eval/hf_batch_convert_cpt_fwe.sh:5` | `select=1` (`1` node) | `1` rank (single-rank convert) | `1` | N/A | `02:00:00` | `workq` |

**Select calculation (SFT):** `select = NHOSTS_TRAIN + 1 spare`. Header `select=5` is the 4N pack value; override at submit for 8N as `select=9`. If `NHOSTS_TRAIN` not set, script derives it as `PBS_SELECT - 1` (`sft_distill_cot_4n.sh:222-231`), else defaults to `4`. Source: `sft_distill_cot_4n.sh:57-67,221-233`.

**CPT uses fewer tiles:** `PPN=8` not `12` because `hidden_dim 11008 % 32 == 0` but `w13` shape `(2, 14336, 5120)` with `14336 = 2^11 * 7` has no factor of 3 — shard degree `12` would require padding and OOM (`17.5 GiB/rank padded` at `PPN=12`). `PPN=8` at `32` ranks is proven safe (`F0` job `12483025` `19.3%` MFU). SFT can use full `12` because it shards only `2B` weights under TRL `fsdp full_shard` with `per_device_train_batch_size 2`, not the large CPT activation regime. Source: `cpt_fineweb_edu_4n.sh:9-10` `05-infra-rl-storage.md:49`.

**Workq budget:** Sunspot `workq` max `128` nodes. This lane budgets `16N` active across 4 concurrent jobs so it packs cleanly:

```
4N (CPT-FWE) + 4N (CPT-olmo25/dolmino75) + 4N (SFT broad-distill) + 2N (GRPO) + 2N (test.sh) = 16N active
```

Header comments in `cpt_fineweb_edu_4n.sh:13`, `sft_distill_cot_4n.sh:57-60,246` and `sft_distill_cot_4n_verification.md:60` all cite this pack. `128N` max means the 16N pack uses `12.5%` — safe for parallel submission without queue contention. For a standalone 8N SFT run (`select=9`), the pack temporarily grows to `20N` (8+4+4+2+2) — still under `128N` but document as sequential if other lanes are active. Source: `sft_distill_cot_4n.sh:58-60` "workq max 128N, this lane budgets ~16N".

**Sunspot XPU device routing (all XPU jobs):**

```bash
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

CPT/SFT also set `CCL_PROCESS_LAUNCHER=pmix` `CCL_OP_SYNC=1`; eval additionally sets `ZE_AFFINITY_MASK=0` for single-tile vLLM. The pretokenize job sets the same two `ONEAPI`/`ZE_FLAT` lines outside `ezpz_setup_job` because it bypasses MPI wrapping (`_pretokenize_distill_cot_mix_1n.sh:98-99`). Source: `cpt_fineweb_edu_4n.sh:26-27` `sft_distill_cot_4n.sh:70-74` `eval/submit_cot_eval.sh:18-20` `_pretokenize_distill_cot_mix_1n.sh:98-99`.

### Aurora — 20B production (not this lane)

Aurora production (20B `agpt` MoE) is the only workload that runs on Aurora, via `submit_agpt_20b_aurora_venv_failover.sh` and `submit_agpt_20b_aurora_venv.sh` with `filesystems=home:flare` and `A=AuroraGPT` (not `datascience`). It uses Aurora's `/flare/AuroraGPT/foremans/` project space, not Tegu. This lane does not submit to Aurora; the CPT/SFT/eval scripts listed above are Sunspot-only (`workq` `A=datascience`). If a true 25/75 dolmino blend is needed, that explicitly requires `qsub -A AuroraGPT -q prod -l filesystems=flare:home -l select=5` per `cpt_olmo25_dolmino75_4n.sh:29,109`. Source: `cpt_olmo25_dolmino75_4n.sh:29`.

### Local (dev / CI)

On a local checkout (no PBS, no Tegu), every script falls back to relative `outputs/`:

- `cpt_fineweb_edu_4n.sh:83-89`: if `TEGU_ROOT` not a directory, `CKPT_DIR` resolves to `${SUBMIT_DIR}/outputs/checkpoints/...`
- `sft_distill_cot_4n.sh:237-239`: if `TEGU_SFT_ROOT` not a directory, `_SFT_ROOT` is `${SUBMIT_DIR}/outputs/sft`
- `hf_batch_convert_cpt_fwe.sh:35-36`: if `TEGU_ROOT` not present, `HF_ROOT` is `$REPO/outputs/evals/cpt-fwe`
- `eval/submit_cot_eval.sh:12,25-27`: `REPO` defaults to Tegu path but falls back to `$HOME/torchtitan` if script not found at Tegu

`bash -n` on any script must pass locally without Tegu mounted — no hard dependency on `/lus/tegu` at parse time. Source: the `[[ -d ... ]]` guards in each script above.

---

## 5. Preflight checklist — 5 commands that prove readiness (python -c imports + dataset_info.json + BASE_MODEL dir)

Run these **on Sunspot login** before any `qsub`. All five must pass; if any fails, do not launch training — fix the gate first. Adapted from `sft_distill_cot_4n_verification.md:72-91` section 5 "Gate — do NOT launch training" and `04-sft-broad-distill.md:179-213` gate checks.

### 1. CPT VENV imports — `torch_remat` + `grain` present

```bash
/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16/bin/python -c "import torch_remat, grain; print('CPT OK:', grain.__version__)"
# Expected: CPT OK: 0.2.18 (or similar)
# Failure: ModuleNotFoundError: No module named 'torch_remat' -> wrong VENV (you pointed at frameworks/2026.1.0 or rl-monarch)
```

Source: `cpt_fineweb_edu_4n.sh:70` probe `python3 -c "import torch_remat, grain; print('CPT venv ok:', grain.__version__)"`.

### 2. SFT/RL VENV imports — `trl` + `torch` XPU present

```bash
/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001/bin/python -c "import trl; import torch; print('SFT OK: trl', trl.__version__, 'torch', torch.__version__, 'xpu', torch.xpu.is_available())"
# Expected: SFT OK: trl 0.13.x torch 2.6.x xpu True  (trl >= 0.12 required; xpu True on XPU login, False on login is OK — XPU availability is compute-only)
# Failure: No module named 'trl' -> bare xpu-torch214 or xpu-torch216; UR loader 0.12 ImportError -> bare xpu-torch214 non-locality
```

Source: `sft_distill_cot_4n.sh:112-113` probes `python3 -c "import trl; print('trl', trl.__version__)"` and `python3 -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available())"`, plus `_pretokenize_distill_cot_mix_1n.sh:101` same `import trl` probe.

For the script's own LD-first path, also verify the symlink farm was created on a prior run:

```bash
ls -l .cache/xpu-ld-first/rl-monarch-torch214-locality-20261001/.complete 2>&1 | head
# Expected after one SFT or pretokenize run: .complete exists; if missing, the next job will create it (not a blocker pre-qsub)
```

Source: `sft_distill_cot_4n.sh:102-110` LD-first block creates `.cache/xpu-ld-first/<venv>/.complete`.

### 3. Pretokenized dataset — `dataset_info.json` exists and is the right artifact

```bash
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json && \
du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192 && \
ls /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/data-*.arrow 2>&1 | wc -l
# Expected: dataset_info.json present, ~3.4G, 8 x data-*.arrow, 91,481 raw -> 36,934 packed @ 8192
# Failure: No such file -> pretok not landed; rebuild with qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh

grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log 2>&1 | tail -5
# Expected: non-trivial [decontam] drop counts per component (13-gram + normalized-exact, per _pretokenize_distill_cot_mix_1n.sh:27-35)
# Failure: no [decontam] lines -> stale cache hit without decontam; re-stage at new recipe hash
```

Source: `sft_distill_cot_4n.sh:152-167` consumer gates on `dataset_info.json` via `_wait_for_pretok()`, `sft_distill_cot_4n_verification.md:10-14` pretokenized gate, `04-sft-broad-distill.md:29-35` verify block.

### 4. BASE_MODEL — `checkpoint-900-hf` exists (re-consolidate hint if only sharded)

```bash
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900{,-hf}/ 2>&1 | head -20
# Expected: checkpoint-900-hf/ exists with config.json + tokenizer.* + model-*.safetensors (IFEval 0.253, pre-collapse deliverable)
# Fallback: if only checkpoint-900/ (96 .distcp shards) exists, re-consolidate:
#   accelerate merge-weights + copy config.json/tokenizer.* per docs/production/sft/README.md
# Failure: neither exists -> sft_distill_cot_4n.sh:140-146 fatal with find outputs/sft -maxdepth 1 hint

ls -lh ~/global_step138650/config.json ~/global_step138650/tokenizer.json 2>&1 | head
# Expected: present (pretokenize tokenizer base, vendored Gemma assets/hf/gemma-7b compatible — no mismatch with checkpoint-900-hf)
# Info only: raw MDS base for E5 control; not required for E1/E2 consumer
```

Source: `sft_distill_cot_4n.sh:115-146` BASE_MODEL fallback chain `checkpoint-900-hf -> checkpoint-900/ (96 .distcp) -> legacy agpt2b-gsm8k path`, `sft_distill_cot_4n_verification.md:32-45` BASE_MODEL audit, `04-sft-broad-distill.md:200-211` BASE_MODEL gate.

### 5. PBS + eval VENV — `frameworks/2026.1.0` + PTI + qstat sanity

```bash
module load frameworks/2026.1.0 pti-gpu 2>&1 | head && \
ls /opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-*/lib64/libpti* 2>&1 | head -3 && \
echo "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-unset} ZE_FLAT=${ZE_FLAT_DEVICE_HIERARCHY:-unset}"
# Expected: module load succeeds, libpti* present, then after submit_cot_eval.sh sets ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT=FLAT
# Failure: module not found -> wrong host (eval module is Sunspot/Aurora only, not local); build eval elsewhere

qstat -Q workq 2>&1 | head -5; echo "---"; qstat -u $USER 2>&1 | head -20
# Expected: workq up, no queue full; any R jobs show expected select (4 for CPT, 5/9 for SFT, 1 for eval/HF)
# Info: Sunspot workq max 128N, this lane budgets 16N active; large queue depth is normal due to shared allocation
```

Source: `eval/submit_cot_eval.sh:16-22` PTI LD + `ONEAPI_DEVICE_SELECTOR` + `ZE_FLAT` + `ZE_AFFINITY_MASK`, `05-infra-rl-storage.md:41-42` eval harness section.

---

**Gate rule:** If checks 3 or 4 fail, do not `qsub` the 4N consumer — it will wait via `_wait_for_pretok()` sleep loop and waste allocation, or fatal on missing BASE_MODEL. Rebuild stage 1 (`qsub _pretokenize_distill_cot_mix_1n.sh`) or re-consolidate `checkpoint-900-hf` first, then re-run this checklist. Source: `sft_distill_cot_4n.sh:156-219` two-stage wire + `sft_distill_cot_4n_verification.md:79-91` "do NOT launch training" gate.

---

*Doc produced 2026-11-16 on `ezpz-cpt111k-eval-sft` @ `a6135a79`. Companion `sft_distill_cot_4n.sh` LD-first fix verified `bash -n OK` + `grep _ld_first` hit.*
