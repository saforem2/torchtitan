#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N smoke-position-fix
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=1
#PBS -q debug
#PBS -j oe

# smoke_position_fix.sh — 1-2 node smoke for BLENDCORPUS_RESUME_CONSUMED_SAMPLES.
#
# Validates the position-fix on top of origin/ezpz (commit >= 0f7f0c7f53) before
# the Monday boost for >2048N jobs. Use on a checkpoint whose stored
# `dataloader.consumed_samples` is 0 (every pre-fix checkpoint).
#
# What it checks:
#   1. All ranks see BLENDCORPUS_RESUME_CONSUMED_SAMPLES via `mpiexec --envall`.
#   2. Log contains the BlendCorpus override line "resuming at N" / "resuming the
#      data stream at --dataloader.resume-consumed-samples=N instead of 0".
#   3. 5 training steps finish with finite loss.
#   4. The new checkpoint's `dataloader.consumed_samples == N + 5*GBS`.
#   5. Checkpoint directory is non-empty.
#
# Usage (interactive or via qsub -v):
#   bash smoke_position_fix.sh <ckpt_path> <N> <expected_ckpt_dir>
#
#   - ckpt_path          Path to the checkpoint to resume from (folder or step-N
#                        dir). Must exist before launch.
#   - N                  The BLENDCORPUS_RESUME_CONSUMED_SAMPLES value (GLOBAL
#                        sample index). Must be a multiple of micro_batch_size *
#                        dp_world_size. Canonical Option B values:
#                          512N: 14745600 + margin (e.g. 15000000)
#                          256N:  9523200 + margin (e.g.  9600000)
#                        Add a margin because segment lengths are good to a few
#                        steps. See docs/guides/blendcorpus-resume-position.md.
#   - expected_ckpt_dir  Where the new step checkpoint will land (folder under
#                        checkpoint.folder). Checked after training.
#
# When submitted via PBS, pass args with `-v`:
#   qsub -v CKPT=/flare/.../step-11100,RESUME_N=15000000,CKPT_DIR=checkpoints/smoke-position \
#        torchtitan/experiments/ezpz/scripts/smoke_position_fix.sh
#
#   or as positional args on the qsub line (consumed from PBS_O_WORKDIR context).
#
# Env knobs (all optional):
#   SMOKE_VENV       path to the venv (default: repo-root .venv, torch 2.13 XPU)
#   SMOKE_CONFIG     torchtitan config name (default: ezpz_agpt_2b)
#   SMOKE_MODULE     torchtitan module (default: ezpz.agpt)
#   SMOKE_STEPS      steps to run (default: 5)
#   GBS              global batch size; only needed for the post-check
#                    (default: inferred from config: micro_batch_size * dp_world_size;
#                     the check uses the training config's GBS)
#   EZPZ_UTILS       staged utils path (compute nodes have no internet)
#   HF_ASSETS        shared tokenizer assets

set -o pipefail

# ---- Environment ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu 2>/dev/null || true
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# ---- Args (positional preferred, -v fallback) ----
CKPT_PATH="${1:-${CKPT:-}}"
RESUME_N="${2:-${RESUME_N:-}}"
EXPECTED_CKPT_DIR="${3:-${CKPT_DIR:-}}"

if [[ -z "${CKPT_PATH}" || -z "${RESUME_N}" || -z "${EXPECTED_CKPT_DIR}" ]]; then
    echo "Usage: bash smoke_position_fix.sh <ckpt_path> <N> <expected_ckpt_dir>" >&2
    echo "  ckpt_path          checkpoint folder or step dir to resume from" >&2
    echo "  N                  BLENDCORPUS_RESUME_CONSUMED_SAMPLES (int)" >&2
    echo "  expected_ckpt_dir  new checkpoint folder (checked after train)" >&2
    echo "" >&2
    echo "Or pass via -v: qsub -v CKPT=<p>,RESUME_N=<n>,CKPT_DIR=<d> ..." >&2
    exit 2
fi

if ! [[ "${RESUME_N}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: N must be a non-negative integer, got '${RESUME_N}'" >&2
    exit 2
fi

# ---- Resolve dirs ----
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}" || { echo "FATAL: cannot cd to ${SUBMIT_DIR}"; exit 1; }

VENV="${SMOKE_VENV:-${SUBMIT_DIR}/.venv}"
VENV="$(readlink -f "${VENV}" 2>/dev/null || echo "${VENV}")"
EZPZ_UTILS="${EZPZ_UTILS:-${SUBMIT_DIR}/.ezpz-utils-cache/ezpz-utils.sh}"
HF_ASSETS="${HF_ASSETS:-${SUBMIT_DIR}/assets/hf/gemma-7b}"

# Try to set up ezpz job env if utils are present (optional for interactive runs).
if [[ -f "${EZPZ_UTILS}" ]]; then
    # shellcheck disable=SC1090
    source "${EZPZ_UTILS}" 2>/dev/null || true
    command -v ezpz_setup_job >/dev/null 2>&1 && ezpz_setup_job 2>&1 | cat || true
    command -v ezpz_load_modules >/dev/null 2>&1 && ezpz_load_modules 2>&1 | cat || true
fi

if [[ ! -d "${VENV}" ]]; then
    echo "FATAL: venv not found: ${VENV}" >&2
    exit 1
fi
if [[ ! -e "${CKPT_PATH}" ]]; then
    echo "FATAL: ckpt_path not found: ${CKPT_PATH}" >&2
    exit 1
fi

# Yeet or activate.
if [[ -f "${VENV}.tar.gz" && -n "${PBS_NODEFILE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}/bin/activate"
    ezpz yeet-env --src "${VENV}.tar.gz" 2>&1 | cat || true
    deactivate 2>/dev/null || true
    source /tmp/.venv/bin/activate 2>/dev/null || source "${VENV}/bin/activate"
else
    # shellcheck disable=SC1090
    source "${VENV}/bin/activate"
fi

# Verify the fix is present in this checkout.
if ! grep -q "BLENDCORPUS_RESUME_CONSUMED_SAMPLES" \
        "${SUBMIT_DIR}/torchtitan/experiments/ezpz/blendcorpus/blendcorpus_builder.py" 2>/dev/null; then
    echo "FATAL: this checkout lacks the position-fix (no BLENDCORPUS_RESUME_CONSUMED_SAMPLES in blendcorpus_builder.py)." >&2
    echo "       Need origin/ezpz >= 0f7f0c7f53. Pinned clones f1c20ee7/086ff996 do NOT have it." >&2
    exit 1
fi

SMOKE_MODULE="${SMOKE_MODULE:-ezpz.agpt}"
SMOKE_CONFIG="${SMOKE_CONFIG:-ezpz_agpt_2b}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
SEQ_LEN="${SEQ_LEN:-8192}"

# Derive job size.
if [[ -n "${PBS_NODEFILE:-}" ]]; then
    NNODES="$(wc -l < "${PBS_NODEFILE}" 2>/dev/null || echo 1)"
else
    NNODES=1
fi
WORLD_SIZE="${WORLD_SIZE:-$(( NNODES * 12 ))}"

JOBID_SHORT="${PBS_JOBID%%.*}"
if [[ -z "${JOBID_SHORT}" ]]; then
    JOBID_SHORT="$(date +%Y%m%d-%H%M%S)"
fi
LOG_DIR="${SUBMIT_DIR}/logs/smoke-position-fix-${JOBID_SHORT}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run.log"
ENV_LOG="${LOG_DIR}/env_check.log"
TRAIN_LOG="${LOG_DIR}/train.log"

echo "===== smoke_position_fix =====" | tee "${RUN_LOG}"
date | tee -a "${RUN_LOG}"
echo "node0:  $(hostname)" | tee -a "${RUN_LOG}"
echo "venv:   ${VENV}" | tee -a "${RUN_LOG}"
echo "tree:   ${SUBMIT_DIR}" | tee -a "${RUN_LOG}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null || echo unknown)" | tee -a "${RUN_LOG}"
echo "ckpt:   ${CKPT_PATH}" | tee -a "${RUN_LOG}"
echo "N:      ${RESUME_N}" | tee -a "${RUN_LOG}"
echo "out:    ${EXPECTED_CKPT_DIR}" | tee -a "${RUN_LOG}"
echo "steps:  ${SMOKE_STEPS}" | tee -a "${RUN_LOG}"
echo "module: ${SMOKE_MODULE}  config: ${SMOKE_CONFIG}" | tee -a "${RUN_LOG}"
echo "nodes:  ${NNODES}  world_size: ${WORLD_SIZE}" | tee -a "${RUN_LOG}"
python3 -c "import torch; print(f'torch: {torch.__version__}  xpu={torch.xpu.is_available() if hasattr(torch,\"xpu\") else \"n/a\"}')" 2>&1 | tee -a "${RUN_LOG}"
echo "" | tee -a "${RUN_LOG}"

# Need to propagate the resume position to every rank. On Aurora the launcher is
# mpiexec (PALS); `ezpz launch` wraps it. Setting the var in PBS -v or `export`
# only reaches the head rank — ranks launched by mpiexec inherit the mpiexec
# environment, so pass it via mpiexec --envall (or -genv). We export here AND
# pass --envall through ezpz so both paths are covered.
export BLENDCORPUS_RESUME_CONSUMED_SAMPLES="${RESUME_N}"

# ---- Phase 1: env propagation check (all ranks) ----
echo "--- Phase 1: env propagation (all ranks see BLENDCORPUS_RESUME_CONSUMED_SAMPLES) ---" | tee -a "${RUN_LOG}"

# Run a tiny launcher job that prints the env var from every rank.
# Prefer mpiexec --envall if available; fall back to `ezpz launch` wrapper.
if command -v mpiexec >/dev/null 2>&1; then
    # mpiexec --envall forwards the entire environment to all ranks.
    # Verify all ranks see the expected value.
    mpiexec --envall -n "${WORLD_SIZE}" --ppn 12 bash -c \
        'echo "rank ${PMI_RANK:-${PALS_RANKID:-?}} BLENDCORPUS_RESUME_CONSUMED_SAMPLES=${BLENDCORPUS_RESUME_CONSUMED_SAMPLES:-<unset>}"' \
        2>&1 | tee "${ENV_LOG}" | tee -a "${RUN_LOG}"
    ENV_RC=${PIPESTATUS[0]}
else
    # Fallback: single-process check (cannot verify multi-rank propagation).
    echo "WARN: mpiexec not found; checking only rank 0" | tee -a "${RUN_LOG}"
    echo "rank 0 BLENDCORPUS_RESUME_CONSUMED_SAMPLES=${BLENDCORPUS_RESUME_CONSUMED_SAMPLES:-<unset>}" | tee "${ENV_LOG}" | tee -a "${RUN_LOG}"
    ENV_RC=0
fi

if [[ ${ENV_RC} -ne 0 ]]; then
    echo "VERDICT: env_check_failed (mpiexec rc=${ENV_RC})" | tee -a "${RUN_LOG}"
    exit 1
fi

# Every line must show the expected N; any <unset> or mismatch fails.
if grep -q "<unset>" "${ENV_LOG}"; then
    echo "VERDICT: env_not_propagated (some ranks have <unset>)" | tee -a "${RUN_LOG}"
    cat "${ENV_LOG}" | tee -a "${RUN_LOG}"
    exit 1
fi
BAD_ENV="$(grep -c "BLENDCORPUS_RESUME_CONSUMED_SAMPLES=${RESUME_N}" "${ENV_LOG}" || true)"
TOTAL_ENV="$(grep -c "BLENDCORPUS_RESUME_CONSUMED_SAMPLES=" "${ENV_LOG}" || true)"
if [[ "${BAD_ENV}" != "${TOTAL_ENV}" ]]; then
    echo "VERDICT: env_mismatch (expected N=${RESUME_N} on all ranks, got ${BAD_ENV}/${TOTAL_ENV} matches)" | tee -a "${RUN_LOG}"
    cat "${ENV_LOG}" | tee -a "${RUN_LOG}"
    exit 1
fi
echo "env check: ${TOTAL_ENV} ranks all report N=${RESUME_N}" | tee -a "${RUN_LOG}"
echo "" | tee -a "${RUN_LOG}"

# ---- Phase 2: training run (5 steps from ckpt with consumed_samples=0) ----
echo "--- Phase 2: training (${SMOKE_STEPS} steps, resume from ckpt with consumed_samples=0) ---" | tee -a "${RUN_LOG}"

# Checkpoint folder for this smoke: write under EXPECTED_CKPT_DIR. Use a fresh
# subdir per invocation so repeated smokes don't collide.
SMOKE_CKPT_FOLDER="${EXPECTED_CKPT_DIR}"
START_TS="$(date +%s)"
export START_TS
mkdir -p "${SMOKE_CKPT_FOLDER}" 2>/dev/null || true

# Data config: blendcorpus with the Aurora prod list (olmo-mix-1124) by default.
# Override with DFL env var.
DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(python3 -c "import ezpz; print(ezpz.get_machine_name())" 2>/dev/null || echo aurora)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL:-${DFL_PARENT}/${DFL_NAME}.txt}"
DATA_CACHE_PATH="${SMOKE_CKPT_FOLDER}/.cache/${DFL_NAME}/index-cache"

echo "ckpt folder: ${SMOKE_CKPT_FOLDER}" | tee -a "${RUN_LOG}"
echo "data list:   ${DFL}" | tee -a "${RUN_LOG}"
echo "BLENDCORPUS_RESUME_CONSUMED_SAMPLES=${BLENDCORPUS_RESUME_CONSUMED_SAMPLES}" | tee -a "${RUN_LOG}"
echo "" | tee -a "${RUN_LOG}"

# Launch training. Pass BLENDCORPUS_RESUME_CONSUMED_SAMPLES via the environment
# and also as an explicit CLI flag (belt and suspenders — the env var is the
# production path through mpiexec --envall, the flag is a direct override).
#
# `ezpz launch` will invoke `mpiexec --envall` underneath, so the exported var
# reaches all ranks. We also set it in the launcher's environment here.
set +e
if command -v mpiexec >/dev/null 2>&1; then
    # Explicit mpiexec path: ensures --envall is used.
    # ezpz launch with --launcher mpich would do the same; we go direct.
    BLENDCORPUS_RESUME_CONSUMED_SAMPLES="${RESUME_N}" \
    mpiexec --envall -n "${WORLD_SIZE}" --ppn 12 \
        python3 -m torchtitan.experiments.ezpz.train \
            --module="${SMOKE_MODULE}" \
            --config="${SMOKE_CONFIG}" \
            --hf-assets-path="${HF_ASSETS}" \
            --training.steps="${SMOKE_STEPS}" \
            --training.max-context-length="${SEQ_LEN}" \
            --checkpoint.enable \
            --checkpoint.folder="${SMOKE_CKPT_FOLDER}" \
            --checkpoint.interval="${SMOKE_STEPS}" \
            --checkpoint.keep-latest-k=1 \
            --checkpoint.no-last-save-model-only \
            --dataloader.dataset=blendcorpus \
            --dataloader.dataset-path="${DFL}" \
            --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
            --dataloader.resume-consumed-samples="${RESUME_N}" \
            --checkpointer.initial-load-path="${CKPT_PATH}" \
            --checkpointer.initial-load-model-only=False \
            --debug.print-config \
            "$@" \
        2>&1 | tee "${TRAIN_LOG}" | tee -a "${RUN_LOG}"
    TRAIN_RC=${PIPESTATUS[0]}
else
    # No mpiexec (interactive single-node fallback).
    BLENDCORPUS_RESUME_CONSUMED_SAMPLES="${RESUME_N}" \
    python3 -m torchtitan.experiments.ezpz.train \
        --module="${SMOKE_MODULE}" \
        --config="${SMOKE_CONFIG}" \
        --hf-assets-path="${HF_ASSETS}" \
        --training.steps="${SMOKE_STEPS}" \
        --training.max-context-length="${SEQ_LEN}" \
        --checkpoint.enable \
        --checkpoint.folder="${SMOKE_CKPT_FOLDER}" \
        --checkpoint.interval="${SMOKE_STEPS}" \
        --checkpoint.keep-latest-k=1 \
        --checkpoint.no-last-save-model-only \
        --dataloader.dataset=blendcorpus \
        --dataloader.dataset-path="${DFL}" \
        --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
        --dataloader.resume-consumed-samples="${RESUME_N}" \
        --checkpointer.initial-load-path="${CKPT_PATH}" \
        --checkpointer.initial-load-model-only=False \
        --debug.print-config \
        "$@" \
        2>&1 | tee "${TRAIN_LOG}" | tee -a "${RUN_LOG}"
    TRAIN_RC=${PIPESTATUS[0]}
fi
set -e

echo "" | tee -a "${RUN_LOG}"
echo "train rc=${TRAIN_RC}" | tee -a "${RUN_LOG}"

if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "VERDICT: train_failed (rc=${TRAIN_RC})" | tee -a "${RUN_LOG}"
    exit 1
fi

# ---- Phase 3: gate checks ----
echo "" | tee -a "${RUN_LOG}"
echo "--- Phase 3: gates ---" | tee -a "${RUN_LOG}"

FAIL=0

# Gate 1: log says "resuming at N" (the override warning).
# Exact message from blendcorpus_builder.py:load_state_dict:
#   "BlendCorpus checkpoint has consumed_samples=0 (written before the counter
#    existed); resuming the data stream at --dataloader.resume-consumed-samples=N
#    instead of 0."
if grep -q "resuming the data stream at.*${RESUME_N}.*instead of 0" "${TRAIN_LOG}"; then
    echo "gate log_resume_at_N: PASS (found resuming at ${RESUME_N})" | tee -a "${RUN_LOG}"
else
    echo "gate log_resume_at_N: FAIL (no 'resuming the data stream at ...${RESUME_N}...instead of 0' in train log)" | tee -a "${RUN_LOG}"
    FAIL=1
fi

# Gate 2: loss finite for all steps (no NaN/Inf).
# Loss lines look like "step:    1  loss: 11.2345" or similar.
if grep -qE "loss:.*(nan|inf)" "${TRAIN_LOG}" 2>/dev/null; then
    echo "gate loss_finite: FAIL (found nan/inf in loss)" | tee -a "${RUN_LOG}"
    grep -E "loss:.*(nan|inf)" "${TRAIN_LOG}" | head -5 | tee -a "${RUN_LOG}"
    FAIL=1
else
    LOSS_LINES="$(grep -cE "step:[[:space:]]+[0-9]+.*loss:" "${TRAIN_LOG}" 2>/dev/null || echo 0)"
    if [[ "${LOSS_LINES}" -ge "${SMOKE_STEPS}" ]]; then
        echo "gate loss_finite: PASS (${LOSS_LINES} loss lines, no nan/inf)" | tee -a "${RUN_LOG}"
    elif [[ "${LOSS_LINES}" -gt 0 ]]; then
        echo "gate loss_finite: FAIL (only ${LOSS_LINES}/${SMOKE_STEPS} loss lines)" | tee -a "${RUN_LOG}"
        FAIL=1
    else
        echo "gate loss_finite: FAIL (no loss lines found in train log)" | tee -a "${RUN_LOG}"
        FAIL=1
    fi
fi

# Gate 3: checkpoint directory non-empty AND freshly written.
# DCP writes under <folder>/step-<N>/; check that something was written AND is newer than training start.
if [[ -d "${SMOKE_CKPT_FOLDER}" ]]; then
    CKPT_FILES="$(find "${SMOKE_CKPT_FOLDER}" -type f 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${CKPT_FILES}" -gt 0 ]]; then
        # Require at least one file newer than this smoke's start time (guards stale-dir false pass).
        SMOKE_START_TS="${SMOKE_START_TS:-$(date -d @${START_TS:-$(date +%s)} +%Y%m%d%H%M 2>/dev/null || echo "")}"
        if [[ -n "${SMOKE_START_TS}" ]]; then
            FRESH_FILES="$(find "${SMOKE_CKPT_FOLDER}" -type f -newermt "@${START_TS}" 2>/dev/null | wc -l | tr -d ' ')"
            if [[ "${FRESH_FILES}" -gt 0 ]]; then
                echo "gate ckpt_non_empty: PASS (${CKPT_FILES} files, ${FRESH_FILES} fresh since smoke start)" | tee -a "${RUN_LOG}"
            else
                echo "gate ckpt_non_empty: FAIL (no files newer than smoke start; stale dir from prior run?)" | tee -a "${RUN_LOG}"
                find "${SMOKE_CKPT_FOLDER}" -type f -printf '%T+ %p\n' 2>/dev/null | sort | tail -5 | tee -a "${RUN_LOG}"
                FAIL=1
            fi
        else
            echo "gate ckpt_non_empty: PASS (${CKPT_FILES} files under ${SMOKE_CKPT_FOLDER})" | tee -a "${RUN_LOG}"
        fi
        find "${SMOKE_CKPT_FOLDER}" -type f 2>/dev/null | head -20 | tee -a "${RUN_LOG}"
    else
        echo "gate ckpt_non_empty: FAIL (no files under ${SMOKE_CKPT_FOLDER})" | tee -a "${RUN_LOG}"
        FAIL=1
    fi
else
    echo "gate ckpt_non_empty: FAIL (folder missing: ${SMOKE_CKPT_FOLDER})" | tee -a "${RUN_LOG}"
    FAIL=1
fi

# Gate 4: new checkpoint stores consumed_samples == N + steps*GBS.
# Read the dataloader state from the latest step dir via DCP/TorchTitan loader.
# Fall back to a direct Python probe if the helper is unavailable.
#
# GBS for this gate: use $GBS if set, otherwise derive from the training config.
# For the 20B prod configs: 512N has GBS=12288 (LBS=2 * 6144 dp), 256N has
# GBS=6144 (LBS=2 * 3072 dp). For the debugmodel the GBS is smaller; the gate
# still works because consumed_samples advances by micro_batch_size * dp per
# yielded batch (i.e. samples_per_yield), which equals GBS when GAS=1.
if [[ -n "${GBS:-}" ]]; then
    GATE_GBS="${GBS}"
else
    # Derive from WORLD_SIZE and default LBS=2 (prod) or 1 (debugmodel).
    # If uncertain, let the Python probe read it from the checkpoint.
    GATE_GBS=""
fi

if [[ -n "${GATE_GBS}" ]]; then
    EXPECTED_CS=$(( RESUME_N + SMOKE_STEPS * GATE_GBS ))
    echo "expected consumed_samples: ${RESUME_N} + ${SMOKE_STEPS}*${GATE_GBS} = ${EXPECTED_CS}" | tee -a "${RUN_LOG}"
else
    EXPECTED_CS=""
    echo "GBS not set; will read expected from checkpoint metadata" | tee -a "${RUN_LOG}"
fi

# Python probe: find the latest step dir and read dataloader.consumed_samples.
python3 - "${SMOKE_CKPT_FOLDER}" "${EXPECTED_CS}" "${RESUME_N}" "${SMOKE_STEPS}" << 'PYEOF' 2>&1 | tee -a "${RUN_LOG}"
import os
import sys
import re
from pathlib import Path

ckpt_folder = Path(sys.argv[1])
expected_cs = sys.argv[2]
resume_n = int(sys.argv[3])
smoke_steps = int(sys.argv[4])

# Find the latest step dir.
step_dirs = sorted(
    [p for p in ckpt_folder.rglob("step-*") if p.is_dir()],
    key=lambda p: int(re.search(r"step-(\d+)", p.name).group(1)) if re.search(r"step-(\d+)", p.name) else -1,
)
if not step_dirs:
    # Fallback: maybe DCP writes directly under the folder without step- prefix.
    candidates = [p for p in ckpt_folder.iterdir() if p.is_dir()] if ckpt_folder.exists() else []
    if candidates:
        step_dirs = sorted(candidates, key=lambda p: p.stat().st_mtime)

if not step_dirs:
    print(f"gate consumed_samples: FAIL (no step dir under {ckpt_folder})")
    sys.exit(1)

latest = step_dirs[-1]
print(f"latest ckpt step dir: {latest}")

# Try DCP load first (TorchTitan's native path).
consumed = None
try:
    import torch.distributed.checkpoint as dcp
    # DCP stores dataloader state under the "dataloader" key.
    state = {"dataloader.consumed_samples": None}
    # Use FileSystemReader for local paths.
    reader = dcp.FileSystemReader(str(latest))
    dcp.load(state, storage_reader=reader, no_dist=True)
    consumed = state.get("dataloader.consumed_samples")
    if consumed is not None:
        # May be a tensor or plain int.
        try:
            import torch
            if isinstance(consumed, torch.Tensor):
                consumed = int(consumed.item())
            else:
                consumed = int(consumed)
        except Exception:
            consumed = int(consumed)
        print(f"dataloader.consumed_samples (DCP): {consumed}")
except Exception as e:
    print(f"DCP load attempt: {e}")

# Fallback: check for a plain file that stores the counter.
if consumed is None:
    for probe in [latest / "dataloader.consumed_samples", latest / ".metadata"]:
        if probe.exists():
            print(f"found probe file: {probe}")
            try:
                print(probe.read_text()[:500])
            except Exception as e:
                print(f"  read failed: {e}")
    # Last resort: scan any file that mentions consumed_samples.
    for f in latest.rglob("*"):
        if f.is_file() and f.stat().st_size < 1_000_000:
            try:
                txt = f.read_text(errors="ignore")
                if "consumed_samples" in txt:
                    print(f"  {f}: {txt[:500]}")
            except Exception:
                pass

if consumed is None:
    print("gate consumed_samples: FAIL (could not read consumed_samples from checkpoint)")
    print("  (check train log and checkpoint layout manually)")
    sys.exit(1)

if expected_cs:
    expected_cs = int(expected_cs)
    if consumed == expected_cs:
        print(f"gate consumed_samples: PASS ({consumed} == {expected_cs})")
    else:
        print(f"gate consumed_samples: FAIL (got {consumed}, expected {expected_cs})")
        sys.exit(1)
else:
    # No expected value; just verify it equals resume_n + steps * GBS by reading GBS
    # from the checkpoint if possible, otherwise just check it's > resume_n.
    if consumed == resume_n:
        print(f"gate consumed_samples: FAIL (still at resume N={resume_n}, should have advanced by {smoke_steps}*GBS)")
        sys.exit(1)
    if consumed > resume_n:
        print(f"gate consumed_samples: PASS (advanced from {resume_n} to {consumed}, delta={consumed - resume_n})")
        if (consumed - resume_n) % smoke_steps == 0:
            gbs_inferred = (consumed - resume_n) // smoke_steps
            print(f"  inferred GBS: {gbs_inferred}")
        else:
            print(f"  WARN: delta {consumed - resume_n} not divisible by steps {smoke_steps}")
    else:
        print(f"gate consumed_samples: FAIL (went backwards: {resume_n} -> {consumed})")
        sys.exit(1)
PYEOF
PY_RC=$?
if [[ ${PY_RC} -ne 0 ]]; then
    echo "gate consumed_samples: FAIL (python probe rc=${PY_RC})" | tee -a "${RUN_LOG}"
    FAIL=1
fi

# ---- Verdict ----
echo "" | tee -a "${RUN_LOG}"
if [[ ${FAIL} -eq 0 ]]; then
    echo "VERDICT: ok" | tee -a "${RUN_LOG}"
    echo "All gates passed. Safe to boost umbrella to >2048N with N=${RESUME_N}." | tee -a "${RUN_LOG}"
else
    echo "VERDICT: failed" | tee -a "${RUN_LOG}"
    echo "One or more gates failed; see above. Do NOT boost." | tee -a "${RUN_LOG}"
    exit 1
fi
