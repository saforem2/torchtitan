#!/bin/bash
# GBS ladder dispatch for Sirius LR finder — 5B adamw scaling law
# GBS=128 already done 2026-11-12 (adamw 2.48e-04 / mano 1.81e-04) — NOT requeued here.
# This wrapper iterates GBS in (32 512 2048) and qsubs submit_lr_finder_sirius.sh
# for each value. Run on a Sirius login node (sirius.alcf.anl.gov).
# GBS=2048: base script auto-halves GAS via LBS=2 when GBS >= 1024 (dp 32, GAS 32 not 64).
#   Without LBS=2 it NCCL-timed-out (GAS 64, 3 steps/33 min); GBS512 GAS 16 succeeded.
#   Walltime 12h covers GBS512+ accumulation. Keep GBS_VALUES=(32 512 2048).
#
# Usage (on sirius):
#   bash torchtitan/experiments/ezpz/scripts/submit_lr_finder_gbs_ladder_sirius.sh
#   # or with dry-run (no qsub, just echo):
#   DRY_RUN=1 bash torchtitan/experiments/ezpz/scripts/submit_lr_finder_gbs_ladder_sirius.sh
#
# Idempotent: re-running re-submits the same 3 GBS jobs; PBS assigns new job IDs
# but does not mutate local state. Queued jobs can be inspected via qstat.
set -euo pipefail

export PATH=/opt/pbs/bin:$PATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/submit_lr_finder_sirius.sh"

# ---------------------------------------------------------------------------
# Config — matches task spec exactly
# ---------------------------------------------------------------------------
GBS_VALUES=(32 512 2048)
MODEL_SIZE=5b
LRF_OPTIMIZERS=adamw
LRF_MAX_LR=1e-1
LRF_INIT_LR=1e-8
LRF_FRACTION=0.15
LRF_STEPS=1000

# Note: GBS=128 was completed 2026-11-12 on Sirius (8N 32 GPUs, 150 steps,
# 1e-8 -> 1e-1 sweep) — advertised LR adamw 2.48e-04 / mano 1.81e-04.
# Do NOT requeue GBS=128 here; this ladder extends with 32, 512, 2048 only.

# ---------------------------------------------------------------------------
# Preconditions (local to Sirius login node, no side effects)
# ---------------------------------------------------------------------------
if [[ ! -f "${BASE_SCRIPT}" ]]; then
    echo "ERROR: base script not found: ${BASE_SCRIPT}" >&2
    exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        echo "WARNING: qsub not found on PATH=${PATH} — set DRY_RUN=1 to preview without dispatch" >&2
    fi
fi

echo "=========================================================="
echo " GBS ladder dispatch — Sirius LR finder (5B adamw)"
echo "   Base script : ${BASE_SCRIPT}"
echo "   GBS values  : ${GBS_VALUES[*]} (GBS=128 done 2026-11-12, skipped)"
echo "   MODEL_SIZE  : ${MODEL_SIZE}"
echo "   Optimizers  : ${LRF_OPTIMIZERS}"
echo "   LR window   : ${LRF_INIT_LR} -> ${LRF_MAX_LR}  fraction=${LRF_FRACTION}  steps=${LRF_STEPS}"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Dispatch loop
# ---------------------------------------------------------------------------
for GBS in "${GBS_VALUES[@]}"; do
    echo ""
    echo "--- GBS=${GBS} ---"
    QSUB_ENV="MODEL_SIZE=${MODEL_SIZE},LRF_OPTIMIZERS=${LRF_OPTIMIZERS},LRF_MAX_LR=${LRF_MAX_LR},LRF_INIT_LR=${LRF_INIT_LR},LRF_FRACTION=${LRF_FRACTION},LRF_STEPS=${LRF_STEPS},LRF_GBS=${GBS}"
    echo "qsub -v ${QSUB_ENV} ${BASE_SCRIPT}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] skipped qsub for GBS=${GBS}"
    else
        JOBID="$(qsub -v "${QSUB_ENV}" "${BASE_SCRIPT}")"
        echo "Submitted GBS=${GBS} -> ${JOBID}"
    fi
done

# ---------------------------------------------------------------------------
# Post-dispatch qstat check
# ---------------------------------------------------------------------------
echo ""
echo "--- qstat check ---"
if command -v qstat >/dev/null 2>&1; then
    qstat -u "${USER}" 2>&1 || echo "(qstat returned non-zero — queue may be empty or unavailable)"
else
    echo "(qstat not on PATH — skipping queue check; on Sirius ensure PATH=/opt/pbs/bin:\$PATH)"
fi

echo ""
echo "Done. 3 GBS jobs dispatched (32, 512, 2048). GBS=128 intentionally skipped (done 2026-11-12)."
