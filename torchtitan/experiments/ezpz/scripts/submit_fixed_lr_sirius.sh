#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-fixed-lr-sirius
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:tegu
#PBS -l select=8
#PBS -q workq
#PBS -j oe
# Fixed-LR convergence on Sirius — validates finder suggested LRs at 1000 steps
# constant LR (NOT LR finder). Model at GBS=128 seq 4096, 8N 32 GPUs.
# Usage: FIXED_LR=2.48e-04 MODEL_SIZE=5b qsub -v FIXED_LR=2.48e-04,MODEL_SIZE=5b torchtitan/experiments/ezpz/scripts/submit_fixed_lr_sirius.sh
# Or test locally: DRY_RUN=1 bash torchtitan/experiments/ezpz/scripts/submit_fixed_lr_sirius.sh
#
# Defaults: 5B 2.48e-04 / 10B 1.28e-04 (Sirius GBS=128 Nov 11-12 findings).
# Pattern mirrors fixed_lr_30b_sunspot.pbs (validated 30B fixed-LR validation).
set -o pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

if [[ -f "${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh" ]]; then
    source "${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi
ezpz_load_modules
ezpz_setup_job
export MPICH_GPU_SUPPORT_ENABLED=0
export EZPZ_MPI_LABEL=1
export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH}"

VENV_DIR="${HOME}/torchtitan/.venv"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log_message INFO "Sirius fixed-LR: ${VENV_DIR}/bin/python"

MODEL_SIZE="${MODEL_SIZE:-5b}"
FIXED_LR="${FIXED_LR:-2.48e-04}"
FIXED_OPTIMIZER="${FIXED_OPTIMIZER:-adamw}"
FIXED_STEPS="${FIXED_STEPS:-1000}"
GBS="${GBS:-128}"
SEQ_LEN=4096
T_TOKENS_PER_STEP=$(( GBS * SEQ_LEN ))
CONFIG="agpt_${MODEL_SIZE}_olmo2tok_smoke"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] Would run: --config ${CONFIG} --optimizer ${FIXED_OPTIMIZER} --optimizer.lr ${FIXED_LR} --training.steps ${FIXED_STEPS} GBS=${GBS}"
    echo "[DRY_RUN] Check bash -n + preflight instead."
    exit 0
fi

echo "=========================================================="
echo " Sirius fixed-LR convergence (NOT finder)"
echo "   MODEL_SIZE : ${MODEL_SIZE}  CONFIG=${CONFIG}"
echo "   FIXED_LR   : ${FIXED_LR}  Optimizer=${FIXED_OPTIMIZER}"
echo "   GBS        : ${GBS}  seq=${SEQ_LEN}  steps=${FIXED_STEPS}"
echo "   Tokens/step: ${T_TOKENS_PER_STEP}  Nproc=32"
echo "=========================================================="

OUT="outputs/fixed_lr_${MODEL_SIZE}_${FIXED_LR}_${PBS_JOBID%%.*}"
mkdir -p "${OUT}"
LOG="${OUT}/run.log"

run_model() {
    "${VENV_DIR}/bin/python" -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt --config "${CONFIG}" \
        --optimizer "${FIXED_OPTIMIZER}" \
        --optimizer.lr "${FIXED_LR}" \
        --training.steps "${FIXED_STEPS}" \
        --training.num-tokens-per-microbatch-per-dp-rank "${SEQ_LEN}" \
        --training.num-tokens-per-train-step "${T_TOKENS_PER_STEP}" \
        --training.max-context-length "${SEQ_LEN}" \
        --metrics.log_freq 1 \
        --metrics.enable_wandb false \
        --checkpoint.enable --checkpoint.folder "checkpoints/fixed_lr_${MODEL_SIZE}_${FIXED_LR}" \
        --checkpoint.interval 50 --checkpoint.no-last-save-model-only \
        --checkpoint.async-mode disabled \
        2>&1 | tee "${LOG}"
    return "${PIPESTATUS[0]}"
}
export VENV_DIR CONFIG FIXED_LR FIXED_OPTIMIZER FIXED_STEPS OUT LOG GBS SEQ_LEN T_TOKENS_PER_STEP
export -f run_model

ezpz launch \
    --nproc $(( 8 * 4 )) \
    --nproc_per_node 4 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    --no-transfer \
    -- \
    bash -c 'run_model'
rc=$?
if [[ "$rc" -ne 0 ]]; then
    printf 'FIXED_LR_FAILED MODEL_SIZE=%s FIXED_LR=%s optimizer=%s rc=%s\n' "${MODEL_SIZE}" "${FIXED_LR}" "${FIXED_OPTIMIZER}" "${rc}" | tee "${OUT}/FAILED"
    exit "$rc"
fi

# Validate: every step 1..FIXED_STEPS has finite loss + grad_norm
"${VENV_DIR}/bin/python" - "${LOG}" "${OUT}/VALIDATED" "${PBS_JOBID%%.*}" "${FIXED_LR}" "${FIXED_STEPS}" "${FIXED_OPTIMIZER}" <<'PY'
import math
import re
import sys
from pathlib import Path
log_path, verdict_path, job, lr, steps_s, opt = sys.argv[1:]
steps = int(steps_s)
text = Path(log_path).read_text(errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
metric = re.compile(r"step:\s+(?P<step>[0-9]+)\s+loss:\s+(?P<loss>[0-9.eE+-]+)\s+grad_norm:\s+(?P<grad>[0-9.eE+-]+)")
rows = {int(m["step"]): (float(m["loss"]), float(m["grad"])) for m in metric.finditer(text)}
for step in range(1, steps + 1):
    if step not in rows:
        raise SystemExit(f"FIXED_LR_INVALID missing metrics step={step}")
    if not all(math.isfinite(v) and v > 0 for v in rows[step]):
        raise SystemExit(f"FIXED_LR_INVALID step={step} values={rows[step]}")
if any(token in text for token in ("Traceback", "SIGSEGV", "died from signal", "OUT_OF_RESOURCES")):
    raise SystemExit("FIXED_LR_INVALID failure signature present")
verdict = f"FIXED_LR_PASS job={job} lr={lr} optimizer={opt} steps={steps} loss_final={rows[steps][0]} grad_final={rows[steps][1]}"
Path(verdict_path).write_text(verdict + "\n")
print(verdict)
PY
