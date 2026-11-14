#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-lr-finder-sirius
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:tegu
#PBS -l select=8
#PBS -q workq
#PBS -j oe
# LR finder on Sirius — shared-NFS mode (no yeet, matches prod hedge).
set -o pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1
# Source ezpz-utils for log_message etc (prod does this)
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
source "${VENV_DIR}/bin/activate"
log_message INFO "Sirius LR-finder shared-NFS: ${VENV_DIR}/bin/python"
MODEL_SIZE="${MODEL_SIZE:-5b}"
export LRF_CONFIG="agpt_${MODEL_SIZE}_olmo2tok_smoke"
export LRF_OPTIMIZERS="${LRF_OPTIMIZERS:-adamw mano}"
export LRF_GBS="${LRF_GBS:-128}"
export LRF_LBS="${LRF_LBS:-1}"
export LRF_SEQ_LEN="${LRF_SEQ_LEN:-4096}"
export LRF_STEPS="${LRF_STEPS:-1000}"
export LRF_MODE="${LRF_MODE:-custom}"
export LRF_FRACTION="${LRF_FRACTION:-0.15}"
export LRF_INIT_LR="${LRF_INIT_LR:-1e-8}"
export LRF_MAX_LR="${LRF_MAX_LR:-1e-1}"
export LRF_AC=full
export LRF_TIMEOUT=6000
export LRF_IDLE_TIMEOUT=1800
export LRF_SKIP_MODULE_LOAD=1
export NGPU_PER_HOST=4
export LRF_USE_CONFIG_DATALOADER=1
export LRF_DUMP_FOLDER="outputs/lr_finder_${LRF_MODE}_${MODEL_SIZE}_olmo2tok_gbs${LRF_GBS}_${LRF_OPTIMIZERS// /-}_${PBS_JOBID%%.*}"
echo "=========================================================="
echo " OLMo-2 LR finder on Sirius (shared-NFS, no yeet)"
echo "   MODEL_SIZE=${MODEL_SIZE} config=${LRF_CONFIG}"
echo "   GBS/LBS=${LRF_GBS}/${LRF_LBS} @ 8N (dp 32, GAS 4)"
echo "   LR window=${LRF_INIT_LR} -> ${LRF_MAX_LR} opts=${LRF_OPTIMIZERS}"
echo "=========================================================="
# Inline the Model×Optimizer loop without yeet-env (shared-NFS path)
for model in ${MODEL_SIZE}; do
  for opt in ${LRF_OPTIMIZERS}; do
    echo "--- [${model}_${opt}] LR-finder ---"
    ezpz launch \
      --nproc $(( 8 * 4 )) \
      --nproc_per_node 4 \
      --auto-retry \
      --spare-nodes auto \
      --timeout ${LRF_IDLE_TIMEOUT} \
      --no-transfer \
      -- \
      ${VENV_DIR}/bin/python -m torchtitan.experiments.ezpz.train \
      --module ezpz.agpt --config "${LRF_CONFIG}" \
      --optimizer "${opt}" \
      --training.steps "${LRF_STEPS}" \
      --training.num-tokens-per-microbatch-per-dp-rank "$(( LRF_LBS * LRF_SEQ_LEN ))" \
      --training.num-tokens-per-train-step "$(( LRF_GBS * LRF_SEQ_LEN ))" \
      --training.max-context-length "${LRF_SEQ_LEN}" \
      --metrics.log_freq 1 \
      --checkpoint.enable --checkpoint.folder "checkpoints/lr_finder_${model}_${opt}" \
      --checkpoint.interval 5 --checkpoint.no-last-save-model-only \
      --checkpoint.async-mode disabled \
      --lr_finder.enable --lr_finder.init_lr "${LRF_INIT_LR}" \
      --lr_finder.max_lr "${LRF_MAX_LR}" --lr_finder.fraction "${LRF_FRACTION}" \
      2>&1 | tee "outputs/lr_finder_${model}_${opt}_${PBS_JOBID%%.*}.log"
    echo "--- [${model}_${opt}] done (exit $?) ---"
  done
done
