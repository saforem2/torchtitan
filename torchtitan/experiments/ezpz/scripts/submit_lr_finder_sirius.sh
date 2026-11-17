#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-lr-finder-sirius
#PBS -l walltime=12:00:00
# walltime 12:00 required for GBS512+ (GAS 16+); 6h killed GBS512 at 127/150 — GBS2048 also 12h with LBS=2
#PBS -l filesystems=home:tegu
#PBS -l select=8
#PBS -q workq
#PBS -j oe
# On Sirius the ladder wrapper may be invoked from $HOME; PBS_O_WORKDIR then
# equals $HOME and the script would look for checkpoints/outputs under the
# wrong tree. Canonicalize to the torchtitan checkout regardless of submit CWD.
# NOTE: test for .git so we do not confuse the repo root (~/torchtitan) with
# its package subdir (~/torchtitan/torchtitan, which always exists).
if [[ -d "${PBS_O_WORKDIR:-}/torchtitan/.git" ]]; then
    cd "${PBS_O_WORKDIR}/torchtitan" || exit 1
    WORKDIR="${PBS_O_WORKDIR}/torchtitan"
else
    cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1
    WORKDIR="${PBS_O_WORKDIR:-$(pwd)}"
fi
if [[ -f "${WORKDIR}/.ezpz-utils-cache/ezpz-utils.sh" ]]; then
    source "${WORKDIR}/.ezpz-utils-cache/ezpz-utils.sh"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi
ezpz_load_modules
ezpz_setup_job
# ezpz_setup_job recomputes WORKING_DIR from os.getcwd() and then
# overrides PBS_O_WORKDIR on mismatch, which can produce
# ~/torchtitan/torchtitan on Sirius. Restore both to the canonical
# checkout so ./assets/hf resolves.
export PBS_O_WORKDIR="${WORKDIR}"
export WORKING_DIR="${WORKDIR}"
cd "${WORKDIR}" || exit 1
export MPICH_GPU_SUPPORT_ENABLED=0
export EZPZ_MPI_LABEL=1
export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH}"

VENV_DIR="${HOME}/torchtitan/.venv"
source "${VENV_DIR}/bin/activate"
# Config registry uses relative ./assets/hf/... (correct from torchtitan checkout).
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
# Auto-adjust LBS at large GBS to keep GAS bounded (prevents NCCL 300s timeout from
# 64 microbatch accumulations — observed on GBS2048 with dp=32 LBS=1 -> GAS=64:
# 3 steps in 33 min vs GBS512 GAS=16 16 steps in 74 min). Smoke config uses
# _use_hf_streaming where num_tokens_per_microbatch_per_dp_rank = LBS*SEQ_LEN,
# so LBS=2 doubles tokens per microbatch safely within 4096 ctx.
if (( LRF_GBS >= 1024 )); then
  if [[ "${LRF_LBS}" == "1" ]]; then
    LRF_LBS=2
    export LRF_LBS
    echo "LRF_GBS=${LRF_GBS} >= 1024 -> auto-raised LRF_LBS=2 to halve GAS" >&2
  else
    echo "WARNING: LRF_GBS=${LRF_GBS} >= 1024 but LRF_LBS=${LRF_LBS} already set — not overriding" >&2
  fi
fi
# Safety: fail if GAS would still exceed 32 (dp 32 on Sirius 8N=32 dp ranks).
_LRF_GAS=$(( LRF_GBS / LRF_LBS / 32 ))
if (( _LRF_GAS > 32 )); then
  echo "ERROR: GAS=${_LRF_GAS} (GBS=${LRF_GBS}/LBS=${LRF_LBS}/dp=32) exceeds 32 — reduce LRF_GBS or increase LRF_LBS" >&2
  exit 1
fi
unset _LRF_GAS
echo "=========================================================="
echo " OLMo-2 LR finder on Sirius (shared-NFS, no yeet)"
echo "   MODEL_SIZE=${MODEL_SIZE} config=${LRF_CONFIG}"
echo "   GBS/LBS=${LRF_GBS}/${LRF_LBS} @ 8N (dp 32, GAS varies with GBS)"
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
      --checkpoint.enable --checkpoint.folder "checkpoints/lr_finder_${model}_${opt}_gbs${LRF_GBS}" \
      --checkpoint.interval 5 --checkpoint.no-last-save-model-only \
      --checkpoint.async-mode disabled \
      --lr_finder.enable --lr_finder.init_lr "${LRF_INIT_LR}" \
      --lr_finder.max_lr "${LRF_MAX_LR}" --lr_finder.fraction "${LRF_FRACTION}" \
      2>&1 | tee "outputs/lr_finder_${model}_${opt}_${PBS_JOBID%%.*}.log"
    echo "--- [${model}_${opt}] done (exit $?) ---"
  done
done
