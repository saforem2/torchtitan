#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-olmo3-polaris
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -j oe

# Aurora-GPT 4B/10B OLMo-3 pre-training on Polaris (A100 40GB).
#
# Models: 5b_olmo2tok (4.64B, dim 3072) and 10b_olmo2tok (9.48B, dim 4096)
#         with the OLMo-3 tokenizer (vocab 100352, byte-identical to
#         OLMo-2-1124-7B on Polaris). Configs agpt_5b_olmo3_polaris /
#         agpt_10b_olmo3_polaris stream Dolma raw JSON via Grain.
#
# Usage:
#   cd /eagle/AuroraGPT/foremans/projects/saforem2/torchtitan
#   # 4B smoke (2 nodes, quick validation):
#   qsub -l select=4 -v NHOSTS_TRAIN=2,MODEL=5b_olmo3_polaris torchtitan/experiments/ezpz/scripts/submit_agpt_olmo3_polaris.sh
#   # 4B production (16 nodes):
#   qsub -l select=18 -v NHOSTS_TRAIN=16,MODEL=5b_olmo3_polaris torchtitan/experiments/ezpz/scripts/submit_agpt_olmo3_polaris.sh
#   # 10B production (32 nodes):
#   qsub -l select=34 -v NHOSTS_TRAIN=32,MODEL=10b_olmo3_polaris torchtitan/experiments/ezpz/scripts/submit_agpt_olmo3_polaris.sh
#
# Chain with: qsub -W depend=afterany:<jobid> -l select=N -v NHOSTS_TRAIN=N,MODEL=... submit_agpt_olmo3_polaris.sh

ulimit -c 0

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
    echo "ERROR: NHOSTS_TRAIN required (use qsub -v NHOSTS_TRAIN=N,MODEL=...)" >&2
    exit 1
fi

MODEL="${MODEL:-5b_olmo3_polaris}"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

cd "${PBS_O_WORKDIR:-$(pwd)}"

ezpz_load_modules
ezpz_setup_job
export MPICH_GPU_SUPPORT_ENABLED=0
export EZPZ_MPI_LABEL=1
export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH}"

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: no venv at ${VENV_DIR}/bin/activate" >&2
    exit 1
fi
source "${VENV_DIR}/bin/activate"

_nnodes_all=$(wc -l < "${PBS_NODEFILE}")
timeout 300 mpiexec -n "${_nnodes_all}" --ppn 1 \
    bash -c "rm -rf /tmp/$(basename "${VENV_DIR}")" 2>/dev/null || true
unset _nnodes_all

_venv_tarball="${VENV_DIR}.tar.gz"
if [[ -f "${_venv_tarball}" ]]; then
    log_message INFO "yeet: broadcasting ${_venv_tarball}"
    ezpz yeet --src "${_venv_tarball}"
else
    log_message INFO "yeet: no ${_venv_tarball}, falling back to rsync mode"
    ezpz yeet
fi
unset _venv_tarball
deactivate

_node_venv="/tmp/$(basename "${VENV_DIR}")"
if [[ ! -f "${_node_venv}/bin/activate" ]]; then
    echo "ERROR: yeet did not produce ${_node_venv}/bin/activate" >&2
    exit 1
fi
source "${_node_venv}/bin/activate"
unset _node_venv

_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    log_message INFO "Killing stale palsd processes: $_stale_palsd"
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd

PPN="${NGPU_PER_HOST:-4}"
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))
NNODES="$NHOSTS_TRAIN"

SEQ_LEN="${SEQ_LEN:-4096}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-1}"
GAS="${GAS:-1}"
GBS=$(( NGPUS_ACTIVE * LBS * GAS / (TP * PP * CP) ))
TOKENS_PER_MICROBATCH_PER_DP_RANK=$(( LBS * SEQ_LEN ))
TOKENS_PER_TRAIN_STEP=$(( GBS * SEQ_LEN ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4000000000000}"
TRAINING_STEPS="${TRAINING_STEPS:-$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))}"

CKPT_KEEP_LATEST_K=0
CKPT_INTERVAL="${CKPT_INTERVAL:-100}"
CKPT_DIR="${CKPT_DIR:-checkpoints/agpt-${MODEL}-n${NNODES}-gbs${GBS}}"

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} on Polaris (OLMo-3, ${NNODES} nodes, GBS=${GBS})"
log_message INFO "machine: $(ezpz_get_machine_name)"
log_message INFO "PBS allocation: $(wc -l < "${PBS_NODEFILE}") nodes total"
log_message INFO "  active: ${NHOSTS_TRAIN}  spare: auto"
log_message INFO "--nproc: ${NGPUS_ACTIVE}  -ppn: ${PPN}"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "GBS: ${GBS}  tokens/step: ${TOKENS_PER_TRAIN_STEP}"
log_message INFO "Checkpoint dir: ${CKPT_DIR}"
log_message INFO "==========================================="

# Polaris A100 needs sync checkpointing (async can be flaky on CUDA)
# Data is via Grain streaming — no blendcorpus DFL needed
mfr_args=()
[[ -n "${MAX_FAILOVER_RETRIES:-}" ]] && mfr_args=(--max-failover-retries "${MAX_FAILOVER_RETRIES}")

ezpz launch \
    --nproc "${NGPUS_ACTIVE}" \
    --nproc_per_node "${PPN}" \
    --auto-retry \
    --spare-nodes auto \
    --timeout "${IDLE_TIMEOUT:-1800}" \
    "${mfr_args[@]}" \
    --no-transfer \
    -- \
    python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode=async \
    --debug.print-config \
    --training.num-tokens-per-microbatch-per-dp-rank="${TOKENS_PER_MICROBATCH_PER_DP_RANK}" \
    --training.num-tokens-per-train-step="${TOKENS_PER_TRAIN_STEP}" \
    --training.max-context-length="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
