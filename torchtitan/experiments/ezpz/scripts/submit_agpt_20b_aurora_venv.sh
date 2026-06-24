#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-20b-sophiag-olmo-mix-n256-v2
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -l select=256
#PBS -q prod
#PBS -j oe

# DEPRECATED (2026-06-24): superseded by
# submit_agpt_20b_aurora_venv_failover.sh, which ALL live 20B production
# uses. The failover wrapper adds bad-node preflight + spare-swap retry
# and is a standalone launcher (it does NOT call this script). This
# plain version is kept only to reproduce pre-failover dispatches; do
# not use it for new production runs.
#
# Aurora-adapted variant of train_agpt_20b_venv.sh.
# - Uses Aurora queue/filesystem/account.
# - Restart from scratch under /flare/AuroraGPT/foremans/runs/agpt-20b-v2/
#   on the new clone (HEAD has the dtype=float32 RMSNorm fix).
# - LBS=2 with the new venv (was LBS=1 on torch 2.10).
# - Defaults to plain CrossEntropyLoss (config=agpt_20b). Set
#   `CONFIG_SUFFIX=_chunkedce qsub ...` to use ChunkedCELoss instead
#   (saves memory at the cost of ~10% throughput).

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# Aurora compute nodes need the ALCF proxy for any outbound HTTP (W&B, HF Hub).
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# Source ezpz-utils (cached locally to avoid bit.ly redirect hangs).
EZPZ_UTILS="$(dirname "$(realpath "$0")")/../../../.ezpz-utils-cache/ezpz-utils.sh"
EZPZ_UTILS="$(realpath "$EZPZ_UTILS" 2>/dev/null || echo "")"
if [[ -z "$EZPZ_UTILS" || ! -f "$EZPZ_UTILS" ]]; then
    EZPZ_UTILS="${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh"
fi
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi
ezpz_setup_job

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
# Prefer tarball broadcast over per-file rsync — single ~3GB sequential read
# per node beats thousands of small-file rsyncs at scale (Lustre metadata).
# Falls back to plain yeet-env if tarball isn't built yet.
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "yeet-env via tarball: .venv.tar.gz"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "yeet-env via rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

# Kill stale palsd processes from previous runs.
_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    log_message INFO "Killing stale palsd processes: $_stale_palsd"
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd

# ---- Configuration ----
MODEL="20b"
NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-2}"
GAS="${GAS:-1}"
GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))

OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"

DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

CKPT_KEEP_LATEST_K="${CKPT_KEEP_LATEST_K:-0}"
CKPT_INTERVAL=100
# CKPT_DIR is overridable via env var so experimental forks (e.g.
# different LR) can write to a separate trajectory without colliding
# with the canonical chain.
CKPT_DIR="${CKPT_DIR:-checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}}"

DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} on ${TRAIN_TOKENS} tokens (Aurora venv, fp32 master)"
log_message INFO "-------------------------------------------"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "NNODES: ${NNODES}"
log_message INFO "WORLD_SIZE: ${WORLD_SIZE:-${NGPUS}}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}"
log_message INFO "Local batch size (LBS): ${LBS}"
log_message INFO "Gradient accumulation steps (GAS): ${GAS}"
log_message INFO "Global batch size (GBS): ${GBS}"
log_message INFO "Training steps calculated as: ${TRAINING_STEPS}"
log_message INFO "DATASET_PATH: ${DFL}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="

# ---- Launch ----
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}${CONFIG_SUFFIX:-}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}" \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
