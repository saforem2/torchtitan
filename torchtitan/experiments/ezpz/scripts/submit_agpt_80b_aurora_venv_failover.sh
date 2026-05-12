#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-80b-failover
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
# select= overridden via qsub -l select=N (e.g. 522 = 512 train + 10 spare)
#PBS -q prod
#PBS -j oe

# Bad-node failover variant for 80B v2 production training.
#
# Submit with:
#   qsub -l select=522 -l walltime=12:00:00 \
#       -v NHOSTS_TRAIN=512,FAILOVER_MAX_RETRIES=2 \
#       scripts/submit_agpt_80b_aurora_venv_failover.sh
#
# Working-config provenance (see CLAUDE.md "v2 — 80B"):
# - 80B v2 path identified 2026-05-05 in 4N smoke (job 12466025):
#   AdamW LR=1e-6, TP=2, AC=full, compile=OFF, fp32-master.
#   Loss descended cleanly 12.98 -> 10.46 over 20 steps.
# - User confirmed working on Sunspot 8N with the same config plus
#   --validator.enable --validator.freq=5 --dataloader.num-workers=2
#   on 2026-05-11.
# - compile=ON crashes for the entire 80B family on torch 2.13 with
#   the DeviceMesh-in-saved-tensors AOT autograd assertion. Don't
#   re-enable compile until upstream fixes that.
#
# Required env: NHOSTS_TRAIN
# Recommended override: FAILOVER_MAX_RETRIES=2 (default 3) — each
# retry pays the 80B model-build + dataloader-init cost (~5-15 min),
# so don't spend too much walltime on retries.

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
    echo "ERROR: NHOSTS_TRAIN env var required (use qsub -v NHOSTS_TRAIN=N)"
    exit 1
fi

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# Source ezpz-utils + failover lib
# NOTE: PBS copies the submit script into /var/spool/pbs/mom_priv/jobs/, so
# `$(dirname "$(realpath "$0")")` is NOT the original scripts dir. Use
# $PBS_O_WORKDIR (the dir from which qsub was run) + the canonical relative
# path under the repo to find sibling files.
SCRIPTS_DIR="${PBS_O_WORKDIR:-$PWD}/torchtitan/experiments/ezpz/scripts"
EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

FAILOVER_LIB="$SCRIPTS_DIR/failover_lib.sh"
[[ -f "$FAILOVER_LIB" ]] || { echo "ERROR: failover_lib.sh not found at $FAILOVER_LIB (PBS_O_WORKDIR=$PBS_O_WORKDIR, PWD=$PWD)"; exit 1; }
source "$FAILOVER_LIB"

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Split PBS_NODEFILE into active (NHOSTS_TRAIN nodes) + spare (rest).
failover_init "$NHOSTS_TRAIN" || exit 1

# Now ezpz_setup_job sees the active subset only.
ezpz_setup_job

source .venv/bin/activate
# yeet to ALL nodes (active + spare) so any spare can swap in instantly.
failover_yeet_all || exit 1
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
# Defaults match the proven 2026-05-05 working config (AdamW LR=1e-6,
# TP=2, AC=full, compile=OFF, fp32-master).
MODEL="80b"
NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-1}"
GAS="${GAS:-1}"
GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))

# 80B uses AdamW LR=1e-6 — SophiaG/Muon are known broken at dim=9216
# (bf16 overflow in Hessian/Newton-Schulz). LR=1.1e-5 NaNs at
# production GBS; LR=1e-6 was stable in the 12466025 smoke.
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-1e-6}"

# Activation checkpoint is required at 80B to fit in tile memory.
ACKPT_MODE="${ACKPT_MODE:-full}"

DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

CKPT_KEEP_LATEST_K="${CKPT_KEEP_LATEST_K:-0}"
CKPT_INTERVAL=100
CKPT_DIR="${CKPT_DIR:-checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}}"
DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} v2 (failover wrapper, ${NNODES} active nodes)"
log_message INFO "-------------------------------------------"
log_message INFO "PBS allocation: $(wc -l < "$FAILOVER_PBS_NODEFILE_ORIG") nodes total"
log_message INFO "  active: $(wc -l < "$FAILOVER_ACTIVE")  spare: $(wc -l < "$FAILOVER_SPARE")"
log_message INFO "FAILOVER_MAX_RETRIES: ${FAILOVER_MAX_RETRIES:-3}"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}"
log_message INFO "TP: ${TP}, AC: ${ACKPT_MODE}, compile: OFF"
log_message INFO "GBS: ${GBS}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="

# ---- Launch with failover ----
# The launch line below mirrors the user's validated Sunspot
# command + the 12466025 smoke config. Key differences from 2B/20B:
#   --parallelism.tensor-parallel-degree=2 (TP must be > 1 for 80B)
#   --activation-checkpoint.mode=full     (required to fit in memory)
#   --compile.no-enable                    (compile crashes on torch 2.13)
#   --parallelism.expert-parallel-degree=1 (dense, no MoE)
failover_run ezpz launch python3 -m torchtitan.experiments.ezpz.train \
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
    --dataloader.num-workers=2 \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --parallelism.tensor-parallel-degree="${TP}" \
    --parallelism.expert-parallel-degree=1 \
    --parallelism.data-parallel-replicate-degree=1 \
    --parallelism.data-parallel-shard-degree=-1 \
    --activation-checkpoint.mode="${ACKPT_MODE}" \
    --compile.no-enable \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
