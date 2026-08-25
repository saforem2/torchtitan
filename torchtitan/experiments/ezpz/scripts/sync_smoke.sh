#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-sync-smoke
#PBS -l walltime=00:30:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe

# Post-upstream-sync smoke for ezpz model code.
#
# Two-phase, fail-fast verification that a sync (or any change to the
# agpt / moe model + parallelize + config-registry surface) didn't
# break the train path:
#
#   1. Import probe -- imports every ezpz module a sync is most likely
#      to break (parallelize, config_registry, activation_checkpoint,
#      trainer). Runs in seconds; catches missing-symbol / renamed-API
#      breakage BEFORE paying for a distributed launch. This is the
#      single most useful check after an upstream merge.
#   2. Per-config train smoke -- runs each config in SMOKE_CONFIGS for
#      a few `--debug.seed=42 --debug.deterministic` steps, captures a
#      per-config rc, and emits step-1/2 loss for eyeball comparison
#      against the pre-merge baseline. Prints a single machine-greppable
#      `VERDICT: ok|failed` at the end.
#
# Runs the production code path: yeet the repo-root .venv.tar.gz to
# /tmp on every compute node, then activate /tmp/.venv. Build the
# tarball first with `ezpz tar-env` if you changed the venv.
#
# Single node (sunspot = 12 tiles/node). Each config is launched
# across all visible ranks.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke.sh
#
# Env knobs (all optional):
#   SMOKE_CONFIGS  space-separated "module:config[:extra cli ...]" specs.
#                  Default: the agpt + moe debugmodels.
#                  moe_debugmodel needs seq_len/lbs shrunk on XPU (see
#                  the default below for why).
#   SMOKE_STEPS    training steps per config (default 2).
#   SEED           debug seed (default 42).

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

STEPS="${SMOKE_STEPS:-2}"
SEED="${SEED:-42}"

# "module:config:extra args" -- one per smoke. moe_debugmodel defaults
# to seq_len=8192 / lbs=8; on XPU (no flash attn -> SDPA MATH fallback)
# that materializes a (8, H, 8192, 8192) attention matrix = 32 GiB
# single alloc -> OOM on 1 tile/rank. Shrink to seq_len=512 / lbs=1 to
# exercise the train loop + AC + loss path (not a perf measurement).
# A TP>1 entry is REQUIRED: TP=1 (pure FSDP) never wraps the attention
# local_map, so it misses the entire sharding-contract surface. The 57th
# sync's q_BLNH rename broke the ezpz attention forks under TP>1 and the
# TP=1 debugmodels passed clean -- the regression only surfaced at 64N.
# agpt_debugmodel @ TP=2 exercises model.parallelize's local_map path on
# a single node (12 tiles).
# Every arm pins BOTH max-context-length AND
# num-tokens-per-microbatch-per-dp-rank, and they must be EQUAL.
#
# Post-#4121 the loader flattens [B, L] -> [T], so T = ntok_per_microbatch.
# rope._reshape_for_broadcast then does rope_cache[:T].view(T, 1, w) against a
# cache that has only max_context_length rows -- so T > max_context_length is a
# hard shape error:
#   RuntimeError: shape '[16384, 1, 8]' is invalid for input of size 65536
# (MEASURED, smoke 8781623 arm 1: agpt_debugmodel defaults to ntok=16384 vs
# max_context_length=8192, i.e. 2 rows.) Production runs exactly one row
# (agpt_2b and agpt_20b both have ntok == max_context_length == 8192), which is
# why it does not hit this -- but the margin is zero.
#
# Setting emit_positions would take the rope_cache[positions] branch instead,
# but that is NOT a workaround here: it puts a DeviceMesh into the
# saved-for-backward set, which AOT autograd rejects (see the emit_positions
# docstring in blendcorpus_builder.py).
#
# There is no training.seq_len and no training.local_batch_size. The real
# fields are num_tokens_per_microbatch_per_dp_rank, num_tokens_per_train_step
# and max_context_length -- both earlier spellings were silently rejected by
# tyro on every run since b8c369fb0.
DEFAULT_CONFIGS=(
    "ezpz.agpt:agpt_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512"
    "ezpz.agpt:agpt_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512 --parallelism.tensor-parallel-degree=2"
    "ezpz.moe:moe_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512"
)
if [[ -n "${SMOKE_CONFIGS:-}" ]]; then
    read -r -a CONFIGS <<< "${SMOKE_CONFIGS}"
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# Production code path: yeet the repo-root .venv.tar.gz (torch 2.13) to
# /tmp on every compute node, then activate /tmp/.venv.
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    ezpz yeet-env --src .venv.tar.gz
else
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/ezpz-sync-smoke-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"

echo "===== ezpz sync smoke =====" | tee "${LOG}"
date | tee -a "${LOG}"
echo "configs: ${CONFIGS[*]}" | tee -a "${LOG}"
echo "steps=${STEPS} seed=${SEED}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# --- Phase 1: import probe (fail fast) ---
echo "--- import probe ---" | tee -a "${LOG}"
python3 -c "
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel
from torchtitan.experiments.ezpz.moe.parallelize import parallelize_moe
from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
from torchtitan.experiments.ezpz.moe.activation_checkpoint import MoeSelectiveAC
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
print('IMPORT_OK')
" 2>&1 | tee -a "${LOG}"

if ! grep -q "IMPORT_OK" "${LOG}"; then
    echo "VERDICT: import_failed" | tee -a "${LOG}"
    exit 1
fi

# --- Phase 2: per-config train smoke ---
declare -a RESULTS=()
OVERALL_OK=1
idx=0
for spec in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    module="${spec%%:*}"
    rest="${spec#*:}"
    config="${rest%%:*}"
    extra="${rest#*:}"
    [[ "${extra}" == "${rest}" ]] && extra=""  # no extra-args segment

    # Label includes the index so the same config can appear more than
    # once (e.g. agpt_debugmodel at TP=1 and TP=2) without its per-entry
    # log or result key colliding.
    label="${idx}-${config}"

    echo "" | tee -a "${LOG}"
    echo "--- [${idx}] ${module} / ${config}: ${STEPS} deterministic steps ${extra:+(${extra})} ---" | tee -a "${LOG}"
    # shellcheck disable=SC2086 -- extra is an intentional word-split arg list
    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module="${module}" \
        --config="${config}" \
        --training.steps="${STEPS}" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --metrics.enable-wandb \
        --checkpoint.no-enable \
        ${extra} \
        2>&1 | tee -a "${LOG_DIR}/${label}.log" | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    RESULTS+=("${label}=${rc}")
    echo "${label} rc=${rc}" | tee -a "${LOG}"
    [[ ${rc} -ne 0 ]] && OVERALL_OK=0
done

echo "" | tee -a "${LOG}"
if [[ ${OVERALL_OK} -eq 1 ]]; then
    echo "VERDICT: ok" | tee -a "${LOG}"
else
    echo "VERDICT: failed (${RESULTS[*]})" | tee -a "${LOG}"
    exit 1
fi
