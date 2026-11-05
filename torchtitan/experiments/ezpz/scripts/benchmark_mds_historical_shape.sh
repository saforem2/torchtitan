#!/bin/bash --login
#PBS -A datascience
#PBS -N mds-historical-shape
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe

# Re-run the MDS model shape behind the 102.9 TFLOP/s/tile Aurora run z2rrt02w
# (2025-07-17, MDS 82e0b2ee, frameworks 2025.0.0) on today's Sunspot MDS stack,
# to separate machine/stack drift from model shape. Arguments are copied from
# that run's argv; only paths, the iteration count, eval/save cadence and
# checkpointing differ. Training starts from scratch.
#   10 layers, hidden 4096, 32 heads, 8 KV heads, FFN 11008, Llama2 vocab,
#   seq 4096, microbatch 1, GBS 192 (accumulation 16 on 12 ranks), ZeRO-1,
#   AdamW, FP32 allreduce accumulation, flash-attention builder.
# Knobs: MDS_SEQ (default 4096), MDS_STEPS (default 32), and MDS_PROFILE=1,
# which records a rank-0 torch.profiler trace of update 12 into
# ${OUT}/tensorboard (MDS --profile pt), the update TT traces.
set -o pipefail
ulimit -c 0

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

CAMPAIGN_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-replicated-20261031
MDS_ROOT="${CAMPAIGN_ROOT}/mds-source"
MDS_HEAD=9f407ed9e8e30920e28272daf8cb7db36c211e21
MDS_VENV="/lus/tegu/projects/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/sunspot/Megatron-DeepSpeed-aurora_frameworks-2025.3.1"
EZPZ_HELPER_ENV="/lus/tegu/projects/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/sunspot/Megatron-DeepSpeed-deepspeed_0.18.5_no_deps_conda_vllm_0.15.0_nre_ipex_2.10.10_pytorch_2.10.0_rel_oneapi_2025.3.1_numpy_2.2.6_python_3.12.8"
PYTHON_OVERLAY="${CAMPAIGN_ROOT}/mds-python-overlay"
DISTUTILS_OVERLAY="${CAMPAIGN_ROOT}/mds-distutils-overlay"
# Llama2-tokenized (uint16, max id < 32000), the books.txt family z2rrt02w used.
BOOKS_LIST="${MDS_ROOT}/ALCF/data-lists/sunspot/books.txt"
TOKENIZER="${MDS_ROOT}/ALCF/tokenizer.model"
MDS_SEQ="${MDS_SEQ:-4096}"
MDS_STEPS="${MDS_STEPS:-32}"
MDS_PROFILE="${MDS_PROFILE:-0}"
PROFILE_ARGS=()
if [[ "${MDS_PROFILE}" == 1 ]]; then
    ((MDS_STEPS >= 12)) || { echo "MDS_PROFILE=1 needs MDS_STEPS >= 12" >&2; exit 2; }
    PROFILE_ARGS=(--profile pt --profile_steps "12,12" --profile-ranks 0)
elif [[ "${MDS_PROFILE}" != 0 ]]; then
    echo "MDS_PROFILE must be 0 or 1" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "${EZPZ_HELPER_ENV}/lib/python3.12/site-packages/ezpz/bin/utils.sh" || exit 90
ezpz_setup_job || exit $?
module load oneapi/release/2025.3.1 hdf5 pti-gpu || exit $?
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VIRTUAL_ENV="${MDS_VENV}"
export PATH="${MDS_VENV}/bin:${EZPZ_HELPER_ENV}/bin:/opt/pbs/bin:${PATH}"
export PYTHONPATH="${MDS_ROOT}:${PYTHON_OVERLAY}:${DISTUTILS_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
hash -r
[[ "$(command -v python)" == "${MDS_VENV}/bin/python" ]] || exit 93
[[ "$(git -C "${MDS_ROOT}" rev-parse HEAD)" == "${MDS_HEAD}" ]] || exit 99
[[ -z "$(git -C "${MDS_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 99
[[ -s "${TOKENIZER}" && -s "${BOOKS_LIST}" ]] || exit 99

JOB="${PBS_JOBID%%.*}"
RUN_NAME="mds-z2rrt02w-shape-seq${MDS_SEQ}${BENCH_TAG:+-${BENCH_TAG}}-${JOB}"
OUT="${CAMPAIGN_ROOT}/outputs/${RUN_NAME}"
mkdir -p "${OUT}/tensorboard" "${OUT}/wandb" "${OUT}/index-cache"
LOG="${OUT}/train.log"
DS_CONFIG="${OUT}/deepspeed-config.json"
cat > "${DS_CONFIG}" <<'JSON'
{
  "fp16": {"enabled": false, "hysteresis": 2, "loss_scale": 0, "min_loss_scale": 1, "loss_scale_window": 1000},
  "bfloat16": {"enabled": true},
  "comms_logger": {"debug": false, "enabled": false, "verbose": false},
  "flops_profiler": {"enabled": false},
  "steps_per_print": 1,
  "train_batch_size": 192,
  "gradient_clipping": 1,
  "zero_optimization": {"stage": 1},
  "wall_clock_breakdown": false,
  "gradient_accumulation_steps": 16,
  "zero_force_ds_cpu_optimizer": false,
  "zero_allow_untested_optimizer": true,
  "train_micro_batch_size_per_gpu": 1
}
JSON

export WANDB_MODE=online WANDB_ENTITY=aurora_gpt WANDB_TEAM=aurora_gpt
export WANDB_PROJECT=torchtitan.ezpz.train WB_PROJECT=torchtitan.ezpz.train
export WANDB_RUN_GROUP=mds-historical-shape-20261103
export WANDB_NAME="${RUN_NAME}" WANDB_RUN_NAME="${RUN_NAME}" WANDB_RUN_ID="${RUN_NAME}"
export WANDB_DIR="${OUT}/wandb"

{
    echo "BEGIN $(date -Is) host=$(hostname -s) job=${JOB} seq=${MDS_SEQ} steps=${MDS_STEPS} profile=${MDS_PROFILE}"
    echo "torch=$(python -c 'import torch; print(torch.__version__)') deepspeed=$(python -c 'import deepspeed; print(deepspeed.__version__)') mds_head=${MDS_HEAD}"
} | tee "${LOG}"

cd "${MDS_ROOT}" || exit 99
# The data-loader workers share tensors over AF_UNIX sockets under TMPDIR.
# PALS sets a per-launch TMPDIR on each rank that exceeds the 108-byte socket
# path limit, so reset it on the rank command line.
ezpz launch --nproc 12 --nproc_per_node 12 -- \
    env TMPDIR=/tmp python "${MDS_ROOT}/pretrain_gpt_alcf.py" \
    --lr 0.0002 --lr-decay-style cosine --lr-warmup-fraction 0.05 \
    --swiglu --hidden-dropout 0 --attention-dropout 0 --normalization rmsnorm \
    --disable-bias-linear --no-query-key-layer-scaling \
    --use-rotary-position-embeddings --untie-embeddings-and-output-weights \
    --num-key-value-heads 8 --ffn-hidden-size 11008 --use-flash-attn-builder \
    --tokenizer-type Llama2Tokenizer --tokenizer-model "${TOKENIZER}" \
    --tensorboard-dir "${OUT}/tensorboard" \
    --deepspeed --no-pipeline-parallel --deepspeed_config="${DS_CONFIG}" \
    --zero-stage=1 --bf16 --shuffle-sample-in-corpus --blend-sample-in-corpus \
    --accumulate-allreduce-grads-in-fp32 --no-bias-gelu-fusion \
    --no-bias-dropout-fusion --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion --optimizer=adamw \
    --tensor-model-parallel-size=1 --pipeline-model-parallel-size=1 \
    --max-position-embeddings="${MDS_SEQ}" --micro-batch-size=1 \
    --ds-sequence-parallel-size=1 --global-batch-size=192 --split=990,10,0 \
    --timing-log-level=1 --eval-interval=100000 --eval-iters=1 \
    --save-interval=1000000 --log-interval=1 \
    --save="${OUT}/checkpoints" \
    --seq-length="${MDS_SEQ}" --num-layers=10 --hidden-size=4096 \
    --train-iters="${MDS_STEPS}" --distributed-backend=ccl --weight-decay=0.1 \
    --adam-beta1=0.9 --adam-beta2=0.95 --adam-eps=0.00001 --clip-grad=1.0 \
    --num-attention-heads=32 --data-cache-path="${OUT}/index-cache" \
    --data-file-list="${BOOKS_LIST}" "${PROFILE_ARGS[@]}" \
    >> "${LOG}" 2>&1
rc=$?
echo "END $(date -Is) rc=${rc} output=${OUT}" | tee -a "${LOG}"
exit "${rc}"
