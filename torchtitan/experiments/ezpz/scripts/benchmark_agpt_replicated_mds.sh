#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt2b-mds-zero0
#PBS -l walltime=04:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Replicated SophiaG/ZeRO-0 baseline for the Sunspot scaling comparison.
# Submit only through the experiment alcf-job-preflight/submit.sh wrapper.
set -o pipefail
ulimit -c 0

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-replicated-20261031}"
RUN_ROOT="${CAMPAIGN_ROOT}"
TT_ROOT="${CAMPAIGN_ROOT}"
MDS_ROOT="${CAMPAIGN_ROOT}/mds-source"
MDS_BASE_HEAD="b5617bac7b4e2303e738ef7db0307915da905941"
MDS_VENV="/lus/tegu/projects/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/sunspot/Megatron-DeepSpeed-aurora_frameworks-2025.3.1"
EZPZ_HELPER_ENV="/lus/tegu/projects/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/sunspot/Megatron-DeepSpeed-deepspeed_0.18.5_no_deps_conda_vllm_0.15.0_nre_ipex_2.10.10_pytorch_2.10.0_rel_oneapi_2025.3.1_numpy_2.2.6_python_3.12.8"
PYTHON_OVERLAY="${CAMPAIGN_ROOT}/mds-python-overlay"
DISTUTILS_OVERLAY="${CAMPAIGN_ROOT}/mds-distutils-overlay"
BOOKS_LIST="${TT_ROOT}/torchtitan/experiments/ezpz/data-lists/sunspot/books.txt"
LOAD_ROOT="${CAMPAIGN_ROOT}/mds-load"
DS_CONFIG_TEMPLATE="${MDS_ROOT}/ds-configs/ds_stage1_mb1_gb384_pp1_bf16_ws12.json"

source "${EZPZ_HELPER_ENV}/lib/python3.12/site-packages/ezpz/bin/utils.sh" || exit 90
ezpz_setup_job || exit $?
# This MDS runtime was built against oneAPI 2025.3.1 and MKL .so.2.
module load oneapi/release/2025.3.1 hdf5 pti-gpu || exit $?
# The oneAPI module resets this to COMPOSITE. Match the TT flat-tile setup
# after all modules have loaded.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

export VIRTUAL_ENV="${MDS_VENV}"
# Use the installed ezpz CLI and the exact MDS interpreter as its payload.
export PATH="${MDS_VENV}/bin:${EZPZ_HELPER_ENV}/bin:/opt/pbs/bin:${PATH}"
export PYTHONPATH="${MDS_ROOT}:${PYTHON_OVERLAY}:${DISTUTILS_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
hash -r
[[ "$(command -v python)" == "${MDS_VENV}/bin/python" ]] || exit 93
EZPZ_LAUNCHER="${EZPZ_HELPER_ENV}/bin/ezpz"
[[ "$(command -v ezpz)" == "${EZPZ_LAUNCHER}" ]] || exit 95
"${MDS_VENV}/bin/python" - <<'PY' || exit 94
import torch

if not torch.xpu.is_available():
    raise SystemExit("Intel XPU is unavailable in the allocated PBS job")
print(f"xpu_count={torch.xpu.device_count()}")
PY

MDS_ZERO_STAGE="${MDS_ZERO_STAGE:-0}"
MDS_OPTIMIZER="${MDS_OPTIMIZER:-sophiag}"
case "${MDS_ZERO_STAGE}" in 0|1) ;; *) echo "MDS_ZERO_STAGE must be 0 or 1" >&2; exit 98 ;; esac
case "${MDS_OPTIMIZER}" in
    # Witness hooks only instrument megatron.optimizer.sophia.
    sophiag) OPT_ARGS=(--optimizer sophiag --sophiag-beta1 .9 --sophiag-beta2 .95 --sophiag-rho .01); WITNESS_ARGS=(--benchmark-sophia-witness) ;;
    adamw) OPT_ARGS=(--optimizer adamw --adam-beta1 .9 --adam-beta2 .95 --adam-eps 1e-5); WITNESS_ARGS=() ;;
    *) echo "MDS_OPTIMIZER must be sophiag or adamw" >&2; exit 98 ;;
esac
RUN_KIND="${RUN_KIND:-zero${MDS_ZERO_STAGE}-${MDS_OPTIMIZER}}"
GRAD_POLICY="${GRAD_POLICY:-production}"
case "${GRAD_POLICY}" in
    production|fp32_wire) ;;
    *) echo "GRAD_POLICY must be production or fp32_wire" >&2; exit 98 ;;
esac
STEPS="${STEPS:-32}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"
EVAL_ITERS="${EVAL_ITERS:-1}"
NODES="${NODES:-2}"
NPROC_PER_NODE=12
if [[ ! "${NODES}" =~ ^[1-9][0-9]*$ || ! "${STEPS}" =~ ^[0-9]+$ || ! "${WARMUP_ITERS}" =~ ^[0-9]+$ ]] || ((STEPS < 10 || STEPS <= WARMUP_ITERS)); then
    echo "Require positive NODES and integer STEPS>=10 greater than WARMUP_ITERS" >&2
    exit 98
fi
ALLOCATED_NODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
if ((ALLOCATED_NODES != NODES)); then
    echo "PBS allocation node count ${ALLOCATED_NODES} differs from NODES=${NODES}" >&2
    exit 98
fi
NPROC=$((NODES * NPROC_PER_NODE))
export BENCH_EXPECTED_WORLD_SIZE="${NPROC}"
GLOBAL_BATCH_SIZE=$((NPROC * 2))
PROFILE="${PROFILE:-0}"
if [[ "${PROFILE}" != 0 && "${PROFILE}" != 1 ]]; then
    echo "PROFILE must be 0 or 1" >&2
    exit 98
fi
if [[ "${PROFILE}" == 1 ]] && ((STEPS < 12)); then
    echo "PROFILE=1 requires at least 12 updates" >&2
    exit 98
fi
if [[ -z "${BENCH_EXPECTED_TT_HEAD:-}" ]]; then
    echo "BENCH_EXPECTED_TT_HEAD must pin the submitted TT checkout" >&2
    exit 99
fi
[[ "$(git -C "${TT_ROOT}" rev-parse HEAD)" == "${BENCH_EXPECTED_TT_HEAD}" ]] || exit 99
if [[ -z "${BENCH_EXPECTED_MDS_HEAD:-}" ]]; then
    echo "BENCH_EXPECTED_MDS_HEAD must pin the submitted MDS checkout" >&2
    exit 99
fi
[[ "$(git -C "${MDS_ROOT}" rev-parse HEAD)" == "${BENCH_EXPECTED_MDS_HEAD}" ]] || exit 99
git -C "${MDS_ROOT}" merge-base --is-ancestor "${MDS_BASE_HEAD}" "${BENCH_EXPECTED_MDS_HEAD}" || exit 99
[[ -z "$(git -C "${MDS_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 99
[[ "$(realpath "${PBS_O_WORKDIR}")" == "$(realpath "${TT_ROOT}")" ]] || exit 99
case "$(realpath "${TT_ROOT}")" in /lus/tegu/projects/datascience/foremans/projects/saforem2/*) ;; *) exit 99 ;; esac
[[ -f "${LOAD_ROOT}/global_step138650/mp_rank_00_model_states.pt" ]] || exit 99
case "$(realpath "${LOAD_ROOT}/global_step138650/mp_rank_00_model_states.pt")" in /lus/tegu/projects/*) ;; *) exit 99 ;; esac
[[ -f "${BOOKS_LIST}" && -d "${PYTHON_OVERLAY}" && -d "${DISTUTILS_OVERLAY}" ]] || exit 99
cd "${MDS_ROOT}" || exit 99
JOBID_SHORT="${PBS_JOBID%%.*}"
RUN_NAME="mds-${RUN_KIND}-${GRAD_POLICY}-ac${MDS_AC:-none}-n${NODES}-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${RUN_ROOT}/outputs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}/tensorboard" "${OUTPUT_DIR}/wandb" \
    "${OUTPUT_DIR}/data-cache/books/index-cache" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
DS_CONFIG="${OUTPUT_DIR}/deepspeed-config.json"
PROVENANCE_DIR="${OUTPUT_DIR}/provenance"
export AGPT_BENCH_BATCH_HASH_DIR="${OUTPUT_DIR}/input-hashes"
export AGPT_BENCH_BATCH_MICROBATCHES="$((STEPS * 2))"
export AGPT_BENCH_BATCH_HELPER="${TT_ROOT}/torchtitan/experiments/ezpz/scripts/agpt_batch_witness.py"
mkdir -p "${PROVENANCE_DIR}"
cp -- "${BASH_SOURCE[0]}" "${PROVENANCE_DIR}/submitted-launcher.sh"
sha256sum "${PROVENANCE_DIR}/submitted-launcher.sh" > "${PROVENANCE_DIR}/submitted-launcher.sha256"
cp -- "${PBS_O_WORKDIR}/torchtitan/experiments/ezpz/scripts/benchmark_agpt_entrypoint.py" "${PROVENANCE_DIR}/benchmark_agpt_entrypoint.py"
git -C "${MDS_ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/mds-head.txt"
printf '%s\n' "${MDS_BASE_HEAD}" > "${PROVENANCE_DIR}/mds-base-head.txt"
git -C "${MDS_ROOT}" diff --binary "${MDS_BASE_HEAD}" HEAD > "${PROVENANCE_DIR}/mds-base-to-head.patch"
cp -- "${MDS_ROOT}/megatron/profiler.py" "${PROVENANCE_DIR}/mds-profiler.py"
git -C "${MDS_ROOT}" diff --binary HEAD > "${PROVENANCE_DIR}/mds-diff.patch"
git -C "${PBS_O_WORKDIR}" rev-parse HEAD > "${PROVENANCE_DIR}/tt-head.txt"
cp -- "${MDS_ROOT}/megatron/optimizer/sophia.py" "${PROVENANCE_DIR}/mds-sophia.py"
cp -- "${MDS_ROOT}/megatron/optimizer/__init__.py" "${PROVENANCE_DIR}/mds-optimizer.py"
cp -- "${MDS_ROOT}/megatron/training.py" "${PROVENANCE_DIR}/mds-training.py"
cp -- "${MDS_ROOT}/megatron/checkpointing.py" "${PROVENANCE_DIR}/mds-checkpointing.py"
cp -- "${MDS_ROOT}/megatron/data/helpers.cpython-312-x86_64-linux-gnu.so" "${PROVENANCE_DIR}/"
sha256sum "${PROVENANCE_DIR}/helpers.cpython-312-x86_64-linux-gnu.so" > "${PROVENANCE_DIR}/native-helpers.sha256"
DS_RUNTIME_ROOT=$("${MDS_VENV}/bin/python" -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("deepspeed").origin).parent / "runtime")')
for source_file in engine.py config.py constants.py bf16_optimizer.py fp16/unfused_optimizer.py; do
    cp -- "${DS_RUNTIME_ROOT}/${source_file}" "${PROVENANCE_DIR}/ds-${source_file//\//-}" || exit 99
done
sha256sum "${LOAD_ROOT}/global_step138650/mp_rank_00_model_states.pt" > "${PROVENANCE_DIR}/checkpoint.sha256"
cp -- "${BOOKS_LIST}" "${PROVENANCE_DIR}/books.txt"
for metadata in "${LOAD_ROOT}"/lr_state_dict_*.yaml; do
    [[ -f "${metadata}" ]] && cp -- "${metadata}" "${PROVENANCE_DIR}/"
done

# Derive a job-unique ZeRO config; the model checkpoint supplies no optimizer state.
"${EZPZ_HELPER_ENV}/bin/python" - "${DS_CONFIG_TEMPLATE}" "${DS_CONFIG}" "${GLOBAL_BATCH_SIZE}" "${GRAD_POLICY}" "${MDS_ZERO_STAGE}" <<'PYCONFIG' || exit 96
import json
import sys

template_path, config_path, global_batch_size, grad_policy, zero_stage = sys.argv[1:]
with open(template_path, encoding="utf-8") as stream:
    config = json.load(stream)
config["train_batch_size"] = int(global_batch_size)
config["train_micro_batch_size_per_gpu"] = 1
config["gradient_accumulation_steps"] = 2
config["zero_optimization"] = {"stage": int(zero_stage)}
config["flops_profiler"]["enabled"] = False
config["comms_logger"]["enabled"] = False
assert "communication_data_type" not in config
assert "grad_accum_dtype" not in config.get("data_types", {})
if grad_policy == "fp32_wire":
    # Preserve the legacy BF16 local accumulation; change only wire precision.
    config["communication_data_type"] = "fp32"
with open(config_path, "x", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
print(f"ds_config={config_path} gbs={global_batch_size} microbatch=1 gas=2 zero_stage={zero_stage} grad_policy={grad_policy}")
PYCONFIG
cp -- "${DS_CONFIG}" "${PROVENANCE_DIR}/deepspeed-config.json"

# Keep the run record online; MDS reads WB_PROJECT before initializing W&B.
export WANDB_MODE=online
export WANDB_TEAM=aurora_gpt
export WANDB_ENTITY=aurora_gpt
export WANDB_PROJECT=torchtitan.ezpz.train
export WB_PROJECT=torchtitan.ezpz.train
export WANDB_RUN_GROUP=agpt-replicated-20261031
export WANDB_RUN_GROUP="${BENCH_WANDB_GROUP:-${WANDB_RUN_GROUP}}"
export WANDB_RUN_NAME="${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_ID="agpt-${RUN_NAME}"
export WANDB_RUN_TAGS="agpt-replicated,sunspot,megatron-deepspeed,zero${MDS_ZERO_STAGE},${MDS_OPTIMIZER},${GRAD_POLICY},${RUN_KIND}"
export WANDB_TAGS="agpt-replicated,sunspot,megatron-deepspeed,zero${MDS_ZERO_STAGE},${MDS_OPTIMIZER},${GRAD_POLICY},${RUN_KIND}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"

PROFILE_ARGS=()
if [[ "${PROFILE}" == 1 ]]; then
    PROFILE_ARGS=(--profile pt --profile_steps 12,12 --profile-ranks 0)
fi
MDS_AC="${MDS_AC:-none}"
case "${MDS_AC}" in
    none) AC_ARGS=() ;;
    full) AC_ARGS=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
    *) echo "MDS_AC must be none or full" >&2; exit 98 ;;
esac

{
    echo "===== AuroraGPT-2B matched Megatron-DeepSpeed run ====="
    date -Is
    echo "host=$(hostname -s)"
    echo "pbs_jobid=${PBS_JOBID:-unset}"
    echo "run_name=${RUN_NAME} run_kind=${RUN_KIND} world_size=${NPROC} ppn=${NPROC_PER_NODE}"
    echo "steps=${STEPS} seed=42 lr=2e-6 warmup=${WARMUP_ITERS} eval_iters=${EVAL_ITERS}"
    echo "ze_flat_device_hierarchy=${ZE_FLAT_DEVICE_HIERARCHY} oneapi_device_selector=${ONEAPI_DEVICE_SELECTOR:-unset}"
    echo "modules=$(module -t list 2>&1 | tr '\n' ';')"
    echo "torch=$(${MDS_VENV}/bin/python -c 'import torch; print(torch.__version__)')"
    echo "deepspeed=$(${MDS_VENV}/bin/python -c 'import deepspeed; print(deepspeed.__version__)')"
    echo "ezpz=$(${EZPZ_HELPER_ENV}/bin/python -c 'import ezpz; print(ezpz.__file__)')"
    echo "mds_head=$(git -C "${MDS_ROOT}" rev-parse HEAD) mds_base_head=${MDS_BASE_HEAD}"
    echo "mds_source_root=${MDS_ROOT} campaign_root=${CAMPAIGN_ROOT}"
    echo "load_root=${LOAD_ROOT} load_tag=global_step138650"
    echo "data_list=${BOOKS_LIST} split=95,5,0 gbs=${GLOBAL_BATCH_SIZE} gas=2 microbatch=1 seq=8192"
    echo "zero_stage=${MDS_ZERO_STAGE} optimizer=${MDS_OPTIMIZER} args=${OPT_ARGS[*]} weight_decay=.1 clip_grad=1.0"
    echo "grad_policy=${GRAD_POLICY} local_accumulation=legacy_default_runtime_unverified"
    echo "optimizer_state=fresh hessian_policy=no_update_hessian_call_in_pinned_source step_bs_default=5120"
    echo "profile=${PROFILE} deepspeed_flops_profiler=disabled activation_checkpoint=${MDS_AC}"
    echo "data_list_sha256=$(sha256sum "${BOOKS_LIST}" | cut -d' ' -f1)"
    echo "ezpz_entrypoint=${EZPZ_LAUNCHER} python=${MDS_VENV}/bin/python"
    echo "output=${OUTPUT_DIR} wandb_mode=${WANDB_MODE} wandb_project=${WB_PROJECT}"
} | tee "${LOG_FILE}"

# Log actual local device properties once per launched rank before training.
ezpz launch \
    --nproc "${NPROC}" --nproc_per_node "${NPROC_PER_NODE}" -- \
    "${MDS_VENV}/bin/python" -c 'import json, os, torch; from ezpz.distributed import get_device_properties, get_local_rank, get_rank, get_world_size; rank=get_rank(); local_rank=get_local_rank(); world_size=get_world_size(in_use=True); assert world_size==int(os.environ["BENCH_EXPECTED_WORLD_SIZE"]), world_size; assert torch.xpu.device_count()==12, torch.xpu.device_count(); torch.xpu.set_device(local_rank); props=get_device_properties(local_rank); raw_props=torch.xpu.get_device_properties(local_rank); print("DEVICE_PREFLIGHT " + json.dumps({"rank":rank,"local_rank":local_rank,"world_size":world_size,"device_count":torch.xpu.device_count(),"device":props,"device_properties":str(raw_props),"torch":torch.__version__,"ze_flat_device_hierarchy":os.environ.get("ZE_FLAT_DEVICE_HIERARCHY")},sort_keys=True), flush=True)' \
    2>&1 | tee -a "${LOG_FILE}"
preflight_rc=${PIPESTATUS[0]}
if [[ ${preflight_rc} -ne 0 ]]; then
    echo "DEVICE_PREFLIGHT_FAILED rc=${preflight_rc}" | tee -a "${LOG_FILE}"
    printf '%s\n' "${preflight_rc}" > "${OUTPUT_DIR}/exit-code.txt"
    tail -n 100 "${LOG_FILE}" >&2
    exit "${preflight_rc}"
fi

exec 3>>"${LOG_FILE}"
BASH_XTRACEFD=3
set -x
ezpz launch \
    --nproc "${NPROC}" --nproc_per_node "${NPROC_PER_NODE}" -- \
    "${MDS_VENV}/bin/python" "${PROVENANCE_DIR}/benchmark_agpt_entrypoint.py" \
    --benchmark-output="${OUTPUT_DIR}/rank-diagnostics" \
    "${WITNESS_ARGS[@]}" \
    --benchmark-script="${MDS_ROOT}/pretrain_gpt_alcf.py" -- \
    --finetune \
    --override-opt_param-scheduler \
    --load "${LOAD_ROOT}" \
    --load-tag global_step138650 \
    --save "${OUTPUT_DIR}/checkpoints" \
    --train-iters "${STEPS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size 1 \
    --seq-length 8192 \
    --max-position-embeddings 131072 \
    --num-layers 12 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --num-key-value-heads 4 \
    --ffn-hidden-size 11008 \
    --tokenizer-type HFTokenizer \
    --tokenizer-model google/gemma-7B \
    --normalization rmsnorm \
    --disable-bias-linear \
    --no-query-key-layer-scaling \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-theta 50000 \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --adjust-word-embedding-init \
    --init-method-std .013964 \
    --word-embedding-init-std .632455532 \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --bf16 \
    --deepspeed \
    --deepspeed_config="${DS_CONFIG}" \
    --zero-stage "${MDS_ZERO_STAGE}" \
    --no-pipeline-parallel \
    --pipeline-model-parallel-size 1 \
    --tensor-model-parallel-size 1 \
    --distributed-backend ccl \
    "${OPT_ARGS[@]}" \
    --weight-decay .1 \
    --clip-grad 1.0 \
    --lr 2e-6 \
    --lr-decay-style constant \
    --lr-warmup-iters "${WARMUP_ITERS}" \
    --lr-decay-iters "${STEPS}" \
    --min-lr 2e-6 \
    --seed 42 \
    --data-file-list "${BOOKS_LIST}" \
    --data-cache-path "${OUTPUT_DIR}/data-cache/books/index-cache" \
    --split 95,5,0 \
    --shuffle-sample-in-corpus \
    --eval-interval 10000 \
    --eval-iters "${EVAL_ITERS}" \
    --save-interval 1000000 \
    --log-interval 1 \
    --log-timers-to-tensorboard \
    --log-optimizer-states-to-tensorboard \
    --log-memory-to-tensorboard \
    --tensorboard-dir "${OUTPUT_DIR}/tensorboard" \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --use-flash-attn-builder \
    "${PROFILE_ARGS[@]}" "${AC_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
rc=${PIPESTATUS[0]}
set +x
exec 3>&-
printf '%s\n' "${rc}" > "${OUTPUT_DIR}/exit-code.txt"
echo "launch_end=$(date -Is) rc=${rc}" | tee -a "${LOG_FILE}"
if [[ ${rc} -ne 0 ]]; then
    echo "VERDICT: failed rc=${rc}" | tee -a "${LOG_FILE}"
    tail -n 100 "${LOG_FILE}" >&2
    exit "${rc}"
fi
echo "VERDICT: ok" | tee -a "${LOG_FILE}"
