#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-replicated-tt-aurora
#PBS -l walltime=02:00:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q debug-scaling
#PBS -j oe

# Aurora version of benchmark_agpt_replicated_tt.sh. For a 256-node scaling
# run, override the PBS request and queue at submission:
#   qsub -q prod -l select=256 -l walltime=12:00:00 -v NODES=256,ARM=ddp-eager,BENCH_EXPECTED_TT_HEAD=<commit>,BENCH_INITIAL_LOAD_PATH=<converted-dcp-step-dir> <this-script>
# Keep the source, tokenizer, checkpoint, and Books shards on flare.
set -o pipefail
ulimit -c 0

export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export NGPU_PER_HOST=12
export NGPU_PER_TILE=6
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

SUBMIT_DIR="${PBS_O_WORKDIR:?submit from the isolated Aurora checkout}"
ROOT="$(readlink -f "${SUBMIT_DIR}")"
cd "${ROOT}" || exit 2
EXPECTED="${BENCH_EXPECTED_TT_HEAD:?provide the pinned TorchTitan commit}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED}" ]] || {
    echo "HEAD $(git rev-parse HEAD) differs from BENCH_EXPECTED_TT_HEAD=${EXPECTED}" >&2
    exit 91
}
[[ -z "$(git diff --name-only HEAD)" ]] || {
    echo "tracked checkout changes are not allowed" >&2
    git diff --name-only HEAD >&2
    exit 92
}
case "${ROOT}" in
    /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-agpt-replicated-20261031) ;;
    *) echo "unexpected worktree realpath: ${ROOT}" >&2; exit 93 ;;
esac

ARM="${ARM:-probe}"
NODES="${NODES:-2}"
STEPS="${STEPS:-32}"
PPN=12
if [[ ! "${NODES}" =~ ^[1-9][0-9]*$ || ! "${STEPS}" =~ ^[0-9]+$ ]] || (( STEPS < 3 )); then
    echo "NODES must be positive and STEPS must be at least 3" >&2
    exit 94
fi
ALLOCATED_NODES="$(sort -u "${PBS_NODEFILE}" | wc -l | tr -d ' ')"
[[ "${ALLOCATED_NODES}" == "${NODES}" ]] || {
    echo "PBS allocation has ${ALLOCATED_NODES} nodes; NODES=${NODES}" >&2
    exit 95
}
NPROC=$((NODES * PPN))
export BENCH_EXPECTED_WORLD_SIZE="${NPROC}"
GBS=$((NPROC * 2))
TOKENS=$((GBS * 8192))
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/agpt-replicated-20261031/aurora-${ARM}-${JOB}"
mkdir -p "${OUT}/provenance" "${OUT}/wandb"
export TORCHINDUCTOR_CACHE_DIR="/tmp/agpt-replicated-${JOB}/inductor"
export TRITON_CACHE_DIR="/tmp/agpt-replicated-${JOB}/triton"
LOG="${OUT}/run.log"
exec > >(tee -a "${LOG}") 2>&1

export WANDB_MODE=online
export WANDB_ENTITY=aurora_gpt
export WANDB_TEAM=aurora_gpt
export WANDB_PROJECT=torchtitan.ezpz.train
export WANDB_RUN_GROUP="${BENCH_WANDB_GROUP:-agpt-replicated-20261031}"
export WANDB_RUN_ID="agpt-aurora-${ARM}-${JOB}"
export WANDB_NAME="aurora-${ARM}-${NODES}n-${JOB}"
export WANDB_RUN_NAME="${WANDB_NAME}"
export WANDB_TAGS="agpt-replicated,aurora,${ARM},${NODES}nodes"
export WANDB_RUN_TAGS="${WANDB_TAGS}"
export WANDB_DIR="${OUT}/wandb"

VENV="${BENCH_VENV:-/flare/AuroraGPT/foremans/venvs/xpu-torch215}"
EZPZ_UTILS="${ROOT}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
source "${EZPZ_UTILS}" || exit 96
load_pinned_ezpz_utils || exit $?
# Aurora's helper loads the framework-matched oneAPI/MPICH modules. Load
# modules first so ezpz_setup_job sees the runtime actually used by the job.
ezpz_load_modules || exit $?
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export NGPU_PER_HOST=12
export NGPU_PER_TILE=6
ezpz_setup_job || exit $?
# The oneAPI module may reset the runtime's default composite tile mapping.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export NGPU_PER_HOST=12
export NGPU_PER_TILE=6

export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || {
    echo "python resolved to $(command -v python), expected ${VENV}/bin/python" >&2
    exit 97
}
[[ -x "${VENV}/bin/ezpz" ]] || { echo "missing ${VENV}/bin/ezpz" >&2; exit 97; }
ORIGINAL_EZPZ_SHEBANG="$(head -n 1 "${VENV}/bin/ezpz")"
[[ "${ORIGINAL_EZPZ_SHEBANG}" == '#!'* ]] || {
    echo "invalid installed ezpz shebang: ${ORIGINAL_EZPZ_SHEBANG}" >&2
    exit 97
}
ORIGINAL_EZPZ_INTERPRETER="${ORIGINAL_EZPZ_SHEBANG#\#!}"
[[ -x "${ORIGINAL_EZPZ_INTERPRETER}" ]] || {
    echo "installed ezpz shebang interpreter is missing: ${ORIGINAL_EZPZ_INTERPRETER}" >&2
    exit 97
}

CONSOLE_DIR="${OUT}/provenance/bin"
mkdir -p "${CONSOLE_DIR}"
printf '%s\n' "${ORIGINAL_EZPZ_SHEBANG}" > "${OUT}/provenance/installed-ezpz-shebang.txt"
"${VENV}/bin/python" - "${VENV}/bin/ezpz" "${CONSOLE_DIR}/ezpz" "${VENV}/bin/python" <<'PY' || exit 98
from pathlib import Path
import sys

source, target, interpreter = sys.argv[1:]
body = Path(source).read_text().partition("\n")[2]
if not body:
    raise SystemExit("installed ezpz console script has no body")
path = Path(target)
path.write_text("#!" + interpreter + "\n" + body)
path.chmod(0o755)
PY
[[ "$(head -n 1 "${CONSOLE_DIR}/ezpz")" == "#!${VENV}/bin/python" ]] || exit 98
export PATH="${CONSOLE_DIR}:${PATH}"
hash -r
[[ "$(command -v ezpz)" == "${CONSOLE_DIR}/ezpz" ]] || exit 98
ezpz launch --help > "${OUT}/provenance/ezpz-launch-help.txt" 2>&1 || exit 98

module -t list > "${OUT}/provenance/modules.txt" 2>&1
if ! grep -Fq 'oneapi/release/2026.1.0' "${OUT}/provenance/modules.txt"; then
    echo "expected Aurora oneapi/release/2026.1.0 module" >&2
    cat "${OUT}/provenance/modules.txt" >&2
    exit 99
fi

git rev-parse HEAD > "${OUT}/provenance/tt-head.txt"
git status --short > "${OUT}/provenance/git-status.txt"
cp "${BASH_SOURCE[0]}" "${OUT}/provenance/submitted-launcher.sh"
sha256sum "${OUT}/provenance/submitted-launcher.sh" > "${OUT}/provenance/submitted-launcher.sha256"
cp "${ROOT}/torchtitan/experiments/ezpz/scripts/benchmark_agpt_entrypoint.py" "${OUT}/provenance/"
env | sort | grep -E '^BENCH_[A-Z0-9_]*(HASH|SHA256)=' > "${OUT}/provenance/benchmark-hashes.env" || true

echo "BEGIN $(date -Is) host=$(hostname -s) job=${PBS_JOBID} head=${EXPECTED}"
echo "ARM=${ARM} nodes=${NODES} world=${NPROC} ppn=${PPN} gbs=${GBS} microbatch=1 gas=2 seq=8192 steps=${STEPS}"
echo "runtime=${VENV} original_ezpz_shebang=${ORIGINAL_EZPZ_SHEBANG} rebound_ezpz=$(command -v ezpz)"
echo "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY} ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"
echo "W&B mode=${WANDB_MODE} group=${WANDB_RUN_GROUP} project=${WANDB_PROJECT} run_id=${WANDB_RUN_ID}"
echo "W&B https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${WANDB_RUN_ID}"

"${VENV}/bin/python" - <<'PY' || exit 100
import blendcorpus
import ezpz
import mpi4py
import torch
from torch.distributed.fsdp import DataParallelMeshDims

assert torch.xpu.is_available(), "Intel XPU is unavailable in the allocated PBS job"
assert torch.xpu.device_count() == 12, torch.xpu.device_count()
print("IMPORT_PREFLIGHT", torch.__version__, torch.version.xpu, ezpz.__file__, blendcorpus.__file__, mpi4py.__file__, DataParallelMeshDims)
PY

DEVICE_PREFLIGHT=(ezpz launch --nproc "${NPROC}" --nproc_per_node "${PPN}" --
    "${VENV}/bin/python" -c 'import json, os, torch; from ezpz.distributed import get_local_rank, get_rank, get_world_size; rank=get_rank(); local_rank=get_local_rank(); world=get_world_size(in_use=True); assert world==int(os.environ["BENCH_EXPECTED_WORLD_SIZE"]), world; assert torch.xpu.device_count()==12, torch.xpu.device_count(); assert 0<=local_rank<12, local_rank; torch.xpu.set_device(local_rank); print("DEVICE_PREFLIGHT " + json.dumps({"rank":rank,"local_rank":local_rank,"world_size":world,"device_count":torch.xpu.device_count(),"device":str(torch.xpu.get_device_properties(local_rank)),"torch":torch.__version__,"ze_flat_device_hierarchy":os.environ.get("ZE_FLAT_DEVICE_HIERARCHY")},sort_keys=True), flush=True)')
if ! "${DEVICE_PREFLIGHT[@]}" > "${OUT}/provenance/device-preflight.log" 2>&1; then
    echo "distributed device preflight failed; output follows:" >&2
    tail -n 100 "${OUT}/provenance/device-preflight.log" >&2
    exit 100
fi

if [[ "${ARM}" == probe || "${ARM}" == probe-bf16 ]]; then
    PROBE_ARGS=()
    if [[ "${ARM}" == probe-bf16 ]]; then
        PROBE_ARGS=(--communication-dtype=bfloat16)
    fi
    COMMAND=(ezpz launch --nproc "${NPROC}" --nproc_per_node "${PPN}" --
        "${VENV}/bin/python" -m torchtitan.experiments.ezpz.scripts.probe_native_ddp_accumulation
        --output "${OUT}/accumulation-probe" "${PROBE_ARGS[@]}")
else
    case "${ARM}" in
        ddp-eager|ddp-losscompile|ddp-bf16-eager|ddp-bf16-losscompile)
            COMM_DTYPE=float32
            [[ "${ARM}" == *bf16* ]] && COMM_DTYPE=bfloat16
            PARALLEL=(--parallelism.data-parallel-shard-degree=1
                --parallelism.data-parallel-replicate-degree="${NPROC}"
                --parallelism.enable-data-parallel-native-ddp
                --parallelism.native-ddp-communication-dtype="${COMM_DTYPE}")
            ;;
        fsdp-eager|fsdp-losscompile)
            PARALLEL=(--parallelism.data-parallel-shard-degree="${NPROC}"
                --parallelism.data-parallel-replicate-degree=1)
            ;;
        *) echo "Unknown ARM=${ARM}" >&2; exit 2 ;;
    esac

    INITIAL_LOAD_PATH="${BENCH_INITIAL_LOAD_PATH:?training arms require BENCH_INITIAL_LOAD_PATH to the converted DCP step directory}"
    [[ "${INITIAL_LOAD_PATH}" == /* && -s "${INITIAL_LOAD_PATH}/.metadata" ]] || {
        echo "invalid DCP path: ${INITIAL_LOAD_PATH} (expected absolute path with .metadata)" >&2
        exit 101
    }
    export MDS_ANNEAL_BASE="${INITIAL_LOAD_PATH}"
    export AGPT_BENCH_BATCH_HASH_DIR="${OUT}/input-hashes"
    export AGPT_BENCH_BATCH_MICROBATCHES="$((STEPS * 2))"
    export TORCHTITAN_AGPT_DTYPE_PROBE=1
    printf 'AGPT_BENCH_BATCH_HASH_DIR=%s\nAGPT_BENCH_BATCH_MICROBATCHES=%s\nTORCHTITAN_AGPT_DTYPE_PROBE=%s\n' \
        "${AGPT_BENCH_BATCH_HASH_DIR}" "${AGPT_BENCH_BATCH_MICROBATCHES}" \
        "${TORCHTITAN_AGPT_DTYPE_PROBE}" > "${OUT}/provenance/agpt-diagnostics.env"

    DATA_LIST="${BENCH_DATA_LIST:-${ROOT}/torchtitan/experiments/ezpz/data-lists/aurora/books.txt}"
    DATA_LIST="$(readlink -f "${DATA_LIST}")"
    HF_ASSETS="${ROOT}/assets/hf/gemma-7b"
    [[ -s "${DATA_LIST}" ]] || { echo "missing Aurora data list: ${DATA_LIST}" >&2; exit 102; }
    for asset in tokenizer.json tokenizer_config.json tokenizer.model special_tokens_map.json; do
        [[ -s "${HF_ASSETS}/${asset}" ]] || {
            echo "missing Gemma tokenizer asset: ${HF_ASSETS}/${asset}" >&2
            exit 102
        }
    done
    mapfile -t DATA_SHARDS < <(awk 'NF >= 2 { print $2 }' "${DATA_LIST}")
    ((${#DATA_SHARDS[@]} > 0)) || { echo "empty data list: ${DATA_LIST}" >&2; exit 102; }
    MISSING_SHARDS=0
    for shard in "${DATA_SHARDS[@]}"; do
        if [[ ! -r "${shard}.bin" || ! -r "${shard}.idx" ]]; then
            echo "missing/unreadable indexed dataset shard pair: ${shard}.bin / ${shard}.idx" >&2
            MISSING_SHARDS=$((MISSING_SHARDS + 1))
        fi
    done
    ((MISSING_SHARDS == 0)) || exit 102
    sha256sum "${DATA_LIST}" > "${OUT}/provenance/books-list.sha256"
    sha256sum "${INITIAL_LOAD_PATH}/.metadata" > "${OUT}/provenance/initial-dcp-metadata.sha256"
    echo "initial_load_path=${INITIAL_LOAD_PATH} dataset=blendcorpus data_list=${DATA_LIST} shards=${#DATA_SHARDS[@]}"
    echo "data_list_sha256=$(sha256sum "${DATA_LIST}" | cut -d' ' -f1) purpose=matched-software-gate"

    COMPILE=(--compile.no-enable)
    [[ "${ARM}" == *losscompile ]] && COMPILE=(--compile.enable --compile.components=loss)
    PROFILE=()
    if [[ "${BENCH_PROFILE:-1}" == 1 && "${STEPS}" -ge 21 ]]; then
        PROFILE=(--profiler.enable-profiling --profiler.profile-freq=7
            --profiler.profiler-warmup=3 --profiler.profiler-active=1
            --profiler.profiler-repeat=1 --profiler.profiler-skip-first=5)
    fi
    COMMAND=(ezpz launch --nproc "${NPROC}" --nproc_per_node "${PPN}" --
        "${VENV}/bin/python" "${OUT}/provenance/benchmark_agpt_entrypoint.py"
        --benchmark-output="${OUT}/rank-diagnostics"
        --benchmark-sophia-witness
        --benchmark-module=torchtitan.experiments.ezpz.train --
        --module=ezpz.agpt --config=agpt_2b_replicated_benchmark
        --training.steps="${STEPS}" --training.max-context-length=8192
        --training.num-tokens-per-microbatch-per-dp-rank=8192
        --training.num-tokens-per-train-step="${TOKENS}"
        "${PARALLEL[@]}" --dataloader.dataset=blendcorpus
        --dataloader.dataset-path="${DATA_LIST}"
        --dataloader.train-iters="${STEPS}" --dataloader.split=95,5,0
        --dataloader.data-cache-path="${OUT}/data-cache/books/index-cache"
        --debug.seed=42 --debug.no-deterministic --metrics.log-freq=1
        --metrics.enable-tensorboard --metrics.enable-wandb
        --dump-folder="${OUT}" --checkpointer.folder="${OUT}/checkpoints"
        --checkpointer.interval=10000 --checkpointer.keep-latest-k=0
        "${COMPILE[@]}" "${PROFILE[@]}" activation-checkpoint:none)
fi

printf '%q ' "${COMMAND[@]}" > "${OUT}/provenance/launch-command.sh"
printf '\n' >> "${OUT}/provenance/launch-command.sh"
"${COMMAND[@]}"
rc=$?
echo "END $(date -Is) rc=${rc} output=${OUT}"
exit "${rc}"
