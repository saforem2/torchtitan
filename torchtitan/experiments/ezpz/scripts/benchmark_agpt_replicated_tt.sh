#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-replicated-tt
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Derived from sync_smoke_sunspot.sh and the native-checkpoint benchmark.
set -o pipefail
ulimit -c 0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT=600
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export WANDB_MODE=online
export WANDB_ENTITY=aurora_gpt
export WANDB_TEAM=aurora_gpt
export WANDB_PROJECT=torchtitan.ezpz.train
export WANDB_RUN_GROUP=agpt-replicated-20261031

ROOT="${PBS_O_WORKDIR:?submit from the isolated checkout}"
cd "${ROOT}" || exit 2
EXPECTED="${BENCH_EXPECTED_TT_HEAD:?provide the pinned commit}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED}" ]] || exit 91
[[ -z "$(git diff --name-only HEAD)" ]] || exit 92
case "$(readlink -f "${ROOT}")" in
    /lus/tegu/projects/datascience/foremans/projects/saforem2/*) ;;
    *) echo "Workdir must be on the project filesystem" >&2; exit 93;;
esac
ARM="${ARM:-probe}"
NODES="${NODES:-2}"
STEPS="${STEPS:-32}"
PPN="${PPN:-12}"
case "${PPN}" in 8|12) ;; *) echo "PPN must be 8 or 12" >&2; exit 2;; esac
NPROC=$((NODES * PPN))
[[ "$(sort -u "${PBS_NODEFILE}" | wc -l)" -eq "${NODES}" ]] || exit 97
export BENCH_EXPECTED_WORLD_SIZE="${NPROC}"
GBS=$((NPROC * 2))
TOKENS=$((GBS * 8192))
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/agpt-replicated-20261031/${ARM}-${JOB}"
mkdir -p "${OUT}/provenance" "${OUT}/wandb"
LOG="${OUT}/run.log"
exec > >(tee -a "${LOG}") 2>&1
export WANDB_RUN_ID="agpt-replicated-${ARM}-${JOB}"
export WANDB_NAME="${ARM}-${NODES}n-${JOB}"
export WANDB_RUN_NAME="${WANDB_NAME}"
export WANDB_DIR="${OUT}/wandb"
export WANDB_TAGS="agpt-replicated,sunspot,${ARM},${NODES}nodes"
export WANDB_RUN_TAGS="${WANDB_TAGS}"
export AGPT_BENCH_BATCH_HASH_DIR="${OUT}/input-hashes"
export AGPT_BENCH_BATCH_MICROBATCHES="$((STEPS * 2))"
export TORCHTITAN_AGPT_DTYPE_PROBE=1
VENV="${BENCH_VENV:-/lus/tegu/projects/datascience/foremans/venvs/xpu-torch214-fixedlr-v2-20260927}"
source "${ROOT}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job || exit $?
ezpz_load_modules || exit $?
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# The pinned venv needs its matching SYCL/UR ahead of the module libraries.
# Keep MPI and libfabric from the system, as in the validated house launcher.
SHIM="${ROOT}/.cache/xpu-ld-first/$(basename "${VENV}")"
mkdir -p "${SHIM}"
for library in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* \
               "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
    [[ -e "${library}" ]] && ln -sf "${library}" "${SHIM}/"
done
export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MDS_ANNEAL_BASE="/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-matched-20261024/outputs/agpt-matched-native-weights/dcp/step-0"
export TORCHINDUCTOR_CACHE_DIR="/tmp/agpt-replicated-${JOB}/inductor"
export TRITON_CACHE_DIR="/tmp/agpt-replicated-${JOB}/triton"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || exit 94
[[ "$(command -v ezpz)" == "${VENV}/bin/ezpz" ]] || exit 95
# This copied venv's installed console script retains its source shebang.
# Rebind a job-local copy; keep the borrowed runtime itself immutable.
CONSOLE_DIR="${OUT}/provenance/bin"
mkdir -p "${CONSOLE_DIR}"
"${VENV}/bin/python" - "${VENV}/bin/ezpz" "${CONSOLE_DIR}/ezpz" "${VENV}/bin/python" <<'PY' || exit 95
from pathlib import Path
import sys
source, target, interpreter = sys.argv[1:]
body = Path(source).read_text().partition("\n")[2]
Path(target).write_text("#!" + interpreter + "\n" + body)
Path(target).chmod(0o755)
PY
export PATH="${CONSOLE_DIR}:${PATH}"
hash -r
ezpz launch --help > "${OUT}/provenance/ezpz-launch-help.txt" 2>&1 || exit 95
"${VENV}/bin/python" - <<'PY' || exit 96
import torch
assert torch.xpu.is_available()
assert torch.xpu.device_count() == 12, torch.xpu.device_count()
print("RUNTIME", torch.__version__, torch.__file__, torch.xpu.device_count())
PY
git rev-parse HEAD > "${OUT}/provenance/tt-head.txt"
git status --short > "${OUT}/provenance/git-status.txt"
cp "${BASH_SOURCE[0]}" "${OUT}/provenance/submitted-launcher.sh"
cp "${ROOT}/torchtitan/experiments/ezpz/scripts/benchmark_agpt_entrypoint.py" "${OUT}/provenance/"
module -t list 2> "${OUT}/provenance/modules.txt"
echo "BEGIN $(date -Is) host=$(hostname -s) job=${PBS_JOBID} sha=${EXPECTED}"
echo "ARM=${ARM} nodes=${NODES} world=${NPROC} ppn=${PPN} gbs=${GBS} microbatch=1 gas=2 seq=8192 steps=${STEPS}"
echo "W&B https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/${WANDB_RUN_ID}"

if [[ "${ARM}" == probe || "${ARM}" == probe-bf16 ]]; then
    COMM_DTYPE=float32
    [[ "${ARM}" == probe-bf16 ]] && COMM_DTYPE=bfloat16
    COMMAND=(ezpz launch --nproc "${NPROC}" --nproc_per_node "${PPN}" --
        "${VENV}/bin/python" -m torchtitan.experiments.ezpz.scripts.probe_native_ddp_accumulation
        --output "${OUT}/accumulation-probe" --communication-dtype="${COMM_DTYPE}")
else
    case "${ARM}" in
        ddp-eager|ddp-losscompile|ddp-bf16-eager|ddp-bf16-losscompile)
            COMM_DTYPE=float32
            [[ "${ARM}" == *bf16* ]] && COMM_DTYPE=bfloat16
            PARALLEL=(--parallelism.data-parallel-shard-degree=1
                --parallelism.data-parallel-replicate-degree="${NPROC}"
                --parallelism.enable-data-parallel-native-ddp
                --parallelism.native-ddp-bucket-cap-mb="${DDP_BUCKET_MB:-25}"
                --parallelism.native-ddp-communication-dtype="${COMM_DTYPE}")
            ;;
        fsdp-eager|fsdp-losscompile)
            PARALLEL=(--parallelism.data-parallel-shard-degree="${NPROC}"
                --parallelism.data-parallel-replicate-degree=1)
            ;;
        *) echo "Unknown ARM=${ARM}" >&2; exit 2;;
    esac
    COMPILE=(--compile.no-enable)
    [[ "${ARM}" == *losscompile ]] && COMPILE=(--compile.enable --compile.components=loss)
    PROFILE=()
    if [[ "${BENCH_PROFILE:-1}" == 1 && "${STEPS}" -ge 21 ]]; then
        PROFILE=(--profiler.enable-profiling --profiler.profile-freq=7
            --profiler.profiler-warmup=3 --profiler.profiler-active=1
            --profiler.profiler-repeat=1 --profiler.profiler-skip-first=5)
    fi
    # Per-step allocator snapshots; an OOM also forces a final dump.
    [[ "${BENCH_MEMORY_SNAPSHOT:-0}" == 1 ]] && PROFILE+=(
        --profiler.enable-memory-snapshot --profiler.memory-snapshot-freq=1)
    AC="${BENCH_AC:-none}"
    case "${AC}" in none|full) ;; *) echo "Unknown BENCH_AC=${AC}" >&2; exit 2;; esac
    echo "activation_checkpoint=${AC} memory_snapshot=${BENCH_MEMORY_SNAPSHOT:-0}"
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
        --dataloader.dataset-path="${ROOT}/torchtitan/experiments/ezpz/data-lists/sunspot/books.txt"
        --dataloader.train-iters="${STEPS}" --dataloader.split=95,5,0
        --dataloader.data-cache-path="${ROOT}/.cache/books-gbs${GBS}/index-cache"
        --debug.seed=42 --debug.no-deterministic --metrics.log-freq=1
        --metrics.enable-tensorboard --metrics.enable-wandb
        --dump-folder="${OUT}" --checkpointer.folder=checkpoints
        --checkpointer.interval=10000 --checkpointer.keep-latest-k=0
        "${COMPILE[@]}" "${PROFILE[@]}" "activation-checkpoint:${AC}")
fi
printf '%q ' "${COMMAND[@]}" > "${OUT}/provenance/launch-command.sh"
printf '\n' >> "${OUT}/provenance/launch-command.sh"
"${COMMAND[@]}"
rc=$?
echo "END $(date -Is) rc=${rc} output=${OUT}"
exit "${rc}"
