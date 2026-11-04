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
# BENCH_CCL_OP_SYNC=0 lets oneCCL return before the device op completes.
# ezpz_setup_job preserves a preset value across module loads.
export CCL_OP_SYNC="${BENCH_CCL_OP_SYNC:-1}"
case "${CCL_OP_SYNC}" in 0|1) ;; *) echo "BENCH_CCL_OP_SYNC must be 0 or 1" >&2; exit 2;; esac
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
DEFAULT_COMPILE_MODE=none
[[ "${ARM}" == *losscompile ]] && DEFAULT_COMPILE_MODE=loss
COMPILE_MODE="${BENCH_COMPILE_MODE:-${DEFAULT_COMPILE_MODE}}"
case "${COMPILE_MODE}" in
    none|loss|model-regions-loss|blocks-loss) ;;
    *) echo "BENCH_COMPILE_MODE must be none, loss, model-regions-loss or blocks-loss" >&2; exit 2;;
esac
case "${ARM}" in
    probe|probe-bf16)
        [[ "${COMPILE_MODE}" == none ]] || {
            echo "BENCH_COMPILE_MODE=${COMPILE_MODE} is only valid for training arms" >&2
            exit 2
        }
        ;;
esac
case "${PPN}" in 8|12) ;; *) echo "PPN must be 8 or 12" >&2; exit 2;; esac
NPROC=$((NODES * PPN))
[[ "$(sort -u "${PBS_NODEFILE}" | wc -l)" -eq "${NODES}" ]] || exit 97
export BENCH_EXPECTED_WORLD_SIZE="${NPROC}"
# Two sequences per rank per update, as microbatch 1 x accumulation 2 or
# microbatch 2 x accumulation 1. Tokens per update are the same either way.
# BENCH_SEQS_PER_RANK and BENCH_SEQ change that budget and the sequence length;
# BENCH_TT_CONFIG replaces the optimizer-derived config (another model shape).
SEQ="${BENCH_SEQ:-8192}"
SEQS_PER_RANK="${BENCH_SEQS_PER_RANK:-2}"
OPTIMIZER="${BENCH_OPTIMIZER:-sophiag}"
case "${OPTIMIZER}" in
    sophiag) TT_CONFIG=agpt_2b_replicated_benchmark; WITNESS=(--benchmark-sophia-witness) ;;
    adamw) TT_CONFIG=agpt_2b_replicated_benchmark_adamw; WITNESS=() ;;
    *) echo "BENCH_OPTIMIZER must be sophiag or adamw" >&2; exit 2;;
esac
TT_CONFIG="${BENCH_TT_CONFIG:-${TT_CONFIG}}"
MICROBATCH="${BENCH_MICROBATCH:-1}"
case "${MICROBATCH}" in 1|2) ;; *) echo "BENCH_MICROBATCH must be 1 or 2" >&2; exit 2;; esac
(( SEQS_PER_RANK % MICROBATCH == 0 )) || { echo "BENCH_SEQS_PER_RANK must be a multiple of BENCH_MICROBATCH" >&2; exit 2; }
GAS=$((SEQS_PER_RANK / MICROBATCH))
GBS=$((NPROC * SEQS_PER_RANK))
TOKENS=$((GBS * SEQ))
JOB="${PBS_JOBID%%.*}"
# BENCH_TAG separates ablation variants of one ARM in one PBS job. When the
# compile mode overrides the ARM-derived default, include it in the run path.
VARIANT_TAG="${BENCH_TAG:-}"
if [[ -n "${BENCH_COMPILE_MODE+x}" && "${COMPILE_MODE}" != "${DEFAULT_COMPILE_MODE}" ]]; then
    VARIANT_TAG="${VARIANT_TAG:+${VARIANT_TAG}-}compile-${COMPILE_MODE}"
fi
TAG="${VARIANT_TAG:+-${VARIANT_TAG}}"
OUT="${ROOT}/outputs/agpt-replicated-20261031/${ARM}${TAG}-${JOB}"
mkdir -p "${OUT}/provenance" "${OUT}/wandb"
LOG="${OUT}/run.log"
exec > >(tee -a "${LOG}") 2>&1
export WANDB_RUN_ID="agpt-replicated-${ARM}${TAG}-${JOB}"
export WANDB_NAME="${ARM}${TAG}-${NODES}n-${JOB}"
export WANDB_RUN_NAME="${WANDB_NAME}"
export WANDB_DIR="${OUT}/wandb"
export WANDB_TAGS="agpt-replicated,sunspot,${ARM},${NODES}nodes,compile-${COMPILE_MODE}"
export WANDB_RUN_TAGS="${WANDB_TAGS}"
export AGPT_BENCH_BATCH_HASH_DIR="${OUT}/input-hashes"
export AGPT_BENCH_BATCH_MICROBATCHES="$((STEPS * GAS))"
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
echo "BENCH_TAG=${BENCH_TAG:-} CCL_OP_SYNC=${CCL_OP_SYNC} fsdp_reshard=${BENCH_FSDP_RESHARD:-default} fsdp_reduce=${BENCH_FSDP_REDUCE_DTYPE:-float32} fsdp_defer=${BENCH_FSDP_DEFER:-0} deterministic=${BENCH_DETERMINISTIC:-0} reduce_probe=${BENCH_REDUCE_PROBE:-0}"
echo "ddp_zero1=${BENCH_DDP_ZERO1:-0} ddp_bucket_mb=${DDP_BUCKET_MB:-25} ddp_bucket_first=${BENCH_DDP_BUCKET_FIRST:-0}"
echo "ARM=${ARM} nodes=${NODES} world=${NPROC} ppn=${PPN} optimizer=${OPTIMIZER} config=${TT_CONFIG} gbs=${GBS} microbatch=${MICROBATCH} gas=${GAS} seq=${SEQ} steps=${STEPS}"
case "${COMPILE_MODE}" in
    none) echo "compile_mode=none components=none" ;;
    loss) echo "compile_mode=loss components=loss" ;;
    model-regions-loss)
        echo "compile_mode=model-regions-loss components=fused_binary_activation,loss full_model_graph=false" ;;
    blocks-loss)
        echo "compile_mode=blocks-loss components=loss per_block_compile=fullgraph" ;;
esac
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
            [[ "${BENCH_DDP_BUCKET_FIRST:-0}" == 1 ]] && PARALLEL+=(
                --parallelism.native-ddp-bucketize-first-iteration)
            ZERO1="${BENCH_DDP_ZERO1:-0}"
            case "${ZERO1}" in
                0) ;;
                float32|bfloat16) PARALLEL+=(--parallelism.native-ddp-zero1
                    --parallelism.native-ddp-zero1-allgather-dtype="${ZERO1}") ;;
                *) echo "BENCH_DDP_ZERO1 must be 0, float32 or bfloat16" >&2; exit 2;;
            esac
            ;;
        fsdp-*|hsdp-*)
            RESHARD="${BENCH_FSDP_RESHARD:-default}"
            REDUCE="${BENCH_FSDP_REDUCE_DTYPE:-float32}"
            case "${RESHARD}" in default|always|never) ;; *) echo "Unknown BENCH_FSDP_RESHARD=${RESHARD}" >&2; exit 2;; esac
            case "${REDUCE}" in float32|bfloat16) ;; *) echo "Unknown BENCH_FSDP_REDUCE_DTYPE=${REDUCE}" >&2; exit 2;; esac
            DEFER=(--parallelism.no-fsdp-defer-gradient-reduction)
            [[ "${BENCH_FSDP_DEFER:-0}" == 1 ]] && DEFER=(--parallelism.fsdp-defer-gradient-reduction)
            FSDP_KNOBS=(--parallelism.fsdp-reshard-after-forward="${RESHARD}"
                --training.mixed-precision-reduce="${REDUCE}" "${DEFER[@]}")
            case "${ARM}" in
                fsdp-eager|fsdp-losscompile)
                    PARALLEL=(--parallelism.data-parallel-shard-degree="${NPROC}"
                        --parallelism.data-parallel-replicate-degree=1 "${FSDP_KNOBS[@]}")
                    ;;
                hsdp-eager|hsdp-losscompile)
                    # HSDP_SHARD ranks shard each replica; default one replica/node.
                    HSDP_SHARD="${HSDP_SHARD:-${PPN}}"
                    (( HSDP_SHARD > 0 && NPROC % HSDP_SHARD == 0 )) || {
                        echo "HSDP_SHARD=${HSDP_SHARD} must divide world ${NPROC}" >&2; exit 2; }
                    echo "hsdp dp_shard=${HSDP_SHARD} dp_replicate=$((NPROC / HSDP_SHARD))"
                    PARALLEL=(--parallelism.data-parallel-shard-degree="${HSDP_SHARD}"
                        --parallelism.data-parallel-replicate-degree="$((NPROC / HSDP_SHARD))"
                        "${FSDP_KNOBS[@]}")
                    ;;
                *) echo "Unknown ARM=${ARM}" >&2; exit 2;;
            esac
            ;;
        *) echo "Unknown ARM=${ARM}" >&2; exit 2;;
    esac
    # BENCH_DETERMINISTIC=1: strict deterministic algorithms (no warn-only).
    DETERMINISTIC=(--debug.no-deterministic)
    case "${BENCH_DETERMINISTIC:-0}" in
        0) ;; 1) DETERMINISTIC=(--debug.deterministic) ;;
        *) echo "BENCH_DETERMINISTIC must be 0 or 1" >&2; exit 2;;
    esac
    # BENCH_REDUCE_PROBE=1 replays update 1 under each FSDP reduction variant
    # (probe_fsdp_reduce_precision.py), then trains normally.
    TRAIN_MODULE=torchtitan.experiments.ezpz.train
    case "${BENCH_REDUCE_PROBE:-0}" in
        0) ;;
        1) TRAIN_MODULE=torchtitan.experiments.ezpz.scripts.probe_fsdp_reduce_precision
           export AGPT_REDUCE_PROBE_OUTPUT="${OUT}/reduce-precision"
           [[ "${ARM}" == fsdp-* ]] || { echo "BENCH_REDUCE_PROBE needs an fsdp ARM" >&2; exit 2; } ;;
        *) echo "BENCH_REDUCE_PROBE must be 0 or 1" >&2; exit 2;;
    esac
    # CompileConfig selects named @local_compile regions. For this AGPT-2B
    # recipe, fused_binary_activation is the only active model region: it uses
    # ComplexRoPE, plain RMSNorm and dense SwiGLU (not cos_sin_rope,
    # gated_rmsnorm or moe_aux_loss). The extra region preserves the model
    # math and uses TorchTitan's existing compiler hook; it is not a whole-model
    # torch.compile graph.
    case "${COMPILE_MODE}" in
        none) COMPILE=(--compile.no-enable) ;;
        loss) COMPILE=(--compile.enable --compile.components=loss) ;;
        model-regions-loss)
            COMPILE=(--compile.enable --compile.components=fused_binary_activation,loss)
            ;;
        blocks-loss)
            # agpt/parallelize.py compiles each TransformerBlock (fullgraph),
            # as the historical components=[model, loss] runs did.
            COMPILE=(--compile.enable --compile.components=loss)
            export EZPZ_AGPT_COMPILE_BLOCKS="${EZPZ_AGPT_COMPILE_BLOCKS:-default}"
            ;;
    esac
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
    TIMING=()
    case "${BENCH_STEP_TIMINGS:-0}" in
        0) ;;
        1) TIMING=(--benchmark-step-timings) ;;
        *) echo "BENCH_STEP_TIMINGS must be 0 or 1" >&2; exit 2;;
    esac
    INITIAL_AUDIT=()
    if [[ -n "${BENCH_INITIAL_PARAMETER_AUDIT_JSON:-}" ]]; then
        [[ -s "${BENCH_INITIAL_PARAMETER_AUDIT_JSON}" ]] || {
            echo "Initial parameter audit requires an existing audit JSON" >&2
            exit 2
        }
        INITIAL_AUDIT=(--benchmark-initial-parameter-audit
            --benchmark-initial-parameter-audit-json="${BENCH_INITIAL_PARAMETER_AUDIT_JSON}")
        case "${BENCH_INITIAL_PARAMETER_AUDIT_ALLOW_DTENSOR:-0}" in
            0)
                [[ "${ARM}" == ddp-* ]] || {
                    echo "Default initial parameter audit requires native DDP" >&2
                    exit 2
                }
                ;;
            1)
                [[ "${ARM}" == fsdp-* || "${ARM}" == hsdp-* ]] || {
                    echo "DTensor initial parameter audit requires FSDP or HSDP" >&2
                    exit 2
                }
                INITIAL_AUDIT+=(--benchmark-initial-parameter-audit-allow-dtensor)
                ;;
            *) echo "BENCH_INITIAL_PARAMETER_AUDIT_ALLOW_DTENSOR must be 0 or 1" >&2; exit 2;;
        esac
    fi
    COMMAND=(ezpz launch --nproc "${NPROC}" --nproc_per_node "${PPN}" --
        "${VENV}/bin/python" "${OUT}/provenance/benchmark_agpt_entrypoint.py"
        --benchmark-output="${OUT}/rank-diagnostics"
        "${WITNESS[@]}" "${TIMING[@]}" "${INITIAL_AUDIT[@]}"
        --benchmark-module="${TRAIN_MODULE}" --
        --module=ezpz.agpt --config="${TT_CONFIG}"
        --training.steps="${STEPS}" --training.max-context-length="${SEQ}"
        --training.num-tokens-per-microbatch-per-dp-rank=$((MICROBATCH * SEQ))
        --training.num-tokens-per-train-step="${TOKENS}"
        "${PARALLEL[@]}" --dataloader.dataset=blendcorpus
        --dataloader.dataset-path="${ROOT}/torchtitan/experiments/ezpz/data-lists/sunspot/books.txt"
        --dataloader.train-iters="${STEPS}" --dataloader.split=95,5,0
        --dataloader.data-cache-path="${ROOT}/.cache/books-gbs${GBS}${BENCH_SEQ:+-seq${SEQ}}/index-cache"
        --debug.seed=42 "${DETERMINISTIC[@]}" --metrics.log-freq=1
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
