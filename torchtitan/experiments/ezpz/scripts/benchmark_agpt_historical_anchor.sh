#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-historical-anchor
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe

# Re-run the historical AGPT-2B TT throughput run 6tc71of4 (2026-07-26,
# 1 node x 12 ranks, global FSDP, compile [model, loss], local batch 2,
# torch 2.13.0a0+gitcf30153) from its own source commit and runtime, to check
# whether today's Sunspot still delivers its 93.75 model TFLOP/s/tile.
# ANCHOR_ROOT must be a clean checkout of ANCHOR_SHA (default 4a72fa1bf).
# ANCHOR_LBS is a space-separated list of local batch sizes to run serially.
# ANCHOR_ENVS is a space-separated list of launch-environment profiles, each
# run at every ANCHOR_LBS:
#   batch   CCL_OP_SYNC=1, CCL_PROCESS_LAUNCHER=pmix, PBS's OMP_NUM_THREADS
#   july    none of the three, as in the interactive 6tc71of4 session
#   nosync  batch without CCL_OP_SYNC and CCL_PROCESS_LAUNCHER
#   noomp   batch without OMP_NUM_THREADS
# ANCHOR_PREWARM, if set, is a command run under `ezpz launch` before the
# first arm (for example a GEMM loop that keeps every tile busy).
# ANCHOR_PAGECACHE_PREWARM=1 reads the conda env and venv shared libraries
# once before the first arm. ANCHOR_SAMPLE_CPU=1 logs the busiest processes
# every 15 s to cpu.log.
set -o pipefail
ANCHOR_ROOT="${ANCHOR_ROOT:?set to a clean checkout of the historical commit}"
ANCHOR_SHA="${ANCHOR_SHA:-4a72fa1bf69526d4c9c0e47304e7d6081246cde3}"
ANCHOR_LBS="${ANCHOR_LBS:-2}"
ANCHOR_STEPS="${ANCHOR_STEPS:-20}"
ANCHOR_ENVS="${ANCHOR_ENVS:-batch}"
CONDA_ENV=/lus/tegu/projects/datasets/software/26.181.0/wheelforge/envs/conda_envs/RC2_vllm_xpu_kernels_0.1.10.1_vllm_0.25.1_nre_pt_2.13.0_rel_one_2026.1.0_np_2.3.5_python_3.12.12
VENV=/lus/tegu/projects/datascience/foremans/projects/saforem2/tmp/2026-07-24-115711/venvs/fw_ssp
cd "${ANCHOR_ROOT}" || exit 2
[[ "$(git rev-parse HEAD)" == "${ANCHOR_SHA}" ]] || exit 91
[[ -z "$(git diff --name-only HEAD)" ]] || exit 92
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
# ANCHOR_ALLOC_CONF=none drops PYTORCH_XPU_ALLOC_CONF (6tc71of4 used expandable segments).
export ZE_FLAT_DEVICE_HIERARCHY=FLAT CCL_LOG_LEVEL=ERROR
export PYTORCH_XPU_ALLOC_CONF="${ANCHOR_ALLOC_CONF:-expandable_segments:True}"
[[ "${PYTORCH_XPU_ALLOC_CONF}" == none ]] && unset PYTORCH_XPU_ALLOC_CONF
export WANDB_MODE=online WANDB_ENTITY=aurora_gpt WANDB_PROJECT=torchtitan.ezpz.train
export WANDB_RUN_GROUP=agpt-historical-anchor-20261102
module add miniforge3/25.11.0-1 cmake ninja pti-gpu hdf5 >/dev/null 2>&1
ml add intel_gpu_umd_aicoe/2026.06.19 >/dev/null 2>&1
unset CMAKE_ROOT
conda activate "${CONDA_ENV}" >/dev/null 2>&1
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
JOB="${PBS_JOBID%%.*}"
OUT="${ANCHOR_ROOT}/outputs/agpt-historical-anchor-${JOB}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
echo "BEGIN $(date -Is) host=$(hostname -s) sha=${ANCHOR_SHA} lbs=${ANCHOR_LBS} envs=${ANCHOR_ENVS} alloc=${PYTORCH_XPU_ALLOC_CONF-unset} steps=${ANCHOR_STEPS}"
python3 -c 'import torch; print("RUNTIME", torch.__version__, torch.__file__, torch.xpu.device_count())' || exit 96
module -t list 2> "${OUT}/modules.txt"
status=0
if [[ "${ANCHOR_SAMPLE_CPU:-0}" == 1 ]]; then
    while sleep 15; do
        echo "== $(date -Is) load=$(cut -d" " -f1-3 /proc/loadavg)"
        ps -eo pid,ppid,pcpu,nlwp,rss,comm --sort=-pcpu | head -20
    done > "${OUT}/cpu.log" 2>&1 &
    sampler=$!
fi
if [[ "${ANCHOR_PAGECACHE_PREWARM:-0}" == 1 ]]; then
    echo "PAGECACHE_BEGIN $(date -Is)"
    find "${CONDA_ENV}" "${VENV}" -name '*.so*' -type f -print0 | xargs -0 -P 16 cat > /dev/null
    echo "PAGECACHE_END $(date -Is)"
fi
if [[ -n "${ANCHOR_PREWARM:-}" ]]; then
    echo "PREWARM_BEGIN $(date -Is) ${ANCHOR_PREWARM}"
    # shellcheck disable=SC2086
    ezpz launch ${ANCHOR_PREWARM} > "${OUT}/prewarm.log" 2>&1
    echo "PREWARM_END $(date -Is) rc=$?"
    grep -h PREWARM "${OUT}/prewarm.log" | sort -t= -k2 -n
fi
for profile in ${ANCHOR_ENVS}; do
for lbs in ${ANCHOR_LBS}; do
    tag="${profile}-lbs${lbs}"
    [[ "${profile}" == batch ]] && tag="lbs${lbs}"
    (
        case "${profile}" in
            batch) export CCL_PROCESS_LAUNCHER=pmix CCL_OP_SYNC=1 ;;
            july) unset CCL_PROCESS_LAUNCHER CCL_OP_SYNC OMP_NUM_THREADS ;;
            nosync) unset CCL_PROCESS_LAUNCHER CCL_OP_SYNC ;;
            noomp) export CCL_PROCESS_LAUNCHER=pmix CCL_OP_SYNC=1; unset OMP_NUM_THREADS ;;
            *) echo "unknown ANCHOR_ENVS profile ${profile}" >&2; exit 2 ;;
        esac
        export WANDB_RUN_ID="agpt-historical-anchor-${tag}-${JOB}"
        export WANDB_NAME="${WANDB_RUN_ID}"
        echo "ANCHOR_BEGIN $(date -Is) ${tag} OMP_NUM_THREADS=${OMP_NUM_THREADS-unset} CCL_OP_SYNC=${CCL_OP_SYNC-unset} CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER-unset}"
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
            --module=ezpz.agpt --config=agpt_2b_real --checkpoint.no-enable \
            --training.local-batch-size="${lbs}" --training.seq-len=8192 \
            --dataloader.num-workers=0 --training.steps="${ANCHOR_STEPS}" \
            > "${OUT}/${tag}.log" 2>&1
    )
    rc=$?
    echo "ANCHOR_END $(date -Is) ${tag} rc=${rc}"
    ((rc != 0)) && status=1
done
done
[[ -n "${sampler:-}" ]] && kill "${sampler}"
echo "END $(date -Is) rc=${status} output=${OUT}"
exit "${status}"
