#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-replicated-matrix
#PBS -l walltime=01:30:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Run MDS and TT arms serially on the same allocated nodes.
# BENCH_PHASES is a space-separated list of KIND:TAG[:KEY=VAL+KEY=VAL...],
# KIND being mds, mdsshape or tt. Overrides apply to that phase only, e.g.
#   "mds:ctl tt:blk:ARM=ddp-bf16-losscompile+BENCH_COMPILE_MODE=blocks-loss
#    mds:gas16:MDS_GAS=16+MDS_OPTIMIZER=adamw+MDS_ZERO_STAGE=1+MDS_ZERO0_PATH=runtime"
# MDS phases default to the historical BF16_Optimizer control used by
# benchmark_agpt_replicated_pair.sh; TT phases call the TT launcher directly.
# mdsshape runs benchmark_mds_historical_shape.sh (the z2rrt02w shape).
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
ROOT="${PBS_O_WORKDIR:?submit from the isolated checkout}"
cd "${ROOT}" || exit 2
[[ "$(git rev-parse HEAD)" == "${BENCH_EXPECTED_TT_HEAD:?pin source SHA}" ]] || exit 91
[[ -z "$(git diff --name-only HEAD)" ]] || exit 92
PHASES="${BENCH_PHASES:?set BENCH_PHASES}"
export TT_ROOT="${ROOT}"
export NODES="${NODES:-2}"
export PPN="${PPN:-12}"
export STEPS="${STEPS:-32}"
export BENCH_STEP_TIMINGS=1
export WANDB_MODE=online
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/agpt-replicated-matrix-${JOB}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
cp "${BASH_SOURCE[0]}" "${OUT}/submitted-matrix.sh"
git rev-parse HEAD > "${OUT}/tt-head.txt"
sort -u "${PBS_NODEFILE}" > "${OUT}/hosts.txt"
[[ "$(wc -l < "${OUT}/hosts.txt")" -eq "${NODES}" ]] || exit 97
SCRIPTS="${ROOT}/torchtitan/experiments/ezpz/scripts"
status=0
for phase in ${PHASES}; do
    IFS=: read -r kind tag extra <<< "${phase}"
    [[ -n "${kind}" && -n "${tag}" ]] || { echo "Malformed phase ${phase}" >&2; exit 2; }
    overrides=()
    if [[ -n "${extra:-}" ]]; then
        IFS=+ read -r -a overrides <<< "${extra}"
        for item in "${overrides[@]}"; do
            [[ "${item}" =~ ^[A-Z_][A-Z0-9_]*=[A-Za-z0-9._-]*$ ]] || {
                echo "Malformed override ${item} in ${phase}" >&2; exit 2; }
        done
    fi
    echo "MATRIX_PHASE_BEGIN $(date -Is) kind=${kind} tag=${tag} overrides=${extra:-} source=${BENCH_EXPECTED_TT_HEAD}"
    case "${kind}" in
        mds)
            env MDS_ZERO_STAGE=0 MDS_ZERO0_PATH=ds0175 MDS_OPTIMIZER=sophiag \
                MDS_AC=none GRAD_POLICY=production PROFILE=1 "${overrides[@]}" \
                RUN_KIND="matrix-${tag}" \
                timeout --kill-after=60 "${BENCH_MDS_TIMEOUT:-1500}" \
                bash --login "${SCRIPTS}/benchmark_agpt_replicated_mds.sh"
            ;;
        mdsshape)
            env "${overrides[@]}" \
                timeout --kill-after=60 "${BENCH_MDS_TIMEOUT:-1500}" \
                bash --login "${SCRIPTS}/benchmark_mds_historical_shape.sh"
            ;;
        tt)
            env ARM=ddp-bf16-losscompile "${overrides[@]}" BENCH_TAG="${tag}" \
                timeout --kill-after=60 "${BENCH_VARIANT_TIMEOUT:-1200}" \
                bash --login "${SCRIPTS}/benchmark_agpt_replicated_tt.sh"
            ;;
        *) echo "Unknown phase kind ${kind}" >&2; exit 2;;
    esac
    rc=$?
    echo "MATRIX_PHASE_END $(date -Is) kind=${kind} tag=${tag} rc=${rc}"
    ((rc != 0)) && status=1
done
echo "MATRIX_COMPLETE $(date -Is) rc=${status} output=${OUT}"
exit "${status}"
