#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-replicated-pair
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Reuse the validated launchers serially on the same allocated nodes.
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
ROOT="${PBS_O_WORKDIR:?submit from the isolated checkout}"
cd "${ROOT}" || exit 2
[[ "$(git rev-parse HEAD)" == "${BENCH_EXPECTED_TT_HEAD:?pin source SHA}" ]] || exit 91
[[ -z "$(git diff --name-only HEAD)" ]] || exit 92
export TT_ROOT="${ROOT}"
export NODES="${NODES:-2}"
export PPN="${PPN:-12}"
export STEPS="${STEPS:-32}"
export BENCH_STEP_TIMINGS=1
export WANDB_MODE=online
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/agpt-replicated-pair-${JOB}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
cp "${BASH_SOURCE[0]}" "${OUT}/submitted-pair.sh"
git rev-parse HEAD > "${OUT}/tt-head.txt"
sort -u "${PBS_NODEFILE}" > "${OUT}/hosts.txt"
[[ "$(wc -l < "${OUT}/hosts.txt")" -eq "${NODES}" ]] || exit 97
export MDS_ZERO_STAGE=0
export MDS_ZERO0_PATH=ds0175
export MDS_OPTIMIZER=sophiag
export MDS_AC=none
export GRAD_POLICY=production
export PROFILE=1
SCRIPTS="${ROOT}/torchtitan/experiments/ezpz/scripts"
status=0
for phase in before tt after; do
    echo "PAIR_PHASE_BEGIN $(date -Is) phase=${phase} source=${BENCH_EXPECTED_TT_HEAD}"
    if [[ "${phase}" == tt ]]; then
        bash --login "${SCRIPTS}/benchmark_agpt_replicated_tt_ablate.sh"
    else
        # RUN_KIND is an existing MDS naming override; keep fresh outputs/run IDs.
        RUN_KIND="zero0-ds0175-sophiag-pair-${phase}" \
            timeout --kill-after=60 "${BENCH_MDS_TIMEOUT:-900}" \
            bash --login "${SCRIPTS}/benchmark_agpt_replicated_mds.sh"
    fi
    rc=$?
    echo "PAIR_PHASE_END $(date -Is) phase=${phase} rc=${rc}"
    ((rc != 0)) && status=1
done
echo "PAIR_COMPLETE $(date -Is) rc=${status} output=${OUT}"
exit "${status}"
