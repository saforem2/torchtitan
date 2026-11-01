#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-tt-ablate
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Run benchmark_agpt_replicated_tt.sh variants serially in one allocation.
# BENCH_VARIANTS is a space-separated list of TAG:SYNC:RESHARD:REDUCE:DEFER,
# for example "base:1:default:float32:0 sync0:0:default:float32:0".
# ARM, PPN, NODES, STEPS and BENCH_EXPECTED_TT_HEAD pass through unchanged.
set -o pipefail
ROOT="${PBS_O_WORKDIR:?submit from the isolated checkout}"
VARIANTS="${BENCH_VARIANTS:?set BENCH_VARIANTS}"
LAUNCHER="${ROOT}/torchtitan/experiments/ezpz/scripts/benchmark_agpt_replicated_tt.sh"
LIMIT="${BENCH_VARIANT_TIMEOUT:-1200}"
status=0
for variant in ${VARIANTS}; do
    IFS=: read -r tag sync reshard reduce defer <<< "${variant}"
    [[ -n "${tag}" && -n "${sync}" && -n "${reshard}" && -n "${reduce}" && -n "${defer}" ]] || {
        echo "Malformed variant ${variant}" >&2; exit 2; }
    echo "VARIANT_BEGIN $(date -Is) tag=${tag} sync=${sync} reshard=${reshard} reduce=${reduce} defer=${defer}"
    BENCH_TAG="${tag}" BENCH_CCL_OP_SYNC="${sync}" BENCH_FSDP_RESHARD="${reshard}" \
        BENCH_FSDP_REDUCE_DTYPE="${reduce}" BENCH_FSDP_DEFER="${defer}" \
        timeout --kill-after=60 "${LIMIT}" bash --login "${LAUNCHER}"
    rc=$?
    echo "VARIANT_END $(date -Is) tag=${tag} rc=${rc}"
    ((rc != 0)) && status=1
done
exit "${status}"
