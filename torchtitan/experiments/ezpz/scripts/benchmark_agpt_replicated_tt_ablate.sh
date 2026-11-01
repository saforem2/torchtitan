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
# An optional sixth field holds comma-separated KEY=VALUE environment
# overrides for that variant only, e.g.
# "z1:1:default:float32:0:ARM=ddp-bf16-losscompile,BENCH_DDP_ZERO1=bfloat16".
# ARM, PPN, NODES, STEPS and BENCH_EXPECTED_TT_HEAD pass through unchanged.
set -o pipefail
# The launcher sets this too; exported here so preflight sees it.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
ROOT="${PBS_O_WORKDIR:?submit from the isolated checkout}"
VARIANTS="${BENCH_VARIANTS:?set BENCH_VARIANTS}"
LAUNCHER="${ROOT}/torchtitan/experiments/ezpz/scripts/benchmark_agpt_replicated_tt.sh"
LIMIT="${BENCH_VARIANT_TIMEOUT:-1200}"
status=0
for variant in ${VARIANTS}; do
    IFS=: read -r tag sync reshard reduce defer extra <<< "${variant}"
    [[ -n "${tag}" && -n "${sync}" && -n "${reshard}" && -n "${reduce}" && -n "${defer}" ]] || {
        echo "Malformed variant ${variant}" >&2; exit 2; }
    overrides=()
    if [[ -n "${extra:-}" ]]; then
        IFS=, read -r -a overrides <<< "${extra}"
        for item in "${overrides[@]}"; do
            [[ "${item}" =~ ^[A-Z_][A-Z0-9_]*=[A-Za-z0-9._-]*$ ]] || {
                echo "Malformed override ${item} in ${variant}" >&2; exit 2; }
        done
    fi
    echo "VARIANT_BEGIN $(date -Is) tag=${tag} sync=${sync} reshard=${reshard} reduce=${reduce} defer=${defer} overrides=${extra:-}"
    env "${overrides[@]}" BENCH_TAG="${tag}" BENCH_CCL_OP_SYNC="${sync}" BENCH_FSDP_RESHARD="${reshard}" \
        BENCH_FSDP_REDUCE_DTYPE="${reduce}" BENCH_FSDP_DEFER="${defer}" \
        timeout --kill-after=60 "${LIMIT}" bash --login "${LAUNCHER}"
    rc=$?
    echo "VARIANT_END $(date -Is) tag=${tag} rc=${rc}"
    ((rc != 0)) && status=1
done
exit "${status}"
