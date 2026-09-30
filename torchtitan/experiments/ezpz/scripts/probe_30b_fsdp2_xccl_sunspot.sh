#!/bin/bash --login
#PBS -A datascience
#PBS -N tt30b-fsdp2-xccl
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
set -o pipefail

D="${PROBE_WORKTREE:?set PROBE_WORKTREE}"
V="${PROBE_VENV:?set PROBE_VENV}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA}"
JOB="${PBS_JOBID%%.*}"
OUT="$D/outputs/30b-fsdp2-xccl-probe-$JOB"
mkdir -p "$OUT"
cd "$D" || exit 2

source "$D/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job
ezpz_load_modules
export VIRTUAL_ENV="$V"
export PATH="$V/bin:/opt/pbs/bin:$PATH"
hash -r
[[ "$(command -v python)" == "$V/bin/python" ]] || exit 93
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || exit 95

export PYTHONPATH="$D${PYTHONPATH:+:$PYTHONPATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export CCL_ATL_SYNC_COLL=1
export CCL_SYCL_KERNEL_SYNC=0
export CCL_KVS_MODE=pmi
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

run_raw() {
    op=$1
    log="$OUT/raw-$op.log"
    mib="0.001 1 16 64 144"
    if [[ "$op" == reduce_scatter ]]; then mib="192 1176"; fi
    WORLD_SIZE=16 EZPZ_PROBE_OP="$op" EZPZ_PROBE_MIBS="$mib" \
        "$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc 16 --nproc_per_node 8 --timeout 600 -- \
        "$V/bin/python" torchtitan/experiments/ezpz/tests/probe_collective_op.py \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    test "$rc" -eq 0 && grep -q ALL_PASSED "$log" || exit "${rc:-1}"
}

run_fsdp() {
    label=$1
    nproc=$2
    ppn=$3
    shift 3
    log="$OUT/$label.log"
    "$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc "$nproc" --nproc_per_node "$ppn" --timeout 900 -- \
        "$V/bin/python" \
        torchtitan/experiments/ezpz/tests/probe_fsdp2_storage_collectives.py \
        "$@" --iterations 3 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    test "$rc" -eq 0 && grep -q FSDP2_XCCL_PROBE_PASS "$log" || exit "${rc:-1}"
}

run_raw all_gather
run_raw reduce_scatter
run_raw all_reduce
run_fsdp pure-fsdp 16 8 --dp-replicate 1 --dp-shard 16
run_fsdp hsdp 48 12 --dp-replicate 3 --dp-shard 16
printf 'FSDP2_XCCL_MATRIX_PASS job=%s commit=%s\n' "$JOB" "$EXPECTED_SHA" | tee "$OUT/VALIDATED"
