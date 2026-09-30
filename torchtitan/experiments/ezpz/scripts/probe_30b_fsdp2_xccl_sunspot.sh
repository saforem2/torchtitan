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

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-$http_proxy}"
source "$D/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job || exit $?
ezpz_load_modules || exit $?
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
    nproc=16
    group_stride=0
    log="$OUT/raw-$op.log"
    case "$op" in
        all_gather)
            # BF16 parameter shards: 192/16 and 1176/16 MiB per rank.
            dtype=bfloat16
            mib="12 73.5"
            ;;
        reduce_scatter)
            # FP32 gradient inputs: full 384 and 2352 MiB tensors.
            dtype=float32
            mib="384 2352"
            ;;
        all_reduce)
            # FP32 HSDP gradient shards: 384/16 and 2352/16 MiB per rank.
            dtype=float32
            mib="24 147"
            nproc=48
            group_stride=16
            ;;
        *)
            exit 98
            ;;
    esac
    WORLD_SIZE="$nproc" EZPZ_PROBE_OP="$op" EZPZ_PROBE_DTYPE="$dtype" \
        EZPZ_PROBE_MIBS="$mib" EZPZ_PROBE_GROUP_STRIDE="$group_stride" \
        "$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc "$nproc" --nproc_per_node 12 --timeout 600 -- \
        "$V/bin/python" torchtitan/experiments/ezpz/tests/probe_collective_op.py \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        exit "$rc"
    fi
    grep -q ALL_PASSED "$log" || exit 96
}

run_fsdp() {
    label=$1
    nproc=$2
    ppn=$3
    shift 3
    log="$OUT/$label.log"
    WORLD_SIZE="$nproc" "$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc "$nproc" --nproc_per_node "$ppn" --timeout 900 -- \
        "$V/bin/python" \
        torchtitan/experiments/ezpz/tests/probe_fsdp2_storage_collectives.py \
        "$@" --iterations 3 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        exit "$rc"
    fi
    grep -q FSDP2_XCCL_PROBE_PASS "$log" || exit 97
}

run_raw all_gather
run_raw reduce_scatter
run_raw all_reduce
run_fsdp pure-fsdp 16 12 --dp-replicate 1 --dp-shard 16
run_fsdp hsdp 48 12 --dp-replicate 3 --dp-shard 16
printf 'FSDP2_XCCL_MATRIX_PASS job=%s commit=%s\n' "$JOB" "$EXPECTED_SHA" | tee "$OUT/VALIDATED"
