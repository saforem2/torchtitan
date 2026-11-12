#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-attn-bwd
#PBS -l walltime=00:30:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe

# One-node attention fwd/bwd microbenchmark at the AGPT-2B shape.
# Runs the TT torch-2.14 venv and the MDS torch-2.10(+IPEX) venv on separate
# cards (tiles 0, 2, 4) concurrently. Submit via alcf-job-preflight/submit.sh.
set -o pipefail
ulimit -c 0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
ROOT="${PBS_O_WORKDIR:?submit from the checkout}"
cd "${ROOT}" || exit 2
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/agpt-replicated-20261031/attn-bwd-bench-${JOB}"
mkdir -p "${OUT}"
BENCH="${ROOT}/torchtitan/experiments/ezpz/scripts/bench_xpu_attention_bwd.py"
CASES="${BENCH_CASES:-all}"
git rev-parse HEAD > "${OUT}/head.txt"
echo "BEGIN $(date -Is) host=$(hostname -s) job=${PBS_JOBID} head=$(cat "${OUT}/head.txt")"

TT_VENV=/lus/tegu/projects/datascience/foremans/venvs/xpu-torch214-fixedlr-v2-20260927
MDS_VENV=/lus/tegu/projects/datascience/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/sunspot/Megatron-DeepSpeed-aurora_frameworks-2025.3.1

(
    source "${ROOT}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
    load_pinned_ezpz_utils || exit $?
    ezpz_load_modules || exit $?
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    SHIM="${ROOT}/.cache/xpu-ld-first/$(basename "${TT_VENV}")"
    mkdir -p "${SHIM}"
    for library in "${TT_VENV}"/lib/libur_* "${TT_VENV}"/lib/libsycl.so* \
                   "${TT_VENV}"/lib/libumf* "${TT_VENV}"/lib/libtcm*; do
        [[ -e "${library}" ]] && ln -sf "${library}" "${SHIM}/"
    done
    export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/attn-bench-${JOB}/inductor-tt"
    export TRITON_CACHE_DIR="/tmp/attn-bench-${JOB}/triton-tt"
    ZE_AFFINITY_MASK=0 "${TT_VENV}/bin/python" "${BENCH}" --tag tt-torch214 --cases "${CASES}"
) > "${OUT}/tt-torch214.log" 2>&1 &
p1=$!

mds_env() {
    module load oneapi/release/2025.3.1 hdf5 pti-gpu || return $?
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    export TORCHINDUCTOR_CACHE_DIR="/tmp/attn-bench-${JOB}/inductor-mds-$1"
    export TRITON_CACHE_DIR="/tmp/attn-bench-${JOB}/triton-mds-$1"
}
( mds_env ipex || exit $?
  ZE_AFFINITY_MASK=2 "${MDS_VENV}/bin/python" "${BENCH}" --tag mds-torch210-ipex --ipex --cases "${CASES}"
) > "${OUT}/mds-torch210-ipex.log" 2>&1 &
p2=$!
( mds_env noipex || exit $?
  ZE_AFFINITY_MASK=4 "${MDS_VENV}/bin/python" "${BENCH}" --tag mds-torch210-noipex --cases "${CASES}"
) > "${OUT}/mds-torch210-noipex.log" 2>&1 &
p3=$!
wait $p1; r1=$?; wait $p2; r2=$?; wait $p3; r3=$?
echo "rc tt=${r1} mds_ipex=${r2} mds_noipex=${r3}"
grep -h '^RESULT ' "${OUT}"/*.log > "${OUT}/results.jsonl"
wc -l "${OUT}/results.jsonl"
for f in "${OUT}"/*.log; do echo "== tail ${f}"; tail -n 5 "${f}"; done
echo "END $(date -Is)"
