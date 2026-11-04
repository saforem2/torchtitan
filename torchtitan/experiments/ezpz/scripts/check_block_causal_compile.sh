#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-blk-compile
#PBS -l walltime=00:20:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe

# One-tile check of block-causal attention gradients, eager vs torch.compile,
# against an FP32 reference (check_block_causal_compile.py), in the TT
# torch-2.14 venv. Submit via alcf-job-preflight/submit.sh from the checkout.
set -o pipefail
ulimit -c 0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
ROOT="${PBS_O_WORKDIR:?submit from the checkout}"
cd "${ROOT}" || exit 2
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/agpt-replicated-20261031/blk-compile-check-${JOB}"
mkdir -p "${OUT}"
TT_VENV=/lus/tegu/projects/datascience/foremans/venvs/xpu-torch214-fixedlr-v2-20260927
echo "BEGIN $(date -Is) host=$(hostname -s) job=${PBS_JOBID} head=$(git rev-parse HEAD)" | tee "${OUT}/run.log"
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
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/blk-compile-${JOB}/inductor"
export TRITON_CACHE_DIR="/tmp/blk-compile-${JOB}/triton"
ZE_AFFINITY_MASK=0 "${TT_VENV}/bin/python" \
    "${ROOT}/torchtitan/experiments/ezpz/scripts/check_block_causal_compile.py" \
    >> "${OUT}/run.log" 2>&1
rc=$?
grep -h '^RESULT ' "${OUT}/run.log"
echo "END $(date -Is) rc=${rc}" | tee -a "${OUT}/run.log"
exit "${rc}"
