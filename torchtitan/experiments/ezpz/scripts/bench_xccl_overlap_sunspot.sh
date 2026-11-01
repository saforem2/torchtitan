#!/bin/bash --login
#PBS -A datascience
#PBS -N xccl-overlap
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# mpiexec reads stdin, so it gets /dev/null; otherwise it eats the config list.
# Sweep oneCCL / Level Zero env knobs over bench_xccl_overlap.py.
# oneCCL reads env once per communicator, so each config is its own launch.
#
# BENCH_CONFIGS: newline-separated "tag|VAR=VAL VAR=VAL" (empty env allowed),
#                or BENCH_CONFIG_FILE with one such line per row.
# BENCH_TOPOS:   space-separated NODESxPPN, e.g. "1x12 2x8 2x12".
# BENCH_ARGS:    extra args for the python script.
# Environment setup follows benchmark_agpt_replicated_tt.sh.
set -o pipefail
ulimit -c 0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT=600

ROOT="${PBS_O_WORKDIR:?submit from the checkout}"
cd "${ROOT}" || exit 2
VENV="${BENCH_VENV:-/lus/tegu/projects/datascience/foremans/venvs/xpu-torch214-fixedlr-v2-20260927}"
JOB="${PBS_JOBID%%.*}"
OUT="${ROOT}/outputs/xccl-overlap/${JOB}"
mkdir -p "${OUT}/traces"
exec > >(tee -a "${OUT}/run.log") 2>&1
source "${ROOT}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job || exit $?
ezpz_load_modules || exit $?
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
SHIM="${ROOT}/.cache/xpu-ld-first/$(basename "${VENV}")"
mkdir -p "${SHIM}"
for library in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* \
               "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
    [[ -e "${library}" ]] && ln -sf "${library}" "${SHIM}/"
done
export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || exit 94
git rev-parse HEAD > "${OUT}/tt-head.txt"
cp "${BASH_SOURCE[0]}" "${OUT}/submitted.sh"
sort -u "${PBS_NODEFILE}" > "${OUT}/hosts.txt"
head -1 "${OUT}/hosts.txt" > "${OUT}/host1.txt"
MASTER_ADDR="$(head -1 "${OUT}/hosts.txt")"
export MASTER_ADDR
BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
SCRIPT="${ROOT}/torchtitan/experiments/ezpz/scripts/bench_xccl_overlap.py"
DEFAULT_CONFIGS='base|CCL_OP_SYNC=1
sync0|CCL_OP_SYNC=0'
CONFIGS="${BENCH_CONFIGS:-${DEFAULT_CONFIGS}}"
# qsub -v cannot carry newlines or commas; a file can.
[[ -n "${BENCH_CONFIG_FILE:-}" ]] && CONFIGS="$(grep -v '^#' "${BENCH_CONFIG_FILE}")"
cp "${BENCH_CONFIG_FILE:-/dev/null}" "${OUT}/configs.txt"
TOPOS="${BENCH_TOPOS:-1x12 2x8}"
echo "BEGIN $(date -Is) job=${PBS_JOBID} hosts=$(tr '\n' ' ' < "${OUT}/hosts.txt") head=$(cat "${OUT}/tt-head.txt")"
port=29400
for topo in ${TOPOS}; do
    nodes="${topo%x*}"; ppn="${topo#*x}"; np=$((nodes * ppn))
    hostfile="${OUT}/hosts.txt"; [[ "${nodes}" == 1 ]] && hostfile="${OUT}/host1.txt"
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        tag="${line%%|*}"; envs="${line#*|}"
        port=$((port + 1))
        echo "RUN_BEGIN $(date -Is) topo=${topo} tag=${tag} env=[${envs}]"
        # shellcheck disable=SC2086
        env ${envs} MASTER_PORT="${port}" BENCH_WORLD="${np}" BENCH_CFG="${tag}" \
            timeout --kill-after=30 "${BENCH_RUN_TIMEOUT:-420}" \
            mpiexec --envall --line-buffer --np "${np}" --ppn "${ppn}" \
            --hostfile "${hostfile}" --cpu-bind="${BIND}" \
            python "${SCRIPT}" --tag "${tag}" --trace-dir "${OUT}/traces" \
            --jsonl "${OUT}/results.jsonl" ${BENCH_ARGS:-} < /dev/null
        rc=$?
        echo "RUN_END $(date -Is) topo=${topo} tag=${tag} rc=${rc}"
    done <<< "${CONFIGS}"
done
echo "END $(date -Is)"
