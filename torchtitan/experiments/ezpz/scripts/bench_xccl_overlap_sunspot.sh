#!/bin/bash --login
#PBS -A datascience
#PBS -N xccl-overlap
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# ezpz launch reads stdin, so it gets /dev/null; otherwise it eats the config list.
# Sweep oneCCL / Level Zero env knobs over bench_xccl_overlap.py.
# oneCCL reads env once per communicator, so each config is its own launch.
#
# BENCH_CONFIGS: newline-separated "tag|VAR=VAL VAR=VAL" (empty env allowed),
#                or BENCH_CONFIG_FILE with one such line per row.
# BENCH_TOPOS:   space-separated NODESxPPN, e.g. "1x12 2x8 2x12".
# BENCH_ARGS:    extra args for the python script.
# Environment setup follows benchmark_agpt_replicated_tt.sh.
# This microbenchmark does not currently initialize W&B. Future use must first
# pass a launch canary, then archive these per-job outputs to the online run.
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
EXPECTED="${BENCH_EXPECTED_TT_HEAD:?provide the pinned TorchTitan commit}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED}" ]] || {
    echo "HEAD $(git rev-parse HEAD) differs from BENCH_EXPECTED_TT_HEAD=${EXPECTED}" >&2
    exit 91
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    echo "tracked checkout changes detected; submit from a clean pinned checkout" >&2
    exit 92
}
case "$(readlink -f "${ROOT}")" in
    /lus/tegu/projects/datascience/foremans/projects/saforem2/*) ;;
    *) echo "Workdir must be on the project filesystem" >&2; exit 93;;
esac
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
[[ -x "${VENV}/bin/ezpz" ]] || { echo "missing ${VENV}/bin/ezpz" >&2; exit 94; }
CONSOLE_DIR="${OUT}/provenance/bin"
mkdir -p "${CONSOLE_DIR}"
"${VENV}/bin/python" - "${VENV}/bin/ezpz" "${CONSOLE_DIR}/ezpz" "${VENV}/bin/python" <<'PY' || exit 95
from pathlib import Path
import sys

source, target, interpreter = sys.argv[1:]
body = Path(source).read_text().partition("\n")[2]
if not body:
    raise SystemExit("installed ezpz console script has no body")
path = Path(target)
path.write_text("#!" + interpreter + "\n" + body)
path.chmod(0o755)
PY
[[ "$(head -n 1 "${CONSOLE_DIR}/ezpz")" == "#!${VENV}/bin/python" ]] || exit 95
export PATH="${CONSOLE_DIR}:${PATH}"
hash -r
[[ "$(command -v ezpz)" == "${CONSOLE_DIR}/ezpz" ]] || exit 95
ezpz launch --help > "${OUT}/provenance/ezpz-launch-help.txt" 2>&1 || exit 95
for flag in --nproc --nproc_per_node --nhosts --hostfile --cpu-bind; do
    grep -q -- "${flag}" "${OUT}/provenance/ezpz-launch-help.txt" || {
        echo "installed ezpz launch is missing required option ${flag}" >&2
        exit 95
    }
done
git rev-parse HEAD > "${OUT}/tt-head.txt"
cp "${BASH_SOURCE[0]}" "${OUT}/submitted.sh"
sort -u "${PBS_NODEFILE}" > "${OUT}/hosts.txt"
ALLOCATED_NODES="$(wc -l < "${OUT}/hosts.txt")"
[[ "${ALLOCATED_NODES}" -eq 2 ]] || {
    echo "expected 2 allocated nodes from PBS select=2, found ${ALLOCATED_NODES}" >&2
    exit 96
}
MASTER_ADDR="$(head -1 "${OUT}/hosts.txt")"
export MASTER_ADDR
BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
SCRIPT="${ROOT}/torchtitan/experiments/ezpz/scripts/bench_xccl_overlap.py"
DEFAULT_CONFIGS='base|CCL_OP_SYNC=1
sync0|CCL_OP_SYNC=0'
CONFIGS="${BENCH_CONFIGS:-${DEFAULT_CONFIGS}}"
# qsub -v cannot carry newlines or commas; a file can.
if [[ -n "${BENCH_CONFIG_FILE:-}" ]]; then
    [[ -r "${BENCH_CONFIG_FILE}" ]] || { echo "cannot read BENCH_CONFIG_FILE=${BENCH_CONFIG_FILE}" >&2; exit 2; }
    CONFIGS="$(grep -v '^#' "${BENCH_CONFIG_FILE}")" || exit 2
fi
[[ -n "${CONFIGS}" ]] || { echo "no benchmark configurations selected" >&2; exit 2; }
printf '%s\n' "${CONFIGS}" > "${OUT}/configs.txt"
TOPOS="${BENCH_TOPOS:-1x12 2x8}"
[[ -n "${TOPOS}" ]] || { echo "no benchmark topologies selected" >&2; exit 2; }
echo "BEGIN $(date -Is) job=${PBS_JOBID} hosts=$(tr '\n' ' ' < "${OUT}/hosts.txt") head=$(cat "${OUT}/tt-head.txt")"
echo "W&B_GATE: no W&B integration; run a separate launch canary before future use, then archive these outputs to the online campaign run."
port=29400
overall_rc=0
for topo in ${TOPOS}; do
    if [[ "${topo}" != *x* ]]; then
        echo "invalid topology '${topo}', expected NODESxPPN" >&2
        overall_rc=1
        continue
    fi
    nodes="${topo%x*}"; ppn="${topo#*x}"
    if [[ ! "${nodes}" =~ ^[1-9][0-9]*$ || ! "${ppn}" =~ ^(8|12)$ ]] || ((nodes > ALLOCATED_NODES)); then
        echo "topology ${topo} exceeds allocation or has invalid counts (allocated nodes=${ALLOCATED_NODES}, allowed PPN=8|12)" >&2
        overall_rc=1
        continue
    fi
    np=$((nodes * ppn))
    hostfile="${OUT}/hosts-${nodes}.txt"
    head -n "${nodes}" "${OUT}/hosts.txt" > "${hostfile}"
    [[ "$(wc -l < "${hostfile}")" -eq "${nodes}" ]] || {
        echo "hostfile ${hostfile} does not contain ${nodes} unique allocated hosts" >&2
        overall_rc=1
        continue
    }
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        tag="${line%%|*}"; envs="${line#*|}"
        port=$((port + 1))
        echo "RUN_BEGIN $(date -Is) topo=${topo} nodes=${nodes} ppn=${ppn} world=${np} hostfile=${hostfile} tag=${tag} env=[${envs}]"
        # shellcheck disable=SC2086
        env ${envs} MASTER_PORT="${port}" BENCH_WORLD="${np}" BENCH_CFG="${tag}" \
            BENCH_EXPECTED_WORLD_SIZE="${np}" \
            timeout --kill-after=30 "${BENCH_RUN_TIMEOUT:-420}" \
            ezpz launch --nproc "${np}" --nproc_per_node "${ppn}" \
            --nhosts "${nodes}" --hostfile "${hostfile}" --cpu-bind "${BIND}" -- \
            "${VENV}/bin/python" "${SCRIPT}" --tag "${tag}" --trace-dir "${OUT}/traces" \
            --jsonl "${OUT}/results.jsonl" ${BENCH_ARGS:-} < /dev/null
        rc=$?
        echo "RUN_END $(date -Is) topo=${topo} tag=${tag} rc=${rc}"
        if ((rc != 0)); then
            overall_rc=1
        fi
    done <<< "${CONFIGS}"
done
echo "END $(date -Is) rc=${overall_rc}"
exit "${overall_rc}"
