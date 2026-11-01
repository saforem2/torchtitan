#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-pbs-startup
#PBS -l walltime=00:03:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe

# Isolate PBS shell and shared-storage startup before loading any runtime.
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
printf 'BATCH_ENTERED time=%s job=%s host=%s\n' "$(date -Is)" "${PBS_JOBID}" "$(hostname -s)"
id
ROOT="${PBS_O_WORKDIR:?submit from a pinned checkout}"
[[ -d "${ROOT}" && -r "${ROOT}" && -w "${ROOT}" ]] || exit 91
cd "${ROOT}" || exit 92
ACTUAL="$(git rev-parse HEAD)" || exit 93
[[ "${ACTUAL}" == "${BENCH_EXPECTED_TT_HEAD:?pin source}" ]] || exit 94
[[ -z "$(git diff --name-only HEAD)" ]] || exit 95
printf 'SHARED_STORAGE_OK workdir=%s sha=%s\n' "${ROOT}" "${ACTUAL}"
printf 'PBS_STARTUP_OK time=%s\n' "$(date -Is)"
