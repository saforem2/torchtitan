#!/bin/bash
# alcf-job-preflight: submit wrapper. Runs preflight and REFUSES to qsub/sbatch
# if it fails.
#
# Why this exists: preflight.sh already exits non-zero, but invoking it in a
# shell chain (`bash preflight.sh ...; qsub job.sh`) ignores that and submits
# anyway. That happened -- preflight printed "PREFLIGHT FAIL, do not submit"
# and the job was queued in the same command. A check whose result you can
# ignore by accident is not a gate.
#
#   usage: submit.sh <worktree> <venv> <machine> <job-script> [qsub args...]
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${1:?usage: submit.sh <worktree> <venv> <machine> <job-script> [qsub args...]}"
V="${2:?}"; M="${3:?}"; JOB="${4:?}"; shift 4

bash "$HERE/preflight.sh" "$W" "$V" "$M" "$JOB"
rc=$?
if [ $rc -ne 0 ]; then
  echo "  REFUSING to submit: preflight failed (exit $rc)." >&2
  echo "  Fix the items marked BAD above, then re-run." >&2
  exit $rc
fi

if command -v /opt/pbs/bin/qsub >/dev/null 2>&1; then
  echo "  preflight passed; submitting $JOB"
  exec /opt/pbs/bin/qsub "$@" "$JOB"
elif command -v sbatch >/dev/null 2>&1; then
  echo "  preflight passed; submitting $JOB"
  exec sbatch "$@" "$JOB"
else
  echo "  no qsub or sbatch found" >&2; exit 127
fi
