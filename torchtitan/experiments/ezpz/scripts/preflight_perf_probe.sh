#!/bin/bash
# Local-only preflight validator for Sirius perf/MFU probes.
# No ssh, no PBS — safe to run from the repo root on any machine.
# Checks:
#  (1) config_registry.py declares activation_checkpoint_mode values none/full/selective
#  (2) agpt_5b_olmo2tok_smoke exists
#  (3) submit_perf_probe_sirius.sh contains all 5 probe labels and exactly 5 launch blocks / case branches
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_REGISTRY="${SCRIPT_DIR}/../agpt/config_registry.py"
PROBE="${SCRIPT_DIR}/submit_perf_probe_sirius.sh"
FAIL=0

echo "[1] config_registry.py has activation_checkpoint_mode values none/full/selective ..."
for mode in none full selective; do
  if grep -q "\"${mode}\"" "${CONFIG_REGISTRY}"; then
    echo "  PASS: activation_checkpoint_mode == \"$mode\" present"
  else
    echo "  FAIL: missing activation_checkpoint_mode \"$mode\""
    FAIL=1
  fi
done

echo "[2] config_registry.py has agpt_5b_olmo2tok_smoke ..."
if grep -q 'def agpt_5b_olmo2tok_smoke' "${CONFIG_REGISTRY}"; then
  echo "  PASS: agpt_5b_olmo2tok_smoke found"
else
  echo "  FAIL: missing agpt_5b_olmo2tok_smoke"
  FAIL=1
fi

echo "[3] submit_perf_probe_sirius.sh has PBS header select=8 + walltime and shared-NFS mode ..."
if grep -q 'select=8' "${PROBE}" && grep -q 'walltime=' "${PROBE}"; then
  echo "  PASS: PBS header has select=8 and walltime"
else
  echo "  FAIL: missing PBS select=8 or walltime"
  FAIL=1
fi
if grep -q 'torchtitan/.venv' "${PROBE}"; then
  echo "  PASS: uses ~/torchtitan/.venv (shared-NFS, no yeet)"
else
  echo "  FAIL: missing ~/torchtitan/.venv reference"
  FAIL=1
fi
if grep -q 'ezpz_load_modules' "${PROBE}" && grep -q 'ezpz_setup_job' "${PROBE}"; then
  echo "  PASS: calls ezpz_load_modules + ezpz_setup_job"
else
  echo "  FAIL: missing ezpz_load_modules / ezpz_setup_job"
  FAIL=1
fi

echo "[4] submit_perf_probe_sirius.sh defines all 5 probes ..."
for label in baseline compile selective none lbs2; do
  if grep -q "probe_${label}\|\"${label}\"\|${label})" "${PROBE}"; then
    echo "  PASS: probe $label present"
  else
    echo "  FAIL: missing probe $label"
    FAIL=1
  fi
done

LAUNCH_COUNT=$(grep -c '^[[:space:]]*ezpz launch' "${PROBE}" || true)
if [[ "$LAUNCH_COUNT" -eq 5 ]]; then
  echo "  PASS: exactly 5 ezpz launch blocks (one per probe, distinct --checkpoint.folder each)"
else
  echo "  FAIL: expected exactly 5 ezpz launch blocks, found $LAUNCH_COUNT"
  FAIL=1
fi

# The case dispatch must have one branch per probe plus the 'all' aggregator.
for branch in 'baseline)' 'compile)' 'selective)' 'none)' 'lbs2)' 'all)'; do
  if grep -qF "$branch" "${PROBE}"; then
    echo "  PASS: case branch $branch present"
  else
    echo "  FAIL: missing case branch $branch"
    FAIL=1
  fi
done
# Distinct checkpoint folders are the clobber guard.
CKPT_FOLDERS=$(grep -o 'checkpoint\.folder[^"]*"[^"]*"' "${PROBE}" | sort -u | wc -l | tr -d ' ')
if [[ "$CKPT_FOLDERS" -eq 5 ]]; then
  echo "  PASS: 5 distinct --checkpoint.folder values"
else
  echo "  FAIL: expected 5 distinct --checkpoint.folder values, found $CKPT_FOLDERS"
  FAIL=1
fi

if [[ $FAIL -eq 0 ]]; then
  echo "All perf preflight checks PASSED."
else
  echo "Some perf preflight checks FAILED."
fi
exit $FAIL
