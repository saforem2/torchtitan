#!/bin/bash
# preflight_fixed_lr.sh — LOCAL-ONLY validator for fixed-LR convergence on Sirius
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/submit_lr_finder_sirius.sh"
FIXED_SCRIPT="${SCRIPT_DIR}/submit_fixed_lr_sirius.sh"
CONFIG_REGISTRY="${SCRIPT_DIR}/../agpt/config_registry.py"
FAILURES=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }
echo "=========================================================="
echo " preflight_fixed_lr — local validation (no PBS, no ssh)"
echo "   SCRIPT_DIR : ${SCRIPT_DIR}"
echo "=========================================================="
echo ""
echo "--- check 1: submit_lr_finder_sirius.sh has LRF_MAX_LR=1e-1 ---"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  fail "missing ${BASE_SCRIPT}"
elif grep -q 'LRF_MAX_LR.*1e-1' "${BASE_SCRIPT}"; then
  pass "LRF_MAX_LR=1e-1 in base script"
else
  fail "LRF_MAX_LR=1e-1 not found in ${BASE_SCRIPT}"
fi
echo ""
echo "--- check 2: config_registry.py has 5b + 10b smoke configs ---"
if [[ ! -f "${CONFIG_REGISTRY}" ]]; then
  fail "missing ${CONFIG_REGISTRY}"
elif grep -q "def agpt_5b_olmo2tok_smoke" "${CONFIG_REGISTRY}" && grep -q "def agpt_10b_olmo2tok_smoke" "${CONFIG_REGISTRY}"; then
  pass "found 5b + 10b smoke configs"
else
  fail "missing 5b or 10b smoke config"
fi
echo ""
echo "--- check 3: submit_fixed_lr_sirius.sh is valid fixed-LR (not finder) ---"
if [[ ! -f "${FIXED_SCRIPT}" ]]; then
  fail "missing ${FIXED_SCRIPT}"
else
  if grep -q 'select=8' "${FIXED_SCRIPT}"; then pass "has select=8"; else fail "missing select=8"; fi
  if grep -q 'walltime=' "${FIXED_SCRIPT}"; then pass "has walltime"; else fail "missing walltime"; fi
  if grep -q 'torchtitan/.venv' "${FIXED_SCRIPT}"; then pass "uses ~/torchtitan/.venv"; else fail "missing ~/torchtitan/.venv"; fi
  if grep -q 'training.steps.*1000\|FIXED_STEPS.*1000' "${FIXED_SCRIPT}"; then pass "has training.steps 1000"; else fail "missing training.steps 1000"; fi
  if grep -q -- '--optimizer.lr' "${FIXED_SCRIPT}"; then pass "has --optimizer.lr override"; else fail "missing --optimizer.lr override"; fi
  # Gated: fixed-LR matrix comments must note TT_CONFIG_JSON pattern and GBS scaling-law gate
  if grep -q 'TT_CONFIG_JSON' "${FIXED_SCRIPT}"; then pass "notes TT_CONFIG_JSON override pattern"; else fail "missing TT_CONFIG_JSON comment (intended --optimizer.lr / TT_CONFIG_JSON pattern)"; fi
  if grep -q 'trajectory_fingerprint' "${FIXED_SCRIPT}"; then pass "notes trajectory_fingerprint gate"; else fail "missing trajectory_fingerprint gate comment"; fi
  if grep -q 'GBS scaling law\|GBS.*32.*512.*2048\|GBS_VALUES' "${FIXED_SCRIPT}"; then pass "notes GBS scaling-law gate (32/512/2048)"; else fail "missing GBS scaling-law gate note (32/512/2048)"; fi
fi
if (( FAILURES == 0 )); then echo " preflight PASSED — all checks ok"; echo "=========================================================="; exit 0; else echo " preflight FAILED — ${FAILURES} check(s) failed"; echo "=========================================================="; exit 1; fi
