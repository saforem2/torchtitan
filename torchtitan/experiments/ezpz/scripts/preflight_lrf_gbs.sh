#!/bin/bash
# preflight_lrf_gbs.sh — LOCAL-ONLY validator for Sirius GBS LR-finder ladder
# Runs on mbph (no ssh, no PBS). Validates:
#   (1) submit_lr_finder_sirius.sh has LRF_MAX_LR=1e-1 default
#   (2) config_registry.py has agpt_5b_olmo2tok_smoke
#   (3) GBS values 32/512/2048 are power-of-2 and world_size-compatible
#   (4) GAS for GBS_VALUES is power-of-2, <=32, finite (LBS-adjusted: 1 for <1024, 2 for >=1024; GBS=2048 GAS=32 via LBS=2 -- see sirius.sh auto-LBS block)
# Usage:
#   bash torchtitan/experiments/ezpz/scripts/preflight_lrf_gbs.sh
# Exits 0 on all PASS, 1 on any FAIL.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/submit_lr_finder_sirius.sh"
LADDER_SCRIPT="${SCRIPT_DIR}/submit_lr_finder_gbs_ladder_sirius.sh"
CONFIG_REGISTRY="${SCRIPT_DIR}/../agpt/config_registry.py"

FAILURES=0

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }

echo "=========================================================="
echo " preflight_lrf_gbs — local validation (no PBS, no ssh)"
echo "   SCRIPT_DIR : ${SCRIPT_DIR}"
echo "=========================================================="
echo ""

# ---------------------------------------------------------------------------
# (1) submit_lr_finder_sirius.sh has LRF_MAX_LR=1e-1 default
# ---------------------------------------------------------------------------
echo "--- check 1: LRF_MAX_LR default in submit_lr_finder_sirius.sh ---"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
    fail "base script not found: ${BASE_SCRIPT}"
else
    # Expect: export LRF_MAX_LR="${LRF_MAX_LR:-1e-1}"
    if grep -q 'LRF_MAX_LR.*:-1e-1' "${BASE_SCRIPT}"; then
        pass "submit_lr_finder_sirius.sh has LRF_MAX_LR default 1e-1"
        # Show the matching line for audit
        grep 'LRF_MAX_LR' "${BASE_SCRIPT}" | sed 's/^/  /'
    else
        fail "submit_lr_finder_sirius.sh missing LRF_MAX_LR=1e-1 default (expected ':-1e-1')"
        if grep -q 'LRF_MAX_LR' "${BASE_SCRIPT}"; then
            echo "  found LRF_MAX_LR lines (wrong default):"
            grep 'LRF_MAX_LR' "${BASE_SCRIPT}" | sed 's/^/  /'
        else
            echo "  no LRF_MAX_LR line found at all"
        fi
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# (2) config_registry.py has agpt_5b_olmo2tok_smoke
# ---------------------------------------------------------------------------
echo "--- check 2: config_registry.py has agpt_5b_olmo2tok_smoke ---"
if [[ ! -f "${CONFIG_REGISTRY}" ]]; then
    fail "config_registry.py not found: ${CONFIG_REGISTRY}"
else
    if grep -q 'def agpt_5b_olmo2tok_smoke' "${CONFIG_REGISTRY}"; then
        pass "config_registry.py has agpt_5b_olmo2tok_smoke"
        grep 'def agpt_5b_olmo2tok_smoke' "${CONFIG_REGISTRY}" | sed 's/^/  /'
    else
        fail "config_registry.py missing def agpt_5b_olmo2tok_smoke"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# (3) GBS values power-of-2 and world_size-compatible
# ---------------------------------------------------------------------------
echo "--- check 3: GBS values power-of-2 + world_size divisibility ---"
GBS_VALUES=(32 512 2048)
# Sirius hedge: 8N * 4 GPUs = 32-way DP. GBS must be divisible by world_size
# so each rank gets an integer micro-batch slice. Also power-of-2 aligns FSDP
# sharding (FFN 8192 % 32 == 0) and keeps GAS math clean.
WORLD_SIZE=32

is_power_of_two() {
    local n=$1
    # n > 0 and (n & (n-1)) == 0
    (( n > 0 && (n & (n - 1)) == 0 ))
}

for gbs in "${GBS_VALUES[@]}"; do
    # power-of-2
    if is_power_of_two "${gbs}"; then
        pass "GBS=${gbs} is power-of-2"
    else
        fail "GBS=${gbs} is NOT power-of-2"
    fi
    # world_size divisibility
    if (( gbs % WORLD_SIZE == 0 )); then
        pass "GBS=${gbs} divisible by world_size=${WORLD_SIZE} (GAS=$(( gbs / WORLD_SIZE )))"
    else
        fail "GBS=${gbs} NOT divisible by world_size=${WORLD_SIZE} (remainder $(( gbs % WORLD_SIZE )))"
    fi
done

# Also validate the ladder script's GBS list matches exactly (32 512 2048)
# This is an extra guard — not strictly required by spec but ensures acceptance.
echo ""
echo "--- check 3b: ladder script GBS list + base reference ---"
if [[ ! -f "${LADDER_SCRIPT}" ]]; then
    fail "ladder script not found: ${LADDER_SCRIPT}"
else
    if grep -q 'submit_lr_finder_sirius\.sh' "${LADDER_SCRIPT}"; then
        pass "ladder script references submit_lr_finder_sirius.sh"
    else
        fail "ladder script does not reference submit_lr_finder_sirius.sh"
    fi
    # Count GBS values — extract the GBS_VALUES line and verify it has 32, 512, 2048
    # We check the literal array definition to ensure exactly those 3.
    if grep -q 'GBS_VALUES=(32 512 2048)' "${LADDER_SCRIPT}"; then
        pass "ladder script has exactly GBS_VALUES=(32 512 2048)"
    else
        fail "ladder script GBS_VALUES line != 'GBS_VALUES=(32 512 2048)'"
        grep 'GBS_VALUES' "${LADDER_SCRIPT}" | sed 's/^/  /' || echo "  (no GBS_VALUES line)"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# (4) GAS for GBS_VALUES: power-of-2, <=32, finite (LBS-adjusted)
# ---------------------------------------------------------------------------
echo "--- check 4: GAS validation (GBS*4096/(LBS*4096*32), LBS rule mirrors sirius.sh) ---"
# Reproduce submit_lr_finder_sirius.sh LBS rule: LBS=1 default, auto-raised to 2 when GBS>=1024.
# Formula: GAS = GBS * SEQ_LEN / (LBS * SEQ_LEN * dp) with SEQ_LEN=4096, dp=32 (Sirius 8N).
# SEQ_LEN cancels algebraically but kept for contract fidelity.
LRF_SEQ_LEN=4096
DP=32
for gbs in "${GBS_VALUES[@]}"; do
    # Reproduce LBS rule from submit_lr_finder_sirius.sh
    if (( gbs >= 1024 )); then
        LBS=2
    else
        LBS=1
    fi
    # GAS = GBS * 4096 / (LBS * 4096 * 32) -- keep as integer math, check divisibility
    _gas_num=$(( gbs * LRF_SEQ_LEN ))
    _gas_den=$(( LBS * LRF_SEQ_LEN * DP ))
    if (( _gas_num % _gas_den != 0 )); then
        fail "GBS=${gbs} LBS=${LBS} GAS not finite: ${gbs}*4096=${_gas_num} not divisible by ${LBS}*4096*32=${_gas_den}"
        continue
    fi
    GAS=$(( _gas_num / _gas_den ))
    # GBS=2048 was GAS=64 under LBS=1 (NCCL timeout); now GAS=32 via LBS=2.
    if (( GAS == 0 )); then
        fail "GBS=${gbs} LBS=${LBS} GAS=${GAS} is zero"
    elif ! is_power_of_two "${GAS}"; then
        fail "GBS=${gbs} LBS=${LBS} GAS=${GAS} is NOT power-of-2"
    else
        pass "GBS=${gbs} LBS=${LBS} GAS=${GAS} is power-of-2"
    fi
    if (( GAS > 32 )); then
        fail "GBS=${gbs} LBS=${LBS} GAS=${GAS} exceeds 32"
    else
        pass "GBS=${gbs} LBS=${LBS} GAS=${GAS} <= 32"
    fi
    # finite already checked via mod; expected: GBS32=1, GBS512=16, GBS2048=32
done
unset _gas_num _gas_den GAS LBS LRF_SEQ_LEN DP

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================================="
if (( FAILURES == 0 )); then
    echo " preflight PASSED — all checks ok"
    echo "=========================================================="
    exit 0
else
    echo " preflight FAILED — ${FAILURES} check(s) failed"
    echo "=========================================================="
    exit 1
fi
