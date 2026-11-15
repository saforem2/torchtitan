#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:15:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# preflight_sft_b3.sh — PBS preflight gate for SFT B3 (agpt2b-b3-instruct-cot-mix)
#
# Validates on a Sunspot workq allocation (1 node, 15 min) that:
#   (1) VENV rl-monarch-torch214-locality-20261001 + LD-first cache is usable
#   (2) torch + trl import and print versions
#   (3) BASE_MODEL outputs/evals/cpt-fwe/step-74000/hf exists with
#       config.json + tokenizer.json + model-*.safetensors
#   (4) pretokenized dataset /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192
#       has dataset_info.json and assistant_masks column with expected counts
#
# Usage (on Sunspot, from repo root):
#   qsub torchtitan/experiments/ezpz/scripts/preflight_sft_b3.sh
#   # optional overrides:
#   # qsub -v BASE_MODEL=/custom/hf,PRETOK_DIR=/custom/pretok torchtitan/experiments/ezpz/scripts/preflight_sft_b3.sh
#
# Do NOT run via local bash -n only; this job is intended for Sunspot workq.
# Local syntax check:  bash -n torchtitan/experiments/ezpz/scripts/preflight_sft_b3.sh
#
set -euo pipefail

REPO="${PBS_O_WORKDIR:-$(pwd)}"
# Sunspot HOME repo (per skill §1); allow override via REPO env if submitted elsewhere.
REPO="${REPO:-/home/foremans/torchtitan}"
VENV="/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001"
# BASE_MODEL per skill §5 — relative to REPO unless absolute
BASE_MODEL="${BASE_MODEL:-outputs/evals/cpt-fwe/step-74000/hf}"
PRETOK_DIR="${PRETOK_DIR:-/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192}"

echo "=========================================================="
echo " preflight_sft_b3 — Sunspot workq 00:15:00 select=1"
echo "   REPO       : ${REPO}"
echo "   VENV       : ${VENV}"
echo "   BASE_MODEL : ${BASE_MODEL}"
echo "   PRETOK_DIR : ${PRETOK_DIR}"
echo "   host       : $(hostname 2>/dev/null || echo unknown)"
echo "   date       : $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date)"
echo "=========================================================="
echo ""

FAILURES=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }

# Resolve BASE_MODEL to absolute if relative (qsub workdir is REPO on Sunspot)
if [[ "${BASE_MODEL}" != /* ]]; then
    BASE_MODEL_ABS="${REPO}/${BASE_MODEL}"
else
    BASE_MODEL_ABS="${BASE_MODEL}"
fi
echo "Resolved BASE_MODEL_ABS=${BASE_MODEL_ABS}"
echo ""

# ---------------------------------------------------------------------------
# (1) ezpz startup + VENV existence
# ---------------------------------------------------------------------------
echo "--- check 1: VENV rl-monarch-torch214-locality-20261001 exists ---"
if [[ ! -d "${VENV}" ]]; then
    fail "VENV missing: ${VENV}"
else
    pass "VENV dir exists: ${VENV}"
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
    fail "VENV python missing: ${VENV}/bin/python"
else
    pass "VENV python exists: ${VENV}/bin/python ($("${VENV}/bin/python" --version 2>&1 || echo 'version check failed'))"
fi
echo ""

echo "--- check 1b: ezpz startup (source + ezpz_setup VENV) ---"
# Machine-agnostic startup per assignment contract — no hard-coded oneapi/ZE_FLAT shims.
# shellcheck source=/dev/null
if ! source <(curl -fsSL https://ezpz.cool/utils.sh) 2>&1; then
    fail "curl https://ezpz.cool/utils.sh failed"
else
    pass "sourced https://ezpz.cool/utils.sh"
fi
# ezpz_setup with explicit VENV path (not ezpz_setup_job)
if ! command -v ezpz_setup >/dev/null 2>&1; then
    fail "ezpz_setup not found after sourcing utils.sh"
else
    pass "ezpz_setup function available"
    if ! ezpz_setup "${VENV}" 2>&1; then
        fail "ezpz_setup ${VENV} failed"
    else
        pass "ezpz_setup ${VENV} succeeded"
    fi
fi
echo ""

echo "--- check 1c: LD-first cache ---"
# VENV rl-monarch-torch214-locality-20261001 requires LD-first for UR/SYCL libs
# (libur_*, libsycl.so*, libumf*, libtcm* win over system). Verify LD_LIBRARY_PATH
# contains the xpu-ld-first cache or VENV lib after ezpz_setup.
if [[ -n "${LD_LIBRARY_PATH:-}" ]] && echo "${LD_LIBRARY_PATH}" | grep -q "xpu-ld-first\|${VENV}/lib"; then
    pass "LD_LIBRARY_PATH contains LD-first/VENV lib: ${LD_LIBRARY_PATH}"
else
    # Fallback: check that the cache dir would be created on next run (B3 scripts create it)
    _ld_first="${REPO}/.cache/xpu-ld-first/$(basename "${VENV}")"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        fail "LD_LIBRARY_PATH does not contain xpu-ld-first or VENV lib (got: ${LD_LIBRARY_PATH}); expected _ld_first=${_ld_first}"
    else
        fail "LD_LIBRARY_PATH empty/unset after ezpz_setup; expected LD-first cache at ${_ld_first}"
    fi
fi
# Also verify VENV actually ships the expected XPU libs
if ls "${VENV}"/lib/libur_* >/dev/null 2>&1 || ls "${VENV}"/lib/libsycl.so* >/dev/null 2>&1; then
    pass "VENV ships XPU libs (libur_*/libsycl.so*)"
else
    fail "VENV missing expected XPU libs at ${VENV}/lib/libur_* and libsycl.so*"
fi
echo ""

# ---------------------------------------------------------------------------
# (2) torch + trl import
# ---------------------------------------------------------------------------
echo "--- check 2: python -c import torch + trl ---"
if python -c "import torch; import trl; print(torch.__version__)" 2>&1; then
    pass "import torch + trl succeeded (torch $(python -c 'import torch; print(torch.__version__)' 2>&1))"
else
    fail "import torch + trl failed"
fi
echo ""

# ---------------------------------------------------------------------------
# (3) BASE_MODEL gate — must exist and contain config + tokenizer + safetensors
# ---------------------------------------------------------------------------
echo "--- check 3: BASE_MODEL=${BASE_MODEL_ABS} ---"
if [[ ! -d "${BASE_MODEL_ABS}" ]]; then
    fail "BASE_MODEL missing: ${BASE_MODEL_ABS} (expected outputs/evals/cpt-fwe/step-74000/hf)"
else
    pass "BASE_MODEL dir exists: ${BASE_MODEL_ABS}"
fi
if [[ -f "${BASE_MODEL_ABS}/config.json" ]]; then
    pass "BASE_MODEL has config.json"
else
    fail "BASE_MODEL missing config.json at ${BASE_MODEL_ABS}/config.json"
fi
# shellcheck disable=SC2012  # ls tokenizer.* is intentional: small known set
if [[ -f "${BASE_MODEL_ABS}/tokenizer.json" ]]; then
    pass "BASE_MODEL has tokenizer.json"
elif [[ -f "${BASE_MODEL_ABS}/tokenizer.model" ]] || [[ -f "${BASE_MODEL_ABS}/tokenizer_config.json" ]] || ls "${BASE_MODEL_ABS}"/tokenizer.* >/dev/null 2>&1; then
    pass "BASE_MODEL has tokenizer.* (fallback): $(ls "${BASE_MODEL_ABS}"/tokenizer.* 2>&1 | tr '\n' ' ')"
else
    fail "BASE_MODEL missing tokenizer.json/tokenizer.* at ${BASE_MODEL_ABS}/"
fi
# shellcheck disable=SC2012  # ls model-*.safetensors count is bounded
if ls "${BASE_MODEL_ABS}"/model-*.safetensors >/dev/null 2>&1; then
    _n_safetensors=$(ls -1 "${BASE_MODEL_ABS}"/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    _sz=$(du -sh "${BASE_MODEL_ABS}" 2>/dev/null | cut -f1 || echo "?")
    pass "BASE_MODEL has model-*.safetensors (${_n_safetensors} files, total ${_sz})"
else
    fail "BASE_MODEL missing model-*.safetensors at ${BASE_MODEL_ABS}/"
fi
echo ""

# ---------------------------------------------------------------------------
# (4) Pretokenized dataset gate
# ---------------------------------------------------------------------------
echo "--- check 4: pretokenized dataset ${PRETOK_DIR} ---"
if [[ ! -d "${PRETOK_DIR}" ]]; then
    fail "PRETOK_DIR missing: ${PRETOK_DIR}"
else
    pass "PRETOK_DIR exists: ${PRETOK_DIR}"
fi
if [[ -f "${PRETOK_DIR}/dataset_info.json" ]]; then
    pass "dataset_info.json present"
    # Show num_examples if parseable
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json, sys
try:
    with open('${PRETOK_DIR}/dataset_info.json') as f:
        info = json.load(f)
    # HF datasets dataset_info.json has splits or dataset_size
    print('  dataset_info.json:', json.dumps(info, indent=2)[:2000])
except Exception as e:
    print(f'  (could not parse dataset_info.json: {e})')
" 2>&1 || true
    fi
else
    fail "PRETOK_DIR missing dataset_info.json at ${PRETOK_DIR}/dataset_info.json"
fi
# Verify arrow/parquet data files exist
# shellcheck disable=SC2012  # ls data-*.arrow count is bounded and globs are safe here
if ls "${PRETOK_DIR}"/data-*.arrow >/dev/null 2>&1 || ls "${PRETOK_DIR}"/*.parquet >/dev/null 2>&1 || [[ -f "${PRETOK_DIR}/state.json" ]]; then
    pass "dataset data files present ($(ls -1 "${PRETOK_DIR}"/data-*.arrow "${PRETOK_DIR}"/*.parquet 2>/dev/null | wc -l | tr -d ' ') arrow/parquet files)"
else
    fail "PRETOK_DIR has no data-*.arrow / *.parquet / state.json"
fi
# Deep check: load via datasets and verify assistant_masks column + counts
echo "  (loading dataset to verify assistant_masks — may take ~10s)"
if python3 -c "
from datasets import load_from_disk
import sys
ds = load_from_disk('${PRETOK_DIR}')
print(f'  dataset: {len(ds)} rows, columns={ds.column_names}')
if 'assistant_masks' not in ds.column_names:
    print('  FAIL: assistant_masks not in columns', file=sys.stderr)
    sys.exit(2)
# Sample counts: fraction of masked tokens, basic sanity
sample = ds[0]
am = sample.get('assistant_masks')
print(f'  row 0 assistant_masks len={len(am) if hasattr(am, \"__len__\") else \"?\"} sum={sum(am) if am is not None else \"?\"} (expect >0 masked tokens)')
# Check a few more rows to ensure masks are not all-zero
nonzero = 0
for i in range(min(100, len(ds))):
    row = ds[i]
    m = row.get('assistant_masks')
    if m is not None and sum(m) > 0:
        nonzero += 1
print(f'  first 100 rows: {nonzero}/100 have nonzero assistant_masks')
if nonzero == 0:
    print('  FAIL: all assistant_masks zero in first 100 rows', file=sys.stderr)
    sys.exit(3)
print('  assistant_masks OK')
" 2>&1; then
    pass "assistant_masks column present and nonzero in sample"
else
    _rc=$?
    if [[ ${_rc} -eq 2 ]]; then
        fail "dataset missing assistant_masks column"
    elif [[ ${_rc} -eq 3 ]]; then
        fail "dataset assistant_masks all zero (no training signal)"
    else
        fail "dataset load/verify failed (rc=${_rc})"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================================="
if (( FAILURES == 0 )); then
    echo " preflight_sft_b3 PASSED — all gates ok"
    echo "   VENV        : ${VENV} (LD-first + torch/trl)"
    echo "   BASE_MODEL  : ${BASE_MODEL_ABS} (config+tokenizer+safetensors)"
    echo "   PRETOK_DIR  : ${PRETOK_DIR} (dataset_info + assistant_masks)"
    echo "=========================================================="
    exit 0
else
    echo " preflight_sft_b3 FAILED — ${FAILURES} check(s) failed"
    echo "=========================================================="
    exit 1
fi
