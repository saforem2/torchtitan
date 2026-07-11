#!/bin/bash --login
# PBS wrapper: run the synthetic-summary step on a single Aurora XPU node.
#
# Step 2 of the "summarize olmo-mix" POC. Summarizes the JSONL text slice
# (from detok_to_text.py) with an instruction-tuned model via plain HF
# transformers generation on one tile. This is a PILOT-scale wrapper
# (debug-scaling, 1 node, 1h) -- not a production data-gen pipeline.
#
# NOTE: no `set -euo pipefail` (venv activate trips unbound vars; see ezpz
# CLAUDE.md). Single node, TP=1: NO `ezpz launch` / mpiexec wrapper -- bare
# python, exactly like the vLLM XPU smoke (an outer launcher adds nothing at
# TP=1 and breaks oneCCL PMIx state).
#
# Submit (from repo root on an Aurora login node):
#   qsub -A AuroraGPT -q debug-scaling -l select=1 -l walltime=01:00:00 \
#     -l filesystems=flare:home \
#     torchtitan/experiments/ezpz/synthetic/submit_summarize.sh
#
# Override defaults with -v, e.g.:
#   qsub ... -v LIMIT=16,IN=outputs/synthetic/wiki_slice.jsonl,MODEL=/path/to/model ...

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

# --- inputs / outputs (override via qsub -v) ---
IN="${IN:-outputs/synthetic/wiki_slice.jsonl}"
OUT="${OUT:-outputs/synthetic/wiki_summaries.jsonl}"
MODEL="${MODEL:-/flare/AuroraGPT/azton/models/Llama-3.1-8B-exvocab}"
MAX_SRC_CHARS="${MAX_SRC_CHARS:-6000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-0}"   # 0 = all docs; set small (e.g. 4) for a smoke

# --- XPU runtime env ---
# Mirror the interactive train launcher (scripts/train_agpt_2b_venv.sh) EXACTLY.
# The .venv is a torch-2.13 XPU build; torch.xpu.is_available() is only True once
# the oneAPI compute runtime is loaded via an EXPLICIT
#   module load oneapi/release/2025.3.1 hdf5 pti-gpu
# BEFORE the ZE_/ONEAPI_ exports and ezpz_setup_job. Relying on ezpz_setup_job
# alone (job 8664725) or `module load frameworks/...` (job 8664598) left XPU at 0.
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# transformers-only; keep it fully offline/local (no hub calls for a local dir)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils) && ezpz_setup_job

# Use the Lustre .venv directly (single node, no yeet needed for a pilot).
source .venv/bin/activate
PY="${PY:-python}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

LOG_DIR="logs/synthetic-summarize-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}" "$(dirname "${OUT}")"

{
  echo "=== synthetic summarize (bare HF generate, 1 XPU tile) ==="
  echo "date:   $(date)"
  echo "IN:     ${IN}"
  echo "OUT:    ${OUT}"
  echo "MODEL:  ${MODEL}"
  echo "LIMIT:  ${LIMIT}  BATCH:${BATCH_SIZE}  MAX_NEW:${MAX_NEW_TOKENS}  MAX_SRC_CHARS:${MAX_SRC_CHARS}"
  echo "PY:     ${PY}"
  echo ""
} | tee "${LOG_DIR}/run.log"

# Pin to a single tile. Under ZE_FLAT_DEVICE_HIERARCHY=FLAT tiles are addressed
# as flat integers (0..11), NOT the composite "device.subdevice" form -- setting
# ZE_AFFINITY_MASK="0.0" here is what made torch.xpu see ZERO devices (jobs
# 8664598/8664725); the probe (no mask) saw all 12. Use a bare integer, and only
# if explicitly requested; otherwise leave all tiles visible and let
# model.to("xpu") land on xpu:0.
[[ -n "${ZE_AFFINITY_MASK_TILE:-}" ]] && export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK_TILE}"

"${PY}" -u -m torchtitan.experiments.ezpz.synthetic.summarize_text \
    --in "${IN}" \
    --out "${OUT}" \
    --model "${MODEL}" \
    --max-src-chars "${MAX_SRC_CHARS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --limit "${LIMIT}" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

rc="${PIPESTATUS[0]}"
echo "VERDICT: rc=${rc}" | tee -a "${LOG_DIR}/run.log"
exit "${rc}"
