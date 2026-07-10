#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=02:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N offline pre-build of the tulu_math_uc_mix cache (tulu-3 0.65 +
# OpenMathInstruct-2 0.15 + ultrachat 0.20, all_exhausted -> ~93.3M rows,
# recipe v2 -> cache hash 44608c8c15bd9714). Runs ONCE on a compute node so the
# 384-rank SFT job just Dataset.load_from_disk() in <5s (no build, no oneCCL
# barrier). With the vectorized interleave (commit b5798f532) the index build is
# ~5s; the remaining time is select(93M) + empty-row filter + save_to_disk.
#
# Compute node (not login) because it writes a large cache (>100GB) and does
# heavy parallel .map/.filter -- do not hammer the login node with that.
#
# Output: cache at ~/.cache/ezpz_sft_mixes/44608c8c15bd9714 ; log in logs/.
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_OFFLINE=1   # all 3 sources are already HF-cached; no downloads

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${SUBMIT_DIR}:${PYTHONPATH:-}"

LOG_DIR="logs/prebuild-openmath-mix-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

python3 - <<'PY' 2>&1 | tee "${LOG_DIR}/run.log"
import time
from torchtitan.experiments.ezpz.rl.datasets_sft import (
    _build_tulu_math_uc_mix, EZPZ_SFT_MIX_CACHE_DIR,
)
t0 = time.time()
ds = _build_tulu_math_uc_mix()   # recipe v2 -> hash 44608c8c15bd9714
print(f"[prebuild] DONE rows={len(ds):,} in {time.time()-t0:.0f}s", flush=True)
print(f"[prebuild] cache root: {EZPZ_SFT_MIX_CACHE_DIR}", flush=True)
PY
echo "=== DONE: log in ${LOG_DIR}/ ==="