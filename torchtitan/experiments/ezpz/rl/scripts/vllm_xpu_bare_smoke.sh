#!/bin/bash --login
#PBS -A datascience
#PBS -N vllm-xpu-bare
#PBS -l walltime=00:15:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Standalone vLLM-XPU smoke (no TRL wrapper). Replays the 2026-06-10
# end-to-end verification from venvs/rl-actors/ (py3.13 + torch 2.12 +
# vllm 0.22 + vllm-xpu-kernels) to confirm the stack still works on a
# fresh allocation. Compare against the earlier vllm_serve_smoke.sh
# which hit a TRL-wrapper platform-detection bug.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/vllm-xpu-bare-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"

MODEL="${MODEL:-outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf}"

echo "=== vllm-xpu bare smoke (rl-actors venv) ===" | tee "${LOG_DIR}/run.log"
echo "MODEL=${MODEL}" | tee -a "${LOG_DIR}/run.log"
date | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Bare vLLM API — no TRL, no server.
venvs/rl-actors/bin/python - <<EOF 2>&1 | tee -a "${LOG_DIR}/run.log"
import torch, vllm
print(f'torch={torch.__version__} vllm={vllm.__version__} xpu={torch.xpu.device_count()}')

from vllm import LLM, SamplingParams

llm = LLM(
    model="${MODEL}",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)
out = llm.generate(["What is 3 + 7 + 2?"], SamplingParams(max_tokens=32, temperature=0))
print("GEN:", out[0].outputs[0].text)
EOF

echo "" | tee -a "${LOG_DIR}/run.log"
echo "VERDICT: complete" | tee -a "${LOG_DIR}/run.log"
