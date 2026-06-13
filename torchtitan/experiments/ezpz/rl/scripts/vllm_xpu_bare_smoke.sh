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
# Force oneCCL to use MPI transport (default before vllm 0.22 flipped
# to OFI). OFI needs libpsm2/libucp which aren't on the runtime LD
# path with just the oneapi module loaded.
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export CCL_PROCESS_LAUNCHER="${CCL_PROCESS_LAUNCHER:-pmix}"
export CCL_OP_SYNC="${CCL_OP_SYNC:-1}"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# vLLM uses ZMQ IPC for its EngineCore IPC, but Linux's sockaddr_un.sun_path
# is limited to 107 chars. The default PBS-job TMPDIR
# (/var/tmp/pbs.<long-jobid>/...) exceeds that once vLLM appends a UUID.
# Override to a short path under /tmp.
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
# Source ezpz so we get a populated `ezpz launch` (mpiexec wrapper)
# that provides the PMIx env vLLM 0.22's EngineCore wants for its
# torch.distributed bootstrap even at TP=1.
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
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
# Run as a real .py file (not a heredoc) so vLLM's
# multiprocessing.spawn worker can `runpy.run_path()` it.
# Launch via `ezpz launch --np 1` so PMIx is initialized — vLLM 0.22's
# EngineCore creates a TP=1 world group via XCCL even at single rank,
# which fails at `MPIR_pmi_init` if launched without an MPI bootstrap.
NP="${NP:-1}"
PPN="${PPN:-1}"
MODEL="${MODEL}" "${SUBMIT_DIR}/.venv/bin/ezpz" launch --np "${NP}" -ppn "${PPN}" \
    "${SUBMIT_DIR}/venvs/rl-actors/bin/python" \
    "${SUBMIT_DIR}/torchtitan/experiments/ezpz/rl/scripts/vllm_xpu_bare_smoke.py" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "VERDICT: complete" | tee -a "${LOG_DIR}/run.log"
