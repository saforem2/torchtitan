#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N fp32-norms-ablation
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=1
#PBS -q debug
#PBS -j oe

# Master-weight-dtype ablation on the agpt debugmodel (~21M params).
#
# Settles: is a NORMS-ONLY fp32 master sufficient, or is the shipped
# full-fp32 master doing real work on the non-norm parameters?
#
#   arm A  bf16 master everywhere       (expected: RMSNorm.weight frozen)
#   arm B  fp32 master for norms only   (EZPZ_FP32_NORMS=1 + dtype=bfloat16)
#   arm C  fp32 master everywhere       (production default)
#
# All three arms run sequentially inside ONE job at identical seed / data /
# optimizer / step count, so any parameter difference is attributable to the
# master dtype alone. Each arm dumps per-parameter statistics to JSON; compare
# with compare_fp32_norms_ablation.py.
#
# Single process on one XPU tile: the debugmodel is tiny, and one rank keeps
# the comparison free of DP-reduction nondeterminism between arms. The FSDP
# mesh still exists at degree 1, which is what installs the
# MixedPrecisionPolicy (parallel_dims._mesh_exist keeps the fsdp mesh alive at
# degree 1 for exactly this reason), so all three arms exercise the real
# master-weight path.
#
# See docs/production/agpt/30b-exp/exp02-fp32-norms-ablation.md.

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate

STEPS="${STEPS:-300}"
SEED="${SEED:-42}"
LR="${LR:-8e-4}"
OUTDIR="${OUTDIR:-outputs/fp32-norms-ablation/${PBS_JOBID%%.*}}"
mkdir -p "${OUTDIR}"

# Single-rank torch.distributed world (gloo/xccl still initializes; FSDP needs
# a process group even at degree 1).
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29577}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

echo "==========================================="
echo "fp32-norms ablation: steps=${STEPS} seed=${SEED} lr=${LR}"
echo "PBS_JOBID=${PBS_JOBID}  OUTDIR=${OUTDIR}"
echo "python: $(which python3)"
python3 -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available())"
echo "==========================================="

for ARM in A B C; do
    echo "--- arm ${ARM} ---"
    python3 -m torchtitan.experiments.ezpz.scripts.oneoff.fp32_norms_ablation \
        --arm "${ARM}" \
        --steps "${STEPS}" \
        --seed "${SEED}" \
        --lr "${LR}" \
        --out "${OUTDIR}/arm${ARM}.json" \
        2>&1 | tee "${OUTDIR}/arm${ARM}.log"
done

echo "=== comparison ==="
python3 torchtitan/experiments/ezpz/scripts/oneoff/compare_fp32_norms_ablation.py \
    "${OUTDIR}/armA.json" "${OUTDIR}/armB.json" "${OUTDIR}/armC.json" \
    | tee "${OUTDIR}/comparison.txt"
