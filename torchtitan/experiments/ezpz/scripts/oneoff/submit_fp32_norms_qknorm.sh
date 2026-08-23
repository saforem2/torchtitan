#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N fp32-norms-qknorm
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -l select=1
#PBS -q debug
#PBS -j oe
# QK-Norm arm of the master-weight-dtype ablation. Section 6 of the 30B
# proposal predicted the bf16 freeze "recurs for any parameter initialized
# near 1.0 -- QK-norm gains being exactly that". debugmodel_qknorm adds 12
# head_dim RMSNorms at init 1.0 on top of the 13 block norms, so this
# measures that prediction directly.
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT CCL_PROCESS_LAUNCHER=pmix CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu" TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 ZE_AFFINITY_MASK=0
OUT="outputs/fp32-norms-ablation/${PBS_JOBID%%.*}-qknorm"; mkdir -p "$OUT"
P=29850
for ARM in A B C; do
  P=$((P+1))
  MASTER_PORT=$P python3 -m torchtitan.experiments.ezpz.scripts.oneoff.fp32_norms_ablation \
    --arm $ARM --steps 300 --seed 42 --config agpt_debugmodel_qknorm_local \
    --out "$OUT/arm$ARM.json" 2>&1 | tee "$OUT/arm$ARM.log"
done
python3 torchtitan/experiments/ezpz/scripts/oneoff/compare_fp32_norms_ablation.py \
  "$OUT/armA.json" "$OUT/armB.json" "$OUT/armC.json" | tee "$OUT/comparison.txt"
