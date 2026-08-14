#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N emb-freeze-probe
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -l select=1
#PBS -q debug
#PBS -j oe
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT CCL_PROCESS_LAUNCHER=pmix CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu" TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
export PYTHONPATH="${PBS_O_WORKDIR:-$(pwd)}:${PYTHONPATH}"
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 ZE_AFFINITY_MASK=0
OUT="outputs/fp32-norms-ablation/${PBS_JOBID%%.*}-emb"; mkdir -p "$OUT"
P=29800
for ARM in A B C; do
  P=$((P+1))
  ARM=$ARM STEPS=300 MASTER_PORT=$P python3 -m torchtitan.experiments.ezpz.scripts.oneoff.emb_visited_probe 2>&1 | tee "$OUT/emb_$ARM.log"
done
echo "=== embedding visited-row summary ==="
grep -h '"arm"' "$OUT"/emb_*.log | tee "$OUT/summary.txt"
