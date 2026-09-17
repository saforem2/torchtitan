#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J sync84-numerics-ab
#SBATCH -o %x-%j.out

set -o pipefail
module load pytorch/2.13.0
D=$SCRATCH/sync84
cd $D

export CUBLAS_WORKSPACE_CONFIG=:4096:8   # required by use_deterministic_algorithms
export TORCH_DEVICE=cuda
export WANDB_MODE=disabled

echo "=== node: $(hostname) ==="
./venv/bin/python -c "import torch;print(torch,torch.__version__,cuda,torch.cuda.is_available())"

for FLAVOR in debugmodel 2b; do
  export AB_FLAVOR=$FLAVOR
  echo "=== flavor=$FLAVOR ==="
  for T in tt-premerge tt-merged; do
    ./venv/bin/python $D/numerics_ab.py $D/$T 2>&1 | grep "^JSONLINE" \
      || echo "  $T FAILED (see full log above)"
  done
done
echo "=== done ==="
