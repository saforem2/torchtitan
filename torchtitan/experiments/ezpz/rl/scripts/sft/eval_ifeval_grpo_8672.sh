#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=02:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
#
# Higher-signal evals for the completed full-mix 8N SFT (checkpoint-8672-hf):
# IFEval (instruction following) + a GRPO smoke (sum_digits), each SFT vs the
# gs138650 baseline. Runs both back-to-back on one 4N allocation:
#   1. IFEval  -- single XPU tile, 2 models (eval_ifeval_sft_vs_baseline.sh)
#   2. GRPO    -- 4N/48 ranks, 50 steps x 2 models (grpo_smoke_sft_vs_baseline.sh)
# Both scripts default (via env) to SFT_MODEL=checkpoint-8672-hf,
# BASELINE_MODEL=${HOME}/global_step138650. See those scripts for details.
#
# Output: IFEval -> outputs/evals/aurora2b-ifeval-<ts>/ ; GRPO ->
# outputs/grpo-smoke/<label>/ ; logs under logs/.
set -o pipefail
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"
DIR="torchtitan/experiments/ezpz/rl/scripts/sft"

echo "########## IFEval (checkpoint-8672-hf vs gs138650) ##########"
bash "${DIR}/eval_ifeval_sft_vs_baseline.sh" || echo "IFEval returned nonzero (continuing to GRPO)"

echo "########## GRPO smoke (checkpoint-8672-hf vs gs138650) ##########"
bash "${DIR}/grpo_smoke_sft_vs_baseline.sh" || echo "GRPO returned nonzero"

echo "=== eval_ifeval_grpo_8672 DONE ==="
