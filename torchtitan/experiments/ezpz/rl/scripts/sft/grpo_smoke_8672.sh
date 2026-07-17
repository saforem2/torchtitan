#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
#
# Standalone GRPO smoke (sum_digits, 50 steps) for the completed full-mix 8N
# SFT, SFT ckpt vs the gs138650 baseline. Split out of the combined
# eval_ifeval_grpo_8672.sh wrapper: that wrapper ran IFEval (~97 min for 2
# models) THEN GRPO in one 2h job, so GRPO was starved and PBS-killed at the
# walltime (jobs 12470889/12470896: walltime 7209 exceeded limit 7200).
# GRPO alone (2 models x ~10 min + init) fits comfortably in 1h.
#
# Model selection via the same env overrides as grpo_smoke_sft_vs_baseline.sh:
#   SFT_MODEL   (default checkpoint-8672-hf) -- pass checkpoint-900-hf for the
#               pre-collapse deliverable.
#   SFT_LABEL   (default sft-step8672)
#   BASELINE_MODEL (default ${HOME}/global_step138650)
set -o pipefail
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"
echo "########## GRPO smoke: ${SFT_LABEL:-sft-step8672} vs baseline ##########"
bash torchtitan/experiments/ezpz/rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh
echo "=== grpo_smoke_8672 DONE ==="
