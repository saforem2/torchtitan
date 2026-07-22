#!/bin/bash
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# PBS wrapper: 1-node (2-tile) Monarch CoT GRPO+LoRA smoke on reason_agpt.
# Runs the interactive agpt2b_grpo_cot.sh on the allocated node. Validates the
# Monarch CoT path clears startup + step 13 before a longer run.
REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9
export WALLTIME_SEC=3000   # leave headroom under the 1h PBS walltime
bash torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_grpo_cot.sh
