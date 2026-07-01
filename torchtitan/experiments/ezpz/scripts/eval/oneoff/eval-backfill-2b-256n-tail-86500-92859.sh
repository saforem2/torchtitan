#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Backfill the 2B 256N chain TAIL after chain completion.
#
# The 2B 256N async chain COMPLETED at step-92,859 (4.674T tokens, 100%)
# on 2026-06-29, but the eval curve stopped at step-86,200. This fills the
# tail at 500-step cadence PLUS the final completion checkpoint (92,859).
#
# Steps (14 ckpts, all confirmed DCP-present + not-yet-evald 2026-07-01):
#   86500 87000 87500 88000 88500 89000 89500
#   90000 90500 91000 91500 92000 92500 92859
#
# Per-ckpt time: ~7 min (HellaSwag dominates) => ~1.5h total, fits 8h walltime.
# eval-2b-v2.sh skip-if-results-exist will skip any ckpt already done if this
# job restarts, so re-submitting is safe.
#
# DCP source: /flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/
#   outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-{N}
# Output (this clone): outputs/evals/agpt-2b-v2-256n/step-{N}/{hf,results}/

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="86500 87000 87500 88000 88500 89000 89500 90000 90500 91000 91500 92000 92500 92859" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
