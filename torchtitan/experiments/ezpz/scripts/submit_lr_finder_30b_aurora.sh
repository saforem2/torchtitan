#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-30b-lr-finder
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -l select=256
#PBS -q prod
#PBS -j oe
#
# 30B LR finder at the PRODUCTION global batch, across adamw / mano / sophiag.
#
# WHY GBS=6144 AND NOT SOMETHING CHEAPER. The optimal LR is batch-size
# dependent, and calibrating at the wrong batch produces a number that does not
# transfer. The 80B learned this expensively: its small-batch finder said
# 1.1e-5, but the real production-batch ceiling was ~14x lower, so runs at the
# "found" LR sat past the NaN cliff. Sweeping at the batch the production run
# will actually use is the entire point.
#
# WHY 256 NODES -- IT IS THE ONLY LEGAL CHOICE, NOT A PREFERENCE.
# 64N/LBS=4 is the throughput sweet spot (exp06: 456 tps, 26.57% MFU) and would
# finish three optimizers in ~7h. But no Aurora queue accepts it. Verified with
# `qstat -Qf`:
#     debug-scaling  nodect <= 256   walltime <= 01:00:00
#     small (prod)   nodect >= 256   walltime <= 12:00:00
#     large          nodect >= 2000  walltime <= 24:00:00
# Everything from 8 to 255 nodes above one hour is a queue dead zone. 256 is the
# smallest count that reaches a multi-hour queue -- and at 256N the LBS*GAS
# budget is exactly 2, which gives LBS=2/GAS=1: the production shape itself.
#
# LBS=5 IS ARITHMETICALLY IMPOSSIBLE HERE, despite being the live chain's value.
# 6144 = 2^11 * 3 has no factor of 5, so
# `global_batch_size % (local_batch_size * batch_degree) == 0`
# (trainer.py:417-423) fails at EVERY node count. Under a pinned GBS=6144 the
# admissible LBS set is powers of two only. This is worth stating because
# exp07's production pick of LBS=5 cannot be carried into a GBS=6144 sweep.
#
# SEQ_LEN=4096, not the script's hardcoded 8192. Every 30B measurement that
# exists is at 4096 (exp05/06/07/08); 8192 has a single datapoint. GBS counts
# SEQUENCES, so 8192 would double tokens/step -- 50.3M instead of 25.2M -- and
# double the wall clock for no calibration benefit, while doubling activation
# memory at a size where exp05 already found AC=none OOMs.
#
# CONFIG: agpt_30b_olmo2tok, NOT agpt_30b. The bare name is the gemma-256k-vocab
# 28.1B variant; the converged chain and every recent measurement use the olmo2
# 26.2B one. run_lr_finder.sh would have composed the wrong one from LRF_MODELS.
#
# LR WINDOW: the script default 1e-6 -> 1.0 is wrong for this problem in two
# ways. The finder probes ONE LR PER STEP (lr_finder.py:118-192: mult =
# (max/init)^(1/sweep_steps), applied to every param_group each step), so 100
# steps over 6 decades is only 16.7 points/decade -- and the 80B answers
# (mano ~3e-6, sophiag ~1e-6) both sit in the FIRST decade, so 83 of the 100
# probes would be spent above 1e-5 where everything has already diverged.
# Worse, init_lr=1e-6 places the expected sophiag minimum AT the first sample,
# making a minimum at or below it unresolvable.
# So: 1e-8 -> 1e-3 (5 decades, a decade of margin either side of the 80B
# answers) at 150 steps = 30 points/decade.
#
# TIMEOUT: LRF_TIMEOUT defaults to 1800s and wraps each run; every viable
# configuration here needs longer, so all three optimizers would be killed and
# reported TIMEOUT. Set explicitly.
#
# NOT SWEEPING MUON: the user asked for adamw/mano/sophiag. (For context, Muon
# is broken at 80B via bf16 overflow in Newton-Schulz at dim=9216; 30B's 6144
# might be survivable, but that is a separate question.)

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

export LRF_MODELS="30b"
export LRF_CONFIG="agpt_30b_olmo2tok"
export LRF_OPTIMIZERS="adamw mano sophiag"

# GBS=6144 = dp(3072) * LBS(2) * GAS(1) at 256 nodes, TP=1.
export LRF_GBS=6144
export LRF_LBS=2
export LRF_SEQ_LEN=4096

# 150 probes over 5 decades = 30 points/decade.
export LRF_STEPS=1000
export LRF_FRACTION=0.15
export LRF_INIT_LR=1e-8
export LRF_MAX_LR=1e-3

# 30B needs full AC (exp05: none OOMs, selective errors) and trains compiled.
export LRF_AC=full

# ~19 s/step x 150 = ~48 min of stepping, plus compile (~10-15 min at this size)
# and startup. 100 min per optimizer, 3 optimizers, inside a 6 h walltime.
export LRF_TIMEOUT=6000
export LRF_IDLE_TIMEOUT=1800

# Keep the three optimizers' CSV/plot/npz from colliding with any existing 30B
# finder outputs (the trainer keys that path on model+optimizer, not on GBS).
export LRF_DUMP_FOLDER="outputs/lr_finder_30b_gbs6144"

echo "=========================================================="
echo " 30B LR finder -- PRODUCTION batch"
echo "   config    = $LRF_CONFIG"
echo "   optimizers= $LRF_OPTIMIZERS"
echo "   GBS       = $LRF_GBS  (256N x 12 x LBS $LRF_LBS x GAS 1, TP=1)"
echo "   seq_len   = $LRF_SEQ_LEN   -> $(( LRF_GBS * LRF_SEQ_LEN )) tokens/step"
echo "   LR window = $LRF_INIT_LR -> $LRF_MAX_LR over $(python3 -c "print(int($LRF_STEPS*$LRF_FRACTION))") probes"
echo "=========================================================="

exec bash torchtitan/experiments/ezpz/scripts/run_lr_finder.sh
