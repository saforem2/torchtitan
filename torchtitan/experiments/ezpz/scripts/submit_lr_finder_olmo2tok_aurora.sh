#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-olmo2tok-lr-finder
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -l select=64
#PBS -q next-eval
#PBS -j oe
#
# LR finder for the OLMo-3-vocab ladder: 4.64B / 9.48B / 26.2B.
# The olmo2tok config/script names are retained for compatibility; the staged
# OLMo-2 tokenizer.json is byte-identical to the OLMo-3 base tokenizer.
#
# WHY THIS EXISTS. agpt() hands every size lr=8e-4 and the only guard
# (_is_80b_config) gates on flavor.startswith("80b"), so none of these three is
# covered. 8e-4 is ~35x the 20B's production 2.28e-5. The 80B shipped its
# inherited default against a documented ~7.4e-7 ceiling and blew up at step 2;
# nothing about these geometries makes that outcome less likely, and the
# difference between a calibrated and an inherited LR is the difference between
# a chain and a wasted allocation.
#
# WHY next-eval, AND WHAT IT CHANGES. `qstat -Qf next-eval` shows NO nodect
# limits at all -- only walltime 00:05:00-06:00:00 and max_queued 20/user. That
# dissolves the constraint submit_lr_finder_30b_aurora.sh was built around:
# there, everything from 8 to 255 nodes above one hour was a queue dead zone,
# forcing 256N purely to reach a multi-hour queue. Here 64N is legal, and 64N
# is the measured throughput sweet spot (exp06: 456 tps, 26.57% MFU at LBS=4).
#
# THE CATCH, AND WHY THE VENV COMES FROM THE 2B-V2 CLONE. next-eval runs a TEST
# bkc image (compute_aurora_test_20260831 / 26.181.0 / oneAPI 2026.1), not the
# PROD image every other queue runs (26.26.0 / oneAPI 2025.3). This repo's
# .venv is based on /opt/aurora/26.26.0/spack/.../python-3.12.12, a path that
# does not exist on the TEST image -- it dies with a bogus "no
# importlib.metadata". The agpt-2b-v2 clone's venv is based on
# $HOME/.local/share/uv/python/cpython-3.14.2, which is image-independent and
# is the one proven to run on compute nodes. So: code from THIS repo, runtime
# from that tarball. LRF_VENV_SRC points the runner at it.
#
# GBS=6144 AT 64 NODES. 64N x 12 ranks = dp 768; 6144 = 768 * 2 * 4, so
# LBS=2/GAS=4. Calibrating at the batch the production run will use is the
# whole point -- the 80B's small-batch finder said 1.1e-5 while the real
# production-batch ceiling was ~14x lower, so runs at the "found" LR sat past
# the NaN cliff.
#
# DATA: the config's OWN dataloader, via LRF_USE_CONFIG_DATALOADER=1. The
# *_smoke configs read precached olmo-mix wiki JSON through Grain and tokenize
# inline with OLMo-3. Every blendcorpus list on Aurora is gemma- or
# Llama-2-tokenized; feeding those ids to a 100,352 embedding is the Polaris
# gibberish failure, and it fails SILENTLY -- ids below the embedding size
# index fine and the loss curve looks plausible.
#
# LR WINDOW: 1e-8 -> 1e-3 (1e-1 for muon), 5-7 decades. Inherited from the 30B
# script for the same reasons: the script default 1e-6 -> 1.0 spends most of
# its probes above 1e-5 where everything has already diverged, and places a
# plausible optimum AT the first sample where a minimum is unresolvable. At
# LRF_FRACTION=0.15 of 1000 steps that is 150 probes = 30 points/decade (21 for
# muon's wider window -- coarser, but a resolved cliff beats a missing one).
#
# OPTIMIZERS: set LRF_OPTIMIZERS, default adamw. The fixed-batch comparison at
# 30B/GBS=960 is COMPLETE and found Mano ahead of AdamW by 0.075 nats at 23.6B
# tokens, so a comparison at these sizes needs Mano measured too; Muon has a
# measured LR (5.68e-04 at GBS=960) but has never been run as a training arm.
#
# A measured LR per optimizer AT THE PRODUCTION BATCH is the precondition for
# any of it: for Mano alone the suggestion spans 4.79e-03 (2B) to ~3e-06 (80B
# at GBS=6144) -- three orders of magnitude for one optimizer. Borrowing a
# number across batch sizes is how the 30B nearly ran SophiaG at 8.5x its
# suggestion and 1.18x below its measured blow-up.
#
# SophiaG sweeps fine (the curve is smooth through the low band) but its
# divergence is state-dependent, not LR-driven: 4/4 replicates at 30B, one with
# the seed pinned and ZERO precursor excursions. A sweep cannot see that, so a
# SophiaG LR from here is a number without a usable arm behind it. Include it
# only to complete the table, never as a launch recommendation, and if an arm
# is ever run then --grad-norm-abort=20.0 is mandatory (it is 0.0 by default).

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

# One size per submission: MODEL_SIZE={5b,10b,30b}, default 5b.
# Three separate jobs rather than LRF_MODELS="5b 10b 30b" in one, so a failure
# in one geometry does not cost the other two their walltime.
MODEL_SIZE="${MODEL_SIZE:-5b}"

export LRF_MODELS="${MODEL_SIZE}"
export LRF_CONFIG="agpt_${MODEL_SIZE}_olmo2tok_smoke"
export LRF_OPTIMIZERS="${LRF_OPTIMIZERS:-adamw}"

# The config owns its dataloader (Grain + OLMo-3 inline tokenization).
export LRF_USE_CONFIG_DATALOADER=1

# Runtime from the image-independent clone venv; see the note above. The venv
# now carries a 2.15 XPU nightly built against the TEST image's oneAPI 2026.1
# runtime, which includes the FSDP spmd-types consumer required after sync 84.
# Keep the TEST-image runtime isolated from the production clone's `.venv` and
# `.venv.tar.gz`; the production umbrella reads those paths at job start.
export LRF_VENV_SRC="/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/.venv.next-eval.tar.gz"
export LRF_ONEAPI_MODULE="oneapi/release/2026.1.0"
export LRF_PREPEND_VENV_LIB=1
export LRF_REJECT_IMPI_RT=1

export LRF_GBS="${LRF_GBS:-6144}"
export LRF_LBS="${LRF_LBS:-2}"
export LRF_SEQ_LEN="${LRF_SEQ_LEN:-4096}"
# Keep the SPMD/FSDP storage mesh divisible by fused parameter dimensions.
# At 64N this is HSDP replicate=96 x shard=8; at a 2N smoke it is 3 x 8.
export LRF_DP_SHARD="${LRF_DP_SHARD:-8}"

export LRF_STEPS="${LRF_STEPS:-1000}"
export LRF_FRACTION="${LRF_FRACTION:-0.15}"
# Window is PER OPTIMIZER, because a sweep that never reaches the cliff yields
# no suggestion at all -- and exits 0 having burned the whole walltime.
# Measured blow-ups at 30B/GBS=960: adamw 3.05e-04, sophiag 3.55e-04,
# mano 5.61e-04, muon 5.68e-03. Muon's is 5.7x ABOVE a 1e-3 top, so the shared
# window would have silently produced nothing for it.
# Each window keeps ~a decade of margin above the known cliff; the batch here
# is 6.4x the batch those were measured at, and optimal LR moves with batch, so
# the margin is doing real work rather than padding.
export LRF_INIT_LR=1e-8
case "${LRF_OPTIMIZERS}" in
    *muon*) export LRF_MAX_LR=1e-1 ;;
    *)      export LRF_MAX_LR=1e-3 ;;
esac

# TP=1: these sizes fit without it, and exp05 measured TP=4 costing 55% of
# throughput on this stack (340 -> 151 tps). The runner's TP block is 80B-only,
# so nothing is set here and the default applies.
export LRF_AC=full

export LRF_TIMEOUT=6000
export LRF_IDLE_TIMEOUT=1800

# Keyed by optimizer set as well as size: run_lr_finder.sh writes under
# ezpz.agpt/<flavor>/<optimizer>/, a path keyed by model+optimizer and NOT by
# GBS or by job, so two concurrent submissions for the same size would
# overwrite each other's CSV/plot/npz.
export LRF_DUMP_FOLDER="outputs/lr_finder_${MODEL_SIZE}_olmo2tok_gbs6144_${LRF_OPTIMIZERS// /-}"

echo "=========================================================="
echo " OLMo-3-vocab LR finder"
echo "   size      = ${MODEL_SIZE}"
echo "   config    = ${LRF_CONFIG}"
echo "   queue     = next-eval (TEST bkc; venv from the 2b-v2 clone)"
echo "   GBS/LBS   = ${LRF_GBS}/${LRF_LBS} @ 64N (dp 768, GAS 4)"
echo "   LR window = ${LRF_INIT_LR} -> ${LRF_MAX_LR}"
echo "   dump      = ${LRF_DUMP_FOLDER}"
echo "=========================================================="

exec bash torchtitan/experiments/ezpz/scripts/run_lr_finder.sh
