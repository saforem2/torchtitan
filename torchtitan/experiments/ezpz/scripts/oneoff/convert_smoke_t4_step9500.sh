#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -l select=4
#PBS -N convert-smoke-t4-9500
#PBS -j oe
#
# Is the t4 constlr seed (step-9500) merely OLD-FORMAT, or is it CORRUPT?
#
# BACKGROUND. The t4 seat (2b-256 constlr-from9500) has produced zero training
# steps across five umbrellas. Two causes have been peeled off already:
#   1. `Missing key ... lm_head.weight`      -> ckpt_key_compat, fixed
#   2. `Missing key ... optimizer.state...`  -> the clone was running a stale
#      copy of that shim with NO optimizer-namespace handling; fixed by copying
#      main's version in (2026-08-20).
# With those gone it now fails one layer deeper, in SophiaG itself:
#       state["hessian"].mul_(beta2)...
#       AttributeError: 'dict' object has no attribute 'mul_'
# i.e. the optimizer state LOADS but arrives nested, not flat -- the documented
# pre-#3623 format migration (docs/reference/known-bugs/pre3623-optim-statedict-  docs-link-check: ignore
# resume.md). The converter bridges exactly that.
#
# WHY THIS IS A TEST AND NOT A FIX. The converter is PROVEN (full 4-link dress
# rehearsal, 2026-07-22, jobs 8686135/8686136). What is NOT established is that
# THIS SEED is sound. step-9200 -- from the same early region of the same chain
# -- is independently corrupt: it resumes to loss ~6.5 both natively AND after
# conversion. step-9500 has never produced a step, so it has never been shown
# to be good. Converting it is therefore a falsifiable experiment:
#
#   PASS  loss ~2.7  grad_norm ~0.7   -> seed fine, it was only the format;
#                                        re-seed the seat and it should run
#   FAIL  loss ~6.5  grad_norm ~18    -> seed is corrupt like step-9200;
#                                        stop converting, re-seed from a later
#                                        known-good ckpt on the 256N chain
#
# Both outcomes are worth the 4 nodes. A FAIL here is the cheaper way to learn
# it than another 266-node umbrella seat dying at startup.
#
# 4 nodes is enough: 2B is pure DP and DCP reshards on load, so node count does
# not affect the converter (noted in the migration memo).

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

MAIN=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd "${PBS_O_WORKDIR:-$MAIN}" || exit 1

SEED_DIR=/flare/AuroraGPT/foremans/runs/agpt-2b-constlr-from9200/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500
export SRC="${SRC:-$SEED_DIR/step-9500}"
# REF only donates the per-param fused/foreach scalar flags, so any NEW-format
# 2B checkpoint works. asyncfix-verify2 is the one the proven run used.
export REF="${REF:-$(ls -d outputs/checkpoints/agpt-2b-asyncfix-verify2/step-* 2>/dev/null | tail -1)}"
export DST="${DST:-$MAIN/outputs/checkpoints/_convert/t4-step9500-newfmt/step-9500}"
export TMP="${TMP:-/tmp/t4_9500_flat.pt}"
export RTMP="${RTMP:-/tmp/t4_ref_flat.pt}"

echo "=========================================================="
echo "[t4] SRC = $SRC"
echo "[t4] REF = $REF"
echo "[t4] DST = $DST"
echo "=========================================================="

if [[ -z "$REF" || ! -d "$REF" ]]; then
    echo "[t4] FATAL: no REF new-format checkpoint found. The converter needs one" >&2
    echo "     to read per-param fused/foreach from. Pass REF=<dir> explicitly." >&2
    exit 2
fi
if [[ ! -d "$SRC" ]]; then
    echo "[t4] FATAL: SRC seed missing: $SRC" >&2
    exit 2
fi

echo ""
echo "### STEP 1/2: convert old -> new optimizer format"
# Resumable: the conversion is deterministic and takes ~15 min, so a re-run
# after a LATER stage failed (the first attempt died in step 2 on a missing
# venv) should not redo it. Skip only when DST is actually usable, not merely
# present -- a half-written DST is exactly what would make the smoke lie.
if [[ -f "$DST/.metadata" ]] && [[ $(ls "$DST" 2>/dev/null | grep -c distcp) -ge 1 ]]; then
    echo "[t4] DST already converted -- skipping (delete it to force a redo)"
    rc=0
else
    .venv/bin/python3 torchtitan/experiments/ezpz/scripts/convert_step9200_optim_flags.py
    rc=$?
fi
echo "[t4] converter exit=$rc"
# Exit code is not the artifact -- verify the DST actually materialized. The
# converter writing nothing and returning 0 is the failure mode this repo
# specializes in.
if [[ $rc -ne 0 ]]; then
    echo "[t4] FATAL: converter failed" >&2; exit 1
fi
n_shards=$(ls "$DST" 2>/dev/null | grep -c distcp)
echo "[t4] DST shards=$n_shards  .metadata=$([[ -f $DST/.metadata ]] && echo yes || echo NO)"
if [[ "$n_shards" -lt 1 || ! -f "$DST/.metadata" ]]; then
    echo "[t4] FATAL: converter exited 0 but produced no usable checkpoint" >&2
    exit 1
fi

echo ""
echo "### STEP 2/2: smoke-resume the CONVERTED seed (8 steps)"
echo "### PASS = loss ~2.7 / grad_norm ~0.7   FAIL = loss ~6.5 / grad_norm ~18"
SMOKE_OUT="$MAIN/outputs/checkpoints/_convert/t4-smoke-9500"
# `ezpz` is a venv entry point, NOT on PATH in a bare PBS shell -- the first
# run of this script converted fine (it calls .venv/bin/python3 explicitly)
# and then died here with "ezpz: command not found", exit 127, wasting the
# allocation after the useful work was already done. Activate first.
source "$MAIN/.venv/bin/activate" || { echo "[t4] FATAL: cannot activate venv" >&2; exit 2; }
# Flags copied VERBATIM from t4's own invocation in umbrella 8764675, not
# adapted from the docs. Attempt 2 died in arg parsing because I guessed
# `--optimizer.name sophiag` (it is `--optimizer=sophiag`) and used
# space-separated flags where this build wants `=`. Reading the working
# command line off the running job is the only reliable source for these.
#   --config is agpt_2b_real (cos_sin RoPE), NOT ezpz_agpt_2b
#   GBS 6144 / LBS 2 / seq 8192 matches the 256N seat this seed belongs to
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config=agpt_2b_real \
    --job.dump-folder="$SMOKE_OUT" \
    --checkpoint.initial-load-path="$DST" \
    --checkpoint.no-enable \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path=torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt \
    --optimizer=sophiag \
    --optimizer.lr=2.28e-5 \
    --lr-scheduler.decay-ratio=0.0 \
    --lr-scheduler.min-lr-factor=1.0 \
    --lr-scheduler.warmup-steps=0 \
    --training.local-batch-size=2 \
    --training.global-batch-size=6144 \
    --training.seq-len=8192 \
    --training.steps=8 \
    2>&1 | tail -70
echo "[t4] smoke exit=$?"

echo ""
echo "=========================================================="
echo "[t4] READ THE LOSS ABOVE, NOT THE EXIT CODE."
echo "  loss ~2.7, grad_norm ~0.7  -> seed OK; it was only the optim format."
echo "                                Re-seed the t4 seat from $DST."
echo "  loss ~6.5, grad_norm ~18   -> seed CORRUPT like step-9200; conversion"
echo "                                cannot save it. Re-seed from a later"
echo "                                known-good ckpt on the 256N chain."
echo "=========================================================="
