#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -l select=5
#PBS -N smoke-t4-converted
#PBS -j oe
#
# Does the t4 constlr seed actually resume once its optimizer state is in the
# post-#3623 flat format -- or is step-9500 corrupt like step-9200?
#
# WHY THIS RUNS THROUGH THE UMBRELLA LAUNCHER. Three earlier attempts
# hand-built the `ezpz launch` invocation and burned an allocation each:
#   1. `ezpz: command not found`   -- no venv activation
#   2. exit 2 in arg parsing       -- `--optimizer.name` vs `--optimizer=`
#   3. "Optimizer SophiaG not added" -- ezpz launch needs
#      --nproc/--nproc_per_node/--hostfile and a `--` separator, and must run
#      from the NODE-LOCAL venv, none of which I had.
# Every fix exposed the next, because the error was reconstructing the command
# SHAPE at all. submit_agpt_multi_autoretry.sh already launches this exact seat
# correctly, so use it: MULTI_ONLY=4 selects t4 alone and
# MULTI_NNODES_OVERRIDE shrinks it. The seat keeps its REAL ckpt dir, config,
# dataset and RoPE flavor -- unlike MULTI_PROFILE=tiny, which rewrites those.
#
# THE ONE THING OVERRIDDEN is the seed. t4's own ckpt dir still holds the
# OLD-format step-9500, so resuming in place just reproduces
#   AttributeError: 'dict' object has no attribute 'mul_'
# (SophiaG's `hessian` arriving nested instead of flat). T4_SEED points
# --checkpoint.initial-load-path at the CONVERTED copy instead.
#
# RoPE: the seat now carries rope=complex (field 12), matching its parent
# 2b_v2_256, which never switched to cos_sin. Before that fix this smoke would
# have loaded complex weights under agpt_2b_real and produced a bad-loss result
# indistinguishable from a corrupt seed.
#
# READ THE LOSS, NOT THE EXIT CODE:
#   ~2.7 / grad_norm ~0.7  -> seed is FINE; it was only the optimizer format.
#                             Re-seed the real seat from the converted ckpt.
#   ~6.5 / grad_norm ~18   -> seed is CORRUPT like step-9200. Conversion cannot
#                             save it; re-seed from a later 256N checkpoint.
#
# 5 nodes = 4 train + 1 spare. 2B is pure DP and DCP reshards on load, so node
# count does not affect whether the seed loads.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

MAIN=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd "${PBS_O_WORKDIR:-$MAIN}" || exit 1

T4_SEED="${T4_SEED:-$MAIN/outputs/checkpoints/_convert/t4-step9500-newfmt/step-9500}"

if [[ ! -f "$T4_SEED/.metadata" ]]; then
    echo "[smoke-t4] FATAL: converted seed missing or unusable: $T4_SEED" >&2
    echo "  Build it with scripts/oneoff/convert_smoke_t4_step9500.sh first." >&2
    exit 2
fi
echo "[smoke-t4] converted seed: $T4_SEED"
echo "[smoke-t4]   shards=$(ls "$T4_SEED" | grep -c distcp)  (1 is expected --"
echo "[smoke-t4]   torch_save_to_dcp writes one consolidated shard; DCP reshards on load)"

# Inject the seed into t4's TRAINERS row (field 8, init_load) without editing
# the production file: rewrite it into a temp copy and run that.
SMOKE_SCRIPT=$(mktemp /tmp/smoke-t4-umbrella-XXXXXX.sh)
python3 - "$MAIN/torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh" \
         "$SMOKE_SCRIPT" "$T4_SEED" <<'PYEOF'
import re, sys
src, dst, seed = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
rows = re.findall(r'^\s+"((?:2b|20b)\|[^"]+)"', s, re.M)
target = next(r for r in rows if "constlr-from9500" in r)
f = target.split("|")
assert len(f) == 12, "expected 12 fields, got %d" % len(f)
assert not f[7], "init_load already set to %r -- refusing to clobber" % f[7]
f[7] = seed
s = s.replace('"%s"' % target, '"%s"' % "|".join(f), 1)
open(dst, "w").write(s)
print("[smoke-t4] injected initial-load-path into the t4 row")
PYEOF
rc=$?
[[ $rc -ne 0 ]] && { echo "[smoke-t4] FATAL: could not inject the seed" >&2; exit 1; }

echo ""
echo "=========================================================="
echo "[smoke-t4] launching t4 alone via the umbrella launcher"
echo "=========================================================="
MULTI_ONLY=4 \
MULTI_NNODES_OVERRIDE=4 \
SPARES=1 \
CKPT_INTERVAL=1000 \
IDLE_TIMEOUT=900 \
bash "$SMOKE_SCRIPT"
echo "[smoke-t4] umbrella exit=$?"
rm -f "$SMOKE_SCRIPT"

echo ""
echo "=========================================================="
echo "[smoke-t4] VERDICT -- read the step lines above"
echo "  loss ~2.7 / gn ~0.7 -> format only; re-seed the seat from the converted ckpt"
echo "  loss ~6.5 / gn ~18  -> seed corrupt; re-seed from a later 256N checkpoint"
echo "  no step lines       -> it failed BEFORE training; read the traceback"
echo "=========================================================="
