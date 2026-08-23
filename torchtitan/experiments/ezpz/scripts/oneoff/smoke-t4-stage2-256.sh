#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -l select=5
#PBS -N smoke-t4-stage2-256
#PBS -j oe
#
# Validate the NEW t4 seat (2B-256 stage-2 dolmino) before it costs 266 nodes
# in a 2098-node umbrella.
#
# THE ONE QUESTION: does step-92859 load under rope=complex and resume at its
# parent's loss (~2.65), rather than the ~7.2 that the 2026-07-18 attempt
# (job 8663177) produced by loading the same seed under agpt_2b_real?
#
# That number IS the test. A flavor mismatch loads cleanly and only shows up as
# loss, so "it started" proves nothing:
#   ~2.6-2.7  -> flavor correct, seed good. Ship the seat.
#   ~7.2      -> flavor still wrong. Do NOT launch at 256N.
#   >>7.2     -> something else; read the traceback.
#
# Secondary: confirm the symlinked blendcorpus index cache HITS. The 648 train
# descriptors are shared with the 512N seat because steps = tok/(gbs*seq_len)
# makes gbs*steps identical at both node counts (291,790,848 samples, which
# reproduces the on-disk "Number of samples 941148" exactly). The 324 VALID
# descriptors will MISS -- eval_samples = gbs*eval_iters does not cancel
# (614,400 vs 1,228,800) -- so a one-time valid-index build at startup is
# EXPECTED, not a misconfiguration.
#
# Runs through the umbrella launcher rather than a hand-built ezpz launch:
# three earlier hand-built attempts each burned an allocation on a different
# reconstruction error. MULTI_ONLY=4 picks this seat alone and keeps its real
# config/dataset/RoPE; MULTI_NNODES_OVERRIDE shrinks it.
#
# NOTE ON GBS: at 4 nodes gbs becomes 4*12*2 = 96, not 6144. That changes the
# cache key, so THIS SMOKE WILL BUILD ITS OWN INDEX and will not exercise the
# symlinked cache. It still answers the flavor question, which is the one that
# can corrupt a chain. Cache-hit verification needs a real 256N launch.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

MAIN=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd "${PBS_O_WORKDIR:-$MAIN}" || exit 1

V=/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz
SEED=$V/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859

if [[ ! -f "$SEED/.metadata" ]]; then
    echo "[smoke] FATAL: seed missing: $SEED" >&2; exit 2
fi
echo "[smoke] seed: $SEED"
echo "[smoke]   shards=$(ls "$SEED" | grep -c distcp)  (3072 expected: 256 nodes x 12)"

# The real seat writes to checkpoints/agpt-2b-stage2-dolmino-n256-gbs6144.
# Point the smoke at a THROWAWAY dir instead: field 5 drives checkpoint.folder,
# and a non-empty one would make torchtitan ignore initial_load_path
# (checkpoint.py:686) -- the trap that wasted job 8771637.
SMOKE_CKPT="checkpoints/_smoke/stage2-256-${PBS_JOBID%%.*}"
SMOKE_ABS="$V/outputs/$SMOKE_CKPT"
if compgen -G "$SMOKE_ABS/step-*" > /dev/null; then
    echo "[smoke] FATAL: $SMOKE_ABS already holds a step-*; seed would be ignored." >&2
    exit 2
fi

SMOKE_SCRIPT=$(mktemp /tmp/smoke-stage2-256-XXXXXX.sh)
python3 - "$MAIN/torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh" \
         "$SMOKE_SCRIPT" "$SMOKE_CKPT" <<'PYEOF'
import re, sys
src, dst, smoke_ckpt = sys.argv[1:4]
s = open(src).read()
rows = re.findall(r'^\s+"((?:2b|20b)\|[^"]+)"', s, re.M)
target = next(r for r in rows if "stage2-dolmino-n256" in r)
f = target.split("|")
assert len(f) == 12, "expected 12 fields, got %d" % len(f)
assert f[11] == "complex", "rope field is %r, expected complex" % f[11]
assert "step-92859" in f[7], "field 8 is not the 92859 seed: %r" % f[7]
f[4] = smoke_ckpt
s = s.replace('"%s"' % target, '"%s"' % "|".join(f), 1)
open(dst, "w").write(s)
print("[smoke] rewrote t4 ckpt_dir -> %s (rope=complex, seed=step-92859)" % smoke_ckpt)
PYEOF
rc=$?
[[ $rc -ne 0 ]] && { echo "[smoke] FATAL: rewrite failed" >&2; exit 1; }

echo ""
echo "=========================================================="
echo "[smoke] launching the stage-2 256N seat alone, 8 steps"
echo "=========================================================="
MULTI_ONLY=4 \
MULTI_NNODES_OVERRIDE=4 \
MULTI_STEPS_OVERRIDE=8 \
SPARES=1 \
CKPT_INTERVAL=100000 \
IDLE_TIMEOUT=900 \
bash "$SMOKE_SCRIPT"
echo "[smoke] umbrella exit=$?"
rm -f "$SMOKE_SCRIPT"

CONSOLE=$(ls -t "$MAIN"/logs/multi-autoretry-"${PBS_JOBID%%.*}"/trainer-0-*.console.log 2>/dev/null | head -1)
echo ""
echo "=========================================================="
if [[ -n "$CONSOLE" && -f "$CONSOLE" ]]; then
    echo "[smoke] config actually launched:"
    grep -m1 -oE 'config=agpt_2b[a-z_]*' "$CONSOLE" | sed 's/^/  /'
    echo "  (want config=agpt_2b -- NOT agpt_2b_real)"
    if grep -q "initial_load_path is provided but the checkpoint.folder exists" "$CONSOLE"; then
        echo "  RESULT: INVALID -- seed ignored; the smoke dir was not empty."
    fi
    echo ""
    echo "[smoke] step lines:"
    grep -E "step: +[0-9]+" "$CONSOLE" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-150 | tail -10 | sed 's/^/  /'
else
    echo "[smoke] WARNING: no console log under logs/multi-autoretry-${PBS_JOBID%%.*}/"
fi
echo "=========================================================="
echo "[smoke] VERDICT -- READ THE STEP-1 LOSS"
echo "  ~2.6-2.7 -> rope=complex correct, seed good. Ship the seat."
echo "  ~7.2     -> flavor STILL wrong (this is what 8663177 showed). Do not launch 256N."
echo "  no steps -> failed before training; read the traceback."
echo "=========================================================="
