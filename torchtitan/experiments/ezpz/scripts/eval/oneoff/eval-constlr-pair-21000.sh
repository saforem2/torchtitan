#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-constlr-pair-21000
#PBS -j oe
#
# Matched-pair downstream eval at step 21,000: the canonical 2B-512 chain
# (decaying LR) vs the constant-LR fork branched from it at step-9200.
#
# WHY: at matched steps the two arms sit within +0.001..+0.008 nats -- inside
# noise (docs/experiments/agpt/aurora/20260817-2b-512-constant-lr-fork.md).
# Loss is not the deliverable: v1-vs-v2 showed two chains can sit close on loss
# and diverge sharply on downstream evals. This asks the eval question directly.
#
# THE TWO ARMS NEED DIFFERENT RoPE FLAVORS. This is the whole reason this is a
# bespoke script and not two invocations of the stock sweep. Per W&B run
# metadata (the authoritative record of what each run executed):
#
#   canonical step-21000 -> run 21grc6o7 (2026-05-26)  -> flavor `2b`      (complex)
#   fork      step-21000 -> run xii94czx (2026-08-16)  -> flavor `2b_real` (cos_sin)
#
# The chains crossed the 2026-06-25 CONFIG_SUFFIX=_real switch at different
# times; the fork launched 2026-08-13, after it, while canonical reached this
# step in May, before it. Converting BOTH with one flavor -- the obvious move
# for a "matched pair" -- scrambles Q/K on one arm and reads as a large fake
# regression. Re-derive with:
#   scripts/eval/rope_flavor_for_step.py --chain <chain> --step <N>
#
# CONVERT_REPO IS THE MAIN REPO, NOT THE PINNED CLONES. Neither pinned clone
# ships agpt/state_dict_adapter.py, so they fall back to the bare
# Llama3StateDictAdapter, which permutes UNCONDITIONALLY -- a `2b_real` request
# there is silently ignored and yields exactly the corrupt export the flavor
# flag exists to prevent. eval-2b-v2.sh refuses this combination (exit 2);
# pointing CONVERT_REPO at the main checkout is the fix, not a workaround.
#
# CONFOUND, STATE IT IN ANY WRITEUP: the fork differs from canonical in BOTH
# the LR schedule AND the RoPE convention. A downstream gap here is therefore
# not attributable to the schedule alone. The RoPE conventions are supposed to
# be numerically equivalent when each is exported with its matching flavor, but
# "supposed to be" is not a measurement.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

MAIN_REPO=/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd "${PBS_O_WORKDIR:-$MAIN_REPO}" || exit 1

STEP="${STEP:-21000}"
# Explicit 0-shot on every task. The eval scripts write `<task>@<N>shot` keys
# PLUS a bare `<task>` alias for whichever group ran LAST, so leaving shots
# implicit lets a later few-shot group overwrite the bare key a reader assumes
# is 0-shot. Pinning one group makes the bare alias unambiguous.
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
export SHOTS_SPEC="${SHOTS_SPEC:-0:${TASKS}}"

echo "=========================================================="
echo "[pair] matched-pair eval at step ${STEP}"
echo "[pair] tasks: ${TASKS}"
echo "[pair] shots: ${SHOTS_SPEC}"
echo "[pair] convert repo (has state_dict_adapter): ${MAIN_REPO}"
echo "=========================================================="

rc_total=0

run_arm () {
    local label="$1" v2repo="$2" ckpt="$3" flavor="$4"
    echo ""
    echo "=========================================================="
    echo "[pair] ARM ${label}  flavor=${flavor}"
    echo "[pair]   ckpt: ${v2repo}/outputs/checkpoints/${ckpt}/step-${STEP}"
    echo "=========================================================="
    STEPS="${STEP}" \
    V2_REPO="${v2repo}" \
    CONVERT_REPO="${MAIN_REPO}" \
    CKPT_NAME="${ckpt}" \
    LABEL="${label}" \
    MODEL_FLAVOR="${flavor}" \
    TASKS="${TASKS}" \
    SHOTS_SPEC="${SHOTS_SPEC}" \
    bash "${MAIN_REPO}/torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh"
    local rc=$?
    echo "[pair] ARM ${label} exit=${rc}"
    # Exit code is NOT the artifact. The stock script SKIPS a step whose
    # results.json already covers the requested tasks and still exits 0, so a
    # no-op run is indistinguishable from a successful one by rc alone.
    # Verified against the file in the summary below.
    [[ $rc -ne 0 ]] && rc_total=1
    return 0
}

run_arm "constlr-canon-21000" \
        "/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz" \
        "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288" \
        "2b"

run_arm "constlr-fork-21000" \
        "/flare/AuroraGPT/foremans/runs/agpt-2b-constlr-from9200/torchtitan-ezpz" \
        "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200" \
        "2b_real"

echo ""
echo "=========================================================="
echo "[pair] ARTIFACT CHECK (not the exit code -- see note above)"
echo "=========================================================="
for label in constlr-canon-21000 constlr-fork-21000; do
    f="outputs/evals/agpt-2b-v2-${label}/step-${STEP}/results/results.json"
    if [[ -f "$f" ]]; then
        echo "[pair] ${label}: results.json present"
        TASKS="${TASKS}" python3 - "$f" "$label" <<'PYSUM'
import json, os, sys
path, label = sys.argv[1], sys.argv[2]
want = [t for t in os.environ.get("TASKS", "").split(",") if t]
try:
    d = json.load(open(path))
except Exception as e:
    print("  [pair] UNREADABLE: %r" % (e,)); sys.exit(0)
missing = [t for t in want
           if t not in d and "%s@0shot" % t not in d]
for t in want:
    row = d.get("%s@0shot" % t, d.get(t))
    if not isinstance(row, dict):
        continue
    # report acc_norm when present (the standard read for these tasks), else acc
    for m in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
        if m in row:
            print("    %-16s %-12s %.4f" % (t, m.split(",")[0], row[m]))
            break
if missing:
    print("  [pair] MISSING TASKS: %s" % ",".join(missing))
PYSUM
    else
        echo "[pair] ${label}: NO results.json -- arm did not produce output"
        rc_total=1
    fi
done

echo ""
echo "[pair] done (rc_total=${rc_total})"
exit $rc_total
