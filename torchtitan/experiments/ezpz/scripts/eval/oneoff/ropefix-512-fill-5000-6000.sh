#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N ropefix-512-fill
#PBS -j oe
#
# Make the 20b-512 ropefix arm uniform: steps 5000 and 6000 currently hold
# ONLY winogrande, while the other 34 hold boolq/openbookqa/piqa/winogrande.
#
# HOW THAT HAPPENED. Those two steps were missing from the sweep's STEPS list
# (it jumped 4900->5100 and 5900->6010), so the original run never created
# their ropefix dirs at all -- it reported "34 ok, 0 skipped, 0 failed", which
# was honest for the list it was given. The gap-fill job (8770172) passed
# TASKS=winogrande, which was right for the 34 dirs that already had the other
# three tasks and WRONG for these two, which started empty. So the fill created
# them with a single task and swapped one asymmetry for another.
#
# Nothing was lost: the arc/hellaswag/mmlu results for these steps live in the
# separate, untouched agpt-20b-v2-512n tree. Only the -ropefix tree is short.
#
# This exists as a script rather than `qsub -v` because a comma-separated
# TASKS value cannot be passed that way -- PBS parses commas in -v as variable
# separators and rejects the whole thing with "cannot send environment with
# the job". Setting the vars inside the script sidesteps the quoting entirely.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}" || exit 1

export CHAIN=20b-512
export STEPS="5000 6000"
export TASKS="boolq,openbookqa,piqa,winogrande"

echo "[fill] CHAIN=$CHAIN STEPS='$STEPS' TASKS=$TASKS"
echo "[fill] the content-aware guard should SKIP winogrande (already present)"
echo "[fill] and run the other three."

bash torchtitan/experiments/ezpz/scripts/eval/oneoff/reeval-ropefix-sweep.sh
rc=$?
echo "[fill] sweep exit=$rc"

# Verify the ARTIFACT, not the exit code: this sweep's own "N ok" counter was
# honest while the input list was short, so a clean exit proves nothing about
# coverage. Assert all four tasks are present on both steps.
echo ""
echo "=========================================================="
echo "[fill] COVERAGE CHECK"
python3 - <<'PYEOF'
import json, os, sys
base = "outputs/evals/agpt-20b-v2-512n-ropefix"
want = {"boolq", "openbookqa", "piqa", "winogrande"}
bad = 0
for step in (5000, 6000):
    p = os.path.join(base, "step-%d" % step, "results", "results.json")
    try:
        have = {k for k in json.load(open(p)) if not k.startswith("mmlu_")}
    except Exception as e:
        print("  step-%d UNREADABLE: %r" % (step, e)); bad += 1; continue
    missing = sorted(want - have)
    print("  step-%d: %s%s" % (step, sorted(have & want),
                               ("   MISSING %s" % missing) if missing else "   complete"))
    if missing:
        bad += 1
sys.exit(1 if bad else 0)
PYEOF
echo "[fill] coverage rc=$?  (0 = both steps hold all four tasks)"
echo "=========================================================="
