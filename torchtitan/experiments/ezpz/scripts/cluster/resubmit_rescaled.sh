#!/bin/bash
# resubmit_rescaled.sh -- re-survey, repoint, and submit the rescaled 80B capture.
#
# The host list inside a pinned .pbs is a SNAPSHOT. Sunspot has nodes that PBS
# reports as `state = free` while /lus/tegu is unreachable on them (one such
# rank exits 127 and kills all 768 -- see
# docs/guides/known-bugs/sunspot-home-mount-check-offlines-nodes.md), so the
# list must be rebuilt from a fresh access test, not from pbsnodes state, and
# not reused from an earlier run.
#
# Usage:  ./resubmit_rescaled.sh [--dry-run]
set -o pipefail

REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
SCRIPT="$REPO/80b_capture_rescaled.pbs"
NEED=32
cd "$REPO" || exit 1
export PATH=/opt/pbs/bin:$PATH

echo "== surveying free nodes for real tegu access =="
bash torchtitan/experiments/ezpz/scripts/cluster/survey_nodes.sh || exit 1

GOOD="$REPO/.cache/patches/verified_good.txt"
[ -s "$GOOD" ] || { echo "no verified nodes; aborting"; exit 1; }

# keep only the ones still free at THIS moment
: > "$REPO/.cache/patches/rs_free.txt"
while read -r n; do
  pbsnodes "$n" 2>/dev/null | grep -qE "state = free" && echo "$n" >> "$REPO/.cache/patches/rs_free.txt"
done < "$GOOD"

HAVE=$(wc -l < "$REPO/.cache/patches/rs_free.txt")
echo "verified AND free: $HAVE (need $NEED)"
if [ "$HAVE" -lt "$NEED" ]; then
  echo "not enough. Options, in order of preference:"
  echo "  - wait for a job to release nodes and re-run this"
  echo "  - drop to 16N: GAS 128 also holds GBS at 25,165,824 exactly"
  echo "    (12N/20N/24N do NOT divide evenly and would shift GBS silently)"
  exit 1
fi

SEL=$(head -"$NEED" "$REPO/.cache/patches/rs_free.txt" \
      | awk '{printf "1:ncpus=208:host=%s+", $1}' | sed 's/+$//')
python3 - "$SCRIPT" "$SEL" <<'PY'
import re, sys
path, sel = sys.argv[1], sys.argv[2]
s = open(path).read()
s2 = re.sub(r"#PBS -l select=[^\n]*\n", "#PBS -l select=" + sel + "\n", s, count=1)
assert s2 != s, "select line not replaced"
assert s2.count("host=") == 32, s2.count("host=")
open(path, "w").write(s2)
print("select repointed to 32 freshly verified hosts")
PY

if [ "$1" = "--dry-run" ]; then
  echo "(dry run) would: qsub $SCRIPT"
  grep -E "^#PBS -N|steps=|warmup-steps=|optimizer-kwargs.lr=" "$SCRIPT" | head -5
  exit 0
fi

echo "== submitting =="
qsub "$SCRIPT"
