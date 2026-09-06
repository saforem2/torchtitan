#!/bin/bash
# survey_nodes.sh -- classify sunspot nodes by whether they can actually reach
# a filesystem, as opposed to whether PBS says they are "free".
#
# WHY THIS EXISTS. Sunspot has nodes that report `state = free` with no comment
# and on which /lus/tegu is unreachable. A rank landing on one cannot cd to the
# repo, so python is not found, it exits 127, and mpiexec tears down every rank
# in the job. Job 12474718 died that way at 0 training steps, killed by
# x1922c7s2b0n0 -- which PBS called healthy. No pbsnodes query finds these.
#
# USAGE
#   ./survey_nodes.sh            # survey all currently-free nodes
#   ./survey_nodes.sh 40         # cap the number of candidates
#
# Writes .cache/patches/verified_good.txt, one hostname per line, and prints
# the failures. Feed that file to build a `#PBS -l select=` host list.
#
# SURVEY IN SMALL BATCHES. A single large survey job requeues on the very
# bad-node lottery it exists to map (a 70-node attempt reached run_count=3).
# Ten-node batches schedule around it. The tradeoff: a batch containing a bad
# node dies whole, so its other nodes stay unclassified. Re-run to pick those
# up, or accept partial coverage -- 47 verified nodes is enough for a 32N job.
#
# THE RESULT IS A SNAPSHOT. Nodes get taken by other jobs and mounts come and
# go. Re-survey before each submit.
set -o pipefail

REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
PROBE="$REPO/80b_capture.pbs"     # any file that must be readable on the node
OUT="$REPO/.cache/patches"
LIMIT="${1:-9999}"
BATCH=10

mkdir -p "$OUT"
cd "$REPO" || exit 1
export PATH=/opt/pbs/bin:$PATH

pbsnodes -avSj 2>/dev/null | tail -n +3 | awk '$2 ~ /free/ {print $1}' \
  | grep -E '^x19' | head -"$LIMIT" > "$OUT/candidates.txt"
N=$(wc -l < "$OUT/candidates.txt")
echo "candidates: $N"
[ "$N" -eq 0 ] && { echo "no free nodes"; exit 1; }

# Job scripts go in a SEPARATE directory. Writing them beside the batch files
# makes the glob below pick up `sv_batch_00.pbs` as if it were a node list --
# which yields a nonsense 3-node job named `sv_sv_batch_00` and silently drops
# real candidates.
JOBS="$OUT/svjobs"
rm -rf "$JOBS"; mkdir -p "$JOBS"
# Stale .o files from an earlier survey would be read as current results.
rm -f "$REPO"/sv_batch_*.o*
rm -f "$OUT"/sv_batch_* "$OUT"/sv_*.pbs
split -l "$BATCH" -d "$OUT/candidates.txt" "$OUT/sv_batch_"

ids=""
for b in "$OUT"/sv_batch_*; do
  tag=$(basename "$b")
  n=$(wc -l < "$b")
  sel=$(awk '{printf "1:ncpus=208:host=%s+", $1}' "$b" | sed 's/+$//')
  {
    echo "#!/bin/bash -l"
    echo "#PBS -A datascience"
    echo "#PBS -q workq"
    echo "#PBS -l select=$sel"
    echo "#PBS -l walltime=00:04:00"
    echo "#PBS -l filesystems=tegu:home"
    echo "#PBS -N $tag"
    echo "#PBS -j oe"
    echo "cd /tmp"
    echo "mpiexec --envall --np $n --ppn 1 -- /bin/sh -c \"/usr/bin/stat -c OK $PROBE >/dev/null 2>&1 && echo GOOD:\\\$(hostname -s) || echo BAD:\\\$(hostname -s)\""
  } > "$JOBS/$tag.pbs"
  id=$(qsub "$JOBS/$tag.pbs" 2>&1 | head -1 | cut -d. -f1)
  ids="$ids $id"
  echo "  $tag -> $id"
done

echo "waiting on:$ids"
for _ in $(seq 1 40); do
  live=0
  for id in $ids; do
    [ -n "$(qstat -f "$id" 2>/dev/null | grep -oE 'job_state = .')" ] && live=$((live+1))
  done
  [ "$live" -eq 0 ] && break
  sleep 30
done

# PBS writes .o files to the SUBMISSION directory, which is $REPO (we cd'd
# there at the top), not $OUT. Be explicit so this still works if the caller
# invokes the script from somewhere else.
OFILES=$(ls -1 "$REPO"/sv_batch_*.o* 2>/dev/null)
if [ -z "$OFILES" ]; then
  echo "no survey output found in $REPO -- every batch failed to run."
  echo "That is itself the finding: the free pool is bad enough that even a"
  echo "$BATCH-node job cannot land. Try a smaller BATCH, or wait."
  exit 1
fi
# shellcheck disable=SC2086
cat $OFILES 2>/dev/null | grep '^GOOD:' | sed 's/^GOOD://' | sort -u \
  > "$OUT/verified_good.txt"
echo "verified good: $(wc -l < "$OUT/verified_good.txt") / $N"
echo "failed the access test (PBS still calls these free):"
cat $OFILES 2>/dev/null | grep '^BAD:' | sed 's/^BAD:/  /' | sort -u
unclassified=$((N - $(wc -l < "$OUT/verified_good.txt") - $(cat $OFILES 2>/dev/null | grep -c '^BAD:')))
[ "$unclassified" -gt 0 ] && echo "unclassified (their batch never completed): $unclassified"
echo "list: $OUT/verified_good.txt"
