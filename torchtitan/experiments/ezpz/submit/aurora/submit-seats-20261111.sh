#!/usr/bin/env bash
# Submit each agpt seat as its own PBS job (seat-autoretry-20261111.pbs), so a
# node failure kills only that seat. Run from an Aurora login node.
#
#   bash submit-seats-20261111.sh            # dry run: validate every seat, submit nothing
#   SUBMIT=1 bash submit-seats-20261111.sh   # validate, then qsub each seat
#   SEATS="0 3" SUBMIT=1 bash ...            # subset
#
# Resume points come from each seat's checkpoint folder (latest step-*):
#   0 20B-512 step-10800   1 20B-256 step-21100   3 4B step-2600   4 10B fresh
set -euo pipefail
export PATH=/opt/pbs/bin:$PATH
REPO=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
PBS_SCRIPT=$REPO/torchtitan/experiments/ezpz/submit/aurora/seat-autoretry-20261111.pbs
SEATS="${SEATS:-0 1 3 4}"
SUBMIT="${SUBMIT:-0}"
WALLTIME="${WALLTIME:-12:00:00}"
STAGGER="${STAGGER:-300}"   # seconds between qsubs (HF / Lustre startup load)

declare -A NAME=([0]=agpt-20b-n512 [1]=agpt-20b-n256 [3]=agpt-v2-4b-n512 [4]=agpt-v2-10b-n512)
declare -A SELECT=([0]=520 [1]=260 [3]=522 [4]=522)

cd "$REPO"
[[ -f "$PBS_SCRIPT" ]] || { echo "FATAL: missing $PBS_SCRIPT"; exit 1; }

# Dry-run every requested seat against a fake nodefile of the right size.
tmp=$(mktemp -d)
for s in $SEATS; do
  [[ -n "${SELECT[$s]:-}" ]] || { echo "FATAL: unknown seat $s"; exit 2; }
  seq -f "fake%04g" 1 "${SELECT[$s]}" > "$tmp/nodes.$s"
  echo "=== dry run seat $s (${NAME[$s]}, select=${SELECT[$s]})"
  if ! MULTI_ONLY=$s DRY_RUN=1 PBS_NODEFILE="$tmp/nodes.$s" PBS_JOBID="dry$s" PBS_O_WORKDIR="$tmp" \
      bash --login "$PBS_SCRIPT" > "$tmp/dry.$s.log" 2>&1; then
    cat "$tmp/dry.$s.log"; echo "FATAL: dry run failed for seat $s"; exit 1
  fi
  grep -E 'trainer [0-9]:|v2 seat|Dry run OK|Nodes:' "$tmp/dry.$s.log"
done

if [[ "$SUBMIT" != "1" ]]; then
  echo "Dry runs OK. Set SUBMIT=1 to qsub."; exit 0
fi

first=1
for s in $SEATS; do
  (( first )) || sleep "$STAGGER"
  first=0
  jid=$(qsub -q prod -A AuroraGPT -l select="${SELECT[$s]}" -l walltime="$WALLTIME" \
    -l filesystems=home:flare -N "${NAME[$s]}" -o "$REPO/logs/" \
    -v "MULTI_ONLY=$s" "$PBS_SCRIPT")
  echo "seat $s ${NAME[$s]} select=${SELECT[$s]} -> $jid"
done
