#!/bin/bash
# healthy_nodes.sh -- list allocation nodes that can actually see $REPO.
#
# Sourced by PBS scripts; sets HEALTHY_FILE and NGOOD.
#
# WHY THIS EXISTS: Sunspot node x1921c4s3b0n0 is missing a project mount
# (mounts=2 vs 3). Any rank landing there exits 127 with "Couldn't change
# directory", and because one bad rank aborts the whole MPI job, a single node
# takes down an N-node allocation in ~1 second.
#
# TWO BUGS THIS ENCODES, both learned expensively:
#
#  1. Run the probe from /tmp, NEVER from $REPO. mpiexec propagates the launch
#     cwd; on a node where $REPO is invisible the chdir fails, that rank exits
#     127, and MPI tears down the entire probe -- so a probe launched from $REPO
#     reports ZERO healthy even when most nodes are fine (job 12472562).
#
#  2. Each rank writes its OWN file; do not race N ranks on one stdout pipe.
#     The shared-pipe version reported 9/62 healthy in job 12472816 with the
#     survivors scattered across five racks -- a pattern that looks like lost
#     output, not clustered hardware failure. File-per-rank has no such failure
#     mode, and it also distinguishes "reported FAIL" from "never reported",
#     which a pipe cannot.
set -o pipefail

_hn_repo="${1:?healthy_nodes.sh: need REPO path}"
_hn_tag="${2:-$PBS_JOBID}"
_hn_dir="$_hn_repo/tmp/healthy.$_hn_tag"

rm -rf "$_hn_dir"; mkdir -p "$_hn_dir"
_hn_nalloc=$(sort -u "$PBS_NODEFILE" | wc -l)

( cd /tmp && mpiexec -n "$_hn_nalloc" -ppn 1 bash -c \
    'h=$(hostname | cut -d. -f1)
     if ls -d '"$_hn_repo"' >/dev/null 2>&1; then echo OK; else echo FAIL; fi > '"$_hn_dir"'/$h' \
) >/dev/null 2>&1

HEALTHY_FILE="$_hn_repo/tmp/healthy_nodes.$_hn_tag"
grep -l OK "$_hn_dir"/* 2>/dev/null | xargs -r -n1 basename > "$HEALTHY_FILE"
NGOOD=$(wc -l < "$HEALTHY_FILE")

_hn_reported=$(ls -1 "$_hn_dir" 2>/dev/null | wc -l)
_hn_failed=$(grep -l FAIL "$_hn_dir"/* 2>/dev/null | wc -l)
_hn_silent=$(( _hn_nalloc - _hn_reported ))

echo "[healthy_nodes] allocated=$_hn_nalloc healthy=$NGOOD failed=$_hn_failed silent=$_hn_silent"
if [ "$_hn_failed" -gt 0 ]; then
  echo "[healthy_nodes] FAILED (no \$REPO): $(grep -l FAIL "$_hn_dir"/* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
fi
if [ "$_hn_silent" -gt 0 ]; then
  # Silent ranks are NOT the same as unhealthy ones -- they may just be slow.
  # Say so rather than quietly treating them as bad.
  echo "[healthy_nodes] WARNING: $_hn_silent node(s) never reported; they are excluded but may be healthy."
fi
export HEALTHY_FILE NGOOD
