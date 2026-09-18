#!/bin/bash
# alcf-job-preflight: run this BEFORE qsub/sbatch. Exits non-zero if anything
# the job needs is missing, so a bad submission never reaches the queue.
#
#   usage: preflight.sh <worktree> <venv> [machine]
#
# Catches the failure classes that cost nine jobs in one session: node-local
# /tmp, missing ZE_FLAT_DEVICE_HIERARCHY, stale worktree, absent tokenizer or
# data-list, and API signatures that changed under us.
set -o pipefail
W="${1:?usage: preflight.sh <worktree> <venv> [machine]}"
V="${2:?usage: preflight.sh <worktree> <venv> [machine]}"
M="${3:-$(hostname -s | sed 's/[0-9-].*//')}"
fail=0
ok()   { printf "  OK    %s\n" "$1"; }
bad()  { printf "  BAD   %s\n" "$1"; fail=1; }

cd "$W" 2>/dev/null || { bad "worktree $W does not exist"; exit 1; }

# --- worktree currency: a pinned worktree misses files recent commits moved ---
if git rev-parse --git-dir >/dev/null 2>&1; then
  h=$(git rev-parse --short HEAD 2>/dev/null)
  u=$(git rev-parse --short '@{u}' 2>/dev/null || echo "$h")
  [ "$h" = "$u" ] && ok "worktree current ($h)" || bad "worktree $h != upstream $u -- fetch/checkout first"
fi

# --- assets ---
[ -s assets/hf/gemma-7b/tokenizer.json ] && ok "tokenizer" || bad "tokenizer assets/hf/gemma-7b"
dl="torchtitan/experiments/ezpz/data-lists"
if [ -d "$dl" ]; then
  found=$(ls -1 "$dl" 2>/dev/null | tr '\n' ' ')
  case "$found" in *"$M"*) ok "data-list for $M";; *) bad "no data-list for '$M' (have: $found)";; esac
fi

# --- python deps, in the venv the JOB will use ---
for m in torch mpi4py ezpz blendcorpus; do
  "$V/bin/python" -c "import $m" 2>/dev/null && ok "py:$m" || bad "py:$m not importable in $V"
done

# --- torch floor: pytorch #181519, by SYMBOL not prose ---
f=$(ls "$V"/lib/python3*/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py 2>/dev/null | head -1)
if [ -n "$f" ]; then
  n=$(grep -c _resolve_spmd_types_for_storage "$f")
  [ "$n" -gt 0 ] && ok "#181519 present ($(wc -l < "$f") lines)" \
                 || bad "#181519 ABSENT -- this torch dies at FSDP wrapping"
fi

# --- XPU geometry: the one that keeps biting ---
case "$M" in
  aurora*|sunspot*|x[0-9]*)
    if [ "${ZE_FLAT_DEVICE_HIERARCHY:-}" = "FLAT" ]; then ok "ZE_FLAT_DEVICE_HIERARCHY=FLAT"
    else bad "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-unset} -- need FLAT for 12 ranks/node"; fi;;
esac

# --- nothing the job reads may live on node-local tmpfs ---
for p in "$@"; do
  case "$p" in /tmp/*) bad "$p is on node-local tmpfs -- invisible to compute nodes";; esac
done

[ $fail -eq 0 ] && echo "  ---> PREFLIGHT PASS" || echo "  ---> PREFLIGHT FAIL, do not submit"
exit $fail
