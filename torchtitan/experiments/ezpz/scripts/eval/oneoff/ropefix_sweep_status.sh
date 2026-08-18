#!/bin/bash
# Is the RoPE re-eval sweep actually finished?
#
# The right test is PER ARM: how many post-switch checkpoints exist in the
# original eval dir, versus how many the -ropefix dir has. Pre-switch steps
# were exported with the matching convention and are correctly NOT re-run, so
# "corrected == original" is the wrong bar and "total >= 74" is worse.
#
# A raw count threshold gave a false COMPLETE on 2026-08-17: 32+34+8 crossed
# 74 while both 20B arms were only half done. 74 was the count of targeted
# checkpoints, not a finish line.
#
# Switch points: 20b_v2_512 step 4401, 20b_v2_256 step 3101,
# 2b_v2_512 step 30401 (see guides/known-bugs/rope-flavor-mismatch.md).
R=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
printf "%-14s %-8s %-6s %-6s %s\n" arm switch need have status
while read -r a sw; do
  [ -z "$a" ] && continue
  # Count post-switch steps that actually HAVE a results.json, not step dirs.
  # A dir can exist with no eval in it (20b-256 step-5200 is one: healthy
  # 3072-shard checkpoint, never evaluated), and counting dirs reported the
  # sweep one step short forever.
  need=0
  for st in $(ls "$R/outputs/evals/agpt-$a" 2>/dev/null | grep '^step-' | sed 's/step-//' | awk -v s="$sw" '$1>=s'); do
    [ -f "$R/outputs/evals/agpt-$a/step-$st/results/results.json" ] && need=$((need+1))
  done
  have=$(ls "$R/outputs/evals/agpt-$a-ropefix" 2>/dev/null | grep -c '^step-')
  if [ "$have" -ge "$need" ]; then st=COMPLETE; else st="pending ($((need-have)) left)"; fi
  printf "%-14s %-8s %-6s %-6s %s\n" "$a" "$sw" "$need" "$have" "$st"
done <<ARMS
20b-v2-512n 4401
20b-v2-256n 3101
2b-v2-512n 30401
ARMS
