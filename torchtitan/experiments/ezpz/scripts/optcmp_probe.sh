#!/bin/bash
# Per-arm status across the WHOLE CHAIN, not just the newest link.
#
# An arm is N chained jobs with N train.log files. Reading only the newest
# resets every statistic each time a link starts and silently understates
# excursion counts. Concatenate all links and dedupe by step.
T=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
for nm in adamw mano sophiag-fresh; do
  logs=$(ls -1 "$T"/outputs/logs/optcmp-${nm}*/*/train.log 2>/dev/null)
  [ -z "$logs" ] && { echo "$nm (no logs yet)"; continue; }
  P=$(cat $logs 2>/dev/null | sed -e 's/\x1b\[[0-9;]*m//g' \
      | grep -aoE 'step: [0-9]+ +loss: +[0-9.]+ +grad_norm: +[0-9.]+' \
      | awk '{print $2, $4, $6}' | sort -k1,1n -u)
  st=$(printf '%s' "$P" | tail -1)
  # Comparison window: steps already at the loss level the diverging
  # replicates occupied (<5.0). A fixed step cutoff is wrong across arms --
  # a fresh init sits at loss ~12 with grad_norm ~50, ordinary early
  # training, not the onset signature (a gn jump under a FLAT loss).
  read -r tot hi mx <<EOF
$(printf '%s' "$P" | awk 'BEGIN{m=0} $2<5.0{c++; if($3>2.0)h++; if($3>m)m=$3} END{printf "%d %d %.2f", c+0, h+0, m}')
EOF
  pct=$(awk -v h="$hi" -v c="$tot" 'BEGIN{if(c>0) printf "%.1f", 100*h/c; else printf "0.0"}')
  links=$(printf '%s\n' "$logs" | wc -l | tr -d ' ')
  echo "$nm links=$links last=[$st] window=$tot over2=$hi (${pct}%) max=$mx"
done
for f in $(ls -1t "$T"/cmp-*.o* 2>/dev/null | head -6); do
  m=$(sed -e 's/\x1b\[[0-9;]*m//g' "$f" 2>/dev/null | grep -aoE 'rc=[0-9]+|OPTCMP_[a-z-]*_DONE' | tr '\n' ' ')
  [ -n "$m" ] && echo "  $(basename "$f"): $m"
done
