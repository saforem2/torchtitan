#!/bin/bash
# Data feeder: pull the v4 reward stream over the existing SSH socket, aggregate
# by policy_version (~= training step), rewrite a local TSV each pass (small: one
# row per version). Columns: ver n mean max p50 frac_nonzero frac_ge_0.5
SOCK=/tmp/sunspot-master.sock
REMOTE_JSONL="${2:-/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/rl_lora_agpt2b_train_v4/rollout_samples.jsonl}"
TSV="${1:-/tmp/rl_reward.tsv}"
TMP="${TSV}.tmp"
printf "ver\tn\tmean\tmax\tp50\tnonzero\tge05\n" > "$TMP"
ssh -o ControlPath="$SOCK" sunspot "python3 - <<'PYEOF'
import json, collections, statistics
byv = collections.defaultdict(list)
try:
    for line in open('$REMOTE_JSONL'):
        try: d = json.loads(line)
        except: continue
        if d.get('is_validation'): continue
        t = d.get('turns') or []
        if not t: continue
        v = t[0].get('max_policy_version')
        if v is None: continue
        try: byv[int(v)].append(float(d.get('reward', 0)))
        except: pass
except FileNotFoundError:
    raise SystemExit
for v in sorted(byv):
    r = byv[v]
    print('%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
        v, len(r), statistics.mean(r), max(r), statistics.median(r),
        sum(1 for x in r if x > 0)/len(r), sum(1 for x in r if x >= 0.5)/len(r)))
PYEOF" 2>/dev/null >> "$TMP"
mv "$TMP" "$TSV"
