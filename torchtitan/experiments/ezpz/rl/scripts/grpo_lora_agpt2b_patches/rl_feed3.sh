#!/bin/bash
# Feeder for the 3-run comparison. One remote pass aggregates v4/v5/v6 reward by
# policy_version, writes a combined local TSV: run  ver  n  mean  max  ge05
SOCK=/tmp/sunspot-master.sock
TSV="${1:-/tmp/rl_reward3.tsv}"
TMP="${TSV}.tmp"
BASE=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs
printf "run\tver\tn\tmean\tmax\tge05\n" > "$TMP"
ssh -o ControlPath="$SOCK" sunspot "python3 - <<'PYEOF'
import json, collections, statistics, os
BASE='$BASE'
for tag in ('v4','v5','v6'):
    path=os.path.join(BASE,'rl_lora_agpt2b_train_%s'%tag,'rollout_samples.jsonl')
    if not os.path.exists(path): continue
    byv=collections.defaultdict(list)
    for line in open(path):
        try: d=json.loads(line)
        except: continue
        if d.get('is_validation'): continue
        t=d.get('turns') or []
        if not t: continue
        v=t[0].get('max_policy_version')
        if v is None: continue
        try: byv[int(v)].append(float(d.get('reward',0)))
        except: pass
    for v in sorted(byv):
        r=byv[v]
        print('%s\t%d\t%d\t%.4f\t%.4f\t%.4f'%(tag,v,len(r),statistics.mean(r),max(r),
              sum(1 for x in r if x>=0.5)/len(r)))
PYEOF" 2>/dev/null >> "$TMP"
mv "$TMP" "$TSV"
