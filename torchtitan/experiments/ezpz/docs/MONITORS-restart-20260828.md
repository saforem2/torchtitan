# Live monitors + state at restart -- 2026-08-28 16:20 UTC

Session monitors do NOT survive a restart. This is what was armed, what
each watched, and the exact commands to re-arm them.

## Live job state at snapshot

| Job | State | Detail |
|---|---|---|
| `7567541` `agpt20b-p1` | **R** since 16:12:10 UTC | 132 nodes, 12 h, `-A AuroraGPT`, resumes step-5600 |
| `7567542` `agpt20b-p2` | **H** `afterany:7567541` | continuation |

**Leg 1 is hung on attempt 1.** `x3111c0s37b1n0` (rank 332) failed with
`cudaErrorDevicesUnavailable` -- the third consecutive job of ours to
land on that zombie node. The failed rank does not exit, so the
collective hangs and only `IDLE_TIMEOUT=3600` will break it. Expect the
retry roughly one hour after 16:16 UTC.

## The one thing to check when the retry fires

`logs/failover-7567541/bad_nodes.txt`, column 2:

- **`scraped`** -> the watchdog fix (commit `8592d2096`) engaged. It
  swapped exactly `x3111c0s37b1n0` for a spare. This is the first live
  trial of that fix; it has never run green on hardware.
- **`blind`** -> the fix did NOT engage. It evicted an innocent node.
  Regression, needs investigation.

4 spares available (132 alloc - 128 train) for a 1-node swap.

## Monitors that were running

### 1. `bkw05n652` -- leg 7567541 retest + resume (PRIMARY, re-arm this)

```bash
C=/lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/torchtitan
F=$C/logs/failover-7567541
prev=""
while true; do
  cur=$(timeout 150 ssh -o BatchMode=yes polaris "export PATH=/opt/pbs/bin:\$PATH
st=\$(qstat -f 7567541 2>/dev/null | tr -d '\n\t' | tr ';' '\n' | grep -oE 'job_state = .' | awk '{print \$3}'); [ -z \"\$st\" ] && st=GONE
att=\$(ls -1 '$F'/attempt-*.log 2>/dev/null | wc -l | tr -d ' ')
cuda=\$(grep -ahc 'CUDA-capable device' '$F'/attempt-*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null); [ -z \"\$cuda\" ] && cuda=0
prov=\$(awk '{print \$2}' '$F'/bad_nodes.txt 2>/dev/null | sort -u | tr '\n' ',')
res=\$(grep -ahoE 'Loading the checkpoint from [^ ]+' '$F'/attempt-*.log 2>/dev/null | tail -1 | grep -oE 'step-[0-9]+')
stp=\$(grep -ahoE 'step: *[0-9]+ *loss: *[0-9.]+' '$F'/attempt-*.log 2>/dev/null | tail -1)
echo \"st=\$st att=\$att cuda=\$cuda prov=\${prov:-none} resume=\${res:-none} | \${stp:-no-steps}\"" 2>/dev/null)
  [ -z "$cur" ] && { sleep 150; continue; }
  if [ "$cur" != "$prev" ]; then
    echo "$cur"
    case "$cur" in
      *"resume=step-5600"*) echo "  RESUME OK -- chain continues from 5600";;
      *"resume=step-"*) echo "  WRONG RESUME STEP -- stop it before it checkpoints";;
    esac
    case "$cur" in
      *"prov=scraped"*) echo "  WATCHDOG FIX WORKING -- correct node named, not blind";;
      *"prov=blind"*) echo "  BLIND AGAIN -- watchdog fix did not engage";;
    esac
    case "$cur" in *"st=GONE"*) echo "  LEG ENDED -- check exit status";; esac
    prev="$cur"
  fi
  sleep 150
done
```

Persistent, 150 s poll. Dedupes on the whole state string.

### 2. `bh9zbliix` -- prod clone regression watch (re-arm, lower priority)

Polls the pinned prod clone every 600 s for `head` + both training-code
fixes. Alerts if `guard=0` or `fold=0`, i.e. a concurrent session pulled
the clone and broke a running leg.

```bash
C=/lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/torchtitan
prev=""
while true; do
  cur=$(timeout 90 ssh -o BatchMode=yes polaris "cd '$C' 2>/dev/null || { echo CLONE_MISSING; exit 0; }
h=\$(git rev-parse --short HEAD); g=\$(grep -c '_last_grad_norm' torchtitan/experiments/ezpz/trainer.py)
f=\$(grep -c '_EZPZ_MAX_CONTEXT_LENGTH' torchtitan/experiments/ezpz/agpt/__init__.py)
echo \"head=\$h guard=\$g fold=\$f\"" 2>/dev/null)
  if [ -n "$cur" ] && [ "$cur" != "$prev" ]; then
    echo "prod clone changed: $cur"
    case "$cur" in
      *"guard=0"*) echo "  REGRESSION: grad-norm guard gone";;
      *"fold=0"*)  echo "  REGRESSION: fold fix gone";;
    esac
    prev="$cur"
  fi
  sleep 600
done
```

Expected steady state: `head=1bb11aac5 guard=2 fold=4`.

### Stopped before restart (do NOT re-arm)

- `bexybj1e9`, `bqt6ed8p5` -- watched dead jobs 7560196/7560197
- `bmw630vmy` -- superseded by `bkw05n652` (duplicate coverage)
- `bepnud02o`, `belmill7d`, `b8spfx1zs` -- tarball rebuild watches; the
  rebuild finished and was verified

## Verified staging (all green at snapshot)

| Item | State |
|---|---|
| `step-5600` | complete, 56 ckpts, `.metadata` present |
| Prod clone | `1bb11aac5` pinned; guard=2, fold=4 |
| Lustre venv | fqdn=1, watchdog=1 |
| Broadcast tarball | fqdn=1, watchdog=1 (gzip verified, 37,439 entries) |
| `/tmp/.venv` on compute | fqdn=1, watchdog=1 (probed `x3006c0s13b1n0`) |

## Open, needs a human

**ALCF ticket, drafted and unsent:**
[`ops/alcf-ticket-zombie-gpu-nodes-20260827.md`](ops/alcf-ticket-zombie-gpu-nodes-20260827.md).
`x3111c0s37b1n0` and `x3007c0s13b1n0` each have one GPU that refuses
`cudaSetDevice` while PBS still advertises `ngpus=4`. Failover routes
around them per-job; only a drain removes them from the pool.
