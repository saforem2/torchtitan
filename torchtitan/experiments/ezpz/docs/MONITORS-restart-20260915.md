# Monitor snapshot -- 2026-09-15 19:10 UTC (ahead of a Claude Code update)

## Live monitors at shutdown

| task | what it watched | status |
|------|-----------------|--------|
| `bdaixcfx2` | Polaris 20B legs 7575650 / 7595518 -- job state, failover verdicts | persistent; dies with the session |

Every earlier monitor (`b1ldc6b1a`, `bpv6w0aaq`, `bppod12fo`, `b4dzwkmsv`,
`bf9yamqbw`, `b06jueiyp`) was already stopped or superseded. Only the one
above needs re-arming, and its job list is now stale -- 7575650 finished, so
re-arm on 7595518 plus whatever successor exists (see below).

## Cluster state at shutdown

- `7575650` (p3): **F**, walltime 12:00:19, `Exit_status=-29` -- clean
  walltime finish. Steps 5901 -> 6263, loss ~1.94.
- `7595518` (p4): **Q**, 132 nodes, `eligible_time = 212h`,
  comment "Not enough free nodes available". No hold. 58 nodes free,
  493 job-exclusive, no reservations -- ordinary queue contention, not a
  PM and not a fault.
- Chain head: **step 6,263** (W&B `thmd4sge`). Checkpoint cadence is 100,
  so the resume point will be step-6200.
- NO successor is queued behind p4. Queue one when p4 starts, or the chain
  stops again.

## Re-arm command

```
BASE=/lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/torchtitan
prev=""
while true; do
  snap=$(ssh -o BatchMode=yes -o ConnectTimeout=20 polaris "export PATH=/opt/pbs/bin:\$PATH
    for j in 7595518:agpt20b-p4; do
      id=\${j%%:*}; nm=\${j##*:}
      st=\$(qstat -x -f \$id 2>/dev/null | awk -F'= ' '/job_state/{print \$2}')
      ex=\$(qstat -x -f \$id 2>/dev/null | awk -F'= ' '/Exit_status/{print \$2}')
      echo \"S \$id=\$st\${ex:+/rc\$ex}\"
      f=\$(ls -1t \$BASE/\$nm.o\$id 2>/dev/null | head -1)
      [ -n \"\$f\" ] && grep -ahoE 'FAILOVER STOP: [a-z_]+|bad nodes: \[[^]]*\]|swapped [0-9]+|invalid device ordinal|CUDA-capable device|system not yet initialized|Traceback|Killed' \"\$f\" 2>/dev/null | tail -4 | sed \"s/^/E \$id /\"
    done" 2>/dev/null)
  if [ -n "$snap" ] && [ "$snap" != "$prev" ]; then echo "$snap" | tr '\n' '|'; prev="$snap"; fi
  sleep 600
done
```

Add the p5 job-id to the `for j in ...` list once it is queued.

## What to check when p4 starts

1. Resume step is **6200** (not 0) and first loss is in the **1.5-2.2** band
   (not ~12.9). Either miss means the checkpoint did not load.
2. Any node fault is attributed `scraped`, not `blind`.
3. `x3111c0s37b1n0` is still in service carrying its stale Aug 22
   `EXECJOB_END` comment -- the PM did not clear it. `x3003c0s25b0n0` was
   taken offline 2026-09-03 for GPU replacement
   (`WHM: GPU 4 UCE: SRAM Uncorrectable - Replace, P1747`).

## Not transferred by an update

- Background monitors (the table above).
- Nothing else: PBS jobs keep running, W&B works from the laptop
  (`~/.netrc` + `.venv/bin/python`), local repo is clean and synced.
