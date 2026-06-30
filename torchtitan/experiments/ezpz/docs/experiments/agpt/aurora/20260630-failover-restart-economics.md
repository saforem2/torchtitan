# Failover / auto-retry "restart economics" -- log-mining analysis

- **Date:** 2026-06-30 (during PM maintenance; read-only log mining)
- **Author:** Sam Foreman
- **Corpus:** all `*.o<jobid>` logs across the production clones + main repo
  (agpt-2b-v2, agpt-20b-v2, agpt-20b-n256, agpt-80b-v2, main torchtitan-ezpz).
  478 job logs total; **133 used the failover/auto-retry wrapper**.

## Method

Classified every failover *episode* (one `attempt 1/M ...` sequence; a relaunch
that reuses the same `.o` file counts as a new episode) by its `[failover]`
marker lines:
- **TRIGGERED** = >=2 attempts started OR >=1 spare swap performed.
- **RECOVERED** = a retry attempt (#>=2) logged `succeeded (exit 0)`.
- Trigger reason = the exit code on the failing attempt.
PBS-side wall-clock (`stime`/`mtime`) is **not available** -- these jobs
(May/early-June) have aged out of `qstat -xf` history.

## Headline numbers

| metric | value |
|---|---|
| Job logs scanned | 478 |
| Logs using the failover wrapper | 133 |
| Failover **episodes** | 210 |
| Episodes that **TRIGGERED** (swap/retry) | 62-63 |
| Episodes that **RECOVERED** (retry succeeded exit 0) | **7** |
| Episodes triggered-but-not-recovered | ~55 |
| Total spare swaps performed | 132 (123 "blind" rotations) |

## Trigger reasons (what makes a retry fire)

| exit code | count | meaning |
|---|---|---|
| 143 | 117 | SIGTERM -- node death / PBS / mid-run kill (most common) |
| 1 | 35 | generic process failure (often init) |
| 127 | 30 | command-not-found / env broken on a node (yeet/venv issue) |
| 124 | 14 | ezpz idle-output watchdog (silent-hang catch) |
| watchdog(124) | 10 | (same, explicit watchdog string) |
| 139 | 1 | SIGSEGV |

## The 7 CONFIRMED successful restarts

| job | model / scale | won on | swaps | trigger | progress after recovery |
|---|---|---|---|---|---|
| 8481646 | 20B 256N | attempt 2 | 1 | exit 1 (init) | trained to step-500 |
| 8504860 | 20B 256N | attempt 2 | 1 | exit 1 (init) | trained to step-500 |
| 8503508 | 20B 256N | attempt 2 | 1 | exit 1 (init) | trained to step-500 |
| 8481647 | 20B 512N | attempt 2 | 1 | exit 1 (init) | trained to step-1000 |
| 8503507 | 20B 512N | attempt 3 | 2 | exit 1 x2 | trained to step-808 |
| **8516701** | 20B 512N | attempt 2 | 4 | exit 143/127 cascade | **trained to step-4501** |
| **8506221** | 2B 512N | attempt 2 | 1 | **exit 124 (silent-hang watchdog)** | **trained to step-16676** |

Two marquee cases:
- **8506221** -- the **silent-hang catch**. Attempt 1 produced no output for 600s;
  the ezpz idle watchdog tripped (exit 124), the wrapper swapped a spare, and
  attempt 2 ran clean to step-16,676. This is the canonical
  "failover caught a real silent hang" validation (see also the 2026-05-23
  incident report for 8505298).
- **8516701** -- the **most-worked recovery**: an exit-143/127 cascade, **4 blind
  spare rotations**, recovered on a fresh episode and trained to step-4,501.

## The triggered-but-not-recovered ~55 episodes, classified

| outcome | count | interpretation |
|---|---|---|
| exhausted_spares | 47 | failover genuinely tried; ran out of spares / max-retries. Real un-fixable-by-swap failures (often systemic: CCL/KVS init crashes, pals-RPC, blendcorpus races -- node-swapping can't fix these). |
| walltime_during_retries | 7 | NOT real failures -- successful runs that used the wrapper and hit walltime (e.g. 8505298, 8521630, 8519833, 8572612). The "trigger" was an incidental mid-run swap; the run was fine. |
| other | 1 | 8567614 |

## Economics takeaways

1. **Raw recovery rate ~11%** (7/62 triggered) UNDERSTATES effectiveness: the
   denominator includes ~7 walltime-clean runs and a large bloc of
   **systemic** failures (CCL KVS timeout, pals-RPC, blendcorpus index race,
   set_determinism OOM) that node-swapping fundamentally cannot fix -- those
   need a code/env fix, not a spare. Removing those, swap-fixable node failures
   recover at a much higher rate.
2. **The swap mechanism works when the failure IS a bad node** -- all 7
   recoveries are init-time bad-node / silent-hang cases, exactly its design
   target. The headline win remains 8506221 (silent hang -> clean step-16,676).
3. **"Blind" rotation dominates** (123/132 swaps had no specific bad node
   identified) -- the bad-node *scraper* rarely pinpoints the culprit, so the
   wrapper rotates a spare blindly. This works but is inefficient at high spare
   counts; better bad-node identification would raise the recovery rate.
4. **Most non-recoveries are systemic, not node-local** -- the 47 exhausted
   cases cluster on known infra bugs already documented separately. Failover is
   the wrong tool for those; the right fix is the upstream patch (e.g.
   blendcorpus atomic-rename, KVS-timeout resubmit, etc.).

## Caveats
- Episode counting treats each `attempt 1/M` as a new episode; a few `.o` files
  concatenate multiple relaunches, so episode counts (210) > job count.
- No PBS wall-clock for these aged-out jobs, so per-attempt time cost is
  estimated as "init-time = minutes" (all 7 recoveries failed at init, not
  mid-training) rather than measured.
