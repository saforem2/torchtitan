# Production Queue-Wait Analysis (Aurora `small`)

> Snapshot: 2026-06-24. Queue wait — not training throughput — is the
> dominant cost on the 512N canonical chains right now. This page
> records the data + the scheduler diagnosis so we stop re-deriving it.

## TL;DR

- **512N is queue-starved.** Both 512N canonical chains (2B + 20B) have
  sat in `small` for **~20 days** (since 2026-06-04) without a slot.
- **256N cycles in hours-to-days.** 256N (260-node) jobs land far more
  often; the worst 256N wait we recorded was ~9 days, most are <1 day.
- **Root cause is pure contention, confirmed by PBS** — not a hold, not
  a bad dependency, not an unsatisfiable request. The 522-node ask
  rarely fits in `small` alongside everyone else.
- **Do NOT re-submit the stuck 512N jobs.** They've accrued ~18 days of
  scheduling priority (`eligible_time`); a fresh submit resets that to
  zero and goes to the back of the queue. Re-submitting makes it worse.

## Live data (from PBS `qtime`, 2026-06-24)

| Job | Trajectory | Nodes | Queued since | Wait so far | PBS comment |
|-----|------------|------:|--------------|------------:|-------------|
| 8521631 | 2B 512N cont10 | 522 | 2026-06-04 11:03 | **~20 d** | Not enough free nodes available |
| 8521632 | 20B 512N cont11 | 522 | 2026-06-04 11:24 | **~20 d** | Not enough free nodes available |
| 8558531 | 2B 256N cont12 | 260 | 2026-06-24 07:19 | <1 d | (cycling) |
| 8558548 | 20B 256N | 260 | 2026-06-24 07:45 | <1 d | (cycling) |

`eligible_time` at snapshot: 8521631 = 430 h, 8521632 = 375 h — i.e.
these jobs have been accruing priority for ~16-18 days. That priority
is the asset that will eventually win them a slot; it is destroyed by
re-submission.

The `H`-held continuations (8534294/95, 8558532, 8558549) are NOT
waiting on the queue — they are `afterany`-held behind their
predecessor and only become eligible once it finishes.

## Why 512N specifically

`qstat -Qf small` at snapshot: `resources_assigned.nodect = 4792`
(nodes already committed to other running jobs). A 522-node *scatter*
placement needs 522 free hosts simultaneously; with ~4.8k nodes
committed elsewhere that window is rare. 256N (260 nodes) fits into
gaps far more often, which is exactly why the 256N chains keep
advancing while 512N stalls.

512N *is* schedulable — earlier 512N dispatches did run (e.g. 8505258,
8509393 on the 20B chain) — it's just infrequent.

## Is it worth re-submitting fresh 512N jobs? — NO

Checked explicitly because it's the tempting move. PBS shows:

- `Hold_Types = n` (no hold), `depend = beforeany:<cont>` only (the
  job is the *predecessor*, not blocked by anything).
- `comment = Not Running: Not enough free nodes available`.

So nothing is wrong with the jobs; they are simply waiting for 522
free nodes. A re-submit:

1. Starts at `eligible_time = 0`, behind the current jobs that have
   ~18 days accrued.
2. Breaks the `afterany` continuation chain (the held conts reference
   the current jobids).

Re-submitting would *lengthen* the wait, not shorten it.

## Real levers (if 512N progress becomes urgent)

1. **Project reservation** — a `pbs_rstat`/reservation for a 512N+
   block is the only reliable way to guarantee the nodes. Worth raising
   at the AuroraGPT sync if 512N throughput matters near-term.
2. **Consolidate on 256N** — 256N gets slots; per-token it is the
   better comparator anyway (the 256N chains carry the bulk of the
   token progress). The 512N chains exist for the canonical large-batch
   trajectory, but if wall-clock progress is the priority, 256N
   delivers it.
3. **Smaller 512N walltime** — shorter jobs backfill into gaps more
   easily, at the cost of more frequent restarts (and more
   `set_determinism` init-crash exposure on 20B).

## Ongoing data collection

`failover_lib.sh::failover_log_queue_wait` (added 2026-06-24) appends
every dispatch's actual queue wait to `logs/queue_wait.csv` at job
start (qtime -> stime), so we have a durable record after jobs age out
of `qstat`. Columns:
`jobid,jobname,nodes,queue,qtime,stime,wait_seconds,wait_hours`.
Aggregate it over time to track whether the 512N starvation is getting
better or worse.
