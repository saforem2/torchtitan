# ALCF ticket: job 8784460 has not scheduled in 101 hours

> Drafted 2026-08-30 18:30 UTC. **NOT SENT** -- for review.
>
> Supersedes the 2026-08-29 draft, which led with a transient
> "Node is in an ineligible state: down" comment. That message has since
> reverted and is not the issue. The case below does not depend on it.

## Ask

Why is `8784460` not being scheduled, and can its score boost be restored?

## The job

| field | value |
|---|---|
| job id | `8784460` |
| project | `AuroraGPT` |
| queue | `large` |
| nodes | 2098 |
| walltime | 12:00:00 |
| queued | Wed Aug 26 13:22 UTC |
| eligible_time | **101:06:16** |
| job_state | `Q` |
| comment | `Not Running: Not enough free nodes available` |

## Why "not enough free nodes" does not explain it

At 18:23 UTC today there were **2,206 free nodes** against a 2,098-node
request. The job fit and did not start. Five minutes later free had fallen to
166, absorbed by other jobs.

That is the pattern rather than a single unlucky sample. **No job of 2,000+
nodes is running on the machine right now** -- the largest is 516 nodes, and
the rest of the top six are 512, 512, 260, 256, 256. Capacity is being consumed
continuously by small jobs, so a 2,098-node request never accumulates the
contiguous block it needs, whatever its eligible time.

## A larger job started ahead of it

On 2026-08-29, job `8791192` (2,304 nodes, another project) started with:

| | `8784460` (ours) | `8791192` |
|---|---|---|
| nodes | 2098 | **2304** |
| eligible_time | **79:34:34** | 00:22:34 |
| score_boost | 0 | 0 |
| Priority | 0 | 0 |
| result | still `Q` | **`R`** |

A *larger* request, with the same boost and 1/200th the eligible time, was
scheduled first. Fair-share may order equal jobs; it is not obvious how it
produces this.

Separately, around 20:22 UTC on 08-29 roughly 3,600 nodes were released at once
(per `last_state_change_time`, free went 1,704 -> 5,292). `8784460` did not
start in the following 30 minutes.

## Ruled out locally

Checked before filing, so these do not need investigating:

- **Reservations.** `R8782410` is 128 nodes, `M8789605` is 6. The
  full-machine maintenance `M8787441` (10,624 nodes) starts Mon Aug 31 14:00 --
  19.6 hours out at the time of writing, against a 12-hour walltime, so the job
  fits before it.
- **Allocation.** `AuroraGPT` has **+1,269,842.6 node-hours available**
  (allocation 15502). The negative balances under `datascience` belong to a
  different project; `8784460` charges `AuroraGPT`.
- **Queue limits.** `large` allows `max_queued = 10` per project and we have 2.
  2,098 sits inside `resources_min.nodect = 2000` /
  `resources_max.nodect = 10624`, and 12h is inside the 24h maximum.
- **Holds and dependencies.** `Hold_Types = n`. The
  `depend = beforeany:8784462` is our own chain successor, which is how these
  runs are always submitted.
- **Node pinning.** `Resource_List.select = 2098` with no host list.

## Context

The previous incarnation of this chain pair carried a score boost of roughly
10M and started within a day of submission. This pair lost it on resubmit, and
`score_boost` currently reads 0. If the boost was tied to the old job ids
rather than the project, restoring it is likely the whole fix.

This is AuroraGPT production pre-training. The chain last trained on 2026-08-26
(job `8773440`, 5h13m of a 12h slot) and has not run since.

## Cluster state at the time of writing

```
Sun 30 Aug 2026 18:28 UTC
free           166   (2,206 five minutes earlier)
job-exclusive 8138
largest running job: 516 nodes
jobs >= 2000 nodes running: 0
```
