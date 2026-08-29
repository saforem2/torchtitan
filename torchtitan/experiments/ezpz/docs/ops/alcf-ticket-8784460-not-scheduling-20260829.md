# ALCF ticket draft: job 8784460 not scheduling after 79h eligible

> Drafted 2026-08-29 20:57 UTC. NOT SENT -- for review first.

## Summary

PBS job `8784460` (queue `large`, 2098 nodes, 12h walltime, project AuroraGPT)
has been queued since 2026-08-26 with **79h34m eligible time** and has not
started, while a **larger** job from another user with **22 minutes** of
eligible time and the same `score_boost = 0` started today.

The scheduler comment on our job changed from "Not enough free nodes available"
to:

```
comment = Not Running: Node is in an ineligible state: down
```

That comment persists while 3,000-5,000 nodes are free, which suggests the job
is being planned onto a node set containing down hardware rather than being
re-planned onto currently-free nodes.

## The comparison

| job | user | nodes | eligible | score_boost | Priority | state |
|---|---|---|---|---|---|---|
| `8784460` | foremans | 2098 | 79:34:34 | 0 | 0 | **Q** |
| `8791192` | (other) | 2304 | 00:22:34 | 0 | 0 | **R** |

A larger job with the same boost and 1/200th the eligible time was scheduled
ahead of ours. Fair-share could account for ordering between equal jobs; it does
not obviously explain a larger job overtaking a three-day-old queue entry.

## Cluster state at the time of writing (20:57 UTC)

```
7078 job-exclusive
3142 free
 338 offline
  26 Stale
  16 down
  13 resv-exclusive
  11 state-unknown
```

Around 20:22 UTC roughly 3,600 nodes were released at once (per
`last_state_change_time`), taking free from 1,704 to 5,292. Our job did not
start in the 30 minutes that followed.

## Ruled out locally

- **Reservations.** `R8782410` is 128 nodes (ends Mon Aug 31 10:55) and
  `M8789605` is 6 nodes. Together 134 -- far too small to block 2098 out of
  3,000+ free.
- **Queue limits.** `large` has `max_queued = 10` per project; we have 2 jobs.
  Our 2098 sits inside `resources_min.nodect = 2000` /
  `resources_max.nodect = 10624`, and 12h is inside the 24h walltime max.
- **Holds / dependencies.** `Hold_Types = n`. The `depend = beforeany:8784462`
  is our own chain successor, which is the normal pattern for these runs and
  does not gate the parent.
- **Node pinning.** `Resource_List.select = 2098` with no host list, so the job
  is not requesting specific nodes.

## Questions for ALCF

1. Why does `8784460` report "Node is in an ineligible state: down" while
   thousands of nodes are free? Is it pinned to a stale placement set?
2. Is there a per-project or fair-share factor putting AuroraGPT behind other
   projects at equal `score_boost`? The previous incarnation of this chain
   carried a ~10M boost and started within a day; this pair lost that on
   resubmit.
3. Can the score boost be restored for `8784460` / `8784462`?

## Local context

The 16 nodes currently `down` include: `x4204c4s0b0n0`, `x4603c5s6b0n0`,
`x4411c0s5b0n0`, `x4005c1s0b0n0`, `x4604c4s0b0n0`, `x4001c1s6b0n0`.
