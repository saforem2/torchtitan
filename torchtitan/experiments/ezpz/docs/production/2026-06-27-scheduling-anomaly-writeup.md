# AuroraGPT scheduling anomaly — 2026-06-27

> **Snapshot: 2026-06-27 ~18:40 CDT / 23:40 UTC.** An eligible 264-node
> AuroraGPT (INCITE) job has sat queued 2.6 h while **6,784 nodes are
> idle** and **zero AuroraGPT jobs are running anywhere on the machine**.
> This is not node contention. This page records the exact evidence for an
> ALCF support ticket / PI escalation.

## The job that won't start

`agpt-2b-n256-sneak2h` (a 2h checkpoint-advancing run for the 2B-256N
canonical chain, resumes step-86200):

| Field | Value |
|---|---|
| Job ID | `8572612` |
| Account | `AuroraGPT` (award_type INCITE-2026) |
| Resource_List.nodect | **264** (256 train + 8 spare) |
| Resource_List.walltime | 02:00:00 |
| **queue** | **`prod`** (still in the routing queue — never routed to `small`) |
| job_state | `Q` |
| **Hold_Types** | **`n`** (no hold) |
| qtime | Sat Jun 27 16:03:23 2026 |
| eligible_time | **02:37:28** and counting |
| project_priority | 25 |
| comment | (none) |
| estimated.start_time | (none) |

No hold, no dependency, eligible 2.6 h, a clean 264-node ask.

## Why "contention" does not explain it

Machine node states at snapshot (`pbsnodes -avS`):

| State | Count |
|---|---|
| **free** | **6,784** |
| job-exclusive | 3,684 |
| down | 15 |
| offline | 105 |

**6,784 nodes are idle.** The job needs 264. It fits **25x over** in the
free pool. And **AuroraGPT has zero running jobs** anywhere on the machine
(verified by parsing every running job's `Account_Name`).

## Jobs that "jumped" AuroraGPT

All **6 multi-node jobs currently running** (>=64 nodes) belong to other
projects; together they hold only ~2,884 of the ~3,684 busy nodes, leaving
the 6,784 idle:

| Project | Jobs | Nodes |
|---|---|---|
| Aurora_testing | 1 | 1024 |
| FusAblator | 1 | 792 |
| GNNMD | 2 | 512 |
| EE-ECP | 1 | 300 |
| CombustTurbine | 1 | 256 |
| **AuroraGPT** | **0** | **0** |

**The decisive comparison:** another project's job `nougat-prod`, **the
same 256-node size**, queued in the same window, **DID route into `small`**
(jobs `8572444`, `8571630` show `queue = small`), while our identically
sized `8572612` remains in `queue = prod` — i.e. it was **never routed out
of the `prod` router into `small`** at all. Same node count, same time,
different outcome by project.

## All AuroraGPT jobs queued at snapshot (nothing running)

| Job | Name | Nodes | Queue |
|---|---|---|---|
| 8521631 | agpt-2b-n512 cont10 | 522 | small |
| 8521632 | agpt-20b-n512 cont11 | 522 | small |
| 8558531 | agpt-2b-n256 cont12 | 260 | small |
| 8558548 | agpt-20b-n256 | 260 | small |
| 8568429 | agpt-multi-failover (umbrella) | 1576 | medium |
| 8572612 | agpt-2b-n256-sneak2h | 264 | prod (unrouted) |

The 512N chains have been eligible since **2026-06-07** (~20 days);
`8568429` (umbrella) has **27 h** of accrued eligible_time. None has been
selected.

## What has been ruled out (all checked 2026-06-26/27)

- **Node scarcity** — 6,784 idle, job needs 264.
- **A hold** — `Hold_Types = n` on every AuroraGPT job.
- **A bad dependency** — the sneak is standalone; the chains' `afterany`
  deps are satisfied (live `depend` fields are clear, jobs are `Q` not `H`).
- **Eligibility/age** — a fresh 0-eligibility submit hit the same wall as a
  56h-accrued canonical job (test 2026-06-26).
- **Walltime** — 2h jobs are blocked the same as 12h jobs now.
- **Allocation** — burn_ratio 0.29 (only ~29% of INCITE-2026 used).
- **Job config** — identical scripts ran successfully earlier (5 sneaks
  landed 2026-06-26).
- **PM reservation** — the Jun-29 06:00 PM is >30 h out; not draining yet.

## What this points to (for ALCF)

An eligible, unheld, well-sized INCITE job not routing out of `prod` into
`small` while 6,784 nodes sit idle and the project has zero running jobs is
consistent with a **project- or queue-level scheduling/priority/routing
issue specific to AuroraGPT**, not normal fair-share contention. The
exact mechanism (a per-project run limit, a routing/priority
misconfiguration, an allocation flag) needs scheduler-admin visibility
(`qmgr`/scheduler logs) that is not available from `qstat`/`pbsnodes`.

**Suggested ask to ALCF:** "Why is AuroraGPT job `8572612` (264 nodes,
2h, eligible 2.6 h, Hold_Types=n) not being routed from `prod` into
`small` and started, when 6,784 nodes are free, AuroraGPT has 0 running
jobs, and same-sized jobs from other projects are routing into `small` in
the same window?"

## Reproduce the snapshot

```bash
qstat -f 8572612 | grep -iE 'job_state|queue =|nodect|eligible_time|Hold_Types|comment'
pbsnodes -avS | awk 'NR>2{s[$2]++} END{for(k in s) print k,s[k]}'   # free count
# running jobs by project:
qstat -f | awk '/^Job Id:/{a="";s="";nd=0} /Account_Name = /{a=$3} \
  /job_state = /{s=$3} /Resource_List.nodect = /{nd=$3} \
  /^$/{if(s=="R"&&nd>=64) print a, nd}'
```
