# Aurora: 2098-node job killed at 9h13m of 24h with `Exit_status = -14`

> **Status: UNEXPLAINED from the user side. Draft ticket below, NOT yet sent.**
>
> Written 2026-08-16 for umbrella job `8756070`. Two separate anomalies hit the
> same job: six `Execution server rejected request` bounces at seat time, then
> an external kill 9h13m into a 24h allocation.
>
> **Read [`sunspot-ccl-allgatherv-outage-20260810.md`](sunspot-ccl-allgatherv-outage-20260810.md)
> before sending this.** That one blamed a cluster fault for what turned out to
> be our own code, and had to be retracted. The difference here is that the
> exclusions below are direct checks rather than inference -- but the honest
> position is still "we cannot see the cause", not "the cluster broke".

## What is actually established

**MEASURED.** All from `qstat -xf 8756070` and the trainer console logs.

| | |
|---|---|
| Job | `8756070`, `agpt-multi-autoretry`, queue `large` |
| Nodes | 2,098 (24,576 ranks) |
| Requested walltime | 24:00:00 |
| `stime` | Sun Aug 16 03:21:43 2026 UTC |
| `obittime` | Sun Aug 16 12:37:55 2026 UTC |
| `resources_used.walltime` | **09:13:43** |
| `Exit_status` | **-14** |
| `run_count` | 6 |

The three live trainers were **mid-stride**, logging clean steps 2m39s before
the kill, with healthy grad norms and normal throughput:

```
t0 2b-512   07:35:49  step: 3312   loss: 2.54346  grad_norm: 0.5144  tps: 2,832
t1 20b-512  07:35:27  step: 9109   loss: 2.38496  grad_norm: 0.2007  tps:   349
t2 20b-256  07:35:16  step: 10381  loss: 2.36638  grad_norm: 0.1648  tps:   421
```

No error, no signal message, no traceback on any slot. The umbrella script's
own log ends at `all 5 trainers launched; waiting...` and never printed a stop
line, so **the script was terminated from outside rather than exiting**.
There is no `.e8756070` stderr file at all.

## Excluded, each by a direct check

| Hypothesis | Check | Result |
|---|---|---|
| Head node died | `pbsnodes x4305c0s2b0n0` | healthy, already `job-exclusive` on another job |
| Allocation exhausted | `sbank-list-allocations -p AuroraGPT` | 1,429,777.4 node-hours available |
| Queue disabled | `qstat -Qf large` | `enabled = True`, `started = True` |
| Reservation collision | `pbs_rstat` | only Mon 15:00 and Wed 14:00 windows |
| Our own fault (NaN/OOM/CCL) | grep all three console logs | nothing; last lines are normal step logs |
| Walltime | `resources_used.walltime` 09:13:43 vs 24:00:00 | not close |

**What we cannot determine:** PBS exposes no user-visible record of which
entity issued the kill. `-14` is a PBS-side termination. Remaining candidates
are an admin action, scheduler preemption, or a system event that marked no
node -- and we have no basis to choose among them.

## The other anomaly on the same job

Before it ran, `8756070` was seated and rejected **six times**:

```
comment = Not Running: PBS Error: Execution server rejected request
run_count = 6
```

Every attempt drew the same `x4305c0s*` rack, whose lead node was:

```
x4305c0s1b0n0  state = offline
  comment = EXECJOB_BEGIN: skipped execjob_end found via tmpfs check;
            reboot required... (job 8756070)
```

**Correcting an earlier reading of ours:** we first took this as our job
offlining one node per attempt. It is not. Only that single node names
`8756070`; at the same moment 13 other `reboot required` nodes belonged to
nine other users' jobs (`8712340`, `8753544`, `8757432`, `8759224`, ...). It
is a cluster-wide condition, and PBS simply kept selecting a node set with a
sick lead. Attempt 6 landed on a healthy set and ran.

Worth asking about in the same ticket, since the `tmpfs check / reboot
required` state appears to be widespread.

## Impact

Low. All three live chains checkpointed minutes before the kill (worst case 81
steps lost, inside the 100-step interval):

- 2B-512 stage-2 dolmino -> `step-3300`
- 20B-512 -> `step-9100`
- 20B-256 -> `step-10300`

The cost is the ~14.8h of unused allocation on 2,098 nodes.

---

## Draft ticket -- REVIEW BEFORE SENDING

**To:** support@alcf.anl.gov
**Subject:** Aurora: 2098-node job 8756070 killed with Exit_status -14 at 9h13m of a 24h allocation

Hello --

Job `8756070` (project AuroraGPT, queue `large`, 2098 nodes, 24h walltime) was
terminated at 2026-08-16 12:37:55 UTC after 09:13:43 of runtime, with
`Exit_status = -14`. I would like to understand what issued the kill, since
nothing on our side accounts for it and I would rather not resubmit into the
same outcome.

From our side this does not look like a job fault:

- All three running trainers logged normal progress until 07:35 local
  (12:35 UTC), roughly two minutes before termination -- steady loss, healthy
  gradient norms, expected throughput. No error, signal, or traceback in any
  rank's output.
- Our launcher script's log ends mid-run with no stop line, which suggests it
  was terminated externally rather than exiting on its own.
- No `.e` stderr file was produced.
- The job's first node, `x4305c0s2b0n0`, is healthy and has since been
  allocated to another job.
- Our allocation has 1.4M node-hours remaining, and the `large` queue is
  enabled and started.

Could you tell from the server-side logs what terminated it (admin action,
preemption, or a system event)?

Separately, and possibly related: this same job was assigned and rejected six
times before it started, each time with
`PBS Error: Execution server rejected request`. Each attempt drew a node set
whose first node, `x4305c0s1b0n0`, was offline with:

```
EXECJOB_BEGIN: skipped execjob_end found via tmpfs check; reboot required
```

At that time roughly 14 nodes cluster-wide were in that same
`reboot required` state across several different users' jobs, so it does not
appear specific to us -- but the scheduler repeatedly selecting a set with an
offline lead node cost us six seat attempts. Is that state being cleared
automatically, or does it need manual intervention?

Happy to provide any additional logs.

Thanks,
Sam Foreman (foremans), AuroraGPT

---

## Related

- [`dispatch-log.md`](../../live/dispatch-log.md) -- the per-slot record
  for this umbrella, including the two failures that *were* ours
  (`Config.job` AttributeError on t3/t4, and the t0 token-budget overshoot).
- [`umbrella-bad-alloc-init.md`](umbrella-bad-alloc-init.md) -- the separate,
  still-open `std::bad_alloc` at CCL init, which t3/t4 also hit here and
  recovered from.
- [`sunspot-ccl-allgatherv-outage-20260810.md`](sunspot-ccl-allgatherv-outage-20260810.md)
  -- the retracted cluster-fault ticket. Cautionary precedent.
