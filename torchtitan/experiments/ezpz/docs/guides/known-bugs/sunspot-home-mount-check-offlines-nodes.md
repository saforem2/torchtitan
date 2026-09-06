# Sunspot: the `home` mount check offlines nodes and requeues jobs forever

**Status (2026-09-06): live site issue. 28 of 129 nodes offline from this
cause alone; only 22 free.** Not caused by our jobs -- the offlining jobs
belong to other users -- but it blocks every multi-node submission we make.

## Symptom

A multi-node job never starts. `qstat` shows it cycling `Q -> R -> E -> Q`
with no walltime accumulated, no `.o` file, and no log directory. The exit
status is the tell:

```
Exit_status = -3
run_count = 16
```

`-3` is PBS for "exec failed, requeue". Each retry grabs a fresh node set,
fails the same way, and the counter climbs. Job `12474709` reached
`run_count = 16` without ever running a single line of its script.

## Cause

The PBS prologue runs a filesystem mount check on every allocated node.
When it fails, PBS **takes the node offline and requeues the job**:

```
EXECJOB_BEGIN: failed mount check: home not mounted(job 12474677)
```

`/home` is mounted on the login nodes, so nothing looks wrong interactively.
It is the compute-node mount that is failing.

This is self-amplifying **for the job that first hits a healthy node**:
that node goes offline and the pool shrinks for everyone.

But a requeuing job is not necessarily the one doing the damage. Ours
(`12474709`, `run_count` 19) appears in **no** node comment -- every one of
the 29 offlined nodes names one of five jobs belonging to `brianhol` and
`zippy`. It was landing on already-dead nodes and bouncing. Check before you
attribute:

```bash
pbsnodes -l | grep -c "<your job id>"      # 0 = your job offlined nothing
pbsnodes -l | grep -oE "job [0-9]+" | sort | uniq -c | sort -rn
```

The practical consequence differs: a job that offlines nodes should be
stopped promptly; a job that merely bounces is only costing you the 64-node
allocation it grabs on each attempt -- which still starves anything queued
behind it.

## How to tell it apart from a bad script

Ladder up. The scale at which it starts failing is the diagnosis, and this
took three probes to pin down because the first two readings each pointed
somewhere wrong:

| probe | scale | payload | run_count | result |
|-------|-------|---------|-----------|--------|
| 12474711 | 1N | `echo` | 1 | `Exit_status = 0`, output written |
| 12474712 | 64N | `echo` | 2+ | requeued, same as the real job |
| 12474709 | 64N | 80B train | 16 | never started |

**A trivial `echo` at 64 nodes fails identically to a real training job.**
That is what rules out your script. Do not skip the trivial-payload probe:
without 12474712 the evidence pointed straight at the one job, and the
obvious next move -- qdel and resubmit -- would have changed nothing while
offlining another sweep of nodes.

Two readings that looked right and were not:

- *"10 of the 64 nodes are unreachable."* An artifact of parsing wrapped
  `qstat -f` output with `tr -d '\n'`, which splices hostnames together.
  PBS wraps continuation lines with a leading tab; strip `\n\t`, not `\n`.
  All 64 nodes were `job-exclusive` and healthy.
- *"My own retries offlined these nodes."* The job ids in the node comments
  (`12474677`, `12474689`, `12474690`, ...) belong to `brianhol` and
  `zippy`. Read the owner off the id before claiming the blame -- or
  assigning it.

## Workaround

Drop `home` from the filesystems request when the job does not need it:

```bash
#PBS -l filesystems=tegu      # not tegu:home
```

Our 80B jobs keep repo, venv, logs and checkpoints on tegu and reference
`$HOME` nowhere, so the `home` in that line was vestigial -- and it is the
only reason the check applied. Confirm with a grep for HOME references on
the script before removing it; the module stack or a conda base can reach
into `$HOME` even when your own lines do not.

## Reporting

Worth a ticket because it degrades the machine for everyone and the offlined
nodes do not come back on their own. Include: the `pbsnodes -l` lines naming
the failure, the count (28 nodes), and the point that `/home` is mounted on
the login node so the failure is compute-side only.

Collect the current list with:

```bash
pbsnodes -l | grep "not mounted"
pbsnodes -l | grep -oE "failed mount check: [a-z]+ not mounted" | sort | uniq -c
```

Related: `project_silent_noop_exit_zero` -- the sibling shape, where work
reports success having done nothing. Here the job reports nothing at all,
which is at least honest, but the 16 silent retries cost an afternoon.
