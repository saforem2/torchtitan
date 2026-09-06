# Sunspot: rack x1921 fails the `home` mount check; 64N jobs cannot run

**Status (2026-09-06): live site issue, ALCF-side. Rack `x1921` is 36 offline
/ 28 job-exclusive -- ZERO free. Rack `x1922` is healthy.** Ongoing since
Fri Sep 4. Not caused by our jobs.

## Symptom

A 64-node job never starts. It cycles `Q -> R -> E -> Q` with no walltime
accumulated, no `.o` file, and no log directory:

```
Exit_status = -3
run_count = 19
```

`-3` is PBS for "exec failed, requeue". Job `12474709` reached `run_count=19`
without executing a line of its script.

## Cause

The PBS prologue mount-checks each allocated node. On `x1921` the `home`
check fails; the node is taken offline and the job requeued:

```
EXECJOB_BEGIN: failed mount check: home not mounted(job 12474677)
```

**All 29 mount-check failures are in `x1921`. Zero in `x1922`.** `/home` is
mounted normally on the login nodes, so nothing looks wrong interactively.

The scale threshold follows directly. `x1922` has 22 free nodes, so:

- a job needing <= 22 nodes can be satisfied entirely from `x1922` and runs;
- a **64-node job cannot**, so it necessarily draws from `x1921`, hits a bad
  node, and requeues -- forever.

## There is no client-side workaround

**Dropping `home` from `#PBS -l filesystems=` does NOT help.** That was the
obvious fix and it is wrong. A matched pair of 8-node probes settles it:

| probe | filesystems | landed on | run_count | result |
|-------|-------------|-----------|-----------|--------|
| 12474714 | `tegu:home` | x1922 | 1 | `Exit_status = 0`, ran |
| 12474715 | `tegu` | x1922 | 1 | `Exit_status = 0`, ran |

**The probe that requested `home` succeeded.** The flag is not the variable;
the rack is. Both probes drew from `x1922` and both passed.

Until ALCF restores `x1921`, a 64N job here cannot be made to run by any
change to our scripts. Options are: wait for the rack, or run at <= 22 nodes.

A 64N probe requesting `tegu` alone (`12474713`) confirms it. Both of its
attempts drew the identical set -- 36 `x1922` + 28 `x1921` -- because that is
the only 64 nodes outside the offline pool, and both failed:

```
run_count = 2      nodes: 64   Counter({'x1922': 36, 'x1921': 28})
```

The scheduler has no other allocation to give. Note those 28 `x1921` nodes are
`job-exclusive`, not offline -- they bounce jobs at the mount check **without
being offlined themselves** (`pbsnodes -l | grep -c 12474713` returns 0). So
the offline count understates the damage: 29 nodes are marked bad, but the
whole rack is unusable.

## PBS gives up on its own

After enough failures PBS system-holds the job:

```
job_state = H
Hold_Types = s
comment = job held, too many failed attempts to run
run_count = 21
```

Worth knowing before you intervene -- our capture job reached this state by
itself. A system hold is released with `qrls`, but do not release it until the
rack is back or it will simply resume burning attempts.

## Diagnosis path, and two dead ends worth knowing

Ladder the scale with a TRIVIAL payload -- the scale at which it breaks is
the diagnosis:

| probe | scale | payload | run_count | result |
|-------|-------|---------|-----------|--------|
| 12474711 | 1N | `echo` | 1 | `Exit_status = 0` |
| 12474712 | 64N | `echo` | 4 | `-3`, never ran |
| 12474709 | 64N | 80B train | 19 | never ran |

A bare `echo` at 64N failing identically to the real job is what rules out
your script.

Then split the suspected variable with a **matched pair at a scale that
runs** -- that is what turned "drop the flag" from a plausible fix into a
refuted one, for the cost of two 5-minute jobs. A fix you have not tested
against its own control is a guess.

Dead ends:

- *"10 of the 64 nodes are unreachable."* A parsing artifact: `qstat -f`
  wraps continuation lines with a **leading tab**, so `tr -d '\n'` splices
  hostnames together. Strip `\n\t`, not `\n`. All 64 were healthy.
- *"My retries offlined these nodes."* `12474709` appears in **zero** node
  comments; all 29 name jobs owned by `brianhol` and `zippy`. It was bouncing
  off nodes that were already dead. Check with
  `pbsnodes -l | grep -c "<job id>"` before claiming or assigning blame.

Also: `E` is a normal transition, not a failure, and `job_state = R` with an
empty `stime` means PBS has allocated but not started. Read `run_count`
before concluding anything from a state letter.

## Operational notes

A job in an exec-failure loop **outruns a plain `qdel`** -- PBS requeues it
faster than the delete lands. Hold it first:

```bash
qhold <id> && qdel -W force <id>
```

## Reporting

Ticket drafted at `sunspot-home-mount-ticket-draft.md`. Regenerate the counts
before sending:

```bash
pbsnodes -l | grep "not mounted"
pbsnodes -l | grep "not mounted" | awk '{print $1}' | cut -c1-5 | sort | uniq -c
pbsnodes -avSj | tail -n +3 | awk '{print substr($1,1,5), $2}' | sort | uniq -c
```

Related: `project_silent_noop_exit_zero` -- the sibling shape, where work
reports success having done nothing.
