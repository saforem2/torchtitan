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

**It is the OFFLINE COUNT, not the rack, that sets the ceiling.** A 16N
capture (`12474716`) drew all 16 of its nodes from `x1921` -- the "bad" rack --
and started on the first attempt. x1921's healthy nodes are fine; only the 36
offline ones are poison, and PBS will not schedule onto those anyway.

So the rule is simply: **a job that can be satisfied entirely from healthy
nodes runs; one that cannot, requeues forever.** With 36 of 129 nodes offline,
64N had no clean allocation to draw and 16N had many. Reducing N is a real
workaround, not a rack-affinity trick -- and there is no way to express "avoid
the sick nodes" in the submit, because PBS already believes it is.

Reduce N while holding the science fixed. For the 80B capture that meant
keeping GBS identical and moving the compensation into GAS:

```
64N:  768 ranks / TP4 = 192 dp,  25,165,824 / (4096*192) = GAS  32
16N:  192 ranks / TP4 =  48 dp,  25,165,824 / (4096* 48) = GAS 128
```

Same tokens/train-step, so it stays a reproduction rather than a new
experiment. Check the arithmetic divides exactly -- 12N and 20N give
fractional GAS and would silently shift GBS.

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

## Pinning to "free" nodes is NOT enough (2026-09-06, later)

The host-pinned 64N capture (`12474718`) launched correctly -- 768/768 GPUs on
all 64 named hosts, right command -- and died at 59 seconds with 0 training
steps:

```
x1922c7s2b0n0...: rank 744 exited with code 127
Couldn't change directory to /lus/tegu/.../torchtitan: No such file or directory
  (57 of these, ~4-5 nodes' worth, out of 768 ranks)
rc=143
```

Exit 127 is command-not-found: that node could not `cd` into the repo, so it
could not find python. One rank dying takes all 768 down with it.

**The fault is not specific to `home`. Some nodes cannot see `tegu` either.**
And the damning detail:

```
$ pbsnodes x1922c7s2b0n0 | grep state
     state = free
```

PBS considered that node healthy. The prologue mount check did not catch it --
which makes sense, since the failing check is what OFFLINES a node, so a node
whose check silently passes (or never runs for that filesystem) stays `free`
while being unusable.

**So `free` is not the same as usable, and a host list filtered by `pbsnodes`
state is not a safe select.** Filter by an actual access test instead: run a
`stat` of a known repo path from every candidate node and keep only the ones
that answer.

```bash
# survey: one rank per candidate node, stat a path on the target filesystem
mpiexec --envall --np $N --ppn 1 /usr/bin/stat -c OKNODE /lus/tegu/.../some_file
```

Nodes that print `OKNODE` are usable; nodes that print `No such file or
directory` are not, whatever `pbsnodes` says about them.

### Reading these logs in the right order

The log is dominated by `Couldn't change directory` -- 57 lines of it -- and
it is tempting to read that as the cause. It is BOTH: the cause on the few
nodes that cannot see the filesystem, and teardown noise everywhere else once
mpiexec starts killing ranks. The line that actually identifies the failure is
the single `exited with code 127`, which names the node. Grep for the exit
codes first, then the signals, and treat repeated messages as an effect until
proven otherwise.

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
