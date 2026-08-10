**To:** support@alcf.anl.gov
**Subject:** Sunspot: node x1921c4s3b0n0 cannot see /lus/tegu project path -- kills every job it lands on

Hello,

Sunspot node **`x1921c4s3b0n0`** appears to be missing a `/lus/tegu` mount. Any
job that lands on it fails within a couple of seconds, and because one bad rank
aborts the whole MPI job, a single node takes down the entire allocation. Most
recently this killed a 72-node job of ours.

The path is fine everywhere else -- login nodes and every other compute node we
have tested read it normally.

---

## Confirmed failure: PBS job 12472827 (72 nodes, 2026-08-10 09:33)

`x1921c4s3b0n0` was in the active node list. The launcher attributes the exit
directly to it:

```
x1921c4s3b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: rank 228 exited with code 127
```

accompanied by 24 instances of:

```
Couldn't change directory to /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan:
    No such file or directory
```

The job died 7 seconds after launch. Our launcher's auto-retry then rotated a
*different* (healthy) node and gave up:

```
[auto-retry] attempt 1 - active=62 hosts, spare=10 hosts
[auto-retry] blind rotation: x1921c1s0b0n0-hsn0... -> x1922c3s1b0n0-hsn0...
[auto-retry] FAILOVER STOP: stuck_pre_training (two consecutive attempts with
             zero step= markers, rc=143)
```

It rotated the wrong node because the error is a generic `chdir` failure on
every rank, which gives automatic failover nothing to attribute.

**Control:** the immediately following job **12472829**, identical in every
respect except that we removed `x1921c4s3b0n0` from the node file, no longer
produced any `exited with code 127`.

## Direct measurement of the mount difference

An 8-node probe (PBS job **12472561**, 2026-08-05) ran `mount | grep -c tegu`
and an `ls` of the project path, one rank per node:

```
x1921c2s4b0n0    tegu=OK   repo=OK   mounts=3
x1921c2s6b0n0    tegu=OK   repo=OK   mounts=3
x1921c2s7b0n0    tegu=OK   repo=OK   mounts=3
x1921c4s0b0n0    tegu=OK   repo=OK   mounts=3
x1921c4s3b0n0    tegu=OK   repo=FAIL mounts=2     <-- the bad node
x1921c4s4b0n0    tegu=OK   repo=OK   mounts=3
x1921c4s5b0n0    tegu=OK   repo=OK   mounts=3
x1921c4s6b0n0    tegu=OK   repo=OK   mounts=3
```

`/lus/tegu` itself resolves on the bad node, but it carries **2** tegu mounts
where healthy nodes carry **3**, and the project directory is not visible. That
suggests one missing or stale mount rather than a total filesystem loss.

Note: this job's stdout has since aged out of our working directory, so the
table above is transcribed from our notes at the time rather than re-read from
the file. Job 12472827 (above) is still on disk and is the primary evidence.

## History

`x1921c4s3b0n0` has failed every time we have seen it, over 2026-08-05 to 08-10:

| PBS job | date | outcome |
|---------|------|---------|
| 12472558 | 08-05 | all ranks exit 127 |
| 12472560 | 08-05 | all ranks exit 127 |
| 12472561 | 08-05 | mount probe: `mounts=2`, project path invisible |
| 12472563 | 08-05 | excluded by our health probe as unhealthy |
| 12472573 | 08-05 | all ranks exit 127 |
| 12472827 | 08-10 | `rank 228 exited with code 127`, 72-node job lost |

Other nodes in the same rack (`c4s0`, `c4s4`, `c4s5`, `c4s6`) are healthy, so
this looks node-local rather than rack-wide.

## Why this is worth fixing rather than routing around

The failure surfaces no filesystem-level error -- only a failed `chdir` and exit
127 -- so it reads as an application bug. We lost several allocations before
probing mounts directly. Automatic node failover does not help, for the reason
shown above: it cannot attribute a uniform error to one node.

We now exclude this hostname explicitly, but that only protects jobs that know
to do it.

## Request

Please check and remount `/lus/tegu` on `x1921c4s3b0n0`, or mark the node
offline until it can be repaired.

One note: on 2026-08-10 the node was in `job-exclusive` state running another
user's job. If it was still in this condition, that job was likely affected too.

Thanks very much,
Sam Foreman (foremans)
Project: datascience
