# ALCF ticket draft: sunspot compute nodes failing `home` mount check

**Ready to send; not sent.** Written 2026-09-06. Verify the counts are still
current before submitting -- regenerate with the commands at the bottom.

---

**Subject:** Sunspot: 28 nodes offlined by `EXECJOB_BEGIN: failed mount check:
home not mounted`; multi-node jobs requeue indefinitely

**Severity:** machine-wide degradation. 42 of 129 nodes offline (28 from this
cause), 6 free at time of writing. Ongoing since Fri Sep 4.

## What is happening

The PBS prologue mount check for `home` is failing on compute nodes. On
failure the node is taken offline and the job is requeued, so any job that
names `home` in `#PBS -l filesystems=` can cycle `Q -> R -> E -> Q`
indefinitely with `Exit_status = -3`, never executing a line of its script and
producing no output file or log.

Our job `12474709` reached **`run_count = 19`** this way.

`/home` is mounted normally on the login nodes, so this is not visible
interactively -- only the compute-node mount is affected.

## Evidence

Node comments:

```
x1921c0s1b0n0  offline  EXECJOB_BEGIN: failed mount check: home not mounted(job 12474677)
x1921c2s1b0n0  offline  EXECJOB_BEGIN: failed mount check: home not mounted(job 12474689)
...
```

Breakdown:

```
     28 failed mount check: home not mounted
      1 failed mount check: flare not mounted
```

Offlining jobs, by owner:

| job | owner | nodes offlined |
|-----|-------|----------------|
| 12474689 | zippy | 8 |
| 12474677 | brianhol | 8 |
| 12474690 | zippy | 7 |
| 12474678 | brianhol | 5 |
| 12474697 | zippy | 2 |

Earliest affected job `ctime`: **Fri Sep 4 17:48:52 2026**.

Full node list:

```
x1921c0s1b0n0 x1921c0s3b0n0 x1921c0s7b0n0 x1921c2s1b0n0 x1921c2s2b0n0
x1921c2s3b0n0 x1921c2s4b0n0 x1921c2s5b0n0 x1921c2s6b0n0 x1921c2s7b0n0
x1921c3s1b0n0 x1921c3s6b0n0 x1921c4s3b0n0 x1921c4s4b0n0 x1921c4s6b0n0
x1921c4s7b0n0 x1921c5s1b0n0 x1921c5s2b0n0 x1921c5s4b0n0 x1921c5s5b0n0
x1921c5s6b0n0 x1921c6s3b0n0 x1921c6s5b0n0 x1921c6s7b0n0 x1921c7s0b0n0
x1921c7s1b0n0 x1921c7s5b0n0 x1921c7s6b0n0 x1921c7s7b0n0
```

## Asks

1. Restore the `home` mount on the compute nodes and return the 28 offlined
   nodes to service (they do not recover on their own).
2. If it is feasible, consider whether a failed mount check should requeue the
   job. The requeue is what turns one bad node into a silent, unbounded loop:
   the job gives no signal that anything is wrong -- no output file, no log,
   no error -- for as long as it keeps retrying. A held job with a comment
   would be far easier to diagnose.

## Regenerating these numbers

```bash
pbsnodes -l | grep "not mounted"
pbsnodes -l | grep -oE "failed mount check: [a-z]+ not mounted" | sort | uniq -c
pbsnodes -l | grep -oE "job [0-9]+" | sort | uniq -c | sort -rn
pbsnodes -avSj | awk 'NR>2{print $2}' | sort | uniq -c
```

Our own workaround (drop `home` from the filesystems request when unused) and
the diagnosis path are documented in
`docs/guides/known-bugs/sunspot-home-mount-check-offlines-nodes.md`.
