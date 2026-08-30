# ALCF ticket -- Polaris nodes with a stuck GPU

**Status:** FILED 2026-08-30 by Sam Foreman. Awaiting ALCF response.
**Nodes:** `x3007c0s13b1n0`, `x3111c0s37b1n0`, `x3003c0s25b0n0`
**Cost so far:** jobs 7560196 (3h03m) + 7560197 (4h04m) + 7567541
(12m56s) = **~7h20m of 130-132 nodes, zero training steps**.

`x3003c0s25b0n0` surfaced later than the other two (job 7567541,
2026-08-29) with a different signature: `invalid device ordinal` rather
than `cudaErrorDevicesUnavailable`. Same practical effect -- one rank
loses its GPU and the allocation hangs.

As of 2026-08-30 21:51 UTC all three still show
`resources_available.ngpus = 4`, and `x3111c0s37b1n0` still carries the
`EXECJOB_END` comment below. `x3111c0s37b1n0` appeared in the estimated
node list for job 7567542, i.e. PBS is still assigning it.

## Suggested ticket text

> Two Polaris compute nodes each have one GPU that refuses
> `cudaSetDevice` with `cudaErrorDevicesUnavailable`, while PBS
> continues to advertise the node as healthy with `ngpus = 4`. Our jobs
> are repeatedly scheduled onto them and lose one rank per node, which
> hangs the whole allocation.
>
> Nodes: `x3007c0s13b1n0`, `x3111c0s37b1n0`, `x3003c0s25b0n0`
> Affected jobs: 7560196, 7560197, 7567541 (user foremans, project
> AuroraGPT)
>
> `pbsnodes x3111c0s37b1n0` reports:
>
> ```
> comment = EXECJOB_END 202608222332: 5 processes failed to terminate.
>           cleaning... (user yudas, jobid 7550518.polaris-pbs-01...)
> ```
>
> so the likely cause is leftover processes from an unrelated job on
> 2026-08-22 that never terminated.
>
> Observed on all three nodes: all four A100s enumerate, 0% utilisation,
> persistence mode on, compute mode Default, and
> `ecc.errors.uncorrected.volatile.total = 0` on every GPU -- so this
> does not look like an ECC/hardware fault. Exactly one GPU per node
> sits at ~5 MiB used while the other three hold ~427 MiB from our
> running ranks, and
> `nvidia-smi --query-compute-apps` shows **no process at all** holding
> the stuck GPU. It nonetheless rejects `cudaSetDevice`.
>
> Could these three nodes be drained and reset?

## Evidence to attach

```
$ pbsnodes x3111c0s37b1n0 | grep -E 'state|comment|ngpus'
     state = job-exclusive
     resources_available.ngpus = 4
     comment = EXECJOB_END 202608222332: 5 processes failed to terminate. ...

$ nvidia-smi --query-gpu=index,memory.used,ecc.errors.uncorrected.volatile.total --format=csv,noheader
0, 5 MiB, 0        <- refuses cudaSetDevice; held by no process
1, 427 MiB, 0
2, 427 MiB, 0
3, 427 MiB, 0
```

Diagnostic signature, identical across every affected job and node: the
5 MiB GPU's index equals the dead rank's **local** rank index
(`x3007c0s13b1n0` rank 13 -> local 1 -> GPU 1; `x3111c0s37b1n0` rank 420
-> local 0 -> GPU 0).

## Why we cannot work around it

Node rotation is the only lever `ezpz launch --auto-retry` has. Two
failover bugs found while chasing this are now fixed (see
[`known-bugs/polaris-failover-detect-machine-fqdn.md`](../guides/known-bugs/polaris-failover-detect-machine-fqdn.md)),
so the scraper now names these nodes correctly and swaps them out. But
PBS keeps re-allocating them to the next job, and the spare pool is
`select - NHOSTS_TRAIN` = 2. Rotation buys one job; it does not remove
the nodes from the pool.

## Check this first next time

`pbsnodes <node> | grep comment` named the root cause in one line, after
a fair amount of inference from `nvidia-smi` GPU-index arithmetic. **A
node can be healthy to the scheduler and unusable in practice.**
