# RETRACTED -- this was NOT a cluster fault

> [!CAUTION]
> **Retracted 2026-08-11. Do NOT send the email below.** It blames the Sunspot
> fabric for an FSDP failure; a minimal reproducer disproves that.
>
> Job **12473017** ran bare `torch.distributed.all_gather_into_tensor` -- the
> exact primitive failing at `distributed_c10d.py:4381` -- with no torchtitan
> involved, on `x1922c5s0/c5s2/c5s6/c5s7`, i.e. INCLUDING the two hosts that
> every failing job had drawn:
>
> ```
> CCLMIN init OK world=48 torch=2.13.0.dev20260519+xpu
> CCLMIN A_TINY OK sum=9024.0
> CCLMIN B_LARGE OK (64 MiB/rank)
> ```
>
> Tiny AND FSDP-scale (64 MiB/rank) all-gathers both succeed, so the collective
> layer is healthy and **the fault is in the torchtitan/FSDP path**.
>
> What I got wrong: I treated "three FSDP jobs failed on different nodes" as
> proof of a fabric fault without ever testing the primitive directly. All three
> ran the full stack -- blendcorpus, W&B, checkpointing, a large model -- before
> reaching any collective, so any of those could have been the cause. The
> cross-chassis spread ruled out ONE bad node; it did not rule out our own code.
>
> The symptom description and job IDs below remain accurate and useful. Only the
> cluster-fault CONCLUSION is withdrawn. Repro script: `tmp/ccl_minimal.pbs`.

---

**To:** support@alcf.anl.gov
**Subject:** Sunspot: all multi-node PyTorch training failing in oneCCL allgatherv_ring (both chassis)

Hello -- following up on the `x1921c4s3b0n0` mount issue (thank you for the
quick fix on that one).

Since around 12:00 today, **every multi-node PyTorch/FSDP job we launch dies at
the first collective** with the same oneCCL error. This is a separate problem
from the mount issue and appears to be cluster-wide.

## Error

```
RuntimeError: oneCCL: allgatherv_ring.hpp:83 allgatherv_ring_blocking:
    EXCEPTION: atl_comm->wait(ep_idx, recv_req)
```

raised from FSDP's parameter all-gather:

```
File ".../torch/distributed/distributed_c10d.py", line 4381, in all_gather_into_tensor
    work = group._allgather_base(output_tensor, input_tensor, opts)
```

It fires immediately at "Training starts" -- i.e. the first FSDP all-gather of
step 1. Nothing trains; no step ever completes.

## It is not our code, not one node, and not one chassis

Three jobs, deliberately varied to isolate the cause:

| PBS job | nodes | our instrumentation | result |
|---------|-------|---------------------|--------|
| 12472864 | 2 (`x1922c4s0`, `x1922c4s2`) | custom fwd hooks | allgatherv_ring |
| 12472866 | 2 (`x1922c5s0`, `x1922c5s2`) | **none** (control) | allgatherv_ring |
| 12472874 | 4 (`x1921c1s0/1/2/4`) | none | allgatherv_ring |

- **Not our instrumentation:** 12472866 is byte-identical to 12472864 with our
  profiling hooks removed, and fails the same way.
- **Not a single bad node:** three disjoint node sets.
- **Not one chassis:** it reproduces on both `x1922` and `x1921`.

The same job configuration (2-node `agpt-2b`, FSDP, XPU) ran normally earlier
this week, and an 8-node training run completed 1600 steps cleanly on 2026-08-08
(job 12472766).

## Environment

- torch `2.13.0.dev20260519+xpu`, project venv on `/lus/tegu`
- launched via `mpiexec` (PALS 1.8) through our `ezpz` wrapper
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `CCL_LOG_LEVEL=ERROR`

## Question

Did anything change in the CCL / libfabric / fabric configuration on Sunspot
today? The timing is close to the `x1921c4s3b0n0` mount repair, so we wondered
whether the two might be related -- though we have no evidence for that beyond
the coincidence, and we may simply be seeing an unrelated fabric issue.

Happy to run any diagnostic you would like, and to provide full logs. This
currently blocks all of our multi-node work on Sunspot.

Thanks,
Sam Foreman (foremans)
Project: datascience

## Update: the PBS server is also unresponsive (2026-08-10, later)

After the CCL failures above, the scheduler itself stopped answering. Every PBS
client call hangs indefinitely -- `qstat -u $USER`, `qstat -Q workq`,
`pbsnodes -a`, and even a bare `qstat -B` all exceeded 120 s with no output,
across four separate attempts over ~15 minutes.

This is **not** login-node contention. Plain `ssh` returns instantly and the
login node is idle:

```
04:22:53  up 21 days,  5 users,  load average: 0.03, 0.01, 0.00
```

So the clients are blocking on the PBS server, not on local CPU. We cannot
submit, query, or cancel jobs at present.

**Correction / refinement.** The clients later stopped hanging, which briefly
looked like recovery -- but they now return **`rc=0` with completely empty
output**, which is worse than a hang because it reads as success:

```
qstat -B          -> (no output)  rc=0
pbsnodes -a       -> (no output)  rc=0
qstat -x 12472874 -> (no output)  rc=0     # a job we ran hours ago
```

`pbsnodes -a | grep -c 'state = free'` therefore reports **0 free nodes**, which
is indistinguishable from a full cluster unless you notice that the *node list
itself* is empty. `/etc/pbs.conf` still points at
`sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov`, and the login node is idle
(load 0.03), so this is the PBS server returning empty result sets rather than a
client or contention problem.

Practical consequence: **do not trust a low "free nodes" count on Sunspot right
now** -- verify the query returns any rows at all before acting on it. We
briefly misread the empty result as "cluster is full".

Whether this is related to the oneCCL failures above is unknown -- we only note
that both appeared the same day, after the `x1921c4s3b0n0` mount repair.


---

## Leading hypothesis (2026-08-11): upstream #4068 vs our XCCL timeout patch

Not yet tested -- recorded so it is not lost.

`96276d865` "Exclude fake-backed axes from `get_all_one_dimensional_meshes`"
(#4068) entered our branch in merge `5de204e6b` on **08-08 11:14**. The last
known-good multi-node run (Wave 3 cosmo 8N, job 12472766, 1600 steps) was
**08-08**; the first `allgatherv_ring` failure was **08-10**. So the change is
on the right side of the timeline, though the exact hour of Wave 3 vs the merge
is still unconfirmed.

What it changed: the function now filters axes to those where
`self._mesh_exist(k, v.size())`. Its own docstring shows the result shrinking:

```
- dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss', 'efsdp'])
+ dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss'])
```

Why that could matter to us specifically: `ezpz/trainer.py` consumes that
function **twice** (lines 89 and 104) in our XCCL timeout monkeypatch --
`_set_pg_timeout` over every returned mesh, then `ProcessGroupXCCL.set_timeout`
over the same set. If an axis FSDP actually collectives on is now excluded, that
group keeps a default timeout our patch previously overrode, which is a
plausible route to a hang/abort inside `allgatherv_ring` -- the error is
`atl_comm->wait(...)`, i.e. a wait that did not complete.

**Consistent with the evidence:** bare `all_gather_into_tensor` on the default
group works fine (job 12473017), and the default group is the one entry our
patch always covers via the explicit `+ [None]`.

### How to test (cheap, 2N)

1. `git stash` nothing -- instead run the 2N smoke at the merge's PARENT
   (`5de204e6b^`) and at `ezpz` HEAD. If the parent trains and HEAD fails, this
   is confirmed and the fix is ours, not upstream's.
2. If confirmed: make the patch iterate the pre-filter axis set, or add the
   FSDP/efsdp groups explicitly rather than relying on the returned dict.

Do NOT re-open the cluster-fault theory: 12473017 already disproved it.
