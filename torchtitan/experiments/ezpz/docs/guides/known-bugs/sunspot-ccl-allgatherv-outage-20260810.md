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
