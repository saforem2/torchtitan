# Intel ticket: XPU graphs cannot capture oneCCL collectives

> [!NOTE]
> **Status: ready to file. Needs an ALCF/Intel account to submit.**
> Gate cleared 2026-08-20 -- see "Why this was held" below.

**Component:** oneCCL / Level Zero / torch XPU graph capture
**Severity:** blocks a shipped feature for all distributed training
**Reported by:** ALCF AuroraGPT (Sunspot)

## Summary

`torch.xpu.graph` capture works, and oneCCL collectives work, but a collective
issued *inside* a capture region raises:

```
RuntimeError: wait method cannot be used for an event associated with a
              command graph.
```

Because every distributed configuration issues collectives inside the training
step (FSDP all-gather, DDP/FSDP all-reduce), XPU graphs are currently unusable
for distributed training. Single-rank capture of pure compute is fine.

## Minimal reproduction

`probe_xpu_graph_collective.py` (attached, 48 lines). 12 ranks, one node, ~20s.

```
world=12 torch=2.13.0a0+gitcf30153
  all_gather OUTSIDE capture: OK
  all_gather INSIDE  capture: FAILS -> RuntimeError: wait method cannot be
                              used for an event associated with a command graph
  matmul     INSIDE  capture: OK
```

The `matmul INSIDE` line rules out a general capture problem. The
`all_gather OUTSIDE` line rules out a general collective problem. The only
failing combination is collective-inside-capture.

## Scope of what DOES capture

Verified separately (job `12473186`), each with a fresh memory pool:

| operation | inside capture |
|---|---|
| bare matmul | OK |
| `nn.Linear` forward | OK |
| raw forward + backward | OK |
| `nn.Linear` forward + backward | OK |
| **oneCCL collective** | **FAILS** |

So forward, backward, autograd and `nn.Module` are all capturable. The
collective is the single remaining blocker.

## Impact

In a real torchtitan run the first captured step fails in
`ProcessGroup.all_gather_single`. All of our production training is
multi-rank, so the feature cannot be used at all. Our wrapper
(`experiments/ezpz/xpu_graph.py`) is committed but **off by default**
(`EZPZ_XPU_GRAPHS=1`) and degrades to eager.

The CUDA equivalent works -- NCCL collectives are capturable under CUDA
graphs -- so this is an XPU-specific gap in an otherwise-shipped feature.

## Environment

```
torch          2.13.0a0+gitcf30153
module         frameworks/2026.1.0
oneCCL         /opt/aurora/26.181.0/oneapi/ccl/latest
backend        XCCL
ZE_FLAT_DEVICE_HIERARCHY=FLAT
platform       Sunspot (Intel Max 1550)
```

## Why this was held (and what turned out to be our bug)

This ticket was drafted 2026-08-16 and immediately gated, because a *second*
capture failure was in play and it was not clear which subsystem owned it:

```
nn.Linear forward -> it->second->use_count > 0 INTERNAL ASSERT FAILED
```

That is an internal assert, not a documented restriction, and the suspicion
was that our wrapper caused it by reusing a single `graph_pool_handle` across
captures.

**That suspicion was correct.** Job `12473184` separated the variables:

```
matmul,           fresh pool   OK
nn.Linear fwd,    fresh pool   OK
matmul,           shared pool  OK
nn.Linear fwd,    shared pool  FAILS -> use_count > 0 INTERNAL ASSERT
```

Pool reuse was the trigger, so the `use_count` assert is **ours, and fixed**
(one pool per graph, `xpu_graph.py:25`). It is deliberately excluded from this
ticket. The collective finding above is independent of it and reproduces with
fresh pools.

## Performance

Not measured, and we are not claiming a number: the graphs-ON arm has never
completed a step. Eager baseline at the probe config is ~3,626 tps (3627/3626
across two runs). Our expectation is that the upside is modest for us -- graphs
mainly remove launch overhead, which matters most for small kernels at small
scale, whereas our steps are large and communication-bound at production node
counts. That is a prediction, not a result. If capture becomes possible we
will measure it rather than assume.

## Related

- [`known-bugs/xpu-graphs-block-oneccl-collectives.md`](../guides/known-bugs/xpu-graphs-block-oneccl-collectives.md) -- full investigation log
- [`intel-ur-die-urEventWait-a2a.md`](./intel-ur-die-urEventWait-a2a.md) -- separate UR event-handling fault, same stack
