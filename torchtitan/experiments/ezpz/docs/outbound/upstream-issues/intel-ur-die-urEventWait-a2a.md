# Intel ticket: `ur_die: urEventWait must not be called for an internal event`

> [!NOTE]
> **Status: ready to file. Needs an ALCF/Intel account to submit.**

**Component:** Intel Unified Runtime (event handling) / oneCCL
**Severity:** hard process abort mid-training, no recovery
**Reported by:** ALCF AuroraGPT (Sunspot)

## Summary

Under Expert Parallelism, a `all_to_all_single` inside the MoE token
dispatcher causes the Unified Runtime to abort the process:

```
ur_die: urEventWait must not be called for an internal event
terminate called without an active exception
Fatal Python error: Aborted
```

This is a UR-level abort -- not a torchtitan assertion, not a CCL error code,
and not catchable from Python. The process dies mid-step.

## Reproduction

Job `12473364`, 2 nodes / 24 ranks, `expert_parallel_degree=2`, LBS=1,
compile off. Fails at step 3 of 5.

```
step: 2  loss: 11.72962  grad_norm: 2.3820  memory: 30.53GiB(47.71%)  tps: 2,508
ur_die: urEventWait must not be called for an internal event
terminate called without an active exception
Fatal Python error: Aborted
```

Main-thread frame at abort:

```
torch/distributed/_functional_collectives.py:583  in all_to_all_single
  experiments/ezpz/moe/token_dispatcher.py:757    in dispatch
  torchtitan/models/common/moe.py:161             in forward
  torch/distributed/tensor/experimental/_func_map.py:247 in _local_map_wrapped
```

The call is a plain `all_to_all_single` over the EP mesh, computing global
token counts per local expert.

## It is not memory pressure

The obvious reading -- an allocation ceiling -- is ruled out by the size sweep.
Per-arm signatures from job `12473367` (2N, 5 steps each):

| config | steps | memory | `ur_die` | outcome |
|---|---|---|:--:|---|
| `moe_debugmodel_ep` | 5/5 | 44% | no | pass |
| `moe_2b_ep` | 3/5 | **48%** | **yes** | SIGABRT |
| `moe_10b_2b_sdpa_ep` | 2/5 | **80%** | **yes** | SIGABRT |
| `moe_10b_2b_sdpa_bmm_ep` | 1/5 | 86% | no | `UR_RESULT_ERROR_OUT_OF_RESOURCES` |
| `moe_7b_ep` | 1/5 | 94% | no | SIGTERM only |
| `moe_10b_2b_sdpa` (no EP) | 5/5 | 74% | no | pass |

The two `ur_die` arms abort at **48%** and **80%** memory, while arms at
**86%** and **94%** fail differently (a genuine resource ceiling, which also
hits the non-EP `moe_7b` at 95%). A smaller model aborting at half the memory
of a larger model that passes is not an allocation problem.

`ur_die` appears in exactly those two arms across all 15 sweep configs and
nowhere else, and those are the same two arms whose faulthandler dumps contain
the a2a frame -- two independent signals agreeing.

## EP alone is not the trigger

`moe_debugmodel_ep` runs 5/5 with EP enabled. Only larger EP configs abort, so
this is size- or traffic-dependent within the a2a, not a property of enabling
expert parallelism.

## Not specific to MoE

The same `ur_die` string appears in an unrelated non-MoE `ezpz fsdp_tp`
multi-node context (ezpz issue #214). MoE's all-to-all appears to provoke a
stack-level event-handling fault rather than owning it.

## `CCL_OP_SYNC` / `CCL_ATL_SYNC_COLL` make it worse, not better

Since this is an event-handling fault, disabling the CCL sync flags looked like
a plausible lever. It is one, in the wrong direction (job `12473471`, 2N):

| config | setting | steps | memory | tps | result |
|---|---|---|---|---|---|
| `moe_2b_ep` | `=1` | 3/5 | 47.72% | 2417 | `ur_die` abort at step 3 |
| `moe_2b_ep` | `=0` | 0/5 | 0.89% | -- | hangs before step 1 |
| `moe_2b_ep` | unset | 0/5 | 0.89% | -- | hangs before step 1 |
| `moe_10b_2b_sdpa` | `=1` | 5/5 | 74.06% | 997 | pass |
| `moe_10b_2b_sdpa` | `=0` | 5/5 | 74.08% | 995 | pass |
| `moe_10b_2b_sdpa` | unset | 5/5 | 74.03% | 998 | pass |

With sync on, the EP config at least trains 3 steps before aborting; with it
off it deadlocks during init (zero step lines, 0.89% memory, killed at timeout
with a libc backtrace and no `ur_die`). A different, earlier failure.

The flags are free on healthy configs -- within 0.3% on throughput and 0.05pp
on memory -- so we keep them at 1.

## Environment

```
torch          2.13.0 (RC4, pytorch_2.13.0_patched_08_02_2026)
module         frameworks/2026.1.0
oneCCL         /opt/aurora/26.181.0/oneapi/ccl/latest
backend        XCCL
ZE_FLAT_DEVICE_HIERARCHY=FLAT
CCL_OP_SYNC=1  CCL_ATL_SYNC_COLL=1
platform       Sunspot (Intel Max 1550), 2 nodes / 24 ranks
```

## What would help

1. Whether `urEventWait` on an internal event is a known UR issue in this
   release, and whether a newer oneCCL / UR fixes it.
2. Whether there is a CCL configuration that avoids the internal-event path
   for `all_to_all_single` (we have ruled out the two sync flags).
3. Whether the abort can be surfaced as a catchable error rather than
   `ur_die` + `terminate`, which gives an application no chance to recover
   or checkpoint.

## Related

- [`known-bugs/moe-ep-a2a-degrades-with-size.md`](../../reference/known-bugs/moe-ep-a2a-degrades-with-size.md) -- full sweep and per-arm evidence
- [`intel-xpu-graphs-cannot-capture-oneccl.md`](./intel-xpu-graphs-cannot-capture-oneccl.md) -- separate capture limitation, same stack
- ezpz issue #214 -- same signature outside MoE
