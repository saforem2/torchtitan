# Sunspot: bare `reduce_scatter_tensor` SIGSEGVs at 48 ranks (2026-08-14)

> [!IMPORTANT]
> **This blocks all 80B work on Sunspot.** Any TP>1 config crashes in the first
> forward pass, because DTensor's redistribute calls `reduce_scatter_tensor`.
> The fault is in the collective itself, NOT in torchtitan, NOT in the ezpz
> experiment code, and NOT the bf16 NaN (Wall 1) we were trying to study.

## Symptom

`agpt_80b` at 4N/TP=4/LBS=1 -- a config that trained **10/10 clean steps on
2026-08-03** (job `12472452`) -- now SIGSEGVs on every rank inside step 1:

```
ezpz/trainer.py:795 train_step
  torchtitan/trainer.py:756 forward_backward_step
    models/common/decoder.py:262 forward
      protocols/module.py:292 forward_with_redistribution
        protocols/module.py:705 _redistribute_outputs
          torch/distributed/tensor/_api.py:703 redistribute
            .../placement_types.py:1311 _reduce_shard_value
              .../_functional_collectives.py:296 reduce_scatter_tensor
                torch/_ops.py:878 redispatch          <- SIGSEGV
```

Job exits rc=139. `Training starts at step 1` is printed; no step ever completes.

## Root cause: the bare collective, not our code

A ~40-line script with **no torchtitan, no DTensor, no model** -- just
`ezpz.setup_torch()` then `dist.reduce_scatter_tensor()` on a 1 KiB bf16 buffer
-- segfaults at 48 ranks (4 nodes x 12):

```
world=48 dev=xpu:0 torch=2.13.0.dev20260519+xpu
x1922c6s5b0n0-hsn0: rank 34 died from signal 11 and dumped core
Execution finished with 139.
```

**Zero collectives complete** -- it dies on the smallest probe, before any
success line prints.

Repro script: `tmp/rs_probe.py` (in the Sunspot checkout).

## Reproduced 3x, different rank each time -- NOT a bad node

| job | test | result |
|---|---|---|
| `12473114` | bare probe, 48 ranks | rank **34** died, signal 11 |
| `12473115` | bare probe, 48 ranks, fresh alloc | rank **12** died, signal 11 |
| `12473116` | bare probe, 48 ranks, fresh alloc | rank **33** died, signal 11 |

Three separate allocations, three different failing ranks, all `rc=139` /
`collectives_ok=0`. The failure moves, so excluding a node does not help.

## Not a code regression

A/B of the identical 80B config at two commits (job `12473110`, worktrees, one
allocation):

| ref | result |
|---|---|
| `01583f349` (the commit that ran clean as `12472452` on 2026-08-03) | rc=139, 60 segv, 0 steps |
| `87c783101` (HEAD) | rc=139, 60 segv, 0 steps |

Identical. The venv is also untouched since 2026-06-02
(`torch 2.13.0.dev20260519+xpu`, py 3.14.2). So neither our code nor the torch
build changed between the clean run and now.

## Environment

- torch `2.13.0.dev20260519+xpu`, python 3.14.2, `.venv` unchanged since 2026-06-02
- backend: torch built-in XCCL (`is_xccl_available() == True`)
- oneAPI CCL: `/opt/aurora/default/oneapi/ccl/latest`
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT`
- nodes drawn from `x1922c6s*` and `x1922c7s*`; all reported `free` in PBS, and
  the crash logs contain **no** Level-Zero / UR device error

## Still open

**Does it reproduce on a single node (12 ranks, intra-node, no fabric?)** The
sub-tests meant to answer this (A: 1 node; C: each node solo) have returned
`rc=127` with no log file across two attempts -- an `ezpz launch` invocation
problem on my side, not a result. That answer decides whether this is a fabric
issue or a node-local oneCCL/driver bug, and it is the first thing ALCF will
ask, so it is worth getting before filing.

## Why this is stated carefully

On 2026-08-10 a similar-looking multi-job CCL failure was reported as a cluster
fault and had to be **retracted**: the bare collective turned out to work fine,
and the real cause was elsewhere. That report inferred hardware from job
failures without testing the primitive.

This one is the opposite case -- the primitive itself was tested first, and it
is what fails. The claim here is deliberately limited to what the probe shows:
bare `reduce_scatter_tensor`, 48 ranks, smallest buffer, three allocations.
