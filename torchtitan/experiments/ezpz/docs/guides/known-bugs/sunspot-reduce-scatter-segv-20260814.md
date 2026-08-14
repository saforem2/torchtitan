# Sunspot: bare `reduce_scatter_tensor` SIGSEGVs at 48 ranks (2026-08-14)

> [!IMPORTANT]
> **RESOLVED by switching stacks (2026-08-14). Use the frameworks RC.** The
> fault is in the oneAPI **2025.3.1** oneCCL that the June `.venv` links
> against. On `frameworks/2026.1.0` (oneCCL 26.181.0 + torch
> `2.13.0a0+gitcf30153`, the RC4 wheelforge conda env) all three collectives
> run clean -- see "The RC fixes it" below. Everything after that point is the
> diagnosis of the broken stack, kept because the underlying oneCCL bug is
> still real and worth reporting.
>
> **This blocks all 80B work on Sunspot when using the June `.venv`.** Any TP>1 config crashes in the first
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

## Reproducer (verified, torch-only)

[`tests/repro_reduce_scatter_segv.py`](../../../tests/repro_reduce_scatter_segv.py)
-- no ezpz, no torchtitan, no model. One node, ~10 seconds:

```bash
mpiexec -n 12 -ppn 12 python3 repro_reduce_scatter_segv.py
```

Verified to reproduce (job `12473121`):

```
node: x1922c6s0b0n0
world=12 torch=2.13.0.dev20260519+xpu dev=xpu:0
x1922c6s0b0n0-hsn0: rank 4 died from signal 11 and dumped core
rc=139
```

A healthy stack prints five `OK reduce_scatter` lines then `ALL_PASSED`. Here
**no OK line is ever reached** -- it dies on the first 1 KiB collective.

The script aborts loudly if it cannot detect the rank environment
(`world <= 1`). That guard exists because an earlier version read
`PMI_RANK`/`PMI_SIZE`, which PALS does **not** set: every process silently came
up as its own 1-rank job, they collided on the rendezvous port, and the run
failed with `EADDRINUSE` having executed no cross-rank collective at all --
a failure that superficially resembled a result. PALS sets `PALS_RANKID`,
`PALS_LOCAL_RANKID`, and `PALS_LOCAL_SIZE` (confirmed in job `12473120`).

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

## When it started

The last Sunspot job that produced training steps is **2026-08-08** (job
`12472766`, mix-cosmo, 1600 clean steps). Every job since is one of my own
80B/diagnostic runs, all of which fail.

Two things that does NOT establish, and should not be read as establishing:

- **It does not date the fault to 08-08.** Nobody ran a non-80B training job
  on Sunspot in that window, so the absence of successes is an absence of
  attempts, not evidence of breakage. The `/tegu/` mounts were also down for
  part of it.
- **It does not prove all training is broken.** Only that *these* jobs are.
  Whether `all_reduce` / `all_gather` still work -- i.e. whether plain
  FSDP/DDP without TP is still viable on Sunspot -- is being measured
  separately (job `12473134` controls). Until that reports, the safe claim is
  narrow: **`reduce_scatter_tensor` is broken, therefore TP>1 is broken.**

## Environment

- torch `2.13.0.dev20260519+xpu`, python 3.14.2, `.venv` unchanged since 2026-06-02
- backend: torch built-in XCCL (`is_xccl_available() == True`)
- oneAPI CCL: `/opt/aurora/default/oneapi/ccl/latest`
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT`
- nodes drawn from `x1922c6s*` and `x1922c7s*`; all reported `free` in PBS, and
  the crash logs contain **no** Level-Zero / UR device error

## It is NODE-LOCAL, not a fabric problem

Job `12473118`, all 12 ranks pinned to a single node (`-ppn 12`, no inter-node
traffic whatsoever):

```
RESULT A-1node-x1922c6s0b0n0  rc=143  collectives_ok=0  segv=1  rank 8 died from signal 11
RESULT B-4node-48r            rc=139  collectives_ok=0  segv=1  rank 18 died from signal 11
```

**A single node reproduces it.** The HSN fabric, inter-node routing, and
multi-node CCL transport are all excluded. This is a node-local oneCCL / Level
Zero / driver fault in `reduce_scatter_tensor` on XPU, reproducible with 12
ranks on one machine and a 1 KiB buffer.

That also makes the repro cheap for ALCF: **1 node, 1 file, ~10 seconds.**

Full reproduction tally -- 5 allocations, 5 different failing ranks:

| job | scale | failing rank |
|---|---|---|
| `12473114` | 4 nodes / 48 ranks | 34 |
| `12473115` | 4 nodes / 48 ranks | 12 |
| `12473116` | 4 nodes / 48 ranks | 33 |
| `12473117` | 4 nodes / 48 ranks | 18 |
| `12473118` | **1 node / 12 ranks** | **8** |
| `12473121` | **1 node / 12 ranks, standalone torch-only repro** | **4** |

## Note on the `--hostfile` void runs

Sub-tests that passed `--hostfile` returned `rc=127` with no log across jobs
`12473115/116/117`. That was mine, not a cluster symptom: the hostfiles were
written with short names after the FQDN was stripped. `-ppn` needs no hostfile
and works. Mentioned only so the `rc=127` lines in those job outputs are not
mistaken for evidence.

## The RC fixes it (job `12473138`)

Same node, same 12 ranks, same probe -- only the software stack differs:

| stack | reduce_scatter | all_gather | all_reduce |
|---|---|---|---|
| `.venv` (torch `2.13.0.dev20260519+xpu`, oneAPI 2025.3.1) | **SIGSEGV** | passes | **SIGSEGV** |
| RC (torch `2.13.0a0+gitcf30153`, oneAPI 2026.1.0) | **passes** | passes | **passes** |

All five buffer sizes to 144 MiB/rank, rc=0, zero segfaults.

**End-to-end confirmation (job `12473139`):** 80B at 4N/TP=4/LBS=1 -- the exact
config that SIGSEGVs on the `.venv` -- trains on the RC:

```
step:  1  loss: 12.93393  grad_norm: 9.8457  memory: 32.43GiB(50.68%)
step:  2  loss: 12.90509  grad_norm: 8.6538  memory: 44.95GiB(70.25%)
step:  3  loss: 12.80237  grad_norm: 8.5700  memory: 44.95GiB(70.25%)
step:  4  loss: 12.64922  grad_norm: 8.1765  memory: 44.95GiB(70.25%)
```

Loss descends, zero segfaults, and memory plateaus at **70.25%** -- matching
the known-good 4N/TP=4 profile from job `12472452` (2026-08-03) exactly.

**How to use it** (the `frameworks/2026.1.0` module alone has NO torch -- that
is what aborted job `12473137`):

```bash
module use /opt/aurora/26.181.0/modulefiles
module load frameworks/2026.1.0          # provides oneAPI 2026.1.0 + oneCCL
source venvs/fw-2026.1-rc2/bin/activate  # RC4 wheelforge conda env + torchtitan deps
unset PYTHONPATH
```

Gate on `import torchtitan.experiments.ezpz.agpt.config_registry`, never on
`import agpt` -- `config/manager.py` masks every missing dependency as one
generic error, so the latter is a false pass.

## It is the REDUCTION, not collectives in general (job `12473135`)

Same node, same 12 ranks, same buffer sizes, same harness -- only the op differs:

| collective | result |
|---|---|
| `all_gather_into_tensor` | **PASSES** all 5 sizes to 144 MiB/rank, rc=0 |
| `reduce_scatter_tensor` | SIGSEGV, 0 collectives |
| `all_reduce` | SIGSEGV, 0 collectives |

The two that fail both perform an **on-device reduction**; `all_gather` only
moves bytes. That points at the reduction kernel / its Level-Zero dispatch,
not at the transport, the fabric, or CCL's communicator setup -- consistent with
`all_gather` exercising the same transport successfully at 144 MiB/rank.

**Scope for us:** this is worse than TP-only. `all_reduce` is what DDP and
FSDP's gradient reduction use, so **all multi-rank training on Sunspot is
affected**, not just TP>1. It is not merely an 80B problem.

## No workaround among the standard CCL knobs (job `12473135`)

All ten variants still segfault:

| variant | result |
|---|---|
| baseline | segv |
| `CCL_ATL_TRANSPORT=ofi` / `=mpi` | segv |
| `CCL_ZE_IPC_EXCHANGE=sockets` / `=pidfd` | segv |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` | segv |
| `CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL=0` | segv |
| `CCL_WORKER_COUNT=1` | segv |
| `CCL_ZE_ENABLE=0` | not a pass -- refuses to build a communicator at all (`ze_data was not initialized`) |
| `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` | **not actually tested** -- COMPOSITE exposes 6 devices/node, so the 12-rank request was rejected (`ngpus must be > 0 and <= 6`) before any collective ran. Retested separately at 6 ranks (job `12473136`). |

## Draft ALCF ticket

> **Subject:** Sunspot: reducing collectives (`reduce_scatter`, `all_reduce`)
> SIGSEGV on XPU -- single node, 12 ranks; `all_gather` is fine
>
> On Sunspot, `torch.distributed.reduce_scatter_tensor` and
> `torch.distributed.all_reduce` both segfault on a 1 KiB bf16 buffer with 12
> ranks on a single node. No fabric is involved; it reproduces intra-node.
>
> **`all_gather_into_tensor` on the identical harness passes all sizes up to
> 144 MiB/rank.** The two failing ops perform an on-device reduction; the
> passing one only moves bytes -- which suggests the reduction kernel or its
> Level Zero dispatch rather than the transport or communicator setup.
>
> This blocks **all** multi-rank training on Sunspot, since `all_reduce` is
> what DDP and FSDP gradient reduction rely on -- not only tensor-parallel jobs.
>
> None of the usual knobs help: `CCL_ATL_TRANSPORT=ofi|mpi`,
> `CCL_ZE_IPC_EXCHANGE=sockets|pidfd`,
> `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`,
> `CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL=0`, `CCL_WORKER_COUNT=1` all still
> segfault (job `12473135`).
>
> Reproducer (torch only, ~10s, attached / at
> `torchtitan/experiments/ezpz/tests/repro_reduce_scatter_segv.py`):
>
> ```bash
> mpiexec -n 12 -ppn 12 python3 repro_reduce_scatter_segv.py
> ```
>
> Output (job `12473121`, node `x1922c6s0b0n0`):
>
> ```
> world=12 torch=2.13.0.dev20260519+xpu dev=xpu:0
> x1922c6s0b0n0-hsn0: rank 4 died from signal 11 and dumped core
> ```
>
> Expected: five `OK reduce_scatter` lines, then `ALL_PASSED`. Observed: dies
> on the first, smallest collective; no OK line is reached.
>
> Reproduced in 6 independent allocations (jobs `12473114`, `12473115`,
> `12473116`, `12473117`, `12473118`, `12473121`) with a different failing rank
> each time (34, 12, 33, 18, 8, 4), at both 48 ranks / 4 nodes and 12 ranks /
> 1 node -- so it is not a single bad node.
>
> Environment: torch `2.13.0.dev20260519+xpu`, python 3.14.2, XCCL backend,
> oneAPI CCL `/opt/aurora/default/oneapi/ccl/latest`,
> `ZE_FLAT_DEVICE_HIERARCHY=FLAT`. Nodes seen: `x1922c6s*`, `x1922c7s*`, all
> reporting `free` in PBS with no Level-Zero/UR device errors in the logs.
> The venv is unchanged since 2026-06-02.
>
> Impact: this blocks all tensor-parallel training on Sunspot, including our
> 80B AuroraGPT runs -- DTensor's redistribute calls `reduce_scatter_tensor`,
> so any TP>1 job dies in the first forward pass. The same 80B config trained
> cleanly on 2026-08-03 (job `12472452`).

## Why this is stated carefully

On 2026-08-10 a similar-looking multi-job CCL failure was reported as a cluster
fault and had to be **retracted**: the bare collective turned out to work fine,
and the real cause was elsewhere. That report inferred hardware from job
failures without testing the primitive.

This one is the opposite case -- the primitive itself was tested first, and it
is what fails. The claim here is deliberately limited to what the probe shows:
bare `reduce_scatter_tensor`, 48 ranks, smallest buffer, three allocations.
