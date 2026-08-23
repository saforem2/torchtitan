# MoE under EP aborts in `all_to_all_single`, worse with model size (2026-08-19)

> [!IMPORTANT]
> **No EP config above `debugmodel` completes, and the failures split into
> TWO causes -- do not attribute them all to a2a.** Per-arm signatures
> (`12473367`):
>
> | config | steps | mem | a2a in dump | signal / error |
> |---|---|---|:--:|---|
> | `debugmodel_ep` | 5/5 | 44% | -- | pass |
> | `2b_ep` | 3/5 | 48% | **yes** | SIGABRT |
> | `10b_2b_sdpa_ep` | 2/5 | 80% | **yes** | SIGABRT |
> | `10b_2b_sdpa_bmm_ep` | 1/5 | 86% | no | `UR_RESULT_ERROR_OUT_OF_RESOURCES` |
> | `7b_ep` | 1/5 | 94% | no | SIGTERM only |
> | `hybridep` | 0/5 | -- | no | never trained |
>
> **The a2a SIGABRT is real but covers only `2b_ep` and `10b_2b_sdpa_ep`.**
> For those two the claim holds strongly: `2b_ep` dies at 48% memory while the
> *larger* `moe_10b_2b_sdpa` passes at 74% and `bmm` passes at 94.5%, so an
> allocation ceiling cannot explain it. `bmm_ep` and `7b_ep` show no a2a frame
> and look like the same resource ceiling as non-EP `7b`.
>
> An earlier version of this page asserted "EP failure is NOT memory" across
> all arms. That overstated a result drawn from `2b_ep` alone.

## Update 2026-08-19: the abort is an Intel UR fault, not our dispatch code

The two a2a arms abort with a specific runtime message, visible just above
the faulthandler dump:

```
ur_die: urEventWait must not be called for an internal event
terminate called without an active exception
Fatal Python error: Aborted
```

That is Intel's Unified Runtime aborting the process, not a torchtitan
assertion and not a CCL error. Three things follow:

1. **EP is not the trigger.** `moe_debugmodel_ep` runs 5/5 with EP enabled.
   Only the larger EP configs abort.

2. **`ur_die` correlates exactly with the a2a arms.** Grepping all 15 sweep
   configs, `ur_die` appears in `2b_ep` and `10b_2b_sdpa_ep` and nowhere
   else -- the same two arms whose faulthandler dumps contain the a2a frame.
   The split in the table above is therefore backed by two independent
   signals, not one.

3. **It is not our code path.** The same `ur_die` string shows up in an
   unrelated non-MoE `ezpz fsdp_tp` multi-node context (ezpz issue #214), so
   this is a stack-level fault that MoE's a2a happens to provoke.

Memory confirms the separation cleanly. The two `ur_die` arms sit at
**47.77%** and **79.74%** -- nowhere near a ceiling -- while the arms without
`ur_die` sit at **86.53%** (`bmm_ep`) and **93.98%** (`7b_ep`). Non-EP
`moe_7b` fails the same way at **95.40%**, which is the control that shows
EP has nothing to do with that second failure mode.

`hybridep` is a third, unrelated thing and is now closed as WONTFIX -- it is
NVIDIA-only by design. See
[hybridep-is-nvidia-only.md](./hybridep-is-nvidia-only.md).

## CCL_OP_SYNC / CCL_ATL_SYNC_COLL: keep them at 1 (2026-08-20)

Both are set to `1` in the standard recipe. Since the abort is an
event-handling fault, turning them off looked like a plausible lever -- and it
is, in the wrong direction. Job 12473471, 2N, 5 steps:

| config | setting | steps | memory | tps | result |
|---|---|---|---|---|---|
| `moe_2b_ep` | `=1` | 3/5 | 47.72% | 2417 | `ur_die` abort at step 3 |
| `moe_2b_ep` | `=0` | 0/5 | 0.89% | -- | **hangs before step 1** |
| `moe_2b_ep` | unset | 0/5 | 0.89% | -- | **hangs before step 1** |
| `moe_10b_2b_sdpa` | `=1` | 5/5 | 74.06% | 997 | PASS |
| `moe_10b_2b_sdpa` | `=0` | 5/5 | 74.08% | 995 | PASS |
| `moe_10b_2b_sdpa` | unset | 5/5 | 74.03% | 998 | PASS |

Two conclusions:

1. **Turning sync off makes the EP config worse, not better.** With `=1` it at
   least trains 3 steps before aborting; with `=0` or unset it deadlocks
   during init -- zero step lines, 0.89% memory, killed at the timeout with a
   libc backtrace and *no* `ur_die` in the log. Different failure, earlier.
   `=0` and unset behave identically here.
2. **They are free on healthy configs.** `moe_10b_2b_sdpa` is within 0.3% on
   throughput (997 / 995 / 998 tps) and 0.05pp on memory across all three
   settings. So the sync flags are not costing measurable performance, and
   there is no reason to drop them.

Control: `agpt_20b` under `spmd_types` fails identically with the flags on and
unset, as expected -- that is a missing FSDP code path, unrelated to
collectives.

**Actionable:** report the `ur_die: urEventWait must not be called for an
internal event` signature to Intel with the a2a repro; it is far more
specific than "SIGABRT in all_to_all_single" and is the right thing to put
in the ticket.

> [!NOTE]
> **Written up and ready to file:**
> [`upstream-issues/intel-ur-die-urEventWait-a2a.md`](../../outbound/upstream-issues/intel-ur-die-urEventWait-a2a.md)
> (2026-08-20). Needs an ALCF/Intel account to submit.

Measured on the post-79th-sync tree (job `12473367`, 2N, LBS=1, 5 steps,
compile off):

| config | steps | memory | |
|---|---|---|---|
| `moe_debugmodel_ep` | 5/5 | 44% | pass |
| `moe_2b_ep` | **3/5** | 48% | SIGABRT |
| `moe_10b_2b_sdpa_ep` | **2/5** | -- | SIGABRT |
| `moe_7b_ep` | **1/5** | 94% | SIGABRT |

The non-EP counterparts pass at every size except 7b:

| config | steps | memory | |
|---|---|---|---|
| `moe_debugmodel` | 5/5 | 44% | pass |
| `moe_2b` | 5/5 | 48% | pass |
| `moe_10b_2b_sdpa` | 5/5 | **74%** | pass |
| `moe_10b_2b_sdpa_bmm` | 5/5 | **94.5%** | pass |
| `moe_7b` | 1/5 | 95% | `UR_RESULT_ERROR_OUT_OF_RESOURCES` |

**Two distinct failures, do not conflate them:**

- **7b (both variants): a size ceiling.** 94-95% memory,
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`. `moe_10b_2b_sdpa_bmm` also runs at 94.5%
  and passes, so this is marginal rather than categorical -- 7b is simply over
  the line at 2N.
- **EP at any size: the a2a dispatch.** Aborts with **signal 6 and no Python
  traceback**; the faulthandler dump points at

      torch/distributed/_functional_collectives.py:583 in all_to_all_single
      experiments/ezpz/moe/token_dispatcher.py:757 in dispatch

  This is the previously-recorded a2a-under-EP SIGABRT (the line moved from
  712 to 757 with upstream churn).

## Reading the failure

A SIGABRT with no Python exception is easy to misread. Useful discriminators
gathered while triaging this:

- **signal 6 (SIGABRT)** with a faulthandler dump -> look for the collective
  in the dump, not for a Python traceback; there will not be one.
- **signal 9 (SIGKILL)** -> HOST OOM, and the reported *device* memory at that
  moment is meaningless (seen on `moe_4b` under `--debug.deterministic`).
- **`OutOfMemoryError`** -> genuine device OOM.
- **`UR_RESULT_ERROR_OUT_OF_RESOURCES` (level_zero error 40)** -> device
  RESOURCE exhaustion (events, command lists), which is *not* the same as
  device memory: the 30B HSDP case hit it at 71.68% with 18 GiB free.

## Not caused by the 79th sync

All of these were tested on the merged tree, but the sync's own fixes
(`c23a544bc` rope annotation, `ae180b3e7` mesh info) took seven MoE configs
from "all fail identically on rope" to passing. The EP aborts are the
pre-existing issue that was previously masked by the earlier failure.

## Not tested

Whether EP degrades with *node count* as well as model size -- everything here
is 2N. If the a2a payload scales with either, that would distinguish a message
size limit from a rank-count one.
