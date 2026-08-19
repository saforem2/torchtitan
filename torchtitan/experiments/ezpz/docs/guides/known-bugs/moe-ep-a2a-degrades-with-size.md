# MoE under EP aborts in `all_to_all_single`, worse with model size (2026-08-19)

> [!IMPORTANT]
> **Every EP config degrades monotonically with model size, and it is NOT
> memory.** `moe_2b_ep` fails at 48% device memory while the *larger*
> `moe_10b_2b_sdpa` passes clean at 74%. The failure is in the expert-parallel
> a2a dispatch path, not an allocation ceiling.

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
