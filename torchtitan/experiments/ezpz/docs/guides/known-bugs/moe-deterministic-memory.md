# `--debug.deterministic` costs ~27% device memory on the MoE path (2026-08-19)

> [!IMPORTANT]
> **`--debug.deterministic` is not incompatible with MoE -- it is expensive.**
> Measured on `moe_2b`, paired within one job: **29.99 GiB without the flag,
> 38.18 GiB with it (+8.19 GiB, +27%).** Configs that already sit near the
> ceiling therefore OOM, and the failure looks nothing like a memory problem
> in the logs.

## Measurements (job `12473336`, 2N, EP=12, LBS=1, 5 steps)

| config | num_experts | det=0 | det=1 |
|---|---:|---|---|
| `moe_debugmodel` | 8 | FAIL (my flag, see below) | FAIL (same) |
| `moe_500m` | 8 | FAIL (my flag) | FAIL (same) |
| **`moe_2b`** | 24 | **5/5, 29.99 GiB** | **died step 2, 38.18 GiB** |
| `moe_4b` | 24 | 5/5, 31.39 GiB | died in compile, host SIGKILL |

Two different failure modes under `det=1`:

- **`moe_2b`: device OOM.** +8.19 GiB of XPU memory over the identical run.
- **`moe_4b`: HOST OOM (signal 9), during compilation, before step 1.** Device
  memory read only 0.92 GiB, which looks like "not a memory problem" and is
  exactly why this was misdiagnosed twice. Deterministic mode disables autotune
  caching, so 12 ranks per node each compile more, simultaneously, and the
  kernel OOM-killer takes the process.

Read `signal 9` as host RAM and `OutOfMemoryError` as device RAM. The device
memory figure at the moment of a host kill is meaningless.

## Consequence for verification runs

CLAUDE.md requires bit-identical loss under `--debug.seed=42
--debug.deterministic` for non-computation changes. On the MoE path that is
only affordable for configs with real headroom:

- **`moe_10b_2b_sdpa` cannot be run deterministically** -- it sits at 57/64 GiB
  already, and +8 GiB does not fit. Verify it by loss comparison instead, and
  say so rather than implying a bit-exact check.
- **`moe_2b` at reduced sequence length** is the config to use when a genuine
  bit-exact MoE check is needed.

## Not caused by any of these (each was checked)

- **Not HSDP.** Pure FSDP (`dp-shard=24`) fails identically to
  `dp-replicate=2`. The 2026-05-21 slides attribute an `aten.normal_` failure
  to HSDP; that attribution does not survive this grid.
- **Not the MoE path being broken.** `moe_10b_2b_sdpa` trains fine without the
  flag -- 6/6 steps, reproduced from the user's own command line.
- **Not sequence length.** `--training.seq-len=2048` failed only when the RC
  environment variables were missing (see below); with them it is 6/6 clean.

## Method notes -- three errors that cost time here

1. **Missing RC environment.** My scripts loaded the right conda env but not
   `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `CCL_OP_SYNC=1`,
   `CCL_ATL_SYNC_COLL=1`. Without them the `seq2048` arm SIGABRT'd and
   produced an `aten.normal_` error that vanished once the vars were set. Any
   ezpz run on this stack needs them; every earlier script sets them.
2. **`--parallelism.expert-parallel-degree=12` against 8-expert models.**
   `moe_debugmodel` and `moe_500m` have `num_experts: 8`, which cannot shard
   across 12 ranks -- `RuntimeError: Split sizes doesn't match total dim 0
   size`. That is an invalid flag, not a determinism result. Check
   `num_experts % ep_degree == 0` before reading anything into a small-model
   failure.
3. **Non-interpolating log filenames.** `label="$cfg-det$det"` collapsed each
   det=0/det=1 pair into one file, so the det=0 logs were overwritten and only
   the console table survived. Quote or brace the variable.
