# Frameworks RC (oneAPI 2026.1.0) -- validation status

> [!IMPORTANT]
> **The RC is the target stack**, not a workaround -- it is the build heading to
> Aurora. Every finding in our docs was measured on the June `.venv`
> (torch `2.13.0.dev20260519+xpu` / oneAPI 2025.3.1), so each is **unverified
> on the RC until re-run**. This page tracks that re-validation.

**Stack:** `frameworks/2026.1.0` (oneAPI 2026.1.0, oneCCL `26.181.0`) +
torch `2.13.0a0+gitcf30153` (RC4 wheelforge conda env).

```bash
module use /opt/aurora/26.181.0/modulefiles
module load frameworks/2026.1.0          # oneAPI + oneCCL. Provides NO torch.
source venvs/fw-2026.1-rc2/bin/activate  # RC4 conda env + torchtitan deps
unset PYTHONPATH
```

Gate on `import torchtitan.experiments.ezpz.agpt.config_registry`, never on
`import agpt` -- `config/manager.py` masks every missing dependency behind one
generic error, so the latter is a false pass.

## Results (job `12473146`, Sunspot, 2-4N, 10 steps each)

| case | verdict | final loss | note |
|---|---|---:|---|
| 2B eager | **PASS** | 11.06 | baseline |
| 2B compile TP=1 | **PASS** | 8.94 | |
| 2B compile **TP=4** | **PASS** | 8.74 | the fw-RC `assert_size_stride` regression is **fixed at 4N** (previously only claimed at 2N) |
| 2B HSDP (shard=12, replicate=2) | **PASS** | 11.73 | |
| MoE EP=1 (`moe_10b_2b_sdpa`) | **PASS** | 9.29 | first confirmation **above 2N**; this config SIGABRT'd on the old stack |
| 80B TP=4 @ 4N | **PASS** | 11.26 | below the NaN regime |

## Fixed by the RC

- **Reducing collectives.** `reduce_scatter_tensor` and `all_reduce` SIGSEGV on
  oneAPI 2025.3.1 (12 ranks, one node, 1 KiB buffer) and run clean on the RC.
  This blocked *all* multi-rank training. Full diagnosis:
  [sunspot-reduce-scatter-segv-20260814](known-bugs/sunspot-reduce-scatter-segv-20260814.md).
- **`torch.compile` at TP=4** -- see the table above.
- **MoE EP=1 above 2N.**

## NOT fixed by the RC -- still open

The RC is a collectives/compiler fix. It does **not** touch the numerics walls:

- **80B bf16 NaN (Wall 1) -- CONFIRMED STILL PRESENT ON THE RC.** Job
  `12473142`, dp=192, bf16: `grad_norm=nan` at **step 30**, loss NaN from 31.
  The nan-abort guard fired at 35 and reclaimed the walltime.

  ```
  step: 28  loss: 11.77  grad_norm: 8.4614
  step: 29  loss: 11.69  grad_norm: 8.6428
  step: 30  loss: 12.12  grad_norm:     nan
  ```

  Step 30 sits squarely in the historical ladder (14 / 17 / 19 / 38), so the
  wall is **stack-independent** -- which strengthens the root cause (a deep
  bf16 residual overflow) rather than weakening it.

  One difference from the recorded signature: grad_norm was **rising** into the
  failure (8.06 -> 8.29 -> 8.46 -> 8.64 -> nan), not dead-flat then instant-NaN.
  Worth noting for diagnosis. LR was only ~1.5e-7 at step 30 (warmup 200), so
  this is not an LR-ceiling effect.
- **The dp ceiling above 264 is unmapped.** Only dp-axis NaN on record is
  dp=372, and that run was TP=2 (confounded). Production wants dp=768-1536.
  Sunspot caps at dp=276 at TP=4, so this needs Aurora.
- **QK-norm TP=4 backward** (`tensor does not have a device`) -- not retested.
- **Pipeline parallelism** -- a torch version floor, unrelated; not retested.

## fp32 activations at dp=192 -- what it revealed (job `12473149`)

Running the SAME config with `--training.mixed-precision-param=float32` at the
dp where bf16 dies exposes how different the two numerically are:

| step | bf16 grad_norm | fp32 grad_norm | bf16 loss | fp32 loss |
|---:|---:|---:|---:|---:|
| 5 | 8.33 | **17,418** | 15.46 | 12.84 |
| 6 | 8.01 | **80,210** | 12.88 | 12.79 |
| 7 | 7.86 | **91,467** | 12.90 | 12.75 |
| 11 | 8.05 | **80,065** | 12.76 | **12.39** |

**bf16 is not mis-reporting a large gradient -- it is computing a different,
smaller one.** bf16 has fp32's full exponent range (max ~3.4e38), so 18,000 is
trivially representable; the 65504 ceiling is *fp16*. What bf16 loses is its
**7-bit mantissa**, which destroys the small-magnitude contributions whose
accumulation across an 84-layer residual stream produces these spikes. This is
exactly the mantissa-not-exponent mechanism the root-cause report describes.

Two consequences:

- **"grad_norm ~= 8" in every historical 80B bf16 log is a real value for a
  degraded gradient**, not a masked reading of the true one. Treat those logs as
  measuring a different optimization problem than the one we intend to solve.
- **Gradient clipping is not broken.** `max_norm=1.0` scales by `1/total_norm`
  using the norm computed *in that run*; the bf16 run genuinely has norm ~8. The
  gradients are wrong, not the clipping. (An earlier draft of this page claimed
  clipping was misbehaving by ~2000x -- that was wrong and is corrected here.)
- **fp32 learns faster**, which is independent corroboration: at step 11 fp32 is
  at loss 12.39 vs bf16 12.76, and accelerating. If bf16 merely mis-reported a
  correct gradient, both would descend identically.

### It CLEARS the wall -- 120/120 steps, zero NaN (COMPLETE)

**Final: `rc=0, steps=120, nan_lines=0`, loss 12.95 -> 8.098.** The run
completed its full budget at production dp with no NaN, through full LR.

bf16 died at step 30 at this dp. fp32 walked straight through it:

```
step: 28  loss: 10.55757  grad_norm: 15.6207
step: 29  loss: 10.39867  grad_norm: 11.2650
step: 30  loss: 10.28401  grad_norm:  7.8810   <- bf16 NaN'd here
step: 31  loss: 10.17815  grad_norm:  9.0314
step: 32  loss: 10.08726  grad_norm:  9.7592
```

**And it does so at a HIGHER learning rate.** The fp32 run uses warmup=40, so at
step 30 it is at LR 7.5e-7; the bf16 run used warmup=200 and failed at only
1.5e-7 -- 5x lower. The comparison is therefore conservative: fp32 survives a
strictly harder condition than the one that killed bf16. (The two runs are not
LR-matched, so this is not a controlled A/B on LR; it is a one-sided result --
fp32 clears a bar bf16 could not, with margin to spare.)

Note also that the grad_norm spikes decay as training settles: 17K-91K over
steps 2-12, then 13-180 by step 20, then single digits by step 30. The early
spikes are a startup transient, not a persistent regime.

Cost, measured at dp=192: memory 36.5 GiB (57%) vs bf16 20.5 GiB (32%);
throughput **16 tps vs 54 -- a 3.4x slowdown**, matching the documented 3-5x.

## Every performance number in our docs is stale

The dp=192/264 clean runs, the NaN step numbers, the MFU and throughput tables,
the scaling studies -- all measured on oneAPI 2025.3.1. They are **not** valid
for the RC and should not be quoted for the Aurora rollout without a re-baseline.
A throughput regression here would be easy to miss and expensive at scale.
