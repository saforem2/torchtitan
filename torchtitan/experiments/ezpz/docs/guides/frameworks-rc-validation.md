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
| 80B TP=4 @ 4N | *(running)* | | |

## Fixed by the RC

- **Reducing collectives.** `reduce_scatter_tensor` and `all_reduce` SIGSEGV on
  oneAPI 2025.3.1 (12 ranks, one node, 1 KiB buffer) and run clean on the RC.
  This blocked *all* multi-rank training. Full diagnosis:
  [sunspot-reduce-scatter-segv-20260814](known-bugs/sunspot-reduce-scatter-segv-20260814.md).
- **`torch.compile` at TP=4** -- see the table above.
- **MoE EP=1 above 2N.**

## NOT fixed by the RC -- still open

The RC is a collectives/compiler fix. It does **not** touch the numerics walls:

- **80B bf16 NaN (Wall 1).** A deep-residual bf16 overflow, optimizer-independent
  (SophiaG @512N step-14, mano @62N step-17, identical signature). fp32-residual
  prototypes moved the wall but did not break it (per-block NaN'd at step 19,
  full-depth at step 38). Under test on the RC right now (job `12473142`,
  600 steps at dp=192).
- **The dp ceiling above 264 is unmapped.** Only dp-axis NaN on record is
  dp=372, and that run was TP=2 (confounded). Production wants dp=768-1536.
  Sunspot caps at dp=276 at TP=4, so this needs Aurora.
- **QK-norm TP=4 backward** (`tensor does not have a device`) -- not retested.
- **Pipeline parallelism** -- a torch version floor, unrelated; not retested.

## Every performance number in our docs is stale

The dp=192/264 clean runs, the NaN step numbers, the MFU and throughput tables,
the scaling studies -- all measured on oneAPI 2025.3.1. They are **not** valid
for the RC and should not be quoted for the Aurora rollout without a re-baseline.
A throughput regression here would be easy to miss and expensive at scale.
