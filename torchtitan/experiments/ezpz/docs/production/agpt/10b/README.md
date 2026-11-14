# Production Training — agpt 10B (9.48B), OLMo-3 tokenizer, 4K

> **Last updated:** 2026-11-14. Steps, checkpoints and queue state below were
> read from disk, logs and the schedulers on 2026-11-14.

The agpt-v2 9.48B model: `10b_olmo2tok` geometry (dim 4096, 48 layers, 32 heads,
8 KV heads, vocab 100352), OLMo-3 tokenizer (byte-identical to OLMo-2-1124-7B),
4096-token sequences, text tokenized on the fly by Grain. Its 4.64B twin is
[agpt 4B](../4b/README.md).

## Status

| Machine | Config | Shape | Head | Now |
|---|---|---|---|---|
| Aurora | `agpt_v2_10b_4k_olmo_mix` | 512N, GBS 6144 | none | queued, `9076371` (fresh start) |
| Sirius | `agpt_10b_olmo3_polaris` | 8N (32 GPUs), GBS 32 | `step-100` | running, `33527` (fresh, step 197 at `15:44 UTC`) |
| Polaris | `agpt_10b_olmo3_polaris` | 16N / 32N | -- | queued since Nov 9, `7832132` / `7832133` |

No 10B run has a checkpoint past `step-100` yet.

## Aurora: never trained at scale

| | |
|---|---|
| Config | `agpt_v2_10b_4k_olmo_mix` (`10b_olmo2tok_dp12`: hidden 11520 for even FSDP sharding) |
| Data | raw olmo-mix-1124 JSON; queued job reads `olmo-mix-dp6144.txt` (sorted), see below |
| Optimizer | AdamW `7e-5`, 2000 warmup. Scaled from the LR finder, not validated at a fixed LR. |
| Checkpoints | `agpt-v2-10b-4k-olmo-mix-n768-20261109/` is empty; the queued job writes `agpt-v2-10b-4k-olmo-mix-n512-20261111/` |
| W&B | `12fv8yur`, `8qp5o62y`, `uzp907ru`, `znmzohyl` (the failed 768N attempts) |

| Job | Dates | Shape | Outcome |
|---|---|---|---|
| `9055264` | -- | 2N | step-1 loss 11.99, the expected value |
| `9053470` seat 4 (umbrella) | Nov 9-11 | 768N, 9216 ranks, GBS 9216 | step-1 loss **17788**, **24202**, **38478** on three fresh attempts, then a oneCCL scale-out allreduce failure (`atl_comm->wait fails`) and `FAILOVER STOP: deterministic_failure`; two more attempts stuck before training |
| `9076371` | queued Nov 11 | 512N, 6144 ranks, GBS 6144 | fresh start |

- Unverified inference from the umbrella write-up: oneCCL scale-out allreduce is
  broken at 9216 ranks on this stack. The 4B trained normally at 6144 ranks, so
  the 10B moved to 512N.
- At 768N every attempt spent ~53 min on one `/home` HF datasets lock before
  step 1. The resubmission uses a per-node `/tmp` cache.
- `9076371` was submitted before the manifest shuffle fix and still reads the
  sorted manifest. To start on the shuffled order, resubmit with
  `-v MULTI_ONLY=4,MANIFEST_V2_10B=/flare/AuroraGPT/foremans/scratch/umbrella-v2-manifests/olmo-mix-dp6144-shuf0.txt`
  before it starts
  ([umbrella write-up](../../2026-11-11-umbrella-2088n.md#resubmission-2026-11-11)).

## Sirius: 8N hedge on Dolma

Full detail: [Sirius hedge report](../sirius-olmo3-5b-10b.md).

| | |
|---|---|
| Config | `agpt_10b_olmo3_polaris` (Dolma raw JSON via Grain) |
| Optimizer | AdamW `4.64e-4` (GBS=6144 finder basin) |
| Batch | 32 GPUs, GBS 32 x 4096 = 131K tokens/step |
| FSDP | FFN 11264, `% 32 == 0` |

| Job | Steps | Loss | Outcome |
|---|---|---|---|
| `33469` | `1 -> 200` | `12.00 -> 6.38` | `E 143`, no spare node (`FAILOVER STOP exhausted`); `step-100` saved |
| `33488` | `200` steps | `12.00 -> 6.19` | Gloo crash on `x3200c0s13b0` at step 200; resubmit rejected (allocation spent) |
| `33526` | fresh `1 -> 2` | `12.01 -> 11.86` | stopped |
| `33527` | fresh `1 -> 200` | `12.03 -> 6.33` | died at step 200 |
| `33528` | fresh `1 -> 19` | `12.02 -> 8.64` | stopped |
| `33529` | fresh `1 -> 200` | `12.01 -> 6.30` | died at step 200 |
| `33537` | fresh `1 -> 197+` | `12.02 -> 6.35` | running Nov 14 18:43 |

**Every attempt dies at step 200.** `33469`, `33488`, `33527` and `33529` each reached
exactly step 200 and stopped with `FAILOVER STOP`. In `33529` the traceback is in
the async checkpoint path (`_wait_for_saving` -> DCP `save` -> `all_reduce("write")`
-> Gloo `gather_object`): `Connection closed by peer`, then other ranks exit 1 or die
on signal 15. Each restart begins at step 1 because the earlier `step-100` was left
corrupt, so the run never gets past 200. Cause not established; one candidate is
host memory while staging the ~100 GB checkpoint. A synchronous-checkpoint run would
show whether the async path is the trigger.

Throughput 182-186 TPS/GPU, MFU 3.8%. The run shows FSDP2 at 32 GPUs trains this
geometry without the uneven-sharding error, which the Polaris 32N job needs.

Charts (Sirius 8N, current attempt `e4ejewmg`, from W&B via `plot_production_wandb.py`):

![Sirius 8N, current attempt `e4ejewmg`: loss, throughput, MFU](n8/figures/production_10b_sirius_8n.svg)

![Sirius 8N, current attempt `e4ejewmg`: grad norm, learning rate, max loss](n8/figures/training_diagnostics_10b_sirius_8n.svg)

![Sirius 8N, current attempt `e4ejewmg`: tokens vs time](n8/figures/tokens_vs_time_10b_sirius_8n.svg)

## Polaris: 16N and 32N on Dolma

Same config as Sirius; GBS 64 at 16N, 128 at 32N. Both jobs have waited since
Nov 9 with `Not enough free nodes`.

## Links

- [Umbrella `9053470` and per-seat resubmission](../../2026-11-11-umbrella-2088n.md)
- [Sirius hedge report](../sirius-olmo3-5b-10b.md)
- [LR finder, 10B](../../../experiments/lr-finder/agpt/10b/README.md)
- [`submit_agpt_olmo3_polaris.sh`](../../../../scripts/submit_agpt_olmo3_polaris.sh),
  [`seat-autoretry-20261111.pbs`](../../../../submit/aurora/seat-autoretry-20261111.pbs)
