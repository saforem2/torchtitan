# Sirius hedge: agpt 5B and 10B on the OLMo-3 tokenizer (Nov 2026)

Compiled 2026-11-14 from the [journal](../../journal.md) entries of Nov 13-14. Numbers
are as logged there; nothing here was re-measured.

## Why

Polaris `7832130` (`18N`, `small`) has been queued since Nov 9 with `Not enough free
nodes` (`81 small / 43 medium` queued at `15:44 UTC` Nov 14). Sirius runs the same two
models meanwhile, so training keeps moving and the 10B sizing gets proven before Polaris
starts it.

## Setup

| | 5B | 10B |
|---|---|---|
| Config | `5b_olmo3_polaris` | `10b_olmo3_polaris` |
| Shape | `2N`, `8` GPUs, `n2-gbs8` | `8N`, `32` GPUs, `n8-gbs32` |
| FSDP | FFN `8192`, `% 32 == 0` | FFN `11264`, `% 32 == 0` |

- Sirius `workq` caps a job at `8N` (`Insufficient ncpus R:512 A:38` on a second `8N`),
  so the two models run as `8N` + `2N`.
- DCP checkpoints do not load across world sizes: `n2-gbs8` cannot resume as `n8-gbs32`.
  A size change is a fresh start, and `CKPT_DIR=checkpoints/agpt-${MODEL}-n${NNODES}-gbs${GBS}`
  keeps them apart.
- `AuroraGPT` on Sirius ran out (`5004812288/5040000000`) on Nov 14. `-A datascience`
  (backfill) works.
- `submit_agpt_olmo3_polaris.sh` fails on Sirius (`cudnn/9.26.0`, `Unknown machine:
  sirius`). The journal names a `submit_agpt_olmo3_sirius.sh` fix; it is not in the
  repo yet. Only [`submit_agpt_olmo3_polaris.sh`](../../../scripts/submit_agpt_olmo3_polaris.sh) is.

## Jobs

| Job | Model | Steps | Loss | Throughput | End |
|---|---|---|---|---|---|
| `33469` | 10B | `1 -> 200` | `12.00` -> `6.38` | `tps 185`, MFU `3.8%` | `E 143` at `01:24` Nov 13: `FAILOVER STOP exhausted`, no spare node; `step-100` (`106G`) |
| `33471` | 5B | `2900 -> 5058` | `6.50` -> `6.14` | `tps 409`, MFU `4.2%` | walltime (`21658` s); `50` ckpts `step-600..5000` |
| `33488` | 10B | `200` steps | `12.00` -> `6.19` | `tps 182` | `Gloo` crash on node `x3200c0s13b0` at `step 200` (`FAILOVER exhausted`); resubmit rejected, allocation spent. The Nov 13 entry calls the 10B resubmit `33486` (resume from `step-100`); the Nov 14 entry calls it `33488` and starts its loss at `12.00`. |
| `33487` | 5B | `5058 -> 7092` | `6.05` -> `5.90` | `tps 416`, MFU `4.27%` | walltime (`21667` s); last good checkpoint `step-7000` |
| `33525` | 5B | `7000 -> 7454+` | `5.97` -> `6.02` | `tps 416`, MFU `4.27%` | running at `15:44 UTC` Nov 14, `-A datascience` |
| `33527` | 10B | fresh `1 -> 197+` | `12.03` -> `6.31` | `tps 186`, MFU `3.81%` | running at `15:44 UTC` Nov 14 |

A Sirius login-node outage (`~00:13`-`15:18 UTC` Nov 13) blocked monitoring, not the
running jobs.

## What it showed

- **10B fits Polaris at 32 GPUs.** FSDP2 on `32` GPUs trained `200` steps (`12.00 ->
  6.38` in `1h15m`) with no uneven-sharding error, so the Polaris `n32` 10B run will not
  hit it.
- **5B trains steadily.** `2900 -> 7454` across four slices, loss `6.50 -> ~6.0`.
- **Walltime kills can corrupt the newest checkpoint.** `33487` was killed at `21667` s
  against a `21600` s limit; resume took `step-7000`, not `step-7092`. The 10B `step-100`
  was left without `.metadata` (`32` shards) and the 10B run restarted fresh.

## LR finder (Nov 11-12, journal only)

`150` points at `GBS128`, `1e-8 -> 1e-1`:

| Model | AdamW | Mano |
|---|---|---|
| 5B | `2.48e-04` | `1.81e-04` |
| 10B | `1.28e-04` | `2.25e-04` |

Mano's LR rises from 5B to 10B while AdamW's falls. These sweeps are not on the
[5B](../../experiments/lr-finder/agpt/5b/README.md) or
[10B](../../experiments/lr-finder/agpt/10b/README.md) LR-finder pages yet, whose
recorded evidence is the GBS=6144 campaign.

## Open

- Commit `submit_agpt_olmo3_sirius.sh`.
- Add the Nov 11-12 GBS128 sweeps to the LR-finder pages.
- Polaris `7832130` still queued.
