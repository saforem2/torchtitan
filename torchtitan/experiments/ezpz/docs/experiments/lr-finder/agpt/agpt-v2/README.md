# AGPT v2 LR-finder results: 5B / 10B / 30B

This section contains learning-rate results for the OLMo-3-vocab AGPT family
(the repository retains the historical `olmo2tok` config names). It is kept
separate from the original 2B/20B/80B family in
[`../agpt-v1/`](../agpt-v1/).

## Current GBS=6144 campaign

| Model | AdamW | SophiaG | current evidence |
|---|---:|---:|---|
| 5B | **7.17e-5** (fine) | **2.80e-6** (coarse) | AdamW fine complete; SophiaG fine walltime-truncated |
| 10B | **4.16e-5** (fine) | `3.24e-7` detector [1] | both fine artifacts complete |
| 30B | pending | **6.48e-7** (coarse) | SophiaG fine running; resumable AdamW fine queued |

[1] The 10B SophiaG detector value is below its fine sweep's sampled range; it
is recorded for provenance, not claimed as a measured fine optimum.

- **[GBS=6144 campaign report, job matrix, and charts](2026-09-18-olmo2tok-ladder-gbs6144-nexteval.md)**

## Prior 30B results at GBS=960

| date | report | result |
|---|---|---|
| 2026-08-23 | [three optimizers](2026-08-23-30b-gbs960-three-optimizers.md) | AdamW 3.05e-5, Mano 5.61e-5, SophiaG 3.55e-5 |
| 2026-08-23 | [Mano detail](2026-08-23-30b-mano-sunspot.md) | Mano arm details |
| 2026-08-30 | [Muon](2026-08-30-30b-gbs960-muon.md) | Muon result and partition analysis |

Committed source CSVs are under [`data/`](data/); campaign figures are under
[`figures/`](figures/).

[Back to the AGPT generation index](../README.md).
