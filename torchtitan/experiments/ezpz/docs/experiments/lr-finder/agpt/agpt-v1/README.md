# AGPT v1 LR-finder results: 2B / 20B / 80B

This section contains learning-rate results for the original dense AGPT model
family. It is kept separate from the OLMo-3-vocab 5B/10B/30B family in
[`../agpt-v2/`](../agpt-v2/).

## Per-model reports

| model | report | scope |
|---|---|---|
| 2B | [2b/README.md](2b/README.md) | small-batch finders and production-batch trend |
| 20B | [20b/README.md](20b/README.md) | small-batch and GBS ladders |
| 80B | [80b/README.md](80b/README.md) | GBS=192 finder, GBS=6144 production batch, and LR-ceiling trend |

## Production-batch guidance at GBS=6144

| Model | AdamW | Mano | SophiaG | behavior |
|---|---:|---:|---:|---|
| 2B | ~8.6e-4 | ~1.6e-3 | ~1.4e-4 | broad clean minima; batch-independent |
| 20B | 1.6e-4 min | 8.6e-5 min | 1.4e-3 min | clean minima; no cliff |
| 80B | use ~5e-7 | use ~3e-6 | use ~1e-6 | AdamW cliffs; Mano/SophiaG retain minima |

These values summarize the detailed model pages; consult those pages before
selecting a production LR. Cross-machine summary figures are in
[`figures/`](figures/).

[Back to the AGPT generation index](../README.md).
