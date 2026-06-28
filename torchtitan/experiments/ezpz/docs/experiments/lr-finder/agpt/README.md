# LR Finder -- agpt (Dense) -- index

Learning-rate-finder results for the dense **agpt** (AuroraGPT) models. This
page is the cross-model index (master recommended-LR table + findings that
span model sizes / machines). Per-model detail lives in its own page:

- **[agpt 2B](2b/README.md)** -- small-batch finders + production-batch trend (in progress)
- **[agpt 20B](20b/README.md)** -- small-batch finders (no trend yet)
- **[agpt 80B](80b/README.md)** -- GBS=192 finder, GBS=6144 production batch, LR-ceiling-vs-GBS trend

For how the finder works, its knobs, and usage, see the
[methodology README](../README.md). Index-level cross-model figures
(`*_comparison`, `*_optimal_lr`) are in [`figures/`](figures/); single-model
figures live under each model page's `figures/`.

## Recommended Learning Rates

Small-batch finder (`sqrt(2/(5*d))` weight init, 5% warmup, Sunspot 2026-04-14,
GBS at the 2-node default):

| Model | AdamW   | Muon    | SophiaG |
|-------|---------|---------|---------|
| 2B    | **1.3e-3**| **2.4e-3**| **3.1e-4**|
| 20B   | **4.0e-4**| **1.7e-4**| **1.8e-5**|
| 80B (small batch, GBS=192) | **1.1e-5** | N/A[1] | N/A[1] |

[1] At GBS=192, Muon/SophiaG NaN'd: bf16 overflow in Newton-Schulz (Muon) and
Hessian estimate (SophiaG) on 9216-dim matrices. **At the production batch
GBS=6144 SophiaG is NOT broken** (real U-min at lr~2.5e-6); only Muon stays
broken. See [agpt 80B](80b/README.md).

> **80B at the PRODUCTION batch (GBS=6144) is different -- the small-batch
> number above does NOT transfer.** Re-running the finder at the real
> production batch (32x larger) shows AdamW's usable LR collapses from
> 1.1e-5 to **~7e-7** (a NaN cliff), so the production default LR=1e-6 sits
> *on the cliff*. **mano** is the best-behaved optimizer at GBS=6144 (clean
> U-min at 1.6e-5).
>
> | Optimizer @ GBS=6144 | usable LR | min loss | behavior |
> |---|---|---|---|
> | **mano** | **~1.6e-5 (use ~3e-6)** | 12.62 | **clean broad U-min, 0 NaN** |
> | **sophiag** | ~2.5e-6 (use ~1e-6) | **12.60** | real U-min, 0 NaN, narrower band |
> | AdamW | ~7e-7 (use ~5e-7) | 12.78 | NaN cliff at 1.36e-6 |
> | muon | -- | -- | broken (NaN from step 7) |
>
> Full detail: [agpt 80B](80b/README.md). **Recommendation: do not run AdamW
> @ 1e-6 at GBS=6144; mano is the safest 80B production optimizer, sophiag a
> viable second.**

## Key Findings (cross-model / cross-machine)

1. **Optimizer sensitivity:** `AdamW (most tolerant) > Muon > SophiaG (most sensitive)`
2. **Model scaling:** Larger models need lower LRs. Muon/SophiaG scale more
   aggressively (~N^-0.5) than AdamW (~N^-0.25).
3. **Blow-up severity:** SophiaG diverges catastrophically (loss 7,000+) vs
   gradual blow-up for AdamW (loss ~60). SophiaG requires tighter LR scheduling.
4. **Cross-hardware consistency:** Suggested LRs match within 2x across Intel
   XPU (Aurora, Sunspot) and NVIDIA A100 (Polaris).
5. **Batch dependence is large-model-specific.** At 80B the usable/ceiling LR
   collapses ~20x with batch and turns into a NaN cliff
   ([trend](80b/README.md#lr-ceiling-vs-gbs-trend-adamw)). At
   [2B](2b/README.md#2026-06-28----lr-ceiling-vs-gbs-trend) the usable LR is
   flat at ~1e-2 across a 128x batch range (0 NaN) -- **no collapse, no cliff.**
   So a small-batch sweep cannot calibrate a *large*-model production LR, but
   small models are themselves forgiving of large batches (likely the same
   dim=9216 bf16 fragility that breaks 80B).

### Cross-Machine Comparison -- agpt 2B

| Optimizer | Aurora (std=0.02) | Sunspot (sqrt(2/5d)) | Polaris (std=0.02) |
|-----------|--------|---------|---------|
| **AdamW** suggested LR | 2e-3 | 1.3e-3 | 2e-3 |
| **Muon** suggested LR | 8e-4 | **2.4e-3** | 1e-3 |
| **SophiaG** suggested LR | 3e-4 | 3.1e-4 | 3e-4 |

### Cross-Machine Comparison -- agpt 20B

| Optimizer | Aurora (std=0.02) | Sunspot (sqrt(2/5d)) | Polaris (std=0.02) |
|-----------|--------|---------|---------|
| **AdamW** suggested LR | 4e-4 | 4.0e-4 | 4e-4 |
| **Muon** suggested LR | 4e-5 | **1.7e-4** | -- |
| **SophiaG** suggested LR | 1e-5 | 1.8e-5 | -- |

**Takeaway:** AdamW and SophiaG are robust to weight init changes. Muon is
sensitive -- `sqrt(2/(5*d))` init allows 3-10x higher LRs vs fixed `std=0.02`.
This is because Muon's orthogonal momentum amplifies gradient scale differences.

### Cross-machine finder curves (2B + 20B side by side)

| Machine | Comparison | Optimal LR |
|---|---|---|
| Aurora | ![Aurora comparison](figures/aurora_comparison.png) | ![Aurora optimal](figures/aurora_optimal_lr.png) |
| Sunspot | ![Sunspot comparison](figures/sunspot_comparison.png) | ![Sunspot optimal](figures/sunspot_optimal_lr.png) |
| Polaris | ![Polaris comparison](figures/polaris_comparison.png) | ![Polaris optimal](figures/polaris_optimal_lr.png) |

## Reports

Per-model pages (each holds all dates / machines / batch sizes for that model):

| Model | Page | Experiments |
|-------|------|-------------|
| 2B  | [2b/README.md](2b/README.md)   | Aurora/Sunspot/Polaris small-batch (Apr 12-21) + production-batch trend (Jun 28: never cliffs) |
| 20B | [20b/README.md](20b/README.md) | Aurora/Sunspot/Polaris small-batch (Apr 12-21) |
| 80B | [80b/README.md](80b/README.md) | GBS=192 finder (Apr 21), GBS=6144 production (Jun 27), LR-ceiling-vs-GBS trend |
