# LR Finder -- agpt 20B

Learning-rate-finder results for the dense **agpt 20B** model. For how the
finder works see the [methodology README](../../README.md); for the
cross-model recommendation table and findings see the
[agpt index](../README.md). Figures in [`figures/`](figures/).

**Recommended (small-batch, `sqrt(2/(5*d))` init):** AdamW **4.0e-4**, Muon
**1.7e-4**, SophiaG **1.8e-5** (Sunspot 2026-04-14).

> **Note:** a production-batch trend sweep (like
> [80B's](../80b/README.md#lr-ceiling-vs-gbs-trend-adamw)) has not been run
> for 20B yet. The numbers below are small-batch (GBS<=192); per the 80B
> finding they may not transfer directly to 20B production batch. 20B is a
> good candidate to fill the middle of the model-size axis once the 2B and
> 80B trends are complete.

---

## 2026-04-14 -- 20B (Sunspot, dim-aware init)

2 nodes / 24 XPU tiles, 100 finder steps (5 warmup + 95 sweep), LR 1e-6 -> 1.0,
weight init `sqrt(2/(5*d))` (dim-aware), blendcorpus (books). (Same job also
swept 2B -- see [2B page](../2b/README.md).)

| Optimizer | Blow-up LR | Suggested LR |
|-----------|-----------|-------------|
| AdamW   | 3.95e-3 | **4.0e-4** |
| Muon    | 1.65e-3 | **1.7e-4** |
| SophiaG | 1.82e-4 | **1.8e-5** |

**Effect of dim-aware init** -- `sqrt(2/(5*d))` (vs fixed `std=0.02`) most
affects Muon: 20B Muon 1.7e-5 -> 1.7e-4 (10x higher). AdamW and SophiaG
relatively unaffected.

![Sunspot 20B finder](figures/sunspot_20b.png)

## 2026-04-21 -- 20B verification + GAS sweep (Sunspot)

Re-ran 20B on torch 2.13 (compile enabled) alongside the 80B finder.

| Optimizer | NaN | Suggested LR | Blow-up |
|-----------|-----|-------------|---------|
| AdamW | 0/100 | ~4.6e-5 | ~4.6e-4 |
| Muon | 0/100 | ~7.4e-4 | ~7.4e-3 |
| SophiaG | 7-9/100 | ~4.6e-7 | ~4.6e-6 |

20B GAS=4/8/16 (GBS 96/192/384) all completed with 0 NaN; suggested LRs not
captured in output (SSH pipe buffering). 20B (dim=5120) works fine for Muon
and SophiaG -- the bf16 overflow that breaks them is specific to 80B
(dim=9216).

## 2026-04-13 -- 20B (Polaris, NVIDIA A100)

2 nodes / 8 A100-40GB, 100 finder steps, LR 1e-6 -> 1.0, fixed `std=0.02` init,
blendcorpus (books), NCCL.

| Optimizer | Min Loss | LR @ Min | Blow-up LR | Suggested LR | Final Loss |
|-----------|----------|----------|-----------|-------------|-----------|
| AdamW   | 11.38 | 3.5e-3 | ~8e-3 | **4e-4** | 85.7 |
| Muon    | 12.59 | 1.7e-4 | ~3e-4 | **2e-5** | 66.0 |
| SophiaG | 11.64 | 1.0e-3 | ~2e-3 | **1e-4** | NaN  |

**Reproduces Aurora** closely: AdamW min loss 11.38 @ LR 3.5e-3 (Aurora:
11.43 @ 4.0e-3), blow-up ~8e-3 (Aurora ~5e-3). The 20B AdamW sweep ran on a
second allocation (4951s wall).

W&B: [AdamW](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/9vx92cl7).

![Polaris 20B finder](figures/polaris_20b.png)

## 2026-04-12 -- 20B (Aurora, first run)

2 nodes / 24 XPU tiles, 100 finder steps, LR 1e-6 -> 1.0, fixed `std=0.02` init,
blendcorpus (books), xccl.

| Optimizer | Min Loss | LR @ Min | Blow-up LR | Suggested LR | Final Loss |
|-----------|----------|----------|-----------|-------------|-----------|
| AdamW   | 11.43 | 4.0e-3 | ~5e-3 | **4e-4** | 57.1   |
| Muon    | 12.51 | 3.8e-4 | ~5e-4 | **4e-5** | 67.9   |
| SophiaG | 12.40 | 1.3e-4 | ~2e-4 | **1e-5** | 7528.9 |

SophiaG shows catastrophic divergence (loss > 7500) vs AdamW/Muon (~60) --
the Hessian-based preconditioning produces violent blow-ups at high LR.

![Aurora 20B finder](figures/aurora_20b.png)

---

## Reports index

| Date | Machine | GBS | Optimizers | Nodes | Key Result |
|------|---------|-----|-----------|-------|------------|
| 2026-04-21 | Sunspot | 96..384 | AdamW, Muon, SophiaG | 2 | torch-2.13 verify + GAS; Muon/SophiaG fine at dim=5120 |
| 2026-04-14 | Sunspot | 48 | AdamW, Muon, SophiaG | 2 | dim-aware init: AdamW 4.0e-4, Muon 1.7e-4 (10x higher) |
| 2026-04-13 | Polaris | 48 | AdamW, Muon, SophiaG | 2 | reproduces Aurora; cross-hardware consistency |
| 2026-04-12 | Aurora | 48 | AdamW, Muon, SophiaG | 2 | AdamW 4e-4; SophiaG catastrophic blow-up (7528) |
