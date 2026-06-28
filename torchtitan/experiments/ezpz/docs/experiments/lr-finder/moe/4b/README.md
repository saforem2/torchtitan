# LR Finder -- moe 4B (24 experts)

Part of the [moe LR-finder index](../README.md); methodology in the
[parent README](../../README.md). Figures in [`figures/`](figures/).

## 2026-04-21 (Sunspot, 2N, torch 2.13, compile on, seq_len=8192, LBS=1)

| Optimizer | NaN | Suggested LR | Blow-up | Status |
|-----------|-----|-------------|---------|--------|
| AdamW   | 0 | 1.22e-3 | 1.22e-2 | clean (representative MoE optimum) |
| Muon    | 0 | -- | -- | ok |
| SophiaG | 1 | -- | -- | mild instability at high LR only |

![moe 4B finder](figures/lr_finder_4B.png)
