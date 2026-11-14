# Verified OLMo-3-1K GBS=128 Sirius evidence — 2026-11-11 to 2026-11-12

Immutable copies of LR-finder CSV artifacts used by the [campaign report](../../2026-11-12-olmo3-gbs128-sirius.md).
All four files have 150 unique LR samples, finite LR/loss fields, and exit-0 terminal logs.
The finder ran the full training step (forward + backward + optimizer) at each swept LR with
exponential increase (Smith/Gugger, `find_optimal_lr` derivative crossing).

| job | model | optimizer | rows | sampled LR range | SHA-256 | interpretation |
|---:|---|---|---:|---|---:|---|
| `33429` | 5B | AdamW | 150 | `1e-08`–`1e-01` | `0d0415f5ed1b25a7897daa53355e8f50945ec3afcee919694e3e988d76e11e87` | min `9.38` at `2.53e-03`; detector blow-up `2.48e-03` → suggested `2.48e-04` |
| `33429` | 5B | Mano | 150 | `1e-08`–`1e-01` | `c1f54cb36b4282696481dc4f9ef6b4ff59c18b0c0ec3beedcf92f37b40beed47` | min `9.75` at `2.27e-03`; detector blow-up `1.81e-03` → suggested `1.81e-04` |
| `33430` | 10B | AdamW | 150 | `1e-08`–`1e-01` | `c05d3b4120d97484d25436718664b5f860f6834e4bf2bcda1ac9a671fe2da0db` | min `9.23` at `1.19e-03`; detector blow-up `1.28e-03` → suggested `1.28e-04` |
| `33450` | 10B | Mano | 150 | `1e-08`–`1e-01` | `9ff98d35fdff9044c10b9a62e0961465399f31cab83480765717b4a3d59c76a3` | min `9.66` at `2.04e-03`; detector blow-up `2.25e-03` → suggested `2.25e-04` |

The `global_batch_size` CSV field is **token** batch size: 524,288 tokens = 128 sequences × 4,096 tokens.
`world_size=32` (8N × 4 GPUs/node, Sirius H100), `seq_len=4096`.

## SHA-256 (full file)

```
0d0415f5ed1b25a7897daa53355e8f50945ec3afcee919694e3e988d76e11e87  sirius-33429-5b-adamw-150.csv
c1f54cb36b4282696481dc4f9ef6b4ff59c18b0c0ec3beedcf92f37b40beed47  sirius-33429-5b-mano-150.csv
c05d3b4120d97484d25436718664b5f860f6834e4bf2bcda1ac9a671fe2da0db  sirius-33430-10b-adamw-150.csv
9ff98d35fdff9044c10b9a62e0961465399f31cab83480765717b4a3d59c76a3  sirius-33450-10b-mano-150.csv
```

Source terminal logs on Sirius: `~/torchtitan/outputs/lr_finder_*_{adamw,mano}_33429.log` etc., each ending `Execution finished with 0` and `LR Finder: suggested LR = ... (blow-up at ...)`.

## Notes

- Job `33429` ran both 5B arms sequentially under one PBS ID (AdamW then Mano), so the Sirius live CSV `lr_finder_data.csv` was overwritten per optimizer. The per-arm CSVs here were reconstructed from the terminal sweep logs (`LR Finder [sweep]: step i/150, lr=..., smoothed_loss=...`) for 5B and preserved directly for 10B (where the live CSV matched the swept range).
- Job `33450` (10B Mano) is a resume from a walltime kill at step 72/150 of `33430`'s Mano companion; the final CSV is the completed 150 after resume from step 90 checkpoint. Both the 5B and 10B finders completed cleanly with finite losses throughout.
- The detector is `blow-up / 10` (`find_optimal_lr`, derivative crossing); suggested LR is the conservative 10×-down value, not the measured blow-up itself.
