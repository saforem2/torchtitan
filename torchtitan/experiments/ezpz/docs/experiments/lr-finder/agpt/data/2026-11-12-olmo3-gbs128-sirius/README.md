# Verified OLMo-3-1K GBS=128 Sirius evidence -- 2026-11-11 to 2026-11-15

Immutable copies of LR-finder CSV artifacts used by the [campaign report](../../2026-11-12-olmo3-gbs128-sirius.md).
All seven files have 150 unique LR samples (10B SophiaG excluded: incomplete), finite LR/loss fields, and exit-0 terminal logs (except 10B SophiaG).
The finder ran the full training step (forward + backward + optimizer) at each swept LR with
exponential increase (Smith/Gugger, `find_optimal_lr` derivative crossing).

| job | model | optimizer | rows | sampled LR range | SHA-256 | interpretation |
|---:|---|---|---:|---|---:|---|
| `33429` | 5B | AdamW | 150 | `1e-08`-`1e-01` | `0d0415f5...` | min `9.38` at `2.53e-03`; detector blow-up `2.48e-03` -> suggested `2.48e-04` |
| `33429` | 5B | Mano | 150 | `1e-08`-`1e-01` | `c1f54cb3...` | min `9.75` at `2.27e-03`; detector blow-up `1.81e-03` -> suggested `1.81e-04` |
| `33542` | 5B | Muon | 150 | `1e-08`-`1e-01` | `0e982695...` | min `9.57` at `6.69e-03`; detector blow-up `6.54e-03` -> suggested `6.54e-04` |
| `33542` | 5B | SophiaG | 150 | `1e-08`-`1e-01` | `6227da95...` | **degenerate** -- see below; real min `10.00` at `1.06e-03` |
| `33430` | 10B | AdamW | 150 | `1e-08`-`1e-01` | `c05d3b41...` | min `9.23` at `1.19e-03`; detector blow-up `1.28e-03` -> suggested `1.28e-04` |
| `33450` | 10B | Mano | 150 | `1e-08`-`1e-01` | `9ff98d35...` | min `9.66` at `2.04e-03`; detector blow-up `2.25e-03` -> suggested `2.25e-04` |
| `33543` | 10B | Muon | 150 | `1e-08`-`1e-01` | `a61b482f...` | min `9.23` at `1.15e-02`; detector blow-up `1.06e-02` -> suggested `1.06e-03` |

SHA-256 prefixes shown; full hashes below.

The `global_batch_size` CSV field is **token** batch size: 524,288 tokens = 128 sequences x 4,096 tokens.
`world_size=32` (8N x 4 GPUs/node, Sirius H100), `seq_len=4096`.

## SHA-256 (full file)

```
0d0415f5ed1b25a7897daa53355e8f50945ec3afcee919694e3e988d76e11e87  sirius-33429-5b-adamw-150.csv
c1f54cb36b4282696481dc4f9ef6b4ff59c18b0c0ec3beedcf92f37b40beed47  sirius-33429-5b-mano-150.csv
0e982695f4a47069c9da84b6e6ee177fd68c6462c293dd13b716357dac48bccd  sirius-33542-5b-muon-150.csv
6227da9513481ccebf5a449aa7f6c4069545f7c99b477ab72e379b55ad59e9a3  sirius-33542-5b-sophiag-150.csv
c05d3b4120d97484d25436718664b5f860f6834e4bf2bcda1ac9a671fe2da0db  sirius-33430-10b-adamw-150.csv
9ff98d35fdff9044c10b9a62e0961465399f31cab83480765717b4a3d59c76a3  sirius-33450-10b-mano-150.csv
a61b482f9d0113cb6a622b48b1848d5ffd44d70cc93a1c868f973e13e37260df  sirius-33543-10b-muon-150.csv
```

Source terminal logs on Sirius: `~/torchtitan/outputs/lr_finder_*_*.log` (e.g. `lr_finder_5b_muon_33542.log`), each ending `Execution finished with 0` and `LR Finder: suggested LR = ... (blow-up at ...)`. 10B SophiaG (`33543` retry) failed at step 25/150 with a checkpoint fingerprint mismatch and has no complete CSV.

## Notes

- Job `33429` ran both 5B arms sequentially under one PBS ID (AdamW then Mano), so the Sirius live CSV `lr_finder_data.csv` was overwritten per optimizer. The per-arm CSVs here were reconstructed from the terminal sweep logs for 5B and preserved directly for 10B.
- Job `33450` (10B Mano) is a resume from a walltime kill at step 72/150 of `33430`'s Mano companion; the final CSV is the completed 150 after resume from step 90 checkpoint.
- `33542` 5B Muon was reconstructed from its PBS output log `lr_finder_5b_muon_33542.log` (the `outputs/lr_finder/ezpz/AgptModel/muon/` dir was later overwritten by the 10B Muon run). LR values below `1e-07` were restored from the exact exponential schedule (`1e-08 * 1.114242875^i`).
- `33542` 5B SophiaG is **degenerate**: the derivative detector returns two minima (`7.00e-06` and `1.01e-03`). The first is noise in the flat early basin (loss barely moves); the second (`1.01e-03` blow-up, `min 10.00 at 1.06e-03`) is the real basin but its suggested `1.01e-04` is not trustworthy -- see campaign report.
- `33543` ran Muon 10B to completion (150/150, `suggested 1.06e-03`), then auto-retried as SophiaG 10B and hit a `trajectory_fingerprint` mismatch (`a002...` vs `8d2e...`) at load, dying at step 0. No 10B SophiaG evidence at GBS=128.
- The detector is `blow-up / 10` (`find_optimal_lr`, derivative crossing); suggested LR is the conservative 10x-down value, not the measured blow-up itself.
