# Interim LR-finder source data

These CSVs are immutable copies of the job-filtered artifacts used for the
2026-09-21 interim chart. The renderer verifies their SHA-256 digests and
requires exactly 150 finite LR/loss rows for each declared PBS job.

| file | PBS job | original artifact | SHA-256 |
|---|---:|---|---|
| `sunspot-12478315-5b-adamw.csv` | 12478315 | `/home/foremans/olmo3-lrf/outputs/lr_finder_5b_olmo3_gbs6144_adamw_home/lr_finder/ezpz/ezpz.agpt/5b_olmo2tok/adamw/lr_finder_data.csv` | `71d1e2ab9da41b24c668bba9a51a9c7a7bbd78a5a60f55ac3626a9977ac3587c` |
| `sunspot-12478327-5b-mano.csv` | 12478327 | `/home/foremans/olmo3-lrf/outputs/lr_finder_5b_olmo3_gbs6144_mano_home/lr_finder/ezpz/ezpz.agpt/5b_olmo2tok/mano/lr_finder_data.csv` | `78be71cf0c0d80745b1175337f1f94904159780f3177e89e1cd9f54950f783cf` |
| `sunspot-12478328-10b-mano.csv` | 12478328 | `/home/foremans/olmo3-lrf/outputs/lr_finder_10b_olmo3_gbs6144_mano_home/lr_finder/ezpz/ezpz.agpt/10b_olmo2tok/mano/lr_finder_data.csv` | `10ccc0c5e207432b85c6678d1b7e8269e8b98b0ddf71ec0fa78b4f1963da7e88` |

The CSV schema records LR/loss metadata, not gradient health. Before these exact
artifacts were allowlisted, their terminal job logs were checked for all 150
completed optimizer steps and absence of non-finite-gradient/skipped-update
markers. This is why validity is tied to both the declared job ID and digest,
rather than inferred from finite losses alone.
