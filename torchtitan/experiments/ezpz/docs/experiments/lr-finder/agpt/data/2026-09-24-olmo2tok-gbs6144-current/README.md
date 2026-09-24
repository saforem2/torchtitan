# Current OLMo-3-vocab GBS=6144 chart inputs

These immutable CSV copies reproduce the stable charts in
`../../figures/olmo2tok-gbs6144/` via
`../../../../../scripts/plot_olmo2tok_coarse_fine.py`.

Included completed arms:

- `sunspot-12478508-5b-adamw-fine.csv`: 100 finite points.
- `sunspot-12478509-10b-adamw-fine.csv`: 100 finite points.
- `sunspot-12478511-5b-sophiag-coarse.csv`: 30 finite points.
- `sunspot-12478569-10b-sophiag-fine.csv`: 100 finite points.
- `sunspot-12478513-30b-sophiag-coarse.csv`: 30 finite points.

Incomplete or partial fine curves are deliberately absent. In particular,
`12478568` exhausted walltime without a completed 5B SophiaG fine artifact,
`12478570` had not produced its terminal 30B SophiaG fine artifact at this
snapshot, and the 30B AdamW replacement `12478581` remained queued.
