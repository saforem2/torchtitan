# 13 — Grain CPT anneal smoke (2N)

## What we run
200-step smoke of `agpt_2b_grain_cpt_anneal` on 2N Sunspot (24 ranks FLAT). Exercises the 3-way anneal mix at warmup weights 70/20/10 (olmo / fineweb-edu / cosmopedia) so the first smoke validates all three sources load and tokenize. Grain tokenizes on-the-fly via `EZPZTokenizer` (~4.5k tps/tile) — **NO pre-tokenize / save_to_disk / pretokenize_to**; mixing ratios stay dynamic for the 100B anneal (70/20/10 -> 20/50/30).

## Where logs / outputs
- PBS stdout: `$PBS_O_WORKDIR/logs/grain-cpt-anneal-smoke-$JOBID/run.log` (+ `train.log`)
- Outputs: `outputs/checkpoints/agpt-2b-grain-cpt-anneal/` (disabled in smoke via `--checkpoint.no-enable`)
- Script: `torchtitan/experiments/ezpz/scripts/grain_cpt_anneal_smoke_2n.sh`
- Submit: `qsub torchtitan/experiments/ezpz/scripts/grain_cpt_anneal_smoke_2n.sh` from `/home/foremans/torchtitan`

## Why
Validate Grain 3-way mix loads before committing 4N 100B anneal. Prior CPT on static 60/40 flatlined MMLU 0.302 -> 0.317 (+0.015/135k), proving textbook/domain inject is needed — the anneal schedule (olmo 70->20, fineweb-edu 20->50, cosmo 10->30) addresses this.

## Expected results
- Loss ~2.3-2.5 (continuing from 2.4 at 38k)
- Throughput comparable to ~4800 tps
- No source-load errors; all 3 corpora stream

## Next steps
- If smoke passes -> launch full 4N 100B anneal
- If fails -> fix failing source path or anneal weights, re-smoke

## Job
Job ID: 12484897 (Q, grain-cpt-anneal-smoke 2N 00:45 workq)
