# W&B gap-backfill: one synthetic run per gap

## Must satisfy (from the user's saved workspace view)

FILTERS the run must PASS:
  model_spec.name     != null,  IN {ezpz.agpt, moe}      -> "ezpz.agpt"
  model_spec.flavor   != null,  IN {2b, 20b}             -> per chain
  machine             != null                            -> "Aurora"
  training.local_batch_size = 2
  training.seq_len          = 8192
  optimizer.name            = "SophiaG"
  dataloader.dataset        = "blendcorpus"
  dataloader.dataset_path   = ".../data-lists/aurora/olmo-mix-1124.txt"
  world_size          IN {512, 3072, 6144, 24, 48}       -> per chain
  Created Timestamp   >= 2026-03-04   (and a second >= 2026-07-13, unchecked)
  Updated Timestamp   >= 2026-04-25

FILTERS that EXCLUDE -- a synthetic run must NOT match these:
  env.created_at != 2026-04-25-171049   -> use the gap's own timestamp
  Name           != likely-sunset-1667  -> any other name

GROUP-BY keys (all must be present or the run lands in a "null" bucket):
  machine, world_size, model_spec.name, model_spec.flavor,
  optimizer.name, training.local_batch_size, training.global_batch_size,
  training.seq_len, checkpoint.folder, Name

## Per-chain values (read from real runs, not invented)

  20b_v2_512: flavor=20b, world_size=6144, gbs=12288,
              ckpt=checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288
  20b_v2_256: flavor=20b, world_size=3072, gbs=6144,
              ckpt=checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144
  2b_v2_256:  flavor=2b,  world_size=3072, gbs=6144,
              ckpt=checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144

## Design decisions

- ONE run per contiguous gap (user's preference). Provenance stays clean and
  nothing mutates a historical run.
- Name pattern: `backfill-<chain>-<firststep>-<laststep>` -- self-describing,
  and cannot collide with the excluded `likely-sunset-1667`.
- Add `backfill: true` + `backfill_source: olog` to config, and tag the run
  `backfill`. Not in any filter, so it does not affect the view, but it makes
  synthetic runs trivially separable later. This matters: without it, a future
  reader cannot distinguish a replayed curve from a measured one.
- `_step` replayed verbatim from the .o records so the curves align.
- env.created_at set to the gap's own wall-clock, NOT the excluded sentinel.

## Scale

  20b_v2_512  2061 olog-only points
  20b_v2_256  2024
  2b_v2_256    322
  = 4407 points that our charts show and W&B does not.
