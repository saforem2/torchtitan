# Unregistered W&B runs: the failure that never announces itself

**Status:** detector added 2026-08-17 (`utils/find_missing_runs.py`); the
underlying drift is structural and will recur.

## The shape of it

`utils/trajectories.py` carries a hand-maintained `wandb_run_ids` list per
chain. Every W&B-derived view -- the overlay charts, the live board, the
exported metric store -- is built from that list. A run that is missing from it
is not an error anywhere: the chain still plots, still shows a live state, still
reports a plausible loss. It just silently omits every step that run logged.

Nothing fails. Nothing warns. The output looks exactly like a correct output.

## Four different wrong things, all the same cause

| Symptom | What it actually was |
|---|---|
| Gaps in the training charts | Crashed W&B runs never synced their tails; the runs that *did* cover those steps were unregistered |
| 20B-512 showing state `R` with a **six-day-old** heartbeat | The seat was writing steps every second. Its current run (`ctbs1be4`) was not in the list |
| A chain rendering as a truncated dir name (`2b-sophiag-olmo-mix-1124-n512-gbs1`) with no token target | The 2B-512 constant-LR fork had **no trajectory entry at all**, at ~21k steps across three umbrellas |
| A curve absent from a figure while regen reported "0 scripts failed" | Separate problem, same family -- see [plotter display lists](#see-also) |

Every one of these was found by accident, while looking at something else.
That is not a detection strategy.

## The detector

```bash
# on Aurora, where W&B credentials live
./.venv/bin/python3 -m torchtitan.experiments.ezpz.utils.find_missing_runs --days 30
./.venv/bin/python3 -m torchtitan.experiments.ezpz.utils.find_missing_runs --chain 20b_v2_512
```

It asks W&B which runs wrote each chain's checkpoint dir, rather than trusting
the list. `metadata.args` is the literal argv a run executed, so
`--checkpoint.folder` says authoritatively which chain a run belongs to -- not
the submit script, not the clone's defaults, not the run's name.

Matching is on the dir **basename**: the same logical chain is reachable by
different absolute prefixes, because a fork's checkpoints live in its own clone
under `/flare/AuroraGPT/foremans/runs/`, not the main repo tree, and each run
records whichever path its clone used.

## Unregistered is NOT the same as missing data

The first scan reported **29 unregistered runs**, and adding all 29 would be
wrong. A crashed attempt that was immediately relaunched writes the same ckpt
dir and logs a step range the relaunch then re-covers; registering it changes
nothing. Others have no usable history at all (`last _step=None` -- died before
logging), or are 50-step smoke tests.

A run is only a real gap if it contributes steps **outside the union already
covered by the registered runs**. Check that before editing:

```python
covered = union of steps over traj["wandb_run_ids"]
new     = steps(candidate) - covered      # empty => redundant, skip it
```

This is why `find_missing_runs.py` is read-only and prints candidates instead
of editing `trajectories.py`. The judgement is not mechanical.

## When to run it

- After any umbrella finishes (each seat mints a new run-id, and a relaunch
  mid-job mints more).
- Before regenerating charts or re-exporting the metric store, since both bake
  in whatever the list says at that moment.
- Whenever the board shows a chain as live with a stale heartbeat -- that
  specific combination is close to diagnostic.

## See also

- `utils/trajectories.py --check-coverage` -- the complementary check: reads
  each chain's on-disk step head, which is the FLOOR its plotted data must
  reach. Coverage says *how far the data should go*; this doc's detector says
  *which runs carry it*.
- Plotter display lists drift the same way and are guarded separately -- see
  `_assert_no_missing_live_chains()` in `utils/plot_production_combined.py` and
  `eval/plot_evals_combined.py`.
- `memory/project_olog_fallback_union_not_replace.md` -- the `.o`-log fallback
  path, for steps that exist in a console log but never reached W&B.
