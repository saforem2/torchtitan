# `production/` move -- the last Step 3 item

> Scoped 2026-08-23. NOT yet executed: the plan requires a quiet window, and
> two umbrellas were queued when this was written.

`production/` is the only directory that contains all four lifecycle kinds at
once, so it does not move as one rename -- it splits four ways.

## Scale

- **187 files** across 8 subtrees plus 6 root files
- **94 code references** from `.py`/`.sh` (not the 44 estimated from an
  earlier grep -- that count missed several tooling files)
- The referencing code includes the **live dashboard**: `prod_dash.py`,
  `trajectories.py`, `check_dashboard_drift.py`, `fill_trajectory_fields.py`,
  `plot_production*.py`, `export_ground_truth.py`, `refresh_all.sh`,
  `update_all_charts.sh`, `check_stale_docs.sh`

That last point is why this waits for a window: the pinned production clones
pull separately, so a path this tooling reads must not move mid-run.

## Mapping

| from | to | kind |
|---|---|---|
| `production/README.md` | `live/dashboard.md` | file |
| `production/dispatch-log.md` | `live/dispatch-log.md` | file |
| `production/loss-dashboard.md` | `live/loss-dashboard.md` | file |
| `production/queue-wait-analysis.md` | `live/queue-wait-analysis.md` | file |
| `production/POST-TRAINING-2B.md` | `live/chains/agpt/2b/post-training.md` | file |
| `production/scaling-performance.md` | `reference/scaling/performance.md` | file |
| `production/figures/` | `live/figures/` | dir (6) |
| `production/metrics/` | `live/metrics/` | dir (11) |
| `production/agpt/30b-exp/` | `records/proposals/30b-exp/` | dir |
| `production/agpt/` | `live/chains/agpt/` | dir (92, after 30b-exp) |
| `production/cpt/` | `live/chains/cpt/` | dir (13) |
| `production/moe/` | `live/chains/moe/` | dir (1) |
| `production/polaris/` | `live/chains/polaris/` | dir (6) |
| `production/rl/` | `live/chains/rl/` | dir (26) |
| `production/sft/` | `live/chains/sft/` | dir (26) |

`metrics/` is not in the original plan. It holds the ground-truth CSVs the
dashboard reads (`manifest.json` + per-chain `*.csv`), which decay with the
chains -- so it is `live/`, beside `figures/`.

**Order matters:** move `agpt/30b-exp/` to `records/proposals/` BEFORE moving
`agpt/`, or it lands in `live/chains/agpt/` and has to be moved twice.

## Method

`scripts/oneoff/reorg_move.py <src> <dst>` per row, one commit each. It takes a
single src -> dst, so the split is several passes. It resolves links by path
rather than string substitution, which is required here: "production" appears
in prose throughout the corpus.

The root FILES are renames, not directory moves -- `reorg_move.py` handles
directories only. Use `git mv` plus a targeted link sweep for those six, or
extend the mover.

## Pre-flight (already done)

- [x] All 5 depth-counted `parents[N]` paths under `production/` converted to
      a marker-anchored `_repo_root()` walk (ffbc9c727). Without this the move
      silently empties five charts -- see
      [[project_moved_scripts_parents_n_silent_empty]].
- [x] `check_doc_links.sh` now flags the `parents[N]` pattern (a9ff2f279).

## Verify after each commit

1. `bash scripts/check_doc_links.sh` -- must stay at 33
2. `grep -rn "docs/live" --include="*.py" --include="*.sh"` -- zero stale
3. Diff any regenerated SVG by DISTINCT COLOR COUNT, not byte size
4. Do NOT let a local `refresh_all.sh` commit the two eval charts: their JSON
   is cluster-side, so they always render empty here
