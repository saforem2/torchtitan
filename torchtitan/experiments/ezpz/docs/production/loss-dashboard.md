# Live production loss dashboard (`prod_dash.py`)

A companion to the GRPO reward dashboard
(`rl/scripts/grpo_lora_agpt2b_patches/rl_dash3.py`) but for the main
AuroraGPT pre-training chains. It overlays the loss curve of every
production chain (2B/20B/80B, all node counts) plus auto-discovered
active experiment forks, and highlights whichever chain is currently
training.

Source: `torchtitan/experiments/ezpz/utils/prod_dash.py`.

## What it tracks

- **Canonical chains** come from `utils/trajectories.py` (the single
  source of truth shared with the committed charts, the stale-doc map,
  and the eval scripts), so the dashboard can never drift from those.
  Every record with `cls in {live, wandb_only}` is shown.
- **Active experiments** are auto-discovered: any `agpt-*` checkpoint dir
  referenced by a PBS `.o` log that is not a canonical chain. Stale ones
  (newest `.o` write older than `PD_EXP_MAX_AGE`, default 7d) are hidden
  unless `--all` is passed; a running job always overrides staleness.

## Data flow

Collection runs **remote over SSH** (the cluster has W&B, the PBS `.o`
logs, and `qstat`); rendering happens **local**. Run it on the cluster
with `PD_LOCAL=1` to skip the SSH hop.

- The expensive part -- per-chain loss history from W&B
  (`scan_history` over ~60 runs, 3-6 min) -- is cached cluster-side at
  `/tmp/prod_dash_backbone_<user>.json` with a TTL (`PD_BACKBONE_TTL`,
  default 1h). Completed-step loss never changes, so a stale cache is
  served immediately (marked `(refreshing)`) while a detached rebuild
  runs, and the next call is warm. Only a first-ever run (no cache)
  blocks on a synchronous build.
- The cheap **live layer** -- `qstat` states + tails of currently
  running `.o` logs -- runs every call (~1-4 s) and supplies the moving
  tip (current step/loss/tps/mfu) plus live/idle coloring.

## Three modes

```bash
# 1. Live kitcat loop (run in a kitty terminal; the rl_dash3 twin).
#    Overlays every chain's loss, live chain bright / idle dim, refresh 30s.
/tmp/kitcat-venv/bin/python torchtitan/experiments/ezpz/utils/prod_dash.py
python .../prod_dash.py --once        # one live frame then exit

# 2. Headless SVG/PNG snapshot for docs/production/.
python .../prod_dash.py --svg figures/production_loss_live.svg

# 3. Stall-aware text status board (no matplotlib needed). Best while the
#    queue is stalled: per-chain last loss, step, % to token target,
#    queue state (R/Q/idle), tps/mfu, and log staleness.
python .../prod_dash.py --board
```

Flags: `--all` (include stale experiments), `--fresh` (force a W&B
rebuild this call), `--tokens` (x-axis in billions of tokens instead of
steps -- comparable across chains since all share the 4.67T olmo-mix
target).

## Env

| var | default | meaning |
|-----|---------|---------|
| `PD_LOCAL` | unset | `1` = read local FS (run on the cluster), else SSH |
| `PD_SSH` | `aurora` | ssh target |
| `PD_SOCK` | `/tmp/aurora-master.sock` | ssh ControlPath |
| `PD_INTERVAL` | `30` | live-loop poll seconds |
| `PD_BACKBONE_TTL` | `3600` | W&B backbone cache TTL (s) |
| `PD_EXP_MAX_AGE` | `604800` | hide experiment forks older than this (s) |
| `PD_SHOW_ALL` | unset | `1` = show every experiment ever run |
| `PD_LIVE_WINDOW` | `300` | `.o`-log write age (s) that counts as "live" |
| `PD_XAXIS` | `step` | `tokens` for a tokens x-axis |
| `PD_REPO` | flare checkout | cluster repo path |
