#!/usr/bin/env python3
"""Local incremental parquet cache of W&B production-run history.

Unlike ``wandb_fetch.py`` (dependency-light: stdlib + lazy wandb, loaded on the
cluster's bare .venv), this module is LOCAL-ONLY and dependency-HEAVY: it uses
pandas + pyarrow to keep a per-run parquet cache under ``~/.cache``. It exists
so the production dashboard can:

  1. Pull straight from W&B on the user's laptop (creds in ``~/.netrc``), which
     sidesteps the flaky Aurora SSH master socket entirely, and
  2. Cache ALL logged metrics per run (not just the 5 the board historically
     showed), so any of the ~24 metrics W&B records (loss, grad_norm, lr, tps,
     tflops, mfu, n_tokens, memory {active,reserved}x{GiB,%}, OOMs, timing,
     validation loss/tps, ...) can be plotted without a re-fetch.

Incrementality: a run's parquet is (re)fetched only when needed. A W&B run in a
terminal state (``finished``/``crashed``/``failed``/``killed``) never changes,
so once cached it is served from disk forever. A run still ``running`` is
re-fetched only for steps PAST the cached max ``_step`` and merged. This keeps a
refresh over ~80 run-ids to just the handful of live runs' new tail.

Public surface:
    CACHE_DIR                         ~/.cache/agpt-prod-dash/wandb (override PD_CACHE_DIR)
    metric_columns(df)                the plottable metric columns present in a frame
    fetch_run(run_id, ...)            -> DataFrame for one run (cache-first, incremental)
    fetch_chain(run_ids, ...)         -> DataFrame for a chain (concat + step-dedup)
    fetch_all(trajectories=None, ...) -> {chain_key: DataFrame} for every prod chain
    discover_metrics(frames)          -> sorted union of metric columns across chains

The lingua franca out of this module is a pandas ``DataFrame`` indexed by
``_step`` (ascending, deduped keeping the latest run's value for a shared step,
matching resume semantics). Callers that want the dependency-light list-of-dicts
shape should keep using ``wandb_fetch`` instead.
"""
from __future__ import annotations

import os
import time

import pandas as pd

PROJECT = "aurora_gpt/torchtitan.ezpz.train"

# Per-run parquet cache. Override with PD_CACHE_DIR (e.g. a repo-local dir).
CACHE_DIR = os.environ.get(
    "PD_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "agpt-prod-dash", "wandb"),
)

# W&B states that will never log another step -> cache is permanent.
_TERMINAL_STATES = frozenset({"finished", "crashed", "failed", "killed"})

# Non-metric bookkeeping columns that scan_history returns but which are not
# themselves plottable y-values (they are axes / metadata).
_NON_METRIC = frozenset({"_step", "_timestamp", "_runtime", "_wandb"})


def _run_parquet(run_id: str) -> str:
    return os.path.join(CACHE_DIR, run_id + ".parquet")


def _meta_path(run_id: str) -> str:
    return os.path.join(CACHE_DIR, run_id + ".state")


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Plottable metric columns in ``df`` (everything except the axes/metadata).

    A column counts as a metric only if it holds at least one non-null numeric
    value -- W&B occasionally returns an all-null column for a metric a run
    never logged, and those should not clutter the selector.
    """
    out = []
    for c in df.columns:
        if c in _NON_METRIC:
            continue
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) and col.notna().any():
            out.append(c)
    return sorted(out)


def _load_cached(run_id: str):
    """Return (df, cached_state) from disk, or (None, None) if absent/unreadable."""
    p = _run_parquet(run_id)
    if not os.path.exists(p):
        return None, None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None, None
    state = None
    try:
        with open(_meta_path(run_id)) as f:
            state = f.read().strip()
    except OSError:
        pass
    return df, state


def _save_cached(run_id: str, df: pd.DataFrame, state: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _run_parquet(run_id) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, _run_parquet(run_id))  # atomic; a crash can't leave a half file
    try:
        with open(_meta_path(run_id), "w") as f:
            f.write(state or "")
    except OSError:
        pass


def _scan_full(run, start_step=None) -> pd.DataFrame:
    """scan_history over a run with NO key filter (grabs every logged metric).

    ``start_step`` (exclusive) limits the scan to the new tail of a live run;
    W&B's scan_history has no server-side step filter, so we still stream all
    rows but drop the already-cached prefix client-side -- cheap vs. re-parsing
    into parquet, and it keeps the merge simple.
    """
    rows = []
    for row in run.scan_history():  # dict per logged step, all keys
        s = row.get("_step")
        if start_step is not None and s is not None and s <= start_step:
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "_step" in df.columns:
        df = df.sort_values("_step").reset_index(drop=True)
    return df


def fetch_run(run_id: str, api=None, project: str = PROJECT,
              force: bool = False, log=None) -> pd.DataFrame:
    """Return one run's full-metric history (cache-first, incremental).

    - Cached + run was terminal when cached + not ``force`` -> serve disk as-is.
    - Cached + run still live (or unknown) -> fetch only steps past the cached
      max ``_step`` and merge.
    - Not cached -> full fetch.
    Returns an empty DataFrame if the run can't be loaded from W&B and nothing
    is cached.
    """
    def _emit(m):
        if log:
            log(m)

    cached_df, cached_state = _load_cached(run_id)
    if (cached_df is not None and not force
            and cached_state in _TERMINAL_STATES):
        _emit("  %s: cache hit (terminal=%s, %d rows)" % (
            run_id, cached_state, len(cached_df)))
        return cached_df

    if api is None:
        import wandb
        api = wandb.Api(timeout=30)
    try:
        run = api.run(project + "/" + run_id)
    except Exception as e:
        if cached_df is not None:
            _emit("  %s: W&B unreachable (%s); serving stale cache" % (
                run_id, type(e).__name__))
            return cached_df
        _emit("  %s: W&B unreachable and no cache; skipping" % run_id)
        return pd.DataFrame()

    state = getattr(run, "state", None) or "unknown"

    if cached_df is not None and not force and not cached_df.empty:
        cached_max = int(cached_df["_step"].max()) if "_step" in cached_df else None
        new_df = _scan_full(run, start_step=cached_max)
        if new_df.empty:
            _emit("  %s: no new steps past %s (state=%s)" % (
                run_id, cached_max, state))
            merged = cached_df
        else:
            merged = pd.concat([cached_df, new_df], ignore_index=True)
            merged = (merged.sort_values("_step")
                      .drop_duplicates("_step", keep="last")
                      .reset_index(drop=True))
            _emit("  %s: +%d new steps (-> max %s, state=%s)" % (
                run_id, len(new_df), int(merged["_step"].max()), state))
        _save_cached(run_id, merged, state)
        return merged

    df = _scan_full(run)
    _save_cached(run_id, df, state)
    _emit("  %s: full fetch %d rows (state=%s)" % (run_id, len(df), state))
    return df


def fetch_chain(run_ids, api=None, project: str = PROJECT,
                force: bool = False, log=None) -> pd.DataFrame:
    """Concat a chain's run-ids into one frame, deduped by ``_step``.

    Runs are fetched in listed order; on a shared step the LATER run wins
    (resume semantics), matching ``wandb_fetch.concat_chain``.
    """
    frames = []
    for rid in run_ids or []:
        df = fetch_run(rid, api=api, project=project, force=force, log=log)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "_step" in out.columns:
        out = (out.sort_values("_step")
               .drop_duplicates("_step", keep="last")
               .reset_index(drop=True))
    return out


def fetch_all(trajectories=None, project: str = PROJECT,
              force: bool = False, log=None) -> dict:
    """Fetch every production chain's full-metric history.

    ``trajectories`` defaults to the manifest in ``trajectories.py`` (all chains
    that carry ``wandb_run_ids``). Returns ``{chain_key: DataFrame}``; chains
    that yield no data are omitted. One ``wandb.Api()`` is reused across runs.
    """
    if trajectories is None:
        from torchtitan.experiments.ezpz.utils import trajectories as _t
        trajectories = _t.TRAJECTORIES
    import wandb
    api = wandb.Api(timeout=30)
    out = {}
    for e in trajectories:
        ids = e.get("wandb_run_ids") or []
        if not ids:
            continue
        key = e["key"]
        if log:
            log("%s: %d run(s)" % (key, len(ids)))
        df = fetch_chain(ids, api=api, project=project, force=force, log=log)
        if not df.empty:
            out[key] = df
    return out


def discover_metrics(frames) -> list[str]:
    """Sorted union of plottable metric columns across a dict/list of frames."""
    seen = set()
    it = frames.values() if isinstance(frames, dict) else frames
    for df in it:
        if df is not None and not df.empty:
            seen.update(metric_columns(df))
    return sorted(seen)


if __name__ == "__main__":
    # Smoke: fetch all chains, report per-chain row counts + discovered metrics.
    import sys
    t0 = time.time()
    def _log(m):
        sys.stderr.write("[wandb_cache +%5.1fs] %s\n" % (time.time() - t0, m))
    force = "--force" in sys.argv
    chains = fetch_all(force=force, log=_log)
    print("=== chains: %d ===" % len(chains))
    for k, df in chains.items():
        steps = (int(df["_step"].min()), int(df["_step"].max())) if "_step" in df else ("?", "?")
        print("  %-22s rows=%-6d steps=%s" % (k, len(df), steps))
    print("=== metrics discovered: %d ===" % len(discover_metrics(chains)))
    for m in discover_metrics(chains):
        print("  ", m)
