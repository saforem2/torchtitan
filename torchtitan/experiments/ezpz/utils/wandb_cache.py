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


def _api(timeout: int = 30):
    """Return a ``wandb.Api()`` that can NEVER block on an interactive prompt.

    With no credentials (no ``~/.netrc`` entry for api.wandb.ai, no
    ``WANDB_API_KEY``), the wandb client asks for an API key on stdin. Inside
    prod_dash_app that read happens on a background worker thread, so the TUI
    just hangs with an invisible prompt until the user interrupts -- and the
    caller's ``try/except`` does not help, because blocking on stdin is not an
    exception. Raise instead, so the caller's existing "enrich skipped" path
    runs and the dashboard degrades to the SSH-sourced 5 metrics.

    Two layers, because the credential check alone is only a heuristic (a
    malformed or expired ~/.netrc entry still looks present):

    1. Refuse up front when no credential source exists at all.
    2. Redirect stdin to /dev/null around the construction, so any prompt the
       client still attempts hits EOF and raises instead of blocking forever.
    """
    import netrc  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if not os.environ.get("WANDB_API_KEY"):
        has_netrc = False
        try:
            hosts = netrc.netrc().hosts
            has_netrc = any("wandb" in h for h in hosts)
        except Exception:
            has_netrc = False
        if not has_netrc:
            raise RuntimeError(
                "no W&B credentials (WANDB_API_KEY unset and no api.wandb.ai "
                "entry in ~/.netrc) -- refusing to let the client prompt on "
                "stdin. Run `wandb login` once to enable the all-metric cache."
            )
    import wandb  # noqa: PLC0415
    # Point stdin at /dev/null for the construction only: a prompt then reads
    # EOF and raises instead of blocking a background thread forever.
    saved = sys.stdin
    with open(os.devnull) as devnull:
        sys.stdin = devnull
        try:
            return wandb.Api(timeout=timeout)
        finally:
            sys.stdin = saved


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
              force: bool = False, log=None, counters=None) -> pd.DataFrame:
    """Return one run's full-metric history (cache-first, incremental).

    - Cached + run was terminal when cached + not ``force`` -> serve disk as-is.
    - Cached + run still live (or unknown) -> fetch only steps past the cached
      max ``_step`` and merge.
    - Not cached -> full fetch.
    Returns an empty DataFrame if the run can't be loaded from W&B and nothing
    is cached.

    ``counters`` (optional dict): a terminal cache-hit increments
    ``counters['hit']`` SILENTLY (no per-run log line) so the caller can print
    one summary instead of one line per run. Interesting events (new fetches,
    new steps on live runs, unreachable/errors) always log via ``log``.
    """
    def _emit(m):
        if log:
            log(m)

    cached_df, cached_state = _load_cached(run_id)
    if (cached_df is not None and not force
            and cached_state in _TERMINAL_STATES):
        # Silent terminal cache-hit: count it, don't spam a line every refresh.
        if counters is not None:
            counters["hit"] = counters.get("hit", 0) + 1
        else:
            _emit("  %s: cache hit (terminal=%s, %d rows)" % (
                run_id, cached_state, len(cached_df)))
        return cached_df

    if api is None:
        api = _api()
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
                force: bool = False, log=None, counters=None) -> pd.DataFrame:
    """Concat a chain's run-ids into one frame, deduped by ``_step``.

    Runs are fetched in listed order; on a shared step the LATER run wins
    (resume semantics), matching ``wandb_fetch.concat_chain``.

    Terminal cache-hits are summarized, not logged per-run: pass a shared
    ``counters`` dict to accumulate across chains (caller prints the total), or
    omit it and this emits a single ``"N/M runs cache-hit"`` line per chain.
    """
    own_counter = counters is None
    counters = counters if counters is not None else {}
    frames = []
    for rid in run_ids or []:
        df = fetch_run(rid, api=api, project=project, force=force, log=log,
                       counters=counters)
        if df is not None and not df.empty:
            frames.append(df)
    if own_counter and log and counters.get("hit"):
        log("  %d/%d runs cache-hit (terminal, served from disk)" % (
            counters["hit"], len(run_ids or [])))
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
    api = _api()
    out = {}
    counters = {}  # shared across all chains -> one grand-total summary
    total_runs = 0
    for e in trajectories:
        ids = e.get("wandb_run_ids") or []
        if not ids:
            continue
        total_runs += len(ids)
        key = e["key"]
        df = fetch_chain(ids, api=api, project=project, force=force, log=log,
                         counters=counters)
        if not df.empty:
            out[key] = df
    if log and counters.get("hit"):
        log("%d/%d runs cache-hit (terminal, served from disk); rest fetched"
            % (counters["hit"], total_runs))
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
