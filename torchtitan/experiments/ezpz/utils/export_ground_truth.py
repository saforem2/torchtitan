#!/usr/bin/env python3
"""Export every AuroraGPT production trajectory to a durable ground-truth store.

WHY THIS EXISTS
---------------
Every plot, dashboard refresh, and README table re-pulled the same ~92 W&B runs
from the cloud. That is slow, needs credentials, needs the ALCF proxy, and --
worse -- it is not reproducible: some steps exist ONLY in a PBS .o log because
their W&B run was purged or never synced (run-ids ``oqqhoxz6``, ``d00iszlc``,
``sn4q74lc`` return CommError from the API today). Re-deriving the union of
"W&B plus the right .o log attached to the right run" has been rediscovered and
re-broken several times; see the long provenance notes in ``trajectories.py``.

This module freezes that union into files, so the expensive, credential-bound,
drift-prone step happens once per export instead of once per plot.

TWO TIERS, AND WHY (measured, not estimated)
--------------------------------------------
The full store is far bigger than a repo should carry. Measured on 2026-08-17,
7 chains / 182,120 rows / 10 metric columns:

    tier                              raw       gzip
    full fidelity                 30.79 MB   13.17 MB
    every 10th step                3.08 MB    1.34 MB
    ~2000 points + all olog        2.12 MB    0.89 MB

Committing the full store was rejected on those numbers. Gzipping it was also
rejected: 13.17 MB is not marginal, and gzip is the wrong tool for a
regenerated git artifact anyway -- git already zlib-compresses blobs, so the
pack-file win is near zero, while gzip destroys delta compression (appending
500 steps to a live chain would store a whole new multi-MB blob every export
instead of a small delta).

So the store is split:

  1. FULL FIDELITY -> ``DEFAULT_FULL_DIR`` on flare, OUTSIDE any git worktree.
     Every step, every metric. This is what a plotter or an analysis should
     read when it can reach the filesystem.

  2. SUMMARY -> committed under ``DEFAULT_REL_SUMMARY_DIR`` in the repo.
     Small enough to live in git (2.12 MB total, largest single file 366 KB),
     and enough to plot a loss curve, sanity-check a chain, and detect drift
     from a laptop with no flare mount and no W&B credentials.

WHAT THE SUMMARY KEEPS, AND WHAT THAT COSTS
--------------------------------------------
Two rules, unioned:

  a. EVERY row whose ``source`` is ``olog``, at full fidelity, no decimation.
     These are the irreplaceable ones -- recovered by regex from a training
     job's stdout, and for some steps the ONLY surviving record because the
     W&B run is already gone from the cloud. There are 4,081 of them across
     the whole store, about 190 KB raw, so preserving all of them costs
     essentially nothing. This is the whole reason the summary is worth
     committing rather than just keeping a manifest.

  b. A uniform backbone of W&B rows at a per-chain step stride chosen so each
     chain lands near ``SUMMARY_TARGET_POINTS`` (2000) rows, plus the chain's
     first and last step always.

2000 is chosen because a chart is ~800-1600 px wide, so 2000 points already
exceeds what a full-width plot can resolve; the summary curve is visually
indistinguishable from the full one. The real cost is in the gaps BETWEEN
samples: at the resulting strides (50 steps for the long 2B chains, 5 for the
20B ones) a transient that lasts fewer steps than the stride -- a single-step
loss spike, a one-off grad-norm excursion -- can fall between samples and be
invisible. Anything looking for short-lived anomalies must read the full store.

The stride is applied to the STEP VALUE, not the row index, and snaps to a
round number (1/2/5/10/...). That makes the selection stable across exports:
step 4600 is in the file whether the chain has 8,000 rows or 80,000, so
re-exporting a live chain APPENDS to the summary instead of rewriting every
line. Row-index decimation would reshuffle the whole file on every export and
make the git history useless.

THE ``source`` COLUMN
---------------------
Each row carries ``wandb`` or ``olog``. Not decoration: an ``olog`` row is a
number scraped from stdout, and a reader comparing chains or chasing an anomaly
needs to know which points came from the cloud and which from a log.

Provenance is derived structurally, not guessed. ``wandb_fetch.parse_olog``
builds records with exactly the six ``OLOG_KEYS``; ``wandb_fetch.fetch_wandb_run``
builds records with exactly the requested ``keys`` (all ten of ``ALL_KEYS``
here, present even when the value is None). So a record holding the W&B-only
keys came from W&B, and one that does not came from a log. ``--verify``
cross-checks that inference against a direct ``parse_olog`` of every fallback
file.

FORMAT: plain CSV
-----------------
Greppable, diffable, and readable in review, and it deltas well on re-export.
Answering "what was the loss at step 4500 of 20b_v2_256" in one command:

    grep '^4500,' <store>/20b_v2_256.csv

Column 1 is ``_step``, so an anchored grep is unambiguous. The header names
every column; ``loss_metrics/global_avg_loss`` is the loss. (Step 4500 is on
the 20B chains' stride-5 backbone, so it is in the committed summary too; on a
2B chain, a step not on the stride is only in the full store.)

RUNNING IT
----------
Run on Aurora: the ``.o`` / console-log fallbacks only exist on that
filesystem, so an export from a laptop would silently drop every recovered
step and quietly shrink the chains.

    export http_proxy=http://proxy.alcf.anl.gov:3128
    export https_proxy=http://proxy.alcf.anl.gov:3128
    ./.venv/bin/python3 torchtitan/experiments/ezpz/utils/export_ground_truth.py \
        --repo-root "$PWD" --verify

Takes roughly 8 minutes, dominated by W&B API paging over the long 2B chains.

Dependency-light by the same contract as ``wandb_fetch``: stdlib only, with
``wandb`` imported lazily. Sibling modules are loaded by file path rather than
by package import, because importing ``torchtitan.experiments.ezpz.*`` runs
that package's ``__init__`` and drags in torch -- unnecessary for reading
numbers, and a hard failure on a bare python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time

# Column written after the metric columns; see the module docstring.
SOURCE_COLUMN = "source"

SOURCE_WANDB = "wandb"
SOURCE_OLOG = "olog"

# A run of missing steps this long or longer is reported as a gap. Chains log
# every step, so anything above a handful is a real hole in the record, but
# small ones are noise from a crash boundary; 50 is the threshold the
# production notes have been using.
DEFAULT_GAP_THRESHOLD = 50

# Full-fidelity store: on flare, deliberately OUTSIDE any git worktree.
DEFAULT_FULL_DIR = "/flare/AuroraGPT/foremans/production-metrics"

# Committed summary, relative to the repo root. NOT under a directory named
# "data", which would have been the obvious name: the root .gitignore has a
# bare ``data`` pattern (line 12) matching a directory called ``data`` at ANY
# depth, so files written there are silently untracked -- exactly the failure
# this store exists to prevent. This location sits beside the production
# READMEs that consume these numbers and follows the existing precedent of
# committed CSVs under docs/ (docs/records/experiments/agpt/aurora/figures/loss_*.csv).
DEFAULT_REL_SUMMARY_DIR = "torchtitan/experiments/ezpz/docs/production/metrics"

MANIFEST_NAME = "manifest.json"

# Target row count per chain in the committed summary; see the module
# docstring for why 2000 and what it costs.
SUMMARY_TARGET_POINTS = 2000

# Strides the summary is allowed to use. Round numbers keep the selection
# predictable and stable as a chain grows.
_NICE_STRIDES = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000)


def _load_module(name, path):
    """Import a module from an explicit file path.

    Bypasses the package chain: ``torchtitan.experiments.ezpz.__init__`` installs
    a triton stub and a device-compat shim, both of which import torch. Reading
    metrics needs neither. Same trick ``prod_dash.py`` uses.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s from %s" % (name, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_repo_root():
    """Repo root inferred from this file's location.

    ``<root>/torchtitan/experiments/ezpz/utils/export_ground_truth.py`` -> four
    directories up. Wrong if the script has been copied elsewhere, which is why
    ``--repo-root`` exists.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def load_helpers(repo_root):
    """Return ``(trajectories_module, wandb_fetch_module)`` from ``repo_root``."""
    base = os.path.join(repo_root, "torchtitan", "experiments", "ezpz", "utils")
    traj = _load_module("_gt_trajectories", os.path.join(base, "trajectories.py"))
    wf = _load_module("_gt_wandb_fetch", os.path.join(base, "wandb_fetch.py"))
    return traj, wf


# ---------------------------------------------------------------------------
# CSV read/write
# ---------------------------------------------------------------------------


def _fmt(value):
    """Render one cell.

    ``None`` becomes the empty string (W&B returns None for a metric a run did
    not log at that step; that is different from zero and must survive the
    round trip). Floats go through ``repr``, which in Python 3 is the shortest
    string that reads back as the identical double -- no precision is lost, so
    a loss printed to five decimals in a chart is byte-identical before and
    after the store.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_chain_csv(path, records, columns):
    """Write ``records`` to ``path`` as CSV with a ``columns`` header.

    Writes to a temporary file and renames, so an interrupted export cannot
    leave a half-written store that looks valid.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for rec in records:
            writer.writerow([_fmt(rec.get(c)) for c in columns])
    os.replace(tmp, path)


def _parse_cell(column, text):
    if text == "":
        return None
    if column == SOURCE_COLUMN:
        return text
    if column == "_step":
        return int(text)
    return float(text)


def read_chain_csv(path):
    """Read a chain CSV back into the list-of-dicts shape ``concat_chain`` emits."""
    out = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return out
        for row in reader:
            out.append({c: _parse_cell(c, v) for c, v in zip(header, row)})
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Reader helpers for consumers
# ---------------------------------------------------------------------------


def full_store_dir():
    """Absolute path of the full-fidelity store (override with AGPT_METRICS_DIR)."""
    return os.environ.get("AGPT_METRICS_DIR", DEFAULT_FULL_DIR)


def summary_store_dir(repo_root=None):
    """Absolute path of the committed summary store inside the repo."""
    root = repo_root or _default_repo_root()
    return os.path.join(root, DEFAULT_REL_SUMMARY_DIR)


def load_chain(key, repo_root=None):
    """Return one chain's FULL-fidelity ground-truth records.

    Reads the durable store on flare (``AGPT_METRICS_DIR``, default
    ``DEFAULT_FULL_DIR``). The whole point of the store is to replace a W&B
    round trip, so a missing file raises instead of quietly falling back to the
    cloud: a silent fallback would reintroduce exactly the credential-bound,
    non-reproducible fetch this module exists to remove, and would hide the
    fact that the store is stale or absent. Off-cluster, use ``load_summary``.

    Returns a list of per-step dicts keyed by the ``wandb_fetch`` metric names
    plus ``source``, sorted ascending by ``_step`` -- the same shape
    ``wandb_fetch.concat_chain`` returns, so a consumer switches over by
    replacing::

        records = wandb_fetch.concat_chain(
            cfg["run_ids"], olog_fallbacks=cfg.get("olog_fallbacks"),
            keys=wandb_fetch.ALL_KEYS, api=api)

    with::

        records = export_ground_truth.load_chain(key)

    and dropping the ``wandb.Api()`` construction. Note the store is a SNAPSHOT:
    for a chain still training it lags by however long ago the export ran (see
    ``exported_at`` and ``cls`` in the manifest).
    """
    path = os.path.join(full_store_dir(), key + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no full-fidelity ground-truth store for chain %r at %s -- "
            "regenerate it on Aurora with `python3 torchtitan/experiments/ezpz/"
            "utils/export_ground_truth.py` (the .o-log fallbacks only exist "
            "there), or call load_summary(%r) for the committed reduced-"
            "resolution copy. Refusing to fall back to W&B silently."
            % (key, path, key)
        )
    return read_chain_csv(path)


def load_summary(key, repo_root=None):
    """Return one chain's COMMITTED summary records (reduced resolution).

    Always available in a checkout, needs no flare mount and no W&B
    credentials. Every ``olog``-sourced row is present at full fidelity; the
    ``wandb`` rows are a uniform step-stride backbone (see the module
    docstring). Good for plotting a loss curve or checking a chain's shape;
    NOT sufficient for anything hunting short-lived anomalies -- use
    ``load_chain`` for that.
    """
    path = os.path.join(summary_store_dir(repo_root), key + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no committed summary for chain %r at %s -- regenerate with "
            "export_ground_truth.py on Aurora." % (key, path)
        )
    return read_chain_csv(path)


def load_manifest(repo_root=None):
    """Return the committed manifest (export timestamp, per-chain summary)."""
    path = os.path.join(summary_store_dir(repo_root), MANIFEST_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError("no manifest at %s" % path)
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def find_gaps(steps, threshold=DEFAULT_GAP_THRESHOLD):
    """Return ``[(prev_step, next_step, missing_count), ...]`` for holes.

    A hole counts when consecutive recorded steps differ by more than
    ``threshold``.
    """
    gaps = []
    ordered = sorted(steps)
    for prev, nxt in zip(ordered, ordered[1:]):
        missing = nxt - prev - 1
        if missing >= threshold:
            gaps.append((prev, nxt, missing))
    return gaps


def tag_sources(records, all_keys):
    """Return ``records`` with a ``source`` value added to each.

    See the module docstring: a W&B record carries every key in ``all_keys``
    (the fetcher builds it from the requested key list), a ``parse_olog``
    record carries only the six OLOG keys. Mutates in place and returns the
    list.
    """
    wanted = set(all_keys)
    for rec in records:
        rec[SOURCE_COLUMN] = SOURCE_WANDB if wanted <= set(rec) else SOURCE_OLOG
    return records


def choose_stride(step_max, target=SUMMARY_TARGET_POINTS):
    """Smallest nice stride whose backbone lands at or under ``target`` rows."""
    for stride in _NICE_STRIDES:
        if step_max / stride <= target:
            return stride
    return _NICE_STRIDES[-1]


def summarize(records, target=SUMMARY_TARGET_POINTS):
    """Return ``(subset, stride)`` -- the committed reduced-resolution view.

    Keeps every ``olog`` row (irreplaceable), plus a step-stride backbone, plus
    the first and last row. Selecting on the step VALUE rather than the row
    index keeps the subset stable as a live chain grows, so a re-export appends
    instead of rewriting the file.
    """
    if not records:
        return [], 1
    steps = [int(r["_step"]) for r in records]
    stride = choose_stride(max(steps), target)
    keep = []
    for i, rec in enumerate(records):
        if (rec[SOURCE_COLUMN] == SOURCE_OLOG
                or int(rec["_step"]) % stride == 0
                or i in (0, len(records) - 1)):
            keep.append(rec)
    return keep, stride


def _olog_steps(wf, traj_record):
    """Steps recoverable directly from this chain's .o fallbacks, as a set.

    Used to cross-check the structural provenance inference. A step here is not
    necessarily tagged ``olog`` in the export -- W&B may also hold it, in which
    case the union keeps the W&B row -- but a step tagged ``olog`` that is NOT
    here would mean the inference is wrong.
    """
    paths = list((traj_record.get("olog_fallbacks") or {}).values())
    if not paths:
        return set()
    records, _ = wf.parse_olog(paths)
    return {int(r["_step"]) for r in records if r.get("_step") is not None}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_chain(traj_record, wf, api, log):
    """Fetch one chain and return ``(records, stats)``."""
    key = traj_record["key"]
    run_ids = traj_record.get("wandb_run_ids") or []
    fallbacks = traj_record.get("olog_fallbacks") or None

    log("[%s] %d run-ids, %d olog fallback(s)"
        % (key, len(run_ids), len(fallbacks or {})))

    records = wf.concat_chain(
        run_ids,
        olog_fallbacks=fallbacks,
        keys=wf.ALL_KEYS,
        api=api,
        log=log,
    )
    tag_sources(records, wf.ALL_KEYS)

    steps = [int(r["_step"]) for r in records if r.get("_step") is not None]
    n_olog = sum(1 for r in records if r[SOURCE_COLUMN] == SOURCE_OLOG)
    stats = {
        "key": key,
        "model": traj_record.get("model"),
        "num_nodes": traj_record.get("num_nodes"),
        "cls": traj_record.get("cls"),
        "gbs": traj_record.get("gbs"),
        "seq_len": traj_record.get("seq_len"),
        "rows": len(records),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "rows_from_wandb": len(records) - n_olog,
        "rows_from_olog": n_olog,
        "wandb_run_ids": list(run_ids),
        "olog_fallbacks": dict(fallbacks or {}),
        "gaps": [
            {"after_step": a, "before_step": b, "missing_steps": n}
            for a, b, n in find_gaps(steps)
        ],
    }
    return records, stats


def _values_match(a, b):
    """True when a written cell read back is the SAME NUMBER.

    Deliberately not an identity check on type. W&B returns a few metrics as
    Python ints (grad_norm 656 on some v1 rows); reading a CSV back always
    yields a float for a metric column, so 656 -> 656.0. That is not a loss of
    fidelity -- it is the same number -- and failing the export over it would
    be a false alarm. What WOULD matter is a changed value, so compare
    numerically and require exactness: for floats ``a == b`` is bit-identical
    (``repr`` is shortest-round-trip), and an int that cannot be represented
    exactly as a double fails here rather than passing silently.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return float(a) == float(b)


def verify_chain(path, records, columns, wf, traj_record, log, label=""):
    """Re-read the written file and prove it matches ``records``.

    Checks, in order:
      1. row count round-trips
      2. every cell round-trips to the same number (see ``_values_match``);
         floats must be bit-identical, so a real precision loss fails here
      3. steps are strictly ascending (the store's ordering contract)
      4. every row tagged ``olog`` is a step that ``parse_olog`` really does
         recover from this chain's fallback files

    Returns a list of problem strings; empty means clean.
    """
    problems = []
    reread = read_chain_csv(path)

    if len(reread) != len(records):
        problems.append(
            "row count changed on round trip: wrote %d, read %d"
            % (len(records), len(reread))
        )
        return problems

    for i, (orig, back) in enumerate(zip(records, reread)):
        for col in columns:
            a, b = orig.get(col), back.get(col)
            if not _values_match(a, b):
                problems.append(
                    "row %d column %s did not round-trip: %r (%s) -> %r (%s)"
                    % (i, col, a, type(a).__name__, b, type(b).__name__)
                )
                if len(problems) > 10:
                    return problems

    steps = [r["_step"] for r in reread]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        problems.append("steps are not strictly ascending in the written file")

    recoverable = _olog_steps(wf, traj_record)
    stray = [
        r["_step"] for r in reread
        if r[SOURCE_COLUMN] == SOURCE_OLOG and r["_step"] not in recoverable
    ]
    if stray:
        problems.append(
            "%d row(s) tagged %s but not recoverable by parse_olog "
            "(provenance inference is wrong); first: %r"
            % (len(stray), SOURCE_OLOG, stray[:5])
        )

    log("  verify%s: %d rows round-tripped, %d olog-tagged, %d problem(s)"
        % (label, len(reread),
           sum(1 for r in reread if r[SOURCE_COLUMN] == SOURCE_OLOG),
           len(problems)))
    return problems


def verify_summary_covers_olog(full_records, summary_records):
    """Every olog row in the full export must survive into the summary.

    The summary's reason to exist is durability for the irreplaceable rows, so
    losing one to decimation is a correctness bug, not a fidelity tradeoff.
    """
    full_olog = {int(r["_step"]) for r in full_records
                 if r[SOURCE_COLUMN] == SOURCE_OLOG}
    summ_olog = {int(r["_step"]) for r in summary_records
                 if r[SOURCE_COLUMN] == SOURCE_OLOG}
    missing = sorted(full_olog - summ_olog)
    if missing:
        return ["summary dropped %d olog row(s) that exist in the full export; "
                "first: %r" % (len(missing), missing[:5])]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Export production training metrics to a durable store"
    )
    ap.add_argument(
        "--repo-root",
        default=_default_repo_root(),
        help="repo root holding trajectories.py and the .o-log fallbacks",
    )
    ap.add_argument(
        "--full-dir",
        default=None,
        help="full-fidelity output dir, kept OUTSIDE git (default: %s)"
        % DEFAULT_FULL_DIR,
    )
    ap.add_argument(
        "--summary-dir",
        default=None,
        help="committed summary output dir (default: <repo-root>/%s)"
        % DEFAULT_REL_SUMMARY_DIR,
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="write only the full store, leave the committed summary alone",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="export only this chain key (repeatable)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="re-read every written file and assert it round-trips",
    )
    ap.add_argument(
        "--summary-points",
        type=int,
        default=SUMMARY_TARGET_POINTS,
        help="target rows per chain in the summary (default %d)"
        % SUMMARY_TARGET_POINTS,
    )
    ap.add_argument(
        "--gap-threshold",
        type=int,
        default=DEFAULT_GAP_THRESHOLD,
        help="report step holes at least this large (default %d)"
        % DEFAULT_GAP_THRESHOLD,
    )
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg):
        sys.stderr.write("[export +%6.1fs] %s\n" % (time.time() - t0, msg))
        sys.stderr.flush()

    repo_root = os.path.abspath(args.repo_root)
    full_dir = os.path.abspath(args.full_dir or full_store_dir())
    summary_dir = os.path.abspath(
        args.summary_dir or os.path.join(repo_root, DEFAULT_REL_SUMMARY_DIR))
    os.makedirs(full_dir, exist_ok=True)
    if not args.no_summary:
        os.makedirs(summary_dir, exist_ok=True)

    traj, wf = load_helpers(repo_root)
    columns = list(wf.ALL_KEYS) + [SOURCE_COLUMN]

    log("repo-root   %s" % repo_root)
    log("full-dir    %s" % full_dir)
    log("summary-dir %s" % (summary_dir if not args.no_summary else "(skipped)"))

    import wandb  # noqa: PLC0415  -- lazy, matching the wandb_fetch contract

    api = wandb.Api(timeout=60)

    chains = []
    skipped = []
    all_problems = []

    for record in traj.TRAJECTORIES:
        key = record["key"]
        if args.only and key not in args.only:
            continue
        if not record.get("wandb_run_ids"):
            # Pure smokes carry no run-ids and were never chart trajectories.
            skipped.append({"key": key, "reason": "no wandb_run_ids",
                            "cls": record.get("cls")})
            log("[%s] skipped: no wandb_run_ids" % key)
            continue

        records, stats = export_chain(record, wf, api, log)
        if not records:
            skipped.append({"key": key, "reason": "no rows returned",
                            "cls": record.get("cls")})
            log("[%s] skipped: fetch returned no rows" % key)
            continue

        full_path = os.path.join(full_dir, key + ".csv")
        write_chain_csv(full_path, records, columns)
        stats["full_file"] = os.path.basename(full_path)
        stats["full_bytes"] = os.path.getsize(full_path)
        stats["full_sha256"] = sha256_file(full_path)
        log("[%s] full    %s (%d rows, %.2f MB)"
            % (key, stats["full_file"], stats["rows"],
               stats["full_bytes"] / 1e6))

        if args.verify:
            problems = verify_chain(full_path, records, columns, wf, record,
                                    log, label=" full")
            all_problems.extend("%s full: %s" % (key, p) for p in problems)
            stats["full_verified"] = not problems

        if not args.no_summary:
            subset, stride = summarize(records, args.summary_points)
            summ_path = os.path.join(summary_dir, key + ".csv")
            write_chain_csv(summ_path, subset, columns)
            stats["summary_file"] = os.path.basename(summ_path)
            stats["summary_rows"] = len(subset)
            stats["summary_step_stride"] = stride
            stats["summary_bytes"] = os.path.getsize(summ_path)
            stats["summary_sha256"] = sha256_file(summ_path)
            log("[%s] summary %s (%d rows, stride %d, %.0f KB)"
                % (key, stats["summary_file"], len(subset), stride,
                   stats["summary_bytes"] / 1e3))
            if args.verify:
                problems = verify_chain(summ_path, subset, columns, wf, record,
                                        log, label=" summary")
                problems += verify_summary_covers_olog(records, subset)
                all_problems.extend("%s summary: %s" % (key, p)
                                    for p in problems)
                stats["summary_verified"] = not problems

        chains.append(stats)

    manifest = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "exported_from": repo_root,
        "full_store_dir": full_dir,
        "columns": columns,
        "gap_threshold": args.gap_threshold,
        "summary_target_points": args.summary_points,
        "source_values": {
            SOURCE_WANDB: "row came from the W&B API",
            SOURCE_OLOG: (
                "row was recovered by parsing a PBS .o / console log; for some "
                "steps this is the only surviving record"
            ),
        },
        "chains": chains,
        "skipped": skipped,
    }
    manifest_path = os.path.join(
        summary_dir if not args.no_summary else full_dir, MANIFEST_NAME)
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, manifest_path)

    tot_full = sum(c["full_bytes"] for c in chains)
    tot_summ = sum(c.get("summary_bytes", 0) for c in chains)
    print("=== %d chain(s): full %.2f MB, committed summary %.2f MB ==="
          % (len(chains), tot_full / 1e6, tot_summ / 1e6))
    print("%-22s %8s %14s %7s %7s %8s %7s  %s"
          % ("chain", "rows", "steps", "wandb", "olog", "summary", "stride",
             "gaps>=%d" % args.gap_threshold))
    for c in chains:
        gaps = ("none" if not c["gaps"] else
                " ".join("%d->%d(%d)" % (g["after_step"], g["before_step"],
                                         g["missing_steps"]) for g in c["gaps"]))
        print("%-22s %8d %14s %7d %7d %8s %7s  %s"
              % (c["key"], c["rows"], "%s..%s" % (c["step_min"], c["step_max"]),
                 c["rows_from_wandb"], c["rows_from_olog"],
                 c.get("summary_rows", "-"), c.get("summary_step_stride", "-"),
                 gaps))
    for s in skipped:
        print("skipped %-20s (%s)" % (s["key"], s["reason"]))

    if all_problems:
        print("\n=== VERIFICATION FAILED: %d problem(s) ===" % len(all_problems))
        for p in all_problems:
            print("  " + p)
        return 1
    if args.verify:
        print("\nverification: all chains round-tripped exactly; every olog row "
              "present in the committed summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
