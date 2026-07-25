#!/usr/bin/env python3
"""Single source of truth for fetching AuroraGPT production loss trajectories.

Historically there were TWO independent implementations of "concatenate a
chain's W&B runs into one loss trajectory, falling back to the PBS .o log when a
run did not sync": one in ``plot_production_wandb.py`` (feeds the committed
charts) and one inlined in ``prod_dash.py``'s remote aggregator (the live
board). They drifted -- one showed a chain "stuck" while the other was correct.
This module holds the ONE canonical implementation both now call.

DEPENDENCY-LIGHT BY CONTRACT: top-level imports are stdlib only, and ``wandb``
is imported lazily inside the functions that need it. This module must NEVER
import numpy, matplotlib, or torch (directly or transitively), because
``prod_dash.py`` loads it on the cluster via ``importlib.spec_from_file_location``
inside a base64'd aggregator that runs on the plain ``.venv`` python and must
stay off those heavy deps. The lingua franca is plain python: a trajectory is a
list of per-step dicts keyed by W&B metric names. Numpy-based callers
(``plot_production_wandb``) adapt on their side.

Public surface:
    OLOG_KEYS            metric keys recoverable from a PBS .o stdout line
    ALL_KEYS             every metric key W&B logs (superset of OLOG_KEYS)
    parse_olog(paths)    -> (records, last): concat .o-log step lines, dedup by
                            step (later files win); ``last`` = final line's tip
    fetch_wandb_run(rid, keys, project=...) -> list[dict]  (one run's history)
    concat_chain(run_ids, olog_fallbacks, keys, project=..., log=...) -> list[dict]
                            robust per-chain concat: for a run with an
                            olog_fallbacks entry, prefer the .o log whenever it
                            reaches at least as far (by max step) as W&B -- this
                            catches partial/truncated W&B history, not just the
                            empty case.

All records are sorted ascending by ``_step`` and deduped keeping the LATEST
run's value for a shared step (resume semantics).
"""
from __future__ import annotations

import re

PROJECT = "aurora_gpt/torchtitan.ezpz.train"

# Metric keys recoverable from a PBS .o per-step stdout line (see OLOG_STEP_RE).
OLOG_KEYS = (
    "_step",
    "loss_metrics/global_avg_loss",
    "grad_norm",
    "throughput(tps)",
    "mfu(%)",
)
# Every metric key W&B logs. The extra keys (timestamp/lr/max_loss/n_tokens)
# are only available from W&B, never from a .o line -- parse_olog leaves them
# absent so numpy adapters can fill NaN.
ALL_KEYS = (
    "_step",
    "_timestamp",
    "grad_norm",
    "lr",
    "loss_metrics/global_avg_loss",
    "loss_metrics/global_max_loss",
    "n_tokens_seen",
    "throughput(tps)",
    "mfu(%)",
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Per-step metric lines in PBS .o files look like:
#   [TS][I][.../metrics:526:log] step: 3300  loss:  2.61866  grad_norm:  0.1672
#   memory: 44.55GiB(69.63%)  tps: 349  tflops: 51.93  mfu: 17.41%
# ANSI escapes wrap each field -- strip them first. We keep the fields that map
# to OLOG_KEYS; memory/tflops are parsed only to anchor the regex.
OLOG_STEP_RE = re.compile(
    r"step:\s+(?P<step>\d+)\s+"
    r"loss:\s+(?P<loss>[\d.]+)\s+"
    r"grad_norm:\s+(?P<grad_norm>[\d.]+)\s+"
    r"memory:\s+[\d.]+GiB\((?P<mem_pct>[\d.]+)%\)\s+"
    r"tps:\s+(?P<tps>[\d,]+)\s+"
    r"tflops:\s+(?P<tflops>[\d.]+)\s+"
    r"mfu:\s+(?P<mfu>[\d.]+)%"
)


def parse_olog(paths):
    """Parse per-step metric lines from one or more PBS .o logs.

    Returns ``(records, last)`` where ``records`` is a list of dicts keyed by
    OLOG_KEYS, sorted ascending by ``_step`` and deduped so a step seen in a
    later file wins (resume semantics); ``last`` is a dict for the final line
    encountered (``step``/``loss``/``tps``/``mfu``) or ``{}`` if nothing parsed.
    Missing/unreadable paths are skipped.
    """
    by_step = {}
    last = {}
    for p in sorted(paths or []):
        try:
            f = open(p, errors="replace")
        except (FileNotFoundError, IsADirectoryError):
            continue
        with f:
            for raw in f:
                m = OLOG_STEP_RE.search(_ANSI_RE.sub("", raw))
                if not m:
                    continue
                step = int(m.group("step"))
                tps = float(m.group("tps").replace(",", ""))
                mfu = float(m.group("mfu"))
                loss = float(m.group("loss"))
                by_step[step] = {
                    "_step": step,
                    "loss_metrics/global_avg_loss": loss,
                    "grad_norm": float(m.group("grad_norm")),
                    "throughput(tps)": tps,
                    "mfu(%)": mfu,
                }
                last = {"step": step, "loss": loss, "tps": tps, "mfu": mfu}
    records = [by_step[s] for s in sorted(by_step)]
    return records, last


def fetch_wandb_run(run_id, keys=OLOG_KEYS, project=PROJECT, api=None):
    """Return one W&B run's history as a list of dicts for the given keys.

    ``keys`` controls cost: pass OLOG_KEYS for a cheap loss curve, ALL_KEYS for
    the full metric set. Returns ``[]`` if the run cannot be found (e.g. it
    logged to a different project) -- callers treat that as a cue to fall back
    to the .o log. Pass an existing ``api`` to reuse one ``wandb.Api()``.
    """
    if api is None:
        import wandb
        api = wandb.Api()
    try:
        run = api.run(project + "/" + run_id)
    except Exception:
        return []
    keys = list(keys)
    out = []
    for row in run.scan_history(keys=keys):
        out.append({k: row.get(k) for k in keys})
    return out


def _max_step(records):
    steps = [r.get("_step") for r in records if r.get("_step") is not None]
    return max(steps) if steps else -1


def concat_chain(
    run_ids,
    olog_fallbacks=None,
    keys=OLOG_KEYS,
    project=PROJECT,
    api=None,
    log=None,
):
    """Concatenate a chain's W&B runs into one trajectory (list of dicts).

    For each run: fetch its W&B history. If the run has an ``olog_fallbacks``
    entry, ALSO parse that .o log and prefer it whenever it reaches at least as
    far as W&B (by max ``_step``) -- this recovers runs that synced partially or
    not at all (e.g. a preflight-smoke reinit swallowed the run, or a crash
    mid-sync). Records are then deduped by step keeping the LATEST run's value.

    ``log`` (optional callable) receives one-line progress strings, matching the
    ``print`` diagnostics the chart plotters emit. ``keys`` is passed through to
    the W&B fetch; the .o fallback always yields OLOG_KEYS (a subset), so absent
    keys stay absent for callers to fill.
    """
    olog_fallbacks = olog_fallbacks or {}
    if api is None:
        try:
            import wandb
            api = wandb.Api()
        except Exception:
            api = None

    def _emit(msg):
        if log is not None:
            log(msg)

    by_step = {}
    for rid in run_ids:
        rows = fetch_wandb_run(rid, keys=keys, project=project, api=api) if api else []
        if rid in olog_fallbacks:
            fp = olog_fallbacks[rid]
            orows, _ = parse_olog([fp])
            w_max, o_max = _max_step(rows), _max_step(orows)
            if o_max >= w_max:
                if o_max < 0:
                    _emit("  %s: both W&B and .o-log empty, skipping" % rid)
                    continue
                _emit("  %s: .o-log %d rows, steps [%d, %d] (W&B reached %s)" % (
                    rid, len(orows), orows[0]["_step"], orows[-1]["_step"],
                    str(w_max) if w_max >= 0 else "empty"))
                rows = orows
            else:
                _emit("  %s: W&B reaches step %d > .o-log %d; keeping W&B" % (
                    rid, w_max, o_max))
        elif rows:
            _emit("  %s: %d rows, steps [%s, %s]" % (
                rid, len(rows), rows[0].get("_step"), rows[-1].get("_step")))
        if not rows:
            _emit("  %s: no rows, skipping" % rid)
            continue
        for r in rows:
            s = r.get("_step")
            if s is not None:
                by_step[int(s)] = r
    return [by_step[s] for s in sorted(by_step)]
