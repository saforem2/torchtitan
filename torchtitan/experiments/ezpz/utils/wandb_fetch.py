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

import os
import re

PROJECT = "aurora_gpt/torchtitan.ezpz.train"

# Metric keys recoverable from a PBS .o per-step stdout line (see OLOG_STEP_RE).
# Cap on bytes read per .o log by parse_olog. Guards against crash-spew logs
# (multi-GB, near-zero step lines) stalling every consumer. 256 MiB holds far
# more step lines than any chart or board displays.
OLOG_MAX_BYTES = int(os.environ.get("EZPZ_OLOG_MAX_BYTES", str(256 * 1024 * 1024)))

OLOG_KEYS = (
    "_step",
    "loss_metrics/global_avg_loss",
    "grad_norm",
    "throughput(tps)",
    "tflops",
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
    "tflops",
    "mfu(%)",
)

# ---------------------------------------------------------------------------
# Non-torchtitan producers log the same QUANTITIES under different key names.
# The v1 MDS (Megatron-DeepSpeed) chain is the live example: it predates
# torchtitan and its runs sit in a different W&B project. Rather than teach
# every consumer a second vocabulary, fetch a run with its OWN key names and
# rename them to the canonical torchtitan ones on the way out, so
# concat_chain / prod_dash / the plotters need no per-source branching.
#
# Keys are {canonical: source}. A canonical key absent from a map is simply
# absent from those records (callers already tolerate missing metrics).
MDS_KEY_ALIASES = {
    "_step": "lm-loss-training/iteration",
    "loss_metrics/global_avg_loss": "lm-loss-training/lm loss",
    "grad_norm": "loss/grad_norm",
    "tflops": "throughput/tflops-lm",
    "n_tokens_seen": "lm-loss-training/consumed_train_tokens",
}
# MDS runs live in their own W&B project, not aurora_gpt/torchtitan.ezpz.train.
MDS_PROJECT = "aurora_gpt/AuroraGPT"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Per-step metric lines in PBS .o files look like:
#   [TS][I][.../metrics:526:log] step: 3300  loss:  2.61866  grad_norm:  0.1672
#   memory: 44.55GiB(69.63%)  tps: 349  tflops: 51.93  mfu: 17.41%
# ANSI escapes wrap each field -- strip them first. We keep the fields that map
# to OLOG_KEYS (loss/grad_norm/tps/tflops/mfu); memory is parsed only to anchor
# the regex.
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
    encountered (``step``/``loss``/``grad_norm``/``tps``/``tflops``/``mfu``) or
    ``{}`` if nothing parsed. Missing/unreadable paths are skipped.
    """
    by_step = {}
    last = {}
    for p in sorted(paths or []):
        try:
            f = open(p, errors="replace")
        except (FileNotFoundError, IsADirectoryError):
            continue
        # A crashed rank storm can make a console log ENORMOUS while containing
        # almost no step lines: the 20B constlr seat failed four times and wrote
        # ~745k traceback lines per crash, leaving 3.1G/2.6G/2.0G logs whose
        # step-line count is ZERO. Line-parsing all of that on every dashboard
        # refresh is what made prod_dash --app hang for 8+ minutes and time out
        # (8.3G total across the trainer logs; even a bare grep is 7s each).
        #
        # Step lines are emitted throughout training, so a prefix is
        # representative: read at most OLOG_MAX_BYTES and stop. Raise
        # EZPZ_OLOG_MAX_BYTES to parse more of a genuinely long healthy run.
        budget = OLOG_MAX_BYTES
        with f:
            for raw in f:
                budget -= len(raw)
                if budget < 0:
                    break
                m = OLOG_STEP_RE.search(_ANSI_RE.sub("", raw))
                if not m:
                    continue
                step = int(m.group("step"))
                tps = float(m.group("tps").replace(",", ""))
                mfu = float(m.group("mfu"))
                loss = float(m.group("loss"))
                grad_norm = float(m.group("grad_norm"))
                tflops = float(m.group("tflops"))
                by_step[step] = {
                    "_step": step,
                    "loss_metrics/global_avg_loss": loss,
                    "grad_norm": grad_norm,
                    "throughput(tps)": tps,
                    "tflops": tflops,
                    "mfu(%)": mfu,
                }
                last = {"step": step, "loss": loss, "grad_norm": grad_norm,
                        "tps": tps, "tflops": tflops, "mfu": mfu}
    records = [by_step[s] for s in sorted(by_step)]
    return records, last


def fetch_wandb_run(run_id, keys=OLOG_KEYS, project=PROJECT, api=None,
                    key_aliases=None):
    """Return one W&B run's history as a list of dicts for the given keys.

    ``keys`` controls cost: pass OLOG_KEYS for a cheap loss curve, ALL_KEYS for
    the full metric set. Returns ``[]`` if the run cannot be found (e.g. it
    logged to a different project) -- callers treat that as a cue to fall back
    to the .o log. Pass an existing ``api`` to reuse one ``wandb.Api()``.

    ``key_aliases`` ({canonical: source}, e.g. MDS_KEY_ALIASES) fetches a run
    that logs the same quantities under other names and renames them to the
    canonical keys, so a non-torchtitan producer needs no downstream special
    casing. Canonical keys with no alias entry are requested as-is.
    """
    if api is None:
        import wandb
        api = wandb.Api()
    try:
        run = api.run(project + "/" + run_id)
    except Exception:
        return []
    keys = list(keys)
    if not key_aliases:
        out = []
        for row in run.scan_history(keys=keys):
            out.append({k: row.get(k) for k in keys})
        return out
    # Ask W&B for the SOURCE names, emit the canonical ones.
    src_of = {k: key_aliases.get(k, k) for k in keys}
    out = []
    for row in run.scan_history(keys=list(src_of.values())):
        out.append({k: row.get(s) for k, s in src_of.items()})
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
    key_aliases=None,
    olog_wins=None,
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

    ``key_aliases`` (see ``fetch_wandb_run``) lets a chain whose producer logs
    other key names -- the v1 MDS chain, ``MDS_KEY_ALIASES`` -- come back under
    the canonical names. It applies to the W&B fetch only: a Megatron run has
    no torchtitan-format .o line to parse.

    ``olog_wins`` is an iterable of run-ids for which the .o log, not W&B, is
    authoritative on steps both sources hold. Only needed when two jobs for one
    chain trained the same steps CONCURRENTLY and W&B kept the run whose
    checkpoints did not survive -- see the block below for the measured
    20b_v2_256 case. Default (empty) keeps W&B winning everywhere.
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
        rows = (
            fetch_wandb_run(rid, keys=keys, project=project, api=api,
                            key_aliases=key_aliases)
            if api else []
        )
        if rid in olog_fallbacks:
            fp = olog_fallbacks[rid]
            orows, _ = parse_olog([fp])
            w_max, o_max = _max_step(rows), _max_step(orows)
            if w_max < 0 and o_max < 0:
                _emit("  %s: both W&B and .o-log empty, skipping" % rid)
                continue
            # UNION, not replace. This used to do `rows = orows` whenever the
            # .o log reached at least as far as W&B, which DISCARDED any W&B
            # history below the log's first step: pointing 20b-256's cxlt0tpe
            # (W&B 6151..6896) at a log covering 6801..7600 closed one gap and
            # opened a new 510-step one at 6295..6805. The two sources describe
            # the same steps of the same run, and the by_step dict below already
            # dedups, so keeping both is strictly better -- each covers what the
            # other missed.
            #
            # WHICH SOURCE WINS ON A SHARED STEP is per-run, not global. The
            # default is W&B (it is the synced record, and normally the .o log
            # is the same process's stdout -- identical values, so the choice
            # is moot). But when two jobs for one chain both seat and train the
            # SAME steps concurrently, the two sources describe DIFFERENT
            # TRAJECTORIES, and the correct answer is whichever one wrote the
            # checkpoints that survived on disk. That is not knowable from W&B.
            #
            # MEASURED case, 20b_v2_256 2026-07-10: W&B run 6yr6ivh4 (steps
            # 3101..3602, 12:09-18:01) and the job behind
            # multi-autoretry-8648363 (3401..4297, 11:07-22:50) both ran. Their
            # loss disagrees on the 3401..3602 overlap by up to 0.0415, mean
            # 0.023, and the SIGN FLIPS -- two diverging trajectories, not
            # rounding. Checkpoint mtimes (step-3600 18:40 ... step-4200
            # Jul11 02:42) continue long past 6yr6ivh4's 18:01 death, so the
            # .o run owns the surviving weights and W&B holds the orphan.
            #
            # `olog_wins` names the run-ids where the .o log is authoritative.
            # Keep it EMPTY unless checkpoint mtimes prove the W&B run is the
            # loser; the default is right everywhere else.
            if orows:
                prefer_olog = rid in (olog_wins or ())
                lo = {int(r["_step"]): r for r in orows
                      if r.get("_step") is not None}
                hi = {int(r["_step"]): r for r in rows
                      if r.get("_step") is not None}
                if prefer_olog:
                    merged = dict(hi)
                    merged.update(lo)
                else:
                    merged = dict(lo)
                    merged.update(hi)
                _emit("  %s: %d W&B + %d .o-log rows -> %d union [%d, %d]%s" % (
                    rid, len(rows), len(orows), len(merged),
                    min(merged), max(merged),
                    "  (.o WINS overlap: concurrent-job collision)"
                    if prefer_olog else ""))
                rows = [merged[s] for s in sorted(merged)]
        elif rows:
            _emit("  %s: %d rows, steps [%s, %s]" % (
                rid, len(rows), rows[0].get("_step"), rows[-1].get("_step")))
        if not rows:
            _emit("  %s: no rows, skipping" % rid)
            continue
        # LAST RUN LISTED WINS a contested step. This is deliberate -- a
        # crashed run's overlapping steps get re-trained by its relaunch, and
        # the relaunch's values are the surviving trajectory -- but it means
        # `run_ids` ORDER IS SEMANTIC, not cosmetic. The lists in
        # trajectories.py are chronological; keep them that way.
        #
        # MEASURED consequence (2026-08-17): appending the older run djmhgmmq
        # (Aug 3) to the END of 20b_v2_256 put its crashed values on top of
        # 2ktrz29u's (Aug 5) re-trained ones, moving loss at steps 7,510-7,590
        # from ~2.33 to ~2.47. Nothing errored; it surfaced only by diffing a
        # re-exported store against the committed one.
        for r in rows:
            s = r.get("_step")
            if s is not None:
                by_step[int(s)] = r
    return [by_step[s] for s in sorted(by_step)]
