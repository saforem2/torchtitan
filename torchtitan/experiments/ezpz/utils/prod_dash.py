#!/usr/bin/env python3
"""Live production-training loss dashboard (kitcat inline / SVG / text board).

Companion to ``rl/scripts/grpo_lora_agpt2b_patches/rl_dash3.py`` but for the
main AuroraGPT pre-training chains instead of GRPO reward. Overlays the loss
curve of every production chain (2B/20B/80B, all node counts) plus any active
experiment forks, and highlights whichever chain is currently training.

Data collection runs REMOTE over SSH (the cluster has W&B, the PBS .o logs, and
qstat); rendering happens LOCAL. The chain list is derived from
``utils/trajectories.py`` (the single source of truth), so it never drifts from
the committed charts. Active experiments (e.g. the constant-LR fork, a HEAD
migration rehearsal) are auto-discovered by checkpoint dir.

Modes:

    # 1. Live kitcat loop (run in a kitty terminal; rl_dash3 twin)
    /tmp/kitcat-venv/bin/python prod_dash.py

    # 2. Headless SVG/PNG snapshot for docs/production/
    python prod_dash.py --svg /path/to/production_loss_live.svg

    # 3. Stall-aware text status board (no matplotlib needed)
    python prod_dash.py --board

    # one live frame then exit (curve + board), for a quick look
    python prod_dash.py --once

    # markers on the curves (any mode above): one shape per chain, or a
    # single shape for all of them
    python prod_dash.py --marker auto
    python prod_dash.py --svg out.svg --marker circle

    # 4. Textual multi-metric TUI (loss / grad_norm / tps / tflops / mfu),
    #    switchable per-metric tabs, live board, auto-refresh. Opt-in; needs
    #    `uv pip install textual textual-plotext` (pure-python, torch-safe).
    #    Falls back to --board if textual isn't installed.
    python prod_dash.py --app

Env:
    PD_INTERVAL   live-loop poll seconds (default 30)
    PD_LOCAL=1    read the local FS instead of SSH (run this ON the cluster)
    PD_SSH        ssh target (default 'aurora')
    PD_SOCK       ssh ControlPath. EMPTY by default so ssh uses the
                  ControlMaster/ControlPath from ~/.ssh/config (which has
                  ControlPersist, i.e. it revives a dead master). Only set
                  this to pin a specific socket.
    PD_BACKBONE_TTL  cluster-side W&B backbone cache TTL, seconds (default 900)
    PD_FRESH=1    force a W&B backbone rebuild this call
    PD_LIVE_WINDOW  seconds since last .o-log write to count a chain "live"
                    (default 300)
    PD_REPO       cluster repo path (default the AuroraGPT flare checkout)
    PD_MARKER     marker style for the curves: none (default) | auto | dot |
                  circle | square | triangle | diamond | x | plus | star.
                  "auto" gives each chain its own marker, cycled in the same
                  order as its color. Also settable with --marker <style>.
    PD_MARKER_EVERY  approx. markers drawn per curve (default 24). Only
                  applies when a marker style is active.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import zlib

# W&B project (also defined inside the _AGG remote script; kept here for the
# board's base-URL header).
PROJECT = "aurora_gpt/torchtitan.ezpz.train"
INTERVAL = float(os.environ.get("PD_INTERVAL", "30"))
LOCAL = os.environ.get("PD_LOCAL") == "1"
SSH_TGT = os.environ.get("PD_SSH", "aurora")
# Empty by default: let ssh resolve ControlMaster/ControlPath from ~/.ssh/config
# so a dead master is transparently re-established. Hardcoding
# -S /tmp/aurora-master.sock overrode the config and, once that hand-started
# master died, every call fell through to Aurora MFA and the dashboard broke.
SOCK = os.environ.get("PD_SOCK", "")
# Backbone = per-chain loss history from W&B. Completed steps never change, so
# a long TTL is safe; the live tip (moving step/loss of a running job) comes
# from the cheap per-call live layer, not the backbone. Default 1h.
BACKBONE_TTL = int(os.environ.get("PD_BACKBONE_TTL", "3600"))
LIVE_WINDOW = float(os.environ.get("PD_LIVE_WINDOW", "300"))
# Hide experiment forks whose newest .o log is older than this (default 7d);
# PD_SHOW_ALL=1 shows every experiment ever run.
EXP_MAX_AGE = int(os.environ.get("PD_EXP_MAX_AGE", str(7 * 86400)))
SHOW_ALL = os.environ.get("PD_SHOW_ALL") == "1"
# Cold W&B backbone build takes ~14 min (scan_history over ~90 runs: one pass
# for the shared metrics + one for lr); the cheap cache-hit path is ~25 s. Give the SSH call room for a cold build so a first
# call doesn't die at the finish line, then cache-hits are instant.
# Must exceed the COLD-BUILD time, not the warm one. Measured 2026-08-20: a
# full 7-chain build takes ~825 s now that lr is fetched in its own
# scan_history pass per run (it was ~400 s before). The old 600 s default sat
# BELOW that, so every cold rebuild over SSH was killed at the 10-minute mark
# and silently produced "0 live / 0 chains" -- which reads as a data problem
# and is really a stopwatch. Warm cache hits are ~25 s and unaffected.
SSH_TIMEOUT = float(os.environ.get("PD_SSH_TIMEOUT", "1800"))
REPO = os.environ.get(
    "PD_REPO",
    "/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz",
)

# Two hue-PARALLEL 12-color palettes. A chain's color index is crc32(key)%12,
# so the SAME chain keeps the SAME hue family across themes -- only the tone
# shifts to stay readable against the background. COLORS_LIGHT is the historical
# palette (Vega category set); COLORS_DARK is a brighter/lighter parallel tuned
# for a dark terminal. Light mode reuses COLORS_LIGHT verbatim, so it is a strict
# no-op vs. the pre-theme behavior; only dark mode changes. See draw_curves.
COLORS_LIGHT = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2",
    "#ff9da6", "#9d755d", "#bab0ac", "#edc948", "#b07aa1", "#86bcb6",
]
COLORS_DARK = [
    "#6ea8dc", "#ffa94d", "#7bc96f", "#ff7b7b", "#5fd0c8", "#d29fd8",
    "#ffc0c8", "#c79a6a", "#d5cfca", "#f6e05e", "#d3a0ce", "#a8ded6",
]
# Back-compat alias: any external caller importing COLORS gets the light set,
# matching the pre-theme default. draw_curves selects the palette at render time.
COLORS = COLORS_LIGHT

# Marker styles. Default is "none" (pure lines) -- the historical behavior and
# still the right choice for a dense canonical chain, where a marker per point
# would be a solid bar. The others earn their keep when curves overlap in
# color-ambiguous ways, or when a chain is so sparse the line reads as a
# near-flat segment and you want to see where the samples actually are.
#
# Cycled per chain in the same crc32 order as the palette, so a given chain
# keeps the same marker across refreshes and across step/tokens axes.
MARKERS_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]
MARKER_MODES = {
    "none": None,          # lines only (default)
    "auto": "cycle",       # per-chain marker from MARKERS_CYCLE
    "dot": ".",
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "x": "X",
    "plus": "P",
    "star": "*",
}
# Points drawn per curve when markers are on. A canonical chain carries ~600
# downsampled points; drawing all of them as markers hides the line under ink,
# so matplotlib's markevery thins them to roughly this many.
MARKER_TARGET = int(os.environ.get("PD_MARKER_EVERY", "24"))


def detect_dark_background():
    """Best-effort: is the terminal/output background dark? Returns True/False.

    Precedence:
      1. PD_THEME=dark|light  -- explicit override, always wins.
      2. COLORFGBG='fg;bg'    -- set by many terminals (rxvt, konsole, some
         iTerm/kitty setups). The trailing field is the background color index;
         0-6 and 8 are dark, 7/9-15 (and '15' white) are light. If the value is
         'fg;;bg' (3 fields) the middle is a decoration color -- take the last.
    Defaults to False (light) when nothing is set -- the historical behavior, so
    an unconfigured terminal renders exactly as before (COLORS_LIGHT)."""
    override = os.environ.get("PD_THEME", "").strip().lower()
    if override in ("dark", "light"):
        return override == "dark"
    cfb = os.environ.get("COLORFGBG", "").strip()
    if cfb:
        parts = cfb.split(";")
        try:
            bg = int(parts[-1])
        except ValueError:
            return False
        # ANSI background index: 0-6 + 8 are dark tones, 7/9-15 are light.
        return bg <= 6 or bg == 8
    return False

# ---------------------------------------------------------------------------
# Remote aggregator: everything that needs cluster-side data lives here. It is
# base64'd and piped to the cluster's .venv python (which has wandb). Emits a
# single JSON object on stdout. The W&B "backbone" (full loss history per
# chain) is expensive, so it is cached on the cluster at /tmp and only rebuilt
# when older than BACKBONE_TTL or when PD_FRESH=1. The cheap "live layer"
# (qstat states + tails of currently-running .o logs) runs every call.
# ---------------------------------------------------------------------------
_AGG = r'''
import json, os, re, sys, glob, time, subprocess, importlib.util, statistics

REPO = %(repo)r
TTL = %(ttl)d
FRESH = %(fresh)d
SHOW_ALL = %(show_all)d
EXP_MAX_AGE = %(exp_max_age)d
# Injected: this aggregator runs REMOTELY as a standalone string, so it cannot
# see the local module's constants. Referencing LIVE_WINDOW without this line
# raises NameError inside live_layer and the whole payload is lost (the board
# then renders "0 live / 0 chains").
LIVE_WINDOW = %(live_window)f
USER = os.environ.get("USER", "foremans")
PROJECT = "aurora_gpt/torchtitan.ezpz.train"
CACHE = "/tmp/prod_dash_backbone_%%s.json" %% USER
os.chdir(REPO)

# Progress goes to STDERR (stdout is reserved for the single JSON payload the
# local side parses). The local fetch() streams stderr live so a cold build
# is no longer a silent multi-minute hang. Each line is timestamped + tagged
# [prod_dash].
_T0 = time.time()
def _log(msg):
    sys.stderr.write("[prod_dash +%%5.1fs] %%s\n" %% (time.time() - _T0, msg))
    sys.stderr.flush()

# --- trajectories.py + wandb_fetch.py imported standalone. Both are
# dependency-light (stdlib + lazy wandb, NO numpy/matplotlib/torch), so
# spec-loading them by path keeps this remote aggregator off the heavy deps the
# cluster .venv would otherwise pull. wandb_fetch is the SAME fetch/concat logic
# the chart plotters use, so the board and the committed charts can't drift.
def _spec_load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

traj = _spec_load("traj", "torchtitan/experiments/ezpz/utils/trajectories.py")
wf = _spec_load("wandb_fetch", "torchtitan/experiments/ezpz/utils/wandb_fetch.py")
CANON = [t for t in traj.TRAJECTORIES if t.get("cls") in ("live", "wandb_only")]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CKPT_RE = re.compile(r"checkpoints/(agpt-[A-Za-z0-9._-]+?)(?:/|\s|$)")

def _downsample(pairs, n=600):
    if len(pairs) <= n:
        return pairs
    k = (len(pairs) + n - 1) // n
    return pairs[::k]

# short chart-name -> W&B/olog metric key. mfu(%%) is %%-escaped because this
# whole _AGG string is percent-formatted in fetch(). "loss" stays first so the
# default view + the back-compat "curve" both key off it.
_METRIC_MAP = {
    "loss": "loss_metrics/global_avg_loss",
    "grad_norm": "grad_norm",
    "tps": "throughput(tps)",
    "tflops": "tflops",
    "mfu": "mfu(%%)",
    # lr is W&B-ONLY: the .o per-step line has no LR field, so parse_olog can
    # never supply it. It is fetched as a SEPARATE scan_history call rather
    # than added to OLOG_KEYS -- see _wandb_records.
    "lr": "lr",
}

def _series_from_records(records):
    """Build {short_metric: [[step, value], ...]} from wandb_fetch records.

    Downsamples the record LIST once (all metrics share the same step grid --
    same W&B row / same .o line), then projects each metric and drops steps
    where that metric is missing (None). Keeps every metric step-aligned."""
    sampled = _downsample(records)
    out = {}
    for short, key in _METRIC_MAP.items():
        pairs = [[r["_step"], r[key]] for r in sampled
                 if r.get("_step") is not None and r.get(key) is not None]
        out[short] = pairs
    return out

# The v1 MDS (Megatron-DeepSpeed) reference chain. It is FROZEN -- 154,391
# iterations, finished long ago -- and lives in a DIFFERENT W&B project
# (MDS_PROJECT = aurora_gpt/AuroraGPT, keys renamed by MDS_KEY_ALIASES:
# `lm-loss-training/lm loss` etc). It is read from the committed CSV instead
# of a cross-project scan_history: that project holds 8,098 runs and this
# chain spans 42 of them, for a curve that can never change again.
#
# The CSV is a FAITHFUL dump, not an approximation -- verified 2026-08-20
# against W&B run ov1dn10t: 1242/1242 loss values identical to the last bit,
# 0 differing, 0 missing. Prefer W&B only if the CSV is ever suspected stale.
MDS_CSV = os.path.join(
    REPO, "torchtitan/experiments/ezpz/docs/production/agpt/2b-mds",
    "loss_data/train_metrics.csv")
# tokens/iter is CONSTANT across all 3 MDS stages: GBS 6144 x seq 8192.
#
# CONFIRMED BIT-EXACT against W&B's own logged `consumed_train_tokens`
# (aurora_gpt/AuroraGPT, alias `lm-loss-training/consumed_train_tokens`):
# at step 154,391 W&B logs 7,770,753,466,368 tokens, and
# 154391 * 6144 * 8192 == 7,770,753,466,368 exactly. Sampled across three runs
# spanning the whole chain (jmxwb00s step 1-172, 5w7wqr9n 149751-153150,
# ov1dn10t 153151-154391) the implied tokens/step is 50,331,648.0 at EVERY
# point. Not an inference -- the run logged it.
#
# Do NOT use the 7770e9/140000 (~55.5M) figure: it divides the budget by
# 140,000 steps when the run reached it at 154,391, overshooting the endpoint
# by 798B. (Fixed in the three eval plotters, commit aec112640.)
MDS_GBS, MDS_SEQ = 6144, 8192
# CSV column -> canonical metric key. mfu is absent from the CSV (Megatron did
# not log it), so the MFU tab simply has no MDS curve; lr likewise.
MDS_COLS = {
    "lm_loss": "loss_metrics/global_avg_loss",
    "grad_norm": "grad_norm",
    "tflops": "tflops",
    "tps_per_gpu": "throughput(tps)",
}

def _mds_records():
    """Parse the frozen MDS CSV into the same record shape as W&B/olog.

    Returns [] when the file is absent so a checkout without it degrades to
    "no MDS curve" rather than breaking the whole backbone."""
    if not os.path.exists(MDS_CSV):
        return []
    import csv  # noqa: PLC0415  (stdlib, only needed on this path)
    out = []
    try:
        with open(MDS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rec = {"_step": int(row["iteration"])}
                except (KeyError, TypeError, ValueError):
                    continue
                for col, key in MDS_COLS.items():
                    v = row.get(col)
                    if v not in (None, ""):
                        try:
                            rec[key] = float(v)
                        except ValueError:
                            pass
                out.append(rec)
    except OSError as e:
        _log("  MDS csv unreadable (%%r) -- chain omitted" %% (e,))
        return []
    return out

def _olog_records(paths):
    """Concat records from .o logs via the shared parser, plus the final line's
    tip. Delegates to wandb_fetch.parse_olog; returns (records, last)."""
    return wf.parse_olog(paths)

def _all_ologs():
    """Every candidate PBS stdout log in the repo (autoretry + umbrella + logs/)."""
    out = []
    out += glob.glob(os.path.join(REPO, "*.o[0-9]*"))
    out += glob.glob(os.path.join(REPO, "logs", "*", "run.log"))
    out += glob.glob(os.path.join(REPO, "torchtitan/experiments/ezpz/scripts/oneoff/*.o[0-9]*"))
    return out

def _dir_index(logs):
    """Map ckpt-dir basename -> [log paths that reference it]. One grep pass."""
    idx = {}
    for p in logs:
        try:
            with open(p, errors="replace") as f:
                head = f.read(200000)  # ckpt dir is named early in the cmd banner
        except (FileNotFoundError, IsADirectoryError):
            continue
        m = CKPT_RE.search(ANSI.sub("", head))
        if m:
            idx.setdefault(m.group(1), []).append(p)
    return idx

def _wandb_records(run_ids, olog_fallbacks, project=None, key_aliases=None):
    """Concat a chain's full per-step W&B history via wandb_fetch.concat_chain
    (same robust olog-fallback rule the charts use: prefer the .o log when it
    reaches at least as far as W&B). Returns the list of records keyed by
    OLOG_KEYS (step + loss + grad_norm + tps + tflops + mfu) -- the caller
    projects to the per-metric series it needs (via _series_from_records).

    ``project``/``key_aliases`` default to the torchtitan project and no
    renaming. The v1 MDS chain overrides both: it lives in a different W&B
    project and logs Megatron key names, which concat_chain maps back to the
    canonical ones so every downstream consumer is unchanged."""
    olog_fallbacks = olog_fallbacks or {}
    # concat_chain resolves relative fallback paths against CWD; the aggregator
    # chdir's to REPO at startup, but be explicit so it works regardless.
    fb = {rid: (fp if os.path.isabs(fp) else os.path.join(REPO, fp))
          for rid, fp in olog_fallbacks.items()}
    records = wf.concat_chain(run_ids, olog_fallbacks=fb, keys=wf.OLOG_KEYS,
                              project=project or PROJECT,
                              key_aliases=key_aliases)
    # lr rides in a SEPARATE pass, merged by step. It must NOT join OLOG_KEYS:
    # scan_history(keys=[...]) returns only rows where EVERY key is present, so
    # one absent key drops the whole row. Measured 2026-08-19: all 6 synthetic
    # backfill runs have no `lr`, so folding lr into OLOG_KEYS returns 0 rows
    # for them -- deleting 3,408 backfilled points from every OTHER metric too
    # and silently reopening the chart gaps those runs exist to close.
    # A separate call means a run lacking lr loses only lr.
    try:
        by_step = {}
        for rid in run_ids or []:
            for row in wf.fetch_wandb_run(rid, keys=("_step", "lr"),
                                          project=project or PROJECT):
                st, lr = row.get("_step"), row.get("lr")
                if st is not None and lr is not None:
                    by_step[st] = lr          # later run wins, as elsewhere
        if by_step:
            for r in records:
                v = by_step.get(r.get("_step"))
                if v is not None:
                    r["lr"] = v
    except Exception as e:
        # Report, do NOT swallow. A bare `pass` here turned a diagnosable
        # failure into "lr is silently absent from every chain", which cost
        # several rebuild cycles to even localize (2026-08-19). lr stays
        # non-fatal, but it must say why it is missing.
        _log("  lr fetch failed (chart will omit it): %%r" %% (e,))
    return records

def _wandb_summary(run_ids):
    """Cheap per-chain W&B metadata (one api.run + .summary read per tried run,
    NO scan_history): tps/mfu/step, url, run-id, heartbeat epoch.

    The chain's LAST run-id is preferred for wandb_id/url, but some chains end
    on a run that never synced to W&B (logged to a different project or died
    mid-sync -- these appear in olog_fallbacks and 'Could not find run'). So we
    walk run_ids from newest to oldest and take tps/mfu/updated from the first
    one that BOTH loads AND has throughput in its summary. All best-effort."""
    if not run_ids:
        return {}
    out = {"wandb_id": run_ids[-1]}
    try:
        import wandb
        api = wandb.Api()
    except Exception:
        return out
    got_metrics = False
    for i, rid in enumerate(reversed(run_ids)):
        try:
            run = api.run(PROJECT + "/" + rid)
        except Exception:
            continue  # unsynced/missing run -> try the previous one
        if i == 0:  # the actual last run loaded: use its url/id
            try:
                out["wandb_url"] = run.url
            except Exception:
                pass
        try:
            s = run.summary
            tps, mfu = s.get("throughput(tps)"), s.get("mfu(%%)")
            if tps is not None:
                out["wb_tps"] = float(tps)
            if mfu is not None:
                out["wb_mfu"] = float(mfu)
            if s.get("_timestamp") is not None:
                out["wb_ts"] = float(s.get("_timestamp"))
        except Exception:
            pass
        try:
            hb = getattr(run, "heartbeatAt", None)
            if hb:
                import calendar
                t = time.strptime(hb.replace("Z", "").split(".")[0],
                                  "%%Y-%%m-%%dT%%H:%%M:%%S")
                out["updated_ts"] = calendar.timegm(t)
        except Exception:
            pass
        if "wb_tps" in out or "wb_mfu" in out:
            got_metrics = True
        # url from the last run + metrics from some run -> done. Otherwise keep
        # walking back to recover throughput from the last SYNCED run.
        if got_metrics and ("wandb_url" in out or i > 0):
            break
    if "updated_ts" not in out and "wb_ts" in out:
        out["updated_ts"] = out["wb_ts"]
    return out

def _last_job_from_paths(paths):
    """Max PBS job-id referenced by a chain's .o log filenames (*.oNNNNN)."""
    best = None
    for p in paths or []:
        m = re.search(r"\.o(\d+)$", p) or re.search(r"[^0-9](\d{6,})/run\.log$", p)
        if m:
            j = int(m.group(1))
            if best is None or j > best:
                best = j
    return str(best) if best is not None else None

_WBRUN_RE = re.compile(r"wandb\.ai/[\w./-]+/runs/([A-Za-z0-9]+)")

def _wandb_ids_from_paths(paths):
    """W&B run-ids parsed from a chain's .o logs, in first-seen order. EVERY
    run logs to W&B (the autoretry scripts print 'View run at .../runs/<id>'),
    so experiment forks discovered from logs -- not trajectories.py -- still
    get their runs. Ordered by log mtime so the newest run is last.

    The 'View run at' banner prints at wandb init (near the top) and again at
    teardown (near the bottom), so we scan only the head+tail (256KiB each) --
    autoretry .o logs can be hundreds of MB and reading them whole is slow."""
    CHUNK = 256 * 1024
    ids, seen = [], set()
    for p in sorted((x for x in paths or [] if os.path.exists(x)),
                    key=lambda x: os.path.getmtime(x)):
        try:
            sz = os.path.getsize(p)
            with open(p, "rb") as f:
                head = f.read(CHUNK)
                if sz > 2 * CHUNK:
                    f.seek(-CHUNK, os.SEEK_END)
                    tail = f.read(CHUNK)
                else:
                    tail = b""
            txt = ANSI.sub("", (head + b"\n" + tail).decode("utf-8", "replace"))
        except (FileNotFoundError, IsADirectoryError, OSError):
            continue
        for m in _WBRUN_RE.finditer(txt):
            rid = m.group(1)
            if rid not in seen:
                seen.add(rid)
                ids.append(rid)
    return ids

def build_backbone():
    _log("cold build: scanning PBS .o logs for ckpt-dir index ...")
    logs = _all_ologs()
    idx = _dir_index(logs)
    _log("indexed %%d log(s); fetching W&B history for %%d canonical chain(s) "
         "(scan_history, the slow part) ..." %% (len(logs), len(CANON)))
    canon_dirs = set()
    chains = {}
    for i, t in enumerate(CANON, 1):
        cd = t.get("ckpt_dir")
        base = os.path.basename(cd) if cd else None
        if base:
            canon_dirs.add(base)
        _log("  [%%d/%%d] %%s: pulling %%d W&B run(s) ..." %% (
            i, len(CANON), t.get("key", "?"),
            len(t.get("wandb_run_ids") or [])))
        records = _wandb_records(t.get("wandb_run_ids") or [],
                                 t.get("olog_fallbacks"),
                                 project=t.get("wandb_project"),
                                 key_aliases=t.get("wandb_key_aliases"))
        series = _series_from_records(records)
        # Distinguish sibling chains that share model+nodes (e.g. the canonical
        # 2b 512N vs the sqrt2-LR fork "2b_v2_512_lr3.22e-5") by appending the
        # key's suffix beyond the standard "<model>_<version>_<nodes>" form.
        std = "%%s_%%s_%%d" %% (t["model"], t["version"], t["num_nodes"])
        extra = t["key"][len(std):].lstrip("_") if t["key"].startswith(std) else ""
        label = "%%s %%dN" %% (t["model"], t["num_nodes"])
        if extra:
            label += " (%%s)" %% extra
        rec = {
            "label": label,
            "model": t["model"], "num_nodes": t["num_nodes"],
            "gbs": t["gbs"], "seq_len": t["seq_len"],
            "token_target": t["token_target"], "kind": "canonical",
            # Cumulative tokens already absorbed before this chain's step 1
            # (stage-2 chains seed from a finished chain's weights). Only the
            # tokens x-axis uses it; step axes are per-chain by definition.
            "prior_tokens": t.get("prior_tokens") or 0,
            "ckpt_base": base,
            # series: per-metric [[step, val]] for the Textual app; curve is the
            # back-compat loss-only alias that render_board/draw_curves/--svg read.
            "series": series, "curve": series.get("loss", []),
        }
        rec.update(_wandb_summary(t.get("wandb_run_ids") or []))
        # last-job from main-repo .o logs, PLUS the trajectory's olog_fallbacks
        # (whose filenames encode job ids, e.g. ...cont1.o8558549). Chains that
        # run from a sibling clone leave no .o log in the main repo, so the
        # fallback filenames are the only job-id source for them.
        lj_paths = list(idx.get(base, [])) + list(
            (t.get("olog_fallbacks") or {}).values())
        rec["last_job"] = _last_job_from_paths(lj_paths)
        chains[t["key"]] = rec
    # MDS: a frozen v1 reference read from CSV, not W&B. Added last among the
    # non-experiment chains so it sorts after the live ones. It has no ckpt_base
    # and no PBS job, so the live layer never touches it -- it is permanently
    # idle by construction, which is correct: it finished in 2026.
    _log("adding the frozen MDS reference chain from CSV ...")
    mds_recs = _mds_records()
    if mds_recs:
        mds_series = _series_from_records(mds_recs)
        chains["2b_v1_mds"] = {
            "label": "2b MDS (v1 ref)",
            "model": "2b", "num_nodes": 256,
            "gbs": MDS_GBS, "seq_len": MDS_SEQ,
            # 7.771T = 154391 * 6144 * 8192, its own completed budget.
            "token_target": 154391 * MDS_GBS * MDS_SEQ,
            "kind": "canonical", "ckpt_base": None,
            "prior_tokens": 0,
            "series": mds_series, "curve": mds_series.get("loss", []),
            "last_job": None,
        }
        _log("  MDS: %%d rows -> %%d plotted points" %% (
            len(mds_recs), len(mds_series.get("loss", []))))

    # active experiments: agpt-* ckpt dirs with a referencing .o log, not
    # canonical. Stale ones (last .o write older than EXP_MAX_AGE seconds) are
    # hidden by default so the board stays focused on the live picture; set
    # SHOW_ALL to include every experiment ever run.
    _log("canonical chains done; scanning for active experiment forks ...")
    now = time.time()
    for base, paths in idx.items():
        if base in canon_dirs:
            continue
        if not base.startswith("agpt-"):
            continue
        # skip obvious non-training smoke/verify/convert artifacts
        if any(w in base for w in ("asyncfix", "debugmodel")):
            continue
        try:
            age = now - max(os.path.getmtime(p) for p in paths if os.path.exists(p))
        except ValueError:
            age = None
        if not SHOW_ALL and (age is None or age > EXP_MAX_AGE):
            continue
        records, _ = _olog_records(paths)
        series = _series_from_records(records)
        if len(series.get("loss", [])) < 2:
            continue
        exp = {
            "label": base.replace("agpt-", ""),
            "model": "2b" if "-2b-" in base or base.endswith("-2b") else "?",
            "num_nodes": None, "gbs": None, "seq_len": 8192,
            "token_target": None, "kind": "experiment",
            "ckpt_base": base,
            "series": series, "curve": series.get("loss", []),
            "last_job": _last_job_from_paths(paths),
        }
        # Every run logs to W&B -- parse the run-id(s) from the fork's .o logs
        # (these forks aren't in trajectories.py) so they get the same
        # wandb/tps/mfu/updated enrichment as canonical chains.
        wb_ids = _wandb_ids_from_paths(paths)
        if wb_ids:
            exp.update(_wandb_summary(wb_ids))
        chains["exp:" + base] = exp
    # Cache the ckpt-base -> log-paths index so the per-call path can stat
    # staleness WITHOUT re-reading every log head over Lustre (that scan is
    # the expensive part: ~2 min for the whole repo). Only paths referenced by
    # a tracked chain are kept.
    tracked = {c.get("ckpt_base") for c in chains.values() if c.get("ckpt_base")}
    slim_idx = {b: p for b, p in idx.items() if b in tracked}
    _log("backbone built: %%d chain(s) total (%%d canonical + experiments)" %% (
        len(chains), len(CANON)))
    return {"chains": chains, "idx": slim_idx, "built_at": time.time()}

def _kick_detached_refresh():
    """Spawn a detached FRESH rebuild so the NEXT call is warm. The worker
    re-enters this same file with PD_FRESH=1 (-> the synchronous build branch),
    writes the cache, and exits. A lock file (touched here, ~build-duration
    validity) prevents piling up concurrent rebuilds."""
    lock = CACHE + ".refreshing"
    try:
        if os.path.exists(lock) and (time.time() - os.path.getmtime(lock)) < 900:
            return  # a refresh is already in flight
        open(lock, "w").close()
    except Exception:
        return
    cmd = ("cd %%s && PD_LOCAL=1 PD_FRESH=1 PD_QUIET=1 .venv/bin/python3 "
           "torchtitan/experiments/ezpz/utils/prod_dash.py --board "
           ">/tmp/prod_dash_refresh.log 2>&1; rm -f %%s" %% (REPO, lock))
    try:
        subprocess.Popen(["bash", "-c", cmd], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            os.remove(lock)
        except Exception:
            pass

def load_backbone():
    # FRESH (the --fresh flag / detached refresh worker) always builds inline.
    if FRESH:
        bb = build_backbone()
        try:
            json.dump(bb, open(CACHE, "w"))
        except Exception:
            pass
        return bb
    # Otherwise serve the cache. If it is stale, serve it ANYWAY (marked stale)
    # and kick a detached rebuild so the next call is warm -- an interactive
    # call must never block on the ~14 min W&B scan_history build.
    if os.path.exists(CACHE):
        try:
            bb = json.load(open(CACHE))
            age = time.time() - os.path.getmtime(CACHE)
            if age >= TTL:
                _log("serving cached backbone (%%dm old, STALE) + kicking a "
                     "detached refresh for the next call" %% (age // 60))
                _kick_detached_refresh()
                bb["stale"] = True
            else:
                _log("serving warm cached backbone (%%dm old)" %% (age // 60))
            return bb
        except Exception:
            pass
    # No cache at all (first-ever run): unavoidable synchronous build.
    _log("no backbone cache -> cold build (this is the ~14 min first-run "
         "wait; subsequent calls are instant from cache)")
    bb = build_backbone()
    try:
        json.dump(bb, open(CACHE, "w"))
    except Exception:
        pass
    return bb

def qstat_jobs():
    try:
        out = subprocess.run(["/opt/pbs/bin/qstat", "-u", USER],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    jobs = []
    for ln in out.splitlines():
        if not re.match(r"^\d+\.", ln):
            continue
        f = ln.split()
        if len(f) < 6:
            continue
        jobs.append({"id": f[0].split(".")[0], "queue": f[2],
                     "name": f[3], "state": f[-2]})
    return jobs

def next_jobs(chains):
    """Map queued/held (Q/H) PBS jobs to chains -> {ckpt_base: next_job_id}.
    Queued jobs have no .o log yet, so read `qstat -f` once per Q/H job for
    Job_Name + Submit_arguments (CKPT_DIR=, NHOSTS_TRAIN=). Match by ckpt-dir
    basename first, else model+nodes from the job name. Lowest id = soonest."""
    pending = [j for j in qstat_jobs() if j["state"] in ("Q", "H")]
    if not pending:
        return {}
    # (model, nodes) -> ckpt_base for canonical chains (fallback match)
    by_modelnodes = {}
    for ch in chains.values():
        if ch.get("kind") == "canonical" and ch.get("ckpt_base"):
            by_modelnodes[(ch.get("model"), ch.get("num_nodes"))] = ch["ckpt_base"]
    bases = {ch.get("ckpt_base") for ch in chains.values() if ch.get("ckpt_base")}
    out = {}
    for j in sorted(pending, key=lambda x: int(x["id"])):
        try:
            det = subprocess.run(["/opt/pbs/bin/qstat", "-f", j["id"]],
                                 capture_output=True, text=True,
                                 timeout=30).stdout
        except Exception:
            continue
        flat = det.replace("\n\t", "").replace("\n ", "")
        base = None
        mck = re.search(r"CKPT_DIR=([^,\s]+)", flat)
        if mck:
            cand = os.path.basename(mck.group(1).rstrip("/"))
            if cand in bases:
                base = cand
        if base is None:
            mnm = re.search(r"Job_Name = agpt-(\d+b)-", flat)
            mnh = re.search(r"NHOSTS_TRAIN=(\d+)", flat)
            if mnm and mnh:
                base = by_modelnodes.get((mnm.group(1), int(mnh.group(1))))
        if base and base not in out:  # first (lowest id) wins
            out[base] = j["id"]
    return out

# Sibling production clones some chains launch from -- their .o logs live in the
# clone, NOT the main REPO, so live_layer must search them too or clone-run jobs
# (e.g. 20b-256 from agpt-20b-n256, 20b-512 resume from agpt-20b-v2) show idle
# despite running. Discovered from each canonical trajectory's ckpt_dir root.
_RUNS = "/flare/AuroraGPT/foremans/runs"
_CLONE_DIRS = [
    _RUNS + "/agpt-2b-v2/torchtitan-ezpz",
    _RUNS + "/agpt-20b-v2/torchtitan-ezpz",
    _RUNS + "/agpt-20b-n256/torchtitan-ezpz",
    _RUNS + "/agpt-2b-constlr-from9200/torchtitan-ezpz",
]

def live_layer():
    """Per running job: resolve its ckpt dir + freshest (step, loss). Only the
    handful of logs belonging to CURRENT qstat jobs are read (cheap), unlike the
    full-repo head scan in build_backbone. Searches the main REPO AND the
    sibling production clones (chains launched from a clone leave their .o log
    there, not in REPO)."""
    jobs = qstat_jobs()
    live = {}  # ckpt_base -> {state, jobid, step, loss, tps, mfu, age}
    states = {}  # ckpt_base -> worst-known state (R>Q>H)
    rank = {"R": 3, "Q": 2, "H": 1, "E": 0}
    search_roots = [REPO] + _CLONE_DIRS
    for j in jobs:
        cand = []
        for root in search_roots:
            cand += glob.glob(os.path.join(root, "*.o" + j["id"]))
            cand += glob.glob(os.path.join(root, "logs", "*" + j["id"], "run.log"))
            cand += glob.glob(os.path.join(
                root, "torchtitan/experiments/ezpz/scripts/oneoff/*.o" + j["id"]))
            # UMBRELLA JOBS: one PBS job runs N trainers, each with its OWN ckpt
            # dir, writing to logs/multi-autoretry-<jobid>/trainer-*.console.log.
            # Without these the only log found is the umbrella .o, whose FIRST
            # ckpt-dir match wins -- so job 8744247 (5 trainers) marked exactly
            # one chain live and showed the other four "idle" while they were
            # actively stepping.
            cand += glob.glob(os.path.join(
                root, "logs", "*" + j["id"], "trainer-*.console.log"))
        # A per-trainer console log names ONE ckpt dir, but the umbrella .o
        # names ALL of them (it prints every trainer's config at startup), so
        # scan for EVERY match, not just the first.
        #
        # search() took only the first, which meant an umbrella job marked
        # exactly one arbitrary seat live and the other four idle -- even
        # though a running umbrella means every seat under it is running.
        # The per-trainer glob above was supposed to cover this, but those
        # console logs do not exist until the trainers actually start; during
        # the venv prestage (minutes, at 2.7 GB x 1310 nodes) the umbrella .o
        # is the ONLY log, and first-match-wins reappeared. Measured on job
        # 8764675: the .o names all 5 ckpt dirs; search() returned just
        # dolmino. finditer() returns all 5.
        bases = []
        for p in cand:
            try:
                head = ANSI.sub("", open(p, errors="replace").read(200000))
            except (FileNotFoundError, IsADirectoryError):
                continue
            for m in CKPT_RE.finditer(head):
                if (m.group(1), p) not in bases:
                    bases.append((m.group(1), p))
        if not bases:
            continue
        # Group each chain's logs so a chain's tip is read only from ITS OWN
        # log, never from a sibling trainer's.
        per_base = {}
        # A log that names MORE THAN ONE ckpt dir is a multiplexed log (the
        # umbrella .o, which prints every trainer's config). It is authoritative
        # for STATE -- the job is running, so every chain it lists is running --
        # but NOT for the per-step tip: its step lines are interleaved across
        # trainers, so reading a tip from it would attribute one trainer's
        # step/loss to all of them. Track those separately.
        shared = {p for p in {q for _, q in bases}
                  if len({b for b, q in bases if q == p}) > 1}
        for b, p in bases:
            per_base.setdefault(b, []).append(p)
        for base, paths in per_base.items():
            if rank.get(j["state"], 0) >= rank.get(states.get(base, "E"), 0):
                states[base] = j["state"]
            if j["state"] != "R":
                continue
            own = [p for p in paths if p not in shared]
            if not own:
                # Only a multiplexed log so far -- the trainers have not written
                # their own console logs yet (venv prestage takes minutes at
                # 1310 nodes). The chain IS running; we just cannot say at which
                # step. Leave state=R with no tip rather than inventing one.
                continue
            _, last = _olog_records(own)
            if not last:
                continue
            paths = own
            try:
                age = time.time() - max(os.path.getmtime(p) for p in paths
                                        if os.path.exists(p))
            except Exception:
                age = None
            # A trainer can finish (or hang) while its umbrella job keeps
            # running. Its console log then stops advancing, and a stale tip
            # would report the chain live forever. Label it "stale" -- what we
            # actually observed (log not advancing) rather than "done", which
            # would assert a clean finish we cannot see from here.
            if age is not None and age > LIVE_WINDOW:
                states[base] = "stale"
                continue
            live[base] = dict(last, jobid=j["id"], age=age)
    return states, live

bb = load_backbone()
idx = bb.get("idx", {})  # cached ckpt-base -> [log paths]; stat-only, no re-read
states, live = live_layer()
nextj = next_jobs(bb["chains"])
now = time.time()
chains_out = {}
for key, ch in bb["chains"].items():
    base = ch.get("ckpt_base")
    paths = idx.get(base, [])
    try:
        ch["log_age"] = round(now - max(os.path.getmtime(p)
                              for p in paths if os.path.exists(p)), 1) if paths else None
    except Exception:
        ch["log_age"] = None
    ch["next_job"] = nextj.get(base)
    # Apply the experiment-staleness filter HERE (emit time) so it works even
    # when reading a cache built before the filter existed. Canonical chains
    # are always shown; a live job overrides staleness.
    if (ch.get("kind") == "experiment" and not SHOW_ALL
            and base not in live
            and (ch["log_age"] is None or ch["log_age"] > EXP_MAX_AGE)):
        continue
    ch["queue_state"] = states.get(base)
    lv = live.get(base)
    if lv:
        ch["live_tip"] = lv  # freshest running-job (step, loss, tps, mfu, age)
    # latest known step/loss for the board (prefer live tip, else curve tail)
    if lv:
        ch["latest_step"], ch["latest_loss"] = lv["step"], lv["loss"]
    elif ch["curve"]:
        ch["latest_step"], ch["latest_loss"] = ch["curve"][-1]
    else:
        ch["latest_step"], ch["latest_loss"] = None, None
    if ch["latest_step"] and ch.get("token_target") and ch.get("gbs"):
        toks = ch["latest_step"] * ch["gbs"] * ch["seq_len"]
        ch["pct_target"] = round(100.0 * toks / ch["token_target"], 2)
        ch["tokens"] = toks
    else:
        ch["pct_target"] = None
        ch["tokens"] = None
    chains_out[key] = ch

# A chain can be TRAINING RIGHT NOW and still be absent from the backbone: the
# backbone is a cached W&B/trajectory snapshot, so a chain whose first steps
# postdate the last rebuild (or that no trajectory lists) has no entry to
# annotate, and the loop above -- which only iterates backbone chains -- drops
# it silently. That is how the constlr-from9200 fork stayed invisible while
# trainer-3 of umbrella job 8744247 was actively stepping it.
# Synthesize a minimal row from the live tip so "what is running" is never a
# function of cache freshness. Marked kind=live-only: no curve/gbs, so it shows
# on the board but is skipped by the (curve-requiring) plot.
_known = {c.get("ckpt_base") for c in chains_out.values() if c.get("ckpt_base")}
for base, lv in live.items():
    if base in _known:
        continue
    chains_out["live:" + base] = {
        "label": base.replace("agpt-", ""),
        "model": "2b" if "-2b-" in base else ("20b" if "-20b-" in base else "?"),
        "num_nodes": None, "gbs": None, "seq_len": None, "token_target": None,
        "kind": "live-only", "ckpt_base": base,
        "series": {}, "curve": [],
        "queue_state": states.get(base), "live_tip": lv,
        "latest_step": lv.get("step"), "latest_loss": lv.get("loss"),
        "pct_target": None, "tokens": None,
        "log_age": lv.get("age"), "last_job": lv.get("jobid"), "next_job": None,
    }

bb["chains"] = chains_out
bb.pop("idx", None)  # don't ship the path index to the client
bb["built_age"] = round(now - bb["built_at"], 1)
bb["generated_at"] = time.time()
print(json.dumps(bb))
'''


# Client-side view of the aggregator's cache. The CACHE constant inside _AGG
# is part of the REMOTE script string, so the client cannot see it -- this
# duplicates the path deliberately rather than importing it.
LOCAL_CACHE = "/tmp/prod_dash_backbone_%s.json" % os.environ.get(
    "USER", "foremans")


def _cached_payload(why: str) -> dict | None:
    """Serve the last good backbone when the aggregator cannot be reached.

    The aggregator already has stale-while-revalidate logic, but it runs on the
    far side of the ssh hop -- so when SSH ITSELF is what failed (cluster down,
    maintenance, no network) that logic never executes and a perfectly good
    local cache sits unread while fetch() returns an empty payload. Observed
    2026-08-24 during an ALCF outage: a 17h-old 9-chain cache was on disk the
    whole time the dashboard rendered nothing.

    Returns None when there is no cache, so the caller still reports the real
    error rather than pretending success.
    """
    if not os.path.exists(LOCAL_CACHE):
        return None
    try:
        bb = json.load(open(LOCAL_CACHE))
    except Exception as e:
        sys.stderr.write("prod_dash: local cache unreadable (%s)\n" % e)
        return None
    if not bb.get("chains"):
        return None
    age_h = (time.time() - os.path.getmtime(LOCAL_CACHE)) / 3600.0
    # Mark it, loudly. A stale dashboard that looks live is worse than no
    # dashboard -- every consumer of this payload should be able to say so.
    bb["stale"] = True
    bb["stale_reason"] = why
    bb["stale_age_hours"] = round(age_h, 1)
    sys.stderr.write(
        "prod_dash: %s -- serving LOCAL CACHE from %.1fh ago "
        "(%d chains). Numbers are NOT live.\n"
        % (why, age_h, len(bb["chains"])))
    return bb


def fetch(stderr_cb=None) -> dict:
    """Run the remote aggregator and return the parsed backbone+live payload.

    stdout carries the single JSON object; the aggregator's stderr is the
    ``[prod_dash +Ns]`` progress stream (cold build ~14 min). Behavior:
      - stderr_cb given: capture stderr and call stderr_cb(line) per line (the
        Textual app pipes this into a RichLog); JSON returned at the end.
      - stderr_cb None, PD_QUIET=1: discard stderr (detached refresh worker).
      - stderr_cb None otherwise: inherit stderr so progress streams straight
        to the terminal live (the kitcat/board CLI path).
    """
    script = _AGG % {"repo": REPO, "ttl": BACKBONE_TTL,
                     "fresh": 1 if os.environ.get("PD_FRESH") == "1" else 0,
                     "show_all": 1 if SHOW_ALL else 0,
                     "exp_max_age": EXP_MAX_AGE,
                     "live_window": LIVE_WINDOW}
    if LOCAL:
        cmd = [sys.executable, "-c", script]
    else:
        b64 = base64.b64encode(script.encode()).decode()
        remote = (
            "P=$([ -x .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3); "
            "echo %s | base64 -d | $P" % b64
        )
        # SSH hardening so a DEAD ControlMaster socket fails fast instead of
        # hanging forever: without these, ssh silently falls back to a fresh
        # connection, hits Aurora's keyboard-interactive MFA, and blocks on the
        # password prompt on the inherited TTY -- which SSH_TIMEOUT cannot reap
        # (a process camped on a TTY read is not "done"). BatchMode=yes refuses
        # every interactive prompt (fail fast with "Permission denied" when the
        # socket is gone), ConnectTimeout bounds the TCP connect, and -n redirects
        # stdin from /dev/null so ssh can never grab the terminal. The live master
        # socket still multiplexes normally (no auth needed) when it is healthy.
        cmd = [
            "ssh", "-n",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=%d" % int(os.environ.get("PD_SSH_CONNECT_TIMEOUT", "10")),
        ]
        # Do NOT pass -S by default. An explicit -S OVERRIDES the ControlPath in
        # ~/.ssh/config, so `ControlPersist yes` does not apply to it: when a
        # hand-started `ssh -M -S /tmp/aurora-master.sock` master dies, nothing
        # revives it and every call here falls through to a fresh connection ->
        # Aurora MFA -> BatchMode refuses -> the dashboard just breaks, while the
        # config-managed master sits alive the whole time. Letting ssh resolve
        # ControlMaster/ControlPath from the config means a dead master is
        # transparently re-established. Set PD_SOCK to opt back in to a specific
        # socket.
        if SOCK:
            cmd += ["-S", SOCK]
        cmd += [SSH_TGT, "cd %s && %s" % (REPO, remote)]
    if stderr_cb is not None:
        # Stream stderr line-by-line to the callback while the JSON accumulates
        # on stdout. Drain BOTH pipes concurrently -- reading stderr to EOF
        # first would deadlock once the child fills the stdout pipe buffer with
        # the (large) JSON payload. A background thread collects stdout.
        import threading
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        except Exception as e:
            sys.stderr.write("prod_dash: aggregator launch failed: %s\n" % e)
            return _cached_payload("aggregator launch failed") or {"chains": {}}
        out_chunks = []
        t_out = threading.Thread(target=lambda: out_chunks.append(p.stdout.read()),
                                 daemon=True)
        t_out.start()
        for line in iter(p.stderr.readline, ""):
            if line:
                try:
                    stderr_cb(line.rstrip("\n"))
                except Exception:
                    pass
        try:
            p.wait(timeout=SSH_TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                stderr_cb("prod_dash: aggregator timed out after %ds" % int(SSH_TIMEOUT))
            except Exception:
                pass
            return _cached_payload("aggregator timed out") or {"chains": {}}
        t_out.join(timeout=10)
        stdout_text = out_chunks[0] if out_chunks else ""
    else:
        stderr_dest = subprocess.DEVNULL if os.environ.get("PD_QUIET") == "1" else None
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr_dest,
                               text=True, timeout=SSH_TIMEOUT)
        except subprocess.TimeoutExpired:
            sys.stderr.write(
                "prod_dash: aggregator timed out after %ds (raise PD_SSH_TIMEOUT "
                "for a cold build)\n" % int(SSH_TIMEOUT))
            return _cached_payload("aggregator timed out") or {"chains": {}}
        stdout_text = r.stdout or ""
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    if not LOCAL:
        # Report what we OBSERVED, then rank causes by likelihood -- do not
        # assert one. The previous text blamed a dead ControlMaster socket
        # unconditionally, which sent a real 8m20s diagnosis (2026-08-19) off
        # after `ssh -MNf` while the socket was healthy the whole time and the
        # actual cause was multi-GB crash-spew .o logs stalling parse_olog.
        # It also interpolated SOCK, which is EMPTY by default, emitting the
        # broken command `ssh -MNf -S  aurora`.
        tail = (stdout_text or "").strip()
        detail = ("  (aggregator stdout was empty)" if not tail
                  else "  last stdout line: %s" % tail.splitlines()[-1][:200])
        sock_hint = ("    ssh -MNf -S %s %s\n" % (SOCK, SSH_TGT) if SOCK
                     else "    ssh -O check %s   # then: ssh -fN %s\n"
                          % (SSH_TGT, SSH_TGT))
        sys.stderr.write(
            "prod_dash: no JSON from aggregator.\n"
            "%s\n"
            "Likely causes, cheapest check first:\n"
            "  1. The aggregator is SLOW, not broken -- a huge .o log makes\n"
            "     parse_olog scan for minutes. Find the big ones:\n"
            "       ssh %s \"du -sh %s/logs/*/trainer-*.console.log | sort -h | tail -3\"\n"
            "     Current per-file cap: EZPZ_OLOG_MAX_BYTES=%d bytes.\n"
            "  2. Cold backbone build exceeded PD_SSH_TIMEOUT=%ds -- raise it.\n"
            "  3. SSH really is down. Verify before assuming:\n"
            "%s"
            % (detail, SSH_TGT, REPO,
               int(os.environ.get("EZPZ_OLOG_MAX_BYTES", 256 * 1024 * 1024)),
               int(SSH_TIMEOUT), sock_hint)
        )
    else:
        sys.stderr.write("prod_dash: no JSON from aggregator\n")
    return _cached_payload("no JSON from aggregator") or {"chains": {}}


# ---------------------------------------------------------------------------
# Text status board (mode 3): no matplotlib needed, works from any python.
# ---------------------------------------------------------------------------
def render_board(payload) -> str:
    chains = payload.get("chains", {})
    now = payload.get("generated_at", time.time())

    def order(item):
        k, c = item
        return (c.get("kind") != "canonical", c.get("model") or "z",
                -(c.get("num_nodes") or 0), k)

    def _age(sec):
        if sec is None:
            return "-"
        if sec < 90:
            return "%.0fs" % sec
        if sec < 5400:
            return "%.0fm" % (sec / 60)
        if sec < 172800:
            return "%.0fh" % (sec / 3600)
        return "%.0fd" % (sec / 86400)

    rows = []
    header = ("chain", "state", "step", "loss", "% tgt", "tps", "mfu",
              "updated", "last", "next", "wandb")
    for key, c in sorted(chains.items(), key=order):
        st = c.get("queue_state") or ("run" if (c.get("log_age") or 1e9) < LIVE_WINDOW else "idle")
        tip = c.get("live_tip") or {}
        step = c.get("latest_step")
        loss = c.get("latest_loss")
        # tps/mfu: prefer the live running-job tip, else the last W&B summary.
        tps = tip.get("tps", c.get("wb_tps"))
        mfu = tip.get("mfu", c.get("wb_mfu"))
        # "updated" = W&B heartbeat age (cleaner than the filesystem log age).
        upd = c.get("updated_ts")
        rows.append((
            c.get("label", key)[:34],
            st,
            "-" if step is None else str(step),
            "-" if loss is None else "%.4f" % loss,
            "-" if c.get("pct_target") is None else "%.1f%%" % c["pct_target"],
            "-" if tps is None else "%.0f" % tps,
            "-" if mfu is None else "%.1f%%" % mfu,
            _age(now - upd) if upd else "-",
            c.get("last_job") or "-",
            c.get("next_job") or "-",
            c.get("wandb_id") or "-",
        ))
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    def fmt(r):
        return "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
    nlive = sum(1 for c in chains.values()
                if c.get("queue_state") == "R"
                or (c.get("log_age") or 1e9) < LIVE_WINDOW)
    bb_age = payload.get("built_age")
    stale = " (refreshing)" if payload.get("stale") else ""
    base_url = "https://wandb.ai/%s/runs/" % PROJECT
    out = [
        "AuroraGPT production loss board  [%s]  %d live / %d chains  (W&B backbone %s%s)"
        % (time.strftime("%H:%M:%S", time.localtime(now)), nlive, len(chains),
           "-" if bb_age is None else "%.0fm old" % (bb_age / 60), stale),
        "W&B: %s<wandb>   'updated' = W&B heartbeat age" % base_url,
        fmt(header),
        "  ".join("-" * w for w in widths),
    ]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Curve overlay (modes 1 & 2): shared by live-kitcat and headless-SVG.
# ---------------------------------------------------------------------------
def _apply_house_style(plt):
    """ambivalent + Iosevka, matching the committed production charts.

    Prefer the shared plot_style helper (single source of the house style).
    If it is unavailable (e.g. run from a python without the ezpz package),
    fall back to loading the ambivalent stylesheet + registering Iosevka
    directly, the way rl_dash3 does -- so styling is guaranteed either way.
    Prefer the 'Iosevka Term' variant if registered (cleaner in plots)."""
    styled = False
    try:
        from torchtitan.experiments.ezpz.utils.plot_style import apply_style
        apply_style()
        styled = True
    except Exception:
        pass
    if not styled:
        import glob as _glob
        import importlib.util as _ilu
        from matplotlib import font_manager as _fm
        for d in (os.path.expanduser("~/.local/share/fonts"),
                  os.path.expanduser("~/Library/Fonts"), "/usr/share/fonts"):
            for f in _glob.glob(os.path.join(d, "**", "Iosevka*.tt*"),
                                recursive=True):
                try:
                    _fm.fontManager.addfont(f)
                except Exception:
                    pass
        try:
            spec = _ilu.find_spec("ambivalent")
            for loc in (spec.submodule_search_locations or []):
                cand = os.path.join(loc, "stylefiles", "ambivalent.mplstyle")
                if os.path.isfile(cand):
                    plt.style.use(cand)
                    break
        except Exception:
            pass
    # Prefer 'Iosevka Term' when present (rl_dash3 convention).
    try:
        from matplotlib import font_manager as _fm
        names = {f.name for f in _fm.fontManager.ttflist
                 if "iosevka" in f.name.lower()}
        pick = next((n for n in ("Iosevka Term", "Iosevka") if n in names), None)
        if pick:
            plt.rcParams["font.family"] = [pick, "DejaVu Sans Mono", "monospace"]
    except Exception:
        pass


def _auto_marker_map(keys):
    """Assign a DISTINCT marker per chain, deterministically.

    Preference comes from crc32 (same hashing the palette uses, so a chain
    keeps its shape across refreshes), but a bare `crc32 % len(MARKERS_CYCLE)`
    collides readily: with 8 shapes and 7 live chains it is more likely than
    not, and it did -- 20b_v2_512 and 20b_v2_256 both drew "X", which defeats
    the point of `auto`. So probe forward to the next free shape.

    Iterating `sorted(keys)` keeps the outcome independent of dict order.
    Beyond len(MARKERS_CYCLE) chains some reuse is unavoidable; color still
    separates those.
    """
    n = len(MARKERS_CYCLE)
    out, used = {}, set()
    for key in sorted(keys):
        start = zlib.crc32(key.encode()) % n
        for off in range(n):
            cand = MARKERS_CYCLE[(start + off) % n]
            if cand not in used:
                break
        else:                      # more chains than shapes -- accept a repeat
            cand = MARKERS_CYCLE[start]
        used.add(cand)
        out[key] = cand
    return out


def _resolve_marker(mode, key, npts, auto_map=None):
    """Marker kwargs for one chain, or {} when markers are off.

    ``mode`` is a MARKER_MODES key; an unknown value falls back to "none"
    rather than raising, so a typo degrades to the historical rendering
    instead of killing a live dashboard loop.

    ``auto_map`` is the result of _auto_marker_map() over the full chain set;
    it is only consulted in "auto" mode. Passing it is what makes shapes
    distinct -- without it, "auto" falls back to the raw hash and may collide.

    ``markevery`` thins the drawn markers to ~MARKER_TARGET per curve. Without
    it a 600-point canonical chain draws 600 markers and the line disappears
    under them; with it a sparse 12-point fork still shows every point,
    because the stride floors at 1.
    """
    style = MARKER_MODES.get(mode)
    if style is None:
        return {}
    if style == "cycle":
        if auto_map and key in auto_map:
            style = auto_map[key]
        else:
            style = MARKERS_CYCLE[zlib.crc32(key.encode()) % len(MARKERS_CYCLE)]
    every = max(1, npts // MARKER_TARGET) if npts > MARKER_TARGET else 1
    return {"marker": style, "markevery": every, "markersize": 4.5,
            "markeredgewidth": 0.0}


def draw_curves(payload, figsize, save_path=None):
    """Render the overlay. ``figsize`` is (width_in, height_in) from
    size_figure() -- already fitted to the terminal window and clamped to the
    kitcat cell limit, so this function does no sizing math of its own."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_house_style(plt)
    plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none",
                         "axes.facecolor": "none"})
    # Theme: pick the hue-parallel palette for the background, and in dark mode
    # override the (otherwise dark) ambivalent text/axes color to a light tone so
    # labels/ticks/legend stay legible against the transparent-over-dark backdrop.
    # Light mode leaves the stylesheet untouched -- a strict no-op vs. before.
    dark = detect_dark_background()
    palette = COLORS_DARK if dark else COLORS_LIGHT
    if dark:
        fg = "#d5d5d5"
        plt.rcParams.update({
            "text.color": fg, "axes.labelcolor": fg, "axes.titlecolor": fg,
            "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg,
        })
    chains = payload.get("chains", {})
    dpi = 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # x-axis: "step" (default, faithful to rl_dash3) or "tokens" (every chain
    # shares the 4.67T olmo-mix target, so tokens = step * gbs * seq_len puts
    # them on a comparable footing despite very different step counts/batches).
    xaxis = os.environ.get("PD_XAXIS", "step")
    marker_mode = os.environ.get("PD_MARKER", "none")
    if marker_mode not in MARKER_MODES:
        _log("unknown PD_MARKER=%r; valid: %s. Falling back to 'none'."
             % (marker_mode, ", ".join(sorted(MARKER_MODES))))
        marker_mode = "none"
    # Assign over ALL chains, not just the ones that survive the len<2 /
    # missing-gbs filters below: a chain dropping out for one frame would
    # otherwise reshuffle everyone else's shapes on the next refresh.
    auto_map = (_auto_marker_map(chains.keys())
                if MARKER_MODES.get(marker_mode) == "cycle" else None)
    parts = []
    for key in sorted(chains):
        c = chains[key]
        curve = c.get("curve") or []
        tip = c.get("live_tip")
        if tip and (not curve or tip["step"] > curve[-1][0]):
            curve = curve + [[tip["step"], tip["loss"]]]
        if len(curve) < 2:
            continue
        toks_per_step = (c.get("gbs") or 0) * (c.get("seq_len") or 0)
        if xaxis == "tokens":
            # Skip chains with no known gbs (experiment forks) rather than
            # plotting their step count on a tokens axis -- mixing scales.
            if not toks_per_step:
                continue
            # Offset by tokens seen BEFORE this chain's step 1. A stage-2 chain
            # restarts its step counter at 1 but its weights already carry the
            # parent run's tokens; starting it at 0 on a CUMULATIVE axis claims
            # it saw its first token alongside the parent.
            prior_b = (c.get("prior_tokens") or 0) / 1e9
            xs = [prior_b + p[0] * toks_per_step / 1e9 for p in curve]  # billions
        else:
            xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        is_live = (c.get("queue_state") == "R"
                   or (c.get("log_age") is not None and c["log_age"] <= LIVE_WINDOW))
        is_exp = c.get("kind") == "experiment"
        color = palette[zlib.crc32(key.encode()) % len(palette)]
        alpha = 1.0 if is_live else 0.30
        lw = 2.4 if is_live else 1.2
        ls = "--" if is_exp else "-"
        z = 5 if is_live else 2
        label = ("* " if is_live else "") + c.get("label", key) + (
            "" if is_live else " (idle)")
        mk = _resolve_marker(marker_mode, key, len(xs), auto_map)
        ax.plot(xs, ys, ls, lw=lw, color=color, alpha=alpha, zorder=z,
                label=label, rasterized=len(xs) > 2000, **mk)
        if is_live:
            parts.append("%s=%.3f@%d" % (c.get("label", key), ys[-1], curve[-1][0]))
    ax.set_xlabel("tokens seen (billions)" if xaxis == "tokens"
                  else "training step (cumulative across resumes)")
    ax.set_ylabel("loss (global avg)")
    ax.set_title("AuroraGPT production loss" + ("  --  " + "  ".join(parts) if parts else ""))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        png = os.path.splitext(save_path)[0] + ".png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        print("Saved: %s" % save_path)
        print("Saved: %s" % png)
    else:
        plt.show()
    plt.close(fig)


def terminal_cells():
    """(cols, rows, cell_w_px, cell_h_px) from TIOCGWINSZ.

    Cells -- not pixels -- are the unit that actually constrains us: kitcat
    renders the figure to a PNG and lays it out as a grid of
    ceil(img_px / cell_px) terminal cells, and the kitty unicode-placeholder
    protocol can address at most 297 cells per axis. Sizing from pixels made
    that grid an uncontrolled derived quantity (see size_figure)."""
    cols, rows, xp, yp = 100, 30, 0, 0
    try:
        import array
        import fcntl
        import termios
        buf = array.array("H", [0, 0, 0, 0])
        fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, buf)
        rows, cols, xp, yp = buf
    except Exception:
        pass
    cols = cols or 100
    rows = rows or 30
    # ws_xpixel/ws_ypixel are 0 in tmux and over plain SSH; kitcat falls back to
    # an 8x16 cell scaled by the device-pixel ratio, so mirror that exactly --
    # if we assume a different cell size than kitcat uses, our cell budget is
    # wrong in precisely the situation that overflows.
    cw = int(xp // cols) if (xp and cols) else 0
    chh = int(yp // rows) if (yp and rows) else 0
    if not cw or not chh:
        scale = 1.0
        try:
            from kitcat.terminal_query import get_dpi_scale
            scale = float(get_dpi_scale()) or 1.0
        except Exception:
            if sys.platform == "darwin":
                scale = 2.0
        cw = cw or max(1, round(8 * scale))
        chh = chh or max(1, round(16 * scale))
    return cols, rows, cw, chh


# kitty unicode placeholders can address at most len(_DIACRITICS)==297 cells per
# axis. Exceeding it is a hard ValueError, not a clipped image.
KITCAT_MAX_CELLS = 297


def size_figure(reserved_rows=0, dpi=100):
    """Figure (width_in, height_in) that fits the CURRENT terminal window.

    Replaces the old pixel-based sizing, which had two failure modes:

      1. It fed TIOCGWINSZ *logical* pixels into figsize, but kitcat re-renders
         at dpi*device_pixel_ratio -- so on a HiDPI display the PNG came out ~2x
         larger than the window in each axis. The resulting placeholder grid
         overflowed both the pane (image wider than the window; the x tick
         wrapped to its own line) and, at wide window sizes, the 297-cell
         protocol limit outright:
             ValueError: image too large for unicode placeholders:
                         needs 76x402 cells, max is 297x297
      2. hpx*0.80 budgeted 80% of the window height for the plot while the
         status board above it is ~10 rows -- so figure + board exceeded the
         window and the terminal scrolled, carrying the title and legend off
         the top. (That is why neither was visible in the reported frame.)

    Sizing from cells fixes both: subtract what the board occupies, convert the
    remaining cells to device pixels, then to inches at the SAME dpi*scale
    kitcat will render with. Clamped to the 297-cell limit so the protocol
    error is unreachable by construction."""
    cols, rows, cw, chh = terminal_cells()
    scale = 1.0
    try:
        from kitcat.terminal_query import get_dpi_scale
        scale = float(get_dpi_scale()) or 1.0
    except Exception:
        if sys.platform == "darwin":
            scale = 2.0
    scale = max(1.0, min(3.0, scale))
    # Leave a 1-col right margin and 1 row of breathing room under the figure so
    # the next prompt/frame does not butt against it.
    # min() only -- a max() floor here has the same defect as a figsize floor:
    # it can request more cells than exist. Clamp at >=1 purely to keep the
    # arithmetic valid when the board fills the window.
    avail_cols = max(1, min(cols - 1, KITCAT_MAX_CELLS))
    avail_rows = max(1, min(rows - reserved_rows - 1, KITCAT_MAX_CELLS))
    # cells -> device px -> inches. kitcat renders at dpi*scale, so dividing the
    # device-pixel extent by (dpi*scale) yields a figsize whose rendered PNG is
    # exactly avail_cols x avail_rows cells.
    win = float(avail_cols * cw) / (dpi * scale)
    hin = float(avail_rows * chh) / (dpi * scale)
    # NO minimum-size floor. An earlier version clamped to max(4.0, w) /
    # max(2.5, h) "for readability"; a unit test over terminal geometries showed
    # that silently re-overflows an ordinary 80x24 SSH window (6.3x2.5in -> 79x16
    # cells requested against 80x14 available). A floor here defeats the entire
    # point of fitting: whatever number we invent is by definition not what fits.
    # A small window gets a small plot.
    return win, hin


def _init_kitcat():
    """kitcat inline backend + HiDPI crispness (ported from rl_dash3)."""
    import matplotlib
    matplotlib.use("module://kitcat")
    # HiDPI: render at device resolution so Retina text isn't upscaled/fuzzy.
    try:
        import array
        import fcntl
        import re as _re
        import termios
        import tty as _tty
        import select

        def _tty_query(seq, timeout=0.3):
            if "TMUX" in os.environ:
                seq = "\033Ptmux;" + seq.replace("\033", "\033\033") + "\033\\"
            try:
                fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            except OSError:
                return ""
            old = termios.tcgetattr(fd)
            try:
                _tty.setraw(fd)
                os.write(fd, seq.encode())
                buf = b""
                while True:
                    ready, _, _ = select.select([fd], [], [], timeout)
                    if not ready:
                        break
                    buf += os.read(fd, 64)
                    if buf.endswith(b"t") or len(buf) > 128:
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                os.close(fd)
            return buf.decode("ascii", errors="replace")

        m = _re.search(r"\033\[\d+;(\d+);(\d+)t", _tty_query("\033[16t"))
        dev = (int(m.group(1)), int(m.group(2))) if m else None
        buf = array.array("H", [0, 0, 0, 0])
        fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, buf)
        rows, _cols, xpix, ypix = buf
        log_ch = ypix / rows if (ypix and rows) else 0.0
        if dev and log_ch:
            backing = dev[0] / log_ch
        elif sys.platform == "darwin":
            backing = 2.0
        else:
            backing = 1.0
        backing = max(1.0, min(3.0, backing))
        if backing > 1.0:
            import kitcat.terminal_query as _ktq
            import kitcat.utils as _ku
            import kitcat.backend as _kb
            for _m in (_ktq, _ku, _kb):
                _m.get_dpi_scale = lambda: backing
            if dev:
                _ku.get_char_cell_width = lambda: max(1, int(round(dev[1])))
                _ku.get_char_cell_height = lambda: max(1, int(round(dev[0])))
    except Exception as e:
        print("kitcat HiDPI patch skipped:", e)


def main():
    global SHOW_ALL
    argv = sys.argv[1:]
    if "--all" in argv:            # include stale experiment forks
        SHOW_ALL = True
    if "--fresh" in argv:          # force a W&B backbone rebuild this call
        os.environ["PD_FRESH"] = "1"
    if "--tokens" in argv:         # x-axis in tokens instead of steps
        os.environ["PD_XAXIS"] = "tokens"
    if "--marker" in argv:         # marker style on the curves
        i = argv.index("--marker")
        if i + 1 >= len(argv):
            sys.stderr.write("prod_dash: --marker needs a style: %s\n"
                             % ", ".join(sorted(MARKER_MODES)))
            return
        style = argv[i + 1]
        if style not in MARKER_MODES:
            sys.stderr.write("prod_dash: unknown marker %r; valid: %s\n"
                             % (style, ", ".join(sorted(MARKER_MODES))))
            return
        os.environ["PD_MARKER"] = style
    if "--app" in argv:            # Textual multi-metric TUI (opt-in)
        try:
            import prod_dash_app
        except ImportError:
            sys.stderr.write(
                "prod_dash: textual not installed -- run\n"
                "  uv pip install textual textual-plotext\n"
                "falling back to --board.\n")
            print(render_board(fetch()))
            return
        prod_dash_app.run_app()
        return
    if "--board" in argv:
        print(render_board(fetch()))
        return
    if "--svg" in argv:
        i = argv.index("--svg")
        path = argv[i + 1]
        # Headless SVG/PNG: no terminal to fit, so use a fixed print-friendly
        # size rather than whatever window happens to be attached.
        draw_curves(fetch(), (12.0, 6.5), save_path=path)
        return
    once = "--once" in argv
    if not once:
        _init_kitcat()
    else:
        import matplotlib
        matplotlib.use("module://kitcat")
    print("Live production loss dashboard (kitcat+ambivalent). Ctrl-C to stop.")
    print("  (first frame on a cold cache builds the W&B backbone -- ~3-6 min, "
          "live progress below; later frames are instant from cache)")
    sys.stdout.flush()
    while True:
        try:
            payload = fetch()
        except Exception as e:
            print("fetch error:", e)
            payload = {"chains": {}}
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        board = render_board(payload)
        print(board)
        # Reserve the rows the board ACTUALLY occupies (it grows with the chain
        # count) plus 1 for the trailing newline kitcat emits. Hardcoding a
        # guess here is what let figure+board exceed the window and scroll the
        # title/legend off the top.
        reserved = board.count("\n") + 2
        draw_curves(payload, size_figure(reserved_rows=reserved))
        if once:
            break
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped.")
            break


if __name__ == "__main__":
    main()
