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
# Cold W&B backbone build takes ~3 min (scan_history over ~60 runs); the cheap
# cache-hit path is ~4 s. Give the SSH call room for a cold build so a first
# call doesn't die at the finish line, then cache-hits are instant.
SSH_TIMEOUT = float(os.environ.get("PD_SSH_TIMEOUT", "600"))
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

def _wandb_records(run_ids, olog_fallbacks):
    """Concat a chain's full per-step W&B history via wandb_fetch.concat_chain
    (same robust olog-fallback rule the charts use: prefer the .o log when it
    reaches at least as far as W&B). Returns the list of records keyed by
    OLOG_KEYS (step + loss + grad_norm + tps + tflops + mfu) -- the caller
    projects to the per-metric series it needs (via _series_from_records)."""
    olog_fallbacks = olog_fallbacks or {}
    # concat_chain resolves relative fallback paths against CWD; the aggregator
    # chdir's to REPO at startup, but be explicit so it works regardless.
    fb = {rid: (fp if os.path.isabs(fp) else os.path.join(REPO, fp))
          for rid, fp in olog_fallbacks.items()}
    return wf.concat_chain(run_ids, olog_fallbacks=fb, keys=wf.OLOG_KEYS,
                           project=PROJECT)

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
                                 t.get("olog_fallbacks"))
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
    # call must never block on the ~3-6 min W&B scan_history build.
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
    _log("no backbone cache -> cold build (this is the ~3-6 min first-run "
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
        base = None
        for p in cand:
            try:
                head = ANSI.sub("", open(p, errors="replace").read(200000))
            except (FileNotFoundError, IsADirectoryError):
                continue
            m = CKPT_RE.search(head)
            if m:
                base = m.group(1)
                break
        if not base:
            continue
        if rank.get(j["state"], 0) >= rank.get(states.get(base, "E"), 0):
            states[base] = j["state"]
        if j["state"] == "R" and cand:
            _, last = _olog_records(cand)
            if last:
                try:
                    age = time.time() - max(os.path.getmtime(p) for p in cand
                                            if os.path.exists(p))
                except Exception:
                    age = None
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
bb["chains"] = chains_out
bb.pop("idx", None)  # don't ship the path index to the client
bb["built_age"] = round(now - bb["built_at"], 1)
bb["generated_at"] = time.time()
print(json.dumps(bb))
'''


def fetch(stderr_cb=None) -> dict:
    """Run the remote aggregator and return the parsed backbone+live payload.

    stdout carries the single JSON object; the aggregator's stderr is the
    ``[prod_dash +Ns]`` progress stream (cold build ~3-6 min). Behavior:
      - stderr_cb given: capture stderr and call stderr_cb(line) per line (the
        Textual app pipes this into a RichLog); JSON returned at the end.
      - stderr_cb None, PD_QUIET=1: discard stderr (detached refresh worker).
      - stderr_cb None otherwise: inherit stderr so progress streams straight
        to the terminal live (the kitcat/board CLI path).
    """
    script = _AGG % {"repo": REPO, "ttl": BACKBONE_TTL,
                     "fresh": 1 if os.environ.get("PD_FRESH") == "1" else 0,
                     "show_all": 1 if SHOW_ALL else 0,
                     "exp_max_age": EXP_MAX_AGE}
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
            return {"chains": {}}
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
            return {"chains": {}}
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
            return {"chains": {}}
        stdout_text = r.stdout or ""
    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    if not LOCAL:
        sys.stderr.write(
            "prod_dash: no JSON from aggregator. If the ssh master socket %s is "
            "dead, BatchMode refuses to prompt (fails fast by design) -- recreate "
            "it with:\n    ssh -MNf -S %s %s\n" % (SOCK, SOCK, SSH_TGT)
        )
    else:
        sys.stderr.write("prod_dash: no JSON from aggregator\n")
    return {"chains": {}}


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


def draw_curves(payload, wpx, hpx, save_path=None):
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
    fig, ax = plt.subplots(
        figsize=(max(5.0, wpx * 0.92 / dpi), max(3.0, hpx * 0.80 / dpi)), dpi=dpi)
    # x-axis: "step" (default, faithful to rl_dash3) or "tokens" (every chain
    # shares the 4.67T olmo-mix target, so tokens = step * gbs * seq_len puts
    # them on a comparable footing despite very different step counts/batches).
    xaxis = os.environ.get("PD_XAXIS", "step")
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
            xs = [p[0] * toks_per_step / 1e9 for p in curve]  # billions of tokens
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
        ax.plot(xs, ys, ls, lw=lw, color=color, alpha=alpha, zorder=z,
                label=label, rasterized=len(xs) > 2000)
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


def terminal_pixels():
    try:
        import fcntl
        import struct
        import termios
        buf = fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, b"\0" * 8)
        _, _, xp, yp = struct.unpack("HHHH", buf)
        if xp and yp:
            return xp, yp
    except Exception:
        pass
    return 1200, 720


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
        draw_curves(fetch(), *terminal_pixels(), save_path=path)
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
        print(render_board(payload))
        draw_curves(payload, *terminal_pixels())
        if once:
            break
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped.")
            break


if __name__ == "__main__":
    main()
