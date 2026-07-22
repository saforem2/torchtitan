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

Three modes (the user asked for all three):

    # 1. Live kitcat loop (run in a kitty terminal; rl_dash3 twin)
    /tmp/kitcat-venv/bin/python prod_dash.py

    # 2. Headless SVG/PNG snapshot for docs/production/
    python prod_dash.py --svg /path/to/production_loss_live.svg

    # 3. Stall-aware text status board (no matplotlib needed)
    python prod_dash.py --board

    # one live frame then exit (curve + board), for a quick look
    python prod_dash.py --once

Env:
    PD_INTERVAL   live-loop poll seconds (default 30)
    PD_LOCAL=1    read the local FS instead of SSH (run this ON the cluster)
    PD_SSH        ssh target (default 'aurora')
    PD_SOCK       ssh ControlPath (default /tmp/aurora-master.sock)
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

INTERVAL = float(os.environ.get("PD_INTERVAL", "30"))
LOCAL = os.environ.get("PD_LOCAL") == "1"
SSH_TGT = os.environ.get("PD_SSH", "aurora")
SOCK = os.environ.get("PD_SOCK", "/tmp/aurora-master.sock")
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

COLORS = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2",
    "#ff9da6", "#9d755d", "#bab0ac", "#edc948", "#b07aa1", "#86bcb6",
]

# ---------------------------------------------------------------------------
# Remote aggregator: everything that needs cluster-side data lives here. It is
# base64'd and piped to the cluster's .venv python (which has wandb). Emits a
# single JSON object on stdout. The W&B "backbone" (full loss history per
# chain) is expensive, so it is cached on the cluster at /tmp and only rebuilt
# when older than BACKBONE_TTL or when PD_FRESH=1. The cheap "live layer"
# (qstat states + tails of currently-running .o logs) runs every call.
# ---------------------------------------------------------------------------
_AGG = r'''
import json, os, re, glob, time, subprocess, importlib.util, statistics

REPO = %(repo)r
TTL = %(ttl)d
FRESH = %(fresh)d
SHOW_ALL = %(show_all)d
EXP_MAX_AGE = %(exp_max_age)d
USER = os.environ.get("USER", "foremans")
PROJECT = "aurora_gpt/torchtitan.ezpz.train"
CACHE = "/tmp/prod_dash_backbone_%%s.json" %% USER
os.chdir(REPO)

# --- trajectories.py imported standalone (avoids the torch-importing package) -
_spec = importlib.util.spec_from_file_location(
    "traj", os.path.join(REPO, "torchtitan/experiments/ezpz/utils/trajectories.py"))
traj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traj)
CANON = [t for t in traj.TRAJECTORIES if t.get("cls") in ("live", "wandb_only")]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([\d.]+)\s+grad_norm:\s+([\d.]+)"
    r".*?tps:\s+([\d,]+).*?mfu:\s+([\d.]+)%%")
CKPT_RE = re.compile(r"checkpoints/(agpt-[A-Za-z0-9._-]+?)(?:/|\s|$)")

def _downsample(pairs, n=600):
    if len(pairs) <= n:
        return pairs
    k = (len(pairs) + n - 1) // n
    return pairs[::k]

def _olog_curve(paths):
    """Concat (step, loss) from .o logs, dedup by step (later files win)."""
    by, last = {}, {}
    for p in sorted(paths):
        try:
            for raw in open(p, errors="replace"):
                m = STEP_RE.search(ANSI.sub("", raw))
                if not m:
                    continue
                s = int(m.group(1))
                by[s] = float(m.group(2))
                last = {"step": s, "loss": float(m.group(2)),
                        "tps": float(m.group(4).replace(",", "")),
                        "mfu": float(m.group(5))}
        except (FileNotFoundError, IsADirectoryError):
            pass
    xs = sorted(by)
    return [(x, by[x]) for x in xs], last

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

def _wandb_curve(run_ids, olog_fallbacks):
    """Concat W&B loss history across a chain's runs; fall back to .o per run."""
    try:
        import wandb
        api = wandb.Api()
    except Exception:
        api = None
    olog_fallbacks = olog_fallbacks or {}
    by = {}
    for rid in run_ids:
        rows = []
        if api is not None:
            try:
                run = api.run(PROJECT + "/" + rid)
                for r in run.scan_history(keys=["_step", "loss_metrics/global_avg_loss"]):
                    s = r.get("_step")
                    lv = r.get("loss_metrics/global_avg_loss")
                    if s is not None and lv is not None:
                        rows.append((int(s), float(lv)))
            except Exception:
                rows = []
        if not rows and rid in olog_fallbacks:
            fp = olog_fallbacks[rid]
            if not os.path.isabs(fp):
                fp = os.path.join(REPO, fp)
            rows, _ = _olog_curve([fp])
        for s, lv in rows:
            by[s] = lv
    xs = sorted(by)
    return [(x, by[x]) for x in xs]

def build_backbone():
    logs = _all_ologs()
    idx = _dir_index(logs)
    canon_dirs = set()
    chains = {}
    for t in CANON:
        cd = t.get("ckpt_dir")
        base = os.path.basename(cd) if cd else None
        if base:
            canon_dirs.add(base)
        curve = _wandb_curve(t.get("wandb_run_ids") or [], t.get("olog_fallbacks"))
        # Distinguish sibling chains that share model+nodes (e.g. the canonical
        # 2b 512N vs the sqrt2-LR fork "2b_v2_512_lr3.22e-5") by appending the
        # key's suffix beyond the standard "<model>_<version>_<nodes>" form.
        std = "%%s_%%s_%%d" %% (t["model"], t["version"], t["num_nodes"])
        extra = t["key"][len(std):].lstrip("_") if t["key"].startswith(std) else ""
        label = "%%s %%dN" %% (t["model"], t["num_nodes"])
        if extra:
            label += " (%%s)" %% extra
        chains[t["key"]] = {
            "label": label,
            "model": t["model"], "num_nodes": t["num_nodes"],
            "gbs": t["gbs"], "seq_len": t["seq_len"],
            "token_target": t["token_target"], "kind": "canonical",
            "ckpt_base": base, "curve": _downsample(curve),
        }
    # active experiments: agpt-* ckpt dirs with a referencing .o log, not
    # canonical. Stale ones (last .o write older than EXP_MAX_AGE seconds) are
    # hidden by default so the board stays focused on the live picture; set
    # SHOW_ALL to include every experiment ever run.
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
        curve, _ = _olog_curve(paths)
        if len(curve) < 2:
            continue
        chains["exp:" + base] = {
            "label": base.replace("agpt-", ""),
            "model": "2b" if "-2b-" in base or base.endswith("-2b") else "?",
            "num_nodes": None, "gbs": None, "seq_len": 8192,
            "token_target": None, "kind": "experiment",
            "ckpt_base": base, "curve": _downsample(curve),
        }
    # Cache the ckpt-base -> log-paths index so the per-call path can stat
    # staleness WITHOUT re-reading every log head over Lustre (that scan is
    # the expensive part: ~2 min for the whole repo). Only paths referenced by
    # a tracked chain are kept.
    tracked = {c.get("ckpt_base") for c in chains.values() if c.get("ckpt_base")}
    slim_idx = {b: p for b, p in idx.items() if b in tracked}
    return {"chains": chains, "idx": slim_idx, "built_at": time.time()}

def load_backbone():
    if not FRESH and os.path.exists(CACHE):
        try:
            age = time.time() - os.path.getmtime(CACHE)
            if age < TTL:
                return json.load(open(CACHE))
        except Exception:
            pass
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

def live_layer():
    """Per running job: resolve its ckpt dir + freshest (step, loss). Only the
    handful of logs belonging to CURRENT qstat jobs are read (cheap), unlike the
    full-repo head scan in build_backbone."""
    jobs = qstat_jobs()
    live = {}  # ckpt_base -> {state, jobid, step, loss, tps, mfu, age}
    states = {}  # ckpt_base -> worst-known state (R>Q>H)
    rank = {"R": 3, "Q": 2, "H": 1, "E": 0}
    for j in jobs:
        cand = glob.glob(os.path.join(REPO, "*.o" + j["id"]))
        cand += glob.glob(os.path.join(REPO, "logs", "*" + j["id"], "run.log"))
        cand += glob.glob(os.path.join(
            REPO, "torchtitan/experiments/ezpz/scripts/oneoff/*.o" + j["id"]))
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
            _, last = _olog_curve(cand)
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


def fetch() -> dict:
    script = _AGG % {"repo": REPO, "ttl": BACKBONE_TTL,
                     "fresh": 1 if os.environ.get("PD_FRESH") == "1" else 0,
                     "show_all": 1 if SHOW_ALL else 0,
                     "exp_max_age": EXP_MAX_AGE}
    if LOCAL:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=SSH_TIMEOUT)
    else:
        b64 = base64.b64encode(script.encode()).decode()
        remote = (
            "P=$([ -x .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3); "
            "echo %s | base64 -d | $P" % b64
        )
        r = subprocess.run(
            ["ssh", "-S", SOCK, SSH_TGT,
             "cd %s && %s" % (REPO, remote)],
            capture_output=True, text=True, timeout=SSH_TIMEOUT)
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    sys.stderr.write("prod_dash: no JSON from aggregator\n")
    if r.stderr:
        sys.stderr.write(r.stderr[-2000:] + "\n")
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

    rows = []
    header = ("chain", "state", "step", "loss", "% tgt", "tps", "mfu", "log age")
    for key, c in sorted(chains.items(), key=order):
        st = c.get("queue_state") or ("run" if (c.get("log_age") or 1e9) < LIVE_WINDOW else "idle")
        tip = c.get("live_tip") or {}
        step = c.get("latest_step")
        loss = c.get("latest_loss")
        age = c.get("log_age")
        age_s = "-" if age is None else (
            "%.0fs" % age if age < 90 else
            "%.0fm" % (age / 60) if age < 5400 else
            "%.1fh" % (age / 3600))
        rows.append((
            c.get("label", key)[:34],
            st,
            "-" if step is None else str(step),
            "-" if loss is None else "%.4f" % loss,
            "-" if c.get("pct_target") is None else "%.1f%%" % c["pct_target"],
            "%.0f" % tip["tps"] if tip.get("tps") else "-",
            "%.1f%%" % tip["mfu"] if tip.get("mfu") else "-",
            age_s,
        ))
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    def fmt(r):
        return "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
    nlive = sum(1 for c in chains.values()
                if c.get("queue_state") == "R"
                or (c.get("log_age") or 1e9) < LIVE_WINDOW)
    bb_age = payload.get("built_age")
    out = [
        "AuroraGPT production loss board  [%s]  %d live / %d chains  (W&B backbone %s)"
        % (time.strftime("%H:%M:%S", time.localtime(now)), nlive, len(chains),
           "-" if bb_age is None else "%.0fm old" % (bb_age / 60)),
        fmt(header),
        "  ".join("-" * w for w in widths),
    ]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Curve overlay (modes 1 & 2): shared by live-kitcat and headless-SVG.
# ---------------------------------------------------------------------------
def draw_curves(payload, wpx, hpx, save_path=None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from torchtitan.experiments.ezpz.utils.plot_style import apply_style
        apply_style()
    except Exception:
        pass
    plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none",
                         "axes.facecolor": "none"})
    chains = payload.get("chains", {})
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(max(5.0, wpx * 0.92 / dpi), max(3.0, hpx * 0.80 / dpi)), dpi=dpi)
    parts = []
    for key in sorted(chains):
        c = chains[key]
        curve = c.get("curve") or []
        tip = c.get("live_tip")
        if tip and (not curve or tip["step"] > curve[-1][0]):
            curve = curve + [[tip["step"], tip["loss"]]]
        if len(curve) < 2:
            continue
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        is_live = (c.get("queue_state") == "R"
                   or (c.get("log_age") is not None and c["log_age"] <= LIVE_WINDOW))
        is_exp = c.get("kind") == "experiment"
        color = COLORS[zlib.crc32(key.encode()) % len(COLORS)]
        alpha = 1.0 if is_live else 0.30
        lw = 2.4 if is_live else 1.2
        ls = "--" if is_exp else "-"
        z = 5 if is_live else 2
        label = ("* " if is_live else "") + c.get("label", key) + (
            "" if is_live else " (idle)")
        ax.plot(xs, ys, ls, lw=lw, color=color, alpha=alpha, zorder=z,
                label=label, rasterized=len(xs) > 2000)
        if is_live:
            parts.append("%s=%.3f@%d" % (c.get("label", key), ys[-1], xs[-1]))
    ax.set_xlabel("training step (cumulative across resumes)")
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
