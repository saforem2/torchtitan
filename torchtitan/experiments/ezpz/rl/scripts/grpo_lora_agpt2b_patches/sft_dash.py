#!/usr/bin/env python3
"""Live CoT-SFT training dashboard (kitcat inline, ambivalent + Iosevka).

Self-contained: reads the run's TRL log directly (over SSH, or locally with
RL_LOCAL=1), parses the ``{'loss': ..., 'mean_token_accuracy': ..., ...}`` dict
lines TRL prints, and plots loss / mean-token-accuracy / grad-norm vs step.
Mirrors rl_dash3.py (same HiDPI fix + style + auto-fit). Edit RUN_LOG_GLOB (or
pass --log) to point at a different run.

Run in a kitty terminal:
    python sft_dash.py                       # pulls the latest sft-*/run.log over SSH
    RL_LOCAL=1 python sft_dash.py            # on the cluster (read local FS)
    python sft_dash.py --log logs/sft-.../run.log
Env: RL_INTERVAL (sec, default 30), RL_LOCAL=1, RL_SSH (default 'sunspot'),
     RL_SOCK (control socket), RL_BACKING (force HiDPI factor).
"""
import os
import sys
import re
import time
import glob
import subprocess

import matplotlib
matplotlib.use("module://kitcat")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import ambivalent


# ---- HiDPI fix: make kitcat render at true device resolution --------------
# (identical to rl_dash3.py -- see that file for the full rationale: kitcat
# renders at logical px on macOS and kitty upscales -> aliasing; we render at
# device px with an unchanged placeholder grid.)
def _tty_query(seq, timeout=0.3):
    import termios
    import tty as _tty
    import select
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


def _csi_t(n):
    m = re.search(r"\033\[\d+;(\d+);(\d+)t", _tty_query("\033[%dt" % n))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _backing_and_cells():
    import array
    import fcntl
    import termios
    buf = array.array("H", [0, 0, 0, 0])
    try:
        fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, buf)
    except Exception:
        pass
    rows, cols, xpix, ypix = buf
    log_cw = xpix / cols if (xpix and cols) else 0.0
    log_ch = ypix / rows if (ypix and rows) else 0.0
    dev = _csi_t(16)
    dev_ch, dev_cw = dev if dev else (0.0, 0.0)
    forced = os.environ.get("RL_BACKING")
    if forced:
        backing = float(forced)
    elif dev_ch and log_ch:
        backing = dev_ch / log_ch
    elif sys.platform == "darwin":
        backing = 2.0
    else:
        backing = 1.0
    backing = max(1.0, min(3.0, backing))
    cw = dev_cw or (log_cw * backing) or (8 * backing)
    ch = dev_ch or (log_ch * backing) or (16 * backing)
    return backing, cw, ch


if sys.stdout.isatty():
    try:
        _BACKING, _DEV_CW, _DEV_CH = _backing_and_cells()
        if _BACKING > 1.0:
            import kitcat.terminal_query as _ktq
            import kitcat.utils as _ku
            import kitcat.backend as _kb
            for _m in (_ktq, _ku, _kb):
                _m.get_dpi_scale = lambda: _BACKING
            _ku.get_char_cell_width = lambda: max(1, int(round(_DEV_CW)))
            _ku.get_char_cell_height = lambda: max(1, int(round(_DEV_CH)))
            print("kitcat HiDPI: backing=%.2f cell=%dx%d(dev)"
                  % (_BACKING, round(_DEV_CW), round(_DEV_CH)))
    except Exception as _e:
        print("kitcat HiDPI patch skipped:", _e)


# ---- config --------------------------------------------------------------
REPO = ("/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
# default: newest CoT-SFT run log; override with --log (accepts MULTIPLE globs to
# overlay several runs on shared panels, e.g.
#   sft_dash.py --log 'logs/grpo-...-12471085/run.log' 'logs/grpo-...-fast-*/run.log'
RUN_LOG_GLOB = os.environ.get(
    "SFT_LOG_GLOB", "logs/sft-agpt2b-gsm8k-r1cot-8n-*/run.log")
RUN_GLOBS = [RUN_LOG_GLOB]  # overridden from argv in __main__
# --auto <glob>: expand to EVERY matching run.log each refresh (auto-discovers new
# runs). Default watches all CoT GRPO runs. Set to "" to disable auto mode.
AUTO_GLOB = os.environ.get("RL_AUTO_GLOB", "")
# distinct colors when overlaying multiple runs.
RUN_COLORS = ["#4c78a8", "#e45756", "#59a14f", "#b279a2", "#f0a24b", "#888888",
              "#54a24b", "#eeca3b", "#b279a2", "#ff9da6"]

INTERVAL = float(os.environ.get("RL_INTERVAL", "30"))
LOCAL = os.environ.get("RL_LOCAL") == "1"
SSH_TGT = os.environ.get("RL_SSH", "sunspot")
SOCK = os.environ.get("RL_SOCK", "/tmp/sunspot-master.sock")

# Metric panels for each run type. TRL's SFTTrainer and GRPOTrainer both emit
# dict-log lines in the same shape, but different KEYS -- so we auto-detect
# which set to plot from the keys present in the parsed rows (see pick_metrics).
SFT_METRICS = [
    ("loss", "training loss", "#4c78a8"),
    ("mean_token_accuracy", "mean token accuracy", "#59a14f"),
    ("grad_norm", "grad norm", "#e45756"),
]
GRPO_METRICS = [
    ("reward", "mean reward", "#4c78a8"),
    ("reward_std", "reward std", "#b279a2"),
    ("frac_reward_zero_std", "frac zero-std groups", "#f0a24b"),
    ("completions/mean_length", "completion length", "#59a14f"),
    ("kl", "KL", "#e45756"),
]


def pick_metrics(rows):
    """Choose the panel set by which keys the run actually logs. GRPO if any
    reward key is present, else SFT. Drops panels no row has data for so a
    partial-metric run doesn't render empty axes."""
    keys = set()
    for r in rows:
        keys.update(r)
    base = GRPO_METRICS if ("reward" in keys) else SFT_METRICS
    present = [m for m in base if m[0] in keys]
    return present or base

# ---- style: ambivalent + Iosevka + transparent ---------------------------
for d in (os.path.expanduser("~/Library/Fonts"), "/Library/Fonts",
          os.path.expanduser("~/.local/share/fonts"), "/usr/share/fonts"):
    for f in glob.glob(os.path.join(d, "**", "Iosevka*.tt*"), recursive=True):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
plt.style.use(ambivalent.STYLES["ambivalent"])
_ios = sorted({f.name for f in fm.fontManager.ttflist if "iosevka" in f.name.lower()})
if _ios:
    plt.rcParams["font.family"] = next(
        (n for n in _ios if n.lower() in ("iosevka", "iosevka term")), _ios[0])
plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none",
                     "axes.facecolor": "none"})


# remote python that resolves the newest matching log and emits its metric rows.
# Handles both TRL SFTTrainer ('loss'/'train_loss'/'mean_token_accuracy') and
# GRPOTrainer ('reward'/'reward_std'/'kl'/'completions/mean_length') dict-lines.
_AGG = r'''
import glob, json, os, re, sys
os.chdir("%s")
paths = sorted(glob.glob(%r))
if not paths:
    print(json.dumps({"rows": [], "log": None})); raise SystemExit
log = paths[-1]
rows = []
final = None
ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# a TRL metric dict has at least one of these training keys.
trigger = re.compile(r"'(?:loss|train_loss|reward)'")
# keys can contain letters, digits, '_' and '/' (e.g. completions/mean_length).
kv = re.compile(r"'([a-zA-Z_][\w/]*)':\s*'?([-\d.eE+]+)'?")
for line in open(log, errors="replace"):
    line = ansi.sub("", line)
    for m in re.finditer(r"\{[^{}]*\}", line):
        blob = m.group(0)
        if not trigger.search(blob):
            continue
        d = {}
        for k, v in kv.findall(blob):
            try: d[k] = float(v)
            except ValueError: pass
        if "train_loss" in d:
            final = d          # SFT end-of-run summary
        elif "loss" in d or "reward" in d:
            rows.append(d)
print(json.dumps({"rows": rows, "final": final, "log": os.path.basename(os.path.dirname(log))}))
'''  # NOTE: format at call time with the CURRENT glob (see fetch), not here --
# an early `% (REPO, RUN_LOG_GLOB)` would bake in the module-load glob and make
# the --log override a no-op.


def fetch(glob_pat=None):
    # Format the remote script HERE (not at module load) so the glob applies.
    agg = _AGG % (REPO, glob_pat if glob_pat is not None else RUN_LOG_GLOB)
    if LOCAL:
        r = subprocess.run([sys.executable, "-c", agg],
                           capture_output=True, text=True, timeout=90)
    else:
        import base64
        b64 = base64.b64encode(agg.encode()).decode()
        remote = "echo %s | base64 -d | python3" % b64
        r = subprocess.run(
            ["ssh", "-o", "ControlPath=" + SOCK, SSH_TGT, remote],
            capture_output=True, text=True, timeout=90)
    import json
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"rows": [], "final": None, "log": None}


# remote one-liner: expand a wildcard into every matching run.log path (newest
# last), so --auto discovers new runs automatically each refresh.
_LIST = (
    "import glob,os,json;"
    "os.chdir(%r);"
    "print(json.dumps(sorted(glob.glob(%r), key=os.path.getmtime)))"
)


def list_runs(glob_pat):
    script = _LIST % (REPO, glob_pat)
    if LOCAL:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=60)
    else:
        import base64
        b64 = base64.b64encode(script.encode()).decode()
        r = subprocess.run(
            ["ssh", "-o", "ControlPath=" + SOCK, SSH_TGT,
             "echo %s | base64 -d | python3" % b64],
            capture_output=True, text=True, timeout=60)
    import json
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("["):
            try:
                return json.loads(line)
            except Exception:
                pass
    return []


def fetch_all():
    """One fetch per run. In --auto mode, AUTO_GLOB is expanded to every matching
    run.log first (so new runs appear automatically); otherwise use RUN_GLOBS."""
    globs = RUN_GLOBS
    if AUTO_GLOB:
        found = list_runs(AUTO_GLOB)
        globs = found or [AUTO_GLOB]  # exact paths -> one series per run
    out = []
    for g in globs:
        d = fetch(g)
        d["glob"] = g
        out.append(d)
    return out


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
    return 1000, 620


def draw(data, wpx, hpx):
    rows = data.get("rows") or []
    metrics = pick_metrics(rows)
    is_grpo = any(m[0] == "reward" for m in metrics)
    dpi = 100
    n = len(metrics)
    fig, axes = plt.subplots(
        n, 1, sharex=True, dpi=dpi,
        figsize=(max(4.0, wpx * 0.92 / dpi), max(3.0, hpx * 0.82 / dpi)))
    if n == 1:
        axes = [axes]
    # GRPO logs 'epoch' too, but step is the more natural x for RL.
    xkey = "step" if (is_grpo and any("step" in r for r in rows)) else "epoch"
    xs = [r.get(xkey, i) for i, r in enumerate(rows)]
    for ax, (key, label, color) in zip(axes, metrics):
        ys = [r.get(key) for r in rows]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    "-", lw=1.8, color=color)
            last = pts[-1][1]
            ax.set_ylabel("%s\n(%.4g)" % (label, last), fontsize=8)
        else:
            ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel(xkey)
    final = data.get("final")
    kind = "GRPO" if is_grpo else "SFT"
    title = "agpt-2b %s -- %s" % (kind, data.get("log") or "(waiting)")
    if final:
        fl = final.get("train_loss")
        title += "  [DONE%s]" % ("" if fl is None else " train_loss=%.4g" % fl)
    elif rows:
        title += "  [%d log points]" % len(rows)
    axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def draw_overlay(runs, wpx, hpx):
    """Overlay several runs on shared panels: one line per run per metric.
    Panel set is the union across runs (GRPO vs SFT auto-detected per run)."""
    allrows = [r for run in runs for r in (run.get("rows") or [])]
    metrics = pick_metrics(allrows)
    is_grpo = any(m[0] == "reward" for m in metrics)
    dpi = 100
    n = len(metrics)
    fig, axes = plt.subplots(
        n, 1, sharex=True, dpi=dpi,
        figsize=(max(4.0, wpx * 0.92 / dpi), max(3.0, hpx * 0.82 / dpi)))
    if n == 1:
        axes = [axes]
    for ri, run in enumerate(runs):
        rows = run.get("rows") or []
        if not rows:
            continue
        color = RUN_COLORS[ri % len(RUN_COLORS)]
        label = (run.get("log") or run.get("glob") or "run%d" % ri)
        # compact label: drop the common prefix, keep the job id / tail
        label = label.replace("grpo-agpt2b-gsm8k-reason-cot-", "").replace(
            "sft-agpt2b-gsm8k-r1cot-", "sft-")
        xkey = "step" if (is_grpo and any("step" in r for r in rows)) else "epoch"
        xs = [r.get(xkey, i) for i, r in enumerate(rows)]
        for ax, met in zip(axes, metrics):
            key = met[0]
            pts = [(x, r.get(key)) for x, r in zip(xs, rows) if r.get(key) is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        "-", lw=1.6, color=color,
                        label=(label if ax is axes[0] else None))
    for ax, met in zip(axes, metrics):
        ax.set_ylabel(met[1], fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("step" if is_grpo else "epoch")
    axes[0].legend(loc="best", fontsize=7, ncol=min(len(runs), 3))
    axes[0].set_title("agpt-2b %s -- %d runs overlaid" %
                      ("GRPO" if is_grpo else "SFT", len(runs)), fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _status_line(data):
    rows = data.get("rows") or []
    tail = rows[-1] if rows else {}
    if "reward" in tail:
        s = "reward=%s std=%s" % (tail.get("reward"), tail.get("reward_std"))
    else:
        s = "loss=%s acc=%s" % (tail.get("loss"), tail.get("mean_token_accuracy"))
    label = (data.get("log") or "(no log yet)")
    return "%s | points=%d last: %s" % (label, len(rows), s)


def main():
    if AUTO_GLOB:
        print("Live dashboard (AUTO: %s, auto-discovers new runs). Ctrl-C to stop."
              % AUTO_GLOB)
    else:
        n = len(RUN_GLOBS)
        print("Live dashboard (%d run%s, kitcat+ambivalent). Ctrl-C to stop."
              % (n, "s" if n > 1 else ""))
    while True:
        try:
            runs = fetch_all()
        except Exception as e:
            runs = []
            print("fetch error:", e)
        # auto mode (or >1 configured glob) -> overlay; else single-run panels.
        multi = bool(AUTO_GLOB) or len(RUN_GLOBS) > 1
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print("[%s]" % time.strftime("%H:%M:%S"))
        for d in runs:
            print("  " + _status_line(d))
        if multi:
            draw_overlay(runs, *terminal_pixels())
        elif runs:
            draw(runs[0], *terminal_pixels())
        # stop when every run is complete -- but NOT in auto mode, where new runs
        # may still appear (keep watching until Ctrl-C).
        if not AUTO_GLOB and runs and all(d.get("final") for d in runs):
            print("all runs complete -- stopping.")
            break
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped.")
            break


if __name__ == "__main__":
    # --log accepts one OR more globs (all argv after --log that aren't flags),
    # so you can overlay several runs:
    #   sft_dash.py --log 'logs/grpo-...-12471085/run.log' 'logs/grpo-...-fast-*/run.log'
    if "--log" in sys.argv:
        i = sys.argv.index("--log") + 1
        globs = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            globs.append(sys.argv[i])
            i += 1
        if globs:
            RUN_GLOBS = globs
            RUN_LOG_GLOB = globs[0]
    # --auto [glob]: overlay+auto-discover EVERY matching run.log (new runs appear
    # on each refresh). Bare --auto defaults to all CoT GRPO runs.
    if "--auto" in sys.argv:
        j = sys.argv.index("--auto") + 1
        if j < len(sys.argv) and not sys.argv[j].startswith("--"):
            AUTO_GLOB = sys.argv[j]
        else:
            AUTO_GLOB = "logs/grpo-agpt2b-gsm8k-reason-cot*/run.log"
    main()
