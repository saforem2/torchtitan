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
# default: newest CoT-SFT run log; override with --log
RUN_LOG_GLOB = os.environ.get(
    "SFT_LOG_GLOB", "logs/sft-agpt2b-gsm8k-r1cot-8n-*/run.log")

INTERVAL = float(os.environ.get("RL_INTERVAL", "30"))
LOCAL = os.environ.get("RL_LOCAL") == "1"
SSH_TGT = os.environ.get("RL_SSH", "sunspot")
SOCK = os.environ.get("RL_SOCK", "/tmp/sunspot-master.sock")

# metrics parsed from each TRL dict-line, and how to plot them.
METRICS = [
    ("loss", "training loss", "#4c78a8"),
    ("mean_token_accuracy", "mean token accuracy", "#59a14f"),
    ("grad_norm", "grad norm", "#e45756"),
]

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


# remote python that resolves the newest matching log and emits its metric rows
_AGG = r'''
import glob, json, os, re, sys
os.chdir("%s")
paths = sorted(glob.glob(%r))
if not paths:
    print(json.dumps({"rows": [], "log": None})); raise SystemExit
log = paths[-1]
rows = []
final = None
# strip ANSI, then match TRL dict-lines like {'loss': '0.36', 'epoch': '1.2', ...}
ansi = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
for line in open(log, errors="replace"):
    line = ansi.sub("", line)
    for m in re.finditer(r"\{[^{}]*'(?:loss|train_loss)'[^{}]*\}", line):
        blob = m.group(0)
        d = {}
        for k, v in re.findall(r"'([a-z_]+)':\s*'?([-\d.e+]+)'?", blob):
            try: d[k] = float(v)
            except ValueError: pass
        if "train_loss" in d:
            final = d
        elif "loss" in d:
            rows.append(d)
print(json.dumps({"rows": rows, "final": final, "log": os.path.basename(os.path.dirname(log))}))
''' % (REPO, RUN_LOG_GLOB)


def fetch():
    if LOCAL:
        r = subprocess.run([sys.executable, "-c", _AGG],
                           capture_output=True, text=True, timeout=90)
    else:
        import base64
        b64 = base64.b64encode(_AGG.encode()).decode()
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
    dpi = 100
    n = len(METRICS)
    fig, axes = plt.subplots(
        n, 1, sharex=True, dpi=dpi,
        figsize=(max(4.0, wpx * 0.92 / dpi), max(3.0, hpx * 0.82 / dpi)))
    if n == 1:
        axes = [axes]
    xs = [r.get("epoch", i) for i, r in enumerate(rows)]
    for ax, (key, label, color) in zip(axes, METRICS):
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
    axes[-1].set_xlabel("epoch")
    final = data.get("final")
    title = "agpt-2b gsm8k-r1cot SFT -- %s" % (data.get("log") or "(waiting)")
    if final:
        title += "  [DONE train_loss=%.4g]" % final.get("train_loss", float("nan"))
    elif rows:
        title += "  [%d log points]" % len(rows)
    axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    print("Live CoT-SFT dashboard (kitcat+ambivalent). Ctrl-C to stop.")
    while True:
        try:
            data = fetch()
        except Exception as e:
            data = {"rows": [], "final": None, "log": None}
            print("fetch error:", e)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        rows = data.get("rows") or []
        tail = rows[-1] if rows else {}
        print("[%s] %s | points=%d last: loss=%s acc=%s" % (
            time.strftime("%H:%M:%S"), data.get("log") or "(no log yet)",
            len(rows), tail.get("loss"), tail.get("mean_token_accuracy")))
        draw(data, *terminal_pixels())
        if data.get("final"):
            print("run complete (train_loss=%.4g) -- stopping."
                  % data["final"].get("train_loss", float("nan")))
            break
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped.")
            break


if __name__ == "__main__":
    if "--log" in sys.argv:
        RUN_LOG_GLOB = sys.argv[sys.argv.index("--log") + 1]
    main()
