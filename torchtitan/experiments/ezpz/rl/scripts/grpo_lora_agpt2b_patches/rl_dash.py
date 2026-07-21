#!/usr/bin/env python3
"""Live agpt-2b GRPO reward dashboard -- kitcat inline rendering, ambivalent +
transparent + Iosevka styling, auto-fit to the terminal's available pixel space.

Run in a kitty terminal:  /tmp/kitcat-venv/bin/python /tmp/rl_dash.py
It loops: refresh the TSV via rl_feed.sh, then redraw inline. Ctrl-C to stop.
"""
import os
import sys
import glob
import time
import subprocess

import matplotlib

matplotlib.use("module://kitcat")  # kitcat backend -> inline kitty graphics
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import ambivalent

TSV = os.environ.get("RL_TSV", "/tmp/rl_reward.tsv")
FEED = os.environ.get("RL_FEED", "/tmp/rl_feed.sh")
JSONL = os.environ.get(
    "RL_JSONL",
    "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/"
    "outputs/rl_lora_agpt2b_train_v4/rollout_samples.jsonl",
)
INTERVAL = float(os.environ.get("RL_INTERVAL", "20"))

# --- register Iosevka + apply ambivalent (transparent) --------------------
for d in (os.path.expanduser("~/Library/Fonts"), "/Library/Fonts"):
    for f in glob.glob(os.path.join(d, "Iosevka*.tt*")):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
plt.style.use(ambivalent.STYLES["ambivalent"])
_ios = sorted({f.name for f in fm.fontManager.ttflist if "iosevka" in f.name.lower()})
if _ios:
    # prefer a plain "Iosevka"/"Iosevka Term" family if present
    pref = next((n for n in _ios if n.lower() in ("iosevka", "iosevka term")), _ios[0])
    plt.rcParams["font.family"] = pref
plt.rcParams["savefig.transparent"] = True
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"


def terminal_pixels():
    """(width_px, height_px) of the terminal, best-effort, for auto-fit."""
    try:
        import fcntl
        import struct
        import termios

        buf = fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, b"\0" * 8)
        rows, cols, xp, yp = struct.unpack("HHHH", buf)
        if xp and yp:
            return xp, yp
    except Exception:
        pass
    return 1000, 640  # fallback


def load_tsv(path):
    rows = []
    try:
        with open(path) as fh:
            header = fh.readline()
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7:
                    rows.append([float(x) for x in parts])
    except FileNotFoundError:
        pass
    return rows


def draw(rows, wpx, hpx):
    dpi = 100
    # leave a small margin so it never overflows the cell grid
    figw = max(4.0, (wpx * 0.92) / dpi)
    figh = max(2.6, (hpx * 0.82) / dpi)
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)
    ax2 = ax.twinx()
    if rows:
        ver = [r[0] for r in rows]
        mean = [r[2] for r in rows]
        mx = [r[3] for r in rows]
        ge05 = [r[6] for r in rows]
        ax.plot(ver, mean, "-o", lw=2, ms=5, color="#4c78a8", label="mean reward")
        ax.plot(ver, mx, "--", lw=1, color="#999999", label="max")
        ax2.plot(ver, ge05, "-s", lw=1.4, ms=4, color="#59a14f", label="frac >= 0.5")
    ax.set_xlabel("policy version (~ training step)")
    ax.set_ylabel("reward")
    ax2.set_ylabel("frac >= 0.5")
    ax.set_ylim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    n = rows[-1][1] if rows else 0
    last = rows[-1][2] if rows else 0.0
    cur_ver = int(rows[-1][0]) if rows else 0
    ax.set_title(
        "agpt-2b GRPO+LoRA (fp32, few-shot, linear reward)  "
        f"v={cur_ver}  mean={last:.3f}  n={int(n)}"
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    print("Live agpt-2b GRPO dashboard (kitcat + ambivalent). Ctrl-C to stop.")
    while True:
        try:
            subprocess.run(["bash", FEED, TSV, JSONL], capture_output=True, timeout=90)
        except Exception:
            pass
        rows = load_tsv(TSV)
        wpx, hpx = terminal_pixels()
        sys.stdout.write("\033[2J\033[H")  # clear + home so it redraws in place
        sys.stdout.flush()
        print(f"[{time.strftime('%H:%M:%S')}] policy versions: {len(rows)}")
        draw(rows, wpx, hpx)
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped.")
            break


if __name__ == "__main__":
    main()
