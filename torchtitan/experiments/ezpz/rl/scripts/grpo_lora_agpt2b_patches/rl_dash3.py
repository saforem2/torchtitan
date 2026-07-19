#!/usr/bin/env python3
"""Live 3-run GRPO comparison dashboard (v4/v5/v6) -- kitcat inline, ambivalent +
transparent + Iosevka, auto-fit to terminal. Run in kitty:
    /tmp/kitcat-venv/bin/python /tmp/rl_dash3.py
"""
import os, sys, glob, time, subprocess, collections
import matplotlib
matplotlib.use("module://kitcat")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import ambivalent

TSV = os.environ.get("RL_TSV", "/tmp/rl_reward3.tsv")
FEED = os.environ.get("RL_FEED", "/tmp/rl_feed3.sh")
INTERVAL = float(os.environ.get("RL_INTERVAL", "25"))

for d in (os.path.expanduser("~/Library/Fonts"), "/Library/Fonts"):
    for f in glob.glob(os.path.join(d, "Iosevka*.tt*")):
        try: fm.fontManager.addfont(f)
        except Exception: pass
plt.style.use(ambivalent.STYLES["ambivalent"])
_ios = sorted({f.name for f in fm.fontManager.ttflist if "iosevka" in f.name.lower()})
if _ios:
    plt.rcParams["font.family"] = next(
        (n for n in _ios if n.lower() in ("iosevka", "iosevka term")), _ios[0])
plt.rcParams.update({"savefig.transparent": True, "figure.facecolor": "none",
                     "axes.facecolor": "none"})

LABELS = {"v4": "v4 default lr2e-5", "v5": "v5 easy(1t,3n) lr2e-5", "v6": "v6 default lr5e-5"}
COLORS = {"v4": "#4c78a8", "v5": "#59a14f", "v6": "#e45756"}


def terminal_pixels():
    try:
        import fcntl, struct, termios
        buf = fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, b"\0" * 8)
        _, _, xp, yp = struct.unpack("HHHH", buf)
        if xp and yp:
            return xp, yp
    except Exception:
        pass
    return 1000, 640


def load(path):
    runs = collections.defaultdict(list)  # tag -> list[(ver,n,mean,max,ge05)]
    try:
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 6:
                    runs[p[0]].append((int(p[1]), int(p[2]), float(p[3]),
                                       float(p[4]), float(p[5])))
    except FileNotFoundError:
        pass
    return runs


def draw(runs, wpx, hpx):
    dpi = 100
    fig, ax = plt.subplots(figsize=(max(4.0, wpx * 0.92 / dpi),
                                    max(2.6, hpx * 0.80 / dpi)), dpi=dpi)
    parts = []
    for tag in ("v4", "v5", "v6"):
        rows = runs.get(tag)
        if not rows:
            continue
        rows.sort()
        ver = [r[0] for r in rows]
        mean = [r[2] for r in rows]
        ax.plot(ver, mean, "-o", lw=2, ms=4, color=COLORS[tag], label=LABELS[tag])
        parts.append(f"{tag}={mean[-1]:.3f}")
    ax.set_xlabel("policy version (~ training step)")
    ax.set_ylabel("mean reward")
    ax.set_ylim(0, 1.02)
    ax.set_title("agpt-2b GRPO+LoRA on XPU -- mean reward vs step   " + "  ".join(parts))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    print("Live 3-run GRPO dashboard (kitcat+ambivalent). Ctrl-C to stop.")
    while True:
        try:
            subprocess.run(["bash", FEED, TSV], capture_output=True, timeout=120)
        except Exception:
            pass
        runs = load(TSV)
        wpx, hpx = terminal_pixels()
        sys.stdout.write("\033[2J\033[H"); sys.stdout.flush()
        tot = {k: (v[-1][0] if v else 0) for k, v in runs.items()}
        print(f"[{time.strftime('%H:%M:%S')}] latest policy versions: {tot}")
        draw(runs, wpx, hpx)
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped."); break


if __name__ == "__main__":
    main()
