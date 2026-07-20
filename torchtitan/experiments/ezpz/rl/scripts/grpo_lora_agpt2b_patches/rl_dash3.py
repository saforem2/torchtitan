#!/usr/bin/env python3
"""Live multi-run GRPO reward dashboard (kitcat inline, ambivalent + Iosevka).

Self-contained: reads each run's rollout_samples.jsonl directly (over SSH if run
locally, or from the local FS if run on the cluster), aggregates reward by
max_policy_version (~ training step), and overlays every run's curve. No separate
feed script. Edit the RUNS table to add/remove experiments.

Run in a kitty terminal:
    /tmp/kitcat-venv/bin/python rl_dash3.py            # local, pulls over SSH
    python rl_dash3.py                                 # on the cluster (RL_LOCAL=1)
Env: RL_INTERVAL (sec, default 30), RL_LOCAL=1 (read local FS instead of SSH),
     RL_SSH (ssh target, default 'sunspot'), RL_SOCK (control socket).
"""
import os
import sys
import json
import time
import glob
import subprocess
import statistics

import matplotlib
matplotlib.use("module://kitcat")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import ambivalent

# ---- the runs to track: (tag, output-subdir, label) ----------------------
BASE = ("/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/"
        "outputs")
RUNS = [
    ("v5",     "rl_lora_agpt2b_train_v5", "v5 easy lr2e-5 r8 (prev best)"),
    ("w1",     "rl_lora_agpt2b_w1",       "w1 easy lr5e-5 r8"),
    ("w2",     "rl_lora_agpt2b_w2",       "w2 easy lr2e-5 r32"),
    ("w3",     "rl_lora_agpt2b_w3",       "w3 easy lr5e-5 16grp"),
    ("shaped", "rl_lora_agpt2b_shaped",   "shaped reward (r32 lr5e-5)"),
]
COLORS = ["#888888", "#4c78a8", "#59a14f", "#e45756", "#b279a2", "#f0a24b"]

INTERVAL = float(os.environ.get("RL_INTERVAL", "30"))
LOCAL = os.environ.get("RL_LOCAL") == "1"
SSH_TGT = os.environ.get("RL_SSH", "sunspot")
SOCK = os.environ.get("RL_SOCK", "/tmp/sunspot-master.sock")

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

# remote python that aggregates ALL runs' reward-by-version in one shot ------
_AGG = r'''
import json, collections, statistics, os
BASE = "%s"
RUNS = %s
out = {}
for tag, sub in RUNS:
    p = os.path.join(BASE, sub, "rollout_samples.jsonl")
    byv = collections.defaultdict(list)
    try:
        for line in open(p):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("is_validation"): continue
            t = d.get("turns") or []
            if not t: continue
            v = t[0].get("max_policy_version")
            if v is None: continue
            try: byv[int(v)].append(float(d.get("reward", 0)))
            except Exception: pass
    except FileNotFoundError:
        continue
    out[tag] = {v: round(statistics.mean(byv[v]), 5) for v in sorted(byv)}
print(json.dumps(out))
'''


def fetch():
    script = _AGG % (BASE, repr([(t, s) for t, s, _ in RUNS]))
    if LOCAL:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=90)
    else:
        # base64 the script so no quoting survives the ssh -> remote-shell hop.
        import base64
        b64 = base64.b64encode(script.encode()).decode()
        remote = "echo %s | base64 -d | python3" % b64
        r = subprocess.run(
            ["ssh", "-o", "ControlPath=" + SOCK, SSH_TGT, remote],
            capture_output=True, text=True, timeout=90)
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


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
    dpi = 100
    fig, ax = plt.subplots(figsize=(max(4.0, wpx * 0.92 / dpi),
                                    max(2.6, hpx * 0.80 / dpi)), dpi=dpi)
    parts = []
    for i, (tag, _sub, label) in enumerate(RUNS):  # noqa: B007  (_sub unused here)
        d = data.get(tag) or {}
        if not d:
            continue
        # JSON object keys are strings; sort numerically, look up by string key.
        keys = sorted(d, key=lambda k: int(k))
        vs = [int(k) for k in keys]
        means = [d[k] for k in keys]
        sm = [statistics.mean(means[max(0, j - 2):j + 1]) for j in range(len(means))]
        ls = "--" if tag == "v5" else "-"
        ax.plot(vs, sm, ls, lw=(1.5 if tag == "v5" else 2.0),
                color=COLORS[i % len(COLORS)], label=label)
        parts.append("%s=%.2f" % (tag, means[-1]))
    ax.axhline(0.248, lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("policy version (~ training step)")
    ax.set_ylabel("mean reward (3-step rolling)")
    ax.set_ylim(0, 1.02)
    ax.set_title("agpt-2b GRPO+LoRA on XPU -- " + "  ".join(parts))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    print("Live GRPO dashboard (%d runs, kitcat+ambivalent). Ctrl-C to stop."
          % len(RUNS))
    while True:
        try:
            data = fetch()
        except Exception as e:
            data = {}
            print("fetch error:", e)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        live = {k: (max(int(x) for x in v) if v else 0) for k, v in data.items()}
        print("[%s] latest policy versions: %s" % (time.strftime("%H:%M:%S"), live))
        draw(data, *terminal_pixels())
        try:
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print("\nstopped.")
            break


if __name__ == "__main__":
    main()
