#!/usr/bin/env python3
"""Ceiling-attack chart: every run re-scored on the SAME char-ratio(power=1)
reward (v5's actual reward) so shaped is comparable on one axis. The
reconstruction is self-validated -- for the char-ratio runs the reconstructed
score matches their stored reward exactly (see beat-v5-sweep.md)."""
import os
import re
import json
import difflib
import collections
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

# Resolve the repo from this file's location, not a hardcoded machine path:
# the plotter is authored on Sunspot but refresh_all.sh runs it on Aurora too,
# where /lus/tegu does not exist. docs/production/rl/grpo/ -> repo root is 7 up.
REPO = Path(__file__).resolve().parents[7]
BASE = str(REPO / "outputs")
RUNS = [
    ("v5", "rl_lora_agpt2b_train_v5", "v5 lr2e-5 r8 (prev best)", "--", 1.6),
    ("w1", "rl_lora_agpt2b_w1", "w1 lr5e-5 r8", "-", 1.4),
    ("w2", "rl_lora_agpt2b_w2", "w2 lr2e-5 r32", "-", 1.4),
    ("w3", "rl_lora_agpt2b_w3", "w3 lr5e-5 16grp", "-", 1.4),
    ("shaped", "rl_lora_agpt2b_shaped", "shaped reward (r32 lr5e-5)", "-", 2.6),
]


def answer_lines(text, tag):
    b = re.findall(r"<\s*%s\s*>(.*?)</\s*%s\s*>" % (tag, tag), text,
                   re.DOTALL | re.IGNORECASE)
    return [l.strip() for l in b[-1].splitlines() if l.strip()] if b else []


def score_p1(text, expected):
    pred = answer_lines(text, "alphabetical_sorted")
    if not pred:
        return 0.0
    pt = "\n".join(l.lower() for l in pred)
    et = "\n".join(l.lower() for l in expected)
    return difflib.SequenceMatcher(None, pt, et).ratio()


def parse_prompt(text):
    m = re.search(r"by (FIRST|LAST) name:\s*(.+)", text)
    if not m:
        return None
    by_first = m.group(1) == "FIRST"
    names = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+", m.group(2).split("\n")[0])
    return by_first, names


def sort_key(name, by_first):
    parts = re.findall(r"[A-Z][a-z]*", name)
    first = parts[0] if parts else name
    last = parts[1] if len(parts) > 1 else ""
    return (first, last) if by_first else (last, first)


def prompt_of(turn):
    for m in (turn.get("prompt_messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content") or ""
    return ""


def recon_by_version(path):
    byv = collections.defaultdict(list)
    try:
        f = open(path)
    except FileNotFoundError:
        return {}
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("is_validation"):
            continue
        turns = d.get("turns") or []
        if len(turns) != 1:
            continue
        t = turns[0]
        v = t.get("max_policy_version")
        if v is None:
            continue
        pr = parse_prompt(prompt_of(t))
        if not pr or not pr[1]:
            continue
        by_first, names = pr
        expected = sorted(names, key=lambda x: sort_key(x, by_first))
        cm = t.get("completion_message") or {}
        text = (cm.get("content") or "") if isinstance(cm, dict) else ""
        byv[int(v)].append(score_p1(text, expected))
    return {v: statistics.mean(byv[v]) for v in sorted(byv)}


fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
finals = {}
for tag, sub, label, ls, lw in RUNS:
    d = recon_by_version(os.path.join(BASE, sub, "rollout_samples.jsonl"))
    if not d:
        continue
    vs = sorted(d)
    means = [d[v] for v in vs]
    sm = [statistics.mean(means[max(0, i - 2):i + 1]) for i in range(len(means))]
    ax.plot(vs, sm, ls, lw=lw, label=label,
            zorder=(5 if tag == "shaped" else 3))
    finals[tag] = statistics.mean(means[-3:])

ax.axhline(0.248, lw=0.8, ls=":", alpha=0.6)
ax.text(2, 0.258, "char-ratio ceiling ~0.25", fontsize=7, alpha=0.7)
ax.set_xlabel("policy version (~ training step)")
ax.set_ylabel("reward re-scored on char-ratio(p1)  [3-step rolling]")
ax.set_ylim(0, 0.6)
ax.set_title("agpt-2b GRPO+LoRA on XPU -- breaking the reward-shape ceiling")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()

OUT = str(Path(__file__).resolve().parent / "aurora2b" / "charts")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, "ceiling-attack." + ext),
                bbox_inches="tight", transparent=True)
print("saved ceiling-attack.{png,svg}")
print("finals (char-ratio p1, last-3-version mean):",
      {k: round(v, 3) for k, v in finals.items()})
