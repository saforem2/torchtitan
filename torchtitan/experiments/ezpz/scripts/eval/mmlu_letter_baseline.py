"""Constant-letter baselines for MMLU, and a letter-prior fit for a results file.

WHY THIS EXISTS. Every AuroraGPT MMLU score sits at chance, and the reason is
that the models answer with a fixed letter prior rather than from the question
(docs/evals/mmlu-letter-prior-at-chance.md). Two consequences make a plain
"is it above 0.25?" test misleading:

  1. The MMLU answer key is NOT uniform. Always answering "D" scores 0.2689
     and always answering "A" scores 0.2295 -- a 0.039 spread, ten times the
     0.0036 standard error. A model at 0.265 has not beaten chance; it has
     beaten "A" and lost to "D".

  2. Per-subject accuracy is over-dispersed and correlates across independent
     runs, which looks like real signal but is fully explained by each
     subject's answer-key letter distribution.

Usage:
    python3 mmlu_letter_baseline.py                    # baselines only
    python3 mmlu_letter_baseline.py path/to/results.json   # + fit that run
"""

import collections
import json
import sys


def letter_baselines():
    """Score of each constant-letter strategy, from the real answer keys."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    counts = collections.Counter(int(a) for a in ds["answer"])
    total = sum(counts.values())
    return {"ABCD"[i]: counts[i] / total for i in range(4)}, total


def subject_keyfreq():
    """Per-subject answer-key letter frequencies, for the prior fit."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    per = collections.defaultdict(collections.Counter)
    for subj, ans in zip(ds["subject"], ds["answer"]):
        per[subj][int(ans)] += 1
    out = {}
    for subj, c in per.items():
        n = sum(c.values())
        out[subj] = [c[i] / n for i in range(4)]
    return out


# The four category rollups carry the mmlu_ prefix but are not subjects;
# averaging them alongside the 57 double-counts and reads ~0.005 high.
CATEGORIES = {"humanities", "stem", "social_sciences", "other"}


def leaf_accuracies(path):
    d = json.load(open(path))
    res = d.get("results", d)
    out = {}
    for k, v in res.items():
        if not k.startswith("mmlu_") or not isinstance(v, dict):
            continue
        if "acc,none" not in v:
            continue
        name = k[len("mmlu_"):]
        if name in CATEGORIES:
            continue
        out[name] = v["acc,none"]
    return out, res.get("mmlu", {}).get("acc,none")


def fit_letter_prior(acc, keyfreq):
    """Least-squares p over acc(subject) = sum_L p_L * keyfreq_L(subject).

    A high R2 means the run is explained by a question-blind letter prior.
    """
    subjects = [s for s in acc if s in keyfreq]
    if len(subjects) < 8:
        return None
    import numpy as np

    A = np.array([keyfreq[s] for s in subjects])
    y = np.array([acc[s] for s in subjects])
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ p
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return p, (1 - ss_res / ss_tot if ss_tot else float("nan")), len(subjects)


def main():
    base, n = letter_baselines()
    print(f"MMLU constant-letter baselines (N={n})")
    for letter, score in sorted(base.items()):
        print(f"  always-{letter}: {score:.4f}")
    best = max(base, key=base.get)
    print(f"  BEST: {best} = {base[best]:.4f}   "
          f"(spread {max(base.values()) - min(base.values()):.4f})")

    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    acc, group = leaf_accuracies(path)
    if not acc:
        print(f"\n{path}: no leaf mmlu subjects found")
        return
    macro = sum(acc.values()) / len(acc)
    print(f"\n{path}")
    print(f"  lm-eval mmlu group key : {group}   <- publish THIS")
    print(f"  leaf macro over {len(acc):2d}     : {macro:.4f}")
    fit = fit_letter_prior(acc, subject_keyfreq())
    if fit:
        p, r2, ns = fit
        print(f"  fitted letter prior    : "
              f"A={p[0]:.2f} B={p[1]:.2f} C={p[2]:.2f} D={p[3]:.2f}")
        print(f"  R2 over {ns} subjects   : {r2:.2f}")
        print("  READ: a high R2 means the scores are explained by a "
              "question-blind letter prior.")


if __name__ == "__main__":
    main()
