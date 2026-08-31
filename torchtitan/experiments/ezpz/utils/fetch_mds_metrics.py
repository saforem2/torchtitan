"""Rebuild the MDS train_metrics.csv from the W&B chain.

The CSV is gitignored and was described only as "pulled separately from the MDS
W&B project". It is aurora_gpt/AuroraGPT, and it is a CHAIN of many runs rather
than one long run -- which is why looking for a single run at iteration 154,391
found nothing.

The metric keys are namespaced (`loss/iteration`, `loss/lm loss`,
`throughput/tps_per_gpu`), not the bare names the CSV uses, so every probe for
`lm_loss`/`iteration` came back empty.

Consumers need: iteration, lm_loss, tps_per_gpu (grad_norm and tflops are read
by plot_loss.py). Later runs win a contested iteration, matching concat_chain's
rule that the last-listed run owns a step.
"""
import csv
import os
import sys

import wandb

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
OUT = os.path.join(
    _REPO,
    "torchtitan/experiments/ezpz/docs/production/agpt/2b-mds/loss_data",
    "train_metrics.csv",
)

KEYS = {
    "iteration": ("loss/iteration", "training/iteration"),
    "lm_loss": ("loss/lm loss", "loss/lm_loss"),
    "grad_norm": ("loss/grad_norm", "training/grad-norm"),
    "tflops": ("throughput/tflops",),
    "tps_per_gpu": ("throughput/tps_per_gpu", "throughput/tokens_per_gpu_per_sec"),
}

api = wandb.Api(timeout=90)

# Collect the chain: every run in the project that logged an iteration.
runs = []
scanned = 0
for r in api.runs("aurora_gpt/AuroraGPT", order="-created_at", per_page=100):
    scanned += 1
    it = r.summary.get("loss/iteration") or r.summary.get("training/iteration")
    if it:
        runs.append((int(it), r))
    if scanned >= 600:
        break
runs.sort(key=lambda t: t[0])
print(f"  scanned {scanned} runs, {len(runs)} in the chain "
      f"(max iter {runs[-1][0] if runs else 0})", flush=True)

want = [k for ks in KEYS.values() for k in ks]
rows = {}
for i, (maxit, r) in enumerate(runs):
    try:
        n = 0
        for h in r.scan_history(keys=["loss/iteration"] + want, page_size=10000):
            it = h.get("loss/iteration") or h.get("training/iteration")
            if it is None:
                continue
            rec = {"iteration": int(it), "run_id": r.id}
            for col, cands in KEYS.items():
                if col == "iteration":
                    continue
                for c in cands:
                    if h.get(c) is not None:
                        rec[col] = h[c]
                        break
            # later runs win a contested iteration
            rows[int(it)] = rec
            n += 1
        if n:
            print(f"    [{i+1}/{len(runs)}] {r.id} -> {n} rows", flush=True)
    except Exception as e:
        print(f"    [{i+1}/{len(runs)}] {r.id} FAILED: {type(e).__name__}", flush=True)

if not rows:
    sys.exit("no rows recovered")

cols = ["iteration", "lm_loss", "grad_norm", "tflops", "tps_per_gpu", "run_id"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for it in sorted(rows):
        w.writerow(rows[it])

ks = sorted(rows)
print(f"  wrote {len(rows)} rows, iterations {ks[0]}..{ks[-1]} -> {OUT}")
