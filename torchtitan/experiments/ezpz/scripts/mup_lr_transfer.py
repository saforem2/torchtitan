#!/usr/bin/env python3
"""muP stage 4: does an LR tuned at a small width transfer to a large one?

The coordinate check (scripts/mup_coord_check.py) proves the parametrization
is internally consistent -- activation coordinates are width-invariant. It says
NOTHING about the payoff, which is the whole reason to want muP: that the
optimal LR measured on a cheap proxy is still optimal on the expensive target.
This measures that.

METHOD. Sweep a small grid of FIXED learning rates at each rung of the ladder
and record final loss. Under muP the loss-vs-LR curves should have their minima
at the SAME eta, because the parametrization already absorbs the width scaling
into the per-group LRs. Under standard parametrization the minimum drifts left
as width grows, which is exactly why every new scale currently needs its own
finder run.

WHY NOT ezpz/lr_finder.py. It writes one `curr_lr` into every param group
(lr_finder.py:141-143 and 193-196), which erases muP's per-group ratios --
under muP the hidden group must sit at eta/m while embeddings, readout and
norms sit at eta. A finder run would therefore measure a model that is not
parametrized the way the config says it is. Discrete fixed-LR runs keep the
grouping intact, at the cost of resolution: this resolves the optimum to a
grid point, not a continuum.

READING THE RESULT. muP transfers if argmin_eta is the same grid point at
every width, or within one step of it. It does NOT transfer if the argmin
walks monotonically with width -- that is the SP signature and means the
parametrization is not doing its job on this model. A flat curve at every
width means the grid is too narrow or the runs too short to discriminate, and
is a HARNESS failure rather than a result.
"""

import argparse
import json
import os
import subprocess
import sys

REPO = "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan"


def run_one(*, flavor: str, lr: float, steps: int, seq_len: int, seed: int,
            out_dir: str, mup: bool, port: int) -> dict:
    """One short training run at a fixed LR. Returns final + mean-tail loss."""
    tag = f"{flavor}_lr{lr:.3e}{'_mup' if mup else '_sp'}"
    log = os.path.join(out_dir, f"{tag}.log")
    argv = [
        sys.executable, "-m", "torchtitan.experiments.ezpz.train",
        "--module", "ezpz.agpt", "--config", flavor,
        f"--training.steps={steps}",
        f"--training.max-context-length={seq_len}",
        f"--training.num-tokens-per-microbatch-per-dp-rank={seq_len}",
        "--compile.no-enable", "--checkpoint.no-enable",
        "--validator.no-enable", "--metrics.no-enable-wandb",
        f"--debug.seed={seed}",
        # Constant LR. A warmup or decay schedule would confound the
        # comparison: the question is which eta is best, not which schedule.
        "--lr-scheduler.warmup-steps=1",
        "--lr-scheduler.decay-ratio=0.0",
    ]
    # SWEEPING eta CORRECTLY IS THE WHOLE MEASUREMENT, AND IT IS EASY TO GET
    # WRONG. The first version of this passed
    #   --optimizer.param-groups.0.optimizer-kwargs.lr=<eta>
    # reasoning that group 0 is the O(1) reference and the rest ride on it.
    # They do not. tyro overrides exactly the group it is told to: group 0 is
    # `^tok_embeddings\.`, ONE parameter. Groups 1-3 -- readout, norms, and
    # the 42-parameter hidden group that does the actual learning -- kept the
    # config default. Every run in a 16x sweep therefore trained at the same
    # effective LR, the loss varied by ~0.006 nats across the grid, and the
    # argmin was picked out of noise. Verified in the logs: group 0 showed
    # lr=0.0032 while groups 1-3 all showed lr=0.0008.
    #
    # There is no CLI form that rescales all four groups coherently, because
    # the hidden group must stay at eta/m while the others sit at eta -- a
    # ratio, not a common value. So the optimizer config is rebuilt per run
    # via MUP_ETA, read by the muP config functions in config_registry.
    env_eta = f"{lr:.12g}"
    env = dict(os.environ)
    env["MUP_ETA"] = env_eta
    env["MASTER_PORT"] = str(port)
    env.setdefault("TORCH_DEVICE", "cpu")
    with open(log, "w") as fh:
        rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    losses = []
    import re
    pat = re.compile(r"step:\s+(\d+)\s+loss:\s+([0-9.]+)")
    with open(log, errors="ignore") as fh:
        for line in fh:
            m = pat.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if m:
                losses.append((int(m.group(1)), float(m.group(2))))
    losses.sort()
    if not losses:
        return {"flavor": flavor, "lr": lr, "rc": rc, "final": None,
                "tail_mean": None, "n": 0, "log": log}
    tail = [v for _, v in losses[-max(1, len(losses) // 5):]]
    return {
        "flavor": flavor, "lr": lr, "rc": rc,
        "final": losses[-1][1],
        "tail_mean": sum(tail) / len(tail),
        "n": len(losses), "last_step": losses[-1][0], "log": log,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavors", required=True,
                    help="comma-separated, smallest first (the base rung)")
    ap.add_argument("--lrs", required=True, help="comma-separated eta grid")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sp", action="store_true",
                    help="run the SP control instead of muP")
    args = ap.parse_args()

    flavors = [f.strip() for f in args.flavors.split(",") if f.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    port = 29800
    for flavor in flavors:
        for lr in lrs:
            port += 1
            r = run_one(flavor=flavor, lr=lr, steps=args.steps,
                        seq_len=args.seq_len, seed=args.seed,
                        out_dir=args.out, mup=not args.sp, port=port)
            rows.append(r)
            print(f"  {flavor:16s} lr={lr:.3e}  rc={r['rc']}  "
                  f"final={r['final']}  tail={r['tail_mean']}", flush=True)

    # Per-flavor argmin over the grid.
    print("\n" + "=" * 74)
    print(f"{'flavor':16s} " + "".join(f"{lr:>11.2e}" for lr in lrs) + "   argmin")
    argmins = {}
    for flavor in flavors:
        vals = []
        for lr in lrs:
            m = [r for r in rows if r["flavor"] == flavor and r["lr"] == lr]
            vals.append(m[0]["tail_mean"] if m and m[0]["tail_mean"] is not None else None)
        cells = "".join(f"{v:>11.4f}" if v is not None else f"{'VOID':>11s}" for v in vals)
        good = [(v, lr) for v, lr in zip(vals, lrs) if v is not None]
        best = min(good)[1] if good else None
        argmins[flavor] = best
        print(f"{flavor:16s} {cells}   {best:.2e}" if best else f"{flavor:16s} {cells}   n/a")

    uniq = {v for v in argmins.values() if v is not None}
    idx = {lr: i for i, lr in enumerate(lrs)}
    spread = (max(idx[v] for v in uniq) - min(idx[v] for v in uniq)) if uniq else -1
    if not uniq:
        verdict = "NO DATA -- every run produced zero loss lines"
    elif len(uniq) == 1:
        verdict = "TRANSFER -- the optimum is the SAME grid point at every width"
    elif spread <= 1:
        verdict = ("TRANSFER (within one grid step) -- optima adjacent; "
                   "tighten the grid to distinguish")
    else:
        verdict = (f"NO TRANSFER -- the optimum moves {spread} grid steps "
                   "across the ladder")
    print("\nVERDICT: " + verdict)

    summary = {"flavors": flavors, "lrs": lrs, "steps": args.steps,
               "mup": not args.sp, "argmins": argmins,
               "argmin_grid_spread": spread, "verdict": verdict, "rows": rows}
    sp = os.path.join(args.out, "transfer_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[stage4] summary -> {sp}")
    return 0 if uniq else 2


if __name__ == "__main__":
    sys.exit(main())
