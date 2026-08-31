"""Compare the z-loss smoke arms from their logs.

Standalone because the running job (12474386) snapshotted the PBS script at
qsub with a too-strict step regex ('loss: +' where the real format has two
spaces), so its own report block will say "no comparable steps". PBS freezing
the script at submit time is documented behaviour here; the logs are fine.
"""
import glob
import os
import re
import sys

ansi = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step: *(\d+) +loss: *([0-9.]+)")
GN = re.compile(r"step: *(\d+).*?grad_norm: *([0-9.]+)")

logd = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("outputs/logs/zloss-smoke/*"))[-1]
print(f"logdir: {logd}\n")


def load(name):
    path = f"{logd}/{name}.log"
    if not os.path.exists(path):
        return None, ""
    txt = ansi.sub("", open(path, errors="replace").read())
    return {int(a): float(b) for a, b in STEP.findall(txt)}, txt


bs, btxt = load("baseline")
zs, ztxt = load("zloss")

if bs is None or zs is None:
    print("  an arm has not finished yet")
    for n, d in (("baseline", bs), ("zloss", zs)):
        print(f"    {n}: {'absent' if d is None else str(len(d)) + ' steps'}")
    raise SystemExit(0)

common = sorted(set(bs) & set(zs))
print(f"baseline steps {len(bs)}, zloss steps {len(zs)}, common {len(common)}")

nan_b = len(re.findall(r"loss: *(nan|inf)", btxt))
nan_z = len(re.findall(r"loss: *(nan|inf)", ztxt))
zrep = len(re.findall(r"z_loss", ztxt))
print(f"nan/inf: baseline {nan_b}, zloss {nan_z}")
print(f"'z_loss' mentions in the zloss arm: {zrep}")

if common:
    print(f"\n{'step':>5} {'baseline':>10} {'zloss':>10} {'delta':>10}")
    show = common[:3] + (["..."] if len(common) > 6 else []) + common[-3:]
    for s in show:
        if s == "...":
            print("  ...")
            continue
        print(f"{s:>5} {bs[s]:>10.5f} {zs[s]:>10.5f} {zs[s] - bs[s]:>+10.5f}")

    deltas = [zs[s] - bs[s] for s in common]
    print(f"\nmean delta {sum(deltas) / len(deltas):+.5f}, "
          f"max |delta| {max(abs(d) for d in deltas):.5f}")

# throughput, so a wrecked step time is visible
def tps(txt):
    v = re.findall(r"tps: *([0-9,]+)", txt)
    return [float(x.replace(",", "")) for x in v]


tb, tz = tps(btxt), tps(ztxt)
if tb and tz:
    mb = sum(tb[-5:]) / len(tb[-5:])
    mz = sum(tz[-5:]) / len(tz[-5:])
    print(f"tps (last 5): baseline {mb:,.0f}  zloss {mz:,.0f}  "
          f"ratio {mz / mb:.3f}")

print()
if nan_z:
    print("VERDICT: z-loss arm produced NaN. Plumbing is broken; stop.")
elif not common:
    print("VERDICT: no comparable steps -- read the logs directly.")
elif zrep == 0:
    print("VERDICT: trains clean, but z_loss is NEVER REPORTED. The penalty")
    print("may be applied and invisible, which makes it untunable. Check that")
    print("the metrics dict reaches the logger before using this at 80B.")
else:
    print("VERDICT: plumbing OK -- trains, finite, penalty reported.")
    print("Says NOTHING about whether z-loss helps: 2B has no logit overflow")
    print("to fix. The 80B arm is the test that matters.")
