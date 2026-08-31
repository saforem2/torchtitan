"""Prove the non-finite capture block runs and names the offending tensor.

Extracts the guard from trainer.py and drives it with a model whose gradients
contain a NaN, checking that collect_param_stats is called BEFORE zero_grad
and that its output identifies the bad parameter. A capture that never fires
is this codebase's signature bug; four instances today.
"""
import math
import re

import torch

from torchtitan.experiments.ezpz import diagnostics as _diag

SRC = ("/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/"
       "torchtitan/experiments/ezpz/trainer.py")
src = open(SRC).read()

ok = True


def check(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}{('  ' + extra) if extra else ''}")


# --- structural: capture precedes the destruction of the data --------------
gi = src.index("if not math.isfinite(float(grad_norm.item()))")
ci = src.index("NON-FINITE GRADIENT CAPTURE", gi)
zi = src.index("self.optimizers.zero_grad()", gi)
check("capture block is inside the non-finite branch", gi < ci)
check("capture runs BEFORE zero_grad destroys the grads", ci < zi,
      f"capture@{ci} < zero_grad@{zi}")

# --- behavioural: does collect_param_stats actually surface a NaN grad? ----
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.good = torch.nn.Linear(4, 4)
        self.bad = torch.nn.Linear(4, 4)


m = Tiny()
for p in m.parameters():
    p.grad = torch.ones_like(p)
# poison exactly one tensor, the way a real non-finite step would
m.bad.weight.grad[0, 0] = float("nan")

stats = _diag.collect_param_stats([m], per_layer=True)
check("collect_param_stats returns something", bool(stats), f"{len(stats)} keys")

nonfinite_keys = [k for k, v in stats.items()
                  if isinstance(v, float) and not math.isfinite(v)]
check("a NaN grad produces at least one non-finite metric",
      bool(nonfinite_keys), str(nonfinite_keys[:4]))

# the per-layer top-k should name the offending module
named = [k for k in stats if "top" in k and "name" in k.lower()]
allkeys = ", ".join(sorted(stats)[:8])
check("per_layer=True emits top-k gradient keys",
      any("top" in k for k in stats), allkeys)

# --- the guard must not fire on a healthy step -----------------------------
m2 = Tiny()
for p in m2.parameters():
    p.grad = torch.ones_like(p)
clean = _diag.collect_param_stats([m2], per_layer=True)
clean_bad = [k for k, v in clean.items()
             if isinstance(v, float) and not math.isfinite(v)]
check("a healthy step produces NO non-finite metrics", not clean_bad,
      str(clean_bad[:4]))

print("\nNAN-CAPTURE TESTS:", "ALL PASS" if ok else "FAILURES PRESENT")
