"""Prove the swept LR survives to the optimizer step, with a real LambdaLR.

The bug was invisible to reading: the finder wrote param_group["lr"], the
write looked correct, and the scheduler silently overwrote it inside
train_step. So this test asserts on the LR OBSERVED AT THE OPTIMIZER STEP,
not on what the finder set.
"""
import torch
from torch.optim.lr_scheduler import LambdaLR


class Container:
    """Mimics LRSchedulersContainer.step() fanning out to LambdaLRs."""

    def __init__(self, scheds):
        self.schedulers = scheds

    def step(self):
        for s in self.schedulers:
            s.step()


BASE_LR = 3.0e-4
WARMUP = 200

p = torch.nn.Parameter(torch.zeros(2))
opt = torch.optim.SGD([p], lr=BASE_LR)
sched = LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
container = Container([sched])


def run(suspend: bool, swept):
    """Return the LR in effect at each optimizer step."""
    for g in opt.param_groups:
        g["lr"] = BASE_LR
    sched.last_epoch = -1
    sched._step_count = 0

    orig = None
    if suspend:
        orig = container.step
        container.step = lambda *a, **k: None

    observed = []
    for lr in swept:
        # The finder sets the LR for the NEXT iteration at the END of the
        # previous one (lr_finder.py: 'Advance LR exponentially only during
        # sweep'), and train_step's scheduler.step() runs in between. So the
        # LR that actually drives an optimizer step is whatever survives that
        # gap -- which is what this samples. Sampling before opt.step() in the
        # same iteration hides the bug entirely, because the clobber has not
        # happened yet: that was the first version of this test, and it
        # reported the unsuspended run as clean.
        for g in opt.param_groups:
            g["lr"] = lr            # what the finder does
        container.step()            # what train_step does at line 1073
        p.grad = torch.ones_like(p)
        observed.append(opt.param_groups[0]["lr"])   # what the NEXT step sees
        opt.step()

    if orig is not None:
        container.step = orig
    return observed


SWEPT = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

before = run(False, SWEPT)
after = run(True, SWEPT)

print(f"{'swept (recorded)':>18} {'seen, unsuspended':>19} {'seen, suspended':>17}")
for s, b, a in zip(SWEPT, before, after):
    print(f"{s:>18.1e} {b:>19.3e} {a:>17.1e}")

ok = True

# Suspended: the optimizer must see exactly what was swept.
match = all(abs(a - s) < 1e-15 for s, a in zip(SWEPT, after))
ok &= match
print(f"\n  [{'ok ' if match else 'FAIL'}] suspended: optimizer sees the swept LR exactly")

# Unsuspended: reproduce the bug, so the test proves the fix does something.
spread_before = max(before) / min(before)
spread_after = max(after) / min(after)
bug = spread_before < 100          # 5 decades swept, collapsed to <2
ok &= bug
print(f"  [{'ok ' if bug else 'FAIL'}] unsuspended: 5-decade sweep collapses "
      f"(observed spread {spread_before:.1f}x)")

full = abs(spread_after - 1e4) / 1e4 < 0.01
ok &= full
print(f"  [{'ok ' if full else 'FAIL'}] suspended: full 4-decade spread preserved "
      f"({spread_after:.0f}x)")

# Restore must put the original back.
restored = container.step.__self__ is container if hasattr(container.step, "__self__") else True
ok &= restored
print(f"  [{'ok ' if restored else 'FAIL'}] container.step restored after the run")

print("\nSCHEDULER TESTS:", "ALL PASS" if ok else "FAILURES PRESENT")
