"""Verify the non-finite grad_norm guard actually SKIPS the optimizer step.

The smoke test (job 12473623) proved the guard is INERT on a healthy run --
10/10 bit-identical against the pre-guard commit. It did not and could not
prove the guard FIRES correctly, because inducing a real NaN needs a divergent
config and would confound the no-op measurement.

That untested half is the half that matters. A guard which is perfectly inert
and then fails to fire is worse than no guard, because it retires the concern
without doing the work.

This isolates the branch logic itself, on CPU, with no distributed backend and
no GPU: build a tiny model, poison the gradient, and assert the weights are
UNCHANGED. Run:

    python3 -m torchtitan.experiments.ezpz.tests.test_nonfinite_grad_guard

The guard in trainer.py reads:

    if not math.isfinite(float(grad_norm.item())):
        self.optimizers.zero_grad()
        logger.error(...)
    else:
        self.optimizers.step()
    self.lr_schedulers.step()

so the invariants under test are: (1) a non-finite grad_norm leaves every
parameter bit-identical, (2) grads are cleared so the poison cannot carry into
the next step, (3) a finite grad_norm still steps normally, and (4) the lr
scheduler advances either way -- skipping a step must not stall the schedule.
"""
import math

import torch


def _apply_guard(optimizer, scheduler, grad_norm):
    """The exact branch structure from FaultTolerantTrainer.train_step."""
    skipped = False
    if not math.isfinite(float(grad_norm.item())):
        optimizer.zero_grad()
        skipped = True
    else:
        optimizer.step()
    scheduler.step()
    return skipped


def _fixture():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    return model, opt, sched


def _backward(model):
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()


def case_nan_skips_step():
    model, opt, sched = _fixture()
    before = model.weight.detach().clone()
    _backward(model)
    # Poison the gradient the way a real divergence would.
    model.weight.grad[0, 0] = float("nan")
    skipped = _apply_guard(opt, sched, torch.tensor(float("nan")))
    assert skipped, "guard did not fire on nan grad_norm"
    assert torch.equal(model.weight.detach(), before), (
        "weights CHANGED on a skipped step -- the guard did not protect them"
    )
    assert model.weight.grad is None or not model.weight.grad.isnan().any(), (
        "nan grad survived zero_grad and would poison the next step"
    )
    return "nan  -> step skipped, weights unchanged, grads cleared"


def case_inf_skips_step():
    model, opt, sched = _fixture()
    before = model.weight.detach().clone()
    _backward(model)
    skipped = _apply_guard(opt, sched, torch.tensor(float("inf")))
    assert skipped, "guard did not fire on inf grad_norm"
    assert torch.equal(model.weight.detach(), before), "weights changed on inf"
    return "inf  -> step skipped, weights unchanged"


def case_finite_still_steps():
    """The critical negative case: the guard must not block healthy training."""
    model, opt, sched = _fixture()
    before = model.weight.detach().clone()
    _backward(model)
    skipped = _apply_guard(opt, sched, torch.tensor(1.5))
    assert not skipped, "guard fired on a FINITE grad_norm -- would halt training"
    assert not torch.equal(model.weight.detach(), before), (
        "weights unchanged on a healthy step -- the optimizer did not run"
    )
    return "1.5  -> step taken, weights updated"


def case_scheduler_advances_when_skipped():
    """A skipped step must not stall the lr schedule."""
    model, opt, sched = _fixture()
    lr0 = opt.param_groups[0]["lr"]
    _backward(model)
    _apply_guard(opt, sched, torch.tensor(float("nan")))
    lr1 = opt.param_groups[0]["lr"]
    assert lr1 < lr0, f"lr did not advance on a skipped step ({lr0} -> {lr1})"
    return f"lr advances even when skipped ({lr0} -> {lr1})"


def case_zero_is_finite():
    """0.0 is finite. A guard using truthiness instead of isfinite would fire."""
    model, opt, sched = _fixture()
    _backward(model)
    skipped = _apply_guard(opt, sched, torch.tensor(0.0))
    assert not skipped, "guard fired on grad_norm=0.0, which is finite"
    return "0.0  -> treated as finite, step taken"


CASES = [
    case_nan_skips_step,
    case_inf_skips_step,
    case_finite_still_steps,
    case_scheduler_advances_when_skipped,
    case_zero_is_finite,
]


def main():
    failures = 0
    for c in CASES:
        try:
            print(f"  PASS  {c.__name__}: {c()}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {c.__name__}: {e}")
    print(f"\n  {len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
