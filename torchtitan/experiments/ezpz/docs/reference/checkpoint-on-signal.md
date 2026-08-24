# Checkpointing on SIGTERM/SIGINT

**Status:** shipped 2026-08-23, on by default (`save_on_signal=True`).
**Code:** `torchtitan/experiments/ezpz/signal_stop.py`, wired in `trainer.py`.

## The problem

Every walltime-bound PBS script in this tree wraps its launch in `timeout N`,
and PBS itself sends SIGTERM at walltime before SIGKILL. Neither was handled.
The process died wherever it stood, and every step computed since the last
`--checkpoint.interval` boundary was discarded.

Concretely, job 12473683 (30B / Mano / 64N, 158 s/step, interval 25):

| | |
|---|---:|
| projected stop (inner `timeout 25200`) | step ~147 |
| last checkpoint | step 125 |
| **discarded** | **~22 steps** |

That is ~1.4B tokens and ~1h of 64-node time, thrown away at the very end of
an 8h allocation. The loss is worst on exactly the runs that can least afford
it: a walltime-bound run has no natural end, so whatever it reached IS the
deliverable.

## Why the existing walltime guard was not enough

`trainer.py` already had a walltime guard that forces a final checkpoint near
a deadline, and this feature deliberately reuses its save-and-flush sequence
rather than duplicating it. But it could not cover this case:

* It only fires when `walltime_deadline_epoch` or `walltime_seconds` is
  configured. Both default to `0`, and job 12473683 set neither -- its log
  contains zero `walltime-ckpt:` lines.
* It reads a clock. A signal from `timeout`, from PBS, or from an operator is
  invisible to it, and an inner `timeout` that disagrees with the PBS
  walltime (the `30b_long` rc=124 lesson) puts the real stop somewhere the
  configured deadline does not predict.

The two are complements: set the deadline when you know it, and the signal
path catches the stop regardless.

## Why the handler does not save

`dcp.save` is a **collective**. Every rank must call it, in the same order,
from the same point in the program. A Python signal handler runs at an
arbitrary bytecode boundary, and the signal reaches each rank at a different
instant. Saving from the handler would have ranks entering the collective from
different places -- or one rank entering while another is mid-allreduce in the
backward pass -- which hangs the job or writes torn shards. That is strictly
worse than losing the tail.

So the handler does the minimum that is safe: **set a flag**. The train loop
polls it between steps, where every rank is already synchronized, and takes
the same `last_step=True` path that bypasses the interval, then blocks on
`maybe_wait_for_saving()` because `close()` does not wait for async saves.

## Two details that are easy to get wrong

**Logging inside a handler can deadlock.** The handler can interrupt a
`logger` call that holds the logging module lock; re-entering logging from the
handler then deadlocks. The notice is written with `os.write(2, ...)`, a
single reentrant syscall.

**A second signal must still kill.** If the first stop is ignored -- the save
wedges, a collective hangs -- an operator needs recourse short of SIGKILL. The
handler restores `SIG_DFL` after the first signal, so a repeat behaves
normally.

## Placement

Installed just before the training loop, not at startup, so it cannot fire
during dataloader build or checkpoint load, where there is no step worth
saving. A signal arriving before step 1 sets the flag and stops cleanly after
step 1 rather than being lost.

A signal during a step waits for that step to finish -- 158 s in the 30B case.
That is intended: interrupting mid-step is what is unsafe.

## Verification

Unit tests (`tests/test_signal_stop.py`, 7 cases) drive **real signals**
through `signal.signal` rather than calling the handler directly, so the
wiring is covered too, including a subprocess test that the second signal
actually terminates the process.

A loop-level integration test reproduces the 12473683 scenario against a fake
checkpointer and asserts the save sequence:

```
saves: 25, 50, 75, 100, 125, 147(last_step=True)   <- tail recovered
```

plus no double-save when a signal lands exactly on an interval boundary, and a
save when the signal arrives at step 1.

Confirmed on the Sunspot RC4 stack that `config.save_on_signal` resolves to
`True` -- the exact expression the loop evaluates. Note the field lives on
`FaultTolerantTrainer.Config`, not `Trainer.Config`; probing the latter shows
a misleading `False`.

## Turning it off

`--save-on-signal=false` restores the previous behavior (die where you stand).
There is no good reason to want this outside of debugging the handler itself.
