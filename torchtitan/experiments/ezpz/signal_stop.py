# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cooperative stop on SIGTERM/SIGINT so a killed run still checkpoints.

WHY THIS EXISTS. Every walltime-bound PBS script in this tree wraps its launch
in `timeout N ...`, and PBS itself sends SIGTERM at walltime before SIGKILL.
Without a handler the process dies wherever it happens to be, discarding every
step computed since the last checkpoint interval. Job 12473683 (30B/Mano/64N,
158 s/step, interval 25) was projected to stop at ~step 147 with its last save
at step 125: ~22 steps, ~1.4B tokens, ~1h of 64-node time, gone.

WHY THE HANDLER DOES NOT SAVE. `dcp.save` is a COLLECTIVE -- every rank must
call it, in the same order, from the same point in the program. A signal
handler runs at an arbitrary bytecode boundary, and the signal reaches each
rank at a different instant. Saving from the handler would have ranks entering
the collective from different places (or one rank entering while another is
mid-allreduce in the backward pass), which hangs the job or writes torn shards
-- strictly worse than losing the tail. So the handler does the minimum that is
safe: set a flag. The train loop polls it between steps, where every rank is
already synchronized, and takes the same forced-save path as the walltime
guard.

WHAT IS SAFE INSIDE A PYTHON SIGNAL HANDLER. CPython runs handlers in the main
thread between bytecodes, so a plain assignment is safe. Logging is NOT: the
handler can interrupt a `logger` call that holds the logging lock, and
re-entering it deadlocks. We therefore write the frame with `os.write(2, ...)`
(a single reentrant syscall) rather than through logging.

SECOND SIGNAL. If the first stop is ignored -- the save wedges, a collective
hangs -- a second signal must still kill the process, or an operator trying to
stop a stuck job has no recourse short of SIGKILL. The handler restores the
default disposition after the first signal, so a repeat behaves normally.
"""

import os
import signal
from typing import Optional


# Module-level, not an attribute on the trainer: the handler must reach this
# without a bound reference, and the signal disposition is per-process anyway.
_stop_requested: bool = False
_stop_signal: Optional[int] = None
_installed: bool = False


def stop_requested() -> bool:
    """Whether a stop signal has been received since installation."""
    return _stop_requested


def stop_signal_name() -> str:
    """Name of the signal that requested the stop (for log messages)."""
    if _stop_signal is None:
        return "none"
    try:
        return signal.Signals(_stop_signal).name
    except ValueError:
        return str(_stop_signal)


def _handler(signum: int, _frame) -> None:
    global _stop_requested, _stop_signal
    _stop_requested = True
    _stop_signal = signum

    # os.write, not logging: a handler that interrupts a logging call and then
    # re-enters logging deadlocks on the module lock. fd 2 is stderr.
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    try:
        os.write(
            2,
            f"\n[signal_stop] caught {name}: will checkpoint and stop after "
            f"the current step. Send it again to exit immediately.\n".encode(),
        )
    except OSError:
        # stderr closed/redirected -- the flag is what matters, not the notice.
        pass

    # Restore default disposition so a SECOND signal kills the process. Without
    # this, a wedged save would swallow every stop attempt.
    try:
        signal.signal(signum, signal.SIG_DFL)
    except (OSError, ValueError, RuntimeError):
        # Not the main thread, or the signal cannot be reset. The flag is still
        # set; the loop will act on it.
        pass


def install(signums=(signal.SIGTERM, signal.SIGINT)) -> bool:
    """Install the cooperative-stop handler. Returns True if anything was set.

    Idempotent. Safe to call from any rank: the flag is per-process and each
    rank observes its own signal, but the loop check that reads it is already
    a synchronization point for all ranks.
    """
    global _installed
    if _installed:
        return True

    installed_any = False
    for signum in signums:
        try:
            signal.signal(signum, _handler)
            installed_any = True
        except (OSError, ValueError, RuntimeError):
            # signal.signal raises off the main thread. Skip that signal
            # rather than failing the run -- losing the tail is much better
            # than not training at all.
            continue

    _installed = installed_any
    return installed_any


def reset_for_test() -> None:
    """Clear module state. For unit tests only."""
    global _stop_requested, _stop_signal, _installed
    _stop_requested = False
    _stop_signal = None
    _installed = False
