# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cooperative SIGTERM/SIGINT stop.

This path only executes while a job is being killed, so it is exactly the kind
of code that rots unnoticed. These tests drive it with real signals rather than
by calling the handler directly, so they also cover the signal.signal wiring.
"""

import os
import signal
import subprocess
import sys

import pytest

from torchtitan.experiments.ezpz import signal_stop


@pytest.fixture(autouse=True)
def _reset():
    signal_stop.reset_for_test()
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)
    signal_stop.reset_for_test()


def test_flag_starts_clear():
    assert signal_stop.stop_requested() is False
    assert signal_stop.stop_signal_name() == "none"


def test_sigterm_sets_flag():
    assert signal_stop.install() is True
    assert signal_stop.stop_requested() is False
    os.kill(os.getpid(), signal.SIGTERM)
    assert signal_stop.stop_requested() is True
    assert signal_stop.stop_signal_name() == "SIGTERM"


def test_sigint_sets_flag():
    assert signal_stop.install() is True
    os.kill(os.getpid(), signal.SIGINT)
    assert signal_stop.stop_requested() is True
    assert signal_stop.stop_signal_name() == "SIGINT"


def test_install_is_idempotent():
    assert signal_stop.install() is True
    assert signal_stop.install() is True
    os.kill(os.getpid(), signal.SIGTERM)
    assert signal_stop.stop_requested() is True


def test_handler_does_not_raise_into_the_loop():
    """A signal must not surface as an exception at an arbitrary bytecode.

    If it did, it would abort the training step it interrupted instead of
    letting the step finish and the loop take the save path.
    """
    signal_stop.install()
    total = 0
    for i in range(1000):
        if i == 500:
            os.kill(os.getpid(), signal.SIGTERM)
        total += i
    assert total == sum(range(1000))
    assert signal_stop.stop_requested() is True


def test_second_signal_kills_the_process():
    """The first signal is cooperative; a second must still terminate.

    Otherwise a wedged save would swallow every stop attempt and leave SIGKILL
    as the only recourse. Run in a subprocess because it dies by design.
    """
    prog = (
        "import os, signal, time, sys;"
        "sys.path.insert(0, %r);"
        "from torchtitan.experiments.ezpz import signal_stop;"
        "signal_stop.install();"
        "os.kill(os.getpid(), signal.SIGTERM);"
        "assert signal_stop.stop_requested();"
        # second one must hit the restored default disposition
        "os.kill(os.getpid(), signal.SIGTERM);"
        "time.sleep(5);"
        "sys.exit(0)"
    ) % os.getcwd()
    p = subprocess.run([sys.executable, "-c", prog], capture_output=True, timeout=30)
    assert p.returncode == -signal.SIGTERM, (
        f"expected death by SIGTERM, got rc={p.returncode} "
        f"stderr={p.stderr.decode()[:400]}"
    )


def test_handler_writes_notice_to_stderr():
    prog = (
        "import os, signal, sys;"
        "sys.path.insert(0, %r);"
        "from torchtitan.experiments.ezpz import signal_stop;"
        "signal_stop.install();"
        "os.kill(os.getpid(), signal.SIGTERM);"
        "sys.exit(0)"
    ) % os.getcwd()
    p = subprocess.run([sys.executable, "-c", prog], capture_output=True, timeout=30)
    assert p.returncode == 0
    assert b"signal_stop" in p.stderr
    assert b"SIGTERM" in p.stderr
