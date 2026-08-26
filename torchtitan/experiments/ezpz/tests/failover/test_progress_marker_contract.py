#!/usr/bin/env python3
"""ezpz's auto-retry progress detector must recognise torchtitan's step lines.

Job 8773440 (2026-08-26) lost 7 of 12 hours to this. ezpz 0.21.x used::

    _PROGRESS_MARKER_RX = re.compile(r"\bstep=\d+")

but torchtitan's metrics logger emits `step: 21800` with a COLON. So every
torchtitan seat scored zero progress no matter how much it trained, and any
seat whose attempt exited non-zero twice was filed `stuck_pre_training` and
ABANDONED with spares still free.

Measured on that job's own logs -- seats that had trained 201 / 504 / 782
steps all classified as no-progress. t3 was killed immediately after a
successful checkpoint at step 21,800.

ezpz >= 0.27.3 widened the pattern to accept `iter|step|epoch|batch|idx`
with either `=` or `:`. This test pins that contract from OUR side: the
detector is in ezpz, but the log format is torchtitan's, and nothing else
tests the pair together.

CPU-only, no cluster. Run:
    python -m pytest torchtitan/experiments/ezpz/tests/failover/test_progress_marker_contract.py -q
"""

import pytest

# A real torchtitan metrics line, ANSI stripped (the classifier sees raw bytes,
# but the marker itself is what matters).
TORCHTITAN_STEP = (
    "[2026-08-26 04:15:59][I][components/metrics:526:log] "
    "step: 21800  loss:  2.73836  grad_norm:  0.3722"
)
EZPZ_ITER = "[2026-08-26 04:15:59][I][ezpz/history:92:update] iter=1200 loss=2.7"


def _rx():
    ezpz_autoretry = pytest.importorskip("ezpz.launch_autoretry")
    return ezpz_autoretry._PROGRESS_MARKER_RX


def test_recognises_torchtitan_colon_form():
    """The exact format our production seats emit. This is the regression."""
    assert _rx().search(TORCHTITAN_STEP), (
        "ezpz's progress detector does not match torchtitan's 'step: N'. "
        "Every seat will score zero progress and healthy runs will be "
        "abandoned as stuck_pre_training -- see job 8773440."
    )


def test_recognises_ezpz_iter_form():
    """ezpz's own examples emit iter=; both must work."""
    assert _rx().search(EZPZ_ITER)


def test_still_rejects_a_log_with_no_training():
    """A seat that really never trained must still read as no-progress."""
    dead = (
        "[2026-08-26 01:06:19] rank 5 died from signal 11\n"
        "[2026-08-26 01:06:19][I][ezpz/launch:913] Execution finished with 143."
    )
    assert not _rx().search(dead)


@pytest.mark.parametrize(
    "line",
    [
        "step: 1",
        "step:1",
        "step = 42",
        "step=42",
        "iter: 7",
        "epoch: 3",
    ],
)
def test_separator_and_keyword_variants(line):
    """Separator carries no information; being strict about it costs recoveries."""
    assert _rx().search(line), f"progress marker not recognised: {line!r}"
