#!/usr/bin/env python3
"""Regression tests for the Polaris bad-node scraper patterns.

Motivated by job 7550301 (2026-08-23): two ranks raised a CUDA
device fault, but the tracebacks were UNLABELED, so the scraper
found no host, failover blind-rotated a HEALTHY node, and the sick
node stayed in the allocation. ~1h of 130 nodes, zero steps.

The fix has two halves and these tests cover both:

  1. `EZPZ_MPI_LABEL=1` makes the launcher pass PALS `--label`, so
     every line is prefixed `<fqdn> <rank>: `.
  2. `ezpz.failover.patterns.polaris` reads that prefix.

The most important test here is NOT that we find the bad node --
it is `test_unlabeled_log_yields_nothing`: on unlabeled input the
scraper must stay SILENT rather than tag the innocent node named by
the watchdog's own SIGTERM. A false positive swaps a healthy node
and leaves the real culprit in place, which is strictly worse than
blind rotation.

Run from repo root:
    python3 torchtitan/experiments/ezpz/tests/failover/test_polaris_scrape.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ezpz.failover.patterns.polaris import normalize_polaris_hostname
from ezpz.failover.scrape import scrape_bad_nodes

H = "hsn.cm.polaris.alcf.anl.gov"


def _scrape(text: str) -> list[str]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".log", delete=False
    ) as fh:
        fh.write(text)
        path = fh.name
    try:
        return scrape_bad_nodes(path, machine="polaris")
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The regression that motivated the module.
# ---------------------------------------------------------------------------

def test_unlabeled_log_yields_nothing() -> None:
    """Job 7550301's real shape: a bare traceback plus a watchdog
    SIGTERM naming an INNOCENT host. Must return [] so the caller
    falls back to blind rotation -- never tag the SIGTERM victim."""
    log = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 528, in <module>\n'
        "    ezpz.distributed.setup_torch()\n"
        "torch.AcceleratorError: CUDA error: CUDA-capable device(s) "
        "is/are busy or unavailable\n"
        f"x3007c0s13b1n0.{H}: rank 57 died from signal 15\n"
    )
    assert _scrape(log) == [], "must not tag the SIGTERM victim"


def test_labeled_cuda_fault_is_attributed() -> None:
    """Same fault WITH --label: the culprit is named."""
    log = (
        f"x3006c0s13b1n0.{H} 57: Traceback (most recent call last):\n"
        f"x3006c0s13b1n0.{H} 57: torch.AcceleratorError: CUDA error: "
        "CUDA-capable device(s) is/are busy or unavailable\n"
        f"x3007c0s13b1n0.{H}: rank 57 died from signal 15\n"
    )
    assert _scrape(log) == [f"x3006c0s13b1n0.{H}"]


def test_cascade_victims_never_tagged() -> None:
    """signal 11/15 and nonzero exits are downstream effects."""
    log = (
        f"x3001c0s1b0n0.{H} 5: torch.AcceleratorError: CUDA error: "
        "CUDA-capable device(s) is/are busy or unavailable\n"
        f"x3002c0s1b0n0.{H}: rank 5 died from signal 15\n"
        f"x3003c0s1b0n0.{H}: rank 9 died from signal 11\n"
        f"x3004c0s1b0n0.{H}: rank 3 exited with code 1\n"
    )
    assert _scrape(log) == [f"x3001c0s1b0n0.{H}"]


# ---------------------------------------------------------------------------
# Other node-fatal signatures.
# ---------------------------------------------------------------------------

def test_cuda_init_variants() -> None:
    log = (
        f"x3010c0s1b0n0.{H} 2: RuntimeError: CUDA error: "
        "no CUDA-capable device is detected\n"
        f"x3011c0s1b0n0.{H} 7: RuntimeError: CUDA error: "
        "initialization error\n"
    )
    assert _scrape(log) == [
        f"x3010c0s1b0n0.{H}",
        f"x3011c0s1b0n0.{H}",
    ]


def test_shepherd_sig9_needs_no_label() -> None:
    """PALS prefixes shepherd kills itself."""
    log = f"x3013c0s1b0n0.{H}: shepherd died from signal 9\n"
    assert _scrape(log) == [f"x3013c0s1b0n0.{H}"]


def test_hsn_suffix_dedupes_to_one_node() -> None:
    """`-hsn0` and the plain form name the SAME node."""
    log = (
        f"x3014c0s1b0n0-hsn0.{H} 1: torch.AcceleratorError: CUDA error: "
        "CUDA-capable device(s) is/are busy or unavailable\n"
        f"x3014c0s1b0n0.{H} 1: RuntimeError: CUDA error: "
        "CUDA-capable device(s) is/are busy or unavailable\n"
    )
    assert _scrape(log) == [f"x3014c0s1b0n0.{H}"]


def test_device_side_assert_not_matched() -> None:
    """A device-side assert is an APPLICATION defect, not a node fault.

    It comes from a failing assertion inside a kernel (out-of-range
    index, bad label, invalid model input), so swapping the node changes
    nothing -- the retry hits the same assert on fresh hardware while
    the loop burns a spare per attempt. Worse than blind rotation, since
    it also retires a healthy node each time.

    Rule for this pattern set: only match conditions where the same code
    would succeed on a different node. (Caught in review of
    saforem2/ezpz#230.)
    """
    log = (
        f"x3020c0s1b0n0.{H} 3: RuntimeError: CUDA error: "
        "device-side assert triggered\n"
    )
    assert _scrape(log) == []


def test_clean_log_yields_nothing() -> None:
    log = "step=1 loss=12.03\nstep=2 loss=11.87\n"
    assert _scrape(log) == []


# ---------------------------------------------------------------------------
# Hostname normalizer.
# ---------------------------------------------------------------------------

def test_normalizer() -> None:
    cases = [
        (f"x3006c0s13b1n0.{H}", f"x3006c0s13b1n0.{H}"),
        (f"x3006c0s13b1n0-hsn0.{H}", f"x3006c0s13b1n0.{H}"),
        (
            "x3006c0s13b1n0.hostmgmt2042.cm.polaris.alcf.anl.gov",
            f"x3006c0s13b1n0.{H}",
        ),
        ("x3006c0s13b1n0.something-else.example.com", None),
        ("some-other-host", None),
        # An Aurora host must NOT normalize as Polaris.
        ("x1234c0s0b0n0.hsn.cm.aurora.alcf.anl.gov", None),
    ]
    for raw, want in cases:
        got = normalize_polaris_hostname(raw)
        assert got == want, f"{raw!r}: got {got!r}, want {want!r}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
