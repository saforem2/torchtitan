# Polaris bad-node scraper patterns (vendored)

`polaris.py` is a drop-in module for `ezpz.failover.patterns`. It is
vendored here (rather than only living in `.venv`) because the venv is
installed from a **pinned** ezpz commit and is rebuilt/re-tarred
periodically -- a fix applied only to `site-packages` silently
disappears on the next rebuild.

## Upstream status

**Merged.** [saforem2/ezpz#230](https://github.com/saforem2/ezpz/pull/230)
(issue [#229](https://github.com/saforem2/ezpz/issues/229)) landed on
`main`; `src/ezpz/failover/patterns/polaris.py` now ships with ezpz
itself (confirmed present in ezpz 0.27.3, 2026-08-25).

This vendored copy is therefore a **fallback**, not the source of truth.
It still matters for any venv pinned to a pre-#230 ezpz: installing from
such a pin gives you an ezpz whose `get_patterns_for_machine("polaris")`
returns `[]` again. Prefer advancing the pin (`uv pip install --no-deps
'git+https://github.com/saforem2/ezpz@main'`) over installing this copy.

## What it fixes

Before this module existed, `get_patterns_for_machine("polaris")`
returned `[]`. Every Polaris failover was therefore **blind**: the
scraper could never name a culprit, so `launch_autoretry` rotated
`active[0]` (an arbitrary, usually healthy node) and left the sick node
in the allocation.

Job **7550301** (2026-08-23, 130 nodes, 20B dolma chain) is the
postmortem. Two ranks raised

    torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are
    busy or unavailable

at `torch.cuda.set_device()`. Both attempts failed identically, the
watchdog fired twice, and the run stopped with `stuck_pre_training`
after ~1 hour of 130 nodes and **zero training steps**.

## Both halves are required

1. **Launcher passes `--label`.** PALS prefixes every output line with
   `<fqdn> <rank>: `. Without it a Python traceback carries no host and
   nothing can attribute it. Enabled by `EZPZ_MPI_LABEL=1`, which the
   Polaris submit script exports; `ezpz/pbs.py` reads it.
2. **These patterns read that prefix.**

On unlabeled logs the patterns match nothing and the caller falls back
to blind rotation -- i.e. this module is a **no-op**, never a source of
false positives.

## Why `--label` is opt-in

Aurora and Sunspot patterns anchor on *unlabeled* `^<host>: ` lines.
Turning labeling on globally would change their log shape and silently
break them. Only the Polaris submit script sets `EZPZ_MPI_LABEL=1`.

## What is deliberately NOT matched

`rank N died from signal {11,15}`. On Polaris the common source of
SIGTERM is the idle-output watchdog's own kill, so the named rank is a
victim of our teardown. In job 7550301 the *only* host-attributed line
in the whole log was exactly this -- naming `x3007c0s13b1n0`, which was
**not** the node that raised the CUDA error.

## Install into a fresh venv

    bash torchtitan/experiments/ezpz/scripts/install_polaris_failover_patterns.sh

Verify:

    python3 -c "from ezpz.failover.patterns import get_patterns_for_machine as g; print([p.name for p in g('polaris')])"

Test:

    python3 torchtitan/experiments/ezpz/tests/failover/test_polaris_scrape.py

## Empirical basis

The `--label` format was confirmed on real hardware, not assumed --
Polaris job **7553963** (2 nodes, debug queue):

    x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov 1: RuntimeError: CUDA error: ...

That probe also confirmed the prefix is applied **per line** to a
multi-line Python traceback on stderr, which is what makes attribution
of a CUDA fault possible at all.
