# Polaris failover patterns were correct and UNREACHABLE

**Status:** fixed 2026-08-27 (`1df90ca00`), verified end-to-end on the
cluster.
**Cost:** job 7560196, 130 nodes, **3h03m**, `Exit_status=124`, zero
training steps.

## Symptom

`ezpz launch --auto-retry` scraped the failed attempt, wrote a node into
`bad_nodes.txt` that had never errored, left both genuinely-bad nodes in
the allocation, and retried. Attempt 2 failed identically on the same
two nodes. `MAX_FAILOVER_RETRIES` is unbounded, so this would have
looped until the walltime.

This looks exactly like
[blind rotation](polaris-failover-blind-rotation.md), and it is not.
That bug is fixed and works.

## Root cause

`_detect_machine()` in `ezpz/failover/scrape.py` resolves the pattern
registry key by lowercasing `ezpz.get_machine()`. On a Polaris login
node that returns an **FQDN**:

```
ezpz.get_machine() -> "polaris-login-04.hsn.cm.polaris.alcf.anl.gov"
registry keys      -> ['aurora', 'perlmutter', 'polaris', 'sunspot']
```

so the lookup raises:

```
ModuleNotFoundError: No module named 'ezpz.failover.patterns.polaris-login-04'
```

Every Polaris pattern is unreachable. The scraper falls back to generic
matching, and the generic noise emitted by ~510 ranks blocked on a hung
collective is:

```
. This may indicate a possible application crash on rank 0 or a network set up issue.
```

which points at the **rank-0 node**. That is what landed in
`bad_nodes.txt` (`x3006c0s37b1n0` -- the launch node, which never
errored).

## Proof

Against the real log from the failed job:

```
scrape_bad_nodes("attempt-1.log")                    -> ModuleNotFoundError
scrape_bad_nodes("attempt-1.log", machine="polaris") -> ['x3007c0s13b1n0...', 'x3111c0s37b1n0...']
```

Those two are the true culprits. The pattern module is fine -- calling
its extractors directly returns exactly those two and nothing else, and
it deliberately does **not** match `died from signal {11,15}` (the
watchdog's own SIGTERM, which named an innocent node in the earlier
bug).

## Fix

Appended to `scripts/install_polaris_failover_patterns.sh`: normalize
the detected string to the **longest registered key it contains**.
Aurora and Sunspot already return bare names, so the match is the
identity there.

Verify with the auto-detect path, no explicit argument:

```bash
python3 -c 'from ezpz.failover.scrape import _detect_machine; print(_detect_machine())'
# must print: polaris
```

## Why every existing check passed

The installer already verified itself -- and could not catch this:

```bash
get_patterns_for_machine("polaris")   # EXPLICIT key -> 5 patterns -> green
```

`tests/failover/test_polaris_scrape.py:45` also passes
`machine="polaris"`. Both test the **policy** (do the regexes match?)
while production uses a different **binding** (can the module be
found?). Same shape as the grad-norm guard that was validated against
replayed logs and then `NameError`'d on step 1.

The new gate re-runs `_detect_machine()` in a fresh interpreter and
fails the install if it is not `polaris`.

## Two traps when shipping this

1. **The Lustre `.venv` is not what runs.** Jobs broadcast
   `.venv.tar.gz` to `/tmp/.venv` on every node. Patching site-packages
   alone ships nothing.
2. **`ezpz tar-env` silently skips.** With a tarball already present it
   logs `already exists, skipping creation` and exits 0. Move the stale
   one aside (do not `rm` -- use `backup`, or `mv` to a dated name) and
   force the rebuild. Then verify the artifact:

```bash
tar -xzOf .venv.tar.gz --wildcards '*/ezpz/failover/scrape.py' | grep -c 'normalize FQDN'
```

A `0` while `tar` is still writing is expected -- the archive is
sequential. Only trust the check after `tar` exits.

## The trigger was not a bad node

`cudaErrorDevicesUnavailable` here was **one GPU per node**, not node
failure. On each culprit all four A100s were present at 0% util, and
exactly one sat at 5 MiB while three held 427 MiB -- and that GPU's
index equalled the dead rank's local index (local 1 <-> GPU 1, local 0
<-> GPU 0). One GPU was still occupied at claim time. Node rotation
would have resolved it.

The failed rank does **not** exit: it prints the traceback and enters
the first collective with no device, hanging the other ranks. Nothing
marks the attempt failed, so auto-retry -- which scrapes on attempt exit
-- only engages once `IDLE_TIMEOUT` (3600 s here) fires.
