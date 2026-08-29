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

## Sequel: the fix was on a path a hung job never reaches

Fixing detection was necessary and **not sufficient**. The very next leg
(7560197) burned **4h04m** over three attempts, exhausted the spare
pool, and trained zero steps -- with every swap tagged `blind`:

```
x3006c0s37b1n0  blind  attempt=1     <- rank-0 node, never errored
x3005c0s13b0n0  blind  attempt=2     <- a SPARE swapped in at attempt 1
FAILOVER STOP: exhausted (no spare nodes left, rc=124)
```

`ezpz/launch_autoretry.py` classifies an idle-output watchdog kill
*before* consulting the scraper:

```python
if effective_rc == _WATCHDOG_RC:      # 124
    if not has_spares:
        return _result(TerminationReason.EXHAUSTED)
    return _result(TerminationReason.BAD_NODE_BLIND)
```

The stated rationale is "the hang IS the silence, so the scraper rarely
finds anything". True for a genuine silent hang; **false here.** A rank
that fails `set_device` prints a full traceback naming its host and
*then* hangs the first collective, so the log is loud. Run against the
three real attempt logs:

```
attempt-1 -> ['x3007c0s13b1n0...', 'x3111c0s37b1n0...']
attempt-2 -> same two
attempt-3 -> same two
```

The scraper had the correct answer three times and was never asked,
while blind rotation evicted three innocent hosts -- one of them a spare
it had just swapped in.

Fixed in the same installer script: on `rc=124`, if the scraper named
hosts, classify `BAD_NODE_KNOWN` so `swap_in()` replaces exactly those.
A genuinely empty scrape still blind-rotates, preserving the documented
behaviour for a real silent hang.

### Why three separate gates all read green

This was the third gate in one session that passed while production
failed:

| Gate | Why it could not catch the bug |
|---|---|
| `get_patterns_for_machine("polaris")` in the installer | passes an **explicit** key; production auto-detects |
| `tests/failover/test_polaris_scrape.py:45` | passes `machine="polaris"` for the same reason |
| (this bug) scraper verified correct on real logs | correct -- but on a code path `rc=124` never enters |

Each tested the **policy** (do the regexes match?) rather than the
**binding** (is this code reached, with these inputs, in production?).
The gate now asserts the watchdog branch can *reach* `BAD_NODE_KNOWN`
and still retains its blind fallback.

## Third act: one missing pattern abandoned a recoverable job

Leg 7567541 (2026-08-28) is the cleanest demonstration of why a scraper
gap is expensive, because it shows both halves at once.

**Attempt 1 -- the watchdog fix working, confirmed on hardware:**

```
[auto-retry] bad nodes: ['x3111c0s37b1n0...'] -- swapped 1
bad_nodes.txt:  x3111c0s37b1n0...  scraped  attempt=1
```

One node named, one swapped, 3 of 4 spares retained. Compare leg
7560197 under the old code: 3 attempts, all `blind`, three innocent
hosts evicted, pool exhausted.

**Attempt 2 -- a NEW error string, not in the pattern set:**

```
x3003c0s25b0n0 rank 495: CUDA error: invalid device ordinal
                          GPU device may be out of range, do you have enough GPUs?
```

A genuine node fault, not a config error: `nvidia-smi -L` listed all 4
A100s, `CUDA_VISIBLE_DEVICES` was unset, the hostfile had no duplicates
(128 lines / 128 unique), ranks 492-495 were placed correctly, and the
identical launch worked on the other 131 nodes. Same one-GPU signature
as before -- index 1 at 0 MiB while the other three held 427 MiB.

`scrape_bad_nodes()` returned `[]` for it.

**The consequence was not "one blind swap". The job died:**

```
FAILOVER STOP: stuck_pre_training (INFERRED, not observed: two consecutive
attempts showed no iter=/step=/epoch=/batch=/idx= line, and the scraper named
no host either, so the run is assumed to be dying before training starts...)
```

with **3 of 4 spares unused**. The gate is:

```python
if (prior_attempt_had_progress is False
    and not has_progress
    and not scraped_bad_nodes):
    return _result(TerminationReason.STUCK_PRE_TRAINING)
```

All three conditions had to hold. The first two were legitimately true.
The third was true **only** because the pattern was missing. So:

> a single unmatched error string -> empty scrape -> the heuristic reads
> "no host implicated" as evidence the *job* is broken -> a recoverable
> run is abandoned with spares in hand.

**The guard is not the bug.** It behaves correctly given its inputs; its
input was wrong. Do not patch `stuck_pre_training` -- widen the pattern
set instead. The message even anticipates this failure in its own text
("...this verdict is wrong and a recoverable job was abandoned").

Fix: added `invalid device ordinal` to `_CUDA_INIT_RX` in the vendored
`failover_patterns/polaris.py`. It satisfies that file's stated rule --
"only match conditions where the same code would succeed on a different
node".

### A patch cannot reach a job already running

Worth knowing before trying: patching `/tmp/.venv` on the head node of a
running job does nothing. `get_patterns_for_machine()` lazy-imports the
machine module **once** and caches it in a module-level `_PATTERNS`
dict:

```python
if machine not in _PATTERNS:
    importlib.import_module(f"ezpz.failover.patterns.{machine}")
```

That import already happened during attempt 1's scrape. Clearing
`__pycache__` does not help either -- the module object is live in
`sys.modules`. A pattern fix reaches the **next** job, via the tarball
broadcast, and only that.

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
