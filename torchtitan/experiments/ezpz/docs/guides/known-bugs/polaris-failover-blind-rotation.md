# Polaris failover was always blind (no bad-node patterns registered)

**Status:** FIXED (2026-08-23, commit `341096f90`)
**Cost of the bug:** job 7550301 -- ~1 hour of 130 nodes, zero training steps
**Checkpoint impact:** none; `step-5600` intact (512 shards, `.metadata` present)

## Symptom

A Polaris training job fails before the first training step and the
failover log reports:

```
[auto-retry] blind rotation: <hostA> -> <hostB>
[auto-retry] attempt 2 (prior rc=124, sleeping 5s)...
[auto-retry] FAILOVER STOP: stuck_pre_training (two consecutive attempts
             with zero step= markers, rc=124)
```

The same failure recurs on the retry even though a spare was swapped in.

## Root cause (two stacked problems)

### 1. Polaris had NO registered scraper patterns

`ezpz.failover.patterns` shipped `aurora.py` and `sunspot.py` -- but no
`polaris.py`. So:

```python
>>> get_patterns_for_machine("polaris")
[]
```

`scrape_bad_nodes` returns `[]` for an empty pattern set, which the
caller correctly interprets as "no actionable hostname" and falls back
to **blind rotation**. `swap_one_blind()` rotates `active[0]` -- an
arbitrary node, by design -- so the *sick* node stays in the allocation
and the retry fails identically. Every Polaris failover ever run took
this path.

### 2. CUDA faults are unattributable without `--label`

Even with patterns registered, the dominant Polaris failure mode is a
CUDA error raised inside a rank's Python process:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy
or unavailable
```

PALS prints a rank's stderr **verbatim, with no host prefix**, unless
the launcher passes `--label`. Unlike an Aurora shepherd kill (which
PALS prefixes itself), this traceback carries nothing to match on.

In job 7550301 the *only* host-attributed line in the entire log was:

```
x3007c0s13b1n0.hsn.cm.polaris.alcf.anl.gov: rank 57 died from signal 15
```

...which is the **idle-output watchdog's own SIGTERM**. That node was a
victim of our teardown, not the culprit. Matching it would have swapped
a third innocent node.

## Fix

Both halves are required.

**1. Launcher labels its output.** `EZPZ_MPI_LABEL=1` makes
`ezpz/pbs.py` append `--label` to the mpiexec command, so every line
becomes `<fqdn> <rank>: <text>`.

Opt-in, not global: Aurora and Sunspot patterns anchor on *unlabeled*
`^<host>: ` lines, so enabling labeling everywhere would silently break
them. Only `submit_agpt_20b_autoretry.sh` (the Polaris script) sets it.

**2. A Polaris pattern module reads that prefix.**
`torchtitan/experiments/ezpz/failover_patterns/polaris.py` registers
five patterns:

| Pattern | Needs `--label`? |
|---|---|
| `polaris.cuda_device_unavailable` | yes |
| `polaris.cuda_init_error` | yes |
| `polaris.gpu_lost` | yes |
| `polaris.shepherd_signal_9` | no (PALS prefixes it) |
| `polaris.gloo_connection_closed` | no (IP is in the message) |

On an unlabeled log the label-dependent patterns match nothing and the
caller falls back to blind rotation -- the module is a **no-op**, never
a source of false positives.

## What is deliberately NOT matched

`rank N died from signal {11,15}`. On Polaris the common source of
SIGTERM is our own watchdog, so the named rank is downstream of the
teardown. This mirrors the innocent-cascade exclusion in `aurora.py`
and `_INNOCENT_RANK_CASCADE_RX` in `launch_autoretry`.

## Verification (real hardware, not simulated)

Probe job **7553963** confirmed the `--label` format -- notably that the
prefix is applied **per line** to a multi-line Python traceback on
stderr, which is what makes attribution possible at all:

```
x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov 1: RuntimeError: CUDA error: ...
```

Probe job **7553977** verified the whole chain through a real
`ezpz launch`:

```
mpiexec --envall --line-buffer --np=8 --ppn=4 --hostfile=... --label ...
LABEL_PRESENT: True
n_patterns: 5
8/8 passed                       # unit tests, on a compute node
SCRAPED: ['x3001c0s13b0n0.hsn.cm.polaris.alcf.anl.gov']
RESULT: PASS - culprit attributed
```

Against the two real logs:

| Log | Expected | Got |
|---|---|---|
| 7550301 (unlabeled, real failure) | `[]` -- stay silent | `[]` |
| 7553977 (labeled, real fault) | the culprit only | culprit only |

## Maintenance -- IMPORTANT

ezpz is installed from a **pinned git commit** and the venv is
periodically rebuilt and re-tarred. A fix applied only to
`site-packages` **silently disappears** on the next rebuild, reverting
Polaris failover to blind rotation with no error.

The module is therefore vendored at
`torchtitan/experiments/ezpz/failover_patterns/polaris.py`. After any
venv rebuild, and **before** `ezpz tar-env`:

```bash
bash torchtitan/experiments/ezpz/scripts/install_polaris_failover_patterns.sh
```

The script verifies registration and exits non-zero if it did not take.

Compute nodes run a **yeeted `/tmp/.venv`** unpacked from `.venv.tar.gz`,
not the live `.venv` -- so the tarball must be rebuilt for the fix to
reach a job. Note `ezpz tar-env` **skips silently (exit 0)** when
`.venv.tar.gz` already exists; move the old one aside first, then
confirm the new tarball actually contains the module:

```bash
tar tzf .venv.tar.gz | grep failover/patterns/polaris.py
```

## Upstream

Filed and fixed upstream so other Polaris users are not left on blind
rotation:

- Issue: [saforem2/ezpz#229](https://github.com/saforem2/ezpz/issues/229)
- PR: [saforem2/ezpz#230](https://github.com/saforem2/ezpz/pull/230)
  (`polaris.py` + the `EZPZ_MPI_LABEL` gate + 14 tests; 61 passing)

Once #230 lands **and** the pinned ezpz commit used by this venv is
advanced past it, the vendored copy under `failover_patterns/` becomes
redundant and the install script can be dropped. Until then the vendored
copy is what actually runs -- do not delete it just because the PR is
merged.

## Related

- `docs/guides/known-bugs/polaris-20b-tokenizer-mismatch.md`
- `torchtitan/experiments/ezpz/failover_patterns/README.md`
- `torchtitan/experiments/ezpz/tests/failover/test_polaris_scrape.py`
