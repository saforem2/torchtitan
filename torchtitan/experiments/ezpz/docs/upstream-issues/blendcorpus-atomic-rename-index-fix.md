# Proposal: atomic-rename fix for the blendcorpus `_build_index_mappings` race (the real runtime fix)

> **Target repo:** `saforem2/blendcorpus` (NOT torchtitan-ezpz). This is the
> durable, barrier-free fix that makes the index cache build correctly at
> runtime at any TP -- eliminating the need for a pre-warm step entirely.

## The problem (recap)

`blendcorpus/data/gpt_dataset.py::_build_index_mappings` builds the per-corpus
`*_doc_idx.npy` / `*_sample_idx.npy` / `*_shuffle_idx.npy` on rank 0, writing
each to its **final filename directly** via `np.save`, then ALL ranks fall
through to `np.load` those same files -- with **no synchronization** between the
write and the read. A non-rank-0 reader that reaches `np.load` while rank 0 is
mid-write sees the final filename present but incomplete:
`EOFError: No data left in file` / `ValueError: mmap length is greater than
file size` / `_pickle.UnpicklingError: invalid load key '\x00'`.

At TP>1 this is not rare: ~(1 - 1/TP) of ranks (those whose TP coordinate != 0)
are not gated by any sibling subgroup barrier and reliably race. Verified on 80B
TP=4 cold-cache (jobs 8574063 32N, 8574172 4N -- ~75% of ranks crashed).

## Why the two existing approaches fall short

1. **`dist.barrier()` between write and read** (tried in `c7eb628e`, reverted in
   `74b09fd`): `_build_index_mappings` is called a *data-dependent* number of
   times (per-corpus x per-split; whether a rank takes the build branch depends
   on cache hit/miss), so the barrier fires a mismatched number of times across
   ranks -> oneCCL `allreduce_scaleout` participation mismatch / hang
   (`atl_comm->wait fails with status: 1`). A deadlock is strictly worse than
   the race, hence the revert.
2. **Pre-warm the cache** (current mitigation, `prewarm_blendcorpus_*.sh`):
   works but is a manual pre-step, and the all-ranks prewarm can itself race at
   TP>1 (must be run single-rank). Operational, not a real fix.

## The fix: atomic publish via temp-file + `os.rename`

`os.rename` (same filesystem) is **atomic** -- including on Lustre. If rank 0
writes each index to a unique temp path, fsyncs, then renames to the final
name, a reader doing `np.load(final)` sees **either**:
- the file absent (final name not yet renamed) -> handle with a short retry, or
- the file fully present and complete (rename published the finished file).

It can NEVER see a partially-written final file. No barrier, no collective, no
deadlock risk, correct at any TP. This is the standard "atomic publish" pattern.

### Writer side (rank 0), in `_build_index_mappings`

Replace each `np.save(idx_path[k], arr, ...)` with a temp-then-rename helper:

```python
import os, tempfile

def _atomic_np_save(final_path, arr):
    # Write to a unique temp in the SAME dir (same fs -> rename is atomic),
    # flush+fsync, then atomically publish via os.replace.
    d = os.path.dirname(final_path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr, allow_pickle=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final_path)   # atomic publish (overwrites if present)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise
```

Use `_atomic_np_save(idx_path["doc"], doc_idx)` etc. Also publish the `.desc`
marker LAST (write desc to a tmp, fsync, rename) -- and treat `.desc` presence
as the "all three idx files are complete" signal (it is written only after the
three `np.save`s succeed).

### Reader side (all ranks), before the `np.load` block

Because a non-builder can arrive before rank 0 has published, add a bounded
spin-wait on the `.desc` marker (the last thing the writer publishes), then load:

```python
import time
deadline = time.time() + 600  # generous; builds are seconds, fs latency varies
while not os.path.exists(idx_path["desc"]):
    if time.time() > deadline:
        raise RuntimeError(f"timed out waiting for index cache {idx_path['desc']}")
    time.sleep(0.5)
doc_idx = np.load(idx_path["doc"], allow_pickle=True, mmap_mode="r")
sample_idx = np.load(idx_path["sample"], allow_pickle=True, mmap_mode="r")
shuffle_idx = np.load(idx_path["shuffle"], allow_pickle=True, mmap_mode="r")
```

The spin-wait is per-rank wall-clock (no collective), so it cannot deadlock the
way a barrier does -- a rank that never sees the file fails its own timeout
rather than hanging the job. (`.desc` is published only after all three idx
files are renamed into place, so its presence guarantees the loads succeed.)

## Why this is safe where the barrier was not

- No `torch.distributed` collective is added, so the data-dependent call-count
  that broke the barrier approach is irrelevant.
- `os.replace` is atomic on POSIX + Lustre; readers never observe a torn file.
- The spin-wait degrades to an immediate load on a cache hit (`.desc` already
  present), so the warm-cache fast path is unchanged.

## Validation plan (once patched in saforem2/blendcorpus)

1. Reinstall the patched blendcorpus into `.venv` (`--no-deps`), rebuild tarball.
2. Re-run the 80B 4N TP=4 smoke from a **cold** `CKPT_DIR` -- must build the
   index and proceed to training with ZERO EOFError/mmap/invalid-load, no
   prewarm.
3. Re-run at 32N TP=4 (the original failure scale) to confirm at scale.
4. Confirm warm-cache path still fast (second run loads instantly).

Until this lands, the operational workaround is the single-rank prewarm
(`scripts/prewarm_blendcorpus_singlerank.sh`).
