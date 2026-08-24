# ENOSPC on /lus/tegu while `df` reports 1.1P free

**Seen:** 2026-08-23, jobs 12473740/41/42 (three concurrent 30B checkpoints).
**Symptom:** `OSError: [Errno 28] No space left on device` inside DCP
`write_data`, killing every rank at the first checkpoint.

## Why `df` lies here

`/lus/tegu` has FOUR OSTs, and they are wildly asymmetric:

| OST | size | used |
|---|---:|---:|
| OST0 | 624 TB | 6-7% |
| OST1 | 624 TB | 6-7% |
| **OST2** | 27 TB | **100%** |
| **OST3** | 27 TB | **100%** |

`df` sums all four and reports ~1.1P free, which is true and useless. A file is
written to the OSTs its stripe layout names, so a file assigned to OST2 or OST3
gets ENOSPC no matter what the total says.

**Use `lfs df /lus/tegu`, not `df`.** Also check `lfs quota -u $USER /lus/tegu`
-- quota exhaustion produces the same errno with a different cause (it was not
the cause here: all limits were 0/unset).

## Why it hit these jobs

The default layout on this filesystem is `stripe_count: 1` with
`stripe_offset: -1` (round robin), so each file lands on exactly ONE OST and
has a ~50% chance of drawing a full one. A DCP checkpoint is thousands of
files, so at 30B scale (241-285 GB per checkpoint, three arms writing at once)
hitting a full OST is not a risk, it is a certainty.

## Fix

Pin the checkpoint directory to the two large OSTs by explicit index:

```bash
lfs setstripe -c 2 -i 0 -o 0,1 "$CKPT"
```

Subdirectories inherit the layout, so the `step-N/` dirs DCP creates are
covered. `optcmp.pbs` does this before every launch.

**Verify it took** -- a `setstripe` that silently failed looks identical until
the next ENOSPC:

```bash
dd if=/dev/zero of=$CKPT/.probe bs=1M count=32
lfs getstripe $CKPT/.probe | tail -4   # obdidx must be 0 and 1 only
rm -f $CKPT/.probe
```

## Notes

* Existing files keep their original layout; `setstripe` only affects files
  created afterward. A directory that already holds checkpoints on a full OST
  needs those files moved, not just a new stripe.
* The checkpoints written BEFORE the failure survived intact, so the recovery
  was a resume, not a restart. Check for `step-N` on disk before assuming the
  work is lost.
* If OST2/OST3 are ever drained or grown, this pin becomes unnecessary but
  stays harmless -- it just concentrates writes on the large OSTs.
