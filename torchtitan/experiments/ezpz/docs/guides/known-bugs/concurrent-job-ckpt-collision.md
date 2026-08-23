# Concurrent-job checkpoint collision on `20b_v2_256`

Investigated 2026-08-16. Read-only audit of the AuroraGPT production
checkpoint directories. Every claim below is labelled MEASURED (observed
directly), INFERRED (deduced from measured facts), or UNKNOWN.

Clock note, load-bearing for the whole analysis: filesystem mtimes on
`/lus/flare` are **UTC** (MEASURED: `date +%Z%z` -> `UTC+0000`), while the
timestamps printed inside the `.o` / `.console.log` files are **CDT
(UTC-5)** (INFERRED from job submit times vs mtimes; consistent to the
minute across 20 independent save events). All times in this document are
normalized to **UTC** unless a log line is quoted verbatim.

## Verdict

**Qualified -- no production checkpoint that the chain actually uses is
corrupt, and no resume ever loaded a losing run's weights.** The surviving
`20b_v2_256` lineage is self-consistent end to end: every resume loaded a
checkpoint written by the run that legitimately preceded it (MEASURED, 16
resume events traced). Two directories -- `step-6200-20260729-142222` and
`step-6300-20260729-142222` -- **are** genuinely mixed, each physically
holding shards from two different jobs at two different rank counts
(MEASURED), and the 256-node version of those two checkpoints is
permanently destroyed (MEASURED: its shards 0-191 were overwritten). But
both were renamed with a timestamp suffix, which makes them invisible to
`step-N` resume lookup, and no log on the filesystem ever references them
(MEASURED). The cost is ~560 GB of dead bytes and two lost (superseded)
checkpoints, not a lineage break. Across all six tracked chains there are
**2 collision episodes, both on `20b_v2_256`**; the other five chains are
clean (MEASURED).

The one thing that deserves to be called out plainly: this was luck, not
design. Nothing in the pipeline detected the collision, refused the second
writer, or flagged the mixed directories. The rename that quarantined them
has no identified source in the repo (UNKNOWN).

## What happened

Two separate episodes in which two jobs held the same chain's checkpoint
directory at the same time.

### Episode 1 -- 2026-07-10, same rank count (benign)

| | run | steps | wall window (UTC) |
|---|---|---|---|
| A | W&B `6yr6ivh4`, job 8647385 | 3101..3603 | 07-10 12:03 -> 18:02 |
| B | `logs/multi-autoretry-8648363/trainer-3-20b-n256.console.log` | 3401..4297 | 07-10 16:02 -> 07-11 03:50 |

MEASURED. Step overlap 3401..3603; wall overlap ~2h. B loaded `step-3400`
(MEASURED), which A had written at 15:38 -- so B is a legitimate
continuation forked off A's last uncontested checkpoint.

Both were 256-node jobs writing 3072 shards, so their writes were
same-shaped: the later writer replaced the earlier one file-for-file. Save
times reconstruct cleanly (MEASURED, from `Finished saving` lines +5h):

- A saved 3200@13:17, 3300@14:29, 3400@15:38, 3500@16:48, 3600@17:58
- B saved 3500@17:23, 3600@18:41, 3700@20:08 ... 4200@02:43

Disk shows `step-3500` mtime 17:22 and `step-3600` mtime 18:40 -> **B won
both** (MEASURED). A's 3500/3600 were overwritten and are gone. Nothing
ever needed them: the next resume in the chain loaded `step-4200`, which is
B's (MEASURED). This is why the loss curves disagree on 3401..3602 with the
sign flipping -- two real trajectories were running -- but the disk keeps
only one of them, consistently.

INFERRED: episode 1 left **no** mixed directory. Every dir in 3300..4200 is
3072 shards with a single mtime cluster (MEASURED).

### Episode 2 -- 2026-07-27..29, different rank counts (the damaging one)

| | run | nodes/ranks | steps | wall window (UTC) |
|---|---|---|---|---|
| C | W&B `v58n7vam`, job 8703284 (`cap_20b256_16n.sh`) | **16N / 192 ranks** | 6001..6302 | 07-27 12:00 -> 07-29 12:59 |
| D | W&B `cxlt0tpe`, job 8698125 trainer-2 | **256N / 3072 ranks** | 6151..6897 | 07-28 16:22 -> 07-29 09:09 |

MEASURED (`NHOSTS=16` in C's log; D's launcher line reads
`Using only [3072/15720] available GPUs`). Step overlap 6151..6302; wall
overlap 16.6h.

C is the "capacity-queue bridge" -- a deliberate small-scale job using
GAS=16 to reproduce GBS=6144 while the 256N allocation was queue-starved.
D loaded `step-6150`, which C wrote (MEASURED). So D is a legitimate
continuation of C. The problem is that **C did not stop.** It kept running
and kept saving every 25 steps into the same directory that D was saving
into every 100 steps.

Because the two jobs had different rank counts they wrote **different
numbers of shards** -- 192 vs 3072 -- so their writes were *not*
same-shaped, and a later writer could not cleanly replace an earlier one.
That is what produced the mixed directories.

## Per-checkpoint provenance

MEASURED (mtime, shard count, internal mtime clustering). Owner assignment
is INFERRED by matching mtime against each run's wall window and save
cadence; confidence noted. `!` marks a step whose mtime regresses relative
to a lower-numbered step -- direct evidence of interleaving.

| step | mtime (UTC) | shards | owning run | conf |
|---|---|---|---|---|
| 3300 | 07-10 14:28 | 3072 | `6yr6ivh4` (A) | high |
| 3400 | 07-10 15:37 | 3072 | `6yr6ivh4` (A) -- last A-owned | high |
| 3500 | 07-10 17:22 | 3072 | 8648363-t3 (B), overwrote A | high |
| 3600 | 07-10 18:40 | 3072 | 8648363-t3 (B), overwrote A | high |
| 3700..4200 | 07-10 20:07 .. 07-11 02:42 | 3072 | 8648363-t3 (B) | high |
| 4225..4350 | 07-13 03:27 .. 05:18 | 3072 | `uvgmafv9` (8661054) | high |
| 4375 | 07-13 05:34 | **0** | interrupted save, empty | high |
| 4400..5100 | 07-17 14:22 .. 23:04 | 3072 | 8663177-t3 | high |
| 5200..6000 | 07-24 08:34 .. 18:11 | 3072 | `5rvusq43` (8681340) | high |
| 6025 | 07-27 16:33 | **192** | `v58n7vam` (C, 16N) | high |
| 6050..6175 | 07-27 20:34 .. 07-28 16:47 | **192** | `v58n7vam` (C, 16N) | high |
| **6200**-`20260729-142222` | 07-28 20:38 | 3072 **MIXED** | C manifest + D orphan bytes | high |
| 6225 | 07-29 00:41 | **192** | `v58n7vam` (C) -- orphan tail | high |
| 6250 | 07-29 04:36 | **192** | `v58n7vam` (C) -- orphan tail | high |
| 6275 | 07-29 08:34 | **192** | `v58n7vam` (C) -- orphan tail | high |
| **6300**-`20260729-142222` `!` | 07-29 12:34 | 3072 **MIXED** | C manifest + D orphan bytes | high |
| 6400 `!` | 07-28 21:49 | 3072 | `cxlt0tpe` (D) | high |
| 6500 `!` | 07-28 23:52 | 3072 | `cxlt0tpe` (D) | high |
| 6600 `!` | 07-29 02:15 | 3072 | `cxlt0tpe` (D) | high |
| 6700 `!` | 07-29 04:34 | 3072 | `cxlt0tpe` (D) | high |
| 6800 `!` | 07-29 07:01 | 3072 | `cxlt0tpe` (D) -- chain resumes here | high |
| 6900..7500 | 08-03 06:04 .. 14:52 | 3072 | 8698754 / later 256N | high |
| 7600-`20260804-015003` | 08-03 16:10 | **0** | interrupted save, empty | high |
| 7600..10300 | 08-05 05:54 onward | 3072 | later 256N | high |

**Out-of-order steps: 6** (MEASURED) -- 6300, 6400, 6500, 6600, 6700, 6800.
All six sit inside episode 2. Their mtimes are *earlier* than the
lower-numbered 6225/6250/6275, because D was writing its 100-step ladder
while C was still writing its 25-step ladder. Outside 6200..6800 the
directory is strictly monotonic across all 118 step dirs from 200 to 10300
(MEASURED).

Note the 192-shard dirs are **not** defective. A 192-rank job writing 192
shards is a well-formed DCP checkpoint; DCP reshards on load. Their shards
are ~1.31 GB each vs ~98 MB for a 3072-shard checkpoint (MEASURED), which
is exactly the 16x ratio you expect.

## Did any resume load the wrong lineage?

**No.** MEASURED -- every resume in the chain, from all umbrella trainer
logs and all standalone `.o` logs that reference this checkpoint folder:

| job | loaded |
|---|---|
| 8558548 | `step-1100` |
| 8558549 / multi-8568429 | `step-2100` |
| 8647385 (`6yr6ivh4`) | `step-3100` |
| **8648363-t3** | **`step-3400`** (A's last uncontested ckpt) |
| 8661054 (`uvgmafv9`) | `step-4200` |
| 8663177-t3 | `step-4350` |
| 8681340 (`5rvusq43`) | `step-5100` |
| **8703284 (`v58n7vam`, 16N)** | **`step-6000`** |
| **8698125-t2 (`cxlt0tpe`, 256N)** | **`step-6150`** (C's 192-shard ckpt) |
| 8698753 / 8698754 | `step-6800` |
| 8714502-t2 | `step-7500` |
| 8714503-t2 | `step-7800` |
| 8730438 | `step-7800`, `step-7900` |
| 8744245-t2 / 8744247-t2 | `step-8300` |
| 8756070-t2 | `step-9800` |

Two loads cross a lineage boundary, and both are **intended handoffs, not
accidents**:

- `8648363-t3` loading `step-3400` -- A's work, adopted by B. Correct.
- `cxlt0tpe` loading `step-6150` -- C's 192-shard bridge checkpoint,
  adopted by the 256N job. Correct, and it worked: the resharded load took
  252.6s, emitted no warnings between load-start and load-end (MEASURED),
  and loss continued smoothly (6151: 2.37689, 6152: 2.35730, ... no spike;
  MEASURED).

Critically, **no resume ever loaded any of C's orphan tail** (6175, 6200,
6225, 6250, 6275, 6300). The chain jumps from `step-6150` straight to
`step-6800`, both times through D's lineage. The orphan tail is dead
weight, not contamination (MEASURED).

INFERRED: the surviving lineage is
`...6000 (5rvusq43, 256N) -> 6001..6150 (C, 16N bridge) -> 6151..6897 (D,
256N) -> 6800 -> onward`, unbroken.

UNKNOWN: whether C's 16N/GAS=16 configuration is *bit-identical* to a 256N
step, as `trajectories.py` asserts. The loss continuity above is consistent
with it but does not prove it. Settling it would need a controlled
same-seed 16N-vs-256N comparison with `--debug.deterministic`.

## Are the DCPs internally consistent?

Mostly yes; two are genuinely mixed. MEASURED for every step dir in
3300..7100 (54 dirs): shard count, min/max internal mtime, largest internal
mtime gap, and size of the trailing mtime cluster.

- **43 dirs** hold exactly 3072 `.distcp` + `.metadata`, single mtime
  cluster. Clean.
- **10 dirs** hold exactly 192 `.distcp` + `.metadata`, single cluster.
  Clean (these are C's, correctly shaped for 192 ranks).
- **2 dirs are MIXED** -- see below.
- **2 dirs are empty** (`step-4375`, `step-7600-20260804-015003`): 0 files,
  no `.metadata`. Interrupted saves; harmless, and `.metadata`-absence is
  already the skip condition used by the eval sweeps.

Three dirs tripped the "internal mtime gap" detector but are benign on
inspection (MEASURED): `step-6250` (gap 121s -- one straggler shard
`__39_0.distcp` plus `.metadata`), `step-6700` (108s -- 3071 files in one
minute, 1 straggler), `step-7100` (111s, same shape). Straggler ranks, not
a second writer.

### The two mixed directories

`step-6200-20260729-142222` and `step-6300-20260729-142222`. Each contains
3072 `.distcp` files split into two mtime clusters, and the split is
**exactly on shard index** (MEASURED):

| dir | shards 0-191 | shards 192-3071 |
|---|---|---|
| `step-6200-*` | written 07-28 20:37, 252,561,448 B each | written 07-28 17:34, 97,997,402 B each |
| `step-6300-*` | written 07-29 12:33 | written 07-28 19:46 |

The file *sizes* settle it beyond doubt: in the same directory,
`__191_0.distcp` is 252 MB (16N shard class) while `__192_0.distcp` is 98 MB
(256N shard class) -- for comparison, in the healthy `step-6400` both are
97,997,402 B (MEASURED). Two jobs' shards are physically coexisting.

The decisive test is what the manifest points at. Both mixed dirs carry a
`.metadata` of **59,044,015 bytes** -- the 192-shard manifest size, versus
698,545,254 for a 3072-shard one (MEASURED). Probing it directly: both
reference `__191_0.distcp` but **not** `__192_0.distcp`, `__1000_0.distcp`,
or `__3071_0.distcp`, whereas healthy `step-6400`/`step-6800` reference all
four (MEASURED).

INFERRED, and this is the key structural finding:

1. The 16N job wrote **last**, so its `.metadata` and its shards 0-191 won.
   Each mixed dir is therefore a **coherent 192-shard checkpoint** -- its
   manifest is internally consistent and every shard it names exists.
2. The 2880 files at indices 192-3071 are **orphans**: real bytes, ~280 GB
   per dir, referenced by nothing. `du` confirms 453 GB per mixed dir vs
   233-239 GB for a normal one (MEASURED).
3. The 256N job's own shards 0-191 were **overwritten** by the 16N job.
   Its version of `step-6200` / `step-6300` is therefore **unrecoverable**
   -- the orphan 2880 can never be completed. That is real, permanent data
   loss (MEASURED), of two checkpoints that were immediately superseded by
   `step-6400`..`step-6800` and were never needed.

So: no directory contains a *silently wrong* mixture that would load and
produce garbage. The failure mode is loud (missing shards -> load error),
not silent. And it cannot even be reached, because both dirs were renamed
with a timestamp suffix and are consequently invisible to `step-N` lookup;
no file anywhere under `logs/` or the 256N clone references either name
(MEASURED).

UNKNOWN: **what performed the rename.** The suffix `20260729-142222`
corresponds to 2026-07-29 14:22 local, after both jobs died. No
`mv`/`rename` of a `step-*` directory exists anywhere in
`torchtitan/experiments/ezpz/` (MEASURED, grepped). It may have been a
manual operator action. Evidence that would settle it: shell history for
that session, or a PBS epilogue/failover script outside the repo.

## Scope: other chains

MEASURED. Two independent detectors run over every on-disk chain:
(a) W&B collision = two runs of the same chain whose step ranges **and**
wall-clock windows both overlap; (b) disk collision = out-of-order mtimes
or multi-cluster step dirs.

| chain | dirs scanned | W&B collisions | out-of-order | mixed dirs |
|---|---|---|---|---|
| `2b_v1_256` | n/a (archived) | 0 | - | - |
| `20b_v1_256` | n/a (archived) | 0 | - | - |
| `2b_v2_256` | 407 | 0 | 0 | 0 |
| `2b_v2_512` | 467 | 0 | 0 | 0 |
| `20b_v2_512` | 113 | 0 | 0 | 0 |
| **`20b_v2_256`** | 118 | **1** | **6** | **2** |
| `2b_v2_512_lr3.22e-5` | n/a (W&B only) | 0 | - | - |

**Re-verified 2026-08-17** with the independent `utils/audit_ckpt_dirs.py`
detector. `20b_v2_512` re-scanned (116 dirs): **0 mixed, 0 out-of-order**, and
exactly the four documented shard counts -- 6144 (93 dirs), 3072 (9), 1536
(10), 192 (2). `20b_v2_256` re-scanned (120 dirs): both mixed dirs found, and
all three benign straggler dirs correctly NOT flagged. So the new tool
reproduces this table's result on a known-clean chain and on the known-bad one,
which is what makes it trustworthy going forward.

**Total: 2 collision episodes, both on `20b_v2_256`.** The W&B detector
finds only episode 2 (`v58n7vam` x `cxlt0tpe`, 150 steps / 16.6h overlap);
episode 1 is invisible to it because the second participant
(job 8648363 trainer-3) has no usable W&B history and survives only as an
`.o`-style console log. INFERRED: **a W&B-only collision scan
under-reports.** Any future check must union W&B ranges with `.o`-log
ranges, exactly as `concat_chain` already does for plotting.

Three benign findings worth recording so they are not mistaken for
collisions later:

- `2b_v2_256` has 3 dirs with a small trailing mtime cluster (`step-36500`,
  `step-50900`, `step-61500`; gaps 13-32s, 3-4 files). Straggler ranks. All
  407 dirs are 3072 shards, strictly monotonic (MEASURED).
- `20b_v2_512` legitimately contains **four** shard counts -- 6144 (512N),
  3072 (256N), 1536 (128N), 192 (16N) (MEASURED). Varying shard count alone
  is *not* evidence of a collision; it is evidence of re-scaled resumes,
  which DCP supports. That chain has zero out-of-order steps and zero mixed
  dirs, i.e. its re-scales were properly serialized -- the same practice as
  episode 2, done safely.
- Empty `step-*` dirs and `.bak-*` dirs appear on several chains; they are
  interrupted saves and already skipped by `.metadata` checks.

## How this happened

The operational cause is simple and is exactly what job holds exist to
prevent: **two jobs for one chain were seated at the same time, both
pointing `--checkpoint.folder` at the same directory.**

- Episode 1: a standalone continuation (8647385) was still running when an
  umbrella job (8648363) seated a trainer for the same chain.
- Episode 2: a small-scale "capacity bridge" (8703284, 16N) was launched to
  make progress while the 256N allocation was queue-starved -- and was
  **not cancelled or allowed to expire** when the 256N job (8698125) finally
  seated 28h later. It then ran alongside it for 16.6h.

Contributing factors, all MEASURED:

1. Nothing in the training entrypoint claims the checkpoint directory.
   There is no lockfile, no PID/jobid stamp, no "another job is writing
   here" refusal. Two writers is an unrepresented state.
2. `keep-latest-k=0` (correct, and deliberately so) means nothing prunes,
   so a loser's orphan checkpoints persist indefinitely and silently
   consume quota.
3. Episode 2 was worse than episode 1 purely because the rank counts
   differed. Same-shape concurrent writes degrade gracefully to
   last-writer-wins; different-shape ones leave a partially-overwritten
   directory that can never be reassembled.
4. The bridge job's whole purpose -- run the same chain at a different
   scale -- is precisely the pattern that makes concurrency destructive
   rather than merely wasteful.

## What to do about it

### Must

1. **Do not delete the two mixed dirs blind.** (Still pending a decision;
   re-verified on disk 2026-08-17 -- both present, 453 GB and 3,072 files
   each.) They are 906 GB combined and
   ~560 GB of that is orphan bytes, but each still holds a *coherent
   192-shard checkpoint* for steps 6200/6300. If those steps have any
   archival value, extract the 192 referenced shards plus `.metadata`
   first; then the 2880 orphans can go. If they have no value, both dirs
   can be dropped wholesale. Either way this is a reclaim decision, not a
   correctness one. (Per project convention use `backup`, not `rm`.)
2. ~~**Add a checkpoint-directory claim.**~~ **DONE 2026-08-17** (`bae88a1f6`,
   `ezpz/ckpt_owner_claim.py`). Rank 0 writes a `.owner` file (jobid, rank
   count, start time) before the checkpointer loads, and reports loudly when a
   different jobid already holds one.

   It **warns rather than refuses**, which is a deliberate departure from the
   "refuse" option written above. A job killed by walltime, a node fault, or a
   PBS `-14` leaves its claim behind, and a starting job cannot tell a stale
   claim from a live one from the inside -- PBS state is not visible there and
   jobids get recycled. Refusing would convert every crashed predecessor into a
   failed resume, which is a worse and far more frequent failure than the
   collision. Claims older than 36h are labelled likely-stale so the warning
   keeps its signal. Verified by 16 assertions in
   `tests/test_ckpt_owner_claim.py` (stdlib-only; runs without torch).
3. ~~**Assert shard count on resume.**~~ **DONE 2026-08-17** as part of the
   audit tool below, which reports any dir whose shard count differs from the
   tree's dominant one. On this tree it correctly flags the 10 dirs at 192
   shards against 107 at 3072. (Still worth adding to the resume path itself
   so it fires in-job, not only on demand.)
4. **Fix the collision detector to union W&B with `.o` logs.** A
   W&B-only scan misses episode 1 entirely. Whatever check gets
   institutionalized must read both sources, since 17 of this chain's 20
   runs ended `crashed` and crashed runs do not sync their tails.

### Optional

5. ~~Add a periodic audit...~~ **DONE 2026-08-17**:
   `utils/audit_ckpt_dirs.py` (`python3 -m ...audit_ckpt_dirs <dir>` or
   `--all-chains`; exits 1 on any MIXED dir so it can gate a resume).

   **The mtime gap alone is not the signal** -- that was the design lesson.
   Three dirs in this very tree (`step-6250`, `step-6700`, `step-7100`) gap
   108-121s from straggler ranks and are benign, so a naive gap check raises
   five alarms of which three are false. The discriminator is **index
   alignment**: a collision's split falls exactly on a shard index, because one
   job overwrote a contiguous prefix. Stragglers scatter. The tool only says
   MIXED when the split is index-aligned, and labels the rest as stragglers.

   VALIDATED against this tree: it finds both known-mixed dirs
   (`step-6200-*`, `step-6300-*`) and stays silent on all three straggler
   dirs.

   Caveat on the size column: shard sizes vary by more than an order of
   magnitude *within* a single writer's cluster (192-rank writer: 1.31 GB at
   shard 0, 252 MB at shard 191). The tool reports a median purely to make the
   two writers visually distinct -- the mtime split and index alignment are the
   evidence, size is a hint.
6. When a capacity-bridge job is launched for a chain that already has a
   queued full-scale job, make one of them a dependency of the other
   (`-W depend=afterany:`) so they cannot be seated concurrently.
7. Record the rename convention. Something quarantined these two dirs by
   appending a timestamp, which is genuinely the right behaviour -- but it
   is not in the repo, so it cannot be relied on and may not happen next
   time. Either implement it deliberately or stop depending on it.
8. Consider stamping the writing jobid into each saved step dir. Provenance
   here had to be reconstructed from mtimes and CDT/UTC arithmetic across
   20 save events; a one-line marker file would have made it a `cat`.
