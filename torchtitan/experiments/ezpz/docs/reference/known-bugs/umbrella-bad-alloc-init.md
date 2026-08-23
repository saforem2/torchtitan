# Umbrella `std::bad_alloc` at init -- intermittent, not yet root-caused

> Last updated: 2026-08-10

**Status: OPEN.** Costs whole trainer slots on a ~2098-node allocation. A
plausible mechanism (concurrent-init contention) is identified but NOT proven --
the evidence is inconsistent across runs and is written up here so the next
person does not re-derive it or over-trust the theory.

## What happens

A trainer dies during distributed init with `MemoryError: std::bad_alloc`,
before or shortly after its first step. Two distinct call sites:

**Trainer 0 (2B-512), 8744245** -- host-memory exhaustion in the seed broadcast:

```
torchtitan/distributed/utils.py:177 in set_determinism
  torch.distributed.broadcast(seed_tensor, src=0)
  distributed_c10d.py:3074 in broadcast -> group.broadcast([tensor], opts)
MemoryError: std::bad_alloc
```

This is the same function as the known 1024N init crash
(`memory/project_1024n_init_crash.md`: "crash at set_determinism init at 12,288
ranks").

**Trainer 2 (20B-256), 8744245** -- CCL bootstrap failure that surfaces as the
same error:

```
CCL_ERROR pmi_resizable_simple_internal.cpp:275 pmrt_kvs_get: failed to get val
CCL_ERROR atl_mpi.cpp:916 comm_create: pmrt_kvs_get: error
CCL_ERROR atl_mpi_comm.cpp:94 init_transport: comm_create error
MemoryError: std::bad_alloc
```

Matches the known CCL KVS timeout pattern
(`memory/project_ccl_kvs_timeout_init_crash.md`: "kvs_get_value timeout at 6144
ranks").

## The timing that suggested contention

8744245 launches all five trainers inside 80 seconds, then three die within a
25-second window about a minute later:

```
23:20:25  launch t0        23:22:20  t3 dies
23:20:45  launch t1        23:22:30  t4 dies
23:21:05  launch t2        23:22:45  t0 dies
23:21:25  launch t3
23:21:45  launch t4        23:30:19  t2 first step   (survived bootstrap)
                           23:40:22  t1 first step   (survived bootstrap)
```

`LAUNCH_STAGGER` defaults to 20s and `sum(nproc)` across the five trainers is
**24,576 ranks** -- twice the 12,288 at which the single-job init crash is
documented. Five PMIx/CCL bootstraps of 3,072-6,144 ranks each overlap almost
completely.

## Why that is NOT yet a conclusion

The same script, the same `LAUNCH_STAGGER=20`, and the same 24,576 ranks
produced three different outcomes:

| umbrella | t0 2B-512 | t1 20B-512 | t2 20B-256 | t3 fork | t4 fork |
|----------|-----------|------------|------------|---------|---------|
| 8714502  | trained   | trained    | trained    | other   | other   |
| 8714503  | trained   | trained    | trained    | **BA**  | **BA**  |
| 8744245  | **BA**    | trained    | **BA**     | other   | other   |

("other" = the checkpoint-load failures, separately diagnosed.)

If launch concurrency alone caused it, 8714502 would not be clean. What the
table does show:

- `bad_alloc` is **intermittent** -- same config, different victims each run.
- It has hit **4 of the 5 slots** across three runs, so it is not tied to a
  particular chain, model size, or checkpoint.
- 8714503 hit only the two LAST-launched trainers, which is consistent with
  contention; 8744245 hit the FIRST and THIRD, which is not.

A node-level condition (a host that happens to be short on free memory when a
rank lands on it) would produce exactly this scatter, and auto-retry's blind
node rotation did not save trainer 2 -- attempt 2 failed the same way.

## Cheapest next test

`LAUNCH_STAGGER` is env-overridable (`LAUNCH_STAGGER="${LAUNCH_STAGGER:-20}"`,
line 107), so the contention theory is falsifiable without touching code:

```
qsub -v LAUNCH_STAGGER=180 ... submit_agpt_multi_autoretry.sh
```

180s x 5 trainers costs 15 minutes of a 24h job -- about 1% -- against slots
currently worth ~40% of the allocation. If `bad_alloc` still appears with the
bootstraps fully separated, contention is ruled out and the next suspect is
per-node free memory at rank-placement time (worth capturing `free -g` from the
node list at launch).

## Also worth fixing regardless

Every one of these reports `FAILOVER STOP: walltime` -- trainer 2 stopped 29
minutes into a 24-hour job. That is the third distinct failure mode wearing the
"walltime" label, and it actively misleads triage. See the 8714502 dispatch
report for the same complaint about setup failures.
