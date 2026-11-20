# 4B/10B v2 512N cannot resume: 6144-rank DCP load stalls past the idle watchdog

> Status: root cause found (2026-11-19), fix in testing. Tracked in #114.
>
> **Root cause: HSDP read amplification, not the plan exchange.** The 4B/10B chains use HSDP (`dp_shard=12`, `dp_replicate=512`); every model/optimizer tensor in the checkpoint has 12 chunks, and `dcp.load` has each of the 512 replicas read its own copy: ~29 TB per 4B load (56.85 GB x 512), ~59 TB per 10B load. The 2B (FSDP over the world) reads each byte once and loads fine. The plan-exchange steps in the original write-up below are superseded. Fix: replica-aware load (only `dp_replicate=0` reads, then broadcasts), on branches `ezpz-dcp-gloo-planning` and `ezpz-replica-aware-load-6ad208b`; see the 2026-11-19 journal entry.
>
> Load-time ladder on Aurora, loading the 4B production `step-20700` with `dp_shard=12`, `dp_replicate=N` (backport on the `6ad208b` tree):
>
> | Nodes | Job | Fix | `dcp.load` + broadcast | Bytes read |
> |---|---|---|---|---|
> | 64 | `9105700` | yes | 147 s | ~57 GB |
> | 64 | `9105991` | no | ~318 s | ~3.6 TB |
> | 128 | `9106167` | yes | 197 s | ~57 GB |
> | 256 | `9106856` | yes | 346 s | ~57 GB |
>
> The 256N run excluded the dataloader state (its `dp_replicate` differs from the checkpoint's) and trained steps 20701-20702 at loss 2.56936 / 2.58342. The CPU regression test is `tests/test_replica_aware_load.py`.

## Problem

The 4B v2 and 10B v2 512N chains cannot resume. Every attempt at 6144 ranks hangs inside the DCP checkpoint load, prints nothing, and is killed by the supervisor's idle watchdog (`IDLE_TIMEOUT=3600`). Neither chain has trained a step since umbrella `9094039` (2026-11-16/17).

| Chain | Checkpoint | Job | Attempts | Outcome |
|---|---|---|---|---|
| 4B v2 512N | `agpt-v2-4b-4k-olmo-mix-n512-20261109/step-20700` | `9091877` (2026-11-18/19, 12h) | 10 | 6x `rc=124` after ~5000 s, 3x `rc=143` after ~2600 s, 10th cut by walltime; no step logged |
| 10B v2 512N | `agpt-v2-10b-4k-olmo-mix-n512-20261111/step-4900` | `9091888` (2026-11-18, ended `-14` after 3h32m) | 5 | `rc=124`/`127`/`143`; no step logged |
| 10B v2 512N | `step-4900` | umbrella `9094039` seat 4 | 3 | each killed mid-load by the watchdog (`rc=143` after 4819 s, 4723 s, ...) |

In `9094039` this looked like contention (two 6144-rank loads plus the 20B-256 save on `flare`; per-rank read rate 1.4 MB/s for 4B, 2.4 MB/s for 10B). The per-chain jobs ran each chain alone and still stalled, so contention is not the whole story.

Last log line of every attempt is the load start, e.g.:

```
[2026-11-19 00:29:45][I][checkpointer/base:327:load] Loading the checkpoint from
  /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/outputs/checkpoints/agpt-v2-4b-4k-olmo-mix-n512-20261109/step-20700.
x4715c4s1b0n0.hsn.cm.aurora.alcf.anl.gov: rank 1882 died from signal 15
```

## Where

- Checkout: `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz` (4B/10B pinned at `6ad208b`), `frameworks/2026.1.0` (torch `2.13.0a0`).
- Logs: `torchtitan-ezpz/logs/multi-autoretry-9091877/trainer-3-v2-4b-n512-attempt-*.console.log`, `multi-autoretry-9091888/trainer-4-v2-10b-n512-attempt-*.console.log`, supervisor output `logs/<job>.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov.OU`.
- Umbrella write-up: `docs/production/2026-11-16-umbrella-9094039.md` ("4B/10B reloads vs `IDLE_TIMEOUT`").

## Reproducer

1. Cheapest first: time the load alone at decreasing scale, no training. On Aurora, `debug-scaling`/`prod`, from the pinned checkout, run the 4B config with `--training.steps` = resume step + 1 against a copy of `step-20700` (do not write into the production checkpoint dir) at 64N, 128N, 256N, 512N. Log wall time from `Loading the checkpoint` to the first step. This gives load time vs ranks.
2. Add progress output to the load so a stall is distinguishable from a slow read: wrap `dcp.load` with a timer thread that prints every 300 s, or set `TORCH_DISTRIBUTED_DEBUG=DETAIL` for one run.
3. During a stalled load, `py-spy dump` a handful of ranks on 2-3 nodes (see the ezpz skill: dump every rank on every node, the split is the evidence) to see whether ranks are in file reads, in a collective (load-plan scatter), or blocked on a few slow ranks.

## Suggested plan

1. **Measure first** (step 1 above). If load time grows superlinearly with ranks, the bottleneck is the plan exchange / metadata, not bytes.
2. **Check the read path:** the 4B checkpoint at 6144 ranks reads at ~1.4 MB/s per rank. Compare against a single-node `dd`/`fio` read of one shard file on the same `flare` path to separate filesystem throughput from DCP overhead. Check Lustre striping on the step dir (`lfs getstripe`); DCP writes one file per rank, and a default stripe count of 1 serializes on few OSTs.
3. **Reduce the plan exchange:** DCP load builds a central plan on the coordinator and scatters it. At 6144 ranks with large `.metadata` (the 20B-512 one is 1.25 GB, see the sibling issue) that is a known scaling hazard. Try `DefaultLoadPlanner` with plan caching if available in this torch, or load with `process_group` set to a gloo group so the plan exchange does not go through oneCCL.
4. **Stop the watchdog from killing a healthy load:** emit a heartbeat line during `dcp.load`, or raise `IDLE_TIMEOUT` for the load phase only, so a slow-but-progressing load is not killed and relaunched into the same wait. This is a mitigation, not a fix.
5. **Fallback:** reshard once at lower scale (load `step-20700` at 128N, save a fresh checkpoint) and resume 512N from that, if the stall is specific to these step dirs.

Done when: 4B resumes from `step-20700` at 512N and logs a training step within the watchdog window, with the load time recorded.
