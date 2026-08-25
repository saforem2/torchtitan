# Production dispatch log

> Last updated: 2026-08-25

Every job that targets a **production pre-training chain** -- individual
submissions AND multi-chain umbrellas -- in one place, because the per-chain
READMEs only record the jobs that advanced *their* chain and the umbrella slots
that failed were invisible in all of them. Reconstructed from
`logs/multi-autoretry-*/trainer-*.console.log` and the PBS `.o` logs.

Reading rules:

- **steps** are the first and last step LOGGED. Checkpoints save every 50, so
  a trainer killed mid-interval persists less than it logged (e.g. logged
  7,897 -> saved 7,800). Take the resumable head from disk, not from here.
- The umbrella summary banner reports `failed: N/5` on EXIT CODE, so a trainer
  that trained for hours and was then SIGTERM'd counts as "failed" there. This
  table records what each slot actually did.
- `FAILOVER STOP: walltime` is printed for several unrelated causes; the
  `outcome` column gives the real one.

## Umbrellas (multi-chain, ~2098N)

| Job | Date | Walltime used/req | t0 | t1 | t2 | t3 | t4 | Productive |
|-----|------|------------------|----|----|----|----|----|-----------:|
| `8663177` | 07-17 | 1h21m / 24h (6%) | 2B-512 35,001->39,604 | 20B-512 no-start | 2B-256 10->6,621 | 20B-256 4,351->5,200 | -- | 3/4 |
| `8680578` | 07-21 | 1h13m / 24h (5%) | no-start | no-start | no-start | -- | -- | 0/3 |
| `8698125` | 07-28 | 1h16m / 24h (5%) | **bad_alloc** | **bad_alloc** | 20B-256 6,151->6,897 | -- | -- | 1/3 |
| `8714337` | 07-29 | 0h19m / 1h (smoke) | smoke 2N | smoke 2N | smoke 2N | smoke 2N | smoke 2N | 5/5 (smoke) |
| `8714502` | 08-05 | 8h01m / 24h (33%) | 2B-512 39,601->41,300 | 20B-512 6,801->7,149 | 20B-256 7,501->7,897 | ckpt-path | ckpt-key | 3/5 |
| `8714503` | 08-07 | 1h51m / 24h (8%) | 2B-512 41,301->43,820 | 20B-512 7,251->7,654 | 20B-256 7,801->8,334 | **bad_alloc** | **bad_alloc** | 3/5 |
| `8744245` | 08-09 | 1h41m / 24h (7%) | **bad_alloc** | 20B-512 7,601->8,000+ | 20B-256 8,301->8,324 **bad_alloc** | ImportError | ImportError | 1/5 |
| `8744247` | 08-13 | **23h18m / 24h (97%)** | **2B-512 43,801->46,429 DONE (4.674T, target reached)** | watchdog-kill | 20B-256 8,301->9,850+ | 2B-512clr 9,201->16,984 (node death) | ckpt-key | 3/5 |
| `8756070` | 08-16 | 9h14m / 24h (38%) | **2B-512 STAGE 2 1->3,312** (dolmino CPT, seed step-46429) | 20B-512 8,701->9,109 | 20B-256 9,801->10,381 | `Config.job` AttributeError | `Config.job` AttributeError | 3/5 |
| `8756957` | 08-16 | **12h00m / 12h (100%)** | **2B-512 STAGE 2 3,301->7,731** (dolmino CPT) | 20B-512 9,101->9,695 | CCL KVS timeout at init | 2B-512clr 16,901->21,309 | **ckpt-key (3rd time)** | 3/5 |
| `8760249` | 08-17 | **12h00m / 12h (100%)** | CCL KVS timeout -> **bad_alloc** at init | 20B-512 9,601->10,200 | 20B-256 10,301->11,100 | CCL KVS segfault at init | **ckpt-key, 4th time -- OPTIMIZER namespace** | 2/5 |
| `8764675` | 08-20 | **12h00m31s / 12h (100%)** | CCL KVS timeout -> segfault at init | 20B-512 10,101->10,699 | 20B-256 11,001->11,800 | **bad_alloc** | SophiaG `update_hessian`: `'dict' object has no attribute 'mul_'` | 2/5 |

**Walltime column.** `8744247` and `8756070` are `resources_used.walltime` from
`qstat -xf` (authoritative). The rest predate PBS history retention and are
derived from the first/last `[multi HH:MM:SS]` banner in the `.o` log, so they
are a **lower bound** -- the banner stops when the umbrella script stops, which
can precede the job's own end. `8744247` shows the gap: its log spans 12h30m
against PBS's true 23h18m.

Read the percentage as *allocation actually used*, not as success -- `8714502`
burned 67% of a 24h slot on nothing, and six of nine umbrellas used under 40%.
That is the single largest source of wasted 2,098-node allocation in this
table, and it is almost entirely startup faults and infra kills rather than
training problems.

**Three umbrellas in a row have now used ~all of their allocation** --
`8744247` 97%, then `8756957` and `8760249` and `8764675` at 100%. All four
asked for 12h or less; the 24h dispatches that kept dying young are the
contrast. Treat 12h as the default ask.

What the last two add to the taxonomy: **t0 died at init in both**, so the 2B
stage-2 chain has been stalled since 08-16 while t1/t2 (the two 20B chains) ran
the full window each time. That is where the +910 and +1,431 steps the
dashboard was missing came from. And the t4 seat's failure has moved PAST the
checkpoint-key rename this log records three times -- in `8760249` the shim
fired but `optimizer.state...qkv_linear.wq.weight.step` was still missing, and
in `8764675` the load SUCCEEDED (62s) and then 3,073 ranks died in SophiaG
`update_hessian` on `'dict' object has no attribute 'mul_'`, i.e. the
pre-#3623 nested->flat optimizer format. Converting step-9500 fixed it: smoke
`8772046` loaded the converted seed in 10.42s and trained 8 clean steps.

**`8756957` is the first umbrella to use 100% of its allocation** (12h00m23s of
12h, `Exit_status = -29` = walltime expiry, not a fault), beating `8744247`'s
97%. Both are the two longest-surviving umbrellas and both ran shorter requests
than the 24h dispatches that kept dying young -- the 12h ask appears easier to
schedule and survive than 24h. Its three live seats all ran the full window;
the two dead ones failed at init and never held a slot productively, which is
the remaining waste and is per-seat, not per-job.

Slot map for the 5-trainer umbrellas: t0=2B-512, t1=20B-512, t2=20B-256,
t3=2B-512 constlr-from9200, t4=2B-256 constlr-from9500.

### t4 (2B-256 constlr) has failed on a key rename three times

`8714502`, `8744247`, and `8756957` all lost the t4 seat to `Missing key in
checkpoint state_dict`. It is the same seed each time
(`constlr-from9500/step-9500`, the highest surviving PRE-DECAY 256N checkpoint,
so it cannot simply be reseeded from something newer), and **two different
upstream renames**:

1. the attention qkv_linear wrapper -- fixed by `ckpt_key_compat.py` after
   `8714502`, which is why `8744247` and `8756957` got further, and
2. `output.weight` -> `lm_head.weight` -- diagnosed 2026-08-17 and fixed in
   `e1320edf9`; the shims now compose, since step-9500 predates BOTH.

In `8756957` this seat sat dead for ~9 hours holding 256 nodes. Its log labels
the death exit 127, which reads as the known pals RPC launch failure and is
not: the real error is 3,072 ranks deep in the log.

**Every production 2B/20B checkpoint on disk still spells the head
`output.weight`.** The live chains are unaffected only because their clones are
pinned to pre-rename code. Any clone that pulls to current HEAD needs the shim,
so pull a prod clone deliberately, not incidentally.

Also in `8756957`: t2 (20B-256) died at init with a CCL KVS timeout
(`kvs_get_value: timeout limit: 60 > 60`) after 173s -- the known 6,144-rank
init fault, unrelated to t4. The auto-retry banner labels it
`FAILOVER STOP: walltime`, which is wrong; read the CCL lines above it.

### Failure causes seen in umbrella slots

| Cause | Slots hit | Status |
|-------|-----------|--------|
| `std::bad_alloc` at init | 8698125 t0/t1, 8714503 t3/t4, 8744245 t0/t2 | **OPEN** -- [known-bugs/umbrella-bad-alloc-init.md](../guides/known-bugs/umbrella-bad-alloc-init.md). Falsifiable next test: `qsub -v LAUNCH_STAGGER=180` (~1% of a 24h job). Did NOT recur in 8744247. |
| node death -> pals RPC on relaunch (`stuck_pre_training`, rc=127) | 8744247 t3 | **Infra, not ours -- and auto-retry behaved correctly.** `x4207c3s4b0n0` stopped answering mid-run (`ping failed ... No reply after 97s`) at step 16,984 after 20.5h of clean training. Auto-retry blind-rotated a spare, but the relaunch hit the known transient `Couldn't forward RPC launch ... Resource temporarily unavailable`; two attempts with zero `step=` markers tripped the `stuck_pre_training` guard, which stopped rather than burning the rest of the allocation. Note this is the ONE failure string that names its actual cause instead of the misleading `walltime`. Loss bounded to 84 steps (ckpt head step-16900, 6,144 shards). |
| idle-watchdog kill during a silent ckpt load | 8744247 t1 | **FIXED by config 08-13.** Not a crash -- the 1800s `IDLE_TIMEOUT` SIGTERM'd a HEALTHY 20B-512 30 min into loading 6,144 DCP shards (a load prints nothing while it works). Control in the same job: 20B-256 = 3,072 shards, loaded in 545s, survived. Auto-retry then blind-rotated a node and attempt 2 lost the CCL KVS race. Reported as `FAILOVER STOP: walltime` 45 min into a 24h job. Replacement `8752939` carries `IDLE_TIMEOUT=5400`. |
| ckpt-path (latest resolved to an aborted fragment) | 8714502 t3 | FIXED 08-06 (fragments moved out of the ckpt tree) |
| ckpt-key (pre-refactor flat attention keys) | 8714502 t4, 8744247 t4 | **FIXED 08-13** (`ae880c32d`). The shim shipped but silently never fired: it resolved `<cwd>/<checkpoint.folder>` while the checkpointer PREPENDS `dump_folder`, so it stat'd a path that does not exist, the blanket `except` returned False, and 3,072 ranks died on `Missing key in checkpoint state_dict`. Now takes `dump_folder=` and WARNS on a non-existent dir. Verified against the real step-9500 metadata (False -> True). |
| ImportError (`config_registry`) | 8744245 t3/t4 | **self-inflicted 08-08, reverted** -- main-repo `trainer.py` copied over a pinned clone that runs older code. Do not copy whole files into pinned clones. |
| `AttributeError: 'Config' object has no attribute 'job'` | 8756070 t3/t4 | **FIXED 08-16 in the clone.** Both constlr slots died on every rank at `trainer.py:671`, `dump_folder=config.job.dump_folder`. The `agpt-2b-constlr-from9200` clone carried a DIVERGENT, older form of the ckpt-key shim fix: its Config has no `.job` node, and the other three clones all read the flat `config.dump_folder` at their line 238. Fixed in place (user-approved, backup `trainer.py.bak-20260816-121850`). Auto-retry had already exhausted its attempts, so the two slots stayed dead for that run -- the fix only helps the next submission. NOTE both slots ALSO hit `std::bad_alloc` on attempt 1 and recovered; the `Config.job` error is what actually killed them, and both surfaced as the usual misleading `FAILOVER STOP: walltime`. |
| **PBS `Exit_status = -14` at 9h13m of 24h (whole job)** | 8756070 (all slots) | **UNEXPLAINED -- external kill. Draft ticket: [known-bugs/aurora-job-8756070-exit-14.md](../guides/known-bugs/aurora-job-8756070-exit-14.md) (NOT yet sent).** Not a training fault: all three live trainers logged clean steps at 07:35:16-07:35:49 local with healthy grad norms (0.14-0.51) and normal throughput, then stopped simultaneously. The umbrella's own log ends at `all 5 trainers launched; waiting...` with no stop line, i.e. the script was terminated from outside rather than exiting. Ruled out by direct check: head node `x4305c0s2b0n0` healthy (already `job-exclusive` on another job); allocation fine (1,429,777 node-hours left); `large` queue `enabled/started = True`; no NaN/OOM/CCL error anywhere; the two visible reservations are Mon and Wed. PBS exposes no user-visible record of who issued the kill. **Second anomaly for this same job** -- see the seat-time row below. All three chains checkpointed minutes before (worst-case loss 81 steps). |
| six `Execution server rejected request` seat bounces | 8756070 | **Infra.** `run_count` reached 6 before the job held. Every attempt was assigned the same `x4305c0s*` rack whose lead node `x4305c0s1b0n0` was `offline / EXECJOB_BEGIN: skipped execjob_end found via tmpfs check; reboot required`. **Correcting an earlier reading in this log's own history:** this was NOT our job breaking nodes one per attempt -- only that single node names 8756070, while 13 other `reboot required` nodes at the same time belonged to nine other users' jobs (`8712340`, `8753544`, `8757432`, ...). It is a cluster-wide condition; PBS simply kept picking a set with a sick lead node. Attempt 6 landed on a healthy set and ran. |

## Individual chain jobs (recent)

| Job | Date | Chain | Nodes | Outcome |
|-----|------|-------|------:|---------|
| `8703284` | 07-27 | 20B-256 | 16 | capacity bridge, GAS=16 -> GBS=6144, 6,001->6,037+ |
| `8730438` | 08-03 | 20B-256 | 260 | Q, never seated; superseded by umbrella |
| `8731654` | 08-04 | 20B-512 chain via 256N | 260 | qdel'd -- bridge arrangement rejected in favour of native |
| `8731758` | 08-04 | 20B-512 | 516 | ran before 8744245; carried the chain to 7,251 |
| `8687863` | 07-29 | 20B-512 chain via 256N | 260 | **zombie**: held, and `afterany:8687862` targets a purged job. Needs a manual `qdel`
| `8748010` | 08-11 | 20B-512 | 516 | Q; qdel'd 08-13 when umbrella 8744247 seated (shared ckpt dir) |
| `8748011` | 08-11 | 2B-512 constlr | 516 | Q; qdel'd 08-13 with 8748010, same reason |

### 2026-08-16: five jobs put on HOLD rather than qdel'd

When umbrella `8756070` seated, the five other queued jobs all targeted
checkpoint dirs it owns. They were `qhold`'d, not `qdel`'d -- reversible with
`qrls`, and it preserves queue positions some of which date to Aug 14. After
`8756070` died, `8756957` (12h umbrella) was released; the other four stay held:

| Job | Nodes | Chain | Why still held |
|-----|------:|-------|----------------|
| `8752939` | 522 | 20B-512 | same ckpt dir as umbrella t1 |
| `8752824` | 266 | 2B-256 constlr | same ckpt dir as umbrella t4 |
| `8756071` | 266 | 20B-256 | same ckpt dir as umbrella t2 |
| `8756072` | 522 | 2B-512 constlr | same ckpt dir as umbrella t3 |
| `8687863` | 260 | 20B-512 via 256N | pre-existing zombie, needs manual `qdel` |

Note `8756957` was submitted 08-14 with `IDLE_TIMEOUT=5400` already set, but
PBS snapshots the script at submit time, so it still carries the **old** t0
`TRAIN_TOKENS` (see below). Harmless under constant LR -- it changes only the
stopping point, and no chain approaches it in a 12h window.

### 2026-08-16: t0 stage-2 token budget was 2.96x too large

`8756070` launched t0 with `--training.steps=70176` = 7.06T tokens of pure
dolmino, against an intended 2.39T. Cause: `TRAIN_TOKENS=7064155541716` was
copied from MDS `train_aGPT_2B_sophiag_stage2.sh`, where it is the
**cumulative** budget through stage 2 (4.674T stage-1 + ~2.39T stage-2). But
stage 2 writes to a NEW ckpt dir, so its step counter starts at zero and the
umbrella's `steps = tok/(gbs*seq_len)` has no notion of tokens already spent.

```
7064155541716 / (12288*8192) = 70,176 steps = 7.06T   (what ran)
2390375382006 / (12288*8192) = 23,742 steps = 2.39T   (intended)
```

Not destructive: with `LR_DECAY_STYLE=constant` (decay_ratio 0.0,
min_lr_factor 1.0) the budget sets only the stopping point, not the schedule
shape, so the 3,312 steps that ran are valid stage-2 progress and step-3300 is
resumable as-is. Fixed in `cc4e22cfa`.
| `8752939` | 08-13 | 20B-512 | 522 | **Q.** Replaces umbrella t1, which the idle watchdog killed. Carries `IDLE_TIMEOUT=5400` |
| `8752824` | 08-13 | 2B-256 constlr | 266 | **Q.** Replaces umbrella t4 (ckpt-key bug, fixed `ae880c32d`) |
| `8754664` | 08-14 | eval, 2B-512 final | 1 | **Done.** Steps 41k/43k/45k/46,429 on the modern ladder. MMLU flat at chance (0.2511 at the endpoint); last 500B tokens moved no metric |
| `8756071` | 08-14 | 20B-256 | 266 | **Q.** Continuation from step-9,800 (umbrella 8744247 ended at walltime) |
| `8756072` | 08-14 | 2B-512 constlr | 522 | **Q.** Continuation from step-16,900 (umbrella t3 lost a node at 16,984) |. |

## Evals (capacity queue)

All complete; none queued as of 2026-08-10.

| Job | Scope | Outcome |
|-----|-------|---------|
| `8729921` / `8731413` | modern block backfill, 256n + 512n | complete (14:37 / 08:44 used) |
| `8735716` / `8735717` | 256n 6,900-7,800 / 512n 6,550-7,100 | complete |
| `8736655` | MDS MMLU probe (7.06T dolmino ckpt) | complete -- mmlu 0.2413, arc_c@25 0.3968 |
| `8736838` | MMLU harness validation vs public models | complete -- Llama-3.2-1B 0.3121, Llama-3.1-8B 0.6530 |
| `8744298` / `8744299` | 256n 7,900-8,300 / 512n 7,150-7,600 | complete; first evals fully under shot-namespaced keys |

**Coverage vs disk:** 20B-256 evaluated through 8,300 = its head. 20B-512
evaluated through 7,600 but the chain is at 8,000 and still advancing under
8744245 -- wait for that job to finish, then eval the whole tail at once rather
than chasing a moving head.

## Keeping this current

Nothing auto-generates this yet. After each dispatch ends:

```bash
# per-trainer step ranges + outcome for one umbrella
for f in logs/multi-autoretry-<JOBID>/trainer-*.console.log; do
  n=$(basename "$f" .console.log)
  a=$(grep -aoE "step: [0-9]+" "$f" | head -1 | grep -oE "[0-9]+")
  b=$(grep -aoE "step: [0-9]+" "$f" | tail -1 | grep -oE "[0-9]+")
  echo "$n ${a:--} -> ${b:--}"
done
```

Then check the outcome column against the four known causes above before
inventing a new one -- three of the four print `FAILOVER STOP: walltime`.
