# Production dispatch log

> Last updated: 2026-08-10

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

| Job | Date | t0 | t1 | t2 | t3 | t4 | Productive |
|-----|------|----|----|----|----|----|-----------:|
| `8663177` | 07-?? | 2B-512 35,001->39,604 | 20B-512 no-start | 2B-256 10->6,621 | 20B-256 4,351->5,200 | -- | 3/4 |
| `8680578` | 07-?? | no-start | no-start | no-start | -- | -- | 0/3 |
| `8698125` | 08-04 | **bad_alloc** | **bad_alloc** | 20B-256 6,151->6,897 | -- | -- | 1/3 |
| `8714337` | 08-05 | smoke 2N | smoke 2N | smoke 2N | smoke 2N | smoke 2N | 5/5 (smoke) |
| `8714502` | 08-05 | 2B-512 39,601->41,300 | 20B-512 6,801->7,149 | 20B-256 7,501->7,897 | ckpt-path | ckpt-key | 3/5 |
| `8714503` | 08-07 | 2B-512 41,301->43,820 | 20B-512 7,251->7,654 | 20B-256 7,801->8,334 | **bad_alloc** | **bad_alloc** | 3/5 |
| `8744245` | 08-09 | **bad_alloc** | 20B-512 7,601->8,000+ | 20B-256 8,301->8,324 **bad_alloc** | ImportError | ImportError | 1/5 |
| `8744247` | -- | H, `afterany:8744245` | | | | | queued |

Slot map for the 5-trainer umbrellas: t0=2B-512, t1=20B-512, t2=20B-256,
t3=2B-512 constlr-from9200, t4=2B-256 constlr-from9500.

### Failure causes seen in umbrella slots

| Cause | Slots hit | Status |
|-------|-----------|--------|
| `std::bad_alloc` at init | 8698125 t0/t1, 8714503 t3/t4, 8744245 t0/t2 | **OPEN** -- [known-bugs/umbrella-bad-alloc-init.md](../guides/known-bugs/umbrella-bad-alloc-init.md) |
| ckpt-path (latest resolved to an aborted fragment) | 8714502 t3 | FIXED 08-06 (fragments moved out of the ckpt tree) |
| ckpt-key (pre-refactor flat attention keys) | 8714502 t4 | shim written + tested (`d83b4f767`), **delivery to the pinned clone still unsolved** |
| ImportError (`config_registry`) | 8744245 t3/t4 | **self-inflicted 08-08, reverted** -- main-repo `trainer.py` copied over a pinned clone that runs older code. Do not copy whole files into pinned clones. |

## Individual chain jobs (recent)

| Job | Date | Chain | Nodes | Outcome |
|-----|------|-------|------:|---------|
| `8703284` | 07-27 | 20B-256 | 16 | capacity bridge, GAS=16 -> GBS=6144, 6,001->6,037+ |
| `8730438` | 08-03 | 20B-256 | 260 | Q, never seated; superseded by umbrella |
| `8731654` | 08-04 | 20B-512 chain via 256N | 260 | qdel'd -- bridge arrangement rejected in favour of native |
| `8731758` | 08-04 | 20B-512 | 516 | ran before 8744245; carried the chain to 7,251 |
| `8687863` | 07-29 | 20B-512 chain via 256N | 260 | **zombie**: held, and `afterany:8687862` targets a purged job. Needs a manual `qdel`. |

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
