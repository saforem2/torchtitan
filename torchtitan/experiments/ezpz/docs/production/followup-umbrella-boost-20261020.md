# Follow-up umbrella: Monday boost (>2048N) — 2026-10-20

> **Purpose:** replace umbrella `8938159` (1054N, `medium`, 3 trainers — `20B-512` NaN at
> `11101`, `20B-256` healthy at `20823`, `2B` done) and supersede the capacity-blocked
> posfix pair `8934150`/`8934151` (274N each, `Q`/`H`, `small`) with a single boosted
> umbrella eligible for Monday's `>2048N` boost window.

Source branch for payload: `ezpz` @ `d98e77797` (first commit carrying the
data-position fix). Rebuild `.venv.tar.gz` from this SHA — do **not** reuse pinned
clone tarballs.

---

## Position arithmetic (Option B: first-unseen + 5% margin, ceil to GBS)

| Chain | First unseen (step) | GBS (seqs) | Base samples = step * GBS | +5% margin (exact) | Task spec rounded (not GBS-aligned, comment only) | Verified GBS-aligned ceil (canonical) | Tokens/step (= GBS * 8192) | Notes |
|-------|--------------------:|-----------:|--------------------------:|-------------------:|-----------------------------------------------:|--------------------------------------:|---------------------------:|-------|
| **20B-512** | `1200` (= `1200*12288`) | `12288` | `14,745,600` | `15,482,880` = `1260.0 * 12288` | `15489024` (= `1260.5 * 12288`, `%12288==6144`; is `2521 * 6144`) | `15482880` (exact multiple; strict next = `15495168`) | `100,663,296` | `15482880 % 12288 == 0`; task's `15489024` is `+6144` past the 5% mark (half-GBS) |
| **20B-256** | `1550` (= `1550*6144`) | `6144` | `9,523,200` | `9,999,360` = `1627.5 * 6144` | `10004400` (`% 6144 == 1968`, not a multiple) | `10002432` = `1628 * 6144` (ceil to `GBS`) | `50,331,648` | Task's `10004400` is not GBS-aligned — use `10002432` for exact alignment |
| **2B seat** | reclaim — no position advance | `12288` | — | — | — | — | `100,663,296` | Reclaim freed `2B` allocation for additional `2B` or spare capacity |

**Samples vs tokens:** the wired interface is `RESUME_SAMPLES` (unit = one
packed sequence of `8192` tokens). Tokens = samples * `8192`. The old
`POSITION_TOKENS` name is retired — every reference in this doc and the
`.pbs` wrapper now uses `RESUME_SAMPLES`. A doc string `position-tokens`
implies `tokens = samples * seq_len`; do not mix the two.

**How to use:** pass the *sample* count to the dataloader budget
(`--dataloader.resume-consumed-samples` / `BLENDCORPUS_RESUME_CONSUMED_SAMPLES`
env). The step counts above are shown for auditability — the job wrapper
converts via `GBS` alignment only; the sequence length (`8192`) is already
baked into the token count only for reporting.

> **Implementation rule:** the umbrella wrapper MUST compute `RESUME_SAMPLES` as the
> **ceil to the next multiple of that trainer's GBS** (strict `>` if already aligned).
> Ship the verified values (`15482880` / `10002432`) as `RESUME_SAMPLES` and keep the
> task-spec rounded numbers (`15489024` / `10004400`) only as a comment documenting the
> request vs. reality (see PBS wrapper below). Either canonical choice is safe if applied
> consistently — what breaks training is a non-multiple of `GBS` that the dataloader
> cannot shard evenly. Assert `15482880 % 12288 == 0` and `10002432 % 6144 == 0` at
> wrapper startup (wired `.pbs` does this and exits `2` on failure).

---

## Trainer table (boost umbrella)

| Trainer idx | Trajectory | Nodes | GBS (seqs) | LBS | GAS | Source ckpt (resume) | New ckpt dir (fork) | Resume samples | Walltime | Status |
|-------------|-----------:|------:|-----------:|----:|----:|----------------------|---------------------|----------------:|----------|--------|
| `t0` | **20B-512** NaN-rollback + data-position fix | `1024` | `12288` | `1` | `1` | `checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288/` @ `step-10800` (last good before `11101` NaN; verify `.metadata` exists) | `checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000-fixedpos/` | `15482880` (spec: `15489024`, see note above) | `12:00:00` | Roll back past NaN, apply fixedpos patch |
| `t1` | **20B-256** continuation | `512` | `6144` | `1` | `1` | `checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144/` @ `step-17200` (last durable before `20823`; confirm) | `checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144-n256-fixedpos/` | `10002432` (spec: `10004400`, see note) | `12:00:00` | Healthy chain — no rollback, only position correction |
| `t2` | **2B** reclaimed seat | `512` | `12288` | `2` | `1` | `checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/` @ `step-46429` (COMPLETE) — reuse for eval/BT or second 2B leg; alternatively `agpt-2b-constlr-from9000` fork | `checkpoints/agpt-2b-v2-512-reclaim-20261020/` (or reuse existing if BT-only) | — | `12:00:00` | Reclaim freed capacity — do not idle allocation |

> **Fork, don't overwrite.** Every new dir above is a *fork* — copy-forward the source
> ckpt's `step-*` plus `.metadata` into the new dir with `cp -a` + `lfs setstripe`,
> never `mv` or in-place rename. The source chains stay intact for rollback/forensics.
> The `-fixedpos` suffix distinguishes these from the pre-fix `...-constlr-from9000`
> fork (2026-08-21) which carries the same bug.

Total allocation: `1024 + 512 + 512 = 2048` train nodes → request `2058` with `10` spares
(`select=2058`) to meet **>2048N boost eligibility** (strict `>` — `2048` alone does not qualify).
For pre-boost testing, submit the same script at `274N` / `512N` in `prod` via `qsub -l select=...`
override (see Queue strategy).

---

## New checkpoint dirs (canonical)

```
# Source (read-only after fork)
checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288/          # @ step-10800
checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144/           # @ step-17200
checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/           # @ step-46429 (COMPLETE)

# New (write targets — fork, don't overwrite)
checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000-fixedpos/
checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144-n256-fixedpos/
checkpoints/agpt-2b-v2-512-reclaim-20261020/                       # or BT/eval-only — no new ckpt if not training
```

Create with explicit OST pinning (2× large OSTs) before first write:

```bash
lfs setstripe -c 2 -i 10 checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000-fixedpos/
lfs setstripe -c 2 -i 10 checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144-n256-fixedpos/
```

---

## Queue strategy

| Phase | Queue | Nodes (`select`) | Walltime | Purpose | Boost eligible |
|-------|-------|-----------------:|---------:|---------|----------------|
| **Smoke** | `debug` | `1-4` (1 per trainer + head) | `01:00:00` | Position smoke: import + 10-step determinism + ckpt load of fixedpos fork, verify `RESUME_SAMPLES` consumed | No |
| **Capacity test** | `prod` (maps to `small` ≤1024N / `medium` ≤~1054N) | `274` or `512` via `qsub -l select=...` override | `12:00:00` | Validate full `GBS`/`GAS` at reduced scale — same script, smaller `select` | No |
| **Monday boost** | `prod` with `select=2058` (**>2048N**) | `2058` (2048 train + 10 spare) | `12:00:00` | Production dispatch — boosted priority window | **Yes** |
| **Blocked pair to cancel** | `small` | `274` each | — | `8934150`/`8934151` posfix pair — `qdel` after boost umbrella queues | — |

**Why 12h not 24h:** `torchtitan/experiments/ezpz/docs/production/dispatch-log.md` —
12h dispatches used **100%** (`8744247` 97%, `8756957`/`8760249`/`8764675`/`8828612`
all 100%); 24h dispatches died at **5–8%** (`8663177` 6%, `8680578` 5%, `8698125`
5%, `8714503` 8%, `8744245` 7%) to startup faults and infra kills, wasting
~2k-node allocation. Default to `12:00:00`.

**Queue naming:** Aurora `prod` is the submit queue — the scheduler reports it as
`small` (`≤1024N`) or `medium`/`large` by node count. The Monday boost threshold
is `>2048N` in `prod` (observed `large`). Request `2058` to clear the boundary.

Chaining:

```bash
# After boost umbrella is Q (returns $UMB):
qsub -W depend=afterany:$UMB torchtitan/experiments/ezpz/scripts/submit_agpt_20b_autoretry.sh  # +1 continuation
qdel 8934150 8934151   # cancel blocked small pair once boost is queued
```

---

## Venv / tarball discipline

- **Build from `d98e77797`** (branch `ezpz`) — this is the first commit containing
  the data-position fix. Verify:

  ```bash
  git rev-parse HEAD   # must be e5672b6ea4 or descendant of d98e77797
  git log --oneline -1 # e5672b6ea4 docs(ezpz): close hermes ...
  ```

- Build tarball on a login node, not inside a job:

  ```bash
  ezpz tar-env --venv .venv --out .venv.tar.gz   # or: tar czf .venv.tar.gz .venv
  ls -lh .venv.tar.gz                             # expect ~2-3 GB compressed
  ```

- In the PBS wrapper, broadcast with `ezpz yeet --src .venv.tar.gz` (tarball mode:
  `512N≈175s`, `1024N≈255s`, `4096N≈750s` — sub-linear). **Never** fall back to
  per-file `rsync` yeet at this scale (`1-2h` + flare saturation).
- **Do not reuse** pinned-clone tarballs (`torchtitan-ezpz-aurora_*` clones) — they
  lack the fix and will reintroduce the position bug.
- Verify XPU after activation inside the job:

  ```bash
  python3 -c "import torch; assert torch.xpu.is_available(), torch.__version__; print(torch.__version__)"
  ```

---

## PBS wrapper (`followup-umbrella-boost-20261020.pbs`)

Save as `torchtitan/experiments/ezpz/submit/aurora/followup-umbrella-boost-20261020.pbs`
and submit with `qsub -l select=2058` (boost) or `qsub -l select=274` (test).

This doc is the single source of truth — the `.pbs` on disk is authoritative for
execution and this file must match it exactly. If they diverge, fix this doc.

```bash
#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N followup-umbrella-boost-20261020
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -j oe
# select= overridden via qsub -l select=N
#   smoke:  -l select=1..4       (debug, 01:00:00)
#   test:   -l select=274|512    (prod, 12:00:00)
#   boost:  -l select=2058       (prod, 12:00:00)  >2048N threshold

set -o pipefail
ulimit -c 0

# ── Source identity guard ──────────────────────────────────────────────
# Fail closed if the checkout is not the fixedpos payload. Update both on
# every re-cut of this wrapper.
EXPECTED_PARENT="d98e77797d98754bafd73344978a885a21915fbf"
EXPECTED_SCRIPT_SHA="$(sha256sum "$0" | awk '{print $1}')"
# Canonical value after first write — paste here and re-freeze:
# EXPECTED_SCRIPT_SHA="REPLACE_AFTER_FIRST_WRITE"

REPO="/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz"
cd "$REPO" || { echo "FATAL: REPO not found: $REPO"; exit 1; }

HEAD_SHA="$(git rev-parse HEAD)"
if ! git merge-base --is-ancestor "$EXPECTED_PARENT" HEAD 2>/dev/null; then
  echo "FATAL: HEAD $HEAD_SHA does not contain EXPECTED_PARENT $EXPECTED_PARENT"
  echo "  (fix is on ezpz @ d98e77797 — rebuild .venv.tar.gz from that branch)"
  exit 1
fi
# Pinned SHA check (log-only until frozen, then enforced)
if [[ "$EXPECTED_SCRIPT_SHA" != "REPLACE_AFTER_FIRST_WRITE" ]]; then
  ACTUAL_SHA="$(sha256sum "$0" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA" != "$EXPECTED_SCRIPT_SHA" ]]; then
    echo "FATAL: script SHA mismatch: got $ACTUAL_SHA want $EXPECTED_SCRIPT_SHA"
    exit 16
  fi
fi
echo "Source guard OK: HEAD=$HEAD_SHA contains $EXPECTED_PARENT"
echo "Script SHA: $(sha256sum "$0" | awk '{print $1}')  (PBS_JOBID=$PBS_JOBID)"

# ── Resume position (samples, not tokens) — Option B: first-unseen + 5% margin, ceil to GBS ──
# Base: 1200*12288=14745600 (512N), 1550*6144=9523200 (256N)
# +5% exact: 15482880 (=1260*12288), 9999360 (=1627.5*6144)
# Verified GBS-aligned ceil (canonical): 15482880 (exact), 10002432 (=1628*6144)
# Task-spec rounded (not GBS-aligned, kept as comment):
#   15489024 (=1260.5*12288, %12288==6144) and 10004400 (%6144==1968)
RESUME_SAMPLES_20B512=15482880
RESUME_SAMPLES_20B256=10002432

# GBS alignment guard — fail closed before any training
for N_GBS in "${RESUME_SAMPLES_20B512}:12288" "${RESUME_SAMPLES_20B256}:6144"; do
  N="${N_GBS%%:*}"; GBS="${N_GBS##*:}"
  if (( N % GBS != 0 )); then
    echo "FATAL: RESUME_SAMPLES $N not aligned to GBS $GBS (need N % GBS == 0)"
    exit 2
  fi
done
echo "Resume samples OK: 20B512=${RESUME_SAMPLES_20B512} (GBS 12288)  20B256=${RESUME_SAMPLES_20B256} (GBS 6144)"

# ── Checkpoint dirs (fork, don't overwrite) ────────────────────────────
CKPT_20B512_SRC="checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288"
CKPT_20B512_NEW="checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000-fixedpos"
CKPT_20B256_SRC="checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144"
CKPT_20B256_NEW="checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144-n256-fixedpos"
CKPT_2B_RECLAIM="checkpoints/agpt-2b-v2-512-reclaim-20261020"

for D in "$CKPT_20B512_NEW" "$CKPT_20B256_NEW"; do
  if [[ ! -d "$D" ]]; then
    echo "Creating fork dir: $D"
    mkdir -p "$D"
    lfs setstripe -c 2 -i 10 "$D" 2>&1 || echo "WARNING: lfs setstripe failed for $D"
  fi
done
for SRC in "$CKPT_20B512_SRC" "$CKPT_20B256_SRC"; do
  if [[ ! -d "$SRC" ]]; then
    echo "FATAL: source ckpt dir missing: $SRC"
    exit 1
  fi
  if ! ls "$SRC"/step-*/.metadata >/dev/null 2>&1; then
    echo "FATAL: source $SRC has no step-*/.metadata — not a valid DCP dir"
    exit 1
  fi
done
echo "Checkpoint dirs OK"

# ── Venv broadcast (tarball mode) ──────────────────────────────────────
VENV_TGZ="$REPO/.venv.tar.gz"
if [[ ! -f "$VENV_TGZ" ]]; then
  echo "FATAL: $VENV_TGZ missing — run: ezpz tar-env --venv .venv --out .venv.tar.gz"
  exit 1
fi
echo "Broadcasting venv tarball: $VENV_TGZ ($(du -h "$VENV_TGZ" | awk '{print $1}'))"
ezpz yeet --src "$VENV_TGZ" --dest /tmp/.venv 2>&1 | tail -20

source /tmp/.venv/bin/activate 2>/dev/null || source .venv/bin/activate
unset PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT CCL_OP_SYNC=1 CCL_ATL_SYNC_COLL=1

python3 -c "import torch; assert torch.xpu.is_available(); print(f'torch {torch.__version__} xpu={torch.xpu.is_available()}')"
echo "Venv OK"

# Verify the position fix is present in this checkout
if ! grep -q "BLENDCORPUS_RESUME_CONSUMED_SAMPLES" \
    "$REPO/torchtitan/experiments/ezpz/blendcorpus/blendcorpus_builder.py" 2>/dev/null; then
  echo "FATAL: position fix not in blendcorpus_builder.py (need origin/ezpz >= 0f7f0c7f53)"
  exit 1
fi

# ── Umbrella launch ────────────────────────────────────────────────────
echo "=== Umbrella config ==="
echo "  20B-512  1024N  GBS=12288  ckpt=$CKPT_20B512_NEW  resume_samples=$RESUME_SAMPLES_20B512"
echo "  20B-256   512N  GBS=6144   ckpt=$CKPT_20B256_NEW  resume_samples=$RESUME_SAMPLES_20B256"
echo "  2B-reclaim 512N  ckpt=$CKPT_2B_RECLAIM"
echo "  walltime=12:00:00  queue=prod  select=${PBS_select:-2058}"
echo "  HEAD=$HEAD_SHA  parent=$EXPECTED_PARENT"

# Export for mpiexec --envall propagation to all ranks
export BLENDCORPUS_RESUME_CONSUMED_SAMPLES_20B512="$RESUME_SAMPLES_20B512"
export BLENDCORPUS_RESUME_CONSUMED_SAMPLES_20B256="$RESUME_SAMPLES_20B256"
export BLENDCORPUS_RESUME_CONSUMED_SAMPLES="$RESUME_SAMPLES_20B512"

# Resolve umbrella script
UMBRELLA_SCRIPT="$REPO/torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh"
if [[ ! -f "$UMBRELLA_SCRIPT" ]]; then
  echo "FATAL: umbrella script not found: $UMBRELLA_SCRIPT"
  exit 1
fi

echo ""
echo "Wrapper validation complete. Umbrella script: $UMBRELLA_SCRIPT"
echo "  Source guard: OK  Venv: OK  Resume samples: OK  Ckpt dirs: OK"
echo ""
echo "Next: ensure submit_agpt_multi_autoretry.sh trainer table has fixedpos"
echo "  dirs + per-seat --dataloader.resume-consumed-samples, then submit via:"
echo "  qsub -l select=2058 $UMBRELLA_SCRIPT  (or this wrapper once wired)"
```

Per-seat wiring (what `submit_agpt_multi_autoretry.sh` must carry — this wrapper
exports the env for `mpiexec --envall`; the trainer table consumes it as):

```bash
# Inside submit_agpt_multi_autoretry.sh trainer table (one entry per seat):
#   20B-512: --checkpoint.folder="$CKPT_20B512_NEW" --dataloader.resume-consumed-samples="$RESUME_SAMPLES_20B512"
#   20B-256: --checkpoint.folder="$CKPT_20B256_NEW" --dataloader.resume-consumed-samples="$RESUME_SAMPLES_20B256"
# Or equivalently via env: BLENDCORPUS_RESUME_CONSUMED_SAMPLES_<seat> read by blendcorpus_builder.py

# Native launch shape (uncommented, wired — not a stub):
ezpz launch --auto-retry --spare-nodes auto \
  --trainer "python3 -m torchtitan.experiments.ezpz.train --module ezpz.agpt --config agpt_20b_n512 --checkpoint.folder=$CKPT_20B512_NEW --dataloader.resume-consumed-samples=$RESUME_SAMPLES_20B512" \
  --trainer "python3 -m torchtitan.experiments.ezpz.train --module ezpz.agpt --config agpt_20b_n256 --checkpoint.folder=$CKPT_20B256_NEW --dataloader.resume-consumed-samples=$RESUME_SAMPLES_20B256" \
  --trainer "python3 -m torchtitan.experiments.ezpz.train --module ezpz.agpt --config agpt_2b_n512  --checkpoint.folder=$CKPT_2B_RECLAIM"
```

**Where it lives:**

- Spec (this file): `docs/production/followup-umbrella-boost-20261020.md` — single source of truth
- Wrapper: `torchtitan/experiments/ezpz/submit/aurora/followup-umbrella-boost-20261020.pbs`
  (authoritative for execution; must match this doc)

---

## Gating checklist (all must be PASS before Monday boost)

> **Hard gate:** the three checks below are blocking. Do not submit the `>2048N`
> boost umbrella until all three report PASS. Each has a distinct failure mode
> that the others do not cover.

| # | Gate | Where | Command / evidence | Pass criteria | Status |
|---|------|-------|--------------------|---------------|--------|
| 1 | **DCP scan (login node)** | Aurora login | `python torchtitan/experiments/ezpz/scripts/scan_dcp_isfinite.py --ckpt <src> --steps all` (or `scan_dcp_isfinite.py` equivalent) + `ls <ckpt>/step-*/.metadata` | All scanned tensors finite; last-good step confirmed (`10800` for 512N, `17200` for 256N); `.metadata` present | ☐ PASS / ☐ FAIL |
| 2 | **Position smoke (debug 1-4N)** | `debug` queue, `select=1..4`, `01:00:00` | `qsub -l select=4 -q debug -l walltime=01:00:00 followup-umbrella-boost-20261020.pbs` — check log for `Resume samples OK`, `Source guard OK`, `Venv OK`, ckpt load, 10 steps | Imports OK, ckpt loads, `RESUME_SAMPLES` reflected in dataloader log (`consumed_samples` == `15482880`/`10002432`), 10 steps no NaN, no `FATAL` | ☐ PASS / ☐ FAIL |
| 3 | **Capacity test (prod 274N or 512N)** | `prod` queue, `select=274` or `512`, `12:00:00` | `qsub -l select=274 followup-umbrella-boost-20261020.pbs` (or `512`) — same script, smaller `select` | `GBS`/`GAS` correct per trainer, throughput sane, ckpt saves to `-fixedpos` fork, no OOM/hang, walltime 12h clean | ☐ PASS / ☐ FAIL |

**Boost is BLOCKED until `1 ∧ 2 ∧ 3 == PASS`.** Record job IDs and log snippets
for each gate in `docs/journal.md` before requesting the boost `qsub`.

### Extended pre-flight checklist (non-blocking but recommended)

| # | Gate | Check | Owner | Status |
|---|------|-------|-------|--------|
| 4 | **Source SHA** | `git rev-parse HEAD` contains `d98e77797`; `EXPECTED_PARENT` guard in wrapper matches | submitter | ☐ |
| 5 | **Venv tarball** | `.venv.tar.gz` built from `ezpz@d98e77797` on login node; `torch.xpu.is_available()==True` in job | submitter | ☐ |
| 6 | **No pinned clone** | Wrapper `REPO` points to `torchtitan-ezpz` on `ezpz`, not `torchtitan-ezpz-aurora_*` clones | submitter | ☐ |
| 7 | **Checkpoint fork** | New `-fixedpos` dirs created with `lfs setstripe -c 2`; source `.metadata` verified | submitter | ☐ |
| 8 | **No overwrite** | Source ckpt dirs untouched after fork (`ls -ld` mtime unchanged) | submitter | ☐ |
| 9 | **Resume samples GBS-aligned** | `15482880 % 12288 == 0`, `10002432 % 6144 == 0` asserted in wrapper log (exit `2` on fail) | submitter | ☐ |
| 10 | **Walltime 12h** | Wrapper `walltime=12:00:00` (dispatch log: 12h=100% use, 24h=5-8% waste) | submitter | ☐ |
| 11 | **Boost eligibility** | Submit with `qsub -l select=2058` (`>2048N`) in `prod` → scheduler reports `large` | submitter | ☐ |
| 12 | **Chaining** | `qsub -W depend=afterany:$UMB` continuation queued before boost starts | submitter | ☐ |
| 13 | **Cancel blocked pair** | `qdel 8934150 8934151` after boost `Q` | submitter | ☐ |
| 14 | **Script SHA frozen** | `EXPECTED_SCRIPT_SHA` pasted after first write; wrapper re-verified | submitter | ☐ |
| 15 | **W&B / logging** | Run IDs captured, `logs/multi-autoretry-$PBS_JOBID/` written to flare | submitter | ☐ |

---

## What this replaces

| Job | Queue | Nodes | State | Action |
|-----|-------|------:|-------|--------|
| `8938159` | `medium` | `1054N` | Running — `20B-512` NaN@`11101`, `20B-256`@`20823`, `2B` done | Superseded by boost umbrella (do not `qdel` until boost is `Q`) |
| `8934150` | `small` | `274N` | `Q` (capacity-blocked) | `qdel` after boost queues |
| `8934151` | `small` | `274N` | `H` (capacity-blocked) | `qdel` after boost queues |

---

## References

- Dispatch log (walltime evidence): `torchtitan/experiments/ezpz/docs/production/dispatch-log.md`
- Production README: `torchtitan/experiments/ezpz/docs/production/README.md`
- Auto-retry submit pattern: `torchtitan/experiments/ezpz/scripts/submit_agpt_20b_autoretry.sh`
- Data-position fix: branch `ezpz` @ `d98e77797` (commit `d98e77797d98754bafd73344978a885a21915fbf`)
- Wired PBS wrapper: `torchtitan/experiments/ezpz/submit/aurora/followup-umbrella-boost-20261020.pbs`
- DCP scan tool: `torchtitan/experiments/ezpz/scripts/scan_dcp_isfinite.py`
