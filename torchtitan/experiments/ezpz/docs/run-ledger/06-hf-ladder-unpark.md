# 06 — HF Ladder Unpark (E1 Scoring Ladder P0-P3, 14 Converts)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` — covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** E1 unparked — **14-step P0-P3 HF ladder** from DCP frontier `143400/190734 = 75.2%` of `100B` budget; script frontier `136800/190700` per `hf_batch_convert_cpt_fwe.sh:10`; batch `hf_batch_convert_cpt_fwe.sh:1-60` `1N 02:00:00 workq tegu:home` with `flavor=2b-mds bfloat16` per `hf_batch_convert_cpt_fwe.sh:43,58` and `convert_to_hf.py:100-258` `staged_export`; fixups `72800` + `13000` in-job per `hf_batch_convert_cpt_fwe.sh:48-53,58`; HF output `TEGU_ROOT/outputs/evals/cpt-fwe/step-$s/hf` per `hf_batch_convert_cpt_fwe.sh:35-36`.
> Every path, PBS field, step number, and gate threshold below cites its source. Inferences labeled `[INFERENCE]`.

---

## 1. P0-P3 ladder — 14 steps with PBS submit plan

### Ladder definition

Source of truth: `SYNTHESIS.md:198-209` §4 Scoring Ladder and `hf_batch_convert_cpt_fwe.sh:10-16` script header. Enumerated **14 steps** exactly as specified (no invented numbers beyond ladder + frontier):

- `SYNTHESIS.md:204` P0 shock horizon `1000, 5000, 10000`
- `SYNTHESIS.md:205` P1 early plateau `25000, 50000, 75000`
- Assignment ladder + `hf_batch_convert_cpt_fwe.sh:10,58` P2 frontier `100000, 111200, 125000, 136800, 143400` — `136800` is script `seq` upper bound `hf_batch_convert_cpt_fwe.sh:58` `$(seq 74400 200 136800)`; `143400` is live assignment snapshot frontier (`01-cpt-frontier.md:24` `143400/190734 = 75.2%`)
- `SYNTHESIS.md:207` P3 terminal `160000, 175000, 190734` — `190734 = 100B / 524288` per `cpt_fineweb_edu_4n.sh:72-78` and `SYNTHESIS.md:207`

`SYNTHESIS.md:206` lists `150000` as the P2 gate checkpoint; it is **deferred until DCP reaches `150k`** (`thread-a-cpt-trajectory.md:557` `150000 gate when DCP reaches 150k`) and is covered by the P3-deferred chain in §3. The canonical 14 for this doc are the P0-P3 ladder points that have DCP or will have DCP within the current budget — `150000` is submitted as the first P3-deferred job once `step-150000` exists (see §3).

### Full ladder table

| Phase | Steps | Count | PBS select / walltime / filesystems | Qsub command | Verify command |
|-------|-------|------:|--------------------------------------|--------------|----------------|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | `select=1` `walltime=02:00:00` `filesystems=tegu:home` `q=workq` `A=datascience` per `hf_batch_convert_cpt_fwe.sh:2-7` | `qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh` — or P0-only manual loop in §3 | `ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-{1000,5000,10000}/hf/model-*.safetensors && ls /lus/tegu/.../outputs/evals/cpt-fwe/step-{1000,5000,10000}/hf/ezpz_export.json` per `hf_batch_convert_cpt_fwe.sh:58,60` |
| **P1 — Early plateau** | `25000, 50000, 75000` | 3 | same `select=1 02:00:00 tegu:home workq` per `hf_batch_convert_cpt_fwe.sh:2-7` | same `qsub .../hf_batch_convert_cpt_fwe.sh` (covers `75000` via `seq 74400 200 136800` per `hf_batch_convert_cpt_fwe.sh:58`); `25000/50000` via P1 manual loop in §3 | `ls -lh /lus/tegu/.../outputs/evals/cpt-fwe/step-{25000,50000,75000}/hf/model-*.safetensors; ls .../step-{25000,50000,75000}/hf/ezpz_export.json \| wc -l` — expect `3` per `hf_batch_convert_cpt_fwe.sh:60` |
| **P2 — Frontier** | `100000, 111200, 125000, 136800, 143400` | 5 | same `select=1 02:00:00 tegu:home workq` per `hf_batch_convert_cpt_fwe.sh:2-7` | same `qsub .../hf_batch_convert_cpt_fwe.sh` covers `100000-136800` via `seq` per `hf_batch_convert_cpt_fwe.sh:58`; `143400` requires ladder override in §3 (beyond script `136800` upper bound) | `ls -lh /lus/tegu/.../outputs/evals/cpt-fwe/step-{100000,111200,125000,136800,143400}/hf/model-*.safetensors; ls .../step-*/hf/ezpz_export.json \| wc -l` — expect `5` for this phase, `11` cumulative P0-P2 per `SYNTHESIS.md:65` `E1` + `02-hf-conversion.md:143-149` |
| **P3 — Terminal (deferred)** | `160000, 175000, 190734` | 3 | same `select=1 02:00:00 tegu:home workq` per `hf_batch_convert_cpt_fwe.sh:2-7` | deferred `qsub` once DCP reaches each step — same command, gated on `[[ -d $DCP_ROOT/step-$s ]]` per `02-hf-conversion.md:244-247` | `ls -lh /lus/tegu/.../outputs/evals/cpt-fwe/step-{160000,175000,190734}/hf/model-*.safetensors; cat .../step-190734/hf/ezpz_export.json` per `hf_export.py:42-51` manifest |
| **Total** | `1000, 5000, 10000, 25000, 50000, 75000, 100000, 111200, 125000, 136800, 143400, 160000, 175000, 190734` | **14** | `select=1 02:00:00 tegu:home` always — `PBS filesystems=tegu:home` validated per `SYNTHESIS.md:215` `POST-TRAINING-2B.md:13` (Sunspot has no `/eagle`) | see §3 for exact copy-paste `qsub` lines | `ls /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf/ezpz_export.json 2>/dev/tty \| wc -l` per `hf_batch_convert_cpt_fwe.sh:60` — expect `11` P0-P2 now, `14` at `190734` |

PBS header verbatim from `hf_batch_convert_cpt_fwe.sh:1-8`:

```
#!/bin/bash --login
#PBS -N hf-batch-cpt-fwe
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
```

DCP + HF routing verbatim from `hf_batch_convert_cpt_fwe.sh:30-36`:

```bash
REPO=/home/foremans/torchtitan
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64   # hf_batch_convert_cpt_fwe.sh:32
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan  # hf_batch_convert_cpt_fwe.sh:35
# HF_ROOT resolves to Tegu on Sunspot, fallback to $REPO/outputs/evals/cpt-fwe on local dev
if [[ -d "$TEGU_ROOT" ]]; then HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe; else HF_ROOT=$REPO/outputs/evals/cpt-fwe; fi  # hf_batch_convert_cpt_fwe.sh:36
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json  # hf_batch_convert_cpt_fwe.sh:37
TOK_SRC=$REPO/assets/hf/gemma-7b  # hf_batch_convert_cpt_fwe.sh:38
```

Converter contract verbatim from `hf_batch_convert_cpt_fwe.sh:43,58` and `convert_to_hf.py:100-235`:

- `flavor=2b-mds` `dtype=bfloat16` — `hf_batch_convert_cpt_fwe.sh:43` echo `flavor=2b-mds dtype=bfloat16` and `hf_batch_convert_cpt_fwe.sh:58` `--model_flavor 2b-mds --export_dtype bfloat16`
- Wrong flavor = silent gibberish per `POST-TRAINING-2B.md:5-6` and `SYNTHESIS.md:211-212` and `convert_to_hf.py:132-164` RoPE comment (Q/K permute applied vs skipped loads without error, only gibberish at generation)
- DCP root live chain `143400/190734 = 75.2%` per assignment snapshot and `01-cpt-frontier.md:24`; script header frontier `136800/190700` per `hf_batch_convert_cpt_fwe.sh:10`; terminal `190734 = 100B / 524288 (64*8192)` per `cpt_fineweb_edu_4n.sh:72-78` and `01-cpt-frontier.md:29-36`
- DCP storage `556` dirs `23G` at `111200` per `POST-TRAINING-2B.md:5,10` and `01-cpt-frontier.md:55,71`; `~717` dirs at `143400` [INFERENCE: `143400/200`]
- CKPT_DIR symlink fix active per `01-cpt-frontier.md:77-90` and `cpt_fineweb_edu_4n.sh:39-44`:

```bash
if [[ -d "${SUBMIT_DIR}/outputs/outputs/checkpoints" && ! -e "${SUBMIT_DIR}/outputs/checkpoints" ]]; then
  ln -s outputs/checkpoints "${SUBMIT_DIR}/outputs/checkpoints"
fi
```

## 2. Fixups verified

### 72800 — copy `config.json` + `tokenizer.*` from `67400`

State per header `hf_batch_convert_cpt_fwe.sh:10-14`:

```
# Frontier: 136800/190700 (12483869 R 4N). HF 67400 READY, 72800 blocked
# (no tokenizer), 13000 empty, 74400-136800 DCP exists but HF missing.
# Fixups:
#   72800 - copy config.json + tokenizer.* from 67400 HF (already converted)
#   13000 - reconvert DCP -> HF (empty HF dir, included in loop)
```

Fixup block `hf_batch_convert_cpt_fwe.sh:48-51`:

```bash
if [[ -d "$HF_ROOT/step-72800/hf" ]]; then
  cp $HF_ROOT/step-67400/hf/config.json $HF_ROOT/step-72800/hf/ && cp $TOK_SRC/tokenizer.* $HF_ROOT/step-72800/hf/ || echo "[WARN] 72800 fixup copy failed (check src exists)"
fi
```

- Source: `HF 67400 READY` per `hf_batch_convert_cpt_fwe.sh:10` — already `9/9` verified on Tegu per `02-hf-conversion.md:129-135`
- Destination: `$HF_ROOT/step-72800/hf/` — `$HF_ROOT=/lus/tegu/.../outputs/evals/cpt-fwe` on Sunspot per `hf_batch_convert_cpt_fwe.sh:35-36`
- Assets: `HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json` per `hf_batch_convert_cpt_fwe.sh:37` + `TOK_SRC=$REPO/assets/hf/gemma-7b` per `hf_batch_convert_cpt_fwe.sh:38` (canonical `gemma-7b` tokenizer for `agpt-2b-mds`)
- Guard: `[[ -d "$HF_ROOT/step-72800/hf" ]]` — no-op if `72800` HF dir not yet present; retried on next batch invocation after reconvert
- Failure signal: `[WARN] 72800 fixup copy failed` per `hf_batch_convert_cpt_fwe.sh:50` if `67400` HF missing

### 13000 — empty dir reconverted

- State: `13000 empty` per `hf_batch_convert_cpt_fwe.sh:11,15`
- Action: included as explicit `13000` prefix in batch loop `hf_batch_convert_cpt_fwe.sh:58` `for s in 13000 $(seq 74400 200 136800)` — no separate handling needed per `hf_batch_convert_cpt_fwe.sh:53` comment
- Converter invocation per `hf_batch_convert_cpt_fwe.sh:58`:

```bash
$REPO/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds --export_dtype bfloat16 || exit 1
cp "$HF_CONFIG" "$HF/config.json"; cp "$TOK_SRC"/tokenizer.* "$HF/"
```

- VENV: `$REPO/torchtitan/.venv/bin/python` = `/home/foremans/torchtitan/torchtitan/.venv/bin/python` per `hf_batch_convert_cpt_fwe.sh:58` and `02-hf-conversion.md:91` (fixed from `$REPO/.venv` in `5fb2bf2ae7`)

### No manual `.lock` needed — `staged_export` handled in `convert_to_hf.py`

Script header `hf_batch_convert_cpt_fwe.sh:18-19`:

```
# Staging lock pattern from hf_export.py (staged_export) is already handled
# inside convert_to_hf.py -- no manual .lock/.staging needed here.
```

Implementation: `convert_to_hf.py:240-258` wraps `_convert_to_hf` in `hf_export.py:17-39` `staged_export`:

```python
def convert_to_hf(...):
    with staged_export(output_dir) as staging:  # convert_to_hf.py:249
        _convert_to_hf(..., staging, ...)
```

`hf_export.py:17-39` `staged_export` acquires `fcntl.flock(LOCK_EX|LOCK_NB)` on `.<name>.lock` at `hf_export.py:22`, refuses overwrite on `hf_export.py:23-27`, writes to `.<name>.staging-*` temp dir at `hf_export.py:28-32`, validates via `_validated_files` at `hf_export.py:35`, then `os.replace(staging, output_dir)` atomically at `hf_export.py:36`:

```python
@contextmanager
def staged_export(output_dir):  # hf_export.py:17
    output_dir = Path(output_dir).absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with (output_dir.parent / f".{output_dir.name}.lock").open("a") as lock:  # hf_export.py:21
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # hf_export.py:22
        if output_dir.is_symlink() or (output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))):  # hf_export.py:23-26
            raise FileExistsError(f"Refusing to overwrite export: {output_dir}")
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))  # hf_export.py:28-31
        try:
            yield staging
            _validated_files(staging)  # hf_export.py:35
            os.replace(staging, output_dir)  # hf_export.py:36
```

Batch script's only idempotency guard is `[[ -f $HF/ezpz_export.json ]] && continue` per `hf_batch_convert_cpt_fwe.sh:58`. Adding manual `.lock`/`.staging` dirs would conflict with the converter's own staging per `02-hf-conversion.md:197-205`.

## 3. Submit commands — P0 immediate, P1-P2 chained, P3 deferred

### P0 immediate — submit the canonical batch as-is

On Sunspot from repo root (where `PBS_O_WORKDIR` resolves to `~/torchtitan` per `hf_batch_convert_cpt_fwe.sh:27-28`):

```bash
qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh
```

Source: `hf_batch_convert_cpt_fwe.sh:22` usage comment and `02-hf-conversion.md:213-217`. This submits `1N 02:00:00 workq tegu:home A=datascience` per `hf_batch_convert_cpt_fwe.sh:2-7` that loops `13000 $(seq 74400 200 136800)` with `flavor=2b-mds bfloat16` via `$REPO/torchtitan/.venv/bin/python` per `hf_batch_convert_cpt_fwe.sh:58`.

**PBS resource rationale:** `select=1 02:00:00 tegu:home` validated per `SYNTHESIS.md:215` `PBS filesystems=tegu:home` (Sunspot lacks `/eagle` per `POST-TRAINING-2B.md:13`) and `02-hf-conversion.md:25` — `02:00:00` is correct for 1N HF batch (not the CPT chain's `12:00:00` per `01-cpt-frontier.md:25` fix `99e918b342`); `tegu:home` mounts both DCP read (`/home`) and HF write (`/lus/tegu`) per `hf_batch_convert_cpt_fwe.sh:33-36`.

### Idempotence via sentinel `ezpz_export.json`

Loop guard `hf_batch_convert_cpt_fwe.sh:58`:

```bash
[[ -f $HF/ezpz_export.json ]] && continue
```

- Sentinel written by `hf_export.py:42-51` `write_export_manifest` at `convert_to_hf.py:226-235` with `{"complete": true, "files": {...}}` and validated by `hf_export.py:54-74` `_validated_files` / `hf_export.py:77-93` `validate_export` (checks `config.json`, `tokenizer.*`, weight shards).
- Batch summary counts it: `echo CONVERTED $(ls $HF_ROOT/step-*/hf/ezpz_export.json 2>/dev/tty | wc -l) HF` per `hf_batch_convert_cpt_fwe.sh:60`.
- Re-running `qsub .../hf_batch_convert_cpt_fwe.sh` is safe — already-converted steps are skipped. `staged_export` at `hf_export.py:23-27` additionally refuses to overwrite a non-empty `output_dir`.
- Verified `74000/74200 3.7G 9/9` already have `ezpz_export.json` per `02-hf-conversion.md:129-139` and `POST-TRAINING-2B.md:5,19` — they will be skipped.

### Submitting only a P0 subset — no `STEPS` env in canonical script, use manual loop

The canonical `hf_batch_convert_cpt_fwe.sh` does **not** support a `STEPS` env override — its loop is hardcoded `for s in 13000 $(seq 74400 200 136800)` per `hf_batch_convert_cpt_fwe.sh:58`. To submit only P0 (or any subset), use the manual ladder override from `02-hf-conversion.md:224-258` adapted from `thread-a-cpt-trajectory.md:588-621` `§5.3.1`:

**P0-only immediate (copy-paste ready, `bash -n` clean):**

```bash
cat > /tmp/hf_ladder_p0.sh <<'PBS'
#!/bin/bash --login
#PBS -N hf-ladder-p0
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
set -euo pipefail
SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}; cd "$SUBMIT_DIR"
REPO=/home/foremans/torchtitan
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b
echo "HF-LADDER-P0 $(date -u +%FT%TZ) host=$(hostname) DCP=$DCP_ROOT HF=$HF_ROOT flavor=2b-mds"
for s in 1000 5000 10000; do
  DCP=$DCP_ROOT/step-$s; HF=$HF_ROOT/step-$s/hf
  [[ -d "$DCP" ]] || { echo "SKIP $s: no DCP at $DCP"; continue; }
  [[ -f "$HF/ezpz_export.json" ]] && { echo "SKIP $s: HF exists at $HF"; continue; }
  echo "CONVERT $s: $DCP -> $HF"
  /home/foremans/torchtitan/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf \
    "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds \
    --export_dtype bfloat16 || exit 1
  cp "$HF_CONFIG" "$HF/config.json"
  cp "$TOK_SRC"/tokenizer.* "$HF/"
  ls -lh "$HF"/model-*.safetensors
done
echo "CONVERTED $(ls "$HF_ROOT"/step-*/hf/ezpz_export.json 2>/dev/null | wc -l) HF total"
PBS
bash -n /tmp/hf_ladder_p0.sh && qsub /tmp/hf_ladder_p0.sh
```

**P1-P2 chained (submit after P0 passes verification in §4, or chain immediately — idempotence makes ordering safe):**

```bash
cat > /tmp/hf_ladder_p1p2.sh <<'PBS'
#!/bin/bash --login
#PBS -N hf-ladder-p1p2
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
set -euo pipefail
SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}; cd "$SUBMIT_DIR"
REPO=/home/foremans/torchtitan
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b
echo "HF-LADDER-P1P2 $(date -u +%FT%TZ) host=$(hostname) DCP=$DCP_ROOT HF=$HF_ROOT flavor=2b-mds"
for s in 25000 50000 75000 100000 111200 125000 136800 143400; do
  DCP=$DCP_ROOT/step-$s; HF=$HF_ROOT/step-$s/hf
  [[ -d "$DCP" ]] || { echo "SKIP $s: no DCP at $DCP"; continue; }
  [[ -f "$HF/ezpz_export.json" ]] && { echo "SKIP $s: HF exists at $HF"; continue; }
  echo "CONVERT $s: $DCP -> $HF"
  /home/foremans/torchtitan/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf \
    "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds \
    --export_dtype bfloat16 || exit 1
  cp "$HF_CONFIG" "$HF/config.json"
  cp "$TOK_SRC"/tokenizer.* "$HF/"
  ls -lh "$HF"/model-*.safetensors
done
echo "CONVERTED $(ls "$HF_ROOT"/step-*/hf/ezpz_export.json 2>/dev/null | wc -l) HF total"
PBS
bash -n /tmp/hf_ladder_p1p2.sh && qsub /tmp/hf_ladder_p1p2.sh
```

Chain P0 then P1-P2 via `qsub` dependency if strict ordering required:

```bash
P0_JOB=$(qsub /tmp/hf_ladder_p0.sh)
qsub -W depend=afterok:$P0_JOB /tmp/hf_ladder_p1p2.sh
```

Or submit both immediately — sentinel `[[ -f $HF/ezpz_export.json ]] && continue` per `hf_batch_convert_cpt_fwe.sh:58` makes concurrent or out-of-order submission safe; `staged_export` `flock` per `hf_export.py:22` serializes concurrent writers to the same `step-$s`.

**P3 deferred (one `qsub` per gate, each gated on DCP existence):**

```bash
cat > /tmp/hf_ladder_p3.sh <<'PBS'
#!/bin/bash --login
#PBS -N hf-ladder-p3
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
set -euo pipefail
SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}; cd "$SUBMIT_DIR"
REPO=/home/foremans/torchtitan
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b
echo "HF-LADDER-P3 $(date -u +%FT%TZ) host=$(hostname) DCP=$DCP_ROOT HF=$HF_ROOT flavor=2b-mds"
for s in 150000 160000 175000 190734; do
  DCP=$DCP_ROOT/step-$s; HF=$HF_ROOT/step-$s/hf
  [[ -d "$DCP" ]] || { echo "SKIP $s: no DCP yet at $DCP (deferred)"; continue; }
  [[ -f "$HF/ezpz_export.json" ]] && { echo "SKIP $s: HF exists at $HF"; continue; }
  echo "CONVERT $s: $DCP -> $HF"
  /home/foremans/torchtitan/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf \
    "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds \
    --export_dtype bfloat16 || exit 1
  cp "$HF_CONFIG" "$HF/config.json"
  cp "$TOK_SRC"/tokenizer.* "$HF/"
  ls -lh "$HF"/model-*.safetensors
done
echo "CONVERTED $(ls "$HF_ROOT"/step-*/hf/ezpz_export.json 2>/dev/null | wc -l) HF total"
PBS
bash -n /tmp/hf_ladder_p3.sh
# Submit only when DCP reaches each step — do NOT qsub until G1-G4 gate at 150k clears per SYNTHESIS.md:223-248
# qsub /tmp/hf_ladder_p3.sh
```

`190734` is the exact terminal per `cpt_fineweb_edu_4n.sh:77-78` `TRAINING_STEPS = 100B / 524288` and `SYNTHESIS.md:207`. `150000` included in P3 script as the gate checkpoint per `SYNTHESIS.md:206` — submitted once `step-150000` DCP exists.

## 4. Verification — 9/9 files + sentinel count

### Per-step `9/9` file check

Each HF must contain `9/9` files per `SYNTHESIS.md:65` and `02-hf-conversion.md:137` — `model-*.safetensors` shards + `config.json` + `tokenizer.*` + `ezpz_export.json`:

```bash
# Per-step size + file list (expect ~3.7G per step on Tegu per POST-TRAINING-2B.md:5,19)
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf/model-*.safetensors
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-74000/hf/

# Batch sentinel count — one ezpz_export.json per successful HF per hf_batch_convert_cpt_fwe.sh:60
ls /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf/ezpz_export.json 2>/dev/tty | wc -l

# Spot-check one HF — expect 9/9 + non-gibberish config + sentinel
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-74000/hf/
cat /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-74000/hf/ezpz_export.json

# Verify every P0-P2 ladder point has all 9/9 — fail fast on omission (the 74k bug pattern per 02-hf-conversion.md:170-177)
for s in 1000 5000 10000 25000 50000 75000 100000 111200 125000 136800 143400; do
  HF=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-$s/hf
  echo "=== step-$s ==="
  ls -lh "$HF"/model-*.safetensors 2>&1 | head -5
  ls -lh "$HF"/config.json "$HF"/tokenizer.json "$HF"/tokenizer_config.json "$HF"/ezpz_export.json 2>&1
  cat "$HF"/ezpz_export.json 2>&1 | head -20
done
```

Sources: `hf_batch_convert_cpt_fwe.sh:58` per-step `ls -lh "$HF"/model-*.safetensors`, `hf_batch_convert_cpt_fwe.sh:60` batch `CONVERTED ... HF` summary, `02-hf-conversion.md:264-278` verification block, `hf_export.py:77-93` `validate_export` required files.

Expected counts:

| Scope | Expected `wc -l` | Source |
|-------|------------------:|--------|
| Already verified (`74000/74200`) | `2` | `POST-TRAINING-2B.md:5,19` `02-hf-conversion.md:129-135` |
| After P0 (`1000/5000/10000`) | `5` (2 existing + 3 new) | `SYNTHESIS.md:204` P0 |
| After P0+P1 (`+25000/50000/75000`) | `8` | `SYNTHESIS.md:204-205` |
| After P0-P2 (`+100000/111200/125000/136800/143400`) | `13` [INFERENCE: 2 existing + 11 ladder; `74000/74200` not in ladder but present] — ladder `11` + existing `2` | `SYNTHESIS.md:209` total `14` with P3 `3` deferred; P0-P2 ladder `11` per this doc |
| At terminal `190734` | `16` [INFERENCE: 2 existing + 14 ladder] — ladder `14` per this doc | `SYNTHESIS.md:208-209` |

`9/9` composition per `SYNTHESIS.md:65` and `hf_export.py:80-93`:

- `model-*.safetensors` + `model.safetensors.index.json` (sharded) per `hf_export.py:87-93` and `HuggingFaceStorageWriter` at `convert_to_hf.py:191-208`
- `config.json` (copied from `HF_CONFIG=.../agpt_2b_mds_config.json` per `hf_batch_convert_cpt_fwe.sh:37,58`)
- `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json` (copied from `TOK_SRC=.../assets/hf/gemma-7b` per `hf_batch_convert_cpt_fwe.sh:38,58`)
- `ezpz_export.json` sentinel with `{"complete": true, "model_flavor": "2b-mds", "rope": ..., "files": {...}}` per `hf_export.py:42-51` and `convert_to_hf.py:226-235`

### `flavor=2b-mds` hard requirement — wrong flavor is silent gibberish

Converter dispatch `convert_to_hf.py:112-122` `model_registry(model_flavor)` — correct flavor `2b-mds` per `hf_batch_convert_cpt_fwe.sh:43,58` and `SYNTHESIS.md:63,211-212` and `POST-TRAINING-2B.md:5-6`:

> `flavor=2b-mds` — wrong flavor = silent gibberish. `torchtitan/experiments/ezpz/eval/convert_to_hf.py` + `hf_batch_convert_cpt_fwe.sh:58`.

Failure mode per `convert_to_hf.py:132-164` RoPE comment: the checkpoint does not record its RoPE convention (`persistent=False`); the adapter reads it from `--model_flavor`; wrong flavor applies or skips the Q/K permute, which loads without error and only shows up as gibberish at generation time:

```
[convert_to_hf] flavor='2b-mds' -> RoPE=cos_sin (Q/K permute SKIPPED).  # convert_to_hf.py:157-164
The checkpoint does not record its convention -- if this does not
match how it was TRAINED, the export is silently corrupt.
```

Verify flavor in sentinel:

```bash
cat /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf/ezpz_export.json | grep -E 'model_flavor|rope|export_dtype'
# Expect: "model_flavor": "2b-mds", "export_dtype": "bfloat16" per hf_batch_convert_cpt_fwe.sh:58
```

Do not substitute the CPT training VENV `xpu-torch216-sync-5a7a622f16` for the converter VENV — they are distinct per `02-hf-conversion.md:165`: HF convert uses `$REPO/torchtitan/.venv/bin/python` per `hf_batch_convert_cpt_fwe.sh:58`; CPT training uses `/lus/tegu/.../venvs/xpu-torch216-sync-5a7a622f16 + LD-first` per `cpt_fineweb_edu_4n.sh:52-70`.

## 5. Downstream gates consuming this batch

Every HF in this ladder is scored by `SYNTHESIS.md:57-88` `E1-E3` and gated by `SYNTHESIS.md:223-265` `§5 G1-G4`. No CPT-direct GSM8K-CoT — raw CPT scores `0/200` expected (no `<think>` contract per `SYNTHESIS.md:219` `S5.3.2` and `02-hf-conversion.md:109`).

| Ladder phase | Steps | Downstream consumer | Gate | Decision rule | Source |
|--------------|-------|---------------------|------|---------------|--------|
| **P0 shock horizon** | `1000, 5000, 10000` | **E1 shock test** — did constant LR `2e-6` avoid dolmino's `step-1000` shock? Score HellaSwag + ARC-Easy + PIQA on each HF vs base | **G1** HellaSwag `>=0.540` (-2pp from `0.560` base) `SE~0.004`; **G1b** ARC-Easy `>=0.621` (-3pp from `0.651`) per `SYNTHESIS.md:231-232` | If `1000` already `-6pp` (dolmino pattern `cpt/README.md:289-291` `step-1000 ~-6-7pp never recovers`), LR-shock hypothesis falsified — stop CPT before `150k` per `SYNTHESIS.md:242-243` | `SYNTHESIS.md:57 E1`, `SYNTHESIS.md:204`, `SYNTHESIS.md:230-232 G1-G1b`, `01-cpt-frontier.md:288-295` dolmino damage modes |
| **P1 early plateau** | `25000, 50000, 75000` | **E2 FWE vs dolmino** — does `60% FWE` kind-shift hold commonsense where dolmino bled `-7.4pp HS / -10.4pp ARC-Easy` at `step-5960` job `8647850`? Score HS + ARC-Easy + PIQA + ARC-Challenge `25-shot` + MMLU `max(CF,MCF)` on each HF; `50000` was `POST-TRAINING-PLAN.md` phase gate | **G1-G3** all commonsense + MMLU `>=0.240` (no regression from chance band `0.2295-0.2689`) per `SYNTHESIS.md:233-235` | Single `-2pp` wobble = noise, continue per `SYNTHESIS.md:248`; monotonic bleed `P0->P1` = ratio or LR bug — stop CPT even before `150k` per `SYNTHESIS.md:242` and `02-hf-conversion.md:145-148` | `SYNTHESIS.md:68-77 E2`, `SYNTHESIS.md:205`, `SYNTHESIS.md:231-235 G1-G3`, `01-cpt-frontier.md:272-285` dolmino table |
| **P2 frontier** | `100000, 111200, 125000, 136800, 143400` | **E2 health at scale + E3 SFT probe** — frontier holds `23G 556 dirs` at `111200` per `POST-TRAINING-2B.md:5,10`; score every P2 HF on commonsense + MMLU; run **one SFT probe (B2 recipe verbatim)** on best P2 checkpoint (highest P2 that passed `G1-G3`, else `74200` fallback) — `gsm8k-r1cot 93 steps 3ep LR 2e-5 cosine GBS384 seq2048` per `SYNTHESIS.md:78-88` | **G1-G3** on base HF + **G4** SFT probe `GSM8K-CoT acc >=0.265` (B2 `0.205 +0.06` at `2sigma` on `200`, `SE~0.028` per `SYNTHESIS.md:236`), **G4b** `format >=0.90`, **G4c** `gen_len <=550` + `unclosed <=10` per `SYNTHESIS.md:236-238` | `G4 >=0.265` + `G4b-c` hold → promote new base, continue to `190k`; `0.15-0.20` noise around B2 → CPT did not move ceiling, pivot to `E7` broad-distill on MDS base; `<=0.10` → CPT base worse, revert to MDS per `SYNTHESIS.md:245-247` | `SYNTHESIS.md:57-66 E1`, `SYNTHESIS.md:78-88 E3`, `SYNTHESIS.md:230-238 G1-G4c`, `02-hf-conversion.md:149` `143400` assignment snapshot |
| **P3 gate at `150000` + terminal `160000/175000/190734`** | `150000` gate; then `160000, 175000, 190734` | **E9 `150k` gate + E10 terminal composition** — convert `150000` once `step-150000` DCP exists, score `G1-G3`, run SFT probe if `G1-G3` hold; `P3` terminal only if `150k` gate clears | **G1-G4** at `150k` per `SYNTHESIS.md:176-183` `E9`: all `G1-G4` hold → continue to `190k`; single `-2pp` wobble → continue, re-check at `190k`; commonsense `<-3pp` or SFT probe `<=0.10` → **stop CPT**, fork broad-distill from last-good checkpoint per `SYNTHESIS.md:242-246` and `SYNTHESIS.md:176-183` | **Near-miss logic** per `SYNTHESIS.md:240-248`: `G1-G2` fail but `G4` passes → continue (FWE `37.5 vs 35.5` MMLU / `60.1 vs 62.3` HS trade expected per `01-cpt-frontier.md:126-127`); `G1-G2` pass but `G4` fails → stop, fork `E7`; everything fails → stop, revert to MDS | `SYNTHESIS.md:176-183 E9`, `SYNTHESIS.md:185-192 E10`, `SYNTHESIS.md:223-248 §5 gates`, `01-cpt-frontier.md:340-360` |

### Gate definitions (from `SYNTHESIS.md:223-239`)

| Gate | Metric | Base | Threshold to **CONTINUE** to `190k` | Source |
|------|--------|------|--------------------------------------|--------|
| **G1** | HellaSwag `acc_norm` `0-shot` | `0.560` | `>=0.540` (`-2pp`, `5sigma`, `SE~0.004`) | `SYNTHESIS.md:231` |
| **G1b** | ARC-Easy `acc` `0-shot` | `0.651` | `>=0.621` (`-3pp`, `SE~0.006`) | `SYNTHESIS.md:232` |
| **G1c** | PIQA `acc_norm` `0-shot` | `0.733` | `>=0.713` (`-2pp`, `SE~0.005`) | `SYNTHESIS.md:233` |
| **G2** | ARC-Challenge `acc_norm` `25-shot` | `~0.31` | `>=0.290` (`-2pp`) | `SYNTHESIS.md:234` |
| **G3** | MMLU `max(CF,MCF)` `5-shot` | `0.241-0.266` chance band `0.2295-0.2689` | `>=0.240` (no regression; clearing band requires prior collapse) | `SYNTHESIS.md:235` |
| **G4** | GSM8K-CoT `acc` `@200` post-SFT (B2 recipe) | B2 `0.205` (`41/200`, `95% CI 0.149-0.260`) | `>=0.265` (`+0.06` at `2sigma`, `SE~0.028`) | `SYNTHESIS.md:236` |
| **G4b** | `format_hit_rate` | B2 `0.985` | `>=0.90` | `SYNTHESIS.md:237` |
| **G4c** | `mean_gen_len` / `n_unclosed` | B2 `282 / ~0` | `<=550 / <=10` | `SYNTHESIS.md:238` |

Do not gate on MMLU MCF pre-SFT or on raw CPT GSM8K-CoT `0/200` — MCF at chance is correct null at `2B` per `SYNTHESIS.md:21` letter prior `p=(0.37,0.32,0.20,0.11)` and `SYNTHESIS.md:219`; CPT `0/200` degenerate loops expected per `SYNTHESIS.md:219` `02-hf-conversion.md:109`.
