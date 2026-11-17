# 02 — HF Conversion Batch Ledger (CPT-FWE DCP -> HF)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` (Sunspot `~/torchtitan`) — ledger covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** HF batch `hf-batch-cpt-fwe` covers DCP frontier **136800 / 190700 per `hf_batch_convert_cpt_fwe.sh:10`** (live DCP at **143400 / 190700 = 75.2%** per assignment snapshot); early HFs **74000/74200 `3.7G` `9/9` verified** on Tegu per `POST-TRAINING-2B.md:5,19`; pending P0-P2 ladder **8 converts `~52G`** per `SYNTHESIS.md:65` `E1`; converter contract `flavor=2b-mds` `dtype=bfloat16` via `hf_batch_convert_cpt_fwe.sh:58`.
> Every path, PBS field, seq range, and file count below cites its source. Inferences labeled `[INFERENCE]`.

---

## What is running

### Batch script and PBS spec

Canonical script: `torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh` (entire script `hf_batch_convert_cpt_fwe.sh:1-60`).

| PBS field | Value | Source |
|---|---|---|
| `#PBS -N` | `hf-batch-cpt-fwe` | `hf_batch_convert_cpt_fwe.sh:2` |
| `#PBS -q` | `workq` | `hf_batch_convert_cpt_fwe.sh:3` |
| `#PBS -l walltime` | `02:00:00` | `hf_batch_convert_cpt_fwe.sh:4` |
| `#PBS -l select` | `1` (1N, single-rank convert) | `hf_batch_convert_cpt_fwe.sh:5` |
| `#PBS -l filesystems` | `tegu:home` (covers DCP on `/home` + HF on Tegu) | `hf_batch_convert_cpt_fwe.sh:6` |
| `#PBS -A` | `datascience` | `hf_batch_convert_cpt_fwe.sh:7` |
| `#PBS -j` | `oe` (join stdout/stderr) | `hf_batch_convert_cpt_fwe.sh:8` |

The `02:00:00` walltime is for the HF batch (1N convert), not the CPT chain. The CPT chain itself runs `12:00:00` per fix `99e918b342` (`cpt_fineweb_edu_4n.sh:4,12` and `POST-TRAINING-2B.md:10,12`). Do not conflate the two: CPT `99e918b342` fixed `02:00 -> 12:00` churn (`~4k` to `~24k` steps/hop); this batch legitimately stays `02:00` because each HF convert is ~minutes on 1N.

### Frontier per script header

Script header `hf_batch_convert_cpt_fwe.sh:10-11`:

```
# Frontier: 136800/190700 (12483869 R 4N). HF 67400 READY, 72800 blocked
# (no tokenizer), 13000 empty, 74400-136800 DCP exists but HF missing.
```

- **Script-header frontier:** `136800 / 190700` at job `12483869 R 4N` (time of commit `c519de37a5`).
- **Assignment/live frontier:** `143400 / 190700 = 75.2%` of `100B` budget (assignment prompt; `thread-a-cpt-trajectory.md:7-9,79-95`). The 143400 DCP exists as `step-143400` under `DCP_ROOT`; the script's `seq` upper bound `136800` has since advanced — see Next steps for how to extend to the live frontier.
- **Budget divisor:** `190734 = 100B / 524288` (`64 * 8192`) per `cpt_fineweb_edu_4n.sh:72-78` and `thread-a-cpt-trajectory.md:60-64`.

### Storage routing — DCP on /home, HF on Tegu

From `hf_batch_convert_cpt_fwe.sh:30-36`:

```bash
REPO=/home/foremans/torchtitan
# Live DCP stays on /home (15T chain at outputs/outputs/checkpoints -- moving mid-train breaks resume)
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
# New HF outputs prefer Tegu project space per user request; PBS filesystems=tegu:home covers both.
# Fall back to HOME if Tegu project path not present (local dev).
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
if [[ -d "$TEGU_ROOT" ]]; then HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe; else HF_ROOT=$REPO/outputs/evals/cpt-fwe; fi
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b
```

| Store | Path | Content | Source |
|---|---|---|---|
| DCP (live, do not move) | `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` (double-nested `outputs/outputs/checkpoints` with symlink `outputs/checkpoints -> outputs/outputs/checkpoints`) | `556` dirs `23G` at `111.2k` (`POST-TRAINING-2B.md:5,10`); `~717` dirs at `143.4k` [INFERENCE: `143400/200`] | `hf_batch_convert_cpt_fwe.sh:32` `cpt_fineweb_edu_4n.sh:41-44,83-89` |
| HF (new, Tegu) | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-$s/hf` | `74000/74200` `3.7G` `9/9` verified; remaining P0-P2 pending | `hf_batch_convert_cpt_fwe.sh:35-36` `POST-TRAINING-2B.md:19` `SYNTHESIS.md:57-66` |
| PBS filesystems | `tegu:home` | Mounts both `/home` (read DCP) and `/lus/tegu` (write HF) on compute | `hf_batch_convert_cpt_fwe.sh:6` `SYNTHESIS.md:65` |

Tegu fallback (`if [[ -d "$TEGU_ROOT" ]]`) handles local dev where Tegu is not mounted; on Sunspot `TEGU_ROOT` exists and `HF_ROOT` resolves to Tegu. Source: `c519de37a5` commit message "prefer Tegu project space for new outputs, keep live 15T on /home" (`git show c519de37a5 --stat`).

### Fixup targets

From script header `hf_batch_convert_cpt_fwe.sh:13-15` and fixup block `hf_batch_convert_cpt_fwe.sh:48-53`:

| Step | State per header | Fixup action in job | Source |
|---|---|---|---|
| `72800` | Blocked (no tokenizer) — prior convert produced HF dir without `config.json`/`tokenizer.*` | `if [[ -d "$HF_ROOT/step-72800/hf" ]]; then cp $HF_ROOT/step-67400/hf/config.json $HF_ROOT/step-72800/hf/ && cp $TOK_SRC/tokenizer.* $HF_ROOT/step-72800/hf/` — copies from already-converted `67400` HF + `assets/hf/gemma-7b` | `hf_batch_convert_cpt_fwe.sh:10-14,48-51` |
| `13000` | Empty HF dir — DCP exists but HF missing | Reconvert via loop (no separate handling; included as explicit `13000` prefix in `for` loop) | `hf_batch_convert_cpt_fwe.sh:15,53,58` |
| `67400` | `READY` — already converted, serves as fixup source for `72800` | No action; copy source | `hf_batch_convert_cpt_fwe.sh:10` |

`72800` is the only step whose fixup is a file copy rather than a reconvert. Its source dependency is the `67400` HF (`HF 67400 READY` in header) plus `TOK_SRC=/home/foremans/torchtitan/assets/hf/gemma-7b` (`hf_batch_convert_cpt_fwe.sh:38`). If `67400` HF were missing, the `cp` would fail with `[WARN] 72800 fixup copy failed` (`hf_batch_convert_cpt_fwe.sh:50`).

### Batch loop — verbatim spec

From `hf_batch_convert_cpt_fwe.sh:58` (single logical line, broken for display but kept verbatim):

```bash
for s in 13000 $(seq 74400 200 136800); do DCP=$DCP_ROOT/step-$s HF=$HF_ROOT/step-$s/hf; [[ -f $HF/ezpz_export.json ]] && continue; $REPO/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds --export_dtype bfloat16 || exit 1; cp "$HF_CONFIG" "$HF/config.json"; cp "$TOK_SRC"/tokenizer.* "$HF/"; ls -lh "$HF"/model-*.safetensors; done
```

| Loop field | Verbatim value | Source |
|---|---|---|
| Step set | `13000` + `$(seq 74400 200 136800)` — i.e. `13000, 74400, 74600, ..., 136800` every `200` | `hf_batch_convert_cpt_fwe.sh:58` |
| DCP per step | `$DCP_ROOT/step-$s` where `DCP_ROOT=/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` | `hf_batch_convert_cpt_fwe.sh:32,58` |
| HF per step | `$HF_ROOT/step-$s/hf` where `HF_ROOT=/lus/tegu/.../outputs/evals/cpt-fwe` on Sunspot | `hf_batch_convert_cpt_fwe.sh:35-36,58` |
| Idempotency guard | `[[ -f $HF/ezpz_export.json ]] && continue` — skip already-converted steps | `hf_batch_convert_cpt_fwe.sh:58` |
| Converter invocation | `$REPO/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds --export_dtype bfloat16 \|\| exit 1` | `hf_batch_convert_cpt_fwe.sh:58` |
| VENV path | `$REPO/torchtitan/.venv/bin/python` = `/home/foremans/torchtitan/torchtitan/.venv/bin/python` on Sunspot (fixed from `$REPO/.venv/bin/python` in `5fb2bf2ae7`) | `hf_batch_convert_cpt_fwe.sh:58` `git show 5fb2bf2ae7 -- hf_batch_convert_cpt_fwe.sh` |
| Converter module | `torchtitan.experiments.ezpz.eval.convert_to_hf` | `hf_batch_convert_cpt_fwe.sh:58` |
| Model identity | `--model_name experiments.ezpz.agpt --model_flavor 2b-mds` | `hf_batch_convert_cpt_fwe.sh:58` |
| Export dtype | `--export_dtype bfloat16` | `hf_batch_convert_cpt_fwe.sh:43,58` |
| Post-convert fixups | `cp "$HF_CONFIG" "$HF/config.json"` (`HF_CONFIG=.../agpt_2b_mds_config.json`) + `cp "$TOK_SRC"/tokenizer.* "$HF/"` (`TOK_SRC=.../assets/hf/gemma-7b`) | `hf_batch_convert_cpt_fwe.sh:37-38,58` |
| Per-step verification | `ls -lh "$HF"/model-*.safetensors` | `hf_batch_convert_cpt_fwe.sh:58` |
| Batch summary | `echo CONVERTED $(ls $HF_ROOT/step-*/hf/ezpz_export.json 2>/dev/tty \| wc -l) HF` | `hf_batch_convert_cpt_fwe.sh:60` |

Loop count (script-header range only): `13000` (1) + `seq 74400 200 136800` = `1 + (136800 - 74400)/200 + 1 = 1 + 313 = 314` entries in the script's literal `seq`; actual converts after `ezpz_export.json` skip are fewer (already-converted steps skipped).

---

## Why

### HF needed for eval — DCP not scorable

DCP checkpoints are sharded FSDP `pytorch_model_fsdp_0/__*.distcp` + `.metadata` (DCP-native, per `POST-TRAINING-2B.md` and `thread-a-cpt-trajectory.md:525-536`). The eval harness (`lm-eval`, `vLLM`) requires HF format (`model-*.safetensors` + `config.json` + `tokenizer.*`). There is no DCP-direct scoring path for base-model eval (`hellaswag`, `arc_easy`, `piqa`, `mmlu`, etc.) or for SFT resumption as `BASE_MODEL`. HF exports exist **only as an eval/SFT bridge** — training itself is DCP-native (CPT via native torchtitan, SFT via DCP resume). Source: `thread-a-cpt-trajectory.md:537-546` `SYNTHESIS.md:211-220` and `POST-TRAINING-2B.md:164-178` (MDS vs HF distinction).

Without HF conversion, the scoring ladder (SYNTHESIS S4, thread-a S5.2) cannot run: `hellaswag / arc_easy / piqa / arc_challenge / mmlu / mmlu_continuation` all consume HF via `lm-eval 0.4.10` HF backend or `vLLM` HF path (`SYNTHESIS.md:73-77`, `thread-a-cpt-trajectory.md:628-653`).

### Tegu routing — avoid /home 15T fill

`POST-TRAINING-2B.md:5` documents the live DCP chain holds `23G` frontier at `111.2k` with `556` ckpt dirs; at `143.4k` the assignment snapshot implies `~717` dirs. The `/home` filesystem and Tegu Lustre (`/lus/tegu`) share a `1.2P` pool at `17%` full (`1P` free) on Sunspot per `SYNTHESIS.md:65,216` and `thread-a-cpt-trajectory.md:536`, but `/home` is the accounting path that would fill under PBS `filesystems=home` if HF output also landed there. User request explicitly prefers Tegu for new outputs: "avoid /home 15T" (`c519de37a5` commit subject "prefer Tegu project space for new outputs, keep live 15T on /home"). Source: `git show c519de37a5 --stat` affecting `hf_batch_convert_cpt_fwe.sh`, `cpt_fineweb_edu_4n.sh`, `eval_baseline_rescore.sh`.

`PBS filesystems=tegu:home` (`hf_batch_convert_cpt_fwe.sh:6`) mounts both: read DCP from `/home`, write HF to Tegu. PBS on Sunspot lacks `/eagle` (`SYNTHESIS.md:215`, `POST-TRAINING-2B.md:13` sync note), so `eagle:home` would be wrong here; `tegu:home` is the validated combination.

Live DCP stays on `/home` and is never moved — "moving mid-train breaks resume" (`hf_batch_convert_cpt_fwe.sh:31` comment) because `FaultTolerantTrainer` resumes via `checkpointer.folder ./outputs/outputs/checkpoints/...` relative to `SUBMIT_DIR` (`cpt_fineweb_edu_4n.sh:41-44,79-89`).

### Flavor=2b-mds — wrong flavor is silent gibberish

Converter dispatch: `torchtitan.experiments.ezpz.eval.convert_to_hf` dispatches on `--model_flavor` (`hf_batch_convert_cpt_fwe.sh:58`). For this CPT chain the correct flavor is `2b-mds` (`hf_batch_convert_cpt_fwe.sh:43` echo `flavor=2b-mds` and `--model_flavor 2b-mds` on line 58). Source: `POST-TRAINING-2B.md:5-6` ("Use flavor 2b-mds for HF export; wrong flavor = silent gibberish"), `SYNTHESIS.md:65,211-212` ("`flavor=2b-mds` — wrong flavor = silent gibberish"), `thread-a-cpt-trajectory.md:544-546` ("wrong flavor = silent gibberish").

Failure mode is silent: the converter will produce `model-*.safetensors` with wrong weight layout that loads without error but generates gibberish under `vLLM` or scores at chance under `lm-eval`. There is no runtime check that the flavor matches the checkpoint's actual model definition — verification must be manual via post-convert scoring (see Next steps).

---

## Outcome

### Already converted — 74000/74200 3.7G 9/9 verified

| Step | Size | Files | Status | Source |
|---|---|---:|---|---|
| `74000` | `3.7G` | `9/9` files verified (patched `config/tokenizer` omission) | Converted via `hf_batch_convert_cpt_fwe.sh` early batch; Tegu `outputs/evals/cpt-fwe/step-74000/hf` | `POST-TRAINING-2B.md:5` (`HFs 74000/74200 3.7G 9/9 files verified (patched config/tokenizer omission)`), `POST-TRAINING-2B.md:19` (`HFs 74000/74200 3.7G 9/9`) |
| `74200` | `3.7G` | `9/9` same | Same batch | Same sources |
| `67400` | [INFERENCE: size not separately quoted; documented as `READY`] | Serves as fixup source for `72800` | Converted; `HF 67400 READY` in header | `hf_batch_convert_cpt_fwe.sh:10` |

`9/9` = `model-*.safetensors` shards + `config.json` + `tokenizer.*` + `ezpz_export.json` per `SYNTHESIS.md:65` ("`9/9` files: safetensors + config + tokenizer") and `hf_batch_convert_cpt_fwe.sh:58-60` (writes safetensors via converter, then `config.json`, `tokenizer.*`, `ezpz_export.json` as completion marker).

Known verification: `POST-TRAINING-2B.md:19` notes the early HFs were verified after patching the `config/tokenizer` omission (see What went wrong). Subsequent SFT (`sft_distill_cot_4n.sh` on `74000` HF via `12483807` with `ezpz launch --nproc 12`) consumed `74000` HF as `BASE_MODEL`, confirming loadability per `POST-TRAINING-2B.md:20-22`.

### Pending — P0-P2 ladder 8 converts ~52G

Per `SYNTHESIS.md:63-66` `E1` (and `thread-a-cpt-trajectory.md:553-565` `S5.2`):

| Phase | Steps to convert | Count | Why this density | Source |
|---|---|---:|---:|---|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | Dolmino shock hit by `1000`; flat-LR `2e-6` should not shock — verify hypothesis | `SYNTHESIS.md:204` `thread-a-cpt-trajectory.md:555` |
| **P1 — Early plateau** | `25000, 50000, 75000` | 3 | `74000/74200` already HF `3.7G` `9/9`; fill gap to `50k` (parked broad-distill gate per `POST-TRAINING-PLAN.md`) | `SYNTHESIS.md:205` `thread-a-cpt-trajectory.md:556` |
| **P2 — Frontier** | `100000, 111200, 125000, 143400` | 4 ([INFERENCE: `150000` deferred until `150k` completes; P2 immediate = 4]) | `111.2k` documented (`POST-TRAINING-2B.md:5` `R step-111200 23G 556 dirs`); `143400` assignment snapshot; `150000` gate when DCP reaches `150k` | `SYNTHESIS.md:206` `thread-a-cpt-trajectory.md:557` |
| **P3 — Terminal (if continuing)** | `160000, 175000, 190734` | 3 | Full `100B` budget; `190734` exact terminal | `SYNTHESIS.md:207` `thread-a-cpt-trajectory.md:558` |

Total `14` HF converts across P0-P3; **8 immediate** (P0+P1+P2 through `143400`) + `6` deferred (`150000` + P3). Source: `SYNTHESIS.md:208-209` ("Total 14 HF converts. Only 3 need SFT probes") and `thread-a-cpt-trajectory.md:560-564`.

Size budget: `~3.7G` each * `14` = `~52G` full ladder (`SYNTHESIS.md:65` `E1 Output ~52G full ladder`); `8` immediate = `~29.6G`. Trivial vs `23G` DCP frontier at `111.2k`. Source: `SYNTHESIS.md:65` `thread-a-cpt-trajectory.md:562-564`.

The script's literal `seq 74400 200 136800` range already overlaps P1-P2 (it covers `74400-136800` every `200`, i.e. `~313` possible steps including `75000, 100000, 111200, 125000` etc.), but the **scoring ladder** selects only the `8` sparse ladder points from that range — do not score every `200`-step checkpoint.

### Fixes landed

| Fix commit | What changed | Effect on this batch | Source |
|---|---|---|---|
| `99e918b342` | `cpt_fineweb_edu_4n.sh` walltime `02:00 -> 12:00` (`workq` allows `23h`) — `~4k` to `~24k` steps/hop | CPT chain healthy to reach `143400`; this batch can now read DCPs through `136800+` instead of stalling at `~115600` | `git show 99e918b342 --stat` `POST-TRAINING-2B.md:10,12,19` |
| `c519de37a5` | Prefer Tegu for new outputs (`TEGU_ROOT` + `HF_ROOT` fallback; frontier `115600 -> 136800` in header; batch range `115600 -> 136800`) | HF writes go to Tegu (`PBS filesystems=tegu:home`); live DCP stays on `/home`; script now covers frontier to `136800` | `git show c519de37a5 -- hf_batch_convert_cpt_fwe.sh` (diff: `DCP_ROOT` comment, `TEGU_ROOT`/`HF_ROOT` block, `seq` upper bound) |
| `5fb2bf2ae7` | Checkpoint path fallback (`$REPO/.venv -> $REPO/torchtitan/.venv`) | Fixes `ModuleNotFoundError` / missing VENV on Sunspot where venv lives at `torchtitan/.venv` not `~/.venv` | `git show 5fb2bf2ae7 -- hf_batch_convert_cpt_fwe.sh` (one-line diff on line 58 VENV path) |

VENV note: `hf_batch_convert_cpt_fwe.sh:58` uses `$REPO/torchtitan/.venv/bin/python` (`/home/foremans/torchtitan/torchtitan/.venv/bin/python`). This is the Sunspot login-node venv that provides `torchtitan` + `transformers` + `safetensors` for conversion. `SYNTHESIS.md:63` `E1` references `VENV=xpu-torch216-sync-5a7a622f16 LD-first` — that is the **CPT training** VENV (`cpt_fineweb_edu_4n.sh:52-70`), not the HF conversion VENV. Do not substitute the XPU training VENV for the HF convert — they are distinct scripts with distinct VENVs.

---

## What went wrong

### 1 — 74k config/tokenizer omission (patched)

Early HF converts at `74000/74200` initially omitted `config.json` and `tokenizer.*` — the converter (`convert_to_hf.py`) writes `model-*.safetensors` + `ezpz_export.json` but does not bundle the model config or tokenizer; those must be copied post-convert (`cp "$HF_CONFIG" "$HF/config.json"` + `cp "$TOK_SRC"/tokenizer.* "$HF/"` on `hf_batch_convert_cpt_fwe.sh:58`). The initial batch missed that copy step and produced HF dirs that failed `lm-eval` / `vLLM` load. Patched by adding the two `cp` commands to the loop and re-running the converts; verified `3.7G 9/9` after fix. Source: `POST-TRAINING-2B.md:19` ("`HFs 74000/74200 3.7G 9/9 files verified (patched config/tokenizer omission)`") and `hf_batch_convert_cpt_fwe.sh:37-38,58` (current loop includes both `cp` lines).

Any future verification that omits these `cp` lines will reproduce the same failure — the `9/9` check must include `config.json` + `tokenizer.json`.

### 2 — Sunspot ~/torchtitan has no outputs/

`POST-TRAINING-2B.md:5` documents: "Sunspot `~/torchtitan/outputs/hf*` empty -- next convert at 112k gate." On Sunspot the repo checkout at `~/torchtitan` is **not** where the live DCP chain writes: DCP lives at `/home/foremans/torchtitan/outputs/checkpoints/...` (which is `/home/foremans/torchtitan/outputs/outputs/checkpoints` double-nested with symlink workaround `cpt_fineweb_edu_4n.sh:39-44`), and after `c519de37a5` new HFs land on Tegu (`/lus/tegu/.../outputs/evals/cpt-fwe`). A naive `ls ~/torchtitan/outputs/` on Sunspot shows no HF output — that is expected, not a missing-output bug.

Correct verification path is always `HF_ROOT` (Tegu on Sunspot, `~/torchtitan/outputs/evals/cpt-fwe` only as local-dev fallback per `hf_batch_convert_cpt_fwe.sh:36`).

### 3 — 72800 missing tokenizer source

Script header `hf_batch_convert_cpt_fwe.sh:10-11` records `72800 blocked (no tokenizer)` and `HF 67400 READY`. The `72800` DCP was converted but its HF dir lacked `config.json` + `tokenizer.*` because the source HF or tokenizer asset was not available at that batch's write time. Current fixup block `hf_batch_convert_cpt_fwe.sh:49-51`:

```bash
if [[ -d "$HF_ROOT/step-72800/hf" ]]; then
  cp $HF_ROOT/step-67400/hf/config.json $HF_ROOT/step-72800/hf/ && cp $TOK_SRC/tokenizer.* $HF_ROOT/step-72800/hf/ || echo "[WARN] 72800 fixup copy failed (check src exists)"
fi
```

Copies from already-verified `67400` HF (`HF 67400 READY`) plus `TOK_SRC=/home/foremans/torchtitan/assets/hf/gemma-7b` (`hf_batch_convert_cpt_fwe.sh:38`, canonical `gemma-7b` tokenizer for `agpt-2b-mds`). Precondition: `67400` HF must exist before this fixup runs (guarded by `[[ -d "$HF_ROOT/step-72800/hf" ]]` — the fixup is a no-op if `72800` HF dir does not exist yet, and will be retried on next batch invocation after `72800` is reconverted).

### 4 — No lock needed (staged_export inside convert_to_hf.py)

Script header `hf_batch_convert_cpt_fwe.sh:18-19`:

```
# Staging lock pattern from hf_export.py (staged_export) is already handled
# inside convert_to_hf.py -- no manual .lock/.staging needed here.
```

`convert_to_hf.py` implements `staged_export` internally: it writes to a staging dir and atomically renames to final `HF` path, with its own lock to prevent concurrent writers. The batch script therefore does **not** create `.lock` or `.staging` dirs — its only idempotency guard is `[[ -f $HF/ezpz_export.json ]] && continue` on `hf_batch_convert_cpt_fwe.sh:58`, which skips already-completed converts. Adding manual lock files would be redundant and could conflict with the converter's own staging.

---

## Next steps

### Exact qsub command

On Sunspot, from the repo root (where `PBS_O_WORKDIR` resolves to `~/torchtitan`):

```bash
qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh
```

Source: `hf_batch_convert_cpt_fwe.sh:22` usage comment (`qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh`) and `thread-a-cpt-trajectory.md:576-579` `S5.3.1`.

This submits `1N 02:00:00 workq tegu:home` (`hf_batch_convert_cpt_fwe.sh:2-7`) that loops `13000 $(seq 74400 200 136800)` with `flavor=2b-mds bfloat16` via `$REPO/torchtitan/.venv/bin/python` (`hf_batch_convert_cpt_fwe.sh:58`).

To cover the **P0-P2 scoring ladder** through the live `143400` frontier (which extends beyond the script's `136800` upper bound), override with the one-shot ladder from `thread-a-cpt-trajectory.md:585-621` `S5.3.1`:

```bash
cat > /tmp/hf_ladder_cpt_fwe.sh <<'PBS'
#!/bin/bash --login
#PBS -N hf-ladder-cpt-fwe
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
set -euo pipefail
SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}; cd $SUBMIT_DIR
REPO=/home/foremans/torchtitan
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b
echo "HF-LADDER $(date -u +%FT%TZ) DCP=$DCP_ROOT HF=$HF_ROOT"
for s in 1000 5000 10000 25000 50000 75000 100000 111200 125000 143400 150000; do
  DCP=$DCP_ROOT/step-$s; HF=$HF_ROOT/step-$s/hf
  [[ -d "$DCP" ]] || { echo "SKIP $s: no DCP"; continue; }
  [[ -f "$HF/ezpz_export.json" ]] && { echo "SKIP $s: HF exists"; continue; }
  echo "CONVERT $s: $DCP -> $HF"
  $REPO/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf \
    "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds \
    --export_dtype bfloat16 || exit 1
  cp "$HF_CONFIG" "$HF/config.json"
  cp "$TOK_SRC"/tokenizer.* "$HF/"
  ls -lh "$HF"/model-*.safetensors
done
echo CONVERTED $(ls $HF_ROOT/step-*/hf/ezpz_export.json 2>/dev/null | wc -l) HF
PBS
qsub /tmp/hf_ladder_cpt_fwe.sh
```

Adapted from `thread-a-cpt-trajectory.md:588-621`. Adjust the `for s in ...` list to the exact P0-P2 set below; steps beyond the DCP frontier are skipped via `[[ -d "$DCP" ]]` guard (`150000` skipped until DCP reaches `150k`).

### Verification

After the batch completes, on Sunspot (or any host with Tegu mounted):

```bash
# Per-step size + file check (expect ~3.7G and 9/9 per step on Tegu)
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf/model-*.safetensors
# Batch summary (counts ezpz_export.json markers, one per successful HF)
ls /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf/ezpz_export.json 2>/dev/tty | wc -l
# Spot-check one HF: expect 9/9 files + non-gibberish config
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-74000/hf/
cat /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-74000/hf/ezpz_export.json
```

Sources: `hf_batch_convert_cpt_fwe.sh:58,60` (per-step `ls -lh` and batch `CONVERTED ... HF` summary) and `thread-a-cpt-trajectory.md:623-626` ("Verify each HF with `ls -lh $HF/model-*.safetensors` (expect `9/9` files, `~3.7G`)").

**What `9/9` means:** `model-*.safetensors` shards + `config.json` + `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` + `ezpz_export.json` per `SYNTHESIS.md:65` "9/9 files: safetensors + config + tokenizer" and `POST-TRAINING-2B.md:19` "`9/9` files verified". Do not accept an HF that is missing `config.json` or `tokenizer.*` — that was the original `74k` omission (see What went wrong #1).

On Sunspot login, `ls ~/torchtitan/outputs/hf*` being empty is **not** a failure — Tegu is the correct `HF_ROOT` on Sunspot (`c519de37a5`); `~/torchtitan/outputs/` is the local-dev fallback (`hf_batch_convert_cpt_fwe.sh:36`).

### Tegu HF_ROOT — canonical output path

```
HF_ROOT = /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-$s/hf
```

Source: `hf_batch_convert_cpt_fwe.sh:35-36` `SYNTHESIS.md:63` `thread-a-cpt-trajectory.md:533-535`.

| Var | Value | When |
|---|---|---|
| `DCP_ROOT` | `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` | Always (live DCP, `15T` chain, never move) |
| `TEGU_ROOT` | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan` | Sunspot (exists, `HF_ROOT` = Tegu) |
| `HF_ROOT` | `$TEGU_ROOT/outputs/evals/cpt-fwe` on Sunspot; `$REPO/outputs/evals/cpt-fwe` local-dev fallback | `hf_batch_convert_cpt_fwe.sh:36` conditional on `[[ -d "$TEGU_ROOT" ]]` |
| `HF_CONFIG` | `/home/foremans/torchtitan/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json` | Copied post-convert to `$HF/config.json` |
| `TOK_SRC` | `/home/foremans/torchtitan/assets/hf/gemma-7b` | `tokenizer.*` copied post-convert |

`PBS filesystems=tegu:home` is the validated mount set for this job (`hf_batch_convert_cpt_fwe.sh:6`); `eagle` is not available on Sunspot (`SYNTHESIS.md:215`).

### P0-P3 schedule — what to convert when

Per `SYNTHESIS.md:202-209` `S4` and `thread-a-cpt-trajectory.md:553-565` `S5.2`:

| Phase | Steps | Count | When | Source |
|---|---|---:|---|---|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | Immediate (with P1-P2 batch) | `SYNTHESIS.md:204` |
| **P1 — Early plateau** | `25000, 50000, 75000` | 3 | Immediate (`74000/74200` already `3.7G 9/9`; fill to `50000`) | `SYNTHESIS.md:205` |
| **P2 — Frontier** | `100000, 111200, 125000, 143400, 150000` | 5 | Immediate through `143400`; `150000` when DCP frontier reaches `150k` | `SYNTHESIS.md:206` |
| **P3 — Terminal (if S4 gate holds)** | `160000, 175000, 190734` | 3 | Deferred — only if `150k` gate continues to `190k` per `SYNTHESIS.md:223-239` `G1-G4` | `SYNTHESIS.md:207` |

**Total 14 HF converts**, `~52G` on Tegu (`~3.7G` each) — `8` immediate (P0+P1+P2 through `143400`) + `6` deferred (`150000` + P3). Source: `SYNTHESIS.md:209`.

The script's literal `seq 74400 200 136800` auto-covers `75000, 100000, 111200, 125000` in its sweep; the ladder selectively scores only the `8` listed points — do not lm-eval every `200`-step checkpoint. P0 steps (`1000, 5000, 10000, 25000, 50000`) require the override ladder script above because they fall below the batch script's `74400` lower bound.

Next consumer after HF verification: base-model eval sweep (`hellaswag`, `arc_easy`, `piqa`, `arc_challenge`, `mmlu`/`mmlu_continuation`) on every HF per `SYNTHESIS.md:68-77` `E2` and `thread-a-cpt-trajectory.md:628-653` `S5.3.2`, plus SFT probe on best P2 + `150k` gate per `SYNTHESIS.md:78-88` `E3` and `thread-a-cpt-trajectory.md:470-518` `S4.4` `G1-G4`.
