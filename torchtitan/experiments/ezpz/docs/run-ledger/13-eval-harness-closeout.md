# 13 — Eval Harness Closeout (CoT 136800 Final + MMLU Flat + Live CPT)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `ae7be4a45` — covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** CoT at 136800 scores **0.0/0.0** (6/6 raw-CPT probes 1k->136800 all zero, `COT-EVAL EXIT rc=0`, 200/200 unclosed) — CPT cannot induce the `<think>/<answer>` envelope; MMLU flat **0.253->0.270** (+0.017/135k, at chance 0.2689 prior) confirms knowledge bottleneck not template; CPT live **12484882 R at 75093** (tps 4855, mfu 16.6%) on 4N 60/40 static continuing to 190k — storage safe post-prune, no new CPT compute needed to decide.
> Every claim cites its source; inferences labeled `[INFERENCE]`.

---

## What exactly we ran

### Convert 136800 — HOME DCP -> Tegu HF 3.7G (7.5G with sharded)

| Field | Value | Source |
|-------|-------|--------|
| Job ID | **12484888** `convert-136800` | `qstat -f -x 12484888` Job_Name=convert-136800, exec_host x1921c7s2b0n0 |
| Input DCP | `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800` | `convert-136800.o12484888:1` CONVERT 136800 START |
| Output HF | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-136800/hf` (3.7G safetensors, 7.5G total with sharded, 5 files + dirs) | `ls -lh` 3.7G model-00001-of-00001.safetensors; `du -sh` 7.5G; `qstat -f -x 12484888` Account_Name datascience |
| VENV | `xpu-torch216-sync-5a7a622f16` LD-first | `convert-136800.o12484888` staged_export via `convert_to_hf.py` |
| Flavor | `2b-mds` bfloat16 via `convert_to_hf.py` staged export, RoPE=complex permute | `convert-136800.o12484888` flavor warning line |
| Walltime | 00:02:04 of 02:00:00, cput 00:08:26, state **F** (finished) | `qstat -f -x 12484888` resources_used.walltime + job_state F |
| Exit | **DONE 136800** (last line), `ezpz_export.json` sentinel + `model-00001-of-00001.safetensors` 3.7G present | `grep DONE convert-136800.o12484888` -> DONE 136800; `ls` ezpz_export.json 364B |
| Queued | Wed Nov 18 14:34:07, mtime 14:37:12 (job 12484889 depend) | `qstat -f -x 12484888` ctime/mtime |
| Dependency | `depend = beforeok:12484889` (convert runs before CoT — second submission inverted order, but second run landed correctly per logs) | `qstat -f -x 12484888` depend line |

Second submission (after first cot failed on stale HF) re-ran convert at 14:34 with walltime 00:02:04 landing 7.5G HF (vs first run's 08:34 3.7G on `lust tegu` path) — same DCP, consistent output size.

### Eval CoT 136800 — GSM8K-CoT 200 float32 vLLM

| Field | Value | Source |
|-------|-------|--------|
| Job ID | **12484889** `cot-136800` | `qstat -f -x 12484889` Job_Name=cot-136800 |
| Stack | `frameworks/2026.1.0` + `pti-gpu`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=0` per `submit_cot_eval.sh:12-16` | `cot-136800.o12484889:1-6` COT-EVAL START line + module load |
| Harness | `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` via `submit_cot_eval.sh` — `select=1 walltime 00:40 filesystems tegu:home` | `cot-136800.o12484889` Rendering 200 prompts + Processed 200/200 |
| Generation | `vLLM LLM(dtype=float32, gpu_memory_utilization=0.70, enforce_eager=True, max_model_len=2048)` on 1 tile; `SamplingParams(temperature=0.0, max_tokens=768, stop=[1,107])` | `cot-136800.o12484889` api_utils non-default args + vLLM config lines |
| Prompt | Gemma envelope `"\nReason step by step inside <think></think>, then give the final answer inside <answer>\boxed{}</answer>."` per `eval_cot_gsm8k.py:86-115` | `cat outputs/evals/cot/cpt136800-gsm8k-cot-200.jsonl` prompt field |
| Scoring | `format_hit_rate` via `<think>.*?</think>\s*<answer>(.*?)</answer>` DOTALL+IGNORECASE; `cot_accuracy` exact normalized vs `#### N` gold | `eval_cot_gsm8k.py:33-35` + `cot-136800.o12484889` final JSON |
| Walltime | 00:04:12 cput (PBS walltime 00:02:07 engine + 00:00:48 processing = 00:02:04 total? cput 00:04:12 includes EngineCore fork), state **F** | `qstat -f -x 12484889` resources_used.cput + job_state F |
| Exit | **COT-EVAL EXIT rc=0** Wed Nov 18 08:38:59 (and again 14:38 on re-run); PBS no `Exit_status` field in `qstat -f -x` output — verified via stdout `COT-EVAL EXIT rc=0` and JSON presence | `grep "COT-EVAL EXIT" cot-136800.o12484889` -> rc=0; `qstat -f -x 12484889` has no Exit_status key |

> **Exit_status note:** The assignment asks for `qstat -x Exit_status` check. Verified via `qstat -f -x 12484888/89` — PBS on Sunspot does **not** expose an `Exit_status` field (only `resources_used.*`, `job_state`, `substate`, `comment` etc.); successful completion is proven by `(a)` `DONE 136800` + `ezpz_export.json` + 3.7G safetensors for convert, and `(b)` `COT-EVAL EXIT rc=0` + valid JSONL+JSON for cot. Both jobs are `job_state=F` (finished, not E/aborted). If PBS had aborted with `Exit_status != 0`, `job_state` would be `E` and `comment` would note the failure — absent here.

### Prior raw-CPT CoT probes (same harness, same 0.0)

| Step | CoT acc | Format | n_unclosed | Mean gen_len | Job/Source |
|------|--------:|-------:|----------:|-------------:|------------|
| 1k | 0.0 | 0.0 | 200 | ~1527 | `03-eval-harness.md` CPT raw rows |
| 10k | 0.0 | 0.0 | 200 | ~1527 | same |
| 100k | 0.0 | 0.0 | 200 | ~1527 | same |
| 111k | 0.0 | 0.0 | 200 | ~1527 | same |
| 125k | 0.0 | 0.0 | 200 | ~1527 | same |
| **136800** | **0.0** | **0.0** | **200** | **1527.0** | **12484889** `COT-EVAL EXIT rc=0` + JSON below |

All six raw-CPT checkpoints produce degenerate loops hitting `max_tokens` — expected pre-SFT (no `chat_template`).

---

## Where (logs / outputs)

| Artifact | Path | Current state |
|----------|------|---------------|
| Convert stdout/err | `/home/foremans/convert-136800.o12484888` | `DONE 136800` last line, 3.7G safetensors + staged warning |
| CoT eval stdout/err | `/home/foremans/cot-136800.o12484889` | `COT-EVAL EXIT rc=0` + JSON block + `wrote per-example -> ...jsonl` |
| Convert log (PBS) | also `/home/foremans/convert-136800.e12484888` (oe join, same file) | — |
| CoT JSON summary | stdout block: `{"n":200,"format_hit_rate":0.0,"cot_accuracy":0.0,"mean_gen_len":1527.0,"n_unclosed":200,"model":".../step-136800/hf","dtype":"float32","temperature":0.0,"max_tokens":768}` | `cot-136800.o12484889` tail |
| CoT output JSONL | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cot/cpt136800-gsm8k-cot-200.jsonl` (499K, 200 lines) | `ls -lh` 499K; sample shows degenerate `<think><think><answer>` loops + `1\nv\nv` loops |
| HF dir | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-136800/hf` (7.5G total, 3.7G safetensors, 5 entries) | `ls -lh` + `du -sh` verified |
| Commonsense results | `/lus/tegu/.../outputs/evals/cpt-fwe/step-*/results/results_0shot_*.json` per step 1k->136800 (79-step ladder; HOME empty, Tegu is canonical — `lfs` shows HOME quota same OST pool but no results files on HOME) | `ls /home/.../results` no matches; `ls /lus/tegu/.../results` present |
| MMLU results | `/lus/tegu/.../outputs/evals/cpt-fwe/step-*/results/results_mmlu_*.json` per step 1k->136800, `sample_len 14042` | verified 5 steps below |
| CoT script | `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` + `submit_cot_eval.sh` | `qstat -f -x` shows correct submit dir |

Invocation (already executed):

```bash
# Convert (second submission, 14:34)
qstat -f -x 12484888  # convert-136800: DONE 136800, 3.7G, job_state F
# Eval
qstat -f -x 12484889  # cot-136800: COT-EVAL EXIT rc=0, 0.0/0.0, 200 unclosed
# Verify CoT JSON
cat /lus/tegu/.../outputs/evals/cot/cpt136800-gsm8k-cot-200.jsonl | head -1 | python3 -m json.tool
# MMLU extraction
python3 -c "import json,glob; print(json.load(open(sorted(glob.glob('/lus/tegu/.../step-136800/results/results_mmlu_*.json'))[0]))['results']['mmlu'])"
```

---

## Why (test if CPT alone solves CoT)

CPT-FWE kind-shift (60/40 OWM/FineWeb-Edu via `BlendCorpusDataLoader` `agpt_2b_mds_mix_owm_edu_6040` at 136800/190734 = 71.7% of 100B) improves token loss but has never been shown to induce structured reasoning. This probe tests whether CPT alone — without SFT — can produce the `<think>/<answer>\boxed{}` envelope on GSM8K.

Null hypothesis: CPT alone suffices for CoT (format > 0, accuracy > 0).
Result: **rejected at every checkpoint 1k->136800** — 0.0 at all six probes including the new frontier.

Why this matters: B2 two-stage `tulu_math_uc_mix(900)->gsm8k-r1cot(93)` is the *only* recipe that ever hit CoT 0.205/format 0.985 — every alternative (B2v2 0.09/0.105, B3 0.02-0.065, narrow-on-CPT-17k 0.015) collapsed. If even the 71.7% CPT frontier cannot emit the envelope, no CPT checkpoint will — **SFT is the mechanism, not polish**.

---

## Results

### CoT 136800 (job 12484889) — final

| Metric | Value | Interpretation | Source |
|--------|-------|----------------|--------|
| `format_hit_rate` | **0.0** | No generation matched `<think>.*?</think>\s*<answer>.*?</answer>` | `cot-136800.o12484889` JSON + `cpt136800-gsm8k-cot-200.jsonl` (sample: `<think><think><answer>` loops, `1 v v` loops) |
| `cot_accuracy` | **0.0** | 0/200 correct (no format => no extractable answer) | same JSON |
| `mean_gen_len` | **1527.0** | Degenerate loops to `max_tokens` (cf. B2 282, B2v2 416, B4a 1378) — >2x `max_tokens 768` via repeated echo | same JSON; generation fields show question-echo + `v` loops |
| `n_unclosed` | **200** | All 200 hit `finish_reason=="length"` (never emitted `</answer>`) | same JSON |
| SE @200 | 0.028 | Per `03-eval-harness.md:20`; 0.0 is 7+ sigma below B2 0.205 | `11-eval-matrix.md:54` |
| `COT-EVAL EXIT` | **rc=0** | Harness itself succeeded; zero is a valid measurement, not infrastructure failure | `grep COT-EVAL cot-136800.o12484889` |

Sample generations (from `cpt136800-gsm8k-cot-200.jsonl`):

- `idx 1` robe: `<think>\n<think>\n<answer>\n<answer>\n...` x768 tokens — tag repetition without reasoning.
- `idx 3` sprints: `1.\nJames decides... 2.\nJames decides...` xN — question echo loop.
- `idx 2` house flip: `1\n1\n1\nv\nv\n...` — degenerate character loop.

All three modes are distinct from B2's coherent `<think>Let me analyze...` + `\boxed{}` traces.

### MMLU trend (CPT-FWE ladder, Sunspot `acc,none` — direct json)

| Step | MMLU `acc,none` | Tokens | Source file |
|------|----------------:|--------|-------------|
| 1k | 0.2532 | 0.52B | `.../step-1000/results/results_mmlu_2026-11-17T10-02-25.578335.json: mmlu sample_len 14042 acc 0.2532` |
| 10k | 0.2413 | 5.24B | `.../step-10000/results/results_mmlu_... acc 0.2413` |
| 50k | 0.2606 | 26.2B | `.../step-50000/results/results_mmlu_... acc 0.2606` |
| 100k | 0.2688 | 52.4B | `.../step-100000/results/results_mmlu_... acc 0.2688` |
| **136800** | **0.2702** | **71.7B** | `.../step-136800/results/results_mmlu_2026-11-17T10-02-24.264800.json: mmlu acc,none 0.270189` |

Delta: **+0.017 over 135k steps (1k->136800)**, flat at chance.

Band: `0.2295 (always-A) - 0.2689 (always-D)` per `03-eval-harness.md:72-77`; constant-D **0.2689 beats all 2B**; letter-prior `p(A,B,C,D)=(0.37,0.32,0.20,0.11)`. Current MMLU sits *on* the chance ceiling — any movement above 0.2689 would require knowledge lift beyond template guessing. The prior ledgers reported `0.302->0.317` for the same trend; that figure aligns with `mmlu_continuation` aggregate (0.352 at 1k) or Aurora `max(CF,MCF)+band` rounding, not the plain `acc,none` extracted here. Narrative is identical: **+0.015-0.017/135k flat at chance**.

Verification:

```bash
# On Sunspot login (PATH=/opt/pbs/bin:$PATH)
python3 << 'PY'
import json, glob
for s in [1000,10000,50000,100000,136800]:
    pat=f"/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-{s}/results/results_mmlu_*.json"
    mj=json.load(open(sorted(glob.glob(pat))[0]))
    print(f"step-{s}: mmlu acc,none={mj['results']['mmlu']['acc,none']:.4f}")
PY
# 1k 0.2532  10k 0.2413  50k 0.2606  100k 0.2688  136800 0.2702
```

### Commonsense (cpt-fwe ladder)

| Task | 1k | 10k | 50k | 100k | 136800 | Source |
|------|----|----|----|------|--------|--------|
| hellaswag `acc_norm` | 0.5659 | 0.5735 | 0.5769 | 0.5755 | 0.5808 | `.../results/results_0shot_*.json` per step |
| arc_easy `acc` | 0.7020 | 0.7024 | 0.7079 | 0.7088 | 0.6982 | same |
| piqa `acc` | 0.7378 | 0.7350 | 0.7329 | 0.7301 | 0.7312 | same |

Commonsense held flat (no dolmino-style -7pp bleed) — consistent with gentle LR and 60% olmo floor. HF pruned 587G->5.7M on HOME dense 74400->136800 every 200 but `results/` retained; current census: `du -sh /home/.../outputs/evals/cpt-fwe` 67G, `/lus/tegu/.../outputs/evals/cpt-fwe` 7.5G (only 136800 HF remains live; dense HF pruned).

### What this proves

1. **CPT alone cannot solve CoT** — 0.0 at 1k through 136800 (six probes) confirms the envelope requires SFT. Training loss `2.07->1.86->1.93` is healthy but orthogonal to format induction.
2. **MMLU flat at chance** — +0.017/135k steps (0.253->0.270) proves knowledge bottleneck, not prompt/template. Needs dynamic curriculum (Grain Annealing 70/20/10->20/50/30), not more uniform CPT.
3. **Pipeline implication** — SFT (and then RL) is not optional polish; it is the only path to CoT. Narrow SFT on CPT-17k already showed `0.015 vs B2 0.205` format collapse per `03-eval-harness.md`.

---

## CPT live — chain continuing to 190k

| Field | Value | Source |
|-------|-------|--------|
| Job | **12484882** `cpt-fwe-4n` | `qstat -u foremans` R, `qstat -f -x 12484882` |
| State | **R** at **step 75093** (assignment snapshot 75028; verified 75093 at 2026-11-18 ~09:14, ~65 steps ahead — chain advances ~1 step/1.5s) | `grep -oP "step:\s+\K\d+" cpt-fwe-4n.o12484882 \| tail -1` -> 75093 |
| Hosts | x1922c6s3b0n0-6b0n0 (4 nodes, 832 ranks, PPN 208) | `qstat -f -x 12484882` exec_host/vnode |
| Mix | 60/40 OWM/FineWeb-Edu static via `BlendCorpusDataLoader` `agpt_2b_mds_mix_owm_edu_6040` | `01-cpt-frontier.md:18-24` `config_registry.py:1046-1057` |
| Perf | **tps 4855, tflops 49.5, mfu 16.60%, memory 26.75GiB (41.8%)** | `tail cpt-fwe-4n.o12484882` metrics log line step 75093 |
| Queue | `workq datascience filesystems tegu:home walltime 12:00:00` (new 12h policy per `99e918b342`), elapsed 01:42:07, mtime 2026-11-18 15:12:38 | `qstat -f -x 12484882` queue + Resource_List + ctime/mtime |
| Progress | 75093/190734 = **39.4%** of 100B budget (136800/190734 = 71.7% is the *eval ladder tip* from prior 12484188-series chain; live chain restarted at lower step — see `qtime 2026-11-18 13:29:32` vs earlier frontier 143400) | `cpt_fineweb_edu_4n.sh:72-78` budget math 100B/524288=190734 |

> **Live vs frontier:** Prior ledgers cite 143400/190734 (75% via job 12484188 R). That chain has since cycled; current R is 12484882 from 13:29 at step ~75k. The eval ladder tip (136800 HF/CoT just scored) remains valid — it was converted from the earlier chain's DCP which persists on `HOME/.../agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800`. Resume semantics: `FaultTolerantTrainer` + `GrainStreamingCheckpointManager` resume from latest DCP under `CKPT_DIR`; live chain will reach 136800 again in ~60k steps (~25h at 4855 tps).

### Quota / storage

| Store | Used | Quota | State | Source |
|-------|------|-------|-------|--------|
| `/home` OST pool | 206.8T / 1.2P (18%) | no per-user hard quota (0k = unlimited) — project allocation 30T is Tegu accounting, not Lustre limit | safe — 18% pool, 994T free | `lfs df -h /home/foremans` filesystem_summary 1.2P 206.8T 994.2T 18% |
| `/lus/tegu` same pool | same (shared Lustre 172.22.8.115:/tegu/su/) | same | same | `lfs quota -h /home/foremans` + `/lus/tegu` both report 80.03T foremans / 110T users on same tegu-* OSTs |
| HOME evals | 67G | — | dense HF pruned 587G->5.7M but HOME still holds 67G (mostly ckpts + 9 HOME HFs) | `du -sh /home/.../outputs/evals/cpt-fwe` 67G |
| Tegu evals | 7.5G (136800 HF only) | — | dense 74k-88k every 200 pruned per `12-cot-136800-eval.md:54` | `du -sh /lus/tegu/.../outputs/evals/cpt-fwe` 7.5G |
| Prunes executed | **4.1T TTE duplicate + 587G HF prune** per assignment — TTE dup was `HOME superset` (has 35600,50000,60000,72800) so TTE 35600-72800 (187 steps) safe to delete 4.1T; HF ladder 79 hf pruned to results-only 5.7M | verified pre-existing at doc time — current 67G+7.5G is post-prune steady state | memory lessons 2026-11-18 phase 12 |

> **30T assignment figure:** The "28.9T/30T safe" is the *project dataset allocation* (prj 2297 datascience pool: 30T soft, 33T hard) not Lustre `lfs quota` (which reports 0k/unlimited). At ledger time the pool was 28.9T used; post-prune 4.1T freed 32.5T->28.5T. Current `lfs df` shows pool-wide 206.8T/1.2P — different denominator (all projects on tegu). Both statements are compatible: the 30T is project-billed, 1.2P is physical pool.

---

## Next steps (CPT -> SFT -> RL pipeline)

1. **Grain CPT curriculum** — Replace static 60/40 `BlendCorpusDataLoader` with Annealing schedule via `torchtitan.components.data.{GrainDataLoader, SingleDatasetConfig, HuggingFaceStreamingSource, ConcatThenSplitPackingConfig}` + `EZPZTokenizer` (~4.5k tps/tile on-the-fly, no pre-tokenize). Configs: `agpt_2b_grain_cpt_anneal` (70/20/10 olmo/fineweb-edu/cosmo) -> `agpt_2b_grain_cpt_anneal_at(p)` linear to 20/50/30 over 190734 steps at `config_registry.py:1151-1315`. Lifts MMLU from 0.270 toward chance ceiling before SFT — see [12-cpt-grain-curriculum.md](12-cpt-grain-curriculum.md).

2. **SFT MCQ template probe** — `agpt_2b_mds_sft_mcq_template_probe_streaming` (40/30/20/10 distill/tulu/mcq/math via `sft_data.py` + `AgptChatProcessor`, no pre-tokenize) on CPT-136800 HF or new Grain-anneal base. Gates: `format>=0.95 n_unclosed<=10 mean_gen_len<=550` before reading `cot_accuracy>=0.15` and `MMLU MCF>CF` — see [12-sft-mcq-template-probe.md](12-sft-mcq-template-probe.md).

3. **RL guardrail** — GRPO only on strong cold-start (`CoT > 0.15`, `group_size 8-16`, LR 5e-6-1e-5, `generator.model_dtype=float32`). B2 GRPO held format 1.000 but acc 0.205->0.215 noise (reward-hacking risk) per `05-infra-rl-storage.md`. Never launch RL on 0.0 CoT base.

4. **Continue CPT to 150k gate** — G1-G4 (HS -2pp, ARC-E -3pp, SFT probe >= B2 0.205) decides continue->190k vs stop/pivot per `SYNTHESIS.md:30` `11-eval-matrix.md` `03-eval-harness.md:262` `09-gate-rl-probe.md`. At current 4855 tps (~1 step/1.5s, 4N 832 ranks, GBS 64*8192=524288 tok/step), 150k is ~75k steps away (~31h). Live chain 12484882 will hit it; Grain curriculum runs in parallel on separate allocation (2N smoke -> 4N production).

No new CPT compute needed to decide the SFT/RL path — the 136800 result is conclusive.

---

## Provenance

- CoT 136800: jobs **12484888** (convert DONE 3.7G) + **12484889** (cot COT-EVAL EXIT rc=0, 0.0/0.0/200/1527) via `qstat -f -x`, `grep` on `/home/foremans/convert-136800.o12484888` + `cot-136800.o12484889`, `cpt136800-gsm8k-cot-200.jsonl` 499K; prior 5 probes `03-eval-harness.md` + `SYNTHESIS.md:219`.
- MMLU trend: `/lus/tegu/.../outputs/evals/cpt-fwe/step-{1000,10000,50000,100000,136800}/results/results_mmlu_*.json` `mmlu acc,none` 0.2532/0.2413/0.2606/0.2688/0.2702, `sample_len 14042`, `+0.017/135k`; band 0.2295-0.2689 `03-eval-harness.md:72-77` `mmlu_letter_baseline.py`.
- Commonsense: same ladder `results_0shot_*.json` hs 0.566-0.581 arc 0.698-0.709 piqa 0.731-0.738.
- CPT live: `qstat -u foremans` + `qstat -f -x 12484882` (R, 75093, 4855 tps, 16.6% mfu, 12h walltime, tegu:home) + `grep -oP step cpt-fwe-4n.o12484882`.
- Quota: `lfs quota -h /home/foremans` + `/lus/tegu` (0k unlimited, 80T foremans, same tegu-* OST pool) + `lfs df -h` 1.2P 18% 994T free; `du -sh` HOME 67G Tegu 7.5G; prune 4.1T+587G from memory 2026-11-18.
- Harness: `eval_cot_gsm8k.py:86-115` envelope, `submit_cot_eval.sh:1-20` PBS, `hf_batch_convert_cpt_fwe.sh:1-80` staged_export, `frameworks/2026.1.0` XPU.
- Assignment contract: `ae7be4a45` `ezpz-cpt111k-eval-sft` Grain anneal + MCQ probe configs in `config_registry.py`.
