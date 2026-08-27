# B4 Cold-Start SFT Fix (finishing-stage + reweighted-mix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat B2's GSM8K CoT accuracy (0.205) after the B3 dilution regression (0.05), via two parallel paths: (A) a short gsm8k-r1cot finishing stage on the B3 base, and (B) a fresh reweighted single-stage SFT with length-filtered OpenR1.

**Architecture:** Path A adds one launcher (2nd-stage SFT over gsm8k-r1cot on the already-consolidated B3 checkpoint, save+eval per epoch). Path B adds an OpenR1 length-filter param + a new `b4_reweight_mix` to `datasets_sft.py`, a pretokenize script (@4096, filtered), and an 8N launcher. Both reuse `consolidate_and_eval_cot.sh`, extended with gen_len/unclosed-answer guardrail reporting. No core edits.

**Tech Stack:** Python 3.12, HF `datasets` + TRL `SFTTrainer` (existing `train_sft.py`), FSDP1, Intel XPU, PBS/`ezpz launch`, gemma chat template, fp32 vLLM eval.

## Global Constraints

- Only edit files under `torchtitan/experiments/ezpz/`. No core/upstream edits.
- All cluster work happens ON the cluster over `ssh -o ControlPath=/tmp/sunspot-master.sock sunspot`. Do NOT create new repo files with the local Write tool AND commit the same path on the cluster (untracked-twin blocks the user's git pull). Create+commit on the cluster only.
- Repo root on cluster: `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan`
- Use `backup` (`~/.local/bin/backup`), never `rm`, for anything not pure regenerable junk.
- Never run pre-commit as verification. PBS scripts must NOT use `set -euo pipefail` (`set -o pipefail` alone OK). Never `pip install` torch/deps.
- Conventional atomic commits (`type(scope): subject`); `git pull` (plain, not --rebase) before every push; stash-pull-pop if dirty; push after each commit batch.
- ASCII only in new/rewritten comments/docstrings.
- Use the repo `.venv/bin` python for datasets/AST unit tests (bare `python3` is 3.6, can't import torchtitan). `source .venv/bin/activate` first.
- Base for Path A: `outputs/evals/cot/b3-3605-hf-diag/` (already consolidated, 7.9G, complete HF). Base for Path B: `/home/foremans/global_step138650` (HF).
- Eval BASE config = `outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf`; TOK_SRC = `~/rl-repro/run/agpt2b-ckpt900`; eos [1,107].
- Data staging (Path B pretok): `/tegu/datasets/datasets/` (== `/lus/tegu/projects/datasets/datasets/`, datasets group).
- Compare all results on the shared 200-problem GSM8K CoT metric vs B2 0.205 / B3 0.05. Guardrail: gen_len (target ~282, not 601), unclosed-`</answer>` count (target ~0, not 26/200).

## File Structure

- `torchtitan/experiments/ezpz/rl/datasets_sft.py` (MODIFY) -- OpenR1 length-filter param + `b4_reweight_mix` registration.
- `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` (MODIFY) -- add mean_gen_len + n_unclosed to the summary JSON.
- `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4a_gsm8k_finish_2n.sh` (CREATE) -- Path A finishing stage.
- `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b4_reweight_mix_1n.sh` (CREATE) -- Path B pretokenize @4096.
- `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4b_reweight_8n.sh` (CREATE) -- Path B 8N run.
- `torchtitan/experiments/ezpz/docs/live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/README.md` (CREATE) -- results tracking.

Tests: unit tests for the OpenR1 length filter (pure function, no torch) and the mix registration + eval-summary shape; the 2N Path A run and Path B pretokenize are the integration checks. Controller submits + monitors all cluster jobs (subagents author scripts + run unit tests only).

---

### Task 1: OpenR1 length-filter param

**Files:**
- Modify: `torchtitan/experiments/ezpz/rl/datasets_sft.py` (`_openr1_format_row`, `_openr1_map_row`, `_build_openr1_math_cot`)
- Test: `/tmp/test_openr1_lenfilter.py` on cluster

**Interfaces:**
- Consumes: existing `_openr1_format_row(ex)` / `_openr1_map_row(ex)` / `_build_openr1_math_cot()` (verified current bodies below).
- Produces: a module-level `OPENR1_MAX_THINK_CHARS` (default read from env `OPENR1_MAX_THINK_CHARS`, fallback `1200`) that `_openr1_format_row` enforces: rows whose `<think>` trace exceeds it are dropped (return None). `_build_b4_reweight_mix` (Task 2) relies on this to remove run-on traces.

- [ ] **Step 1: Write the failing test** on the cluster (single-quoted heredoc to `/tmp/test_openr1_lenfilter.py`):

```python
import os, sys
sys.path.insert(0, "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
from torchtitan.experiments.ezpz.rl.datasets_sft import _openr1_format_row

def _row(trace, ans="5"):
    gen = "<think>" + trace + "</think> the answer is \\boxed{" + ans + "}"
    return {"problem": "Q?", "generations": [gen],
            "correctness_math_verify": [True], "is_reasoning_complete": [True], "answer": ans}

# short trace kept (default cutoff 1200)
os.environ.pop("OPENR1_MAX_THINK_CHARS", None)
short = _row("2+3=5")
out = _openr1_format_row(short)
assert out is not None and "<think>2+3=5</think>" in out["completion"][0]["content"], out
print("SHORT kept: PASS")

# long trace dropped (>1200 chars)
long_trace = "x" * 1300
assert _openr1_format_row(_row(long_trace)) is None, "long trace must be dropped"
print("LONG dropped: PASS")

# env override tightens the cutoff
os.environ["OPENR1_MAX_THINK_CHARS"] = "3"
import importlib, torchtitan.experiments.ezpz.rl.datasets_sft as m
importlib.reload(m)
assert m._openr1_format_row(_row("12345")) is None, "trace len 5 > cutoff 3 -> drop"
assert m._openr1_format_row(_row("ab")) is not None, "trace len 2 <= cutoff 3 -> keep"
print("ENV override: PASS")
print("ALL_OK")
```

- [ ] **Step 2: Run to verify it fails** -- `ssh ... 'cd <repo> && source .venv/bin/activate && python3 /tmp/test_openr1_lenfilter.py'`. Expected: FAIL (long trace currently kept -> AssertionError, since no filter exists yet).

- [ ] **Step 3: Add the filter** via a `/tmp/patch_lenfilter.py` on the cluster. Add near the other OpenR1 module constants (after `_OPENR1_SUFFIX`):

```python
# Max <think> trace length (chars) kept from OpenR1. The B3 regression traced to
# long-form R1 run-on traces (mean gen_len 601, 26/200 never closed </answer>)
# that taught a verbose style diluting a 2B's short-arithmetic competence. Drop
# traces above this so only short, useful reasoning is trained. Env-tunable.
OPENR1_MAX_THINK_CHARS = int(os.environ.get("OPENR1_MAX_THINK_CHARS", "1200"))
```

Ensure `import os` is at module scope (it is -- used elsewhere). Then in `_openr1_format_row`, right after `trace = tm.group(1).strip()`, add:

```python
        if len(trace) > OPENR1_MAX_THINK_CHARS:
            continue  # run-on trace: skip this generation (may fall through to None)
```

(Placing it inside the per-generation loop as `continue` -- not `return None` -- lets a shorter later generation for the same problem still qualify.)

- [ ] **Step 4: Run to verify it passes** -- same command. Expected: `ALL_OK`.

- [ ] **Step 5: AST check + commit**

```bash
ssh ... 'cd <repo> && source .venv/bin/activate && python3 -c "import ast; ast.parse(open(\"torchtitan/experiments/ezpz/rl/datasets_sft.py\").read()); print(\"AST OK\")"'
ssh ... 'cd <repo> && git add torchtitan/experiments/ezpz/rl/datasets_sft.py && git stash && git pull && git stash pop && git add torchtitan/experiments/ezpz/rl/datasets_sft.py && git commit -m "feat(ezpz/sft): OPENR1_MAX_THINK_CHARS length filter (drop run-on R1 traces)" && git push'
```

---

### Task 2: b4_reweight_mix registration

**Files:**
- Modify: `torchtitan/experiments/ezpz/rl/datasets_sft.py` (add after the `b3_instruct_cot_mix` registration)

**Interfaces:**
- Consumes: `_materialized_mix_load_or_build(component_names, weights, seed, caps=None)` (has a `caps` param from B3), `register_sft_dataset`, `SFTDataset`, and registry names `gsm8k-r1cot`, `OpenR1-Math-220k` (now length-filtered per Task 1), `tulu-3-sft-mixture`, `ultrachat-200k`.
- Produces: registry entry `"b4_reweight_mix"`.

- [ ] **Step 1: Write the failing test** (`/tmp/test_b4_mix.py`):

```python
import sys
sys.path.insert(0, "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
from torchtitan.experiments.ezpz.rl.datasets_sft import SFT_REGISTRY
assert "b4_reweight_mix" in SFT_REGISTRY, sorted(SFT_REGISTRY)
ds = SFT_REGISTRY["b4_reweight_mix"]
assert ds.name == "b4_reweight_mix" and callable(ds.build)
import inspect
src = inspect.getsource(ds.build)
for comp in ["gsm8k-r1cot", "OpenR1-Math-220k", "tulu-3-sft-mixture", "ultrachat-200k"]:
    assert comp in src, f"missing component {comp}"
assert "OpenMathInstruct-2" not in src, "OpenMathInstruct-2 should be DROPPED in b4"
print("B4_MIX_OK")
```

- [ ] **Step 2: Run to verify it fails** -- `AssertionError` (b4_reweight_mix not registered).

- [ ] **Step 3: Add the builder + registration** (`/tmp/patch_b4_mix.py`), after the `b3_instruct_cot_mix` `register_sft_dataset(...)` block:

```python
def _build_b4_reweight_mix(seed: int = 42):
    """B4 reweighted mix -- fixes the B3 dilution regression
    (docs/live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/design.md):
      0.40 gsm8k-r1cot        (was 0.15 in b3 -- restore in-distribution short CoT)
      0.15 OpenR1-Math-220k   (LENGTH-FILTERED via OPENR1_MAX_THINK_CHARS -- short
                               traces only; the run-on ones caused the regression)
      0.30 tulu-3-sft-mixture (general instruction-following)
      0.15 ultrachat-200k     (multi-turn chat)
    OpenMathInstruct-2 DROPPED (added math breadth the 2B could not convert to
    accuracy). Same materialized-mix cache + all_exhausted interleave as b3.
    """
    return _materialized_mix_load_or_build(
        component_names=[
            "gsm8k-r1cot",
            "OpenR1-Math-220k",
            "tulu-3-sft-mixture",
            "ultrachat-200k",
        ],
        weights=[0.40, 0.15, 0.30, 0.15],
        seed=seed,
    )


register_sft_dataset(
    SFTDataset(
        name="b4_reweight_mix",
        build=_build_b4_reweight_mix,
        description=(
            "B4 reweighted mix: 40%% gsm8k-r1cot + 15%% length-filtered "
            "OpenR1-Math-220k + 30%% tulu-3 + 15%% ultrachat-200k "
            "(OpenMathInstruct-2 dropped). Fixes the B3 long-CoT dilution."
        ),
    )
)
```

- [ ] **Step 4: Run to verify it passes** -- `B4_MIX_OK`.

- [ ] **Step 5: AST check + commit** (`feat(ezpz/sft): b4_reweight_mix (gsm8k-r1cot 0.40 + short OpenR1 + tulu + ultrachat)`).

---

### Task 3: eval gen_len + unclosed-answer guardrail reporting

**Files:**
- Modify: `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` (the summary JSON + per-row)

**Interfaces:**
- Consumes: the existing `outputs` loop (each `o` is a vLLM RequestOutput; `o.outputs[0].text`, `o.outputs[0].finish_reason`). Sampling already uses `stop=["</answer>"]`, so `finish_reason == "length"` means the generation hit max_tokens WITHOUT closing `</answer>` (an unclosed run-on).
- Produces: two new summary fields `mean_gen_len` and `n_unclosed`, plus `finish_reason` in each per-example row. These are the B3-lesson guardrail metrics; downstream READMEs report them.

- [ ] **Step 1: Write the failing test** (`/tmp/test_eval_summary.py`) -- a pure unit test of a small helper (no vLLM). Refactor the summary math into a testable function `summarize(texts, finish_reasons, fmts, corrects)` returning the dict, and assert:

```python
import sys
sys.path.insert(0, "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
from torchtitan.experiments.ezpz.scripts.eval.eval_cot_gsm8k import summarize
s = summarize(texts=["ab", "cdef"], finish_reasons=["stop", "length"],
              fmts=[True, False], corrects=[True, False])
assert s["n"] == 2
assert s["mean_gen_len"] == 3.0            # (2+4)/2
assert s["n_unclosed"] == 1                # one finish_reason == "length"
assert s["format_hit_rate"] == 0.5
assert s["cot_accuracy"] == 0.5
print("EVAL_SUMMARY_OK")
```

- [ ] **Step 2: Run to verify it fails** -- ImportError (no `summarize`).

- [ ] **Step 3: Refactor + add fields** (`/tmp/patch_eval.py`). Add a module-level `summarize(...)`:

```python
def summarize(texts, finish_reasons, fmts, corrects):
    n = len(texts)
    n_format = sum(int(x) for x in fmts)
    n_correct = sum(int(x) for x in corrects)
    n_unclosed = sum(1 for fr in finish_reasons if fr == "length")
    mean_gen_len = round(sum(len(t) for t in texts) / n, 1) if n else 0.0
    return {
        "n": n,
        "format_hit_rate": round(n_format / n, 4) if n else 0.0,
        "cot_accuracy": round(n_correct / n, 4) if n else 0.0,
        "mean_gen_len": mean_gen_len,
        "n_unclosed": n_unclosed,
    }
```

Then in `main()`, collect `finish_reasons.append(o.outputs[0].finish_reason)` in the loop, add `"finish_reason": o.outputs[0].finish_reason` to each row, and build the printed summary from `summarize(...)` merged with the static fields (`model`, `dtype`, `temperature`, `max_tokens`). Keep the existing keys so nothing downstream breaks.

- [ ] **Step 4: Run to verify it passes** -- `EVAL_SUMMARY_OK`.

- [ ] **Step 5: AST check + commit** (`feat(ezpz/eval): report mean_gen_len + n_unclosed in CoT eval (B3-lesson guardrail)`).

---

### Task 4: Path A launcher (finishing stage on B3 base, per-epoch ckpts)

**Files:**
- Create: `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4a_gsm8k_finish_2n.sh`

**Interfaces:**
- Consumes: base `outputs/evals/cot/b3-3605-hf-diag/`; `--sft_dataset gsm8k-r1cot` (inline tokenize, 7473 rows); `train_sft.py`.
- Produces: `outputs/sft/agpt2b-b4a-gsm8k-finish/checkpoint-{per-epoch}` -- 3 checkpoints (one per epoch) for the controller to eval.

Mirror the proven SFT launcher structure (e.g. `agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh` header/env/ezpz_setup_job/venv), scaled to 2N, with these specifics:

- [ ] **Step 1: Write the script** (heredoc writer). Key lines:

```bash
#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# B4a: short gsm8k-r1cot FINISHING stage on the B3 base (restores B2's winning
# in-distribution short-CoT final stage). Saves 1 ckpt/epoch (3 epochs) for
# per-epoch eval -- B4a's base is more capable than B2's so 3 epochs may overfit
# 7473 examples; per-epoch eval finds the sweet spot.
set -o pipefail
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_DATASETS_OFFLINE=0   # gsm8k is small + may need download; proxy on
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"; source .venv/bin/activate
BASE_MODEL="${BASE_MODEL:-$SUBMIT_DIR/outputs/evals/cot/b3-3605-hf-diag}"
CKPT_DIR="outputs/sft/agpt2b-b4a-gsm8k-finish"
LOG_DIR="logs/sft-b4a-gsm8k-finish-${PBS_JOBID%%.*}"; mkdir -p "${LOG_DIR}"
[[ -f "${BASE_MODEL}/model.safetensors" ]] || { echo "FATAL: B3 base missing ${BASE_MODEL}"; exit 1; }
ezpz launch --np 24 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset gsm8k-r1cot \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --bf16 --fsdp full_shard \
    --logging_steps 5 \
    --save_strategy epoch \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true
echo "=== DONE B4a: ckpts in ${CKPT_DIR} (one per epoch) ===" | tee -a "${LOG_DIR}/run.log"
```

Note: confirm `--save_strategy epoch` is accepted by EzpzSFTConfig/TRL (it is a standard HF TrainingArguments value; verify no ezpz override forces "steps"). If `epoch` is unsupported, fall back to `--save_strategy steps --save_steps <~1 epoch in steps>` (7473 rows / GBS -- compute from GBS=24*2*8=384 -> ~19 steps/epoch, so save_steps ~19). Verify before finalizing.

- [ ] **Step 2: `bash -n` syntax check** -> `SYNTAX_OK`.
- [ ] **Step 3: Commit** (`feat(ezpz/sft): B4a gsm8k-r1cot finishing-stage launcher on B3 base`).
- [ ] **Step 4 (controller):** submit; confirm base loads (no key errors), loss descends, 3 per-epoch checkpoints save. Then eval each (Task 6).
- [ ] **Step 5 (controller):** record per-epoch eval numbers in the README (Task 6).

---

### Task 5: Path B pretokenize (@4096, filtered mix) + 8N launcher

**Files:**
- Create: `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b4_reweight_mix_1n.sh`
- Create: `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4b_reweight_8n.sh`

**Interfaces:**
- Consumes: `b4_reweight_mix` (Task 2), `OPENR1_MAX_THINK_CHARS` (Task 1), base `~/global_step138650`.
- Produces: pretokenized dataset `/tegu/datasets/datasets/agpt2b-b4-reweight-len4096/` (consumed by the 8N launcher via `--pretokenized_dataset`); then `outputs/sft/agpt2b-b4b-reweight-8n/checkpoint-*`.

Mirror `_pretokenize_b3_instruct_cot_mix_1n.sh` (the proven B3 pretokenize) with deltas: `--sft_dataset b4_reweight_mix`, `--max_length 4096`, `OUT_DIR=/tegu/datasets/datasets/agpt2b-b4-reweight-len4096`, keep `HF_HOME`/`HF_DATASETS_CACHE`/`EZPZ_SFT_MIX_CACHE_DIR` on `/tegu/datasets`, `export OPENR1_MAX_THINK_CHARS=1200` (so the filter is active during the mix build), `NPROC=${NPROC:-32}`, proxy on, 12h walltime (filtered/4096 mix is far smaller than B3 -> expect ~1-2h). The 8N launcher mirrors `agpt2b_b3_instruct_cot_mix_8n.sh` with the new pretok path, `--max_length 4096`, resume + afterany-chainable, no `--auto-retry`.

- [ ] **Step 1: Write both scripts** (heredoc writers). `bash -n` both.
- [ ] **Step 2: Commit** (`feat(ezpz/sft): B4b pretokenize (@4096 filtered mix) + 8N launcher`).
- [ ] **Step 3 (controller):** submit pretokenize; verify it builds the mix (report OpenR1 filter drop-rate + total rows -- expect fewer than B3's 13.3M since OpenMath dropped + OpenR1 filtered), packs @4096, saves `dataset_info.json`, reports size.
- [ ] **Step 4 (controller):** after pretok lands, submit the 8N run (+ afterany continuation). Monitor loss + checkpoints.
- [ ] **Step 5 (controller):** record job ids + status in README.

---

### Task 6: Eval all B4 checkpoints + document

**Files:**
- Create: `torchtitan/experiments/ezpz/docs/live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/README.md`

**Interfaces:**
- Consumes: B4a per-epoch checkpoints (Task 4), B4b final checkpoint (Task 5), `consolidate_and_eval_cot.sh` (+ Task 3's gen_len/unclosed reporting).
- Produces: the comparison verdict (B2 0.205 / B3 0.05 / B4a-ep{1,2,3} / B4b) on the shared metric + guardrails.

- [ ] **Step 1 (controller):** eval each B4a per-epoch checkpoint via `consolidate_and_eval_cot.sh` (CKPT=<...>/checkpoint-N, HF_OUT=/tmp, BASE=729-hf, TOK_SRC). Record cot_accuracy, format_hit_rate, mean_gen_len, n_unclosed for each.
- [ ] **Step 2 (controller):** eval the B4b final (and a mid checkpoint) the same way.
- [ ] **Step 3:** write the README: the B3-regression diagnosis summary, both B4 recipes, per-checkpoint eval table (acc/format/gen_len/unclosed vs B2 0.205 / B3 0.05), and the verdict -- did either path beat 0.205? which structure won? Cross-link from `docs/live/chains/rl/plans/cot.md` + `docs/live/chains/sft/README.md`.
- [ ] **Step 4:** commit README + append the B4 result to cot.md (`docs(ezpz/sft): B4 results -- <verdict> vs B2 0.205`).

---

## Self-Review

**Spec coverage:** finishing-stage (Task 4) / reweighted mix (Task 2) / OpenR1 length filter (Task 1) / gen_len+unclosed guardrail (Task 3) / @4096 pretok + 8N (Task 5) / eval + doc (Task 6) -- all design sections covered.

**Placeholder scan:** no TBD/TODO; every code step shows real code. The `--save_strategy epoch` verification (Task 4) and the OpenR1 filter drop-rate (Task 5) are explicit pre-submit/report checks with stated fallbacks, not placeholders.

**Type/name consistency:** `OPENR1_MAX_THINK_CHARS` (Task 1) -> used by `_build_b4_reweight_mix` via the OpenR1 component (Task 2) + set in the Path B pretok env (Task 5). `b4_reweight_mix` (Task 2) -> `--sft_dataset` in Task 5. `summarize(...)` (Task 3) -> keys `mean_gen_len`/`n_unclosed` reported in Task 6. Pretok path `/tegu/datasets/datasets/agpt2b-b4-reweight-len4096` written in Task 5, consumed same task. B3 base path `outputs/evals/cot/b3-3605-hf-diag` (Task 4) verified on disk. Consistent.

**Open risks carried into execution:** (1) `--save_strategy epoch` support (Task 4 verifies, has save_steps fallback). (2) OpenR1 filter drop-rate at 1200 chars unknown (Task 5 reports; lower weight not cutoff if too aggressive). (3) 2B ceiling -- if neither path clears ~0.22-0.25 the lever is a bigger base, decided after numbers land (design risk section).
