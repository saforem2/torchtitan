# B3 Cold-Start SFT (instruct + CoT rebuild) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the agpt-2b CoT cold-start as ONE TRL SFT from the stage-3 pretrain base (`gs138650`) on a balanced instruction + chain-of-thought data mix, then eval it on the 200-problem GSM8K CoT metric vs B2 (0.205).

**Architecture:** Add one new dataset loader (`OpenR1-Math-220k`, reformatted to the `<think>/<answer>\boxed{}` envelope) and one named mix (`b3_instruct_cot_mix`) to the existing `datasets_sft.py`. Stage all data offline (download + pretokenize @ 8192) to `/tegu/datasets/datasets/` on an interactive/1N node with proxy. Run the SFT at <=8N reading the pretokenized copy offline. Consolidate and eval with the existing CoT eval tooling. No trainer or core changes.

**Tech Stack:** Python 3.12, HF `datasets` + TRL `SFTTrainer` (existing `train_sft.py`), FSDP1, Intel XPU (no flash-attn), PBS/`ezpz launch`, gemma chat template.

## Global Constraints

- Only edit files under `torchtitan/experiments/ezpz/`. No core/upstream edits. (CLAUDE.md)
- All cluster work happens ON the cluster over the ssh ControlPath (`ssh -o ControlPath=/tmp/sunspot-master.sock sunspot`). Do NOT create new repo files with the local Write tool AND commit the same path on the cluster (leaves an untracked twin that blocks the user's git pull). Create+commit on the cluster only.
- Repo root on cluster: `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan`
- Use `backup` (`~/.local/bin/backup`), never `rm`, for anything not pure regenerable junk.
- Never run pre-commit as verification (user handles linting).
- PBS scripts must NOT use `set -euo pipefail` (venv activate has unbound vars). `set -o pipefail` alone is OK (the existing pretokenize script uses it).
- Never `pip install` torch or its deps.
- Conventional atomic commits (`type(scope): subject`); one logical change per commit. `git pull` (plain, not --rebase) before every push; stash-pull-pop if dirty. Push after each commit batch.
- ASCII only in new/rewritten comments and docstrings (`->` not arrows, `--` not em-dash).
- Data staging target: `/tegu/datasets/datasets/` == real path `/lus/tegu/projects/datasets/datasets/` (datasets group, separate quota; already holds `fineweb-edu-100BT`).
- Base model: `/home/foremans/global_step138650` (HF format, staged). Pretrain seq_len was 8192, so `max_length=8192` is positionally safe.
- `max_length=8192`, 1 epoch, LR 2e-5, gemma template, `assistant_only_loss=True`, packing, bf16, FSDP full_shard. Scale <=8N (gs138650 lineage is pinned <=8N).
- Eval metric = 200-problem GSM8K CoT via `scripts/eval/eval_cot_gsm8k.py`; compare to B2 acc 0.205 / format 0.985. Success = accuracy up, format >= 0.95.

## File Structure

- `torchtitan/experiments/ezpz/rl/datasets_sft.py` (MODIFY) -- add `_build_openr1_math_cot` loader + `OpenR1-Math-220k` registration + `_build_b3_instruct_cot_mix` + `b3_instruct_cot_mix` registration.
- `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b3_instruct_cot_mix_1n.sh` (CREATE) -- 1N offline download + pretokenize @ 8192 to `/tegu/datasets/datasets/`.
- `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b3_instruct_cot_mix_smoke_2n.sh` (CREATE) -- 2N smoke: batch-fit @ 8192, format learns, loss sane.
- `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b3_instruct_cot_mix_8n.sh` (CREATE) -- 8N full run reading the pretokenized copy offline.
- `torchtitan/experiments/ezpz/rl/scripts/eval/submit_b3_eval.sh` (CREATE, or reuse `scripts/eval/submit_cot_eval.sh`) -- consolidate + 200-problem eval.
- `torchtitan/experiments/ezpz/docs/live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/README.md` (CREATE -- not yet written)  <!-- docs-link-check: ignore --> -- run tracking per ezpz golden rule #2.

Tests: this is a data-pipeline + training project, not a unit-testable library. The "tests" are (a) a small local dry-run of the OpenR1 loader on a handful of rows asserting the envelope + filtering, and (b) the 2N smoke as the integration test. Each is an explicit task step below.

---

### Task 1: OpenR1-Math-220k loader (envelope reformat + correctness filter)

**Files:**
- Modify: `torchtitan/experiments/ezpz/rl/datasets_sft.py` (add loader + registration after the `gsm8k-r1cot` block, ~line 445)
- Test: a standalone dry-run script `/tmp/test_openr1_loader.py` on the cluster (5-row assertion; not committed)

**Interfaces:**
- Consumes: `SFTDataset`, `register_sft_dataset`, the `{"prompt":[...],"completion":[...]}` row shape from `_build_gsm8k_r1cot`.
- Produces: registry entry `"OpenR1-Math-220k"` -> a `Dataset` of `{prompt:[{role:user}], completion:[{role:assistant, content:"<think>...</think>\n<answer>\\boxed{ans}</answer>"}]}` rows.

**OpenR1 schema (verified on cluster):** columns include `problem` (str), `generations` (list[str], each already wrapped in `<think>...</think>` and typically ending with `\boxed{...}`), `answer` (str, messy), `correctness_math_verify` (list[bool], parallel to `generations`), `is_reasoning_complete` (list[bool]).

- [ ] **Step 1: Write the failing dry-run test on the cluster**

Write `/tmp/test_openr1_loader.py` (via a single-quoted heredoc over ssh so no quote-mangling):

```python
import re, sys
sys.path.insert(0, "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
# Import ONLY the pure formatting helper (no torch/datasets import needed for the unit check).
from torchtitan.experiments.ezpz.rl.datasets_sft import _openr1_format_row

# A representative OpenR1 row (mirrors the verified schema).
row = {
    "problem": "What is 2+3?",
    "generations": [
        "<think>\nWe add 2 and 3 to get 5.\n</think>\n\nThe answer is \\boxed{5}.",
        "<think>bad</think> no box here",
    ],
    "correctness_math_verify": [True, False],
    "is_reasoning_complete": [True, True],
    "answer": "5",
}
out = _openr1_format_row(row)
assert out is not None, "should pick the correct+boxed generation"
comp = out["completion"][0]["content"]
assert comp.startswith("<think>"), comp
assert "</think>\n<answer>\\boxed{5}</answer>" in comp, comp
assert out["prompt"][0]["role"] == "user"
assert "What is 2+3?" in out["prompt"][0]["content"]

# A row with NO correct generation -> filtered (returns None).
row_bad = {"problem": "x", "generations": ["<think>y</think> nobox"],
           "correctness_math_verify": [False], "is_reasoning_complete": [True], "answer": "1"}
assert _openr1_format_row(row_bad) is None, "no correct gen -> drop"

# A row whose correct generation has no \boxed{} -> filtered.
row_nobox = {"problem": "x", "generations": ["<think>z</think> the answer is 7"],
             "correctness_math_verify": [True], "is_reasoning_complete": [True], "answer": "7"}
assert _openr1_format_row(row_nobox) is None, "correct but no boxed -> drop"
print("OPENR1_LOADER_OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ssh -o ControlPath=/tmp/sunspot-master.sock sunspot 'cd <repo> && python3 /tmp/test_openr1_loader.py'`
Expected: FAIL with `ImportError: cannot import name '_openr1_format_row'` (function not written yet).

- [ ] **Step 3: Write the loader + helper in datasets_sft.py**

Add (via a `/tmp/patch_openr1.py` script run on the cluster, inserting after the `gsm8k-r1cot` registration block). The pure helper is separated out so it is unit-testable without torch:

```python
# OpenR1-Math-220k -- rich R1-distilled long-form CoT, reformatted into our
# <think>/<answer>\boxed{} envelope (matches gsm8k-r1cot + the eval).
_OPENR1_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_OPENR1_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPENR1_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)


def _openr1_format_row(ex):
    """Pick a verified-correct generation, extract its <think> trace + boxed
    answer, and reformat to our envelope. Returns a formatted row dict, or
    None if no correct generation with an extractable boxed answer exists
    (caller drops None rows so we never train a malformed envelope).

    OpenR1 generations already carry <think>...</think>; we keep that trace
    verbatim and normalize the tail to a single <answer>\\boxed{}</answer>.
    """
    gens = ex.get("generations") or []
    correct = ex.get("correctness_math_verify") or []
    complete = ex.get("is_reasoning_complete") or []
    for i, gen in enumerate(gens):
        if i < len(correct) and not correct[i]:
            continue
        if i < len(complete) and not complete[i]:
            continue
        if not gen:
            continue
        tm = _OPENR1_THINK_RE.search(gen)
        if not tm:
            continue
        trace = tm.group(1).strip()
        # boxed answer: prefer the LAST \boxed{} anywhere in the generation
        boxes = _OPENR1_BOXED_RE.findall(gen)
        if not boxes:
            continue
        ans = boxes[-1].strip()
        if not trace or not ans:
            continue
        completion = f"<think>{trace}</think>\n<answer>\\boxed{{{ans}}}</answer>"
        return {
            "prompt": [{"role": "user", "content": ex["problem"] + _OPENR1_SUFFIX}],
            "completion": [{"role": "assistant", "content": completion}],
        }
    return None


def _build_openr1_math_cot():
    """open-r1/OpenR1-Math-220k reformatted to the <think>/<answer> envelope.

    Selects a verified-correct (correctness_math_verify) + complete generation
    per problem, keeps its R1 reasoning trace, and normalizes the final answer
    to <answer>\\boxed{ans}</answer>. Rows with no correct+boxed generation are
    dropped (map returns the sentinel, then filter removes it). Needs the HF
    download cached first (see _pretokenize_b3_instruct_cot_mix_1n.sh).
    """
    from datasets import load_dataset

    raw = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    cols = raw.column_names

    def _map(ex):
        out = _openr1_format_row(ex)
        if out is None:
            # sentinel: empty prompt/completion -> dropped by the filter below
            return {"prompt": [], "completion": []}
        return out

    mapped = raw.map(_map, remove_columns=cols)
    return mapped.filter(lambda ex: bool(ex["prompt"]) and bool(ex["completion"]))


register_sft_dataset(
    SFTDataset(
        name="OpenR1-Math-220k",
        build=_build_openr1_math_cot,
        description=(
            "open-r1/OpenR1-Math-220k R1-distilled long-form CoT, reformatted "
            "into the <think>/<answer>\\boxed{} envelope. Selects a verified- "
            "correct generation per problem; rows without a correct+boxed "
            "generation are dropped. The rich-reasoning half of b3_instruct_cot_mix."
        ),
    )
)
```

Ensure `import re` is available at module scope (it is used elsewhere in the file; if the regexes are at module top, add `import re` at the top of the file if not already present -- check first).

- [ ] **Step 4: Run the dry-run test to verify it passes**

Run: `ssh ... 'cd <repo> && python3 /tmp/test_openr1_loader.py'`
Expected: `OPENR1_LOADER_OK`

- [ ] **Step 5: AST check + commit**

```bash
ssh ... 'cd <repo> && python3 -c "import ast; ast.parse(open(\"torchtitan/experiments/ezpz/rl/datasets_sft.py\").read()); print(\"AST OK\")"'
ssh ... 'cd <repo> && git add torchtitan/experiments/ezpz/rl/datasets_sft.py && git stash && git pull && git stash pop && git add torchtitan/experiments/ezpz/rl/datasets_sft.py && git commit -m "feat(ezpz/sft): OpenR1-Math-220k loader (envelope reformat + correctness filter)" && git push'
```

---

### Task 2: b3_instruct_cot_mix named mix

**Files:**
- Modify: `torchtitan/experiments/ezpz/rl/datasets_sft.py` (add after the `tulu_math_uc_mix` registration, ~line 775)

**Interfaces:**
- Consumes: `_materialized_mix_load_or_build(component_names, weights, seed, stopping_strategy)`, `register_sft_dataset`, `SFTDataset`, and the 5 component registry names: `tulu-3-sft-mixture`, `OpenR1-Math-220k` (Task 1), `gsm8k-r1cot`, `ultrachat-200k`, `OpenMathInstruct-2`.
- Produces: registry entry `"b3_instruct_cot_mix"`.

- [ ] **Step 1: Write the failing test**

Write `/tmp/test_b3_mix.py` on the cluster:

```python
import sys
sys.path.insert(0, "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
from torchtitan.experiments.ezpz.rl.datasets_sft import SFT_REGISTRY
assert "b3_instruct_cot_mix" in SFT_REGISTRY, sorted(SFT_REGISTRY)
ds = SFT_REGISTRY["b3_instruct_cot_mix"]
assert ds.name == "b3_instruct_cot_mix"
assert callable(ds.build)
print("B3_MIX_REGISTERED_OK")
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh ... 'cd <repo> && python3 /tmp/test_b3_mix.py'`
Expected: `AssertionError` (b3_instruct_cot_mix not in registry).

- [ ] **Step 3: Add the mix builder + registration**

Via `/tmp/patch_b3_mix.py` on the cluster, insert after the `tulu_math_uc_mix` registration:

```python
def _build_b3_instruct_cot_mix(seed: int = 42):
    """Balanced instruction + CoT SFT mix for the B3 cold-start rebuild
    (docs/live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/design.md):
      0.30 tulu-3-sft-mixture (general instruction-following)
      0.25 OpenR1-Math-220k   (rich long-form R1 CoT, our envelope)
      0.15 gsm8k-r1cot        (in-distribution CoT, eval-matching format)
      0.15 ultrachat-200k     (multi-turn chat)
      0.15 OpenMathInstruct-2 (math breadth)
    45%% instruction/chat, 55%% math/CoT. Folds the CoT envelope INTO one SFT
    (vs B2's two-stage tulu-math -> gsm8k-r1cot lineage). Uses the same
    materialized-mix cache + all_exhausted interleave as tulu_math_uc_mix.
    """
    return _materialized_mix_load_or_build(
        component_names=[
            "tulu-3-sft-mixture",
            "OpenR1-Math-220k",
            "gsm8k-r1cot",
            "ultrachat-200k",
            "OpenMathInstruct-2",
        ],
        weights=[0.30, 0.25, 0.15, 0.15, 0.15],
        seed=seed,
    )


register_sft_dataset(
    SFTDataset(
        name="b3_instruct_cot_mix",
        build=_build_b3_instruct_cot_mix,
        description=(
            "B3 cold-start mix: 30%% tulu-3 + 25%% OpenR1-Math-220k (CoT) + "
            "15%% gsm8k-r1cot + 15%% ultrachat-200k + 15%% OpenMathInstruct-2. "
            "Combined instruction+CoT rebuild from gs138650."
        ),
    )
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `ssh ... 'cd <repo> && python3 /tmp/test_b3_mix.py'`
Expected: `B3_MIX_REGISTERED_OK`

- [ ] **Step 5: AST check + commit**

```bash
ssh ... 'cd <repo> && python3 -c "import ast; ast.parse(open(\"torchtitan/experiments/ezpz/rl/datasets_sft.py\").read()); print(\"AST OK\")"'
ssh ... 'cd <repo> && git add ...datasets_sft.py && git stash && git pull && git stash pop && git add ...datasets_sft.py && git commit -m "feat(ezpz/sft): b3_instruct_cot_mix (balanced instruct+CoT mix for B3)" && git push'
```

---

### Task 3: Offline download + pretokenize script (1N, @ 8192, to /tegu/datasets)

**Files:**
- Create: `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b3_instruct_cot_mix_1n.sh`

**Interfaces:**
- Consumes: `train_sft.py --sft_dataset b3_instruct_cot_mix --pretokenize_to <dir>` (build-only path); `HF_HOME`/`HF_DATASETS_CACHE`; base tokenizer at `~/global_step138650`.
- Produces: pretokenized dataset at `/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192/` (input_ids + assistant_masks + seq_lengths) consumed by Tasks 4 and 5 via `--pretokenized_dataset`.

Mirror `_pretokenize_tulu_math_uc_mix_1n.sh` exactly, with these deltas: `--sft_dataset b3_instruct_cot_mix`, `--max_length 8192`, output under `/tegu/datasets/datasets/`, `HF_HOME`/`HF_DATASETS_CACHE=/tegu/datasets/datasets/hf` (so the raw HF downloads land in the shared staging area), proxy ON (download needs it), `dataset_num_proc=32` (the hard-won memory lesson: 96/64 SIGBUS the packing map; 32 is the safe default), 12h walltime.

- [ ] **Step 1: Write the script on the cluster**

Create the file via a single-quoted heredoc over ssh (writer script to `/tmp/write_pretok.sh`, then run it). Content:

```bash
#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N offline download + PRE-TOKENIZE of b3_instruct_cot_mix @ 8192 for the B3 SFT.
# Downloads the 5 HF datasets to the shared /tegu/datasets staging area (proxy on),
# builds the OpenR1 envelope + interleaved mix, and runs TRL tokenize+pack ONCE via
# --pretokenize_to so the 8N job loads it with --pretokenized_dataset and starts
# training in <60s (avoids 96 ranks racing runtime tokenize -- the known SIGTERM
# bottleneck). num_proc=32: 96/64 spike the packing map to ~26TB vmem and SIGBUS;
# 32 is the safe EzpzSFTConfig default. Runs on ONE rank, no collective.
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# Shared dataset staging area (datasets group, separate quota).
export HF_HOME=/tegu/datasets/datasets/hf
export HF_DATASETS_CACHE=/tegu/datasets/datasets/hf/datasets
# mix cache also under the shared area so the 8N job finds the interleave build
export EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

BASE_MODEL="${HOME}/global_step138650"
OUT_DIR="/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192"
LOG_DIR="logs/pretokenize-b3-instruct-cot-mix-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 1N DOWNLOAD+PRETOKENIZE: b3_instruct_cot_mix -> ${OUT_DIR} (len=8192, base=gs138650) ===" \
    | tee -a "${LOG_DIR}/run.log"

python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset b3_instruct_cot_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --pretokenize_to "${OUT_DIR}" \
    --max_length 8192 \
    --dataset_num_proc 32 \
    --output_dir "${LOG_DIR}/_pretok_scratch" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "=== DONE: pretokenized dataset at ${OUT_DIR} ===" | tee -a "${LOG_DIR}/run.log"
du -sh "${OUT_DIR}" 2>/dev/null | tee -a "${LOG_DIR}/run.log"
```

Note: verify `train_sft.py` accepts `--dataset_num_proc` and `--pretokenize_to` with `--output_dir` still required; if `--pretokenize_to` short-circuits before training, `--output_dir` may still be a required arg (set it to a scratch path as above). Check the arg definitions before finalizing.

- [ ] **Step 2: Syntax check**

Run: `ssh ... 'cd <repo> && bash -n torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b3_instruct_cot_mix_1n.sh && echo SYNTAX_OK'`
Expected: `SYNTAX_OK`

- [ ] **Step 3: Commit the script**

```bash
ssh ... 'cd <repo> && git add torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b3_instruct_cot_mix_1n.sh && git stash && git pull && git stash pop && git add ...  && git commit -m "feat(ezpz/sft): 1N offline download+pretokenize for b3_instruct_cot_mix @ 8192" && git push'
```

- [ ] **Step 4: Submit the pretokenize job**

Run: `ssh ... 'cd <repo> && PATH=/opt/pbs/bin:$PATH qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_b3_instruct_cot_mix_1n.sh'`
Expected: a job id. This is a long job (download + tokenize the multi-million-row mix @ 8192; could be several hours). Monitor for: OpenR1 download completes, filter drop-rate is reported (sanity: should keep a large fraction), tokenize+pack completes, `du -sh` of the output prints.

- [ ] **Step 5: Verify the pretokenized output exists + note its size**

Run: `ssh ... 'ls -la /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192/ && du -sh /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192/'`
Expected: a `dataset_info.json` + arrow shards present; size reported (flag if it threatens the datasets-group quota).

---

### Task 4: 2N smoke launcher (batch-fit @ 8192 + format-learning sanity)

**Files:**
- Create: `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b3_instruct_cot_mix_smoke_2n.sh`

**Interfaces:**
- Consumes: the pretokenized dataset from Task 3 (`--pretokenized_dataset /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192`); base `~/global_step138650`.
- Produces: a short checkpoint under `outputs/sft/agpt2b-b3-instruct-cot-mix-smoke-2n/` proving the config fits at 8192 and loss descends.

Mirror the 8N tulu-math launcher's `ezpz launch ... train_sft.py` block, scaled to 2N (`--np 24 -ppn 12`), with `--max_length 8192`, a SMALL micro-batch (start `--per_device_train_batch_size 1` since 8192 is 8x the proven-fitting 1024 token budget), `--max_train_samples` to cap it short (e.g. 4000) so it is a real smoke, `--save_steps 25`, offline (`HF_HUB_OFFLINE=1`), `EZPZ_SFT_MIX_CACHE_DIR` + pretokenized path pointing at /tegu/datasets.

- [ ] **Step 1: Write the smoke script on the cluster** (heredoc writer -> run). Key lines:

```bash
#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# 2N smoke: B3 mix @ 8192 -- prove the batch config FITS on XPU (no flash-attn)
# and loss descends before the 8N run. Caps to a few thousand samples.
set -o pipefail
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HUB_OFFLINE=1
export HF_HOME=/tegu/datasets/datasets/hf
export EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"; source .venv/bin/activate
BASE_MODEL="${HOME}/global_step138650"
PRETOK_DIR="/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192"
CKPT_DIR="outputs/sft/agpt2b-b3-instruct-cot-mix-smoke-2n"
LOG_DIR="logs/b3-smoke-2n-${PBS_JOBID%%.*}"; mkdir -p "${LOG_DIR}"
[[ -d "${PRETOK_DIR}" ]] || { echo "FATAL: pretokenized dir missing ${PRETOK_DIR}"; exit 1; }
ezpz launch --np 24 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_train_samples 4000 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 8192 \
    --bf16 --fsdp full_shard \
    --logging_steps 5 \
    --save_strategy steps --save_steps 25 \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true
echo "=== DONE smoke: ${CKPT_DIR}, log ${LOG_DIR} ===" | tee -a "${LOG_DIR}/run.log"
```

Note: confirm `--max_train_samples` is a real `train_sft.py` arg (the pretokenize doc mentions it was used in the 2N smoke). If the pretokenized path ignores `--max_train_samples` (because prep is skipped), instead cap via a smaller pretokenized subset OR accept a short run bounded by walltime + `--max_steps` if supported. Verify before submit.

- [ ] **Step 2: Syntax check** -- `bash -n ... && echo SYNTAX_OK` -> `SYNTAX_OK`

- [ ] **Step 3: Commit** -- `git ... commit -m "feat(ezpz/sft): 2N B3 smoke launcher (@8192 batch-fit sanity)" && git push`

- [ ] **Step 4: Submit + monitor the smoke**

Run: `ssh ... 'cd <repo> && PATH=/opt/pbs/bin:$PATH qsub ...smoke_2n.sh'`
Watch for: (a) NO OOM / OUT_OF_RESOURCES at `micro_batch=1 @ 8192` (the primary risk -- if it OOMs, the fallback is documented: drop to 4096 or length-bucket, re-pretokenize); (b) loss descends over the logged steps; (c) a checkpoint saves at step 25. Success = fits + loss down.

- [ ] **Step 5: Record the smoke result** in the README (Task 6). If OOM: note it, and STOP -- do not launch 8N; revisit length/batch with the user.

---

### Task 5: 8N full-run launcher

**Files:**
- Create: `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b3_instruct_cot_mix_8n.sh`

**Interfaces:**
- Consumes: the pretokenized dataset (Task 3); the batch config CONFIRMED by the Task 4 smoke; base `~/global_step138650`.
- Produces: the B3 checkpoint under `outputs/sft/agpt2b-b3-instruct-cot-mix-8n/` (the deliverable to eval).

Mirror the tulu-math 8N launcher exactly (`--np 96 -ppn 12`, `--resume_from_checkpoint`, `--save_steps 50`, afterany-chainable, NO `--auto-retry` per the tulu-math script's note), swapping in the B3 pretokenized dir, `--max_length 8192`, and the smoke-confirmed `--per_device_train_batch_size` / `--gradient_accumulation_steps` (choose GAS so effective GBS is comparable to the tulu-math run's 6144; with mb=1 @ 8192 across 96 ranks, GBS = 96 * 1 * GAS -- pick GAS to land near 6144-equivalent tokens/step given the 8x longer sequences, or match the smoke's proven fit).

- [ ] **Step 1: Write the 8N script on the cluster** (heredoc writer). Base it on the confirmed smoke config; set `#PBS -l select=8`, `--np 96 -ppn 12`, `--save_strategy steps --save_steps 50`, `--resume_from_checkpoint "${CKPT_DIR}"`, `--num_train_epochs 1`, `--max_length 8192`, offline env + /tegu paths as in Task 4.

- [ ] **Step 2: Syntax check** -- `bash -n ... && echo SYNTAX_OK`

- [ ] **Step 3: Commit** -- `git ... commit -m "feat(ezpz/sft): 8N B3 full-run launcher" && git push`

- [ ] **Step 4: Submit the 8N run (only after the smoke is green)**

Run: `ssh ... 'cd <repo> && PATH=/opt/pbs/bin:$PATH qsub ...8n.sh'`. Chain a continuation with `qsub -W depend=afterany:<jobid> ...8n.sh` if 1 epoch exceeds the walltime (the tulu-math full mix was ~54B tokens / multi-link; B3 @ 8192 is longer per-seq -- size the chain accordingly). Monitor loss + periodic checkpoints.

- [ ] **Step 5: Record job id + status** in the README (Task 6).

---

### Task 6: Consolidate + eval + document B3

**Files:**
- Create: `torchtitan/experiments/ezpz/rl/scripts/eval/submit_b3_eval.sh` (or reuse `scripts/eval/submit_cot_eval.sh` if it accepts an FSDP-checkpoint consolidation path)
- Create: `torchtitan/experiments/ezpz/docs/live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/README.md`  <!-- docs-link-check: ignore -->

**Interfaces:**
- Consumes: a B3 checkpoint from Task 5 (`outputs/sft/agpt2b-b3-instruct-cot-mix-8n/checkpoint-N`); `scripts/eval/eval_cot_gsm8k.py`; `accelerate merge-weights` (FSDP->HF); `scripts/eval/fix_ckpt_eos.py`.
- Produces: `cot_accuracy` + `format_hit_rate` on 200 GSM8K problems, compared to B2 (0.205 / 0.985).

- [ ] **Step 1: Write the eval wrapper** (mirror `consolidate_and_eval_cot.sh`: FSDP `checkpoint-N` -> HF via `accelerate merge-weights`, copy config/tokenizer from `checkpoint-729-hf` + `fix_ckpt_eos [1,107]`, then `eval_cot_gsm8k.py --limit 200 --dtype float32`). Reuse the `HF_OUT` override so consolidation can target node-local /tmp if quota is tight.

- [ ] **Step 2: Syntax check + commit** the wrapper.

- [ ] **Step 3: Submit the eval** on the best/last B3 checkpoint. Expected output: a JSON with `n: 200, format_hit_rate, cot_accuracy`.

- [ ] **Step 4: Write the README** documenting: base (gs138650), mix + weights, max_length 8192, pretokenize job id + dataset size + OpenR1 filter drop-rate, smoke result (fit/loss), 8N job id(s) + final loss, and the eval verdict (B3 acc/format vs B2 0.205/0.985). Cross-link from `docs/live/chains/rl/plans/cot.md` and `docs/live/chains/sft/README.md`.

- [ ] **Step 5: Commit the README + update cot.md** with the B3 result.

```bash
ssh ... 'cd <repo> && git add <README> <cot.md> && git stash && git pull && git stash pop && git add ... && git commit -m "docs(ezpz/sft): B3 run + eval verdict vs B2" && git push'
```

---

## Self-Review

**Spec coverage:** base gs138650 (Tasks 3-5) / balanced mix (Task 2) / OpenR1 envelope reformat (Task 1) / max_length 8192 (Tasks 3-5) / offline staging to /tegu/datasets (Task 3) / <=8N scale (Tasks 4-5) / 200-problem eval vs B2 (Task 6) / README tracking (Task 6) -- all covered.

**Placeholder scan:** no TBD/TODO; every code step shows real code; the two "verify the arg exists" notes (Tasks 3, 4) are explicit pre-submit checks against `train_sft.py`, not placeholders -- the implementer confirms `--dataset_num_proc` / `--max_train_samples` / `--pretokenize_to`+`--output_dir` interaction before finalizing, with a stated fallback.

**Type/name consistency:** `_openr1_format_row` (Task 1 helper, unit-tested) -> used by `_build_openr1_math_cot` -> registered `OpenR1-Math-220k` -> consumed by name in `_build_b3_instruct_cot_mix` (Task 2). Pretokenized dir `/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192` is written in Task 3 and consumed verbatim in Tasks 4-5. Row shape `{prompt:[...],completion:[...]}` matches `_build_gsm8k_r1cot`. Consistent.

**Open risks carried into execution:** (1) 8192 batch-fit on XPU is unproven -> Task 4 smoke is the gate before Task 5; fallback (4096 / length-bucket) documented. (2) OpenR1 filter drop-rate unknown -> reported in Task 3. (3) `--max_train_samples` behavior on the pretokenized path -> verified in Task 4.
