#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-20b-v2
#PBS -j oe
#
# Convert + eval the v2 20B SophiaG checkpoints (fp32 master) and
# produce results that can be directly compared against the v1
# bf16-tainted eval table in docs/evals/agpt/20b/README.md.
#
# v2 ckpt path:
#   /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/
#     outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288/step-{N}
#
# Output (in this clone, under outputs/evals/agpt-20b-v2/):
#   step-100/hf/      HF safetensors + tokenizer
#   step-100/results/ lm-eval JSON
#
# Steps to evaluate are passed via STEPS env var (space-separated). At
# write time only step-100 + step-200 are saved; later runs of this
# script will pick up newer ckpts as they land.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md — venv
# activate has unbound vars and would trigger on first source.
set -o pipefail

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_ENABLE_HF_TRANSFER=0

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1
echo "PWD: $(pwd)"
echo "Modules loaded."

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

# This eval pipeline runs against the bare frameworks/2025.3.1 module
# stack (NOT the user venv) per CLAUDE.md — user venv has transformers
# 5.6.2 which breaks lm-eval's HF backend.
#
# But the convert_to_hf step needs torchtitan + the v2 model registry —
# so source the v2 venv for the conversion, then deactivate before
# the lm-eval step.
V2_REPO="${REPO:-/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz}"
# Default to the canonical 512N chain (gbs12288); override CKPT_NAME +
# LABEL to evaluate other trajectories (e.g. the 256N comparator).
V2_CKPT_NAME="${CKPT_NAME:-agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288}"
# LABEL is appended to the output dir so 256N + 512N evals can
# coexist under outputs/evals/agpt-20b-v2-<LABEL>/.
LABEL="${LABEL:-512n}"

STEPS="${STEPS:-100 200 300}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"

for step in $STEPS; do
    DCP_DIR="${V2_REPO}/outputs/checkpoints/${V2_CKPT_NAME}/step-${step}"
    HF_DIR="outputs/evals/agpt-20b-v2-${LABEL}/step-${step}/hf"
    RESULTS_DIR="outputs/evals/agpt-20b-v2-${LABEL}/step-${step}/results"

    if [[ ! -d "$DCP_DIR" ]]; then
        echo "[SKIP] 20b-v2 step-${step}: no DCP at ${DCP_DIR}"
        continue
    fi
    if [[ -f "${RESULTS_DIR}/results.json" ]]; then
        # Content-aware skip: only skip if the existing results.json already
        # holds every requested task (SHOTS_SPEC groups, else TASKS). A plain
        # existence check made the modern-suite backfill a no-op (old files
        # have only the 0-shot commonsense tasks). See patch_skip_guard.py.
        if SHOTS_SPEC="${SHOTS_SPEC:-}" TASKS="${TASKS}" python3 - "${RESULTS_DIR}/results.json" <<'PYCHK'
import json, os, sys
path = sys.argv[1]
spec = os.environ.get("SHOTS_SPEC", "").strip()
if spec:
    want = set()
    for g in spec.split(";"):
        _, _, tl = g.partition(":")
        want |= {t for t in tl.split(",") if t}
else:
    want = set(os.environ.get("TASKS", "").split(","))
try:
    have = set(json.load(open(path)).keys())
except Exception:
    have = set()
# mmlu expands to a group key + mmlu_* subtasks; treat prefix as satisfied.
missing = [t for t in want
           if t and t not in have
           and not any(k == t or k.startswith(t + "_") for k in have)]
sys.exit(0 if not missing else 1)
PYCHK
        then
            echo "[SKIP] 20b-v2 step-${step}: all requested tasks already present"
            continue
        fi
        echo "[RERUN] 20b-v2 step-${step}: results.json missing requested task(s); running + merging"
    fi

    echo ""
    echo "============================================================"
    echo "20B v2 step-${step}"
    echo "============================================================"

    # Use absolute paths because step 1 (conversion) cd's into the v2
    # clone, then step 2 (lm-eval) cd's back here.
    EVAL_CLONE="$(pwd)"
    HF_DIR_ABS="${EVAL_CLONE}/${HF_DIR}"
    RESULTS_DIR_ABS="${EVAL_CLONE}/${RESULTS_DIR}"

    # ---- Step 1: DCP -> HF (run from v2 clone so torchtitan resolves) ----
    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] Converting DCP -> HF (via v2 venv, from ${V2_REPO})..."
        mkdir -p "${HF_DIR_ABS}"
        # Run conversion from the v2 clone so its torchtitan + venv +
        # local model registry are all on PYTHONPATH. Use PYTHONPATH=.
        # to force-add the v2 clone's torchtitan/ ahead of any system
        # python tree.
        (
            cd "${V2_REPO}" || exit 1
            source .venv/bin/activate
            echo "  subshell: pwd=$(pwd)"
            echo "  subshell: which python3=$(which python3)"
            echo "  subshell: torchtitan check..."
            PYTHONPATH=".:${PYTHONPATH:-}" python3 -c "import torchtitan; print('  torchtitan from:', torchtitan.__file__)" \
                || { echo "  ERROR: torchtitan import failed"; exit 1; }
            PYTHONPATH=".:${PYTHONPATH:-}" python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
                "${DCP_DIR}" \
                "${HF_DIR_ABS}" \
                --model_name "experiments.ezpz.agpt" \
                --model_flavor "20b" \
                --export_dtype "bfloat16"
        ) || { echo "[1/2] Conversion FAILED — skipping eval for step ${step}"; continue; }
        # Copy HF config + tokenizer assets from the eval clone (we
        # already have these, no need to pull from v2 clone).
        cp "${EVAL_CLONE}/torchtitan/experiments/ezpz/eval/configs/agpt_20b_config.json" \
            "${HF_DIR_ABS}/config.json"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer.{json,model} "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer_config.json "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/special_tokens_map.json "${HF_DIR_ABS}/"
        echo "[1/2] Conversion done."
    else
        echo "[1/2] HF already converted, skipping."
    fi

    # Sanity check: bail if conversion produced no model file.
    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] No model file in ${HF_DIR_ABS} — skipping eval"
        continue
    fi

    # ---- Step 2: lm-eval (bare frameworks venv + tt-lm-eval overlay) ----
    echo "[2/2] Running lm-eval..."
    source venvs/aurora/tt-lm-eval/bin/activate
    mkdir -p "${RESULTS_DIR_ABS}"
    HF_DIR_ABS="${HF_DIR_ABS}" RESULTS_DIR_ABS="${RESULTS_DIR_ABS}" TASKS="${TASKS}" SHOTS_SPEC="${SHOTS_SPEC:-}" LIMIT="${LIMIT:-}" \
    python3 << 'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None
from lm_eval import evaluator

hf_dir = os.environ["HF_DIR_ABS"]
results_dir = os.environ["RESULTS_DIR_ABS"]

# SHOTS_SPEC groups tasks by few-shot count: "shots:task,task;shots:task".
# A single lm-eval call uses one global num_fewshot, so mixed-shot suites
# (mmlu-5, arc_challenge-25, hellaswag-0) require one call PER shot group.
# Falls back to all of $TASKS at 0-shot when SHOTS_SPEC is unset (legacy).
shots_spec = os.environ.get("SHOTS_SPEC", "").strip()
_lim = os.environ.get("LIMIT", "").strip()
eval_limit = int(_lim) if _lim and int(_lim) > 0 else None
if shots_spec:
    groups = []
    for grp in shots_spec.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        shots_str, _, tlist = grp.partition(":")
        groups.append((int(shots_str), [t for t in tlist.split(",") if t]))
else:
    groups = [(0, os.environ["TASKS"].split(","))]

merged = {}
n_shot = {}
for shots, tset in groups:
    if not tset:
        continue
    print(f"  [lm-eval] {len(tset)} task(s) @ {shots}-shot: {','.join(tset)}", flush=True)
    r = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_dir}",
        tasks=tset,
        batch_size=2,
        num_fewshot=shots,
        device="xpu:0",
        limit=eval_limit,
    )
    # A task can appear in more than one shot group (arc_challenge runs 0-shot
    # in the commonsense block AND 25-shot in the modern block). Writing both
    # to the bare task key makes the later group silently overwrite the
    # earlier, producing a column that mixes shot counts with no way to tell
    # which is which. Namespace every task by its shot count, and keep the
    # bare key pointing at the LAST write so older readers/plotters still work.
    for _task, _metrics in r["results"].items():
        merged[f"{_task}@{shots}shot"] = _metrics
        merged[_task] = _metrics
    # lm-eval reports the shots actually used per task; preserve it so a
    # results.json is self-describing even for the bare keys.
    for _task, _n in (r.get("n-shot") or {}).items():
        n_shot[_task] = _n
        n_shot[f"{_task}@{shots}shot"] = _n

# Merge into any existing results.json so the 0-shot dashboard and the modern
# suite coexist (a step may be re-run to ADD tasks, not replace them).
out_path = f"{results_dir}/results.json"
existing = {}
if os.path.exists(out_path):
    try:
        existing = json.load(open(out_path))
    except Exception:
        existing = {}
existing.update(merged)
if n_shot:
    _prev = existing.get("n-shot") or {}
    _prev.update(n_shot)
    existing["n-shot"] = _prev
with open(out_path, "w") as f:
    json.dump(existing, f, indent=2)
for task, metrics in merged.items():
    if not isinstance(metrics, dict):
        continue
    # prefer acc_norm, then acc, then exact_match (gsm8k), then flexible EM
    val = (metrics.get("acc_norm,none")
           or metrics.get("acc,none")
           or metrics.get("exact_match,strict-match")
           or metrics.get("exact_match,none")
           or metrics.get("exact_match,flexible-extract", "?"))
    print(f"  {task}: {val:.4f}" if isinstance(val, float) else f"  {task}: {val}")
PYEOF
    deactivate
    echo "[2/2] Eval done. results: ${RESULTS_DIR_ABS}/results.json"
done

echo ""
echo "=== 20B v2 eval complete ==="
