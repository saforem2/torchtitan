#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-2b-v2
#PBS -j oe
#
# Convert + eval the v2 2B SophiaG checkpoints (fp32 master) and
# produce results that can be directly compared against the v1 256N
# eval table in docs/evals/agpt/2b/README.md (and v1 256N+512N eval
# results sit at outputs/evals/agpt-2b/).
#
# v2 ckpt paths (both 256N and 512N v2 runs):
#   /flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/
#     outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-{N}
#     outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-{N}
#
# Default: eval the 256N v2 checkpoint at every 200 steps from 200 to 2000
# (10 ckpts). Override via STEPS / CKPT_NAME env vars.
#
# Output (in this clone):
#   outputs/evals/agpt-2b-v2/step-{N}/{hf,results}/

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_ENABLE_HF_TRANSFER=0

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1
echo "PWD: $(pwd)"
echo "Modules loaded."

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

# V2_REPO controls where the DCP checkpoints LIVE (env-overridable so we
# can also eval ckpts in the legacy /flare/.../projects/saforem2/torchtitan/
# clone, where the chain pre-2026-04-30 was written).
V2_REPO="${V2_REPO:-/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz}"
# USE_LEGACY_CONVERTER=1 -> call convert_to_hf_legacy.py (handles the
# pre-qkv_linear FQN layout used by ckpts in the legacy clone). Default 0
# uses the canonical convert_to_hf.py.
USE_LEGACY_CONVERTER="${USE_LEGACY_CONVERTER:-0}"
# CONVERT_REPO controls where the conversion ENV + script live (the .venv
# and torchtitan/experiments/ezpz/eval/convert_to_hf.py). This MUST be a
# clone that has the eval/ subdir + working torch 2.13 venv — i.e. the
# canonical v2 clone. Default to PWD (where this script is executed from)
# so it works without override; only override if you have a newer eval env
# in a different clone.
CONVERT_REPO="${CONVERT_REPO:-/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz}"
# Default to the canonical 512N chain (gbs12288); override CKPT_NAME +
# LABEL to evaluate other trajectories (e.g. the abandoned 256N
# one-shot).
V2_CKPT_NAME="${CKPT_NAME:-agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288}"
# LABEL is appended to the output dir so 256N + 512N evals can
# coexist under outputs/evals/agpt-2b-v2-<LABEL>/.
LABEL="${LABEL:-512n}"

STEPS="${STEPS:-1000 2000 3000 4000 5000}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
# MODEL_FLAVOR / EVAL_CONFIG_JSON select the model shape at convert + eval time.
# Default = the stock vocab-256128 "2b" flavor (olmo-mix bases). The MDS base is
# vocab 256000, so its arm MUST pass MODEL_FLAVOR=2b-mds EVAL_CONFIG_JSON=agpt_2b_mds_config.json
# -- otherwise a 256000-weight model is loaded under a 256128 config = gibberish
# (the tokenizer-mismatch trap). Each base is evaluated with ITS OWN vocab.
# RoPE FLAVOR -- MODEL_FLAVOR IS REQUIRED. There is no safe default.
#
# The flavor selects the RoPE convention at convert time, and the checkpoint
# does NOT record which one it was trained with (both rope caches are
# persistent=False, so nothing lands on disk). Pass the wrong one and the
# adapter applies (or skips) the Q/K permute: the export loads fine and only
# fails as gibberish at generation.
#
# Why there is no default: THE CHAINS SWITCHED CONVENTION MID-FLIGHT. Commit
# 5ffb850a1 (2026-06-25) flipped CONFIG_SUFFIX to _real, and the running chains
# picked it up on their next resume. W&B run metadata (the authoritative record
# of what each run actually executed) shows:
#
#   20b_v2_256   agpt_20b  -> agpt_20b_real  at 2026-07-10  (loss 2.69 -> 6.14)
#   20b_v2_512   agpt_20b  -> agpt_20b_real  at 2026-07-05  (loss 2.57 -> 6.03)
#   2b_v2_512    agpt_2b   -> agpt_2b_real   at 2026-08-05  (no spike)
#   2b_v2_256    agpt_2b   throughout                        (never switched)
#
# So the correct flavor depends on WHICH STEP you are converting, not on which
# chain. A per-chain constant is wrong for at least one step of most chains.
#
# To find the right value for a given step, ask W&B for the run that produced
# it (see scripts/eval/rope_flavor_for_step.py), or read the registry table in
# docs/reference/known-bugs/rope-flavor-mismatch.md.
if [[ -z "${MODEL_FLAVOR:-}" ]]; then
    cat >&2 <<'ERRMSG'
[eval-2b-v2] ERROR: MODEL_FLAVOR is required and has no default.

  The agpt chains changed RoPE convention mid-flight (2026-06-25 onward), so
  the correct flavor depends on the STEP being converted. Guessing silently
  corrupts the export -- it loads fine and only fails as gibberish.

  Find it:
    python3 torchtitan/experiments/ezpz/scripts/eval/rope_flavor_for_step.py \
        --chain <2b_v2_512|2b_v2_256|20b_v2_512|20b_v2_256> --step <N>

  Then re-run with e.g.  MODEL_FLAVOR=2b_real  or  MODEL_FLAVOR=2b
  Details: docs/reference/known-bugs/rope-flavor-mismatch.md
ERRMSG
    exit 2
fi
# The pinned v2 clones do NOT ship agpt/state_dict_adapter.py -- they fall back
# to the bare Llama3StateDictAdapter, which applies the Q/K permute
# UNCONDITIONALLY, so a cos_sin flavor is silently ignored there. Refuse rather
# than emit a corrupt export.
#
# CHECK CONVERT_REPO, NOT V2_REPO. The conversion imports torchtitan from
# CONVERT_REPO (see step 1 below: "venv from CONVERT_REPO, DCP from V2_REPO"),
# so that is the checkout whose adapter decides whether the permute happens.
# V2_REPO only supplies the DCP bytes. This guard tested V2_REPO and therefore
# refused a CORRECT invocation: the matched-pair eval (job 8766898) passed
# CONVERT_REPO=<main repo>, which HAS the adapter, and was blocked anyway --
# losing the fork arm while the canonical arm completed fine.
if [[ "$MODEL_FLAVOR" == *_real ]] \
   && [[ ! -e "${CONVERT_REPO}/torchtitan/experiments/ezpz/agpt/state_dict_adapter.py" ]]; then
    cat >&2 <<ERRMSG
[eval-2b-v2] ERROR: MODEL_FLAVOR='${MODEL_FLAVOR}' (cos_sin) but the CONVERT_REPO
  ${CONVERT_REPO}
  has no agpt/state_dict_adapter.py, so it would use the bare
  Llama3StateDictAdapter and permute anyway -- producing exactly the corrupt
  export this flag is meant to avoid.

  Convert from a checkout that HAS the adapter, or update the clone.
  See docs/reference/known-bugs/rope-flavor-mismatch.md
ERRMSG
    exit 2
fi
echo "[eval-2b-v2] MODEL_FLAVOR='${MODEL_FLAVOR}' (explicit; no default exists -- see rope-flavor-mismatch.md)"
EVAL_CONFIG_JSON="${EVAL_CONFIG_JSON:-agpt_2b_config.json}"

for step in $STEPS; do
    DCP_DIR="${V2_REPO}/outputs/checkpoints/${V2_CKPT_NAME}/step-${step}"
    HF_DIR="outputs/evals/agpt-2b-v2-${LABEL}/step-${step}/hf"
    RESULTS_DIR="outputs/evals/agpt-2b-v2-${LABEL}/step-${step}/results"

    if [[ ! -d "$DCP_DIR" ]]; then
        echo "[SKIP] 2b-v2 step-${step}: no DCP at ${DCP_DIR}"
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
            echo "[SKIP] 2b-v2 step-${step}: all requested tasks already present"
            continue
        fi
        echo "[RERUN] 2b-v2 step-${step}: results.json missing requested task(s); running + merging"
    fi

    echo ""
    echo "============================================================"
    echo "2B v2 step-${step}"
    echo "============================================================"

    EVAL_CLONE="$(pwd)"
    HF_DIR_ABS="${EVAL_CLONE}/${HF_DIR}"
    RESULTS_DIR_ABS="${EVAL_CLONE}/${RESULTS_DIR}"

    # ---- Step 1: DCP -> HF (run from CONVERT_REPO so torchtitan resolves) ----
    # NOTE: V2_REPO can point at any clone (e.g. legacy) so DCP can be sourced
    # from there, but CONVERT_REPO MUST be the canonical v2 clone with .venv +
    # torchtitan/experiments/ezpz/eval/convert_to_hf.py. Don't conflate them.
    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] Converting DCP -> HF (venv from ${CONVERT_REPO}, DCP from ${V2_REPO})..."
        # Validate CONVERT_REPO has what we need before launching the subshell.
        if [[ ! -f "${CONVERT_REPO}/.venv/bin/activate" ]]; then
            echo "  ERROR: ${CONVERT_REPO}/.venv/bin/activate missing — set CONVERT_REPO to a clone with a built .venv"
            echo "[1/2] Conversion FAILED — skipping eval for step ${step}"
            continue
        fi
        if [[ "${USE_LEGACY_CONVERTER}" == "1" ]]; then
            CONVERT_PY="torchtitan/experiments/ezpz/eval/convert_to_hf_legacy.py"
        else
            CONVERT_PY="torchtitan/experiments/ezpz/eval/convert_to_hf.py"
        fi
        if [[ ! -f "${CONVERT_REPO}/${CONVERT_PY}" ]]; then
            echo "  ERROR: ${CONVERT_REPO}/${CONVERT_PY} missing"
            echo "[1/2] Conversion FAILED — skipping eval for step ${step}"
            continue
        fi
        mkdir -p "${HF_DIR_ABS}"
        (
            cd "${CONVERT_REPO}" || exit 1
            source .venv/bin/activate
            echo "  subshell: pwd=$(pwd)"
            echo "  subshell: which python3=$(which python3)"
            echo "  subshell: converter=${CONVERT_PY}"
            PYTHONPATH=".:${PYTHONPATH:-}" python3 -c "import torchtitan; print('  torchtitan from:', torchtitan.__file__)" \
                || { echo "  ERROR: torchtitan import failed"; exit 1; }
            PYTHONPATH=".:${PYTHONPATH:-}" python3 "${CONVERT_PY}" \
                "${DCP_DIR}" \
                "${HF_DIR_ABS}" \
                --model_name "experiments.ezpz.agpt" \
                --model_flavor "${MODEL_FLAVOR}" \
                --export_dtype "bfloat16"
        ) || { echo "[1/2] Conversion FAILED — skipping eval for step ${step}"; continue; }
        cp "${EVAL_CLONE}/torchtitan/experiments/ezpz/eval/configs/${EVAL_CONFIG_JSON}" \
            "${HF_DIR_ABS}/config.json"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer.{json,model} "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer_config.json "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/special_tokens_map.json "${HF_DIR_ABS}/"
        echo "[1/2] Conversion done."
    else
        echo "[1/2] HF already converted, skipping."
    fi

    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] No model file in ${HF_DIR_ABS} — skipping eval"
        continue
    fi

    # ---- Step 2: lm-eval (frameworks venv + tt-lm-eval overlay) ----
    echo "[2/2] Running lm-eval..."
    source venvs/aurora/tt-lm-eval/bin/activate
    mkdir -p "${RESULTS_DIR_ABS}"
    # batch_size=8 is fine for 2B on single XPU (vs 2 for 20B).
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
for shots, tset in groups:
    if not tset:
        continue
    print(f"  [lm-eval] {len(tset)} task(s) @ {shots}-shot: {','.join(tset)}", flush=True)
    r = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_dir}",
        tasks=tset,
        batch_size=8,
        num_fewshot=shots,
        device="xpu:0",
        limit=eval_limit,
    )
    merged.update(r["results"])

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
with open(out_path, "w") as f:
    json.dump(existing, f, indent=2)
for task, metrics in merged.items():
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
echo "=== 2B v2 eval complete ==="
