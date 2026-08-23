#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -l filesystems=flare:home
#PBS -N reeval-ropefix
#PBS -j oe
#
# Re-evaluate every published eval that was converted with the WRONG RoPE
# flavor. Parameterised by CHAIN so three of these run concurrently.
#
#   qsub -v CHAIN=20b-512 ...   34 steps
#   qsub -v CHAIN=20b-256 ...   36 steps
#   qsub -v CHAIN=2b-512  ...    4 steps
#
# WHY: the chains switched complex -> cos_sin mid-flight, but the eval scripts
# hardcoded the complex flavor. MEASURED correction (job 8760246/8760307, same
# ckpt + harness, only --model_flavor differing):
#   20B step 5000  arc_c 0.3123 -> 0.3575  (+0.045)
#   20B step 6000  arc_c 0.2713 -> 0.3660  (+0.095)
#   2B  step 46429 hellaswag 0.4753 -> 0.5384, mmlu 0.2511 -> 0.2579
# The published "ARC-C decays monotonically" curve is an artifact: corrected,
# the model improves throughout. See
# docs/experiments/agpt/aurora/20260816-arc-c-decay-vs-rope-permute.md
#
# DISK IS THE BINDING CONSTRAINT. A 20B HF export is 78 GB; 70 of them is
# 5.5 TB and flare has ~31 PB free but this dir does not need 5 TB of garbage.
# Each export is DELETED immediately after its eval succeeds. Set KEEP_HF=1 to
# retain them (only do this for a handful of steps).
#
# Converts from the MAIN repo: the pinned v2 clones ship no
# agpt/state_dict_adapter.py and would apply the Q/K permute regardless of the
# flavor flag, silently reproducing the very bug this fixes.

set -o pipefail   # NOT -u: lmod init reads ZSH_EVAL_CONTEXT, venv activate has unbound vars

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1

MAIN=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
RUNS=/lus/flare/projects/AuroraGPT/foremans/runs
CHAIN="${CHAIN:?set CHAIN=20b-512|20b-256|2b-512}"
KEEP_HF="${KEEP_HF:-0}"

case "$CHAIN" in
  20b-512)
    CKPT=$RUNS/agpt-20b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288
    OUT=$MAIN/outputs/evals/agpt-20b-v2-512n-ropefix
    FLAVOR=20b_real; CFG=agpt_20b_config.json
    # 5000 and 6000 were MISSING from this list (it jumped 4900->5100 and
    # 5900->6010) while the 20b-256 list below has both. The sweep therefore
    # reported "34 ok, 0 skipped, 0 failed" and was complete FOR THE LIST IT
    # WAS GIVEN -- the list was the bug, so nothing flagged it. Both
    # checkpoints exist (6144 shards each) and had already been arc/hellaswag
    # evaluated, so only winogrande was absent. Added 2026-08-21.
    STEPS="${STEPS:-4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6010 6100 6200 6300 6400 6500 6550 6600 6700 6800 6900 7000 7100 7150 7200 7250 7300 7400 7500 7600}"
    TASKS="${TASKS:-arc_challenge,hellaswag,arc_easy}" ;;
  20b-256)
    CKPT=$RUNS/agpt-20b-n256/torchtitan-ezpz/outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144
    OUT=$MAIN/outputs/evals/agpt-20b-v2-256n-ropefix
    FLAVOR=20b_real; CFG=agpt_20b_config.json
    STEPS="${STEPS:-4000 4200 4400 4600 4800 5000 5100 5300 5400 5500 5600 5700 5800 5900 6000 6100 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300}"
    TASKS="${TASKS:-arc_challenge,hellaswag,arc_easy}" ;;
  2b-512)
    CKPT=$RUNS/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288
    OUT=$MAIN/outputs/evals/agpt-2b-v2-512n-ropefix
    FLAVOR=2b_real; CFG=agpt_2b_config.json
    STEPS="${STEPS:-35000 37000 39000 39600}"
    TASKS="${TASKS:-mmlu,arc_challenge,arc_easy,hellaswag,winogrande,piqa,openbookqa,boolq}" ;;
  *) echo "FATAL: unknown CHAIN=$CHAIN"; exit 2 ;;
esac

cd "$MAIN" || exit 1
[[ -e torchtitan/experiments/ezpz/agpt/state_dict_adapter.py ]] || {
    echo "FATAL: no agpt/state_dict_adapter.py here -- would permute anyway."; exit 2; }

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_ENABLE_HF_TRANSFER=0

n_ok=0; n_skip=0; n_fail=0
for step in $STEPS; do
    DCP=$CKPT/step-$step
    HF=$OUT/step-$step/hf
    RES=$OUT/step-$step/results
    [[ -d "$DCP" ]]              || { echo "[skip] no DCP step-$step"; n_skip=$((n_skip+1)); continue; }
    # Skip only if the existing results.json already holds EVERY requested
    # task. Testing mere file existence made a task-fill run a silent no-op:
    # jobs 8766059/8766061 exited in 1s with "0 ok, 36 skipped, 0 failed" and
    # Exit_status=0, because the arc/hellaswag pass had already written a file
    # at every step. Exit 0 plus no work looks identical to success.
    if [[ -f "$RES/results.json" ]] && RES_JSON="$RES/results.json" WANT="$TASKS" python3 -c '
import json, os, sys
want = {t for t in os.environ["WANT"].split(",") if t}
try:
    have = set(json.load(open(os.environ["RES_JSON"])))
except Exception:
    sys.exit(1)          # unreadable -> re-run it
sys.exit(0 if want <= have else 1)
'; then
        echo "[skip] have step-$step (all of: $TASKS)"; n_skip=$((n_skip+1)); continue
    fi

    echo "=== [$CHAIN] step $step: convert ($FLAVOR) ==="
    mkdir -p "$HF" "$RES"
    (
        source .venv/bin/activate 2>/dev/null
        PYTHONPATH=".:${PYTHONPATH:-}" python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
            "$DCP" "$HF" --model_name "experiments.ezpz.agpt" \
            --model_flavor "$FLAVOR" --export_dtype "bfloat16"
    ) || { echo "[FAIL] convert step-$step"; n_fail=$((n_fail+1)); rm -rf "$HF"; continue; }

    cp torchtitan/experiments/ezpz/eval/configs/$CFG "$HF/config.json"
    cp assets/hf/gemma-7b/tokenizer.{json,model} "$HF/"
    cp assets/hf/gemma-7b/tokenizer_config.json "$HF/"
    cp assets/hf/gemma-7b/special_tokens_map.json "$HF/"

    echo "=== [$CHAIN] step $step: lm-eval ==="
    source venvs/aurora/tt-lm-eval/bin/activate
    HF_DIR="$HF" RES_DIR="$RES" TASKS="$TASKS" python3 <<'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *a, **k: None
from lm_eval import evaluator
hf, res = os.environ["HF_DIR"], os.environ["RES_DIR"]
out = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={hf},dtype=bfloat16,trust_remote_code=True",
    tasks=os.environ["TASKS"].split(","), device="xpu:0", batch_size=8,
)
with open(os.path.join(res, "results.json"), "w") as fh:
    json.dump(out["results"], fh, indent=2)
PYEOF

    if [[ -f "$RES/results.json" ]]; then
        echo "[OK] step-$step"; n_ok=$((n_ok+1))
        # 78 GB per 20B export -- reclaim immediately or 70 steps is 5.5 TB.
        [[ "$KEEP_HF" == "1" ]] || rm -rf "$HF"
    else
        echo "[FAIL] lm-eval step-$step"; n_fail=$((n_fail+1)); rm -rf "$HF"
    fi
done
echo "=== [$CHAIN] DONE: $n_ok ok, $n_skip skipped, $n_fail failed ==="
