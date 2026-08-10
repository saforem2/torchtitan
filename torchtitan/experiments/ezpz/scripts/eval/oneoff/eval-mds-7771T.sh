#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-mds-7771T
#PBS -j oe
#
# The two genuine 7.771T MDS checkpoints -- never evaluated on anything.
#
# These are the most-trained models in the project: global_step154391,
# 7,771,000,000,000-ish tokens, ~1.66x the ENTIRE v2 budget. They were missed
# because they live in different working directories from the rest of the MDS
# tree (stage3-mix/ and stage3/, not optimizer-experiments/Megatron-DeepSpeed/),
# so every earlier sweep walked right past them.
#
#   stage3-mix  295 ckpts  loss 2.033  ALCF/data-lists/aurora/stage1-33-stage2-33-stage3-34.txt
#   stage3      283 ckpts  loss 0.880  ALCF/data-lists/aurora/nvidia-math1-code2.txt
#
# Same architecture, same optimizer/LR, same token count, DIFFERENT finishing
# mix -- so this is a controlled test of whether the finishing data moves MMLU,
# which is the open question from the 2026-08-05 investigation. The previously
# probed "7.77T" checkpoint was actually 7.064T on dolmino-fused (ntok7770B in
# a directory name is a TARGET, not consumed tokens) and scored mmlu 0.2413.
#
# Prediction worth recording BEFORE the run: if MMLU is data-limited rather
# than capability-limited, the math/code finisher should NOT help MMLU (it is
# not academic-MC content) while the stage1/2/3 mix might. If BOTH sit at
# 0.24 at 7.771T, that is the strongest evidence yet that olmo-mix-derived
# pretraining simply does not contain what MMLU tests.
#
# gsm8k is included because the math/code checkpoint is the one model we have
# that might actually do arithmetic -- every other AuroraGPT ckpt scores ~0.
# 6h walltime: 2 ckpts x (convert + mmlu-57 + arc_c-25 + gsm8k).

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

EXP=/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments
STAGE_PFX="AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05"
STAGE_SFX="tokHF_tmgoogle_gemma-7b_flash"
STEP=154391

for arm in stage3-mix stage3; do
    SRC="${EXP}/${arm}/Megatron-DeepSpeed/checkpoints/${STAGE_PFX}_ntok7770B_${STAGE_SFX}/global_step${STEP}/mp_rank_00_model_states.pt"
    HF_DIR="outputs/evals/agpt-2b-mds-7771T/${arm}/step-${STEP}/hf"
    RES_DIR="outputs/evals/agpt-2b-mds-7771T/${arm}/step-${STEP}/results"

    echo ""
    echo "############################################################"
    echo "# ${arm} @ global_step${STEP}"
    echo "############################################################"
    date
    if [[ ! -f "${SRC}" ]]; then
        echo "[SKIP] no model states at ${SRC}"
        continue
    fi

    if [[ ! -f "${HF_DIR}/model-00001-of-00001.safetensors" ]]; then
        echo "[1/2] Converting MDS -> HF ..."
        python3 torchtitan/experiments/ezpz/eval/mds_to_hf.py \
            --mds_checkpoint "${SRC}" --output_dir "${HF_DIR}" 2>&1 || {
            echo "[ERROR] conversion failed for ${arm}"; continue; }
    else
        echo "[1/2] HF checkpoint exists, skipping conversion."
    fi
    date

    echo "[2/2] lm-eval: commonsense-0 + mmlu-5 + arc_c-25 + gsm8k-5"
    mkdir -p "${RES_DIR}"
    HF_DIR="${HF_DIR}" RES_DIR="${RES_DIR}" ARM="${arm}" python3 << 'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *a, **k: None
from lm_eval import evaluator

hf_dir, res_dir, arm = os.environ["HF_DIR"], os.environ["RES_DIR"], os.environ["ARM"]

# Shot-namespaced keys per 3e1877170: arc_challenge is scored at BOTH 0-shot
# (commonsense block) and 25-shot (modern block), and a bare task key silently
# loses which one produced the number.
GROUPS = [
    (0, ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa",
         "openbookqa", "boolq"]),
    (5, ["mmlu"]),
    (25, ["arc_challenge"]),
    (5, ["gsm8k"]),
]

merged, n_shot = {}, {}
for shots, tasks in GROUPS:
    print(f"  [lm-eval] {len(tasks)} task(s) @ {shots}-shot", flush=True)
    try:
        r = evaluator.simple_evaluate(
            model="hf", model_args=f"pretrained={hf_dir}", tasks=tasks,
            batch_size=8, num_fewshot=shots, device="xpu:0")
    except Exception as e:
        print(f"    FAILED @ {shots}-shot: {e}")
        continue
    for t, m in r["results"].items():
        merged[f"{t}@{shots}shot"] = m
        merged[t] = m
    for t, n in (r.get("n-shot") or {}).items():
        n_shot[t] = n
        n_shot[f"{t}@{shots}shot"] = n

out = f"{res_dir}/results.json"
existing = {}
if os.path.exists(out):
    try:
        existing = json.load(open(out))
    except Exception:
        existing = {}
existing.update(merged)
if n_shot:
    prev = existing.get("n-shot") or {}
    prev.update(n_shot)
    existing["n-shot"] = prev
json.dump(existing, open(out, "w"), indent=2)

print(f"\n  ==== {arm} @ 7.771T ====")
for t, m in (("mmlu", "acc,none"), ("arc_challenge@25shot", "acc_norm,none"),
             ("hellaswag", "acc_norm,none"), ("arc_easy", "acc,none"),
             ("gsm8k", "exact_match,strict-match")):
    v = (merged.get(t) or {}).get(m)
    if isinstance(v, float):
        flag = "  <-- ABOVE CHANCE" if t == "mmlu" and v >= 0.28 else ""
        print(f"    {t}: {v:.4f}{flag}")
print(f"  results: {out}")
PYEOF
    date
done

echo ""
echo "=== done ==="
