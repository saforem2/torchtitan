#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-mds-stage3-ladder
#PBS -j oe
#
# A LADDER across the MDS stage-3 finishing phase, on both arms.
#
# 8747067 evaluated only the TIPS (global_step154391) and found the finishing
# mix matters enormously at the endpoint:
#
#                     stage3-mix   stage3 (math/code)
#   arc_challenge@25    0.4164       0.3703
#   hellaswag           0.5874       0.4215
#   arc_easy            0.7138       0.6216
#   gsm8k               0.0167       0.0303
#   mmlu                0.2463       0.2591   (both chance)
#
# The endpoint says math/code forgot ~16 points of HellaSwag to buy gsm8k that
# is still ~0. What it CANNOT say is WHEN that happened -- immediately on the
# data switch, or gradually across 14k steps. Both arms branch from ~140,300
# and run to 154,391 in 50-step increments (281 / 283 ckpts), so a ladder over
# the same steps on both arms makes the forgetting curve directly comparable.
#
# Ladder: every 2000 steps, 140400 -> 154391 (8 points/arm, 16 conversions).
# MMLU is DELIBERATELY EXCLUDED -- it is settled at chance across 5 configs, a
# 20x token span and 4 finishing mixes, against a harness validated on
# Llama-3.2-1B (0.3121) / Llama-3.1-8B (0.6530). Spending the slow 56k-request
# pass on 16 more predictable ~0.25 values buys nothing; the tips already
# anchor it. arc_c@25 + gsm8k + commonsense are where the signal is.

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

EXP=/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments
PFX="AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05"
SFX="tokHF_tmgoogle_gemma-7b_flash"
STEPS="140400 142400 144400 146400 148400 150400 152400 154391"

for arm in stage3-mix stage3; do
  for step in $STEPS; do
    SRC="${EXP}/${arm}/Megatron-DeepSpeed/checkpoints/${PFX}_ntok7770B_${SFX}/global_step${step}/mp_rank_00_model_states.pt"
    HF_DIR="outputs/evals/agpt-2b-mds-7771T/${arm}/step-${step}/hf"
    RES_DIR="outputs/evals/agpt-2b-mds-7771T/${arm}/step-${step}/results"

    echo ""
    echo "======== ${arm} @ global_step${step} ========"
    date
    [[ -f "${SRC}" ]] || { echo "[SKIP] no states at ${SRC}"; continue; }

    # Content-aware skip: the tips were already done by 8747067.
    if [[ -f "${RES_DIR}/results.json" ]] && \
       python3 -c "
import json,sys
j=json.load(open('${RES_DIR}/results.json'))
need=['hellaswag','arc_easy','arc_challenge@25shot','gsm8k']
sys.exit(0 if all(k in j for k in need) else 1)" 2>/dev/null; then
      echo "[SKIP] ${arm} step-${step}: requested tasks already present"
      continue
    fi

    if [[ ! -f "${HF_DIR}/model-00001-of-00001.safetensors" ]]; then
      echo "converting MDS -> HF ..."
      python3 torchtitan/experiments/ezpz/eval/mds_to_hf.py \
          --mds_checkpoint "${SRC}" --output_dir "${HF_DIR}" 2>&1 | tail -3 || {
          echo "[ERROR] conversion failed"; continue; }
    fi

    HF_DIR="${HF_DIR}" RES_DIR="${RES_DIR}" ARM="${arm}" STEP="${step}" python3 << 'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *a, **k: None
from lm_eval import evaluator

hf_dir, res_dir = os.environ["HF_DIR"], os.environ["RES_DIR"]
arm, step = os.environ["ARM"], os.environ["STEP"]

GROUPS = [
    (0, ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]),
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

os.makedirs(res_dir, exist_ok=True)
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

vals = []
for t, m in (("arc_challenge@25shot", "acc_norm,none"),
             ("hellaswag", "acc_norm,none"),
             ("arc_easy", "acc,none"),
             ("gsm8k", "exact_match,strict-match")):
    v = (merged.get(t) or {}).get(m)
    if isinstance(v, float):
        vals.append(f"{t.split('@')[0]}={v:.4f}")
print(f"  LADDER {arm} step-{step}: " + "  ".join(vals))
PYEOF
    date
  done
done

echo ""
echo "=== ladder done ==="
