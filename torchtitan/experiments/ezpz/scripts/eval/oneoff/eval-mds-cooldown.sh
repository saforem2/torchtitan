#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-mds-cooldown
#PBS -j oe
#
# The three cooldown forks under automate-cooldown/ -- never evaluated.
#
#   cooldown-3  global_step52650
#   cooldown-4  global_step72500
#   cooldown-5  global_step92400
#
# One checkpoint each, 3.7 GB, ws3072 production geometry. (The other dirs in
# automate-cooldown/ are ws24 / gb48 / gb384 dev runs -- NOT production, and
# deliberately excluded.)
#
# Why they matter: the 2026-07-28 anneal A/B concluded the LR SCHEDULE is not
# the lever at 10B tokens (constant-LR beat WSD-decay-to-zero on both bases).
# These are cooldowns at 52k/72k/92k steps -- a much longer horizon and a
# different mechanism than that A/B tested. If a cooldown produced a real
# capability jump over the constant-LR trajectory at the same step, that
# qualifies the "schedule does not matter" finding; if not, it corroborates it
# at 5-9x the token count.
#
# Full modern ladder INCLUDING mmlu here, unlike the stage-3 ladder: these are
# only 3 checkpoints, they sit at very different points on the token axis
# (~2.6T / 3.6T / 4.6T), and they are the one family where a schedule-driven
# MMLU effect has never been checked at all.

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

CD=/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/automate-cooldown/Megatron-DeepSpeed/checkpoints

for pair in "cooldown-3:52650" "cooldown-4:72500" "cooldown-5:92400"; do
  arm="${pair%%:*}"; step="${pair##*:}"
  SRC="${CD}/${arm}/global_step${step}/mp_rank_00_model_states.pt"
  HF_DIR="outputs/evals/agpt-2b-mds-cooldown/${arm}/step-${step}/hf"
  RES_DIR="outputs/evals/agpt-2b-mds-cooldown/${arm}/step-${step}/results"

  echo ""
  echo "======== ${arm} @ global_step${step} ========"
  date
  [[ -f "${SRC}" ]] || { echo "[SKIP] no states at ${SRC}"; continue; }

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

print(f"\n  ==== {arm} @ step-{step} ====")
for t, m in (("mmlu", "acc,none"), ("arc_challenge@25shot", "acc_norm,none"),
             ("hellaswag", "acc_norm,none"), ("arc_easy", "acc,none"),
             ("gsm8k", "exact_match,strict-match")):
    v = (merged.get(t) or {}).get(m)
    if isinstance(v, float):
        print(f"    {t}: {v:.4f}")
print(f"  results: {out}")
PYEOF
  date
done

echo ""
echo "=== cooldown evals done ==="
