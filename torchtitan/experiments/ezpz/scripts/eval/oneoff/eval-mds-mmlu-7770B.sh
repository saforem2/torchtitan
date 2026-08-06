#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -l select=1
#PBS -N eval-mds-mmlu-7770B
#PBS -j oe
#
# Does MMLU EVER leave chance in this stack? Run MMLU-5 on the most-trained
# 2B checkpoint we have: MDS stage-3 at 7.77T tokens (step-140352).
#
# Why this checkpoint. MMLU sits at 4-way chance (~0.25) on every 20B and 2B
# checkpoint measured so far, including the COMPLETE 2B-256 v2 chain at its
# full 4.674T-token target (0.2437 at the finish, 0.2443 at 1.8T -- no trend).
# That kills the easy "not enough tokens yet" explanation. This checkpoint has
# 1.66x the tokens of the entire v2 budget and the strongest commonsense
# scores of any 2B we have (ARC-Easy 0.7037, ARC-C ~0.34, Winogrande ~0.59),
# so if a 2B in this stack can move MMLU at all, it moves here.
#
# Two clean outcomes:
#   moves  -> harness is fine, MMLU is just very late/large-scale emergent,
#             and 20B at 0.4T is nowhere near it. Stop worrying.
#   chance -> three configs spanning 4T-7.8T tokens all pinned at 0.25 points
#             at the harness or the data, not at capability. Chase it before
#             spending more 20B eval time on MMLU.
#
# eval_mds_sweep.sh never ran MMLU: it hardcodes
# tasks=[hellaswag,arc_easy,arc_challenge,winogrande] at num_fewshot=0. This
# reuses its conversion path (mds_to_hf.py) and adds the modern block.
#
# Debug queue, 1h wall. Budget: ~4 min MDS->HF for a 3.7 GB 2B checkpoint,
# then MMLU-5. The 20B did a 4,687-request loglikelihood block in 25:48; 2B is
# ~10x smaller, so MMLU-57 should fit comfortably. arc_challenge-25 is included
# because it is cheap once the model is loaded and gives a same-shot-count
# comparison against the 20B modern block.

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

CKPT_PFX="/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/Megatron-DeepSpeed/checkpoints"
STAGE_PFX="AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05"
STAGE_SFX="tokHF_tmgoogle_gemma-7b_flash"

STAGE="${STAGE:-ntok7770B}"
STEP="${STEP:-140352}"

SRC="${CKPT_PFX}/${STAGE_PFX}_${STAGE}_${STAGE_SFX}/global_step${STEP}/mp_rank_00_model_states.pt"
HF_DIR="outputs/evals/agpt-2b-mds/${STAGE}/step-${STEP}/hf"
RES_DIR="outputs/evals/agpt-2b-mds/${STAGE}/step-${STEP}/results"

if [[ ! -f "${SRC}" ]]; then
    echo "[FATAL] no model states at ${SRC}"
    exit 1
fi

echo "=== MDS MMLU probe: ${STAGE} @ global_step${STEP} (7.77T tokens) ==="
date

if [[ ! -f "${HF_DIR}/model-00001-of-00001.safetensors" ]]; then
    echo "[1/2] Converting MDS -> HF..."
    python3 torchtitan/experiments/ezpz/eval/mds_to_hf.py \
        --mds_checkpoint "${SRC}" \
        --output_dir "${HF_DIR}" 2>&1 || { echo "[FATAL] conversion failed"; exit 1; }
else
    echo "[1/2] HF checkpoint exists, skipping conversion."
fi
date

echo "[2/2] Running lm-eval (mmlu-5, arc_challenge-25)..."
mkdir -p "${RES_DIR}"
HF_DIR="${HF_DIR}" RES_DIR="${RES_DIR}" python3 << 'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None
from lm_eval import evaluator

hf_dir = os.environ["HF_DIR"]
res_dir = os.environ["RES_DIR"]

# Same shot counts as the 20B modern block so the numbers are comparable.
# Namespaced by shot count for the same reason as eval-20b-v2.sh: a bare
# task key silently loses which shot count produced it.
merged, n_shot = {}, {}
for shots, tasks in ((5, ["mmlu"]), (25, ["arc_challenge"])):
    print(f"  [lm-eval] {tasks} @ {shots}-shot", flush=True)
    r = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_dir}",
        tasks=tasks,
        batch_size=8,
        num_fewshot=shots,
        device="xpu:0",
    )
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
with open(out, "w") as f:
    json.dump(existing, f, indent=2)

print()
print("  ==== HEADLINE ====")
for t in ("mmlu", "arc_challenge"):
    m = merged.get(t) or {}
    v = m.get("acc,none") or m.get("acc_norm,none")
    if isinstance(v, float):
        verdict = "AT CHANCE" if t == "mmlu" and v < 0.27 else ""
        print(f"  {t}: {v:.4f}  {verdict}")
print(f"  results: {out}")
PYEOF
date
echo "=== done ==="
