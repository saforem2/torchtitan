#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -l select=1
#PBS -N eval-mmlu-harness-check
#PBS -j oe
#
# Is our MMLU harness measuring anything at all?
#
# MMLU sits at 4-way chance on EVERY AuroraGPT checkpoint measured, across a
# 20x token span and two model scales:
#
#   20B-256          0.39T   0.2599
#   20B-512          0.71T   0.2655
#   2B-512           3.99T   0.2473
#   2B-256 COMPLETE  4.674T  0.2437   (100% of target)
#   2B MDS stage-3   7.77T   0.2413   (best-ever ARC-C 0.3968 on the same ckpt)
#
# There is no upward trend -- the most-trained model scores LOWEST. Every other
# task on those same checkpoints improves monotonically (hellaswag 0.405 ->
# 0.561 over the completed 2B run). A capability that has not emerged yet would
# still drift; a pinned 0.24 looks like a measurement problem.
#
# This runs three PUBLIC models with well-known MMLU through our EXACT path:
# same tt-lm-eval venv, same simple_evaluate(model="hf", num_fewshot=5,
# device="xpu:0"). They span a wide range, so we get a ladder rather than a
# single point:
#
#   Llama-3.2-1B   published ~0.32
#   Llama-3.2-3B   published ~0.56
#   Llama-3.1-8B   published ~0.66
#
#   tracks the ladder -> harness is CLEAN. The AuroraGPT result is real and is
#                        about the DATA (does olmo-mix-1124 contain anything
#                        MMLU-shaped?), which is a genuine finding.
#   all ~0.25        -> the bug is OURS and has silently invalidated every MMLU
#                        number in the project.
#
# All three are already in the local HF cache with weights, so no download and
# no proxy dependency. Sizes 2.4G / 6.0G / 15G; 1B first so we get a verdict
# even if the 1h wall bites.

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1          # cached only -- fail loudly rather than hang on a download

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

RES_DIR="outputs/evals/harness-check-mmlu"
mkdir -p "${RES_DIR}"

RES_DIR="${RES_DIR}" python3 << 'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None
from lm_eval import evaluator

res_dir = os.environ["RES_DIR"]

# (hf id, published 5-shot MMLU for reference)
MODELS = [
    ("meta-llama/Llama-3.2-1B", 0.32),
    ("meta-llama/Llama-3.2-3B", 0.56),
    ("meta-llama/Llama-3.1-8B", 0.66),
]

out = {}
for name, published in MODELS:
    print(f"\n===== {name} (published MMLU-5 ~ {published:.2f}) =====", flush=True)
    try:
        r = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={name}",
            tasks=["mmlu"],
            batch_size=8,
            num_fewshot=5,
            device="xpu:0",
        )
        got = (r["results"].get("mmlu") or {}).get("acc,none")
        out[name] = {"published": published, "measured": got}
        print(f"  measured: {got}")
    except Exception as e:  # keep going; a later model may still answer it
        out[name] = {"published": published, "error": str(e)[:300]}
        print(f"  FAILED: {e}")

with open(f"{res_dir}/results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n==== VERDICT ====")
ok = bad = 0
for name, d in out.items():
    m, p = d.get("measured"), d["published"]
    if m is None:
        print(f"  {name}: ERROR -- {d.get('error','')[:80]}")
        continue
    # "at chance" = within noise of the 0.25 four-way floor
    if m < 0.28:
        print(f"  {name}: {m:.4f} vs published {p:.2f}  <-- AT CHANCE, should not be")
        bad += 1
    else:
        print(f"  {name}: {m:.4f} vs published {p:.2f}  ok")
        ok += 1
print()
if bad and not ok:
    print("  HARNESS IS BROKEN: known-good models score at chance through our path.")
    print("  Every MMLU number in the project is suspect.")
elif ok and not bad:
    print("  HARNESS IS CLEAN: it reproduces published MMLU on known models.")
    print("  The AuroraGPT MMLU result is real -- look at the data, not the eval.")
else:
    print("  MIXED -- see per-model lines above.")
print(f"  results: {res_dir}/results.json")
PYEOF

echo "=== done ==="
