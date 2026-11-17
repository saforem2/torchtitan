#!/bin/bash
# hf_ladder_p0p3_generator.sh — emits PBS qsub scripts for CPT-FWE P0-P3 HF conversion
#
# Usage:
#   bash hf_ladder_p0p3_generator.sh P0   # writes /tmp/hf_ladder_p0.sh (then: qsub /tmp/hf_ladder_p0.sh)
#   bash hf_ladder_p0p3_generator.sh P1
#   bash hf_ladder_p0p3_generator.sh P2
#   bash hf_ladder_p0p3_generator.sh P3   # steps not yet on disk -> SKIP no-op, still safe to qsub
#   bash hf_ladder_p0p3_generator.sh all  # writes all 4
#
# Emitted script mirrors the known-good hf_batch_convert_cpt_fwe.sh pattern:
#   - cwd = /home/foremans/torchtitan (repo root, makes `import torchtitan...` work)
#   - VENV xpu-torch216-sync-5a7a622f16 with LD-first SYCL/UR symlink dir on LD_LIBRARY_PATH
#   - sentinel ezpz_export.json guards re-converts
set -euo pipefail

PHASE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

P0_STEPS="1000 5000 10000"
P1_STEPS="25000 50000 75000"
P2_STEPS="100000 111200 125000 136800 143400"
P3_STEPS="150000 160000 175000 190734"

emit() {
  local tag="$1" steps="$2"
  local out="/tmp/hf_ladder_${tag,,}.sh"
  cat > "$out" <<'EOS'
#!/bin/bash --login
#PBS -N hf-batch-cpt-fwe-__TAG__
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe

set -euo pipefail

cd /home/foremans/torchtitan
VENV=/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16
# LD-first: venv UR/SYCL libs win over system (mirrors hf_batch_convert_cpt_fwe.sh)
_ld_first=/home/foremans/torchtitan/.cache/xpu-ld-first/xpu-torch216-sync-5a7a622f16
if [[ ! -e "${_ld_first}/.complete" ]]; then
  mkdir -p "${_ld_first}"
  for _lib in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
    [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"
  done
  touch "${_ld_first}/.complete"
fi
export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

DCP_ROOT=/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
HF_ROOT="$TEGU_ROOT/outputs/evals/cpt-fwe"
HF_CONFIG=/home/foremans/torchtitan/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=/home/foremans/torchtitan/assets/hf/gemma-7b

echo "HF-LADDER __TAG__ START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
echo "  HF_ROOT=$HF_ROOT"
echo "  STEPS: __STEPS__"

for s in __STEPS__; do
  DCP="$DCP_ROOT/step-$s"
  HF="$HF_ROOT/step-$s/hf"
  [[ -d "$DCP" ]] || { echo "SKIP $s (no DCP)"; continue; }
  [[ -f "$HF/ezpz_export.json" ]] && { echo "SKIP $s (HF exists)"; continue; }
  echo "CONVERT $s"
  python -m torchtitan.experiments.ezpz.eval.convert_to_hf "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds --export_dtype bfloat16 || exit 1
  cp "$HF_CONFIG" "$HF/config.json"
  # HF export already wrote config.json + tokenizer.* via write_hf_assets; the
  # legacy cp overwrites are kept for dense 2b-mds chain but must include the
  # tokenizer JSONs or lm-eval bos_token_id is None (see Tegu 79-HF backfill).
  cp "$TOK_SRC"/tokenizer_config.json "$HF/" 2>/dev/null || true
  cp "$TOK_SRC"/special_tokens_map.json "$HF/" 2>/dev/null || true
  cp "$TOK_SRC"/tokenizer.* "$HF/"
  ls -lh "$HF"/model-*.safetensors
done

echo "CONVERTED $(ls "$HF_ROOT"/step-*/hf/ezpz_export.json 2>/dev/null | wc -l) HF total"
EOS
  sed -i.bak -e "s/__TAG__/$tag/g" -e "s/__STEPS__/$steps/g" "$out" && rm -f "$out.bak"
  echo "$out"
}

case "$PHASE" in
  P0) emit P0 "$P0_STEPS" ;;
  P1) emit P1 "$P1_STEPS" ;;
  P2) emit P2 "$P2_STEPS" ;;
  P3) emit P3 "$P3_STEPS" ;;
  all) emit P0 "$P0_STEPS"; emit P1 "$P1_STEPS"; emit P2 "$P2_STEPS"; emit P3 "$P3_STEPS" ;;
  *) echo "Usage: $0 {P0|P1|P2|P3|all}" >&2; exit 1 ;;
esac
