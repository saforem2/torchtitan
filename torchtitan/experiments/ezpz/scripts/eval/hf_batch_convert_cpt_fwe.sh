#!/bin/bash --login
#PBS -N hf-batch-cpt-fwe
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
#
# Batch convert CPT FineWeb-Edu DCP -> HF for MMLU-first roadmap.
# Frontier: 111.2k/190.7k LIVE (12483856 R). HF 67400 READY, 72800 blocked
# (no tokenizer), 13000 empty, 74400-111000 DCP exists but HF missing.
#
# Fixups:
#   72800 - copy config.json + tokenizer.* from 67400 HF (already converted)
#   13000 - reconvert DCP -> HF (empty HF dir)
# Batch:  74400 -> 111000 every 200 via seq (185 steps)
#
# Staging lock pattern from hf_export.py (staged_export) is already handled
# inside convert_to_hf.py -- no manual .lock/.staging needed here.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh
#

set -euo pipefail

REPO=/home/foremans/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9

DCP_BASE=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
HF_BASE=$REPO/outputs/evals/cpt-fwe

# Canonical per-step paths (spec): DCP=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-$s
#                                  HF=$REPO/outputs/evals/cpt-fwe/step-$s/hf

echo "HF-BATCH-CPT-FWE START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) repo=$REPO"
echo "  DCP_BASE=$DCP_BASE"
echo "  HF_BASE=$HF_BASE"
echo "  flavor=2b-mds dtype=bfloat16"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
convert_step() {
    local s="$1"
    local DCP=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-$s
    local HF=$REPO/outputs/evals/cpt-fwe/step-$s/hf

    echo "--- step-$s DCP=$DCP HF=$HF ---"

    if [[ ! -f "$DCP/.metadata" ]]; then
        echo "[SKIP] step-$s no DCP .metadata at $DCP"
        return 0
    fi

    # Skip if HF already complete (has weight shards + manifest).
    if [[ -f "$HF/model.safetensors" || -f "$HF/model-00001-of-*.safetensors" || -f "$HF/model-00001-of-00001.safetensors" ]]; then
        # Validate before skipping -- catches half-written or tokenizer-missing HFs.
        if "$CONVERT_PY" -c "from torchtitan.experiments.ezpz.eval.hf_export import validate_export; validate_export('$HF')" 2>&1; then
            echo "[SKIP] step-$s HF already valid at $HF"
            return 0
        else
            echo "[WARN] step-$s HF exists but invalid -- reconverting (rm -rf $HF)"
            rm -rf "$HF"
        fi
    fi

    mkdir -p "$(dirname "$HF")"

    echo "[CONVERT] step-$s $DCP -> $HF"

    if ! $HOME/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf $DCP $HF --model_name experiments.ezpz.agpt --model_flavor 2b-mds --export_dtype bfloat16 2>&1; then
        echo "[FAIL] step-$s convert failed rc=$?"
        return 1
    fi

    # convert_to_hf writes config.json + tokenizer via write_hf_assets (staged_export
    # lock pattern already handled inside convert_to_hf). Belt-and-suspenders cp:
    # (no-op if already present from adapter, but required by spec for explicitness)
    cp -n "$HF_BASE/step-67400/hf/config.json" "$HF/config.json" 2>/dev/null || true
    for tok in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do cp -n "$HF_BASE/step-67400/hf/$tok" "$HF/$tok" 2>/dev/null || true; done
    # Fallback: if no 67400 HF yet, assets already written by adapter; nothing to copy.

    if ! "$CONVERT_PY" -c "from torchtitan.experiments.ezpz.eval.hf_export import validate_export; validate_export('$HF')" 2>&1; then
        echo "[FAIL] step-$s validate_export failed"
        return 1
    fi

    echo "[OK] step-$s converted and validated: $HF"
}

fix_72800() {
    local SRC="$HF_BASE/step-67400/hf"
    local DST="$HF_BASE/step-72800/hf"
    echo "=== FIXUP 72800 (copy tokenizer from 67400) ==="
    echo "  SRC=$SRC"
    echo "  DST=$DST"

    if [[ ! -d "$SRC" ]]; then
        echo "[FAIL] 72800 fixup: source HF missing: $SRC"
        return 1
    fi
    if [[ ! -d "$DST" ]]; then
        echo "[FAIL] 72800 fixup: dest HF missing: $DST (expected blocked dir with no tokenizer)"
        return 1
    fi

    # Copy config.json + tokenizer.* from 67400 HF (READY) into 72800 HF.
    for fname in config.json tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
        if [[ -f "$SRC/$fname" ]]; then
            echo "[COPY] $fname $SRC -> $DST"
            cp -v "$SRC/$fname" "$DST/"
        fi
    done

    # Validate after copy.
    if "$CONVERT_PY" -c "from torchtitan.experiments.ezpz.eval.hf_export import validate_export; validate_export('$DST')" 2>&1; then
        echo "[OK] 72800 fixup validated: $DST"
    else
        echo "[FAIL] 72800 fixup validate_export failed: $DST"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 1) Fix 72800: copy config.json + tokenizer.* from 67400 HF
# ---------------------------------------------------------------------------
fix_72800 || echo "[WARN] 72800 fixup encountered errors (see above)"

# ---------------------------------------------------------------------------
# 2) Fix 13000: reconvert (empty HF -> fresh convert)
# ---------------------------------------------------------------------------
echo "=== FIXUP 13000 (reconvert empty HF) ==="
# Remove empty/stale HF dir if present so staged_export can claim it.
if [[ -d "$HF_BASE/step-13000/hf" ]]; then
    if [[ -z "$(ls -A "$HF_BASE/step-13000/hf" 2>/dev/null)" ]]; then
        echo "[CLEAN] removing empty HF dir $HF_BASE/step-13000/hf"
        rmdir "$HF_BASE/step-13000/hf" 2>/dev/null || rm -rf "$HF_BASE/step-13000/hf"
    else
        echo "[INFO] 13000 HF not empty -- will validate/skip or reconvert via convert_step"
    fi
fi
if ! convert_step 13000; then
    echo "[WARN] 13000 reconvert failed"
fi

# ---------------------------------------------------------------------------
# 3) Batch converts 74400 -> 111000 every 200
# ---------------------------------------------------------------------------
echo "=== BATCH 74400 -> 111000 step 200 ==="

FAILED=0
OK=0
SKIPPED=0

for s in $(seq 74400 200 111000); do
    DCP="$DCP_BASE/step-$s"
    HF="$HF_BASE/step-$s/hf"

    # Log and run; don't abort batch on single failure.
    set +e
    convert_step "$s"
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        # Distinguish OK vs SKIP by checking log marker; count conservatively.
        if [[ -f "$HF/ezpz_export.json" ]]; then
            OK=$((OK + 1))
        else
            SKIPPED=$((SKIPPED + 1))
        fi
        echo "[RESULT] step-$s rc=0 OK=$OK SKIP=$SKIPPED FAIL=$FAILED"
    else
        FAILED=$((FAILED + 1))
        echo "[RESULT] step-$s rc=$rc OK=$OK SKIP=$SKIPPED FAIL=$FAILED"
    fi
done

echo "HF-BATCH-CPT-FWE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) OK=$OK SKIPPED=$SKIPPED FAILED=$FAILED"
if [[ $FAILED -gt 0 ]]; then
    echo "BATCH finished with $FAILED failures -- check logs above"
    exit 1
fi
