#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# Stage 3 of the MDS154391 lineage: continue-SFT the immutable Stage-2
# checkpoint on teacher-free STaR traces that the model itself produced and
# that passed exact-answer, envelope, termination, reasoning-depth,
# duplicate, and GSM8K-test-decontamination filters.
#
# The corpus comes from stage3_star_generate.py; this script refuses to train
# unless that artifact exists, reports zero test collisions, and carries at
# least STAGE3_MIN_ACCEPTED accepted problems.
set -u
set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214}"
MODEL_PATH="${MODEL_PATH:-/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/stage2/agpt2b-mds154391-step600-gsm8k-r1cot-2n-r2/final}"
STAR_DIR="${STAR_DIR:?set STAR_DIR to the stage3_star_generate.py output directory}"
CKPT_DIR="${CKPT_DIR:-/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/stage3/sft-${PBS_JOBID%%.*}}"
STAGE3_MIN_ACCEPTED="${STAGE3_MIN_ACCEPTED:-2000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LR="${LR:-5e-6}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-25}"
: "${EXPECTED_COMMIT:?submit with -v EXPECTED_COMMIT=<exact-pushed-sha>}"

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export CCL_ATL_SYNC_COLL=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup "${VENV}"
cd "${SUBMIT_DIR}" || exit 11
actual_commit=$(git rev-parse HEAD) || exit 15
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
    echo "FATAL: commit mismatch expected=${EXPECTED_COMMIT} actual=${actual_commit}"
    exit 15
}
git diff --quiet && git diff --cached --quiet || { echo "FATAL: dirty worktree"; exit 15; }

ACCEPTED="${STAR_DIR%/}/accepted_sft.jsonl"
REPORT="${STAR_DIR%/}/report.json"
[[ -s "${MODEL_PATH}/model.safetensors" ]] || { echo "FATAL: base model missing"; exit 16; }
[[ -s "${ACCEPTED}" ]] || { echo "FATAL: accepted corpus missing: ${ACCEPTED}"; exit 16; }
[[ -s "${REPORT}" ]] || { echo "FATAL: corpus report missing: ${REPORT}"; exit 16; }
[[ ! -e "${CKPT_DIR}" ]] || { echo "FATAL: output exists: ${CKPT_DIR}"; exit 17; }

python3 - "${REPORT}" "${ACCEPTED}" "${STAGE3_MIN_ACCEPTED}" <<'PY' || exit 18
import json
import sys

report_path, accepted_path, minimum = sys.argv[1], sys.argv[2], int(sys.argv[3])
report = json.loads(open(report_path).read())
rows = sum(1 for line in open(accepted_path) if line.strip())
if report["test_collisions"]:
    raise SystemExit(f"FATAL: {report['test_collisions']} GSM8K-test collisions")
if report["accepted_problems"] != rows:
    raise SystemExit(
        f"FATAL: report claims {report['accepted_problems']} accepted rows; file has {rows}"
    )
if rows < minimum:
    raise SystemExit(f"FATAL: {rows} accepted rows below required {minimum}")
print(
    "STAGE3_SFT_CORPUS_OK rows=%d acceptance_rate=%.4f" % (rows, report["acceptance_rate"])
)
PY

export EZPZ_STAGE3_STAR_PATH="${ACCEPTED}"
LOG_DIR="logs/stage3-sft-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
model_sha=$(sha256sum "${MODEL_PATH}/model.safetensors" | cut -d' ' -f1)
echo "STAGE3_SFT_START commit=${actual_commit} model_sha=${model_sha} corpus=${ACCEPTED} out=${CKPT_DIR} $(date -Is)" \
    | tee "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset stage3-star \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length "${MAX_LENGTH}" \
    --seed 42 \
    --bf16 --fsdp full_shard \
    --gradient_checkpointing \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 4 \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log"
rc=${PIPESTATUS[0]}
echo "STAGE3_SFT_DONE rc=${rc} $(date -Is)" | tee -a "${LOG_DIR}/run.log"
exit "${rc}"
