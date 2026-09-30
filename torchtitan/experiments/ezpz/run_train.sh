#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# Set COMM_BACKEND="fake" for dry-run validation without GPU execution:
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_BACKEND="fake" ./run_train.sh

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi

MODEL="${MODEL:-2b}"

_fallback_dfl="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"
DFL="${DFL:-${DATA_FILE_LIST:-${_fallback_dfl}}}"

MODULE=${MODULE:-"ezpz.agpt"}
CONFIG=${CONFIG:-"ezpz_agpt_${MODEL}"}

NGPU=${NGPU:-${NGPUS:-${WORLD_SIZE:-4}}}
COMM_BACKEND=${COMM_BACKEND:-""}

export LOG_RANK=${LOG_RANK:-0}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

CHECKPOINT_DIR="aGPT-${MODEL}-ws${NGPU}-$(basename "${DFL}")"

if [[ -n "$COMM_BACKEND" && "$COMM_BACKEND" != "fake" ]]; then
    echo "COMM_BACKEND must be empty or fake, got: ${COMM_BACKEND}" >&2
    exit 1
fi

if [ "$COMM_BACKEND" = "fake" ]; then
    echo "Running with fake process groups"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.experiments.ezpz.train \
        --module "${MODULE}" \
        --config "${CONFIG}" \
        --comm.backend=fake \
        --training.steps 1 \
        "$@"
else
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --debug.print_config \
        --module "${MODULE}" \
        --config "${CONFIG}" \
        --training.dataset_path "${DFL}" \
        --checkpoint.folder "${CHECKPOINT_DIR}" \
        "$@"
fi
