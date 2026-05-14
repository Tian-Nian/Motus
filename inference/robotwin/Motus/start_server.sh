#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WEIGHTS_ROOT="/mnt/pfs/pg4hw0/niantian/model_weights"
CHECKPOINT_PATH_DEFAULT="${REPO_ROOT}/checkpoints/robotwin/pytorch_model/mp_rank_00_model_states.pt"
WAN_PATH_DEFAULT="${WEIGHTS_ROOT}/Wan2.2-TI2V-5B"
VLM_PATH_DEFAULT="${WEIGHTS_ROOT}/Qwen3-VL-2B-Instruct"

GPU_ID="${1:-0}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8094}"
CHECKPOINT_PATH="${4:-${CHECKPOINT_PATH_DEFAULT}}"
WAN_PATH="${5:-${WAN_PATH_DEFAULT}}"
VLM_PATH="${6:-${VLM_PATH_DEFAULT}}"
LOG_DIR="${7:-${SCRIPT_DIR}/logs_server}"
TASK_NAME="${8:-robotwin}"

if [[ ! -f "${CHECKPOINT_PATH}" && ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "Error: checkpoint path not found: ${CHECKPOINT_PATH}"
    exit 1
fi

if [[ ! -d "${WAN_PATH}" ]]; then
    echo "Error: WAN path not found: ${WAN_PATH}"
    exit 1
fi

if [[ ! -d "${VLM_PATH}" ]]; then
    echo "Error: VLM path not found: ${VLM_PATH}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=============================================================="
echo "Motus RoboTwin Server"
echo "=============================================================="
echo "Repo Root:        ${REPO_ROOT}"
echo "GPU ID:           ${GPU_ID}"
echo "Host:             ${HOST}"
echo "Port:             ${PORT}"
echo "Checkpoint:       ${CHECKPOINT_PATH}"
echo "WAN Path:         ${WAN_PATH}"
echo "VLM Path:         ${VLM_PATH}"
echo "Log Dir:          ${LOG_DIR}"
echo "Task Name:        ${TASK_NAME}"
echo "=============================================================="

cd "${REPO_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
python "${SCRIPT_DIR}/server.py" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --log_dir "${LOG_DIR}" \
    --task_name "${TASK_NAME}"