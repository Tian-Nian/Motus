#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_SH="/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/etc/profile.d/conda.sh"
MOTUS_PYTHON_DEFAULT="/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/envs/motus/bin/python"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate motus
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate motus
fi

PYTHON_BIN="${PYTHON_BIN:-$MOTUS_PYTHON_DEFAULT}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python)"
fi

CONFIG_FILE="configs/robotwin_xspark.yaml"
DEEPSPEED_CONFIG="configs/zero2_stage2.json"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-}"
RUN_NAME="${RUN_NAME:-robotwin_xspark}"
REPORT_TO="${REPORT_TO:-tensorboard}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints_xspark}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT="$({ "$PYTHON_BIN" - <<'PY'
import socket

for port in range(29500, 29600):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("No free master port found in 29500-29599")
PY
    })"
fi

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using MASTER_PORT=$MASTER_PORT"
echo "Using PYTHON_BIN=$PYTHON_BIN"
echo "Using DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"

if [ "$TORCHRUN_BIN" = "torchrun" ]; then
    "$PYTHON_BIN" -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank=0 \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        train/train.py \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --config "$CONFIG_FILE" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --run_name "$RUN_NAME" \
        --report_to "$REPORT_TO"
else
    "$TORCHRUN_BIN" \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --config "$CONFIG_FILE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO"
fi