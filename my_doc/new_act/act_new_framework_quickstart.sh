#!/usr/bin/env bash
# Quickstart: Train ACT on new LeRobot framework (v0.3.4) and run async inference
# Usage:
#   bash act_new_framework_quickstart.sh help
#   bash act_new_framework_quickstart.sh convert     # v2.1 -> v3.0 dataset
#   bash act_new_framework_quickstart.sh smoke       # short smoke run
#   bash act_new_framework_quickstart.sh train       # longer training
#   bash act_new_framework_quickstart.sh server      # start policy server
#   bash act_new_framework_quickstart.sh client      # start robot client (edit ports/cameras first)

set -euo pipefail

# --- Paths (edit if needed) ---
NEW_FRAMEWORK_DIR="/home/chenqingyu/robot/new_lerobot/lerobot-20251011"
DATASET_V21="/home/chenqingyu/robot/new_lerobot/grasp_dataset"
DATASET_V30="/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30"
OUT_DIR_SMOKE="/home/chenqingyu/robot/outputs/act_grasp_v3_smoke"
OUT_DIR_TRAIN="/home/chenqingyu/robot/outputs/act_grasp_v3"
HF_CACHE="/home/chenqingyu/robot/.cache/hf_v30"

# Required by new framework; use a non-empty string for local datasets
REPO_ID="grasp_dataset"

# Robot/Async (edit for your setup)
SERVER_HOST="0.0.0.0"
SERVER_PORT="8080"
SERVER_FPS="30"
SERVER_INFER_LAT="0.02"
SERVER_OBS_TIMEOUT="1"

ROBOT_TYPE="koch_follower"
ROBOT_PORT="/dev/ttyACM0"
# Dataset cameras are observation.images.laptop and observation.images.phone
ROBOT_CAMERAS='{ laptop: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, phone: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30} }'
TASK_TEXT="抓取工作台上的物体并放入收纳盒。"
CLIENT_SERVER_ADDR="127.0.0.1:8080"
POLICY_DEVICE="cuda"
ACTIONS_PER_CHUNK="100"
CHUNK_THRESH="0.5"

PRETRAINED_PATH_SMOKE="$OUT_DIR_SMOKE/checkpoints/last/pretrained_model"
PRETRAINED_PATH_TRAIN="$OUT_DIR_TRAIN/checkpoints/last/pretrained_model"

cmd_train() {
  if command -v lerobot-train >/dev/null 2>&1; then
    echo "[i] Using lerobot-train entrypoint"
    lerobot-train "$@"
  else
    echo "[i] Using module fallback"
    python -m lerobot.scripts.lerobot_train "$@"
  fi
}

ensure_env() {
  export HF_DATASETS_CACHE="$HF_CACHE"
  export LR_NEW_FRAMEWORK_DIR="$NEW_FRAMEWORK_DIR"
  python - <<'PYBLOCK'
import os, sys, importlib, subprocess
try:
    import lerobot
    print('[i] lerobot import OK')
except Exception:
    path = os.environ.get('LR_NEW_FRAMEWORK_DIR')
    if not path:
        raise SystemExit('[!] LR_NEW_FRAMEWORK_DIR not set')
    print('[i] Installing new framework in editable mode:', path)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', path])
    importlib.import_module('lerobot')
    print('[i] lerobot import OK after install')
PYBLOCK
}

help_msg() {
cat <<EOF
Commands:
  help       Show this help
  convert    Convert v2.1 dataset -> v3.0 (writes to $DATASET_V30)
  smoke      Short ACT training (sanity check)
  train      Longer ACT training
  server     Start async policy server (GPU side)
  client     Start robot client (Koch example; edit ports/cameras)

Notes:
- Ensure Python >= 3.10 and CUDA/PyTorch match your system.
- Cameras keys must match dataset cameras: laptop, phone.
- For local training, keep --policy.push_to_hub=false.
EOF
}

convert_dataset() {
  ensure_env
  echo "[i] Converting dataset v2.1 -> v3.0"
  python scripts/local_convert_v21_to_v30.py \
    --input-dir "$DATASET_V21" \
    --output-dir "$DATASET_V30" \
    --data-mb 100 --video-mb 500
  echo "[i] Done. Output: $DATASET_V30"
}

smoke_train() {
  ensure_env
  cmd_train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.root="$DATASET_V30" \
    --dataset.repo_id="$REPO_ID" \
    --batch_size=8 \
    --num_workers=8 \
    --steps=200 \
    --save_freq=200 \
    --eval_freq=0 \
    --output_dir="$OUT_DIR_SMOKE" \
    --job_name=act_grasp_v3_smoke \
    --wandb.enable=false
}

full_train() {
  ensure_env
  cmd_train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.root="$DATASET_V30" \
    --dataset.repo_id="$REPO_ID" \
    --batch_size=24 \
    --num_workers=8 \
    --steps=100000 \
    --save_freq=5000 \
    --eval_freq=0 \
    --output_dir="$OUT_DIR_TRAIN" \
    --job_name=act_grasp_v3 \
    --wandb.enable=false
}

start_server() {
  ensure_env
  python - <<'PYBLOCK'
from importlib.util import find_spec
import sys
if find_spec('lerobot.async_inference') is None:
    print('[!] async_inference not found in lerobot; check installation.')
    sys.exit(1)
PYBLOCK
  python -m lerobot.async_inference.policy_server \
    --host="$SERVER_HOST" \
    --port="$SERVER_PORT" \
    --fps="$SERVER_FPS" \
    --inference_latency="$SERVER_INFER_LAT" \
    --obs_queue_timeout="$SERVER_OBS_TIMEOUT"
}

start_client() {
  ensure_env
  local pretrained_path="${1:-}"
  if [[ -z "$pretrained_path" ]]; then
    echo "[!] Provide pretrained path: smoke|train|<absolute_path>"
    echo "    smoke -> $PRETRAINED_PATH_SMOKE"
    echo "    train -> $PRETRAINED_PATH_TRAIN"
    return 1
  fi
  if [[ "$pretrained_path" == "smoke" ]]; then
    pretrained_path="$PRETRAINED_PATH_SMOKE"
  elif [[ "$pretrained_path" == "train" ]]; then
    pretrained_path="$PRETRAINED_PATH_TRAIN"
  fi
  python -m lerobot.async_inference.robot_client \
    --robot.type="$ROBOT_TYPE" \
    --robot.port="$ROBOT_PORT" \
    --robot.cameras="$ROBOT_CAMERAS" \
    --task="$TASK_TEXT" \
    --server_address="$CLIENT_SERVER_ADDR" \
    --policy_type=act \
    --pretrained_name_or_path="$pretrained_path" \
    --policy_device="$POLICY_DEVICE" \
    --actions_per_chunk="$ACTIONS_PER_CHUNK" \
    --chunk_size_threshold="$CHUNK_THRESH" \
    --fps="$SERVER_FPS"
}

case "${1:-help}" in
  help)    help_msg ;;
  convert) convert_dataset ;;
  smoke)   smoke_train ;;
  train)   full_train ;;
  server)  start_server ;;
  client)  shift || true; start_client "${1:-}" ;;
  *)       echo "Unknown command: $1"; help_msg; exit 1 ;;
 esac