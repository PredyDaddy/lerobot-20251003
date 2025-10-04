#!/usr/bin/env bash
set -euo pipefail

# Activate conda env if needed
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot

POLICY_PATH="/mnt/data/cqy/workspace/lerobot-20251003/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model"
ROBOT_PORT="/dev/ttyUSB0"
FPS=20
USE_AMP=true
DEVICE=cuda
MAX_REL_TARGET=10.0

CAMERAS_JSON='{"laptop": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "phone": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}'

python my_workspace/inference/koch_inference.py \
  --policy.path "$POLICY_PATH" \
  --device "$DEVICE" \
  --use_amp "$USE_AMP" \
  --fps "$FPS" \
  --robot.port "$ROBOT_PORT" \
  --robot.cameras "$CAMERAS_JSON" \
  --robot.max_relative_target "$MAX_REL_TARGET" \
  --robot.id koch_diff_inf

