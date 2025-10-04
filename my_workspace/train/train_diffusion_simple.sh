#!/bin/bash

# 简化版 Diffusion Policy 训练脚本
# 使用大部分默认参数，只配置必要的参数

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
# 获取项目根目录（脚本所在目录的上两级）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# 检查conda环境
echo "检查当前conda环境: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "lerobot_v3" ]; then
    echo "警告: 当前不在lerobot_v3环境中"
    echo "请手动运行: conda activate lerobot_v3"
    echo "然后再运行此脚本"
    exit 1
fi
echo "当前环境: $CONDA_DEFAULT_ENV ✓"

# 检查数据集是否存在（使用相对路径）
DATASET_PATH="${PROJECT_ROOT}/grasp_dataset_v30"
if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集路径不存在: $DATASET_PATH"
    exit 1
fi

# 设置输出目录路径（使用相对路径，包含时间戳和随机数避免冲突）
OUTPUT_BASE="${PROJECT_ROOT}/output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RANDOM_ID=$(shuf -i 1000-9999 -n 1)
OUTPUT_PATH="${OUTPUT_BASE}/diffusion_train_${TIMESTAMP}_${RANDOM_ID}"
echo "输出目录: $OUTPUT_PATH"

# 确保输出目录不存在
if [ -d "$OUTPUT_PATH" ]; then
    echo "错误: 输出目录已存在，重新生成"
    RANDOM_ID=$(shuf -i 1000-9999 -n 1)
    OUTPUT_PATH="${OUTPUT_BASE}/diffusion_train_${TIMESTAMP}_${RANDOM_ID}"
    echo "新的输出目录: $OUTPUT_PATH"
fi

# 必要的训练参数配置
POLICY_TYPE="diffusion"
DEVICE="cuda"
BATCH_SIZE=64  # Diffusion Policy通常需要较小的batch size
NUM_WORKERS=8
STEPS=500000
SAVE_FREQ=4000
LOG_FREQ=200
EVAL_FREQ=0

# 生成带时间戳的日志文件名
LOG_FILE="logs/diffusion_train_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "开始Diffusion Policy训练（简化版）"
echo "时间: $(date)"
echo "数据集: $DATASET_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "策略类型: $POLICY_TYPE"
echo "批次大小: $BATCH_SIZE"
echo "训练步数: $STEPS"
echo "========================================"
echo "注意：大部分参数使用默认值"
echo "========================================"

# 运行训练脚本（只使用必要参数，其他使用默认值）
python ${PROJECT_ROOT}/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=grasp_dataset_v30 \
    --dataset.root=$DATASET_PATH \
    --policy.type=$POLICY_TYPE \
    --policy.device=$DEVICE \
    --policy.push_to_hub=false \
    --output_dir=$OUTPUT_PATH \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --log_freq=$LOG_FREQ \
    --eval_freq=$EVAL_FREQ \
    --wandb.enable=false \
    --job_name=diffusion_grasp_train \
    2>&1 | tee $LOG_FILE

# 检查训练是否成功完成
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "========================================"
    echo "训练完成!"
    echo "时间: $(date)"
    echo "日志文件: $LOG_FILE"
    echo "模型保存在: $OUTPUT_PATH"
    echo "========================================"
else
    echo "========================================"
    echo "训练失败!"
    echo "请检查日志文件: $LOG_FILE"
    echo "========================================"
    exit 1
fi