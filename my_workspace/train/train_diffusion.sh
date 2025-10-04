#!/bin/bash

# Diffusion Policy训练脚本
# 基于新框架的LeRobot训练脚本
# 数据集: grasp_dataset_v30
# 策略: Diffusion Policy

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

# 训练参数配置
POLICY_TYPE="diffusion"
DEVICE="cuda"
BATCH_SIZE=64  # Diffusion Policy通常需要较小的batch size
NUM_WORKERS=8
STEPS=500000
SAVE_FREQ=4000
LOG_FREQ=200
EVAL_FREQ=0

# Diffusion Policy特定参数
N_OBS_STEPS=2
HORIZON=16
N_ACTION_STEPS=8

# 生成带时间戳的日志文件名
LOG_FILE="logs/diffusion_train_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "开始Diffusion Policy训练"
echo "时间: $(date)"
echo "数据集: $DATASET_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "策略类型: $POLICY_TYPE"
echo "批次大小: $BATCH_SIZE"
echo "训练步数: $STEPS"
echo "观察步数: $N_OBS_STEPS"
echo "预测范围: $HORIZON"
echo "动作步数: $N_ACTION_STEPS"
echo "========================================"

# 运行训练脚本（使用相对路径）
python ${PROJECT_ROOT}/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=grasp_dataset_v30 \
    --dataset.root=$DATASET_PATH \
    --policy.type=$POLICY_TYPE \
    --policy.device=$DEVICE \
    --policy.push_to_hub=false \
    --policy.n_obs_steps=$N_OBS_STEPS \
    --policy.horizon=$HORIZON \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.vision_backbone=resnet18 \
    --policy.use_group_norm=true \
    --policy.spatial_softmax_num_keypoints=32 \
    --policy.crop_shape="[84,84]" \
    --policy.crop_is_random=true \
    --policy.down_dims="[512,1024,2048]" \
    --policy.kernel_size=5 \
    --policy.n_groups=8 \
    --policy.diffusion_step_embed_dim=128 \
    --policy.use_film_scale_modulation=true \
    --policy.noise_scheduler_type=DDPM \
    --policy.num_train_timesteps=100 \
    --policy.beta_schedule=squaredcos_cap_v2 \
    --policy.prediction_type=epsilon \
    --policy.clip_sample=true \
    --policy.clip_sample_range=1.0 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_weight_decay=1e-6 \
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