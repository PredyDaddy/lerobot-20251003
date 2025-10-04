#!/bin/bash

# ============================================================================
# Diffusion Policy 推理脚本
# ============================================================================
# 功能：在Koch机械臂上运行训练好的Diffusion模型进行实时推理
# 作者：AI Assistant
# 日期：2025-10-04
# 基于：LeRobot lerobot-record脚本
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "项目根目录: $PROJECT_ROOT"

# 检查conda环境
if [ "$CONDA_DEFAULT_ENV" != "lerobot_v3" ]; then
    echo "错误: 当前不在lerobot_v3环境中"
    echo "请运行: conda activate lerobot_v3"
    exit 1
fi
echo "✓ Conda环境: $CONDA_DEFAULT_ENV"

# ============================================================================
# 硬件配置（请根据实际情况修改）
# ============================================================================

# Koch机械臂USB端口
ROBOT_PORT="/dev/ttyUSB0"

# 相机设备路径（请根据实际情况修改）
CAMERA_LAPTOP_INDEX=0
CAMERA_PHONE_INDEX=2

# 相机分辨率和帧率（必须与训练数据集一致）
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# ============================================================================
# 模型配置
# ============================================================================

# Diffusion模型检查点路径
MODEL_PATH="${PROJECT_ROOT}/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请检查训练输出目录"
    exit 1
fi
echo "✓ 模型路径: $MODEL_PATH"

# ============================================================================
# 推理配置
# ============================================================================

# 评估数据保存路径
EVAL_DATA_ROOT="${PROJECT_ROOT}/eval_data/diffusion_eval_$(date +%Y%m%d_%H%M%S)"

# 推理参数
NUM_EPISODES=5
EPISODE_TIME_S=30
CONTROL_FPS=30

# 任务描述
TASK_DESCRIPTION="Grasp and place object"

# 安全参数
MAX_RELATIVE_TARGET=10.0

# 可视化
DISPLAY_DATA=true

# ============================================================================
# 硬件检查
# ============================================================================

echo ""
echo "========================================"
echo "硬件检查"
echo "========================================"

# 检查机器人端口
if [ ! -e "$ROBOT_PORT" ]; then
    echo "错误: 机器人端口不存在: $ROBOT_PORT"
    echo "可用端口:"
    ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  未找到USB串口设备"
    exit 1
fi
echo "✓ 机器人端口: $ROBOT_PORT"

# 检查端口权限
if [ ! -r "$ROBOT_PORT" ] || [ ! -w "$ROBOT_PORT" ]; then
    echo "警告: 端口权限不足"
    echo "尝试授权: sudo chmod 666 $ROBOT_PORT"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查相机设备
echo "检查相机设备..."
if [ -e "/dev/video${CAMERA_LAPTOP_INDEX}" ]; then
    echo "✓ 相机laptop: /dev/video${CAMERA_LAPTOP_INDEX}"
else
    echo "警告: 相机laptop设备不存在: /dev/video${CAMERA_LAPTOP_INDEX}"
fi

if [ -e "/dev/video${CAMERA_PHONE_INDEX}" ]; then
    echo "✓ 相机phone: /dev/video${CAMERA_PHONE_INDEX}"
else
    echo "警告: 相机phone设备不存在: /dev/video${CAMERA_PHONE_INDEX}"
fi

# ============================================================================
# 安全提示
# ============================================================================

echo ""
echo "========================================"
echo "安全提示"
echo "========================================"
echo "1. 确保机械臂周围无障碍物"
echo "2. 准备好随时按下紧急停止按钮"
echo "3. Diffusion推理延迟较高，注意观察"
echo "4. 按Ctrl+C可随时安全停止"
echo "5. 当前安全限幅: ${MAX_RELATIVE_TARGET}度/步"
echo ""
echo "注意: Diffusion需要2步观测历史，首次推理会有延迟"
echo ""
read -p "确认已阅读安全提示，按Enter继续..." 

# ============================================================================
# 运行推理
# ============================================================================

echo ""
echo "========================================"
echo "开始Diffusion推理"
echo "========================================"
echo "模型: Diffusion Policy"
echo "检查点: $MODEL_PATH"
echo "控制频率: ${CONTROL_FPS}Hz"
echo "Episode数量: $NUM_EPISODES"
echo "Episode时长: ${EPISODE_TIME_S}秒"
echo "评估数据: $EVAL_DATA_ROOT"
echo "========================================"
echo ""

# 构建相机配置JSON
CAMERA_CONFIG="{
    laptop: {
        type: opencv,
        index_or_path: ${CAMERA_LAPTOP_INDEX},
        width: ${CAMERA_WIDTH},
        height: ${CAMERA_HEIGHT},
        fps: ${CAMERA_FPS}
    },
    phone: {
        type: opencv,
        index_or_path: ${CAMERA_PHONE_INDEX},
        width: ${CAMERA_WIDTH},
        height: ${CAMERA_HEIGHT},
        fps: ${CAMERA_FPS}
    }
}"

# 运行lerobot-record进行推理
lerobot-record \
    --robot.type=koch_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.cameras="$CAMERA_CONFIG" \
    --robot.id=koch_diffusion_inference \
    --robot.max_relative_target=$MAX_RELATIVE_TARGET \
    --policy.path="$MODEL_PATH" \
    --dataset.repo_id=None \
    --dataset.root="$EVAL_DATA_ROOT" \
    --dataset.num_episodes=$NUM_EPISODES \
    --dataset.episode_time_s=$EPISODE_TIME_S \
    --dataset.fps=$CONTROL_FPS \
    --dataset.single_task="$TASK_DESCRIPTION" \
    --display_data=$DISPLAY_DATA

# ============================================================================
# 完成
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "推理完成!"
    echo "========================================"
    echo "评估数据已保存到: $EVAL_DATA_ROOT"
    echo ""
    echo "后续分析:"
    echo "1. 查看Rerun可视化回放"
    echo "2. 分析成功率和轨迹质量"
    echo "3. 对比ACT和Diffusion的性能差异"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "推理失败!"
    echo "========================================"
    echo "请检查:"
    echo "1. 机器人连接是否正常"
    echo "2. 相机是否可用"
    echo "3. 模型路径是否正确"
    echo "4. GPU显存是否充足（Diffusion需要更多显存）"
    echo "========================================"
    exit 1
fi

