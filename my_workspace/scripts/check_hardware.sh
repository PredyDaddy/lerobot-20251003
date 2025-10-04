#!/bin/bash

# ============================================================================
# Koch机械臂硬件检查脚本
# ============================================================================
# 功能：检查推理前的所有硬件和软件配置
# 作者：AI Assistant
# 日期：2025-10-04
# ============================================================================

echo "========================================"
echo "Koch机械臂硬件检查"
echo "========================================"
echo ""

# ============================================================================
# 1. USB端口检查
# ============================================================================
echo "1. USB端口检查："
echo "----------------------------------------"

USB_DEVICES=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null)
if [ -z "$USB_DEVICES" ]; then
    echo "  ✗ 未找到USB串口设备"
    echo "  提示：检查机械臂是否已连接"
else
    echo "$USB_DEVICES" | while read device; do
        echo "  ✓ 找到设备: $device"
        
        # 检查权限
        if [ -r "$device" ] && [ -w "$device" ]; then
            echo "    权限: OK"
        else
            echo "    权限: 不足（需要读写权限）"
            echo "    解决: sudo chmod 666 $device"
        fi
    done
fi
echo ""

# ============================================================================
# 2. 相机设备检查
# ============================================================================
echo "2. 相机设备检查："
echo "----------------------------------------"

VIDEO_DEVICES=$(ls /dev/video* 2>/dev/null)
if [ -z "$VIDEO_DEVICES" ]; then
    echo "  ✗ 未找到视频设备"
    echo "  提示：检查相机是否已连接"
else
    echo "$VIDEO_DEVICES" | while read device; do
        echo "  ✓ 找到设备: $device"
    done
    
    # 使用v4l2-ctl获取详细信息（如果可用）
    if command -v v4l2-ctl &> /dev/null; then
        echo ""
        echo "  详细信息："
        v4l2-ctl --list-devices 2>/dev/null | head -20
    fi
fi
echo ""

# ============================================================================
# 3. 用户权限检查
# ============================================================================
echo "3. 用户权限检查："
echo "----------------------------------------"

# 检查dialout组（串口权限）
if groups | grep -q dialout; then
    echo "  ✓ 用户在dialout组（串口权限）"
else
    echo "  ✗ 用户不在dialout组"
    echo "    解决: sudo usermod -a -G dialout $USER"
    echo "    注意: 需要重新登录生效"
fi

# 检查video组（相机权限）
if groups | grep -q video; then
    echo "  ✓ 用户在video组（相机权限）"
else
    echo "  ✗ 用户不在video组"
    echo "    解决: sudo usermod -a -G video $USER"
    echo "    注意: 需要重新登录生效"
fi
echo ""

# ============================================================================
# 4. Python环境检查
# ============================================================================
echo "4. Python环境检查："
echo "----------------------------------------"

# 检查conda环境
if command -v conda &> /dev/null; then
    CONDA_ENV=$(conda info | grep "active environment" | awk '{print $4}')
    if [ "$CONDA_ENV" = "lerobot_v3" ]; then
        echo "  ✓ Conda环境: $CONDA_ENV"
    else
        echo "  ✗ 当前环境: $CONDA_ENV"
        echo "    期望环境: lerobot_v3"
        echo "    解决: conda activate lerobot_v3"
    fi
else
    echo "  ✗ Conda未安装"
fi

# 检查Python版本
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  ✓ $PYTHON_VERSION"
else
    echo "  ✗ Python未找到"
fi
echo ""

# ============================================================================
# 5. 关键依赖检查
# ============================================================================
echo "5. 关键依赖检查："
echo "----------------------------------------"

# 检查PyTorch
python -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}')" 2>/dev/null || \
    echo "  ✗ PyTorch未安装"

# 检查OpenCV
python -c "import cv2; print(f'  ✓ OpenCV: {cv2.__version__}')" 2>/dev/null || \
    echo "  ✗ OpenCV未安装"

# 检查LeRobot
python -c "import lerobot; print(f'  ✓ LeRobot已安装')" 2>/dev/null || \
    echo "  ✗ LeRobot未安装"

# 检查Dynamixel SDK
python -c "import dynamixel_sdk; print(f'  ✓ Dynamixel SDK已安装')" 2>/dev/null || \
    echo "  ✗ Dynamixel SDK未安装"

# 检查serial
python -c "import serial; print(f'  ✓ PySerial已安装')" 2>/dev/null || \
    echo "  ✗ PySerial未安装"
echo ""

# ============================================================================
# 6. GPU检查（可选）
# ============================================================================
echo "6. GPU检查（可选）："
echo "----------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "    GPU: %s\n    显存: %s / %s\n", $1, $3, $2}'
    
    # 检查CUDA
    python -c "import torch; print(f'  ✓ CUDA可用: {torch.cuda.is_available()}'); \
               print(f'    CUDA版本: {torch.version.cuda}') if torch.cuda.is_available() else None" 2>/dev/null
else
    echo "  - GPU不可用（将使用CPU推理）"
fi
echo ""

# ============================================================================
# 7. 相机功能测试
# ============================================================================
echo "7. 相机功能测试："
echo "----------------------------------------"

python << 'EOF' 2>/dev/null
import cv2
import sys

cameras_found = []
for i in range(10):  # 检查前10个设备
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  ✓ 相机{i}: {width}x{height} @ {fps:.1f}fps")
        cameras_found.append(i)
        cap.release()

if not cameras_found:
    print("  ✗ 未找到可用相机")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "  ✗ 相机测试失败"
fi
echo ""

# ============================================================================
# 8. 模型检查点检查
# ============================================================================
echo "8. 模型检查点检查："
echo "----------------------------------------"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 检查ACT模型
ACT_MODEL="${PROJECT_ROOT}/output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model"
if [ -d "$ACT_MODEL" ]; then
    echo "  ✓ ACT模型: $ACT_MODEL"
    if [ -f "$ACT_MODEL/config.json" ]; then
        echo "    配置文件: 存在"
    fi
    if [ -f "$ACT_MODEL/model.safetensors" ]; then
        MODEL_SIZE=$(du -h "$ACT_MODEL/model.safetensors" | cut -f1)
        echo "    模型文件: $MODEL_SIZE"
    fi
else
    echo "  ✗ ACT模型未找到"
fi

# 检查Diffusion模型
DIFF_MODEL="${PROJECT_ROOT}/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model"
if [ -d "$DIFF_MODEL" ]; then
    echo "  ✓ Diffusion模型: $DIFF_MODEL"
    if [ -f "$DIFF_MODEL/config.json" ]; then
        echo "    配置文件: 存在"
    fi
    if [ -f "$DIFF_MODEL/model.safetensors" ]; then
        MODEL_SIZE=$(du -h "$DIFF_MODEL/model.safetensors" | cut -f1)
        echo "    模型文件: $MODEL_SIZE"
    fi
else
    echo "  ✗ Diffusion模型未找到"
fi
echo ""

# ============================================================================
# 总结
# ============================================================================
echo "========================================"
echo "检查完成"
echo "========================================"
echo ""
echo "下一步："
echo "1. 解决上述标记为 ✗ 的问题"
echo "2. 确保机械臂和相机已正确连接"
echo "3. 运行推理脚本："
echo "   bash my_workspace/inference/run_act_inference.sh"
echo "   或"
echo "   bash my_workspace/inference/run_diffusion_inference.sh"
echo ""

