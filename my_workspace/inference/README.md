# Koch机械臂推理脚本说明

本目录包含用于在Koch机械臂上部署训练好的ACT和Diffusion模型的推理脚本。

---

## 📁 文件列表

### Shell脚本（推荐新手使用）

1. **`run_act_inference.sh`**
   - ACT模型推理脚本
   - 自动化配置和硬件检查
   - 包含安全提示和错误处理

2. **`run_diffusion_inference.sh`**
   - Diffusion模型推理脚本
   - 针对Diffusion特性优化
   - 包含延迟提示

### Python脚本（高级用户）

3. **`koch_inference.py`**
   - 通用推理脚本（支持ACT和Diffusion）
   - 提供更多命令行参数
   - 适合自定义配置

---

## 🚀 快速开始

### 方法1：使用Shell脚本（推荐）

```bash
# 1. 检查硬件
bash ../scripts/check_hardware.sh

# 2. 修改脚本中的相机配置（如果需要）
vim run_act_inference.sh
# 修改 CAMERA_LAPTOP_INDEX 和 CAMERA_PHONE_INDEX

# 3. 运行推理
bash run_act_inference.sh
```

### 方法2：使用Python脚本

```bash
# ACT推理
python koch_inference.py \
    --policy_path ../../output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model \
    --robot_port /dev/ttyUSB0 \
    --camera_laptop_index 0 \
    --camera_phone_index 2 \
    --num_episodes 5 \
    --save_data

# Diffusion推理
python koch_inference.py \
    --policy_path ../../output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model \
    --robot_port /dev/ttyUSB0 \
    --camera_laptop_index 0 \
    --camera_phone_index 2 \
    --num_episodes 5 \
    --save_data
```

---

## ⚙️ 配置说明

### 必须修改的配置

在运行脚本前，根据实际硬件修改以下配置：

1. **机器人端口**（`ROBOT_PORT`）
   - 默认：`/dev/ttyUSB0`
   - 查找：`ls /dev/ttyUSB*`

2. **相机索引**（`CAMERA_LAPTOP_INDEX`, `CAMERA_PHONE_INDEX`）
   - 默认：0 和 2
   - 查找：运行 `bash ../scripts/check_hardware.sh`

### 可选配置

- `NUM_EPISODES`: Episode数量（默认5）
- `EPISODE_TIME_S`: 每个episode时长（默认30秒）
- `MAX_RELATIVE_TARGET`: 安全限幅（默认10度）
- `DISPLAY_DATA`: 是否启用可视化（默认true）

---

## 📊 输出说明

### 评估数据

推理数据保存在 `../../eval_data/` 目录：

```
eval_data/
├── act_eval_20251004_120000/
│   ├── meta/
│   │   ├── info.json
│   │   └── stats.json
│   ├── videos/
│   │   ├── chunk-000/
│   │   │   ├── observation.images.laptop.mp4
│   │   │   └── observation.images.phone.mp4
│   └── data/
│       └── chunk-000/
│           └── episode_000000.parquet
└── diffusion_eval_20251004_130000/
    └── ...
```

### 可视化

如果启用了 `--display_data`，会自动打开Rerun可视化界面，实时显示：
- 相机图像
- 机器人状态
- 预测动作
- 执行轨迹

---

## 🔍 故障排查

### 常见错误

1. **`DeviceNotConnectedError`**
   - 检查USB连接
   - 检查端口权限：`sudo chmod 666 /dev/ttyUSB0`

2. **`cv2.error: !_src.empty()`**
   - 相机索引错误
   - 运行硬件检查脚本确认正确的索引

3. **`FileNotFoundError: model.safetensors`**
   - 模型路径错误
   - 检查 `output/` 目录中的实际路径

4. **推理延迟过高**
   - 检查GPU是否可用：`nvidia-smi`
   - Diffusion模型需要更多计算资源

### 调试技巧

```bash
# 1. 测试机器人连接
lerobot-calibrate --robot.type=koch_follower --robot.port=/dev/ttyUSB0

# 2. 测试相机
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# 3. 测试模型加载
python -c "
from lerobot.policies.factory import make_policy
policy = make_policy.from_pretrained('output/act_train_xxx/checkpoints/152000/pretrained_model')
print(f'Model loaded: {policy.name}')
"
```

---

## 📚 相关文档

- **快速开始**: `../docs/quick_start_inference.md`
- **完整指南**: `../docs/koch_inference_deployment_guide.md`
- **硬件检查**: `../scripts/check_hardware.sh`

---

## ⚠️ 安全提示

**每次运行前必须**：
1. 清理机械臂周围障碍物
2. 确认紧急停止按钮可用
3. 检查相机视野
4. 准备随时按Ctrl+C停止

**首次运行建议**：
- 降低episode时长（如10秒）
- 增加安全限幅（如5度）
- 仔细观察机器人动作

---

## 🤝 贡献

如果发现问题或有改进建议，请：
1. 记录详细的错误信息
2. 提供硬件配置
3. 描述复现步骤

---

**祝推理顺利！** 🎉

