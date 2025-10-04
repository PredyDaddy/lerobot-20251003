# Koch机械臂推理快速开始指南

> **5分钟快速上手推理部署**

---

## 🚀 快速开始（3步）

### 步骤1：硬件检查

```bash
# 运行硬件检查脚本
bash my_workspace/scripts/check_hardware.sh
```

**确保以下项目都显示 ✓**：
- USB端口存在且有权限
- 至少2个相机设备可用
- 用户在dialout和video组
- Conda环境为lerobot_v3
- 所有Python依赖已安装

### 步骤2：修改相机配置

根据硬件检查结果，编辑推理脚本中的相机索引：

```bash
# 编辑ACT推理脚本
vim my_workspace/inference/run_act_inference.sh

# 修改这两行（根据实际设备）
CAMERA_LAPTOP_INDEX=0  # 改为你的laptop相机索引
CAMERA_PHONE_INDEX=2   # 改为你的phone相机索引
```

### 步骤3：运行推理

```bash
# 激活环境
conda activate lerobot_v3

# 运行ACT推理
bash my_workspace/inference/run_act_inference.sh

# 或运行Diffusion推理
bash my_workspace/inference/run_diffusion_inference.sh
```

---

## 📝 推理脚本对比

### ACT推理

**优点**：
- 推理速度快（50-100ms）
- 控制频率高（30Hz）
- 适合精细操作

**命令**：
```bash
bash my_workspace/inference/run_act_inference.sh
```

**配置**：
- 模型：`output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model`
- Chunk size: 100步
- 推理频率：每100步推理1次

### Diffusion推理

**优点**：
- 动作更平滑
- 适合复杂长时序任务

**注意**：
- 推理延迟较高（200-500ms）
- 需要2步观测历史
- 首次推理会有延迟

**命令**：
```bash
bash my_workspace/inference/run_diffusion_inference.sh
```

**配置**：
- 模型：`output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model`
- Horizon: 16步
- 推理频率：每8步推理1次

---

## 🔧 常见问题快速修复

### 问题1：端口权限不足

```bash
# 临时解决
sudo chmod 666 /dev/ttyUSB0

# 永久解决
sudo usermod -a -G dialout $USER
# 重新登录
```

### 问题2：相机无法打开

```bash
# 检查相机设备
ls /dev/video*

# 测试相机
ffplay /dev/video0  # Ctrl+C退出

# 添加video组权限
sudo usermod -a -G video $USER
```

### 问题3：找不到模型

```bash
# 检查模型路径
ls -la output/act_train_*/checkpoints/*/pretrained_model/
ls -la output/diffusion_train_*/checkpoints/*/pretrained_model/

# 如果路径不同，修改脚本中的MODEL_PATH变量
```

### 问题4：机器人连接失败

```bash
# 检查USB设备
ls -l /dev/ttyUSB*

# 检查端口是否被占用
lsof /dev/ttyUSB0

# 重新插拔USB线
```

---

## 🎯 Python脚本使用（高级）

如果需要更多自定义选项，使用Python脚本：

### ACT推理

```bash
python my_workspace/inference/koch_inference.py \
    --policy_path output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model \
    --robot_port /dev/ttyUSB0 \
    --camera_laptop_index 0 \
    --camera_phone_index 2 \
    --num_episodes 5 \
    --episode_time_s 30 \
    --save_data \
    --display_data
```

### Diffusion推理

```bash
python my_workspace/inference/koch_inference.py \
    --policy_path output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model \
    --robot_port /dev/ttyUSB0 \
    --camera_laptop_index 0 \
    --camera_phone_index 2 \
    --num_episodes 5 \
    --episode_time_s 30 \
    --save_data \
    --display_data
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--policy_path` | 模型检查点路径 | 必填 |
| `--robot_port` | USB端口 | /dev/ttyUSB0 |
| `--camera_laptop_index` | Laptop相机索引 | 0 |
| `--camera_phone_index` | Phone相机索引 | 2 |
| `--num_episodes` | Episode数量 | 5 |
| `--episode_time_s` | Episode时长（秒） | 30 |
| `--max_relative_target` | 安全限幅（度） | 10.0 |
| `--save_data` | 保存评估数据 | False |
| `--display_data` | 启用可视化 | True |

---

## 📊 评估数据分析

推理完成后，评估数据保存在 `eval_data/` 目录：

```bash
# 查看评估数据
ls -la eval_data/

# 使用Rerun可视化回放
lerobot-dataset-viz --repo-id eval_data/act_eval_TIMESTAMP

# 分析成功率（需要手动标注）
# 可以在Rerun中查看每个episode的执行情况
```

---

## ⚠️ 安全检查清单

**每次运行前必须检查**：

- [ ] 机械臂周围无障碍物
- [ ] 紧急停止按钮可用
- [ ] 相机视野清晰
- [ ] USB连接稳定
- [ ] 电机温度正常
- [ ] 工作空间安全

**运行中监控**：

- [ ] 观察Rerun可视化
- [ ] 监听异常声音
- [ ] 检查动作是否合理
- [ ] 准备随时按Ctrl+C

---

## 📚 更多信息

详细文档：`my_workspace/docs/koch_inference_deployment_guide.md`

包含：
- 完整的硬件检查步骤
- 推理流程详解
- ACT vs Diffusion技术对比
- 异常处理机制
- 性能优化建议

---

## 🆘 获取帮助

如果遇到问题：

1. 查看详细文档
2. 运行硬件检查脚本
3. 检查日志输出
4. 查看LeRobot官方文档

**常用命令**：

```bash
# 查看LeRobot版本
python -c "import lerobot; print(lerobot.__version__)"

# 测试机器人连接
lerobot-calibrate --robot.type=koch_follower --robot.port=/dev/ttyUSB0

# 查看可用相机
python -c "import cv2; [print(f'Camera {i}') for i in range(10) if cv2.VideoCapture(i).isOpened()]"
```

---

**祝推理顺利！** 🎉

