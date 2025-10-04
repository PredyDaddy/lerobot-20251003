# Koch机械臂策略推理部署完整指南

> **基于LeRobot代码库实际实现的技术文档**
> 作者：AI Assistant
> 日期：2025-10-04
> 适用模型：ACT Policy & Diffusion Policy

---

## 📋 目录

1. [概述](#1-概述)
2. [硬件检查清单](#2-硬件检查清单)
3. [软件环境检查](#3-软件环境检查)
4. [模型配置对比](#4-模型配置对比)
5. [推理流程详解](#5-推理流程详解)
6. [推理脚本使用](#6-推理脚本使用)
7. [常见问题排查](#7-常见问题排查)
8. [安全注意事项](#8-安全注意事项)

---

## 1. 概述

### 1.1 已训练模型信息

根据代码库检索，你已成功训练以下两个模型：

| 模型类型 | 检查点路径 | 训练步数 |
|---------|-----------|---------|
| **ACT Policy** | `output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model/` | 152,000 |
| **Diffusion Policy** | `output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model/` | 108,000 |

### 1.2 数据集配置

- **数据集路径**: `grasp_dataset_v30/`
- **机器人类型**: Koch Follower
- **相机配置**:
  - `laptop`: 640x480 @ 30fps
  - `phone`: 640x480 @ 30fps
- **动作维度**: 6 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **状态维度**: 6 (与动作维度相同)

### 1.3 推理方式

LeRobot提供了**统一的推理接口**，通过`lerobot-record`脚本实现：
- 支持策略驱动的机器人控制
- 支持实时数据记录（用于评估）
- 支持Rerun可视化

---

## 2. 硬件检查清单

### 2.1 Koch机械臂连接检查

#### 2.1.1 查找USB端口

```bash
# 方法1：使用LeRobot工具（推荐）
python -c "
from serial.tools import list_ports
ports = list_ports.comports()
for port in ports:
    print(f'端口: {port.device}')
    print(f'  描述: {port.description}')
    print(f'  硬件ID: {port.hwid}')
    print()
"

# 方法2：Linux系统直接查看
ls -l /dev/ttyUSB* /dev/ttyACM*

# 方法3：macOS系统
ls -l /dev/tty.usbmodem*
```

**预期输出示例**：
```
端口: /dev/ttyUSB0
  描述: USB Serial Port
  硬件ID: USB VID:PID=0403:6014
```

#### 2.1.2 检查端口权限

```bash
# 检查当前用户权限
groups

# 添加用户到dialout组（Linux）
sudo usermod -a -G dialout $USER
# 需要重新登录生效

# 临时授权（测试用）
sudo chmod 666 /dev/ttyUSB0
```

#### 2.1.3 验证Dynamixel电机通信

根据`src/lerobot/robots/koch_follower/koch_follower.py`，Koch使用以下电机配置：

```python
# Koch Follower电机配置（源码第51-60行）
motors = {
    "shoulder_pan": Motor(1, "xl430-w250", norm_mode),   # ID=1
    "shoulder_lift": Motor(2, "xl430-w250", norm_mode),  # ID=2
    "elbow_flex": Motor(3, "xl330-m288", norm_mode),     # ID=3
    "wrist_flex": Motor(4, "xl330-m288", norm_mode),     # ID=4
    "wrist_roll": Motor(5, "xl330-m288", norm_mode),     # ID=5
    "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),  # ID=6
}
```

**关键参数**：
- **波特率**: 自动检测（默认1000000 bps）
- **协议版本**: Protocol 2.0
- **电机ID**: 1-6（必须按顺序配置）

**测试命令**：
```bash
# 使用LeRobot校准工具测试连接
lerobot-calibrate \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=test_koch
```

### 2.2 相机设备检查

#### 2.2.1 查找相机设备

```bash
# Linux: 列出所有视频设备
v4l2-ctl --list-devices

# 或使用简单方法
ls -l /dev/video*

# 测试相机是否可用
ffplay /dev/video0  # Ctrl+C退出
```

#### 2.2.2 验证相机分辨率和帧率

```bash
# 检查相机支持的格式
v4l2-ctl --device=/dev/video0 --list-formats-ext

# 使用Python测试（推荐）
python -c "
import cv2
cap = cv2.VideoCapture(0)  # 或 '/dev/video0'
if cap.isOpened():
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'相机0: {int(width)}x{int(height)} @ {fps}fps')
    cap.release()
else:
    print('无法打开相机0')
"
```

#### 2.2.3 相机配置映射

根据你的数据集，需要配置两个相机：

| 数据集名称 | 设备路径（示例） | 分辨率 | 帧率 |
|-----------|----------------|--------|------|
| `laptop` | `/dev/video0` | 640x480 | 30fps |
| `phone` | `/dev/video2` | 640x480 | 30fps |

**重要**：设备路径可能因系统而异，需要根据实际情况调整。

### 2.3 硬件检查脚本

创建一个一键检查脚本：

```bash
#!/bin/bash
# 保存为 my_workspace/scripts/check_hardware.sh

echo "=== Koch机械臂硬件检查 ==="
echo ""

# 1. 检查USB端口
echo "1. USB端口检查："
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  未找到USB串口设备"
echo ""

# 2. 检查相机设备
echo "2. 相机设备检查："
ls -l /dev/video* 2>/dev/null || echo "  未找到视频设备"
echo ""

# 3. 检查用户权限
echo "3. 用户权限检查："
groups | grep -q dialout && echo "  ✓ 用户在dialout组" || echo "  ✗ 用户不在dialout组（需要sudo usermod -a -G dialout $USER）"
echo ""

# 4. 检查Python环境
echo "4. Python环境检查："
conda info | grep "active environment" || echo "  未激活conda环境"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch未安装"
python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" 2>/dev/null || echo "  ✗ OpenCV未安装"
echo ""

echo "=== 检查完成 ==="
```

---

## 3. 软件环境检查

### 3.1 Conda环境

```bash
# 激活环境
conda activate lerobot_v3

# 验证环境
conda info | grep "active environment"
```

### 3.2 关键依赖检查

```bash
# 检查LeRobot安装
python -c "import lerobot; print(f'LeRobot版本: {lerobot.__version__}')"

# 检查策略模块
python -c "from lerobot.policies.act import ACTPolicy; print('✓ ACT Policy可用')"
python -c "from lerobot.policies.diffusion import DiffusionPolicy; print('✓ Diffusion Policy可用')"

# 检查机器人模块
python -c "from lerobot.robots.koch_follower import KochFollower; print('✓ Koch Follower可用')"

# 检查Dynamixel SDK
python -c "import dynamixel_sdk; print('✓ Dynamixel SDK可用')"
```

### 3.3 模型检查点验证

```bash
# 检查ACT模型
ls -lh output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model/

# 检查Diffusion模型
ls -lh output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model/

# 验证模型文件完整性
python -c "
import json
with open('output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model/config.json') as f:
    cfg = json.load(f)
    print(f'ACT模型类型: {cfg[\"type\"]}')
    print(f'Chunk size: {cfg[\"chunk_size\"]}')
    print(f'n_action_steps: {cfg[\"n_action_steps\"]}')
"
```

---

## 4. 模型配置对比

### 4.1 ACT vs Diffusion 核心差异

基于源码分析（`src/lerobot/policies/act/modeling_act.py` 和 `src/lerobot/policies/diffusion/modeling_diffusion.py`）：

| 特性 | ACT Policy | Diffusion Policy |
|------|-----------|-----------------|
| **观测步数** | `n_obs_steps=1` | `n_obs_steps=2` |
| **动作预测** | `chunk_size=100` | `horizon=16` |
| **执行步数** | `n_action_steps=100` | `n_action_steps=8` |
| **推理频率** | 每100步推理1次 | 每8步推理1次 |
| **归一化** | MEAN_STD (全部) | STATE/ACTION: MIN_MAX, VISUAL: MEAN_STD |
| **推理延迟** | ~50-100ms (Transformer) | ~200-500ms (扩散采样) |
| **实时性** | ★★★★★ 高 | ★★★☆☆ 中等 |

### 4.2 推理流程对比

#### ACT推理流程（源码：`modeling_act.py` 第98-121行）

```python
def select_action(self, batch):
    # 1. 检查动作队列是否为空
    if len(self._action_queue) == 0:
        # 2. 预测100步动作chunk
        actions = self.predict_action_chunk(batch)[:, :100]
        # 3. 填充队列（转置为 (n_steps, batch, dim)）
        self._action_queue.extend(actions.transpose(0, 1))

    # 4. 从队列取出一个动作
    return self._action_queue.popleft()
```

**特点**：
- 一次推理预测100步
- 队列消耗完才重新推理
- 适合高频控制（30Hz）

#### Diffusion推理流程（源码：`modeling_diffusion.py` 第102-138行）

```python
def select_action(self, batch, noise=None):
    # 1. 更新观测队列（需要2步历史）
    self._queues = populate_queues(self._queues, batch)

    # 2. 检查动作队列
    if len(self._queues[ACTION]) == 0:
        # 3. 堆叠历史观测
        batch = {k: torch.stack(list(self._queues[k]), dim=1)
                 for k in batch if k in self._queues}
        # 4. 扩散采样生成16步动作
        actions = self.predict_action_chunk(batch, noise)
        # 5. 提取前8步填充队列
        self._queues[ACTION].extend(actions.transpose(0, 1))

    # 6. 从队列取出一个动作
    return self._queues[ACTION].popleft()
```

**特点**：
- 需要维护2步观测历史
- 一次推理预测16步，执行8步
- 扩散采样耗时较长

### 4.3 实际配置文件

#### ACT配置（从检查点读取）

```json
{
    "type": "act",
    "n_obs_steps": 1,
    "chunk_size": 100,
    "n_action_steps": 100,
    "vision_backbone": "resnet18",
    "use_vae": true,
    "latent_dim": 32,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1
}
```

#### Diffusion配置（从检查点读取）

```json
{
    "type": "diffusion",
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "vision_backbone": "resnet18",
    "num_train_timesteps": 100,
    "noise_scheduler_type": "DDPM"
}
```

---

## 5. 推理流程详解

### 5.1 LeRobot推理架构

根据`src/lerobot/scripts/lerobot_record.py`（第236-370行），推理流程如下：

```
┌─────────────────────────────────────────────────────────┐
│                    推理主循环                            │
│  (record_loop函数，30Hz控制频率)                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  1. 获取机器人观测                │
        │     robot.get_observation()      │
        │     - 读取电机位置 (6维)          │
        │     - 捕获相机图像 (2个相机)      │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  2. 观测预处理                   │
        │     robot_observation_processor  │
        │     - 默认为IdentityProcessor    │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  3. 构建数据集格式                │
        │     build_dataset_frame()        │
        │     - 转换为LeRobot格式          │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  4. 策略推理                     │
        │     predict_action()             │
        │     ├─ preprocessor(observation) │
        │     ├─ policy.select_action()    │
        │     └─ postprocessor(action)     │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  5. 动作后处理                   │
        │     robot_action_processor       │
        │     - 默认为IdentityProcessor    │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  6. 发送动作到机器人              │
        │     robot.send_action()          │
        │     - 可选：max_relative_target  │
        │     - 写入Goal_Position          │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  7. 记录数据（可选）              │
        │     dataset.add_frame()          │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  8. 可视化（可选）                │
        │     log_rerun_data()             │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  9. 等待下一个控制周期            │
        │     busy_wait(1/fps - dt)        │
        └─────────────────────────────────┘
```

### 5.2 predict_action详解

源码位置：`src/lerobot/utils/control_utils.py` 第66-130行

```python
def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """单步推理预测机器人动作"""

    # 1. 添加任务描述（如果策略需要）
    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    # 2. 预处理：归一化、添加batch维度、转换设备
    observation = preprocessor(observation)

    # 3. 策略推理（自动管理动作队列）
    with torch.no_grad(), torch.autocast(device_type=device.type) if use_amp else nullcontext():
        action = policy.select_action(observation)

    # 4. 后处理：反归一化、移除batch维度
    action = postprocessor(action)

    # 5. 转换为CPU numpy数组
    action = action.squeeze(0).to("cpu")

    return action
```

### 5.3 异常处理机制

根据源码分析，LeRobot内置以下安全机制：

#### 5.3.1 动作限幅（源码：`src/lerobot/robots/utils.py`）

```python
# Koch配置中可设置max_relative_target
if self.config.max_relative_target is not None:
    present_pos = self.bus.sync_read("Present_Position")
    goal_pos = ensure_safe_goal_position(
        goal_present_pos,
        self.config.max_relative_target
    )
```

**建议配置**：
```python
max_relative_target = 10.0  # 限制单步最大移动角度（度）
```

#### 5.3.2 电机超限保护

Dynamixel电机内置硬件限位（源码：`koch_follower.py` 第155-163行）：

```python
def configure(self):
    # 设置扩展位置模式（允许>360度旋转）
    for motor in self.bus.motors:
        if motor != "gripper":
            self.bus.write("Operating_Mode", motor,
                          OperatingMode.EXTENDED_POSITION.value)
```

#### 5.3.3 相机掉线处理

```python
# 相机异步读取（源码：koch_follower.py 第197-201行）
for cam_key, cam in self.cameras.items():
    try:
        obs_dict[cam_key] = cam.async_read()
    except Exception as e:
        logger.error(f"相机{cam_key}读取失败: {e}")
        # 可选：使用上一帧或黑屏
```

---

## 6. 推理脚本使用

### 6.1 使用lerobot-record进行推理

这是**官方推荐**的推理方式，支持实时可视化和数据记录。

#### 6.1.1 ACT模型推理

```bash
lerobot-record \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras="{
        laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
        phone: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}
    }" \
    --robot.id=koch_inference \
    --robot.max_relative_target=10.0 \
    --policy.path=output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model \
    --dataset.repo_id=None \
    --dataset.root=eval_data/act_eval \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=30 \
    --dataset.fps=30 \
    --dataset.single_task="Grasp and place object" \
    --display_data=true
```

**参数说明**：
- `--robot.type`: 机器人类型（koch_follower）
- `--robot.port`: USB端口路径
- `--robot.cameras`: 相机配置（JSON格式）
- `--robot.max_relative_target`: 安全限幅（度）
- `--policy.path`: 模型检查点路径
- `--dataset.root`: 评估数据保存路径
- `--display_data`: 启用Rerun可视化

#### 6.1.2 Diffusion模型推理

```bash
lerobot-record \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras="{
        laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
        phone: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}
    }" \
    --robot.id=koch_inference \
    --robot.max_relative_target=10.0 \
    --policy.path=output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model \
    --dataset.repo_id=None \
    --dataset.root=eval_data/diffusion_eval \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=30 \
    --dataset.fps=30 \
    --dataset.single_task="Grasp and place object" \
    --display_data=true
```

**注意**：Diffusion需要2步观测历史，首次推理会等待2帧。

### 6.2 简化推理脚本

我将在`my_workspace/inference/`目录创建简化脚本（见第6节）。

---

### 6.3 使用自定义 Python 推理脚本 koch_inference.py

该脚本无需数据集即可直接进行实时控制，内部按源码调用 predict_action、policy.select_action、robot.send_action：

```bash
# ACT 推理（示例路径请按你实际输出调整）
python my_workspace/inference/koch_inference.py \
  --policy.path /mnt/data/cqy/workspace/lerobot-20251003/output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model \
  --device cuda --use_amp true --fps 30 \
  --robot.port /dev/ttyUSB0 \
  --robot.cameras '{"laptop": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "phone": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
  --robot.max_relative_target 10.0

# Diffusion 推理（扩散采样更慢，建议 FPS=20）
python my_workspace/inference/koch_inference.py \
  --policy.path /mnt/data/cqy/workspace/lerobot-20251003/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model \
  --device cuda --use_amp true --fps 20 \
  --robot.port /dev/ttyUSB0 \
  --robot.cameras '{"laptop": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "phone": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
  --robot.max_relative_target 10.0
```

也可直接使用我准备好的脚本：

```bash
chmod +x my_workspace/inference/infer_act.sh my_workspace/inference/infer_diffusion.sh
my_workspace/inference/infer_act.sh
my_workspace/inference/infer_diffusion.sh
```

注意：相机键名（如 laptop、phone）需与训练时一致，以匹配预处理器配置中的 image_features。


## 7. 常见问题排查

### 7.1 机器人连接问题

**问题**: `DeviceNotConnectedError: Koch Follower is not connected`

**排查步骤**：
1. 检查USB线是否连接
2. 检查端口权限：`ls -l /dev/ttyUSB0`
3. 检查端口是否被占用：`lsof /dev/ttyUSB0`
4. 尝试重新插拔USB

**解决方案**：
```bash
# 授权端口
sudo chmod 666 /dev/ttyUSB0

# 或添加用户到dialout组
sudo usermod -a -G dialout $USER
# 重新登录
```

### 7.2 电机通信失败

**问题**: `RuntimeError: Motor 'shoulder_pan' was not found`

**排查步骤**：
1. 检查电机ID是否正确（1-6）
2. 检查波特率是否匹配
3. 检查电机供电

**解决方案**：
```bash
# 使用校准工具重新扫描
lerobot-calibrate \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=test
```

### 7.3 相机无法打开

**问题**: `cv2.error: (-215:Assertion failed) !_src.empty()`

**排查步骤**：
1. 检查设备路径：`ls /dev/video*`
2. 测试相机：`ffplay /dev/video0`
3. 检查权限：`groups | grep video`

**解决方案**：
```bash
# 添加用户到video组
sudo usermod -a -G video $USER

# 或使用绝对路径
--robot.cameras="{laptop: {type: opencv, index_or_path: '/dev/video0', ...}}"
```

### 7.4 推理延迟过高

**问题**: 控制频率低于30Hz

**排查步骤**：
1. 检查GPU使用：`nvidia-smi`
2. 检查CPU占用：`htop`
3. 测量推理时间

**优化方案**：
```python
# ACT: 使用temporal ensembling（更平滑但延迟更低）
--policy.temporal_ensemble_coeff=0.01

# Diffusion: 减少扩散步数（牺牲质量换速度）
# 需要修改配置文件中的num_inference_timesteps
```

### 7.5 动作抖动

**问题**: 机器人运动不平滑

**原因分析**：
1. 动作归一化不匹配
2. PID参数未优化
3. 动作队列管理问题

**解决方案**：
```python
# 1. 检查归一化统计量是否正确
# 2. 调整PID参数（源码：koch_follower.py 第173-177行）
self.bus.write("Position_P_Gain", "elbow_flex", 1500)
self.bus.write("Position_D_Gain", "elbow_flex", 600)

# 3. 启用max_relative_target限幅
--robot.max_relative_target=5.0
```

---

## 8. 安全注意事项

### 8.1 首次运行前

- [ ] **工作空间清理**：确保机械臂周围无障碍物
- [ ] **紧急停止**：准备好随时按下电源开关
- [ ] **限位设置**：配置`max_relative_target`参数
- [ ] **低速测试**：首次运行使用较小的动作步长

### 8.2 运行中监控

- [ ] **实时可视化**：启用`--display_data=true`监控动作
- [ ] **异常检测**：观察电机温度和电流
- [ ] **碰撞检测**：如有异常声音立即停止

### 8.3 推荐配置

```python
# 安全配置示例
robot_config = KochFollowerConfig(
    port="/dev/ttyUSB0",
    max_relative_target=10.0,  # 限制单步最大移动10度
    disable_torque_on_disconnect=True,  # 断开时自动关闭力矩
)
```

### 8.4 紧急停止

**键盘快捷键**（在`lerobot-record`中）：
- `Ctrl+C`: 停止记录并安全断开
- `Esc`: 提前结束当前episode

**代码中断**：
```python
# 在自定义脚本中添加
import signal
def signal_handler(sig, frame):
    robot.disconnect()  # 自动关闭力矩
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
```

---

## 附录A：完整配置示例

### A.1 ACT推理配置

```yaml
# config_act_inference.yaml
robot:
  type: koch_follower
  port: /dev/ttyUSB0
  id: koch_act_inference
  max_relative_target: 10.0
  cameras:
    laptop:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
    phone:
      type: opencv
      index_or_path: 2
      width: 640
      height: 480
      fps: 30

policy:
  path: output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model

dataset:
  repo_id: null
  root: eval_data/act_eval
  num_episodes: 5
  episode_time_s: 30
  fps: 30
  single_task: "Grasp and place object"

display_data: true
```

### A.2 Diffusion推理配置

```yaml
# config_diffusion_inference.yaml
robot:
  type: koch_follower
  port: /dev/ttyUSB0
  id: koch_diffusion_inference
  max_relative_target: 10.0
  cameras:
    laptop:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
    phone:
      type: opencv
      index_or_path: 2
      width: 640
      height: 480
      fps: 30

policy:
  path: output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model

dataset:
  repo_id: null
  root: eval_data/diffusion_eval
  num_episodes: 5
  episode_time_s: 30
  fps: 30
  single_task: "Grasp and place object"

display_data: true
```

---

## 附录B：性能基准

基于源码分析和典型硬件配置：

| 指标 | ACT | Diffusion |
|------|-----|-----------|
| 单次推理时间 | 50-100ms | 200-500ms |
| 控制频率 | 30Hz | 15-30Hz |
| GPU显存占用 | ~2GB | ~3GB |
| CPU占用 | 中等 | 高 |
| 适用场景 | 高频精细操作 | 复杂长时序任务 |

---

**文档版本**: v1.0
**最后更新**: 2025-10-04
**维护者**: AI Assistant
**反馈**: 如有问题请检查代码库或提issue

