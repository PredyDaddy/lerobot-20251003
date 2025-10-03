# CLAUDE.md

此文件为Claude Code (claude.ai/code)在此代码库中工作时提供指导。

## 项目概述

这是一个LeRobot代码库 - 基于Hugging Face最先进的机器人学习库的定制版本，专门适配了Koch机械臂和中文文档。LeRobot在PyTorch中提供模型、数据集和工具，用于真实世界机器人技术，专注于模仿学习和强化学习。

## 开发命令

### 环境设置
```bash
# 创建Python 3.10虚拟环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 安装LeRobot
pip install -e .

# 针对特定环境的仿真安装
pip install -e ".[aloha, pusht]"

# 安装开发依赖
pip install -e ".[dev]"

# 登录WandB进行实验跟踪
wandb login
```

### 代码质量
```bash
# 使用Ruff进行代码检查
ruff check .

# 使用Ruff进行代码格式化
ruff format .

# 使用Bandit进行安全扫描
bandit -r lerobot/

# 运行pre-commit钩子
pre-commit run --all-files
```

### 测试
```bash
# 运行端到端测试（来自Makefile）
make test-end-to-end

# 运行特定策略测试
make test-act-ete-train
make test-diffusion-ete-train
make test-tdmpc-ete-train

# 使用pytest运行测试
pytest tests/
```

### 核心脚本
```bash
# 训练策略
python lerobot/scripts/train.py \
    --policy.type=act \
    --env.type=aloha \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --wandb.enable=true

# 评估策略
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10

# 可视化数据集
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0

# 控制真实机器人
python lerobot/scripts/control_robot.py \
    --robot.type=koch \
    --control.type=teleoperate
```

## 代码架构

### 核心结构
- **lerobot/scripts/**: 主要可执行脚本（train.py, eval.py, control_robot.py等）
- **lerobot/cameras/**: 相机实现（opencv, intelrealsense）
- **lerobot/robots/**: 机器人实现（koch, aloha, so100等）
- **lerobot/motors/**: 电机控制器（dynamixel, feetech）
- **lerobot/policies/**: 策略实现（ACT, Diffusion, TDMPC, VQBeT）
- **lerobot/datasets/**: 数据集处理和转换
- **lerobot/envs/**: 环境配置和工具
- **lerobot/teleoperators/**: 遥操作接口
- **lerobot/utils/**: 训练、日志等工具
- **lerobot/configs/**: 配置类和YAML文件

### 可用组件
- **策略**: act, diffusion, tdmpc, vqbet
- **机器人**: koch, koch_bimanual, aloha, so100
- **相机**: opencv, intelrealsense
- **电机**: dynamixel, feetech
- **环境**: aloha, pusht, xarm

### 配置系统
- 使用基于dataclass的配置，支持命令行覆盖
- 策略配置在 `lerobot/policies/*/configuration_*.py`
- 环境配置在 `lerobot/envs/configs.py`
- 主训练配置在 `lerobot/configs/train.py`

### 数据集格式
- LeRobotDataset格式，使用HuggingFace数据集和Arrow/parquet后端
- 视频以MP4格式存储以节省空间
- 通过 `delta_timestamps` 支持时间帧关系
- 元数据以JSON/JSONL格式存储

## 关键文件位置

### 定制Koch实现
- **机器人**: `src/lerobot/robots/koch_follower/`
- **遥操作**: `src/lerobot/teleoperators/koch_leader/`
- **配置**: 在相应子目录中可用

### 配置文件
- **pyproject.toml**: Python项目配置和依赖
- **Makefile**: 测试自动化和Docker构建
- **lerobot/configs/**: 策略和训练配置

### 硬件配置
- **机器人配置**: `src/lerobot/robots/*/config_*.py`
- **相机配置**: `src/lerobot/cameras/configs.py`
- **电机配置**: `src/lerobot/motors/configs.py`

## 开发指南

### 代码质量
- 使用Ruff进行代码检查，行长度110
- 使用Bandit进行安全扫描
- 通过 `pip install -e ".[dev]"` 可用pre-commit钩子

### 添加新组件
- **新策略**: 在 `lerobot/__init__.py` 中更新 `available_policies` 和 `available_policies_per_env`，实现配置和建模类
- **新机器人**: 在 `lerobot/__init__.py` 中更新 `available_robots`，在 `lerobot/robots/` 目录中添加机器人定义
- **新环境**: 在 `lerobot/__init__.py` 中更新 `available_tasks_per_env` 和 `available_datasets_per_env`
- **新数据集**: 在 `lerobot/__init__.py` 中更新 `available_datasets_per_env`

### 模型训练
- 启用 `--wandb.enable=true` 时支持WandB日志记录
- 检查点保存在 `outputs/train/` 中，频率可配置
- 使用 `--resume=true` 和配置路径恢复训练

### 依赖
- 需要PyTorch 2.2+
- 关键依赖：transformers, diffusers, gymnasium, opencv-python, rerun-sdk
- 特定机器人/环境的可选依赖以extras形式提供

## 重要说明

此代码库包含定制的Koch机器人实现，扩展了基础LeRobot功能以适配特定硬件配置。Koch特定代码位于机器人和遥操作器的专用子目录中。

### 可用命令
- `lerobot-train`: 在数据集上训练策略
- `lerobot-eval`: 评估训练好的策略
- `lerobot-record`: 从机器人记录训练数据
- `lerobot-replay`: 回放记录的数据集
- `lerobot-teleoperate`: 遥操作机器人
- `lerobot-dataset-viz`: 在rerun中可视化数据集
- `lerobot-info`: 显示系统信息

## 开发规范
- 在我跟你讨论的时候，不需要完全顺着我的意思，我在不确定的时候会询问你某件事情是否需要做，如果你觉得我的做法不对，可以直接的告诉我做这件事情的意义不大
- 在每次任务的开始先使用sequential-thinking来思考，认真思考，thinkharder, ultratkink, 避免犯错
- 在这个项目里面运行python代码务必使用leroboot_v3这个conda环境。
- 这个项目的conda环境为lerobot，先用conda activate lerobot激活环境再使用终端命令
- 在使用python安装包的时候尽可能的使用清华园，如果没有才考虑这个其他的