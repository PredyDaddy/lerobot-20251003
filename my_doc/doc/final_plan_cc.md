# SmolVLA训练完整实施方案 (最终版)

**文档版本**: 1.0
**创建日期**: 2025-10-02
**作者**: CC (整合C4.5、Codex、GPT-5三人方案)
**项目**: Koch机械臂SmolVLA集成项目

---

## 执行摘要

基于对C4.5、Codex、GPT-5三人技术方案的深入分析，并结合现有代码仓库的实际情况，本文档制定了在lerobot-20251011新框架中训练SmolVLA的完整实施方案。

### 核心结论
- **推荐方案**: 在lerobot-20251011新框架中直接训练SmolVLA，保留现有new_lerobot环境用于ACT训练
- **关键挑战**: 数据集格式从v2.1升级到v3.0的不可逆转换
- **资源需求**: 24GB+ GPU显存，32GB+内存，150GB+存储空间
- **预估时间**: 微调训练1-2天，从头训练3-6天

---

## 一、项目现状分析

### 1.1 当前环境配置

**现有环境 (new_lerobot)**:
- **框架版本**: LeRobot v2.1
- **数据集**: grasp_dataset (700 episodes, 237,875 frames)
- **任务**: "Place the bracelet into the box" (单任务)
- **相机配置**: 双相机 (laptop: 480×640, phone: 480×640)
- **机器人状态**: 6维 (main_shoulder_pan, main_shoulder_lift, main_elbow_flex, main_wrist_flex, main_wrist_roll, main_gripper)
- **现有模型**: ACT, ACT-DINOv2
- **训练脚本**: koch_train.sh

**目标环境 (lerobot-20251011)**:
- **框架版本**: LeRobot v3.0
- **新增算法**: SmolVLA (Vision-Language-Action model)
- **架构变化**: 从`lerobot.common.*`到`lerobot.*`，引入processor系统

### 1.2 关键技术差异

| 方面 | 当前环境 (v2.1) | 新框架 (v3.0) | 影响 |
|------|----------------|---------------|------|
| **数据集格式** | episode_*文件格式 | file_*分片格式 | ⚠️ **不兼容，需转换** |
| **导入路径** | `lerobot.common.*` | `lerobot.*` | 代码结构重大变化 |
| **训练入口** | `lerobot/scripts/train.py` | `lerobot-train` | CLI命令变化 |
| **处理器系统** | 简单预处理 | 完整processor pipeline | SmolVLA强依赖 |
| **依赖版本** | 基础依赖 | transformers>=4.52.0等 | 新增重量级依赖 |

### 1.3 代码仓库分析发现

**重要发现**:
1. **TRT优化目录**: 存在`trt_*`系列目录，表明已有模型优化经验
2. **现有训练输出**: `outputs/train/`目录结构完整，有现成的训练基础设施
3. **数据集备份**: 存在`grasp_dataset_backup`，说明有数据安全意识
4. **脚本体系**: 完整的koch_*系列脚本，已形成成熟的工作流
5. **文档完善**: 多个中文技术文档，说明重视知识沉淀

**潜在优势**:
- 已有TensorRT优化经验，可应用于SmolVLA推理优化
- 现有训练基础设施可直接复用
- 成熟的数据采集和处理流程
- 完整的机器人控制脚本

**需要考虑的问题**:
- 多个TRT目录占用存储空间，需要规划清理
- 现有conda环境可能有依赖冲突
- 需要保持现有ACT训练流程不受影响

---

## 二、技术方案深度分析

### 2.1 方案对比矩阵

| 方案 | 工作量 | 风险 | 长期维护 | 数据兼容性 | 推荐度 |
|------|--------|------|----------|------------|--------|
| **迁移SmolVLA到现有环境** | 1-2周 | 🔴 高 | 极差 | ✅ 完全兼容 | ❌ **不推荐** |
| **直接在新框架训练** | 1-2天 | 🟡 中 | 优秀 | ⚠️ 需转换 | ✅ **强烈推荐** |
| **混合方案** | 2-3天 | 🟢 低 | 良好 | ✅ 兼容 | ✅ **最佳选择** |

### 2.2 选择混合方案的核心理由

**技术优势**:
1. **风险最小化**: 保留现有生产环境，零风险到ACT训练
2. **功能最大化**: 新框架完全支持SmolVLA的所有特性
3. **维护简单**: 避免复杂的代码迁移和依赖冲突
4. **回退容易**: 出现问题可立即回到现有方案

**业务优势**:
1. **快速部署**: 2-3天即可完成SmolVLA训练
2. **对比研究**: 可同时训练和对比ACT、ACT-DINOv2、SmolVLA
3. **技能积累**: 团队可掌握两套框架的使用
4. **未来准备**: 为后续算法升级打下基础

---

## 三、实施计划详细步骤

### 阶段1: 环境准备 (预计0.5天)

#### 3.1.1 Conda环境创建

```bash
# 创建独立环境，避免与现有lerobot环境冲突
cd /home/chenqingyu/robot/lerobot-20251011
conda create -y -n lerobot_smolvla python=3.10
conda activate lerobot_smolvla

# 验证Python版本 (必须3.10+)
python --version  # 期望: Python 3.10.x
```

#### 3.1.2 依赖安装顺序

```bash
# 1. 安装基础LeRobot
pip install -e .

# 2. 安装SmolVLA专用依赖
pip install -e ".[smolvla]"

# 3. 验证关键依赖
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import transformers
print(f'Transformers: {transformers.__version__}')

from lerobot.policies.smolvla import SmolVLAPolicy
print('✓ SmolVLA imported successfully')
"
```

#### 3.1.3 环境配置管理

```bash
# 创建环境配置文件
cat > ~/.lerobot_smolvla_config << 'EOF'
# SmolVLA训练环境配置
export LEROBOT_HOME="/home/chenqingyu/robot/lerobot-20251011"
export PYTHONPATH="${LEROBOT_HOME}/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${LEROBOT_HOME}/.cache/huggingface"
export OMP_NUM_THREADS=8

# 性能优化
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
EOF

# 创建便捷激活脚本
cat > activate_smolvla.sh << 'EOF'
#!/bin/bash
echo "激活SmolVLA训练环境..."
conda activate lerobot_smolvla
source ~/.lerobot_smolvla_config
echo "✓ 环境激活完成"
EOF

chmod +x activate_smolvla.sh
```

### 阶段2: 数据集转换 (预计2-4小时)

#### 3.2.1 数据安全备份

```bash
# 多重备份策略
cd /home/chenqingyu/robot/new_lerobot

# 1. 创建完整备份
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp -r grasp_dataset grasp_dataset_backup_${TIMESTAMP}

# 2. 创建增量备份 (节省空间)
rsync -av --link-dest=grasp_dataset_backup \
    grasp_dataset/ grasp_dataset_sync_${TIMESTAMP}/

# 3. 验证备份完整性
python -c "
import json
info1 = json.load(open('grasp_dataset/meta/info.json'))
info2 = json.load(open(f'grasp_dataset_backup_${TIMESTAMP}/meta/info.json'))
assert info1['total_frames'] == info2['total_frames']
print('✓ 备份验证通过')
"
```

#### 3.2.2 数据集转换执行

```bash
cd /home/chenqingyu/robot/lerobot-20251011
source activate_smolvla.sh

# 创建转换脚本
cat > scripts/convert_grasp_dataset.py << 'EOF'
#!/usr/bin/env python3
"""
grasp_dataset数据集转换脚本
v2.1 -> v3.0格式转换
"""

import argparse
import shutil
import json
from pathlib import Path
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_info, convert_tasks, convert_data, convert_videos, convert_episodes_metadata
)

def main():
    parser = argparse.ArgumentParser(description="转换grasp_dataset到v3.0格式")
    parser.add_argument("--source", required=True, help="源数据集路径")
    parser.add_argument("--target", required=True, help="目标数据集路径")
    parser.add_argument("--data-mb", type=int, default=100, help="数据分片大小(MB)")
    parser.add_argument("--video-mb", type=int, default=500, help="视频分片大小(MB)")

    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    print(f"源数据集: {source}")
    print(f"目标数据集: {target}")

    # 检查源数据集
    if not source.exists():
        print(f"❌ 源数据集不存在: {source}")
        return False

    # 检查版本
    info_path = source / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "")
    print(f"源数据集版本: {version}")

    if version != "v2.1":
        print(f"⚠️ 警告: 源数据集版本不是v2.1，可能无法正确转换")

    # 清理目标目录
    if target.exists():
        print(f"清理目标目录: {target}")
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)

    try:
        print("开始转换...")

        # 1. 转换基本信息
        print("1/5 转换基本信息...")
        convert_info(source, target, args.data_mb, args.video_mb)

        # 2. 转换任务信息
        print("2/5 转换任务信息...")
        convert_tasks(source, target)

        # 3. 转换数据
        print("3/5 转换数据...")
        episodes_meta = convert_data(source, target, args.data_mb)

        # 4. 转换视频
        print("4/5 转换视频...")
        episodes_videos_meta = convert_videos(source, target, args.video_mb)

        # 5. 转换episodes元数据
        print("5/5 转换episodes元数据...")
        convert_episodes_metadata(source, target, episodes_meta, episodes_videos_meta)

        print(f"✅ 转换完成!")
        print(f"输出目录: {target}")

        # 验证转换结果
        print("验证转换结果...")
        target_info_path = target / "meta" / "info.json"
        with open(target_info_path) as f:
            target_info = json.load(f)

        print(f"✓ 新版本: {target_info['codebase_version']}")
        print(f"✓ Episodes: {target_info['total_episodes']}")
        print(f"✓ Frames: {target_info['total_frames']}")

        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF

chmod +x scripts/convert_grasp_dataset.py

# 执行转换
python scripts/convert_grasp_dataset.py \
    --source /home/chenqingyu/robot/new_lerobot/grasp_dataset \
    --target datasets/grasp_dataset_v30 \
    --data-mb 100 --video-mb 500
```

#### 3.2.3 转换结果验证

```python
# 创建验证脚本
cat > scripts/verify_dataset_v30.py << 'EOF'
#!/usr/bin/env python3
"""
v3.0数据集验证脚本
"""

import sys
from pathlib import Path
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

def verify_dataset_v3(dataset_path: str):
    """验证v3.0数据集"""
    print("=" * 60)
    print("v3.0数据集验证")
    print("=" * 60)

    dataset_path = Path(dataset_path)

    # 1. 基础目录结构检查
    print("\n[1/8] 目录结构检查...")
    required_dirs = ["data", "meta", "videos"]
    for dir_name in required_dirs:
        if (dataset_path / dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ 不存在")
            return False

    # 2. 关键文件检查
    print("\n[2/8] 关键文件检查...")
    required_files = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet"
    ]
    for file_name in required_files:
        if (dataset_path / file_name).exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ❌ {file_name} 不存在")
            return False

    # 3. 加载元数据
    print("\n[3/8] 元数据加载...")
    try:
        meta = LeRobotDatasetMetadata(
            repo_id="grasp_dataset",
            root=dataset_path
        )
        print(f"  ✓ 元数据加载成功")
        print(f"    版本: {meta.info['codebase_version']}")
        print(f"    Episodes: {meta.total_episodes}")
        print(f"    Frames: {meta.total_frames}")
        print(f"    FPS: {meta.fps}")
        print(f"    机器人类型: {meta.robot_type}")
    except Exception as e:
        print(f"  ❌ 元数据加载失败: {e}")
        return False

    # 4. 版本验证
    print("\n[4/8] 版本验证...")
    version = meta.info['codebase_version']
    if version.startswith('v3.'):
        print(f"  ✓ 版本正确: {version}")
    else:
        print(f"  ❌ 版本错误: {version} (期望v3.x)")
        return False

    # 5. 特征检查
    print("\n[5/8] 特征检查...")
    features = meta.features
    expected_features = [
        "action",
        "observation.state",
        "observation.images.laptop",
        "observation.images.phone",
        "task_index"
    ]

    for feature in expected_features:
        if feature in features:
            print(f"  ✓ {feature}: {features[feature]['shape']}")
        else:
            print(f"  ❌ 缺失特征: {feature}")
            return False

    # 6. 数据集加载测试
    print("\n[6/8] 数据集加载测试...")
    try:
        dataset = LeRobotDataset(
            repo_id="grasp_dataset",
            root=dataset_path
        )
        print(f"  ✓ 数据集加载成功 ({len(dataset)} 样本)")
    except Exception as e:
        print(f"  ❌ 数据集加载失败: {e}")
        return False

    # 7. 样本数据测试
    print("\n[7/8] 样本数据测试...")
    try:
        # 测试多个样本
        for i in [0, len(dataset)//2, len(dataset)-1]:
            sample = dataset[i]

            # 检查关键数据
            assert 'observation.images.laptop' in sample
            assert 'observation.images.phone' in sample
            assert 'observation.state' in sample
            assert 'action' in sample
            assert 'task' in sample  # SmolVLA需要

            # 检查数据形状
            assert sample['observation.images.laptop'].shape == (3, 480, 640)
            assert sample['observation.images.phone'].shape == (3, 480, 640)
            assert sample['observation.state'].shape == (6,)
            assert sample['action'].shape == (6,)

        print(f"  ✓ 样本数据验证通过")

    except Exception as e:
        print(f"  ❌ 样本数据测试失败: {e}")
        return False

    # 8. Task信息检查
    print("\n[8/8] Task信息检查...")
    try:
        tasks = meta.tasks
        print(f"  ✓ 任务数量: {len(tasks)}")

        if len(tasks) > 0:
            task = tasks[0]
            print(f"  ✓ 任务描述: '{task.get('task', 'N/A')}'")

            # 验证task字段
            sample = dataset[0]
            if 'task' in sample and sample['task']:
                print(f"  ✓ Task字段存在: '{sample['task']}'")
            else:
                print(f"  ⚠️ Task字段缺失或为空")
                return False
        else:
            print(f"  ❌ 没有找到任务信息")
            return False

    except Exception as e:
        print(f"  ❌ Task信息检查失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ v3.0数据集验证完全通过!")
    print("=" * 60)

    # 显示对比信息
    print("\n数据集对比:")
    print(f"  Episodes: {meta.total_episodes}")
    print(f"  Frames: {meta.total_frames}")
    print(f"  时长: {meta.total_frames / meta.fps:.1f} 秒")
    print(f"  任务: '{meta.tasks[0]['task'] if meta.tasks else 'N/A'}'")

    return True

if __name__ == "__main__":
    dataset_path = "/home/chenqingyu/robot/lerobot-20251011/datasets/grasp_dataset_v30"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    success = verify_dataset_v3(dataset_path)
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/verify_dataset_v30.py

# 运行验证
python scripts/verify_dataset_v30.py
```

### 阶段3: 训练基础设施搭建 (预计0.5天)

#### 3.3.1 工作目录结构

```bash
# 创建完整的工作目录结构
mkdir -p {logs,outputs/train,outputs/eval,outputs/test,scripts,configs,checkpoints}

# 目录说明
echo "目录结构创建完成:"
echo "  logs/          - 训练日志"
echo "  outputs/train/ - 训练输出"
echo "  outputs/eval/  - 评估结果"
echo "  outputs/test/  - 测试输出"
echo "  scripts/       - 辅助脚本"
echo "  configs/       - 配置文件"
echo "  checkpoints/   - 模型检查点"
```

#### 3.3.2 监控工具安装

```bash
# 安装监控工具
pip install wandb tensorboard gpustat

# WandB配置 (可选)
wandb login  # 如果需要上传到云端

# TensorBoard配置
cat > scripts/start_tensorboard.sh << 'EOF'
#!/bin/bash
source activate_smolvla.sh
echo "启动TensorBoard..."
tensorboard --logdir=outputs --port=6006 --host=0.0.0.0
EOF

chmod +x scripts/start_tensorboard.sh
```

#### 3.3.3 训练配置模板

```python
# 创建训练配置模板
cat > configs/smolvla_config.py << 'EOF'
"""
SmolVLA训练配置模板
包含不同场景的推荐配置
"""

class SmolVLAConfig:
    """SmolVLA训练配置基类"""

    # 基础配置
    policy_type = "smolvla"
    pretrained_model = "lerobot/smolvla_base"

    # 数据配置
    dataset_root = "/home/chenqingyu/robot/lerobot-20251011/datasets/grasp_dataset_v30"
    dataset_repo_id = "grasp_dataset"

    # 训练配置
    device = "cuda"
    batch_size = 8
    num_workers = 4
    steps = 50000
    save_freq = 1000
    eval_freq = 0  # 不在训练时评估

    # 优化器配置
    optimizer_lr = 5e-5
    optimizer_betas = (0.9, 0.95)
    optimizer_weight_decay = 1e-10
    optimizer_grad_clip_norm = 10.0

    # 学习率调度
    scheduler_warmup_steps = 2000
    scheduler_decay_steps = 40000
    scheduler_decay_lr = 2.5e-6

    # 模型配置
    use_amp = True
    amp_dtype = "bf16"  # 或 "fp16"

    # 日志配置
    wandb_enable = False
    wandb_project = "smolvla_koch"

    @classmethod
    def get_finetune_config(cls):
        """微调配置 (推荐)"""
        config = cls()
        config.pretrained_model = "lerobot/smolvla_base"
        config.batch_size = 8
        config.optimizer_lr = 5e-5
        config.steps = 50000
        return config

    @classmethod
    def get_scratch_config(cls):
        """从头训练配置"""
        config = cls()
        config.policy_type = "smolvla"  # 不使用预训练模型
        config.batch_size = 16
        config.optimizer_lr = 1e-4
        config.steps = 200000
        return config

    @classmethod
    def get_lightweight_config(cls):
        """轻量级配置 (适用于小显存GPU)"""
        config = cls.get_finetune_config()
        config.batch_size = 4
        config.gradient_accumulation_steps = 4  # 等效batch_size=16
        config.use_amp = True
        config.amp_dtype = "fp16"
        return config

    @classmethod
    def get_test_config(cls):
        """测试配置"""
        config = cls.get_finetune_config()
        config.batch_size = 2
        config.steps = 100
        config.save_freq = 50
        config.dataset_episodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return config

# 配置验证
if __name__ == "__main__":
    configs = [
        ("微调配置", SmolVLAConfig.get_finetune_config()),
        ("从头训练", SmolVLAConfig.get_scratch_config()),
        ("轻量级", SmolVLAConfig.get_lightweight_config()),
        ("测试配置", SmolVLAConfig.get_test_config())
    ]

    for name, config in configs:
        print(f"\n{name}:")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.optimizer_lr}")
        print(f"  Steps: {config.steps}")
        print(f"  AMP: {config.use_amp}")
EOF
```

### 阶段4: 训练脚本开发 (预计0.5天)

#### 3.4.1 通用训练脚本

```python
# 创建通用训练脚本
cat > scripts/train_smolvla.py << 'EOF'
#!/usr/bin/env python3
"""
SmolVLA训练脚本
支持多种配置模式和监控功能
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def create_training_command(config, output_dir, log_file):
    """创建训练命令"""
    cmd = [
        "python", "src/lerobot/scripts/lerobot_train.py",
        f"--policy.path={config.pretrained_model}",
        f"--policy.device={config.device}",
        f"--dataset.root={config.dataset_root}",
        f"--dataset.repo_id={config.dataset_repo_id}",
        f"--batch_size={config.batch_size}",
        f"--num_workers={config.num_workers}",
        f"--policy.optimizer_lr={config.optimizer_lr}",
        f"--policy.optimizer_betas={config.optimizer_betas}",
        f"--policy.optimizer_weight_decay={config.optimizer_weight_decay}",
        f"--policy.optimizer_grad_clip_norm={config.optimizer_grad_clip_norm}",
        f"--policy.scheduler_warmup_steps={config.scheduler_warmup_steps}",
        f"--policy.scheduler_decay_steps={config.scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={config.scheduler_decay_lr}",
        f"--steps={config.steps}",
        f"--save_freq={config.save_freq}",
        f"--eval_freq={config.eval_freq}",
        f"--output_dir={output_dir}",
        f"--job_name=smolvla_koch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        f"--use_amp={config.use_amp}",
        f"--amp_dtype={config.amp_dtype}",
        f"--wandb.enable={config.wandb_enable}",
        f"--wandb.project={config.wandb_project}"
    ]

    # 移除None值的参数
    cmd = [arg for arg in cmd if not arg.endswith("None")]

    return cmd

def save_config(config, config_file):
    """保存配置到文件"""
    config_dict = {}
    for key in dir(config):
        if not key.startswith('_') and not callable(getattr(config, key)):
            config_dict[key] = getattr(config, key)

    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

def monitor_training(log_file):
    """监控训练过程"""
    print(f"监控训练日志: {log_file}")
    print("按Ctrl+C停止监控")

    try:
        import subprocess
        tail_proc = subprocess.Popen(['tail', '-f', log_file])
        tail_proc.wait()
    except KeyboardInterrupt:
        print("\n停止监控")
        tail_proc.terminate()

def main():
    parser = argparse.ArgumentParser(description="SmolVLA训练脚本")
    parser.add_argument("--config", choices=["finetune", "scratch", "lightweight", "test"],
                       default="finetune", help="训练配置")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--monitor", action="store_true", help="监控训练过程")
    parser.add_argument("--dry-run", action="store_true", help="只显示命令，不执行")

    args = parser.parse_args()

    # 加载配置
    from configs.smolvla_config import SmolVLAConfig

    config_map = {
        "finetune": SmolVLAConfig.get_finetune_config(),
        "scratch": SmolVLAConfig.get_scratch_config(),
        "lightweight": SmolVLAConfig.get_lightweight_config(),
        "test": SmolVLAConfig.get_test_config()
    }

    config = config_map[args.config]

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"outputs/train/smolvla_{args.config}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(f"logs/smolvla_{args.config}_{timestamp}.log")

    # 保存配置
    config_file = output_dir / "train_config.json"
    save_config(config, config_file)
    print(f"配置已保存: {config_file}")

    # 创建训练命令
    cmd = create_training_command(config, output_dir, log_file)

    print("=" * 60)
    print(f"SmolVLA训练 - {args.config}配置")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file}")
    print(f"预估时间: {config.steps / 1000:.1f} 小时 (假设1步/秒)")
    print("=" * 60)

    # 显示命令
    print("训练命令:")
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
    print("=" * 60)

    if args.dry_run:
        print("Dry run模式，不执行训练")
        return

    # 创建日志目录
    log_file.parent.mkdir(exist_ok=True)

    # 开始训练
    print(f"开始训练: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        import subprocess

        # 启动训练进程
        with open(log_file, 'w') as f:
            f.write(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            # 启动训练
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True)

            # 实时写入日志
            for line in proc.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()

        # 等待训练完成
        return_code = proc.wait()

        with open(log_file, 'a') as f:
            if return_code == 0:
                f.write(f"\n✅ 训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"\n✅ 训练完成!")
            else:
                f.write(f"\n❌ 训练失败: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                print(f"\n❌ 训练失败!")

        # 如果需要监控
        if args.monitor and return_code == 0:
            monitor_training(log_file)

        sys.exit(return_code)

    except KeyboardInterrupt:
        print("\n训练被中断")
        with open(log_file, 'a') as f:
            f.write(f"\n训练中断: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练出错: {e}")
        with open(log_file, 'a') as f:
            f.write(f"\n训练出错: {e} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/train_smolvla.py
```

#### 3.4.2 便捷训练脚本

```bash
# 创建便捷训练脚本
cat > scripts/train_finetune.sh << 'EOF'
#!/bin/bash
source activate_smolvla.sh
echo "开始SmolVLA微调训练..."
python scripts/train_smolvla.py --config finetune --monitor
EOF

cat > scripts/train_test.sh << 'EOF'
#!/bin/bash
source activate_smolvla.sh
echo "开始SmolVLA测试训练..."
python scripts/train_smolvla.py --config test --dry-run
EOF

cat > scripts/train_lightweight.sh << 'EOF'
#!/bin/bash
source activate_smolvla.sh
echo "开始SmolVLA轻量级训练..."
python scripts/train_smolvla.py --config lightweight --monitor
EOF

chmod +x scripts/train_*.sh
```

### 阶段5: 小规模测试 (预计0.5天)

#### 3.5.1 测试计划

```bash
# 运行测试验证整个流程
echo "=== SmolVLA训练流程测试 ==="

# 1. 环境测试
echo "1. 环境测试..."
source activate_smolvla.sh
python -c "
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.datasets import LeRobotDataset
import torch

print('✓ SmolVLA导入成功')
print('✓ 数据集导入成功')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
print(f'✓ GPU数量: {torch.cuda.device_count()}')
"

# 2. 数据集测试
echo "2. 数据集测试..."
python scripts/verify_dataset_v30.py

# 3. 干跑测试
echo "3. 干跑测试..."
python scripts/train_smolvla.py --config test --dry-run

# 4. 实际测试训练 (10分钟)
echo "4. 实际测试训练..."
echo "这将运行100步，预计需要2-3分钟"
read -p "是否继续? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/train_smolvla.py --config test
fi

echo "✅ 测试完成!"
```

#### 3.5.2 测试验证清单

```python
# 创建测试验证脚本
cat > scripts/test_checklist.py << 'EOF'
#!/usr/bin/env python3
"""
SmolVLA训练前测试清单
"""

import sys
import torch
from pathlib import Path

def run_checklist():
    """运行测试清单"""
    print("=" * 60)
    print("SmolVLA训练前测试清单")
    print("=" * 60)

    tests = []

    # 1. 环境检查
    print("\n[1/6] 环境检查...")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")

        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")

        from lerobot.policies.smolvla import SmolVLAPolicy
        print(f"  ✓ SmolVLA导入成功")

        if torch.cuda.is_available():
            print(f"  ✓ CUDA可用: {torch.cuda.device_count()}个GPU")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  ❌ CUDA不可用")
            return False

        tests.append(True)
    except Exception as e:
        print(f"  ❌ 环境检查失败: {e}")
        tests.append(False)

    # 2. 数据集检查
    print("\n[2/6] 数据集检查...")
    dataset_path = Path("/home/chenqingyu/robot/lerobot-20251011/datasets/grasp_dataset_v30")
    if dataset_path.exists():
        try:
            from lerobot.datasets import LeRobotDataset
            dataset = LeRobotDataset("grasp_dataset", root=dataset_path)
            print(f"  ✓ 数据集加载成功 ({len(dataset)} 样本)")

            # 检查样本
            sample = dataset[0]
            required_keys = ['observation.images.laptop', 'observation.images.phone',
                           'observation.state', 'action', 'task']

            for key in required_keys:
                if key in sample:
                    print(f"  ✓ {key}: {sample[key].shape}")
                else:
                    print(f"  ❌ 缺失: {key}")
                    return False

            tests.append(True)
        except Exception as e:
            print(f"  ❌ 数据集检查失败: {e}")
            tests.append(False)
    else:
        print(f"  ❌ 数据集不存在: {dataset_path}")
        tests.append(False)

    # 3. GPU内存检查
    print("\n[3/6] GPU内存检查...")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total - allocated

            print(f"  GPU {i}:")
            print(f"    总内存: {total:.1f} GB")
            print(f"    已使用: {allocated:.1f} GB")
            print(f"    可用: {free:.1f} GB")

            if free < 20:  # 至少需要20GB
                print(f"    ⚠️ GPU内存可能不足")
            else:
                print(f"    ✓ GPU内存充足")

        tests.append(True)
    else:
        tests.append(False)

    # 4. 存储空间检查
    print("\n[4/6] 存储空间检查...")
    import shutil

    # 检查输出目录空间
    output_dir = Path("/home/chenqingyu/robot/lerobot-20251011/outputs")
    if output_dir.exists():
        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free / 1024**3

        print(f"  输出目录空间:")
        print(f"    总空间: {total/1024**3:.1f} GB")
        print(f"    已使用: {used/1024**3:.1f} GB")
        print(f"    可用: {free_gb:.1f} GB")

        if free_gb < 50:
            print(f"    ⚠️ 存储空间可能不足 (建议至少50GB)")
        else:
            print(f"    ✓ 存储空间充足")

        tests.append(free_gb >= 50)
    else:
        print(f"    ❌ 输出目录不存在")
        tests.append(False)

    # 5. 预训练模型检查
    print("\n[5/6] 预训练模型检查...")
    try:
        from lerobot.policies.smolvla import SmolVLAPolicy

        # 这会触发模型下载
        print("  正在检查预训练模型下载...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        print(f"  ✓ 预训练模型加载成功")

        param_count = sum(p.numel() for p in policy.parameters()) / 1e6
        print(f"  ✓ 参数量: {param_count:.1f}M")

        tests.append(True)
    except Exception as e:
        print(f"  ❌ 预训练模型检查失败: {e}")
        print(f"    可能需要网络连接或等待下载")
        tests.append(False)

    # 6. 依赖版本检查
    print("\n[6/6] 依赖版本检查...")
    critical_deps = {
        'torch': '2.2.1',
        'transformers': '4.52.0',
        'accelerate': '1.7.0',
        'safetensors': '0.4.3'
    }

    dep_ok = True
    for dep, min_version in critical_deps.items():
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"  {dep}: {version}")

            # 简单版本检查 (可能不够精确)
            try:
                if version >= min_version:
                    print(f"    ✓ 版本满足要求")
                else:
                    print(f"    ⚠️ 版本可能过低 (建议 >= {min_version})")
                    dep_ok = False
            except:
                print(f"    ? 无法比较版本")

        except ImportError:
            print(f"  ❌ {dep} 未安装")
            dep_ok = False

    tests.append(dep_ok)

    # 总结
    print("\n" + "=" * 60)
    passed = sum(tests)
    total = len(tests)

    print(f"测试结果: {passed}/{total} 通过")

    if all(tests):
        print("✅ 所有测试通过，可以开始训练!")
        return True
    else:
        print("❌ 部分测试未通过，请解决问题后再训练")

        if not tests[0]: print("  - 环境配置有问题")
        if not tests[1]: print("  - 数据集配置有问题")
        if not tests[2]: print("  - GPU内存不足")
        if not tests[3]: print("  - 存储空间不足")
        if not tests[4]: print("  - 预训练模型问题")
        if not tests[5]: print("  - 依赖版本问题")

        return False

if __name__ == "__main__":
    success = run_checklist()
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/test_checklist.py

# 运行测试清单
python scripts/test_checklist.py
```

### 阶段6: 正式训练 (预计1-5天)

#### 3.6.1 微调训练 (推荐)

```bash
# 运行微调训练
echo "=== SmolVLA微调训练 ==="
echo "配置: 从预训练模型lerobot/smolvla_base微调"
echo "预计时间: 6-12小时 (50,000步)"
echo "GPU需求: 24GB+ 显存"
echo ""

# 选择配置
echo "请选择配置:"
echo "1. 标准微调 (batch_size=8, 24GB显存)"
echo "2. 轻量微调 (batch_size=4, 12GB显存，梯度累积)"
echo "3. 自定义配置"

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo "启动标准微调训练..."
        ./scripts/train_finetune.sh
        ;;
    2)
        echo "启动轻量微调训练..."
        ./scripts/train_lightweight.sh
        ;;
    3)
        echo "启动自定义训练..."
        python scripts/train_smolvla.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
```

#### 3.6.2 从头训练 (可选)

```bash
# 从头训练 (仅在有大量数据和时间时)
echo "=== SmolVLA从头训练 ==="
echo "警告: 从头训练需要2-5天时间和大量GPU资源"
echo "建议先尝试微调训练"
echo ""

read -p "确定要继续从头训练吗? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "启动从头训练..."
    source activate_smolvla.sh
    python scripts/train_smolvla.py --config scratch --monitor
fi
```

#### 3.6.3 训练监控

```python
# 创建训练监控脚本
cat > scripts/monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""
SmolVLA训练实时监控
"""

import sys
import time
import psutil
from pathlib import Path
import torch

def monitor_training():
    """监控训练状态"""
    print("=" * 60)
    print("SmolVLA训练监控")
    print("=" * 60)

    # 查找最新的训练日志
    log_dir = Path("logs")
    log_files = sorted(log_dir.glob("smolvla_*.log"))

    if not log_files:
        print("没有找到训练日志文件")
        return

    latest_log = log_files[-1]
    print(f"监控日志: {latest_log}")

    # 查找输出目录
    output_dirs = list(Path("outputs/train").glob("smolvla_*"))
    if output_dirs:
        output_dir = max(output_dirs, key=lambda x: x.stat().st_mtime)
        print(f"输出目录: {output_dir}")

    print("=" * 60)

    try:
        while True:
            # GPU状态
            if torch.cuda.is_available():
                print(f"\n[{time.strftime('%H:%M:%S')}] GPU状态:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    utilization = allocated / total * 100

                    print(f"  GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({utilization:.1f}%)")

            # CPU状态
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            print(f"CPU: {cpu_percent:.1f}%")
            print(f"内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")

            # 训练进度 (从日志提取)
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()

                # 查找最新的step信息
                for line in reversed(lines[-10:]):
                    if "step" in line.lower() and "loss" in line.lower():
                        print(f"训练: {line.strip()}")
                        break
            except:
                pass

            # 检查checkpoint
            if output_dirs:
                checkpoints_dir = output_dir / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoints = sorted([d.name for d in checkpoints_dir.iterdir() if d.is_dir()])
                    if checkpoints:
                        print(f"最新checkpoint: {checkpoints[-1]}")

            time.sleep(30)  # 30秒更新一次

    except KeyboardInterrupt:
        print("\n监控停止")

if __name__ == "__main__":
    monitor_training()
EOF

chmod +x scripts/monitor_training.py

# 启动监控
python scripts/monitor_training.py
```

### 阶段7: 模型评估 (预计0.5天)

#### 3.7.1 基础评估脚本

```python
# 创建模型评估脚本
cat > scripts/evaluate_model.py << 'EOF'
#!/usr/bin/env python3
"""
SmolVLA模型评估
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.datasets import LeRobotDataset
from tqdm import tqdm

def evaluate_model(checkpoint_path: str, dataset_path: str, num_samples: int = 1000):
    """评估模型性能"""
    print("=" * 60)
    print("SmolVLA模型评估")
    print("=" * 60)

    # 加载模型
    print(f"\n[1/4] 加载模型: {checkpoint_path}")
    try:
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = policy.to(device)
        policy.eval()

        param_count = sum(p.numel() for p in policy.parameters()) / 1e6
        print(f"✓ 模型加载成功 ({param_count:.1f}M 参数)")
        print(f"✓ 设备: {device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 加载数据集
    print(f"\n[2/4] 加载数据集: {dataset_path}")
    try:
        dataset = LeRobotDataset("grasp_dataset", root=dataset_path)
        print(f"✓ 数据集加载成功 ({len(dataset)} 样本)")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # 评估指标计算
    print(f"\n[3/4] 评估指标计算 ({num_samples} 样本)...")

    mse_errors = []
    mae_errors = []

    try:
        with torch.no_grad():
            for i in tqdm(range(min(num_samples, len(dataset))), desc="评估进度"):
                sample = dataset[i]

                # 准备输入
                batch = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0).to(device)
                    else:
                        batch[key] = value

                # 模型推理
                pred_action = policy.select_action(batch)
                true_action = sample['action'].unsqueeze(0).to(device)

                # 计算误差
                mse = torch.mean((pred_action - true_action) ** 2).item()
                mae = torch.mean(torch.abs(pred_action - true_action)).item()

                mse_errors.append(mse)
                mae_errors.append(mae)

        print(f"✓ 评估完成")

    except Exception as e:
        print(f"❌ 评估过程失败: {e}")
        return

    # 结果分析
    print(f"\n[4/4] 结果分析")

    mse_errors = np.array(mse_errors)
    mae_errors = np.array(mae_errors)

    print(f"\n总体指标:")
    print(f"  MSE: {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
    print(f"  MAE: {np.mean(mae_errors):.6f} ± {np.std(mae_errors):.6f}")

    # 按关节分析
    print(f"\n关节详细分析:")
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]

    try:
        # 重新计算每个关节的误差
        joint_mae = np.zeros((len(mae_errors), 6))

        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in sample.items()}

                pred_action = policy.select_action(batch)
                true_action = sample['action'].unsqueeze(0).to(device)

                joint_mae[i] = torch.abs(pred_action - true_action).cpu().numpy()[0]

        for i, name in enumerate(joint_names):
            joint_errors = joint_mae[:, i]
            print(f"  {name:15s}: {np.mean(joint_errors):.6f} ± {np.std(joint_errors):.6f}")

    except Exception as e:
        print(f"  关节详细分析失败: {e}")

    # 保存结果
    results = {
        "total_samples": len(mse_errors),
        "mse_mean": float(np.mean(mse_errors)),
        "mse_std": float(np.std(mse_errors)),
        "mae_mean": float(np.mean(mae_errors)),
        "mae_std": float(np.std(mae_errors)),
        "model_path": checkpoint_path,
        "dataset_path": dataset_path
    }

    output_file = f"outputs/eval/evaluation_results_{Path(checkpoint_path).name}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果已保存: {output_file}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="评估SmolVLA模型")
    parser.add_argument("--checkpoint", help="模型checkpoint路径")
    parser.add_argument("--dataset", default="/home/chenqingyu/robot/lerobot-20251011/datasets/grasp_dataset_v30",
                       help="数据集路径")
    parser.add_argument("--samples", type=int, default=1000, help="评估样本数量")

    args = parser.parse_args()

    # 默认checkpoint路径
    if not args.checkpoint:
        default_path = "outputs/train/smolvla_finetune/checkpoints/last"
        if Path(default_path).exists():
            args.checkpoint = default_path
        else:
            print("请指定checkpoint路径")
            sys.exit(1)

    evaluate_model(args.checkpoint, args.dataset, args.samples)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/evaluate_model.py
```

#### 3.7.2 推理速度测试

```python
# 创建推理速度测试脚本
cat > scripts/benchmark_inference.py << 'EOF'
#!/usr/bin/env python3
"""
SmolVLA推理速度基准测试
"""

import time
import torch
from pathlib import Path
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.datasets import LeRobotDataset

def benchmark_inference(checkpoint_path: str, dataset_path: str, num_iterations: int = 100):
    """测试推理速度"""
    print("=" * 60)
    print("SmolVLA推理速度基准测试")
    print("=" * 60)

    # 加载模型
    print(f"\n[1/3] 加载模型...")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.eval()

    print(f"✓ 模型加载成功")
    print(f"  设备: {device}")

    # 准备测试数据
    print(f"\n[2/3] 准备测试数据...")
    dataset = LeRobotDataset("grasp_dataset", root=dataset_path)
    sample = dataset[0]

    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value

    print(f"✓ 测试数据准备完成")

    # 预热
    print(f"\n[3/3] 预热和测试...")
    with torch.no_grad():
        for _ in range(10):
            _ = policy.select_action(batch)

    if device == "cuda":
        torch.cuda.synchronize()

    # 基准测试
    times = []
    print(f"开始基准测试 ({num_iterations} 次迭代)...")

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            _ = policy.select_action(batch)

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{num_iterations}")

    # 统计结果
    times = np.array(times) * 1000  # 转换为毫秒

    print(f"\n" + "=" * 60)
    print(f"推理速度测试结果")
    print(f"=" * 60)
    print(f"测试设备: {device}")
    print(f"测试次数: {num_iterations}")
    print(f"")
    print(f"推理时间统计 (ms):")
    print(f"  平均时间: {np.mean(times):.2f} ms")
    print(f"  中位数: {np.median(times):.2f} ms")
    print(f"  标准差: {np.std(times):.2f} ms")
    print(f"  最小值: {np.min(times):.2f} ms")
    print(f"  最大值: {np.max(times):.2f} ms")
    print(f"")
    print(f"推理速度:")
    avg_fps = 1000 / np.mean(times)
    min_fps = 1000 / np.max(times)
    max_fps = 1000 / np.min(times)

    print(f"  平均FPS: {avg_fps:.2f}")
    print(f"  最低FPS: {min_fps:.2f}")
    print(f"  最高FPS: {max_fps:.2f}")
    print(f"")

    # 实时性评估
    target_fps = 30  # 30Hz控制要求
    target_time = 1000 / target_fps  # 33.33ms

    print(f"实时性评估 (目标: {target_fps}Hz, {target_time:.1f}ms):")
    if np.mean(times) <= target_time:
        print(f"  ✅ 满足实时性要求")
    else:
        print(f"  ❌ 不满足实时性要求")
        print(f"  需要优化: {(np.mean(times) / target_time - 1) * 100:.1f}% 超时")

    # 优化建议
    print(f"\n优化建议:")
    if np.mean(times) > target_time:
        print(f"  1. 使用torch.compile优化模型")
        print(f"  2. 启用混合精度推理 (fp16)")
        print(f"  3. 考虑模型量化")
        print(f"  4. 使用异步推理")
    else:
        print(f"  推理速度满足要求，无需额外优化")

    print(f"=" * 60)

if __name__ == "__main__":
    import numpy as np

    checkpoint_path = "outputs/train/smolvla_finetune/checkpoints/last"
    dataset_path = "/home/chenqingyu/robot/lerobot-20251011/datasets/grasp_dataset_v30"

    benchmark_inference(checkpoint_path, dataset_path)
EOF

chmod +x scripts/benchmark_inference.py
```

### 阶段8: 与现有模型对比 (预计0.5天)

#### 3.8.1 模型对比脚本

```python
# 创建模型对比脚本
cat > scripts/compare_models.py << 'EOF'
#!/usr/bin/env python3
"""
SmolVLA vs ACT vs ACT-DINOv2 模型对比
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_smolvla_model(checkpoint_path: str):
    """加载SmolVLA模型"""
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.eval()

    return policy, device

def load_act_model(model_path: str):
    """加载ACT模型 (在新环境中可能不可用)"""
    # 这里需要根据实际情况调整
    # 由于ACT模型在旧环境中，可能需要不同的加载方式
    print("ACT模型加载需要在新环境中实现")
    return None, None

def compare_inference_speed(models: dict, dataset_path: str):
    """对比推理速度"""
    print("\n" + "=" * 60)
    print("推理速度对比")
    print("=" * 60)

    # 加载测试数据集
    from lerobot.datasets import LeRobotDataset
    dataset = LeRobotDataset("grasp_dataset", root=dataset_path)
    sample = dataset[0]

    results = {}

    for model_name, model_info in models.items():
        if model_info['model'] is None:
            print(f"{model_name}: 跳过 (模型不可用)")
            continue

        model = model_info['model']
        device = model_info['device']

        print(f"\n测试 {model_name}...")

        # 准备输入
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(device)
            else:
                batch[key] = value

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model.select_action(batch)

        if device == "cuda":
            torch.cuda.synchronize()

        # 测试推理速度
        import time
        times = []

        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model.select_action(batch)

                if device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

        times = np.array(times) * 1000  # 转换为毫秒

        avg_time = np.mean(times)
        fps = 1000 / avg_time

        results[model_name] = {
            'avg_time_ms': avg_time,
            'fps': fps,
            'std_time_ms': np.std(times)
        }

        print(f"  平均时间: {avg_time:.2f} ms")
        print(f"  推理FPS: {fps:.2f}")
        print(f"  实时性: {'✅' if avg_time <= 33.33 else '❌'}")

    return results

def compare_model_size(models: dict):
    """对比模型大小"""
    print("\n" + "=" * 60)
    print("模型大小对比")
    print("=" * 60)

    results = {}

    for model_name, model_info in models.items():
        if model_info['model'] is None:
            print(f"{model_name}: 跳过 (模型不可用)")
            continue

        model = model_info['model']

        # 计算参数量
        param_count = sum(p.numel() for p in model.parameters()) / 1e6

        # 估算模型大小 (假设fp32)
        model_size_mb = param_count * 4  # fp32 = 4 bytes per parameter

        results[model_name] = {
            'params_m': param_count,
            'size_mb': model_size_mb
        }

        print(f"{model_name}:")
        print(f"  参数量: {param_count:.2f}M")
        print(f"  模型大小: {model_size_mb:.1f}MB")

    return results

def generate_comparison_report(speed_results: dict, size_results: dict, output_file: str):
    """生成对比报告"""
    print("\n" + "=" * 60)
    print("模型对比总结")
    print("=" * 60)

    if not speed_results and not size_results:
        print("没有可用的对比数据")
        return

    # 生成文本报告
    report = []
    report.append("# SmolVLA vs 其他模型对比报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## 1. 推理速度对比")

    if speed_results:
        report.append("\n| 模型 | 推理时间(ms) | FPS | 实时性 |")
        report.append("|------|-------------|-----|--------|")

        for model_name, results in speed_results.items():
            real_time = "✅" if results['avg_time_ms'] <= 33.33 else "❌"
            report.append(f"| {model_name} | {results['avg_time_ms']:.2f} | {results['fps']:.2f} | {real_time} |")

    report.append("\n## 2. 模型大小对比")

    if size_results:
        report.append("\n| 模型 | 参数量(M) | 模型大小(MB) |")
        report.append("|------|----------|--------------|")

        for model_name, results in size_results.items():
            report.append(f"| {model_name} | {results['params_m']:.2f} | {results['size_mb']:.1f} |")

    report.append("\n## 3. 结论")

    if speed_results:
        fastest_model = min(speed_results.keys(), key=lambda x: speed_results[x]['avg_time_ms'])
        report.append(f"\n- **最快模型**: {fastest_model} ({speed_results[fastest_model]['avg_time_ms']:.2f}ms)")

        real_time_models = [name for name, results in speed_results.items() if results['avg_time_ms'] <= 33.33]
        if real_time_models:
            report.append(f"- **满足实时性**: {', '.join(real_time_models)}")
        else:
            report.append("- **满足实时性**: 无")

    if size_results:
        smallest_model = min(size_results.keys(), key=lambda x: size_results[x]['params_m'])
        report.append(f"- **最小模型**: {smallest_model} ({size_results[smallest_model]['params_m']:.2f}M参数)")

    report.append("\n## 4. 建议")
    report.append("\n- 如果追求推理速度: 优先选择满足实时性要求的模型")
    report.append("- 如果考虑模型大小: 平衡性能和资源消耗")
    report.append("- 建议在实际机器人上进行最终测试")

    # 保存报告
    report_text = "\n".join(report)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {output_file}")
    print("\n报告内容:")
    print(report_text)

def main():
    # 模型路径配置
    models = {
        'SmolVLA': {
            'path': 'outputs/train/smolvla_finetune/checkpoints/last',
            'loader': load_smolvla_model
        },
        'ACT': {
            'path': '/home/chenqingyu/robot/new_lerobot/outputs/train/grasp5/checkpoints/last/pretrained_model',
            'loader': load_act_model
        },
        'ACT-DINOv2': {
            'path': '/home/chenqingyu/robot/new_lerobot/outputs/train/grasp5/checkpoints/last/pretrained_model',
            'loader': load_act_model  # 需要区分
        }
    }

    dataset_path = "/home/chenqingyu/robot/lerobot-20251011/datasets/grasp_dataset_v30"

    print("=" * 60)
    print("SmolVLA模型对比分析")
    print("=" * 60)

    # 加载模型
    loaded_models = {}

    for model_name, model_info in models.items():
        print(f"\n加载 {model_name} 模型...")
        loader = model_info['loader']
        path = model_info['path']

        try:
            model, device = loader(path)
            loaded_models[model_name] = {
                'model': model,
                'device': device,
                'path': path
            }
            print(f"✓ {model_name} 加载成功")
        except Exception as e:
            print(f"❌ {model_name} 加载失败: {e}")
            loaded_models[model_name] = {
                'model': None,
                'device': None,
                'path': path
            }

    # 对比分析
    speed_results = compare_inference_speed(loaded_models, dataset_path)
    size_results = compare_model_size(loaded_models)

    # 生成报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"outputs/eval/model_comparison_{timestamp}.md"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    generate_comparison_report(speed_results, size_results, output_file)

    print("\n" + "=" * 60)
    print("对比分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/compare_models.py
```

---

## 四、风险评估与缓解措施

### 4.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **数据集转换失败** | 中 | 高 | 1. 多重备份策略<br>2. 官方转换脚本<br>3. 详细验证流程 |
| **GPU内存不足** | 高 | 中 | 1. 梯度累积<br>2. 混合精度训练<br>3. 轻量级配置 |
| **推理速度不满足实时性** | 中 | 中 | 1. 模型优化<br>2. 量化技术<br>3. 异步推理 |
| **依赖冲突** | 中 | 低 | 1. 独立conda环境<br>2. 版本锁定<br>3. 容器化部署 |
| **模型效果不佳** | 中 | 中 | 1. 预训练微调<br>2. 超参数调优<br>3. 数据增强 |

### 4.2 项目风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **训练时间超预期** | 中 | 低 | 1. 并行训练<br>2. 云GPU备选<br>3. 早停策略 |
| **存储空间不足** | 低 | 中 | 1. 存储规划<br>2. 数据压缩<br>3. 清理旧数据 |
| **现有训练受影响** | 低 | 高 | 1. 环境隔离<br>2. 配置备份<br>3. 快速回滚方案 |

### 4.3 业务风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **SmolVLA效果不如ACT** | 中 | 高 | 1. 对比测试<br>2. 性能基准<br>3. 回退方案 |
| **学习成本高** | 中 | 低 | 1. 详细文档<br>2. 培训计划<br>3. 技术支持 |
| **长期维护复杂** | 中 | 中 | 1. 自动化脚本<br>2. 监控体系<br>3. 知识沉淀 |

---

## 五、资源需求详细规划

### 5.1 硬件资源

**GPU配置**:
- **推荐配置**: RTX 4090 (24GB) × 1
- **最低配置**: RTX 3090 (24GB) × 1
- **备选方案**: A5000 (24GB) 或云GPU

**CPU配置**:
- **推荐配置**: 8核+ CPU
- **内存需求**: 32GB+ (推荐64GB)

**存储需求**:
```
原始数据集: ~50GB
转换后数据: ~50GB
训练输出: ~20GB
日志文件: ~5GB
备份空间: ~50GB
总计: ~175GB
```

### 5.2 软件资源

**系统要求**:
- Ubuntu 20.04/22.04 LTS
- CUDA 11.8+ / 12.0+
- Python 3.10+

**关键依赖**:
```python
pytorch>=2.2.1
transformers>=4.52.0
accelerate>=1.7.0
safetensors>=0.4.3
num2words>=0.5.14
```

### 5.3 时间资源

**详细时间分解**:
| 阶段 | 预计时间 | 说明 |
|------|---------|------|
| 环境准备 | 2-4小时 | 包括依赖安装和配置 |
| 数据集转换 | 2-4小时 | 700 episodes数据 |
| 小规模测试 | 1-2小时 | 验证完整流程 |
| 微调训练 | 6-12小时 | 50,000步 |
| 从头训练 | 2-5天 | 200,000步 |
| 模型评估 | 1-2小时 | 性能和速度测试 |
| 对比分析 | 1-2小时 | 与ACT模型对比 |
| **总计(微调)** | **12-24小时** | **推荐方案** |
| **总计(从头)** | **3-6天** | **备选方案** |

---

## 六、成功标准与验收指标

### 6.1 技术指标

**训练完成标准**:
- ✅ Loss稳定下降并收敛
- ✅ 训练完成预定步数
- ✅ 模型checkpoint保存成功
- ✅ 无明显过拟合现象

**性能指标**:
- **动作预测MSE**: < 0.01 (基于实际数据调整)
- **推理速度**: > 30 FPS (满足实时控制要求)
- **成功率**: > 80% (实际机器人测试)
- **模型大小**: < 2GB (部署友好)

### 6.2 对比基准

**与ACT模型对比**:
| 指标 | SmolVLA目标 | ACT基准 | 说明 |
|------|-------------|---------|------|
| 推理速度 | > 25 FPS | > 50 FPS | ACT更快 |
| 模型大小 | < 2GB | < 500MB | ACT更小 |
| 任务成功率 | > 80% | > 85% | ACT经验丰富 |
| 训练时间 | < 1天 | < 6小时 | ACT更快 |

**验收标准**:
- ✅ 推理速度满足实时性要求
- ✅ 模型成功率不低于ACT的90%
- ✅ 训练过程可复现
- ✅ 可以在实际机器人上运行

### 6.3 交付物清单

**代码交付物**:
- [ ] 完整的训练脚本 (`scripts/train_smolvla.py`)
- [ ] 评估脚本 (`scripts/evaluate_model.py`)
- [ ] 推理测试脚本 (`scripts/benchmark_inference.py`)
- [ ] 对比分析脚本 (`scripts/compare_models.py`)
- [ ] 监控工具 (`scripts/monitor_training.py`)

**模型交付物**:
- [ ] 训练好的SmolVLA模型权重
- [ ] 模型配置文件
- [ ] 训练日志和checkpoint
- [ ] 性能评估报告

**文档交付物**:
- [ ] 技术实施方案 (本文档)
- [ ] 训练配置说明
- [ ] 故障排除指南
- [ ] 模型对比分析报告

---

## 七、项目实施时间表

### 7.1 详细时间线

```
第1天 (4-6小时):
├── 环境准备 (2小时)
├── 依赖安装 (1小时)
├── 配置验证 (1小时)
└── 小规模测试 (1小时)

第2天 (6-8小时):
├── 数据集备份 (0.5小时)
├── 数据集转换 (4小时)
├── 转换验证 (1小时)
└── 测试训练 (1小时)

第3天 (6-12小时):
├── 微调训练启动 (0.5小时)
├── 训练监控 (8-10小时)
├── 中期检查 (1小时)
└── 训练完成处理 (0.5小时)

第4天 (2-4小时):
├── 模型评估 (1小时)
├── 推理速度测试 (1小时)
├── 对比分析 (1小时)
└── 报告生成 (1小时)

总计: 18-30小时 (推荐方案)
```

### 7.2 里程碑节点

| 里程碑 | 时间节点 | 验收标准 |
|--------|----------|----------|
| **M1: 环境就绪** | 第1天结束 | 所有依赖安装成功，测试通过 |
| **M2: 数据准备完成** | 第2天结束 | 数据集转换成功，验证通过 |
| **M3: 训练启动** | 第3天开始 | 微调训练正常启动 |
| **M4: 训练完成** | 第3天结束 | 训练完成50,000步 |
| **M5: 模型验证** | 第4天结束 | 评估完成，性能达标 |
| **M6: 项目交付** | 第4天结束 | 所有交付物完成 |

---

## 八、后续工作规划

### 8.1 短期优化 (1-2周)

**性能优化**:
1. **推理加速**:
   - 使用torch.compile优化
   - 模型量化 (INT8/FP16)
   - TensorRT部署

2. **模型调优**:
   - 超参数精细调整
   - 数据增强策略
   - 学习率调度优化

**功能完善**:
1. **自动化工具**: 训练流程完全自动化
2. **监控体系**: 实时性能监控和告警
3. **部署方案**: 生产环境部署脚本

### 8.2 中期发展 (1-2月)

**多任务扩展**:
1. **数据收集**: 收集更多类型的任务数据
2. **任务设计**: 设计多样化的抓取任务
3. **多任务训练**: 利用SmolVLA的多任务能力

**系统集成**:
1. **ROS集成**: 与ROS系统集成
2. **Web界面**: 开发模型管理和监控界面
3. **API服务**: 提供模型推理API服务

### 8.3 长期规划 (3-6月)

**算法升级**:
1. **新算法测试**: 测试LeRobot生态的新算法
2. **模型融合**: 探索模型融合和集成方法
3. **在线学习**: 实现在线学习和适应能力

**产业化应用**:
1. **标准化流程**: 建立标准化的训练部署流程
2. **性能优化**: 针对具体应用场景的深度优化
3. **知识转移**: 技术知识向团队转移

---

## 九、关键技术细节

### 9.1 SmolVLA算法特点

**架构组成**:
- **视觉编码器**: SmolVLM2-500M-Video-Instruct
- **动作专家网络**: MLP架构
- **语言理解**: 支持自然语言指令

**训练方法**:
- **Flow Matching**: 替代传统行为克隆
- **多模态输入**: 图像+状态+语言
- **序列预测**: 预测动作序列而非单步

**适用场景**:
- ✅ 多任务场景
- ✅ 语言条件化任务
- ⚠️ 单任务场景 (可能overkill)
- ❌ 简单重复任务 (ACT可能更好)

### 9.2 数据集格式关键差异

**v2.1 → v3.0 主要变化**:

| 方面 | v2.1 | v3.0 | 影响 |
|------|------|------|------|
| **episode文件** | 每个episode一个文件 | 多个episode合并存储 | 存储效率提升 |
| **元数据格式** | JSONL | Parquet | 查询性能提升 |
| **数据路径模板** | `episode_{index}` | `file_{index}` | 需要适配代码 |
| **统计信息** | 分散存储 | 集中存储 | 更好的数据管理 |

### 9.3 训练配置关键参数

**微调推荐配置**:
```python
# 基础配置
pretrained_model = "lerobot/smolvla_base"
batch_size = 8  # 根据GPU内存调整
learning_rate = 5e-5  # 微调用较小学习率
steps = 50000  # 微调需要更少步数

# 优化器配置
optimizer = "AdamW"
betas = (0.9, 0.95)
weight_decay = 1e-10
grad_clip_norm = 10.0

# 学习率调度
scheduler = "cosine_decay"
warmup_steps = 2000
decay_steps = 40000
min_lr = 2.5e-6

# 性能优化
use_amp = True  # 混合精度训练
amp_dtype = "bf16"  # bfloat16优先
```

### 9.4 故障排除指南

**常见问题及解决方案**:

1. **CUDA Out of Memory**
   ```bash
   # 解决方案1: 减小batch_size
   --batch_size=4

   # 解决方案2: 梯度累积
   --batch_size=2 --gradient_accumulation_steps=4

   # 解决方案3: 混合精度
   --use_amp=true --amp_dtype=fp16
   ```

2. **数据集加载错误**
   ```bash
   # 检查数据集版本
   python -c "import json; print(json.load(open('datasets/grasp_dataset_v30/meta/info.json'))['codebase_version'])"

   # 重新验证数据集
   python scripts/verify_dataset_v30.py
   ```

3. **预训练模型下载失败**
   ```bash
   # 设置HF缓存目录
   export HF_HOME="/path/to/cache"

   # 手动下载
   python -c "from lerobot.policies.smolvla import SmolVLAPolicy; SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')"
   ```

4. **推理速度慢**
   ```python
   # 解决方案1: torch.compile
   policy = torch.compile(policy, mode="reduce-overhead")

   # 解决方案2: 混合精度推理
   policy = policy.half()

   # 解决方案3: 减小输入尺寸
   policy.config.resize_imgs_with_padding = (384, 384)
   ```

---

## 十、结论与建议

### 10.1 核心结论

基于对C4.5、Codex、GPT-5三人技术方案的深入分析和整合，结合代码仓库的实际情况，得出以下核心结论：

1. **技术可行性**: ✅ **高度可行**
   - 新框架完全支持SmolVLA训练
   - 数据集转换技术成熟
   - 现有硬件资源满足要求

2. **实施风险**: 🟡 **中等可控**
   - 主要风险在数据集转换和GPU内存
   - 有完整的缓解措施
   - 可以快速回退到现有方案

3. **投入产出**: ✅ **性价比高**
   - 时间投入: 1-2天 (微调)
   - 技术收益: 掌握最新VLA技术
   - 长期价值: 为后续算法升级打基础

4. **业务价值**: 🟡 **需要验证**
   - SmolVLA在单任务场景的优势不确定
   - 需要通过实际测试验证效果
   - 建议作为技术探索而非生产替代

### 10.2 最终建议

**强烈推荐的实施方案**:

1. **采用混合方案**: 在lerobot-20251011中训练SmolVLA，保留现有环境
2. **选择微调训练**: 从预训练模型开始，而非从头训练
3. **分阶段实施**: 测试 → 微调 → 评估 → 对比 → 决策
4. **并行开发**: SmolVLA和ACT训练同时进行，便于对比

**关键成功因素**:
- ✅ 环境隔离，避免影响现有训练
- ✅ 数据备份，确保数据安全
- ✅ 小规模测试，验证完整流程
- ✅ 性能对比，客观评估效果
- ✅ 快速决策，不合适的及时调整

### 10.3 项目价值

**技术价值**:
- 掌握最新的Vision-Language-Action技术
- 积累大模型微调经验
- 为多任务机器人控制打基础

**团队价值**:
- 提升团队技术能力
- 丰富算法选择
- 增强技术储备

**业务价值**:
- 探索更先进的机器人控制方法
- 为复杂任务提供可能的技术方案
- 保持技术前瞻性

### 10.4 下一步行动

1. **立即行动** (今天):
   - 确认硬件资源可用性
   - 创建项目目录结构
   - 开始环境准备

2. **短期目标** (本周):
   - 完成数据集转换和验证
   - 完成小规模测试训练
   - 启动正式微调训练

3. **中期目标** (下周):
   - 完成模型训练和评估
   - 进行与ACT模型的对比
   - 形成技术决策建议

4. **长期规划** (下月):
   - 根据结果决定后续投入
   - 考虑多任务扩展
   - 规划产业化应用

---

**文档结束**

祝SmolVLA训练项目成功！如有问题，请参考本文档的故障排除章节或寻求技术支持。

**项目成功关键**: 保持灵活性，及时调整，确保与现有生产环境的兼容性。🚀

---

*本文档整合了C4.5的详细实施经验、Codex的技术分析能力和GPT-5的系统性思维，结合代码仓库的实际情况，提供了完整的SmolVLA训练实施方案。*