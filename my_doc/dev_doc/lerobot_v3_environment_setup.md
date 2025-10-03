# lerobot_v3环境搭建指南

**文档版本**: 1.0
**创建日期**: 2025-10-02
**作者**: Claude Code
**项目**: Koch机械臂lerobot-20251011环境配置

---

## 📋 环境概述

本文档记录了在`lerobot-20251011`项目中搭建`lerobot_v3`独立conda环境的完整过程，该环境专门用于SmolVLA训练和LeRobot v3.0框架验证。

### 环境基本信息
- **环境名称**: `lerobot_v3`
- **Python版本**: 3.10.18
- **框架版本**: LeRobot v0.3.4
- **目标用途**: SmolVLA训练、ACT模型验证、新框架测试

---

## 🔧 系统环境检查

### 当前硬件配置
```bash
# 检查当前目录
pwd
# 输出: /home/chenqingyu/robot/new_lerobot/lerobot-20251011

# 检查系统版本
conda --version && python --version
# 输出: conda 25.5.1, Python 3.13.5

# 检查GPU状态
nvidia-smi
# 输出: NVIDIA GeForce RTX 4090 Laptop GPU (16GB显存)
```

### 关键系统信息
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU (16GB显存)
- **CUDA版本**: 12.9 (驱动版本 575.64.03)
- **平台**: Linux x86_64
- **工作目录**: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011`

---

## 🛠️ 环境搭建步骤

### 第1步: 创建conda环境

```bash
# 创建独立的Python 3.10环境
conda create -y -n lerobot_v3 python=3.10

# 初始化conda环境（如果需要）
eval "$(conda shell.bash hook)"

# 激活环境
conda activate lerobot_v3

# 验证Python版本
python --version
# 期望输出: Python 3.10.18
```

**注意**: 使用清华园镜像加速下载：
```bash
# conda已配置清华园镜像源
# Channels: https://mirrors.tuna.tsinghua.edu.cn/anaconda/
```

### 第2步: 安装LeRobot v3.0基础依赖

```bash
# 进入项目目录
cd /home/chenqingyu/robot/new_lerobot/lerobot-20251011

# 激活环境并安装基础依赖
eval "$(conda shell.bash hook)"
conda activate lerobot_v3

# 使用清华园镜像安装LeRobot基础包
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装过程要点**:
- 使用`-e`标志进行可编辑安装
- 启用清华园镜像`-i https://pypi.tuna.tsinghua.edu.cn/simple`
- 设置超时时间为10分钟以处理大型依赖包下载

**主要安装的包**:
- PyTorch 2.7.1+cu126 (包含CUDA支持)
- datasets>=4.0.0
- diffusers>=0.27.2
- huggingface-hub>=0.34.2
- opencv-python-headless>=4.9.0
- wandb>=0.20.0
- 以及其他必要的依赖

### 第3步: 安装SmolVLA专用依赖

```bash
# 继续在激活的环境中安装SmolVLA扩展依赖
pip install -e .[smolvla] -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**SmolVLA专用依赖包括**:
- transformers>=4.52.0
- accelerate>=1.7.0
- num2words>=0.5.14
- safetensors>=0.4.3

---

## ✅ 环境验证

### PyTorch和CUDA验证

```python
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
```

**期望输出**:
```
PyTorch: 2.7.1+cu126
CUDA available: True
CUDA version: 12.6
GPU count: 1
GPU 0: NVIDIA GeForce RTX 4090 Laptop GPU
```

### 核心组件导入验证

```python
# 正确导入所有核心组件
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
print('✓ SmolVLAConfig导入成功')

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('✓ SmolVLAPolicy导入成功')

from lerobot.policies.act.modeling_act import ACTPolicy
print('✓ ACTPolicy导入成功')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('✓ LeRobotDataset导入成功')

import lerobot
print(f'LeRobot version: {lerobot.__version__}')
```

**期望输出**:
```
✓ SmolVLAConfig导入成功
✓ SmolVLAPolicy导入成功
✓ ACTPolicy导入成功
✓ LeRobotDataset导入成功
LeRobot version: 0.3.4
```

### 可用策略检查

```python
import lerobot.policies
print('Available policies:', [x for x in dir(lerobot.policies) if not x.startswith('_')])
```

**输出应包含**:
```
['ACTConfig', 'DiffusionConfig', 'SmolVLAConfig', 'TDMPCConfig', 'VQBeTConfig',
 'act', 'diffusion', 'smolvla', 'tdmpc', 'vqbet', ...]
```

---

## 📦 环境配置总结

### 成功安装的关键组件

| 组件类型 | 名称 | 版本 | 状态 |
|---------|------|------|------|
| **Python环境** | conda env | 3.10.18 | ✅ |
| **LeRobot框架** | lerobot | 0.3.4 | ✅ |
| **深度学习** | PyTorch | 2.7.1+cu126 | ✅ |
| **Transformers** | transformers | 4.56.2 | ✅ |
| **数据集** | datasets | 4.1.1 | ✅ |
| **计算机视觉** | opencv-python | 4.12.0.88 | ✅ |
| **模型加速** | accelerate | 1.10.1 | ✅ |
| **可视化** | wandb | 0.22.1 | ✅ |

### 可用的策略模型
- ✅ **ACTPolicy** - Action Chunking Transformer
- ✅ **SmolVLAPolicy** - Vision-Language-Action模型
- ✅ **DiffusionPolicy** - 扩散模型策略
- ✅ **TDMPCPolicy** - TD-MPC策略
- ✅ **VQBeTPolicy** - VQ-BeT策略

### GPU支持状态
- ✅ CUDA 12.6 完全支持
- ✅ cuDNN 9.5.1.17 加速
- ✅ 16GB RTX 4090 显存可用

---

## 🚀 环境使用指南

### 激活环境

```bash
# 方法1: 使用conda命令
conda activate lerobot_v3

# 方法2: 初始化后激活
eval "$(conda shell.bash hook)"
conda activate lerobot_v3
```

### 验证环境状态

```bash
# 快速验证脚本
python -c "
import torch
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.act import ACTPolicy
from lerobot.datasets import LeRobotDataset
import lerobot
print(f'✅ LeRobot {lerobot.__version__} 环境就绪')
print(f'✅ PyTorch {torch.__version__} + CUDA {torch.version.cuda}')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"
```

### 下一步建议

环境搭建完成后，建议按以下顺序进行开发：

1. **数据集准备** (推荐优先级: ⭐⭐⭐⭐⭐)
   - 将grasp_dataset从v2.1转换为v3.0格式
   - 验证数据集加载和格式正确性

2. **ACT模型验证** (推荐优先级: ⭐⭐⭐⭐)
   - 在新框架中训练ACT模型验证流程
   - 与现有ACT模型性能对比

3. **SmolVLA训练** (推荐优先级: ⭐⭐⭐⭐⭐)
   - 从预训练模型开始微调
   - 配置训练参数和监控

4. **异步推理测试** (推荐优先级: ⭐⭐⭐)
   - 部署Policy Server
   - 测试Robot Client

---

## ⚠️ 注意事项和最佳实践

### 环境管理
- **独立性**: `lerobot_v3`环境与现有`lerobot`环境完全独立，避免依赖冲突
- **激活脚本**: 建议创建环境激活脚本以简化使用流程
- **备份策略**: 定期备份conda环境配置

### 性能优化
- **GPU显存**: RTX 4090的16GB显存足够SmolVLA训练需求
- **混合精度**: 可启用AMP训练节省显存
- **批处理大小**: 建议从batch_size=8开始调整

### 常见问题解决

1. **导入错误**:
   ```bash
   # 确保正确激活环境
   conda activate lerobot_v3
   # 检查PYTHONPATH设置
   echo $PYTHONPATH
   ```

2. **CUDA问题**:
   ```bash
   # 验证CUDA版本兼容性
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **依赖冲突**:
   ```bash
   # 重新安装特定包
   pip install package_name --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

---

## 📚 相关文档

- **项目主文档**: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011/CLAUDE.md`
- **技术方案**: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011/my_doc/doc/`
- **ACT框架**: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011/my_doc/new_act/`

---

## 🎯 总结

`lerobot_v3`环境已经成功搭建并验证完成！该环境具备以下能力：

- ✅ **完整的LeRobot v3.0框架支持**
- ✅ **SmolVLA模型训练能力**
- ✅ **CUDA加速和GPU计算**
- ✅ **多种策略模型支持**
- ✅ **独立隔离的开发环境**

现在可以开始进行数据集转换、模型训练和技术验证工作。

**环境状态**: 🟢 **就绪**
**下一步**: 数据集v2.1到v3.0转换
**预计开始时间**: 立即可用

---

*本文档记录了环境搭建的完整过程，供后续参考和团队协作使用。*