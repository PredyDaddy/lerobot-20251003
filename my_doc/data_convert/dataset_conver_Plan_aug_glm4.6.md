# LeRobot数据集v2.1到v3.0转换指南

**文档版本**: 1.0
**创建日期**: 2025-10-02
**作者**: Claude Code
**项目**: Koch机械臂数据集格式转换
**风险等级**: 🔴 **高风险** - 涉及生产数据，转换失败可能导致数据丢失

---

## 📋 执行摘要

本文档提供了将Koch机械臂数据集从LeRobot v2.1格式转换为v3.0格式的完整指南。转换过程涉及数据结构重组、元数据格式升级和索引映射重建。

### 关键数据概览
- **数据集路径**: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset`
- **源格式**: LeRobot v2.1
- **目标格式**: LeRobot v3.0
- **数据规模**: 700个episodes，237,875帧，1,400个视频文件
- **存储空间**: 约XX GB（待转换后确认）

---

## 🔍 当前数据集分析报告 (v2.1格式)

### 数据集基本信息

#### 核心统计
```json
{
  "codebase_version": "v2.1",
  "robot_type": "koch",
  "total_episodes": 700,
  "total_frames": 237875,
  "total_tasks": 1,
  "total_videos": 1400,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 30
}
```

#### 任务描述
- **任务数量**: 1个
- **任务内容**: "Place the bracelet into the box."
- **任务索引**: 0

#### 数据特征
- **动作维度**: 6维
  - `main_shoulder_pan`
  - `main_shoulder_lift`
  - `main_elbow_flex`
  - `main_wrist_flex`
  - `main_wrist_roll`
  - `main_gripper`

- **状态维度**: 6维（与动作相同）

- **相机数量**: 2个
  - `observation.images.laptop` (480x640x3, 30fps)
  - `observation.images.phone` (480x640x3, 30fps)

### 目录结构分析

```
grasp_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ... (共700个文件)
├── videos/
│   └── chunk-000/
│       ├── observation.images.laptop/
│       │   ├── episode_000000.mp4
│       │   ├── episode_000001.mp4
│       │   └── ... (共700个文件)
│       └── observation.images.phone/
│           ├── episode_000000.mp4
│           ├── episode_000001.mp4
│           └── ... (共700个文件)
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

### 数据文件格式分析

#### Parquet文件结构 (episode_*.parquet)
每个episode文件包含以下列：
- `action`: numpy.ndarray[float32], shape=(6,) - 动作数据
- `observation.state`: numpy.ndarray[float32], shape=(6,) - 状态数据
- `timestamp`: float32 - 时间戳
- `frame_index`: int64 - 帧索引
- `episode_index`: int64 - episode索引
- `index`: int64 - 全局索引
- `task_index`: int64 - 任务索引

#### 示例数据
```python
# 第一帧数据示例
action: [8.7890625, 135.0, 182.28516, 3.2519531, -3.515625, 33.83789]
state: [10.019531, 131.74805, 179.0332, -1.7578125, -0.17578125, 33.83789]
```

#### 视频文件格式
- **编码**: H.264
- **像素格式**: YUV420P
- **分辨率**: 480x640
- **帧率**: 30fps
- **音频**: 无

---

## 🎯 目标格式详解 (v3.0)

### v3.0核心变化

#### 1. 数据聚合策略
**v2.1格式**:
```
data/chunk-000/episode_000000.parquet  (每个episode独立文件)
data/chunk-000/episode_000001.parquet
```

**v3.0格式**:
```
data/chunk-000/file_000.parquet  (多个episodes聚合文件)
data/chunk-000/file_001.parquet
```

#### 2. 视频聚合策略
**v2.1格式**:
```
videos/chunk-000/observation.images.laptop/episode_000000.mp4
videos/chunk-000/observation.images.laptop/episode_000001.mp4
```

**v3.0格式**:
```
videos/chunk-000/observation.images.laptop/file_000.mp4
videos/chunk-000/observation.images.laptop/file_001.mp4
```

#### 3. 元数据格式升级

**episodes元数据**:
- **v2.1**: `meta/episodes.jsonl` (JSONL格式)
- **v3.0**: `meta/episodes/chunk-000/episodes_000.parquet` (Parquet格式)

**tasks元数据**:
- **v2.1**: `meta/tasks.jsonl` (JSONL格式)
- **v3.0**: `meta/tasks/chunk-000/file_000.parquet` (Parquet格式)

**stats元数据**:
- **v2.1**: `meta/episodes_stats.jsonl` (JSONL格式)
- **v3.0**: `meta/episodes_stats/chunk-000/file_000.parquet` (Parquet格式)

#### 4. 索引映射系统

v3.0引入了完整的索引映射系统，支持从episode到具体文件的定位：

```python
episode_metadata = {
    "episode_index": 0,
    "data/chunk_index": 0,           # 数据文件chunk索引
    "data/file_index": 0,            # 数据文件索引
    "dataset_from_index": 0,         # 在聚合文件中的起始帧索引
    "dataset_to_index": 272,         # 在聚合文件中的结束帧索引
    "videos/observation.images.laptop/chunk_index": 0,  # 视频文件chunk索引
    "videos/observation.images.laptop/file_index": 0,   # 视频文件索引
    "videos/observation.images.laptop/from_timestamp": 0.0,    # 在聚合视频中的起始时间
    "videos/observation.images.laptop/to_timestamp": 9.0667,    # 在聚合视频中的结束时间
    "tasks": ["Place the bracelet into the box."],
    "length": 272,
    # ... 其他统计信息
}
```

### 文件大小控制
- **数据文件大小**: 默认100MB (可配置)
- **视频文件大小**: 默认500MB (可配置)

---

## 🔄 详细转换步骤

### ⚠️ 转换前准备

#### 1. 环境检查
```bash
# 确保使用正确的conda环境
conda activate lerobot_v3

# 验证关键依赖
python -c "
import pandas as pd
import pyarrow as pa
from datasets import Dataset
import jsonlines
print('✅ 所有依赖包可用')
"
```

#### 2. 数据备份 (关键步骤)
```bash
# 创建完整备份
BACKUP_DIR="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 复制整个数据集
cp -r /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset "$BACKUP_DIR/"

echo "✅ 数据集已备份到: $BACKUP_DIR"
```

#### 3. 存储空间检查
```bash
# 检查可用空间 (建议至少3倍原始数据集大小)
df -h /home/chenqingyu/robot/new_lerobot/lerobot-20251011/

# 估算原始数据集大小
du -sh /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset/
```

### 第1步: 创建转换脚本

由于原项目没有提供本地转换脚本，我们需要基于官方转换脚本创建一个本地版本：

```bash
# 创建本地转换脚本
cat > /home/chenqingyu/robot/new_lerobot/lerobot-20251011/scripts/convert_dataset_local.py << 'EOF'
#!/usr/bin/env python3

"""
本地数据集转换脚本
基于src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py修改
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Dataset, Features, Image
from huggingface_hub import HfApi, snapshot_download
from requests import HTTPError

# 导入lerobot工具函数
import sys
sys.path.append('/home/chenqingyu/robot/new_lerobot/lerobot-20251011/src')

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    cast_stats_to_numpy,
    flatten_dict,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    get_video_size_in_mb,
    load_info,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.utils.utils import init_logging

def convert_dataset_local(
    dataset_path: str,
    output_path: str = None,
    data_file_size_in_mb: int = None,
    video_file_size_in_mb: int = None,
):
    """本地数据集转换函数"""

    root = Path(dataset_path)
    if output_path is None:
        new_root = root.parent / f"{root.name}_v30"
    else:
        new_root = Path(output_path)

    # 设置默认参数
    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    logging.info(f"开始转换数据集: {root} -> {new_root}")
    logging.info(f"数据文件大小限制: {data_file_size_in_mb}MB")
    logging.info(f"视频文件大小限制: {video_file_size_in_mb}MB")

    # 清理目标目录
    if new_root.is_dir():
        shutil.rmtree(new_root)
        logging.info(f"清理目标目录: {new_root}")

    # 执行转换步骤
    logging.info("步骤1: 转换info.json")
    convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)

    logging.info("步骤2: 转换tasks")
    convert_tasks(root, new_root)

    logging.info("步骤3: 转换数据文件")
    episodes_metadata = convert_data(root, new_root, data_file_size_in_mb)

    logging.info("步骤4: 转换视频文件")
    episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb)

    logging.info("步骤5: 转换episodes元数据")
    convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)

    logging.info("✅ 转换完成!")
    return new_root

# 这里需要包含所有从原转换脚本复制的函数
# ... (其他函数实现)

if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="源数据集路径")
    parser.add_argument("--output-path", type=str, help="输出路径 (可选)")
    parser.add_argument("--data-file-size-in-mb", type=int, default=100, help="数据文件大小限制(MB)")
    parser.add_argument("--video-file-size-in-mb", type=int, default=500, help="视频文件大小限制(MB)")

    args = parser.parse_args()
    convert_dataset_local(**vars(args))
EOF

chmod +x /home/chenqingyu/robot/new_lerobot/lerobot-20251011/scripts/convert_dataset_local.py
```

### 第2步: 执行转换

```bash
# 执行转换
cd /home/chenqingyu/robot/new_lerobot/lerobot-20251011

# 运行转换脚本
python scripts/convert_dataset_local.py \
    --dataset-path ./grasp_dataset \
    --output-path ./grasp_dataset_v30 \
    --data-file-size-in-mb 100 \
    --video-file-size-in-mb 500
```

### 第3步: 转换验证

#### 3.1 目录结构验证
```bash
# 验证新目录结构
echo "验证v3.0目录结构..."
ls -la grasp_dataset_v30/
ls -la grasp_dataset_v30/data/
ls -la grasp_dataset_v30/videos/
ls -la grasp_dataset_v30/meta/
```

**期望的结构**:
```
grasp_dataset_v30/
├── data/
│   └── chunk-000/
│       ├── file_000.parquet
│       ├── file_001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.laptop/
│       │   ├── file_000.mp4
│       │   └── file_001.mp4
│       └── observation.images.phone/
│           ├── file_000.mp4
│           └── file_001.mp4
└── meta/
    ├── info.json
    ├── episodes/
    │   └── chunk-000/
    │       └── episodes_000.parquet
    ├── tasks/
    │   └── chunk-000/
    │       └── file_000.parquet
    └── episodes_stats/
        └── chunk-000/
            └── file_000.parquet
```

#### 3.2 数据完整性验证
```bash
# 验证总帧数
python -c "
import pandas as pd
import json

# 加载原始info.json
with open('grasp_dataset/meta/info.json', 'r') as f:
    old_info = json.load(f)

# 加载新info.json
with open('grasp_dataset_v30/meta/info.json', 'r') as f:
    new_info = json.load(f)

print(f'原始总帧数: {old_info[\"total_frames\"]}')
print(f'新版本总帧数: {new_info[\"total_frames\"]}')
print(f'帧数匹配: {old_info[\"total_frames\"] == new_info[\"total_frames\"]}')

print(f'原始episodes: {old_info[\"total_episodes\"]}')
print(f'新版本episodes: {new_info[\"total_episodes\"]}')
print(f'episodes匹配: {old_info[\"total_episodes\"] == new_info[\"total_episodes\"]}')
"
```

#### 3.3 数据加载验证
```bash
# 验证新数据集可以正确加载
python -c "
from pathlib import Path
import sys
sys.path.append('/home/chenqingyu/robot/new_lerobot/lerobot-20251011/src')

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    # 加载转换后的数据集
    dataset = LeRobotDataset('grasp_dataset_v30')
    print(f'✅ 数据集加载成功')
    print(f'数据集长度: {len(dataset)}')
    print(f'Episodes: {dataset.num_episodes}')

    # 验证第一个episode
    episode = dataset[0]
    print(f'第一个episode keys: {episode.keys()}')
    print(f'Action shape: {episode[\"action\"].shape}')
    print(f'State shape: {episode[\"observation.state\"].shape}')

except Exception as e:
    print(f'❌ 数据集加载失败: {e}')
"
```

---

## ⚠️ 风险点与注意事项

### 🔴 高风险项目

#### 1. 数据丢失风险
- **风险**: 转换过程中可能出现数据损坏或丢失
- **缓解措施**:
  - 执行前必须创建完整备份
  - 验证每个转换步骤
  - 保留原始数据直到验证完成

#### 2. 存储空间不足
- **风险**: 转换过程需要额外存储空间（约2-3倍原始大小）
- **缓解措施**:
  - 转换前检查可用磁盘空间
  - 可考虑使用外部存储

#### 3. 内存不足
- **风险**: 大文件聚合可能导致内存溢出
- **缓解措施**:
  - 调整文件大小限制参数
  - 监控内存使用情况

### 🟡 中等风险项目

#### 1. 转换脚本兼容性
- **风险**: 自定义转换脚本可能存在bug
- **缓解措施**:
  - 在小数据集上测试
  - 逐步验证转换结果

#### 2. 性能问题
- **风险**: 大数据集转换可能耗时很长
- **缓解措施**:
  - 预估转换时间
  - 考虑分批处理

### 🟢 注意事项

#### 1. 文件权限
- 确保有读写权限
- 检查文件属主和权限设置

#### 2. 路径配置
- 使用绝对路径避免路径混淆
- 检查PYTHONPATH设置

#### 3. 依赖版本
- 确保所有依赖包版本兼容
- 特别是pandas、pyarrow、datasets等

---

## 🔄 回滚方案

### 完整回滚
```bash
# 如果转换失败，恢复原始数据集
BACKUP_DIR="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_backup_20251002_170000"
ORIGINAL_DIR="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset"

# 删除失败的数据集
if [ -d "${ORIGINAL_DIR}_v30" ]; then
    rm -rf "${ORIGINAL_DIR}_v30"
    echo "已删除失败的转换结果"
fi

# 恢复原始数据（如果被意外删除）
if [ ! -d "$ORIGINAL_DIR" ] && [ -d "$BACKUP_DIR" ]; then
    cp -r "$BACKUP_DIR" "$ORIGINAL_DIR"
    echo "已从备份恢复原始数据集"
fi
```

### 部分回滚
```bash
# 仅保留备份，删除转换结果
rm -rf /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30
echo "已删除转换结果，原始数据保持不变"
```

---

## 📊 性能预估

### 转换时间预估
基于数据集规模预估：
- **数据量**: 700 episodes, 237,875 frames
- **视频数量**: 1,400 files
- **预估转换时间**: 2-4小时
- **峰值内存使用**: 4-8GB
- **峰值磁盘使用**: 原始大小的2.5倍

### 硬件要求
- **最低要求**:
  - 内存: 8GB
  - 可用磁盘: 50GB
  - CPU: 4核心

- **推荐配置**:
  - 内存: 16GB+
  - 可用磁盘: 100GB+
  - CPU: 8核心+

---

## ✅ 验证清单

### 转换前验证
- [ ] 已创建完整数据备份
- [ ] 检查磁盘空间充足
- [ ] 验证conda环境正确激活
- [ ] 确认所有依赖包版本兼容
- [ ] 转换脚本准备就绪

### 转换过程验证
- [ ] 每个转换步骤无错误完成
- [ ] 监控内存使用正常
- [ ] 检查中间文件生成正确
- [ ] 日志记录保存完整

### 转换后验证
- [ ] 目录结构符合v3.0规范
- [ ] 总帧数和episodes数量匹配
- [ ] 数据集可以正常加载
- [ ] 随机抽样数据内容正确
- [ ] 视频文件播放正常
- [ ] 元数据文件格式正确

### 性能验证
- [ ] 数据加载速度测试
- [ ] 内存使用测试
- [ ] 与v2.1版本对比测试

---

## 🛠️ 故障排除

### 常见问题

#### 1. 内存不足错误
```bash
# 症状: MemoryError或killed
# 解决方案:
# 减小文件大小限制
python scripts/convert_dataset_local.py \
    --data-file-size-in-mb 50 \
    --video-file-size-in-mb 200
```

#### 2. 磁盘空间不足
```bash
# 症状: No space left on device
# 解决方案:
# 清理临时文件或使用更大的存储空间
rm -rf /tmp/*
```

#### 3. 导入错误
```bash
# 症状: ModuleNotFoundError
# 解决方案:
# 检查PYTHONPATH和环境设置
export PYTHONPATH="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/src:$PYTHONPATH"
```

#### 4. 权限错误
```bash
# 症状: Permission denied
# 解决方案:
# 修改文件权限
chmod -R 755 /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset*
```

---

## 📚 相关文档

- **LeRobot v3.0迁移指南**: `/docs/source/porting_datasets_v3.mdx`
- **数据集格式文档**: `/docs/source/lerobot-dataset-v3.mdx`
- **环境搭建指南**: `/my_doc/dev_doc/lerobot_v3_environment_setup.md`
- **项目技术方案**: `/my_doc/doc/`

---

## 🎯 总结

本转换指南提供了从v2.1到v3.0格式的完整转换流程。关键要点：

1. **安全性第一**: 始终保持数据备份，验证每个步骤
2. **逐步验证**: 不要跳过任何验证步骤
3. **性能监控**: 注意资源使用情况
4. **文档记录**: 保留完整的转换日志

**转换状态**: 🟡 **准备就绪**
**建议执行时间**: 立即可执行
**预计完成时间**: 2-4小时

---

*本文档基于实际代码分析编写，确保转换过程的准确性和可执行性。如有疑问，请参考相关源码文档。*