# v2.1 → v3.0 数据集转换实施指南（dataset_conver_Plan_aug_gpt5）

本指南基于当前仓库实际数据与源码，提供生产可用、可回滚、可验证的 grasp_dataset 从 v2.1 升级到 v3.0 的方案。

- 源数据集：/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset（v2.1）
- 目标数据集：/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30（v3.0）
- 适用代码：lerobot-20251011（CODEBASE_VERSION="v3.0"）

---

## 0. 风险声明（必读）
- 生产数据，严禁覆盖原始目录。所有转换输出至全新目录 grasp_dataset_v30。
- 转换失败可能导致结果不一致，必须严格执行“备份→转换→验证”三步。
- v2.1 与 v3.0 不兼容。新框架将对 v2.1 直接抛 BackwardCompatibilityError（见 src/lerobot/datasets/backward_compatibility.py）。

---

## 1. 当前 v2.1 数据集分析报告（基于实际盘上数据）
数据根：lerobot-20251011/grasp_dataset

### 1.1 顶层结构
- data/（分 chunk 存放 episode_*.parquet）
- images/（存在 observation.images.* 目录，v2.1 中可能用于缩略图/中间产物，转换不依赖）
- meta/
  - info.json（版本、特征、路径模板、统计汇总）
  - tasks.jsonl（任务索引与文本）
  - episodes.jsonl（每条 episode 元信息）
  - episodes_stats.jsonl（每条 episode 的统计信息）
- videos/（分 chunk 与相机键存放 episode_*.mp4）

### 1.2 meta/info.json（关键字段）
实测内容摘录：
- codebase_version: "v2.1"
- robot_type: "koch"
- total_episodes: 700
- total_frames: 237875
- fps: 30
- data_path: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
- video_path: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
- features:
  - action: float32, shape=[6]
  - observation.state: float32, shape=[6]
  - observation.images.laptop: dtype=video, 480×640×3, fps=30
  - observation.images.phone: dtype=video, 480×640×3, fps=30
  - 辅助索引：timestamp、frame_index、episode_index、index、task_index

### 1.3 meta/tasks.jsonl
- 单任务：{"task_index": 0, "task": "Place the bracelet into the box."}
- total_tasks=1（与 info.json 一致）

### 1.4 data/ 结构
- data/chunk-000/episode_000000.parquet … episode_000699.parquet（共700个，目录列举显示被截断）

### 1.5 videos/ 结构
- videos/chunk-000/observation.images.laptop/episode_000000.mp4 …（700个）
- videos/chunk-000/observation.images.phone/episode_000000.mp4 …（700个）

### 1.6 关键信息小结
- 相机：2 路（laptop、phone），键名与 info.json.features 一致。
- 状态维度：6；动作维度：6。
- 700 episodes / 237,875 帧；FPS=30；单任务场景。

---

## 2. v3.0 目标格式详解（基于源码 src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py）

### 2.1 目录与文件映射（OLD → NEW）
- data/chunk-000/episode_000000.parquet → data/chunk-000/file_000.parquet（按文件大小聚合多 episode）
- videos/chunk-000/<camera>/episode_000000.mp4 → videos/<camera>/chunk-000/file_000.mp4（每相机独立聚合）
- meta/tasks.jsonl → meta/tasks/chunk-000/file_000.parquet（随后也可聚合为 meta/tasks.parquet 或等价结构）
- meta/episodes.jsonl → meta/episodes/chunk-000/episodes_000.parquet（聚合 episode 元信息）
- meta/episodes_stats.jsonl → meta/episodes_stats/chunk-000/file_000.parquet（每 episode 统计聚合）
- meta/info.json：更新 codebase_version="v3.0"；写入 data_files_size_in_mb / video_files_size_in_mb；更新 data_path / video_path 模板；为非视频特征补齐 fps 字段。

注：DEFAULT_DATA_PATH/DEFAULT_VIDEO_PATH 在 utils 中定义，v3.0 模板分别为：
- data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet
- videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4

### 2.2 索引与统计映射
- convert_data() 输出 episodes_metadata：
  - episode_index、data/chunk_index、data/file_index
  - dataset_from_index、dataset_to_index（聚合后在新文件内的全局帧范围）
- convert_videos()/convert_videos_of_camera() 输出 per-camera 映射：
  - videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}
  - 多相机 episode 数量必须一致（否则报错）
- convert_episodes_metadata() 合并：legacy episodes.jsonl + episodes_stats.jsonl + 新的 data/video 映射 → 写入 meta/episodes/*
- write_stats()：按 episode 级统计聚合出全局 stats 写入 meta/stats.json（v3.0）

### 2.3 v3.0 必备结构（转换完成后应具备）
- data/chunk-*/file_*.parquet（至少一组）
- videos/<camera>/chunk-*/file_*.mp4（每相机至少一组）
- meta/info.json（codebase_version=v3.0）
- meta/tasks/*.parquet（或汇总任务映射）
- meta/episodes/chunk-*/episodes_*.parquet
- meta/episodes_stats/chunk-*/file_*.parquet
- meta/stats.json（全局统计）

---

## 3. 转换步骤详解（命令+预期输出）

### 3.1 备份（必做，不覆盖源）
```bash
cd /home/chenqingyu/robot/new_lerobot/lerobot-20251011
cp -r grasp_dataset grasp_dataset_backup_$(date +%Y%m%d_%H%M%S)
```
预期：生成 grasp_dataset_backup_YYYYMMDD_HHMMSS 目录，体量与源接近。

### 3.2 本地转换脚本（若尚未创建）
在仓库中新增 scripts/local_convert_v21_to_v30.py，调用源码中的转换函数，移除 Hub 依赖（仅本地目录→目录）：

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_info, convert_tasks, convert_data, convert_videos, convert_episodes_metadata
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-mb', type=int, default=100)
    p.add_argument('--video-mb', type=int, default=500)
    args = p.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    convert_info(src, dst, args.data_mb, args.video_mb)
    convert_tasks(src, dst)
    eps_meta = convert_data(src, dst, args.data_mb)
    eps_v_meta = convert_videos(src, dst, args.video_mb)
    convert_episodes_metadata(src, dst, eps_meta, eps_v_meta)

if __name__ == '__main__':
    main()
```

### 3.3 执行转换
```bash
python scripts/local_convert_v21_to_v30.py \
  --input-dir /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset \
  --output-dir /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30 \
  --data-mb 100 --video-mb 500
```
预期日志片段（示例）：
- Converting info from ... → v3.0
- Converting tasks from ...
- convert data files: 700it [...]
- convert videos of observation.images.laptop: 700it [...]
- convert videos of observation.images.phone: 700it [...]
- Converting episodes metadata from ...
- 写入 meta/stats.json 完成

### 3.4 目录结果（应见到示例）
- grasp_dataset_v30/
  - data/chunk-000/file_000.parquet（若 100MB 上限，可能 file_000, file_001 …）
  - videos/observation.images.laptop/chunk-000/file_000.mp4（同理分片）
  - videos/observation.images.phone/chunk-000/file_000.mp4
  - meta/info.json（codebase_version=v3.0）
  - meta/tasks/...parquet
  - meta/episodes/chunk-000/episodes_000.parquet
  - meta/episodes_stats/chunk-000/file_000.parquet
  - meta/stats.json

---

## 4. 转换后验证清单（必须全部通过）

### 4.1 基础结构与版本
```bash
[ -f grasp_dataset_v30/meta/info.json ] && echo OK
[ -f grasp_dataset_v30/meta/stats.json ] && echo OK
[ -d grasp_dataset_v30/data ] && echo OK
[ -d grasp_dataset_v30/videos ] && echo OK
```
检查 info.json：
- codebase_version == "v3.0"
- data_path/ video_path 模板为 file_* 形式
- data_files_size_in_mb / video_files_size_in_mb 存在
- 非视频特征（action、observation.state）含 fps 字段

### 4.2 元数据与任务
- 存在 meta/episodes/chunk-*/episodes_*.parquet
- 存在 meta/episodes_stats/chunk-*/file_*.parquet
- 存在 meta/tasks/*.parquet（或等价聚合）

### 4.3 程序化加载与抽样
```python
from pathlib import Path
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
root = Path('/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30')
meta = LeRobotDatasetMetadata(repo_id='grasp_dataset', root=root)
assert meta.info['codebase_version'].startswith('v3.')
print('episodes=', meta.total_episodes, 'frames=', meta.total_frames, 'fps=', meta.fps)

ds = LeRobotDataset('grasp_dataset', root=root)
print('len(ds)=', len(ds))
s = ds[0]
for k in ['observation.images.laptop','observation.images.phone','observation.state','action']:
    assert k in s
```
期望：
- episodes≈700；frames≈237875；fps=30
- sample 含 2 路图像、6 维状态、6 维动作、task 字段

### 4.4 多相机一致性
- 两个相机的 episode 数量一致（转换脚本已强校验，不一致会报错）
- 视频 file_* 分片与 episodes_*.parquet 的索引映射一致

---

## 5. 风险点与注意事项
- 存储与 I/O：视频拼接体量较大，建议目标目录位于 NVMe；必要时调小 --data-mb/--video-mb。
- 显存/内存：转换本身主要受内存与磁盘带宽影响；避免并行执行其他重 I/O 任务。
- 相机键名：保持 observation.images.laptop / phone 一致；v3.0 将按此键在 videos/<key>/ 下组织。
- images/ 目录：v2.1 中存在，但 convert_image_keys() 仅识别 features 中 dtype=="image" 的键。当前 info.json 中相机键为 dtype=="video"，不会作为 image 键写入 data parquet。
- 任务一致性：单任务数据 task_index 恒为 0；转换后 tasks.parquet 中应包含该映射。

---

## 6. 回滚方案
- 源目录 grasp_dataset 始终只读；转换写入 grasp_dataset_v30。
- 如验证未通过：删除 grasp_dataset_v30，修正参数或数据后重新转换；必要时从 grasp_dataset_backup_* 恢复。

---

## 7. 转换脚本使用指南（本地）

### 7.1 一次性转换
```bash
python scripts/local_convert_v21_to_v30.py \
  --input-dir /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset \
  --output-dir /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30 \
  --data-mb 100 --video-mb 500
```

### 7.2 复跑与清理
- 修改 --data-mb/--video-mb 可影响分片粒度。
- 复跑前如需清空目标目录，手动 rm -rf grasp_dataset_v30 或让脚本自动删除已存在的目标目录（参考 3.2 示例）。

---

## 8. 附录：关键源码定位（便于审计）
- 转换脚本：src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
  - convert_info / convert_tasks / convert_data / convert_videos / convert_episodes_metadata
- 版本常量：src/lerobot/datasets/lerobot_dataset.py → CODEBASE_VERSION = "v3.0"
- 兼容性报错：src/lerobot/datasets/backward_compatibility.py → BackwardCompatibilityError（针对 v2.1）

以上内容已基于实际数据与源码校验，按本指南执行可实现安全、可回滚、可验证的 v2.1 → v3.0 转换。
