# LeRobot 数据集 v2.1 → v3.0 本地转换指南（Codex 版）

本指南基于你当前仓库与真实数据编写，覆盖现状盘点、v3.0 目标结构、转换步骤、风险与回滚、验证清单，并给出可直接运行的本地转换脚本用法。所有示例均以实际路径与源码为依据。

- 源数据集: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset`
- 当前版本: v2.1（meta/info.json 明确标注）
- 目标版本: v3.0（新框架严格要求）

---

## 1. 当前数据集分析报告（v2.1）

- 顶层结构
  ```bash
  ls -la --group-directories-first /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset
  ```

- 元数据（实际存在）
  ```bash
  ls -la /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset/meta
  ```
  关键结论：
  - `meta/info.json`: `codebase_version` 为 `v2.1`；`total_episodes: 700`；`total_frames: 237875`；`fps: 30`
  - `meta/tasks.jsonl`: 单任务（示例：Place the bracelet into the box.）
  - `meta/episodes.jsonl`、`meta/episodes_stats.jsonl` 存在
  - `meta/stats.json` 缺失（v2.1 常见）

- 数据/视频组织
  ```bash
  # Episodes parquet 数量
  find /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset/data -maxdepth 2 -type f -name 'episode_*.parquet' | wc -l
  # 700

  # 相机与视频文件
  find /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset/videos -type f -name 'episode_*.mp4' | wc -l
  # 1400
  find /home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset/videos -mindepth 2 -maxdepth 2 -type d | sed 's#.*/videos/##' | sort
  # chunk-000/observation.images.laptop
  # chunk-000/observation.images.phone
  ```

- 特征维度（基于真实 parquet 读取）
  ```bash
  python - <<'PY'
  import pyarrow.parquet as pq
  p = "/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset/data/chunk-000/episode_000000.parquet"
  t = pq.read_table(p)
  row0 = {c: t[c][0].as_py() for c in t.column_names}
  for k in ['observation.state','action']:
      v = row0.get(k)
      print(k, 'len=', len(v))
  PY
  # observation.state len=6, action len=6
  ```

---

## 2. v3.0 目标格式详解（基于源码）

参考：`src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`、`src/lerobot/datasets/utils.py`

- 路径模板（utils.py）
  - `DEFAULT_DATA_PATH`: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
  - `DEFAULT_VIDEO_PATH`: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
- 元数据升级
  - `meta/tasks.parquet`（代替 `tasks.jsonl`）
  - `meta/episodes/chunk-XXX/file-XXX.parquet`（代替 `episodes.jsonl`）
  - `meta/stats.json`（聚合统计，代替 `episodes_stats.jsonl`）
- info.json 更新（convert_info）
  - `codebase_version` 置为 `v3.0`
  - 替换 `data_path`/`video_path` 为上述模板；为非视频特征写入 `fps`

必需目录与文件（v3.0）：
- `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`
- `meta/episodes/chunk-XXX/file-XXX.parquet`
- `data/chunk-XXX/file-XXX.parquet`
- `videos/<video_key>/chunk-XXX/file-XXX.mp4`

聚合与索引（convert_* 逻辑）：
- 文件滚动阈值：`DEFAULT_DATA_FILE_SIZE_IN_MB=100`、`DEFAULT_VIDEO_FILE_SIZE_IN_MB=500`
- 编号推进：`update_chunk_file_indices()`；`DEFAULT_CHUNK_SIZE=1000`
- `convert_episodes_metadata()` 将 legacy 元数据 + stats + data/video 索引合并写入 `meta/episodes/*.parquet` 并写出 `meta/stats.json`

---

## 3. 转换步骤详解（本地、离线、安全）

强烈建议：先备份、再输出到新目录，确保可回滚。

### 3.1 备份与准备

```bash
SRC="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset"   # v2.1 源
DST="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30" # v3.0 目标
BK="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v21_backup"

cp -a "$SRC" "$BK"               # 只读备份
jq -r '.codebase_version,.total_episodes,.total_frames,.fps' "$SRC/meta/info.json"
[[ -e "$DST" ]] && echo "[!] $DST exists" && exit 1 || true
mkdir -p "$DST"
```

### 3.2 本地转换脚本（不依赖 Hub）

创建 `scripts/local_convert_v21_to_v30.py`：
```bash
mkdir -p scripts
cat > scripts/local_convert_v21_to_v30.py <<'PY'
#!/usr/bin/env python3
import argparse, logging
from pathlib import Path
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_info, convert_tasks, convert_data, convert_videos, convert_episodes_metadata,
)
from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--data-mb', type=int, default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    ap.add_argument('--video-mb', type=int, default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    args = ap.parse_args()

    src = Path(args.input_dir).resolve()
    dst = Path(args.output_dir).resolve()
    if not src.exists():
        raise SystemExit(f'[!] Source not found: {src}')
    if dst.exists():
        raise SystemExit(f'[!] Output already exists: {dst}')

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info(f'Source: {src}')
    logging.info(f'Target: {dst}')

    convert_info(src, dst, args.data_mb, args.video_mb)
    convert_tasks(src, dst)
    episodes_metadata = convert_data(src, dst, args.data_mb)
    episodes_videos_metadata = convert_videos(src, dst, args.video_mb)
    convert_episodes_metadata(src, dst, episodes_metadata, episodes_videos_metadata)
    logging.info('✓ Conversion completed')

if __name__ == '__main__':
    main()
PY
chmod +x scripts/local_convert_v21_to_v30.py
```

执行转换：
```bash
python scripts/local_convert_v21_to_v30.py \
  --input-dir  "/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset" \
  --output-dir "/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30" \
  --data-mb 100 --video-mb 500
```

### 3.3 转换后快速自检

```bash
DST="/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30"

# 版本与模板
jq -r '.codebase_version, .data_path, .video_path' "$DST/meta/info.json"

# 任务表
python - <<'PY'
import pandas as pd
p = "/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30/meta/tasks.parquet"
df = pd.read_parquet(p)
print(df.head()); print('rows=', len(df))
PY

# episodes 与 stats
python - <<'PY'
import pandas as pd, json
root = "/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30"
ep = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
print(ep.head()); print('rows in shard=', len(ep))
print('stats keys sample=', list(json.load(open(f"{root}/meta/stats.json")).keys())[:6])
PY

# 文件存在性
find "$DST/data" -type f -name 'file-*.parquet' | wc -l
find "$DST/videos" -type f -name 'file-*.mp4' | wc -l
```

---

## 4. 风险点与注意事项

- 切勿原地覆盖源目录；务必输出到新目录并先做只读备份。
- 聚合写入需额外磁盘空间（建议 ≥ 源大小 ×1.2）。
- 自检示例依赖 `pyarrow`/`pandas`；建议在独立虚拟环境执行。
- 若发现统计或索引异常，删除目标目录，回滚备份，定位损坏 episode/视频后重跑。
- 相机键名保持与 v2.1 一致（`observation.images.laptop/phone`）。

---

## 5. 转换脚本使用指南（local_convert_v21_to_v30.py）

- 位置：`lerobot-20251011/scripts/local_convert_v21_to_v30.py`
- 用法：
  ```bash
  python scripts/local_convert_v21_to_v30.py \
    --input-dir /path/to/grasp_dataset \
    --output-dir /path/to/grasp_dataset_v30 \
    --data-mb 100 --video-mb 500
  ```
- 前置：先在当前环境安装本仓为包（`pip install -e .`）。

---

## 6. 验证清单（必须逐项通过）

- `meta/info.json.codebase_version == "v3.0"`；`data_path`/`video_path` 为 file 模式
- `total_episodes`/`total_frames` 与源一致（700/237875）
- `meta/stats.json` 存在，任务表 `meta/tasks.parquet` 可读
- 随机读取 `data/chunk-000/file-000.parquet` 列含 `action`、`observation.state`、`task_index`
- 新框架可加载：
  ```bash
  python - <<'PY'
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  ds = LeRobotDataset(repo_id='grasp_dataset', root='/home/chenqingyu/robot/new_lerobot/lerobot-20251011/grasp_dataset_v30', episodes=[0])
  print('keys:', ds[0].keys())
  PY
  ```

---

## 7. 回滚方案

- 删除 `grasp_dataset_v30`，用 `grasp_dataset_v21_backup` 覆盖恢复源目录；修复问题后重跑转换。

---

## 8. 参考（源码锚点）

- `src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`：`convert_info/convert_tasks/convert_data/convert_videos/convert_episodes_metadata`
- `src/lerobot/datasets/utils.py`：`DEFAULT_*_PATH`、`DEFAULT_*_SIZE_IN_MB`、`update_chunk_file_indices()`、`write_*`

（建议先在少量 episode 子集上做试转与验证，确认通过后再对全量数据执行。）
