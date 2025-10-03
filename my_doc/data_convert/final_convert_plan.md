# LeRobot v2.1 → v3.0 本地转换最终方案（final_convert_plan）

本方案综合三份建议文档（codex / glm4.6 / gpt5）并结合当前代码仓库实际实现，给出一套可本地落地、可回滚、可验证的 v2.1 → v3.0 转换计划。已对源码进行审计，确认涉及的关键函数、路径与约束条件均存在且可用。

- 源数据集（v2.1）: `grasp_dataset`
- 目标数据集（v3.0）: `grasp_dataset_v30`
- 代码仓库根: `/home/chenqingyu/robot/new_lerobot/lerobot-20251011`

## 一、目标与范围
- 目标：将现有本地 v2.1 数据集按 v3.0 规范重排与升级（聚合 data、合并视频、升级 meta），不依赖 Hugging Face Hub 推拉。
- 范围：仅目录到目录的本地转换；不改写源目录；输出完整 v3.0 结构与元数据，满足 `LeRobotDataset` 本地加载。

风险声明（吸收 glm4.6 与 gpt5 强调）
- 风险等级：高（生产数据）。严禁覆盖源目录，务必先完整备份；转换-验证-再切换。
- v2.1 与 v3.0 不兼容：旧版数据在 v3.0 代码中会触发 BackwardCompatibilityError（参见 `src/lerobot/datasets/backward_compatibility.py`）。

## 二、现状确认（依据真实数据与源码）
- `grasp_dataset/meta/info.json`：`codebase_version="v2.1"`、`total_episodes=700`、`total_frames=237875`、`fps=30`，`video_path` 为 episode_* 旧模板；有 2 路视频键：`observation.images.laptop`、`observation.images.phone`（dtype=video）。
- 旧式元数据存在：`meta/episodes.jsonl`、`meta/episodes_stats.jsonl`、`meta/tasks.jsonl`；`meta/stats.json`（v2.1）不存在。
- 数据与视频：`data/chunk-000/episode_*.parquet`（700 个）；`videos/chunk-000/<camera>/episode_*.mp4`（每相机 700 个）。
- 源码核对：
  - 转换函数：`src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`
    - `convert_info`、`convert_tasks`、`convert_data`、`convert_videos`、`convert_episodes_metadata` 均支持以本地 `root/new_root` 目录为输入输出。
  - 常量与模板：`src/lerobot/datasets/utils.py`
    - `DEFAULT_DATA_PATH = data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
    - `DEFAULT_VIDEO_PATH = videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
    - 写入函数：`write_info`、`write_tasks`、`write_episodes`、`write_stats`。
  - 加载校验：`src/lerobot/datasets/lerobot_dataset.py`（`LeRobotDatasetMetadata`/`LeRobotDataset` 可指向本地 `root` 加载 v3.0 数据集）。

结论：仓库已具备离线转换所需的全部函数与路径模板，计划可行。

## 三、v3.0 目标结构与关键变更
- 目录结构（新增/调整）
  - `data/chunk-XXX/file_XXX.parquet`（按文件大小聚合多 episode）
  - `videos/<video_key>/chunk-XXX/file_XXX.mp4`（每相机独立聚合）
  - `meta/info.json`（升级 `codebase_version`→`v3.0`；新增 `data_files_size_in_mb`、`video_files_size_in_mb`；更新 `data_path`/`video_path` 模板；非视频特征补齐 `fps`）
  - `meta/tasks.parquet`（取代 `tasks.jsonl`）
  - `meta/episodes/chunk-XXX/file-XXX.parquet`（聚合 episodes 元数据与映射）
  - `meta/stats.json`（由 episode 级统计聚合得到）
- 聚合与编号
  - 数据分片阈值：`DEFAULT_DATA_FILE_SIZE_IN_MB=100`、`DEFAULT_VIDEO_FILE_SIZE_IN_MB=500`
  - 文件编号推进：`update_chunk_file_indices()`；`DEFAULT_CHUNK_SIZE=1000`
  - 多相机一致性：各相机 episode 数量必须一致，否则抛错。

## 四、执行步骤（本地、离线、安全）

### 4.1 备份与目录准备
- 只读备份源目录，避免误操作：
  - 备份到 `grasp_dataset_backup_YYYYMMDD_HHMMSS`
  - 输出目录设为 `grasp_dataset_v30`（须不存在）

建议命令：
- `cd /home/chenqingyu/robot/new_lerobot/lerobot-20251011`
- `cp -r grasp_dataset grasp_dataset_backup_$(date +%Y%m%d_%H%M%S)`
- 确保 `grasp_dataset_v30` 不存在，否则先清理
- 检查磁盘空间：`df -h .` 与 `du -sh grasp_dataset/`（建议预留≥2.5–3× 原始体量）

### 4.2 依赖与环境
- Python 3.10+（建议使用独立虚拟环境/conda 环境）
- 依赖包：`pandas`、`pyarrow`、`datasets`、`tqdm`、`jsonlines`、`pyav`
- 路径/环境：
  - 使用绝对路径，避免路径混淆
  - 设定 `PYTHONPATH=.` 或将 `src` 加入 `PYTHONPATH`
  - 确保有读写权限（如必要可 `chmod -R 755 grasp_dataset*`）
- 存储空间：建议预留≥原始数据体量的 2.5–3×（备份 + 新版聚合文件 + 临时文件）

说明：视频拼接使用 PyAV 的无重编码拼接（`concatenate_video_files`），一般不依赖系统 ffmpeg；需保证 PyAV 可用。

### 4.3 新增本地转换脚本（仓库内）
- 位置：`scripts/local_convert_v21_to_v30.py`
- 作用：直接调用仓库内转换函数，完成目录→目录的本地转换；参数化 data/video 聚合阈值。
- 伪代码要点：
  - 解析 `--input-dir`、`--output-dir`、`--data-mb`、`--video-mb`
  - 顺序执行：`convert_info` → `convert_tasks` → `convert_data` → `convert_videos` → `convert_episodes_metadata`
  - `convert_data` 与 `convert_videos` 返回的 episodes 映射作为 `convert_episodes_metadata` 输入
  - 如目标目录已存在，可先行删除或提示退出，保证可重复执行

执行示例：
- `python scripts/local_convert_v21_to_v30.py --input-dir grasp_dataset --output-dir grasp_dataset_v30 --data-mb 100 --video-mb 500`

备注：三份文档中提供的脚本示例与此一致；仓库内函数签名与调用参数已对齐，可直接落地。

### 4.4 执行转换
- 按上一步命令运行，观察日志：
  - Converting info …
  - Converting tasks …
  - convert data files: 700it …
  - convert videos of observation.images.laptop: 700it …
  - convert videos of observation.images.phone: 700it …
  - Converting episodes metadata …
  - 写入 meta/stats.json 完成

### 4.5 转换后验证（必须通过）
- 结构存在：
  - `grasp_dataset_v30/meta/info.json`、`meta/stats.json`
  - `grasp_dataset_v30/data/chunk-*/file_*.parquet`
  - `grasp_dataset_v30/videos/<camera>/chunk-*/file_*.mp4`
  - `grasp_dataset_v30/meta/episodes/chunk-*/file_*.parquet`
  - `grasp_dataset_v30/meta/tasks.parquet`
- `info.json` 核对：
  - `codebase_version == "v3.0"`
  - `data_path`/`video_path` 为 `file_*` 模板
  - `data_files_size_in_mb`、`video_files_size_in_mb` 存在
  - 非视频特征（如 `action`、`observation.state`）带 `fps` 字段
- 本地程序化加载抽检：
  - 使用 `LeRobotDatasetMetadata(repo_id='grasp_dataset', root='grasp_dataset_v30')` 读取
  - 断言 `total_episodes≈700`、`total_frames≈237875`、`fps=30`
  - 使用 `LeRobotDataset('grasp_dataset', root='grasp_dataset_v30')` 抽样 `ds[0]`，应含 2 路图像、6 维 `state` 与 `action`
- 多相机一致性：两个相机的 episode 数量一致（转换脚本已强校验）
- 额外抽查：随机播放若干输出视频 `file_*.mp4`，确认可正常播放且时长递增拼接合理

### 4.6 清理与回滚
- 如验证失败：删除 `grasp_dataset_v30`，修正参数或数据后重跑
- 永远保留 `grasp_dataset_backup_*` 备份，确保可回退

## 五、参数建议与边界场景
- 分片大小（默认）：`--data-mb=100`、`--video-mb=500`。磁盘紧张或 I/O 厚重时可调小以获得更多更小的分片。
- 仅数值数据（无视频）场景：`convert_videos` 会返回 `None`，流程仍可完成。
- `images/` 目录：v2.1 中可能存在，但以 `features` 声明为准。当前数据中相机 dtype=video，不会作为 image 键写入 data parquet。
- 多相机不一致：若各相机 episode 数不一致，脚本会报错并终止；需先修复源数据。
- 内存与 I/O：`convert_data` 使用 pandas 读写，`concat_data_files` 会将待并入的 parquet 汇总到内存中；700 集（默认参数）在常规工作站可运行，但建议转换时避免并发大 I/O 任务；目标目录建议位于 NVMe 盘。

故障排除（吸收 glm4.6）
- MemoryError/进程被杀：减小 `--data-mb`/`--video-mb`
- No space left on device：清理临时文件或更换更大磁盘
- ModuleNotFoundError：检查 `PYTHONPATH` 与依赖是否安装
- Permission denied：修正目录权限为可读写

## 六、时间、资源与硬件建议（参考）
- 时间：2–4 小时，受磁盘带宽与视频拼接速度影响
- 磁盘：建议预留≥原始数据体量 2.5–3×（备份 + 新版 + 中间文件）
- 内存：峰值约 4–8GB；最低 8GB，推荐 16GB+
- 计算：主要受 I/O 限制；CPU/GPU 不是主要瓶颈

## 七、可行性审计（基于源码）
- 已核对关键函数与常量：
  - `convert_dataset_v21_to_v30.py` 中的 `convert_info/convert_tasks/convert_data/convert_videos/convert_episodes_metadata` 与 `utils.py` 中的 `DEFAULT_*_PATH`、`write_*` 等函数存在且满足本地目录调用
  - 视频拼接使用 `video_utils.concatenate_video_files`（基于 PyAV），无需外部 ffmpeg
  - `LeRobotDatasetMetadata`/`LeRobotDataset` 支持指定 `root` 指向本地目录进行加载与校验
- 目标结构模板与三份文档一致，无冲突
- 结论：本计划与当前仓库实现全面对齐，可直接落地执行

## 八、执行 Checklist（落地复用）
- [ ] 备份源目录：`grasp_dataset_backup_*`
- [ ] 准备输出目录：`grasp_dataset_v30` 不存在
- [ ] 安装依赖：`pandas`、`pyarrow`、`datasets`、`tqdm`、`jsonlines`、`pyav`
- [ ] 设置 `PYTHONPATH=.`
- [ ] 创建并检查 `scripts/local_convert_v21_to_v30.py`
- [ ] 运行转换命令（确认参数）
- [ ] 逐项验证结构、info、元数据与加载抽样
- [ ] 保留备份，必要时回滚

---

附：如需先在子集上试跑，可临时将 10–20 个 episode 复制到一处小数据集目录，按相同脚本与参数执行，确认流程与输出结构无误后再对全量数据执行。
