# SmolVLA 在 lerobot-20251011 新框架中的完整训练方案（Final Plan by Codex）

本文汇总并统一三位同事（Claude 4.5、Codex、GPT‑5）的方案与实践经验，并结合新框架源码实际阅读校对关键差异与易错点，产出可直接落地的端到端训练手册。目标：在 lerobot-20251011 新框架中，用现有抓取数据集训练 SmolVLA，过程稳定、可回滚、可观测、便于后续扩展。

---

## 0. 来源与代码校对（重要分歧的定论）

- 三份原文档的共同主张：
  - 不把 SmolVLA 迁回旧仓库（new_lerobot）训练，直接在新框架中训练；必要时采用“双环境/双轨并行”。
  - 先小规模冒烟再长跑；独立虚拟环境；对显存/IO 做轻量化与稳定性兜底。

- 关于“数据集是否必须从 v2.1 转到 v3.0”的分歧：
  - Claude 4.5 与 Codex：必须先转换到 v3.0。
  - GPT‑5：认为 v2.x 可直接兼容。

- 代码校对（以本仓 `lerobot-20251011` 为准）：
  - `src/lerobot/datasets/lerobot_dataset.py` 中 `CODEBASE_VERSION = "v3.0"`；`LeRobotDatasetMetadata.load_metadata()` 会调用 `check_version_compatibility()`（见 `datasets/utils.py` 与 `datasets/backward_compatibility.py`）。
  - `datasets/backward_compatibility.py` 明确对 v2.1 抛 `BackwardCompatibilityError`，提示使用 v3.0 转换脚本。结论：新框架严格要求数据集为 v3.0，v2.1 不能直接加载。
  - `pyproject.toml` 中 `smolvla` extras 依赖为 `transformers>=4.52.0, accelerate>=1.7.0, safetensors>=0.4.3, num2words>=0.5.14`；训练入口脚本为 `lerobot-train`。
  - `configs/policies.py` 默认 `push_to_hub=True`，若未提供 `policy.repo_id` 将在 `configs/train.py:TrainPipelineConfig.validate()` 报错；本地训练需显式关闭。
  - 训练流程：`scripts/lerobot_train.py` 使用 `make_dataset` → `make_policy` → `make_pre_post_processors`（SmolVLA 强依赖 Processor），并在 batch 上应用 preprocessor 后训练；`GradScaler` 仅在 `policy.use_amp` 时生效。

结论：需先把 `grasp_dataset` 转为 v3.0；训练时显式 `--policy.push_to_hub=false` 以避免 Hub 校验报错；按新框架 Processor 机制组织输入/输出与正则化。

---

## 1. 总体路线与决策

- 采用“双轨并行”与最小风险策略：
  - 旧仓（`new_lerobot`）继续跑 ACT / ACT‑DINOv2，不做大改，便于回滚。
  - 新框架（`lerobot-20251011`）专门用于 SmolVLA 训练与评估。
  - 数据层面：将 `./grasp_dataset` 从 v2.1 一次性转换为 v3.0（输出到新目录），保持原数据完整备份。

- 训练策略优先级：
  - 优先微调官方预训练（`--policy.path=lerobot/smolvla_base`），显存/收敛更稳。
  - 数据/GPU 充足、域差异大时再考虑 from‑scratch（`--policy.type=smolvla` + `load_vlm_weights` 等配置），风险更高。

---

## 2. 环境准备（独立虚拟环境）

- 路径与结构（当前 CWD）：`/home/chenqingyu/robot/new_lerobot/`，新框架位于子目录 `lerobot-20251011/`。
- 要求：Python ≥ 3.10，PyTorch 与 CUDA 版本与机器匹配；建议独立虚拟环境以避免旧版 `lerobot` 包冲突。

示例：

```bash
cd /home/chenqingyu/robot/new_lerobot/lerobot-20251011

# 建议：使用 conda 或 venv
conda create -y -n lerobot_v3 python=3.10
conda activate lerobot_v3

# 安装新框架与 SmolVLA 额外依赖
pip install -e .
pip install -e .[smolvla]

# 验证入口与依赖
python -c "import lerobot, sys; print(lerobot.__version__, sys.version)"
python -c "from lerobot.policies.smolvla import SmolVLAPolicy; print('SmolVLA import OK')"
python -m lerobot.scripts.lerobot_train --help | head -n 5
```

注意：
- `torchcodec` 在部分平台不可用，视频解码会回退到 `pyav`；也可在训练时显式 `--dataset.video_backend=pyav`。
- 首次使用 `transformers` 加载 VLM/Tokenizer 可能需联网缓存（如受限场景可预下载到本机缓存后离线使用）。

---

## 3. 数据集从 v2.1 → v3.0 的转换与验证

新框架严格校验 `meta/info.json` 的 `codebase_version`（必须 `v3.0`），并要求元数据、数据与视频的路径/分片结构符合 v3.0 规范（Parquet + 按 chunk/file 组织）。

### 3.1 转换输出目录与备份

```bash
cd /home/chenqingyu/robot/new_lerobot
# 建议先做备份
cp -r grasp_dataset grasp_dataset_backup_$(date +%Y%m%d)

# 目标输出目录（本地不走 Hub）
export DS_V21=/home/chenqingyu/robot/new_lerobot/grasp_dataset
export DS_V30=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30
mkdir -p "$DS_V30"
```

### 3.2 本地转换思路（无 Hub 依赖）

`src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` 是面向 Hub 的转换脚本（默认 snapshot_download + push）。在本地场景可按脚本内部逻辑“本地化”：

- 关键步骤（对应脚本函数）：
  - `convert_info(...)`：将 `codebase_version` 改为 `v3.0`，调整 `data_path`/`video_path` 模板（`data/chunk-{chunk}/file-{file}.parquet`、`videos/{video_key}/chunk-{chunk}/file-{file}.mp4`），为非视频特征补齐 `fps`；保留 `features`，注意视频特征的 `fps` 由 `video_info` 提供。
  - `convert_tasks(...)`：把 `meta/tasks.jsonl` 转为 `meta/tasks.parquet`（索引为 task 文本，列含 `task_index`）。
  - `convert_data(...)`：把 `data/chunk-XXX/episode_YYYYYY.parquet` 聚合到 `data/chunk-XXX/file_ZZZ.parquet`，并在 episodes 元数据中记录 `dataset_from_index`/`dataset_to_index` 与 `data/chunk_index`/`data/file_index` 映射。
  - `convert_videos(...)`：对每个相机 `videos/chunk-XXX/<cam>/episode_YYYYYY.mp4` 做级联拼接到 `videos/<cam>/chunk-XXX/file_ZZZ.mp4`，并记录 `from_timestamp`/`to_timestamp` 与分片索引。
  - `convert_episodes_metadata(...)`：将 episode 级元数据（index、length、tasks、data/video 索引、每集统计）写为 `meta/episodes/chunk-XXX/episodes_YYY.parquet`；聚合全局 `stats`。

- 本地可编写一个简短的 Python 包装脚本，调用上述逻辑（去掉 Hub 相关调用），输入 `DS_V21`、输出到 `DS_V30`。如需，我可以后续为你补一个 `local_convert_v21_to_v30.py` 的最小脚本模板。

### 3.3 转换后自检

```bash
# 使用新框架自带可视化/加载脚本快速抽样验证
python -m lerobot.scripts.lerobot_dataset_viz \
  --dataset.root="$DS_V30" \
  --dataset.repo_id=grasp_dataset \
  --dataset.streaming=false

# Python 交互快速读取一条
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='grasp_dataset', root='$DS_V30', episodes=[0])
print('version:', ds.meta.info['codebase_version'])
print('features:', list(ds.meta.features.keys())[:8])
item = ds[0]
print('keys in sample:', list(item.keys())[:12])
print('has task:', 'task' in item, 'lang tokens:', 'observation.language.tokens' in item)
PY
```

期望：
- `codebase_version == "v3.0"`；
- 存在 `meta/tasks.parquet`、`meta/episodes/chunk-***/episodes_***.parquet`；
- 取样 `item` 含有语言字段 `task`（Tokenizer 会在预处理期生成 `observation.language.tokens/attention_mask`）。

---

## 4. 训练要点与配置（SmolVLA 专项）

### 4.1 核心机制（与旧框架的差异）

- 训练入口：`python -m lerobot.scripts.lerobot_train`（等同 `lerobot-train`）。
- Policy 工厂：`policies/factory.py` 基于 `policy` 配置生成 SmolVLA，并通过 `make_pre_post_processors` 组装 Processor Pipeline：
  - Preprocessor（顺序）：Rename（占位）、AddBatchDim、SmolVLANewLine（补 `task` 末尾换行）、Tokenizer、Device、Normalizer（按数据集统计与 `normalization_mapping`）。
  - Postprocessor：Unnormalizer（输出动作回到原始尺度）、Device(`cpu`)。
- 预训练微调时（`--policy.path=...`）：训练脚本会用 `dataset.meta.stats` 覆盖 Normalizer/Unnormalizer 的 `stats`，并设置 `device_processor` 到目标设备，确保与本地数据统计对齐（见 `scripts/lerobot_train.py`）。

### 4.2 关键 CLI 与默认坑位

- 强烈建议显式关闭 Hub 推送：`--policy.push_to_hub=false`（否则未设 `--policy.repo_id` 会报错）。
- 指定设备与 AMP：`--policy.device=cuda --policy.use_amp=true`（节省显存）。
- 数据集根与“repo_id”：使用本地根目录 + 任意字符串 repo_id（如 `grasp_dataset`），即可完全离线。
- 评估与保存：本地训练通常 `--eval_freq=0`，`--save_freq` 视磁盘与需求调整（如 `1000~5000`）。
- 视频解码后端：如遇解码报错或平台不支持 `torchcodec`，加 `--dataset.video_backend=pyav`。

### 4.3 显存与时序参数（SmolVLAConfig 关联）

- 默认图像预处理：`resize_imgs_with_padding=(512, 512)`；可改小（如 384）降低显存与 IO 压力。
- 时序窗口：`chunk_size=50, n_action_steps=50`（<= `chunk_size`），值越大，规划时域越长、单步延迟越高。
- 向量维度对齐：`max_state_dim=32, max_action_dim=32`，原始 6 维状态/动作会自动右侧零填充（`modeling_smolvla.pad_vector`）。
- 冻结策略：默认 `freeze_vision_encoder=True, train_expert_only=True, train_state_proj=True`；可在后期尝试解冻小步微调。

### 4.4 训练优化与调度（默认较稳）

- Optimizer 预设（AdamW）：`lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=1e-10, grad_clip_norm=10`。
- Scheduler 预设（CosineWarmup）：`warmup_steps=1000, decay_steps=30000, decay_lr=2.5e-6`。
- `--use_policy_training_preset=true`（默认）时自动注入上述预设；否则需手动传 `--optimizer.* --scheduler.*`。

---

## 5. 推荐命令模板（可直接运行）

以下命令假设：
- 新框架路径：`/home/chenqingyu/robot/new_lerobot/lerobot-20251011`
- 转换后数据集：`/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30`

### 5.1 微调预训练（推荐起步）

```bash
cd /home/chenqingyu/robot/new_lerobot/lerobot-20251011
conda activate lerobot_v3

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda --policy.use_amp=true \
  --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
  --dataset.repo_id=grasp_dataset \
  --batch_size=16 --num_workers=8 \
  --steps=20000 --save_freq=5000 --eval_freq=0 \
  --output_dir=../outputs/train/smolvla_grasp \
  --job_name=smolvla_grasp \
  --wandb.enable=false
```

轻量化（≤12GB 显存）可叠加：

```bash
  --policy.resize_imgs_with_padding="(384,384)" \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --batch_size=8
```

### 5.2 从零开始训练（仅在数据/GPU 充足时）

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.device=cuda --policy.use_amp=true \
  --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
  --dataset.repo_id=grasp_dataset \
  --batch_size=16 --num_workers=8 \
  --steps=200000 --save_freq=5000 --eval_freq=0 \
  --output_dir=../outputs/train/smolvla_scratch \
  --job_name=smolvla_scratch \
  --wandb.enable=false
```

---

## 6. 日志、断点续训与评估

- 本地日志：训练过程中周期性打印（`log_freq`），策略保存至 `--output_dir`；`save_freq` 控制保存间隔。
- 断点续训：
  - 首次训练后，目录含 `train_config.json`；续训使用：
    - `--resume=true --config_path=<output_dir>/train_config.json`
  - 脚本会基于 checkpoint 载入优化器/调度器/processor 状态（见 `scripts/lerobot_train.py`）。
- WandB：`--wandb.enable=true` 可启用云端可观测（注意内网/令牌）；
- 在线评估：`--eval_freq>0` 且配置 `env` 时，脚本会在训练期定期 `eval_policy_all()`；对真实数据通常 `--eval_freq=0`，改用离线评估脚本更实际。

---

## 7. 性能与稳定性建议（Checklist）

- I/O：若视频解码瓶颈明显，尝试：
  - `--dataset.video_backend=pyav`
  - 降低 `--policy.resize_imgs_with_padding`（如 384）
  - 提高 `--num_workers`（8~16）
- 显存：开启 AMP、减小 batch、减小 `chunk_size/n_action_steps`；必要时降低分辨率。
- 学习率：默认预设较稳；震荡大时小幅降低 `optimizer_lr` 或增加 `warmup_steps`。
- 梯度裁剪：默认 10；若爆炸可调至 5/1。
- 随机性与复现：设定 `--seed`；记录 torch/cuDNN/驱动版本。

---

## 8. 数据字段与特征映射（v2.1 → v3.0 要点）

- 路径与格式变化：
  - 数据：`data/chunk-XXX/file_YYY.parquet`（v3.0 将多个 episode 聚合到一个 file）。
  - 视频：`videos/<camera_key>/chunk-XXX/file_YYY.mp4`（每相机分别聚合）。
  - 元数据：`meta/episodes/chunk-XXX/episodes_YYY.parquet`、`meta/tasks.parquet`、`meta/stats.json`、`meta/info.json`（`codebase_version: v3.0`）。
- 关键字段：
  - `features` 中视觉键形如 `observation.images.<cam>`，SmolVLA 内部再做 `resize/pad`。
  - `task`：通过 `meta/tasks.parquet` 中 `task_index` 映射注入到 sample（Tokenizer 依赖）。
  - `fps`：非视频特征在 `stats` 里补齐；视频帧率在视频 info 中。

---

## 9. 常见报错与排障

- BackwardCompatibilityError（加载数据集时报）：
  - 说明当前数据集为 v2.1；请先按第 3 节转换为 v3.0。
- Hub 推送相关错误：
  - 未提供 `--policy.repo_id` 时，若 `--policy.push_to_hub=true` 会报错；本地训练请设为 false。
- Tokenizer 报错 / 首次下载失败：
  - 检查网络与凭据；离线场景可预下载到本机缓存（HF_HOME），再离线加载。
- 视频解码 AVError/Codec：
  - 加 `--dataset.video_backend=pyav`；或按平台安装兼容版本的 `av`。
- OOM：
  - 降低分辨率、batch、`chunk_size/n_action_steps`；开启 AMP。

---

## 10. 可选：脚本封装（便于与现有流程并行）

示例 `koch_train_smolvla.sh`（新框架目录下）：

```bash
#!/usr/bin/env bash
set -euo pipefail

conda activate lerobot_v3
export PYTHONWARNINGS="ignore::UserWarning"

DATA_ROOT="/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30"
OUT_DIR="../outputs/train/smolvla_grasp"

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda --policy.use_amp=true \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id=grasp_dataset \
  --batch_size=16 --num_workers=8 \
  --steps=20000 --save_freq=5000 --eval_freq=0 \
  --output_dir="${OUT_DIR}" \
  --job_name=smolvla_grasp \
  --wandb.enable=false
```

---

## 11. 你可能没考虑到但很关键的点（来自源码）

- Processor 覆盖逻辑：当使用 `--policy.path` 时，训练脚本会用当前数据集的 `stats` 覆盖 Normalizer/Unnormalizer，并把 DeviceProcessor 指向训练设备。若直接从零训练（`--policy.type`），则 `dataset_stats` 通过 `make_pre_post_processors` 传入，确保归一化与反归一化一致。
- `resolve_delta_timestamps`：训练前会依据 `SmolVLAConfig.observation_delta_indices/action_delta_indices` 与数据集 `fps` 生成 `delta_timestamps`，并传给 Dataset（用于对齐时序、校验时间戳；默认观测取 t=0、动作为 0..chunk_size-1）。`info.json` 的 `fps` 不正确会导致校验失败或不一致。
- `EpisodeAwareSampler`：仅当策略配置含 `drop_n_last_frames` 时启用（SmolVLA 默认没有），常规走随机打乱的 DataLoader。
- `PreTrainedConfig.push_to_hub` 默认 true：这是新框架“可发布”理念的默认值，不改掉在本地训练会强要求 `repo_id`，易踩坑。
- 图像维度：数据集的图像通常为 HWC；`dataset_to_policy_features` 在构造 PolicyFeature 时会识别并转换为 CHW；训练 batch 再经 preprocessor/processor 转为 torch 张量。
- 训练完成后的 Push：若将来需要发布，`train.py` 中在训练结束会调用 `policy.push_model_to_hub(cfg)` 以及 processor 的 `push_to_hub`；本地只需关闭即可。

---

## 12. 落地执行顺序（Checklist）

1) 创建并激活新环境，安装 `.[smolvla]`（第 2 节）
2) 备份 `grasp_dataset`，将其转换到 `grasp_dataset_v30`（第 3 节）
3) 用 `lerobot_dataset_viz` 与 Python 抽样验证（第 3.3 节）
4) 运行 200~500 steps 冒烟（第 5.1 节，`--steps=200 --save_freq=200`）
5) 观察显存/IO 与 loss 曲线，按第 7 节清单做必要调优
6) 正式长跑，配置保存与 WandB（可选）
7) 如需断点续训，使用 `--resume=true --config_path=<output_dir>/train_config.json`

---

如需：我可以为你补一个仅依赖本地文件的 `local_convert_v21_to_v30.py`（按本框架 `convert_dataset_v21_to_v30.py` 的实现裁剪 Hub 逻辑），以及把第 5 节模板整理成 `koch_train_smolvla.sh` 放在 `lerobot-20251011/` 目录下，方便与你现有 ACT 流程并行管理。

