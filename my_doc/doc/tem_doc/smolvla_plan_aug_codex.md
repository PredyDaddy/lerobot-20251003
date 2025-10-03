# SmolVLA 集成方案（2025 年 8 月）

## 1. 背景
- 现有生产环境仍在 `new_lerobot` 仓库中运行，每天的训练流程依赖 `lerobot/scripts/train.py`（参考 `lerobot/scripts/train.py:1`），数据保存在 `./grasp_dataset`（LeRobot v2.1 数据格式）。
- 新拉取的上游版本 `lerobot-20251011`（发布版本 v0.3.4）在 `src/lerobot` 中提供了全新的代码结构，并原生支持 SmolVLA 算法。
- 目标是在保持 ACT 训练流程稳定的前提下，使用现有抓取数据集训练 SmolVLA。

## 2. 架构差异分析
- **包结构与 API**
  - 旧仓库的大部分模块位于 `lerobot/common/...`；新仓库将结构扁平化到 `lerobot/...`（例如 `lerobot/common/policies/factory.py` 与 `lerobot-20251011/src/lerobot/policies/factory.py:1`）。
  - 训练入口从 `python lerobot/scripts/train.py` 切换为 `lerobot.scripts.lerobot_train`（见 `lerobot-20251011/src/lerobot/scripts/lerobot_train.py:1`），并在主循环前注入策略级的预处理与后处理管线。
  - 异步推理（策略服务器、gRPC 通信）只在新版仓库提供（`lerobot-20251011/src/lerobot/async_inference/policy_server.py:1`）。
- **数据集管线**
  - 旧版加载器仅支持代码库版本 `v2.1`（`lerobot/common/datasets/lerobot_dataset.py:64`），使用 JSONL 元数据和按 episode 切分的 parquet 文件。
  - 新版要求 `v3.0` 数据集（`lerobot-20251011/src/lerobot/datasets/lerobot_dataset.py:52`），若仍是旧版本会抛出 `BackwardCompatibilityError`（`lerobot-20251011/src/lerobot/datasets/backward_compatibility.py:23`）。
  - v3.0 将元数据转存为 parquet，引入数据流式加载，并统一常量命名（如 `lerobot-20251011/src/lerobot/utils/constants.py:15` 中的 `OBS_LANGUAGE_TOKENS`）。
- **SmolVLA 依赖关系**
  - 策略实现位于 `lerobot-20251011/src/lerobot/policies/smolvla/*.py`，依赖新版处理器框架（`lerobot-20251011/src/lerobot/processor/...`），这些模块在 `new_lerobot` 中不存在。
  - `lerobot-20251011/pyproject.toml` 中 `smolvla = [...]`（约第 123 行）声明了额外依赖：`transformers>=4.52.0`、`accelerate>=1.7.0`、`num2words`、`safetensors>=0.4.3` 等。
- **配置与命令行差异**
  - 配置类位置及字段发生变化（旧版 `lerobot/configs/train.py` 对应新版 `lerobot-20251011/src/lerobot/configs/train.py`），并新增推送 Hub、流式数据等选项，同时要求 Python ≥3.10。
  - 新仓库通过 setuptools 暴露 `lerobot-train`、`lerobot-eval` 等命令行入口。

## 3. 方案对比
- **方案 A：在 `lerobot-20251011` 中直接训练 SmolVLA**
  - 优势：原生支持 SmolVLA（含处理器、异步推理），与官方文档一致，后续升级只需同步上游，数据集升级完成后还可解锁 PI0、HilSERL 等新策略。
  - 劣势：需要先将数据集转换到 v3.0，需要新建环境，并暂时维护一份 ACT 脚本直到迁移结束。
- **方案 B：将 SmolVLA 迁移至 `new_lerobot`**
  - 优势：无需切换环境，可继续使用现有训练脚本。
  - 劣势：需要回移处理器栈、常量、配置、异步通信、SmolVLA 依赖以及数据归一化逻辑，回归风险高且后续同步成本大，同时数据集仍停留在 v2.1 的旧格式。

## 4. 推荐结论
建议选择 **方案 A**。直接在 `lerobot-20251011` 中训练既能获得官方支持，又能避免重写处理器与数据集 v3.0 基础设施。一次性的数据集转换成本远低于在旧仓库重复实现大量功能。

## 5. 实施步骤（方案 A）
- **环境准备**：创建/激活全新的 Python ≥3.10 虚拟环境，在新仓库目录执行 `pip install -e lerobot-20251011[smolvla]`，并根据现有部署固定 CUDA/PyTorch 版本。
- **数据集升级至 v3.0**：
  - 参考 `lerobot-20251011/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`，去除与 Hugging Face Hub 交互的部分，对本地 `grasp_dataset` 运行转换，输出到新目录（例如 `grasp_dataset_v30`），并在 `info.json` 中更新 `codebase_version: v3.0`。
  - 使用新版加载器抽样验证（`HF_DATASETS_CACHE=... python -m lerobot.scripts.lerobot_dataset_viz --dataset.root=...`）。
- **训练脚本调整**：
  - 新增 SmolVLA 启动脚本（如 `koch_smolvla_train.sh`），调用 `lerobot-train`，指定转换后的数据集路径、预训练模型（`--policy.path=lerobot/smolvla_base` 或本地权重）、批大小、步数以及 `--dataset.root=/abs/path/to/grasp_dataset_v30`。
  - ACT 相关脚本暂时继续指向旧仓库，待后续迁移。
- **功能验证**：
  - 先用 `--steps=50 --save_freq=50` 做冒烟测试，确认预处理与分词流程正常。
  - 关注显存占用（默认冻结 VLM 但 512×512 图像仍需约 8–10 GB 显存）。
- **支撑流程迁移**：
  - 调整 `PYTHONPATH` 或 shell 辅助脚本，使 SmolVLA 任务加载 `lerobot-20251011`，而现有 ACT 任务仍指向 `new_lerobot`。
  - SmolVLA 训练稳定后，再规划 ACT 流程迁移，减少双仓库维护时间。
- **文档与知识传递**：
  - 在 `koch.md` 或内部文档中记录数据集转换步骤、新训练命令、关键环境变量。
  - 将定制的转换脚本与数据一起归档，方便复用。

## 6. 风险与缓解措施
- 转换脚本默认依赖 Hugging Face Hub，本地化时需确保统计信息与视频索引不被破坏。缓解措施：在数据副本上演练，并加入 episode 数量、基础统计校验。
- 不同环境间的依赖漂移（如 `torch`、`transformers`）可能影响 ACT 任务。缓解措施：隔离虚拟环境，明确记录激活命令。
- SmolVLA 默认配置（50 步动作块、512×512 图像）可能超出显存限制。缓解措施：先使用较小的 batch/chunk，逐步评估再扩大。
- 同时维护两套仓库容易引发 `PYTHONPATH` 配置错误。缓解措施：编写 shell 别名或封装脚本，显式导出对应路径。

## 7. 后续机会
- SmolVLA 训练验证完成后，将 ACT 流程迁移到 `lerobot-20251011`，统一训练脚本与数据格式。
- 考虑将转换后的数据集推送到私有 Hugging Face 仓库，利用官方的数据流式加载与可视化工具，并作为备份。
- 评估异步推理能力（`lerobot-20251011/src/lerobot/async_inference/policy_server.py`），在部署 SmolVLA 时解耦策略推理与控制执行。

## 8. 新框架训练落地方案（深化）
为便于将 `/home/chenqingyu/robot/new_lerobot/grasp_dataset` 中的旧版数据平滑迁移到 `lerobot-20251011` 训练流水线，建议按以下步骤执行，并在每个阶段留出回滚与验证点。

### 8.1 环境与依赖
- 建议新建独立 Conda 或 venv 环境（例如 `conda create -n lerobot_v034 python=3.10`），避免与原 ACT 环境的包互相污染。
- 进入新环境后，在 `lerobot-20251011` 目录下安装：
  ```bash
  pip install -e .[smolvla]
  ```
  若本地 CUDA 版本与 PyTorch 默认 wheel 不兼容，可先安装指定版本的 `torch/torchvision/torchaudio`，再执行上面命令。
- 若需要 WandB、TensorBoard 等监控，可提前在该环境中配置好登陆信息。

### 8.2 数据集转换与验证
1. **复制数据以防误操作**：
   ```bash
   cp -r /home/chenqingyu/robot/new_lerobot/grasp_dataset \
         /home/chenqingyu/robot/new_lerobot/grasp_dataset_v21_backup
   ```
2. **本地化转换脚本**：基于 `lerobot-20251011/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` 新建脚本（例如 `scripts/local_convert_v21_to_v30.py`），需做的关键改动：
   - 移除 `snapshot_download` 与 `HfApi` 相关逻辑，仅处理本地目录。
   - 将输出目录指向 `/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30`。
   - 保留数据合并、统计写入、视频转储等核心逻辑。
3. **执行转换**：
   ```bash
   python scripts/local_convert_v21_to_v30.py \
       --input-dir /home/chenqingyu/robot/new_lerobot/grasp_dataset \
       --output-dir /home/chenqingyu/robot/new_lerobot/grasp_dataset_v30
   ```
4. **验证关键项**：
   - 检查 `meta/info.json` 中 `codebase_version` 是否为 `"v3.0"`，`data_path`/`video_path` 是否替换为新的 `file_{index}` 模式。
   - 对比 `meta/episodes/*` 中 episode 数是否与原始一致，随机抽取若干帧确认视频帧可读。
   - 使用新版工具抽样：
     ```bash
     HF_DATASETS_CACHE=/tmp/hf_cache_v30 \
     python -m lerobot.scripts.lerobot_dataset_viz \
         --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
         --dataset.repo_id=None \
         --episode-index=0
     ```
   - 若出现统计缺失，可利用转换脚本生成的日志交叉排查；必要时回滚到备份目录重新执行。

### 8.3 训练配置与脚本
1. **准备训练脚本**（例如 `koch_smolvla_train.sh`）：
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   export PYTHONPATH="/home/chenqingyu/robot/new_lerobot/lerobot-20251011:${PYTHONPATH:-}"
   export HF_DATASETS_CACHE=/home/chenqingyu/robot/.cache/hf_v30

   lerobot-train \
     --policy.path=lerobot/smolvla_base \
     --policy.device=cuda \
     --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
     --dataset.repo_id=None \
     --batch_size=16 \
     --num_workers=8 \
     --steps=200000 \
     --save_freq=5000 \
     --eval_freq=0 \
     --output_dir=/home/chenqingyu/robot/outputs/smolvla_grasp \
     --job_name=smolvla_grasp \
     --wandb.enable=false
   ```
   - 若需要使用本地初始权重，可将 `--policy.path` 指向本地目录。
   - `batch_size`、`steps` 需结合显存与数据量调整；初次运行可先设为 `--steps=200 --save_freq=200` 做冒烟测试。
2. **超参与特性关注点**：
   - `SmolVLAPolicy` 支持微调时冻结视觉编码器（默认开启），若想解冻可在配置覆写 `--policy.freeze_vision_encoder=false`。
   - 数据集若无语言描述，可在训练命令中通过 `--dataset.single_task="..."` 补充固定指令，或在转换脚本中写入合适的任务描述。
   - 可考虑使用梯度累积（`--optimizer.gradient_accumulation_steps`）缓解显存压力。

### 8.4 训练前检查清单
- 确认 GPU 驱动、CUDA 版本与 PyTorch 匹配，执行 `python -c "import torch; print(torch.cuda.is_available())"` 应返回 `True`。
- 验证 `grasp_dataset_v30` 的磁盘占用，确保输出目录所在磁盘留有至少 2 倍训练期间的缓存空间。
- 若启用 WandB，提前设置 `WANDB_API_KEY` 并根据网络限制决定 `--wandb.mode=offline/disabled`。

### 8.5 监控与调试建议
- **日志**：`lerobot-train` 默认输出到控制台，可在脚本中追加 `2>&1 | tee logs/smolvla_train_$(date +%Y%m%d_%H%M%S).log` 保留记录。
- **显存监控**：使用 `watch -n 1 nvidia-smi` 或 `gpustat`，评估是否需要调整 `batch_size`、`chunk_size` 或 `num_steps`。
- **中断恢复**：如需断点续训，保留 `--output_dir`，并补充 `--resume=true --config_path=<output_dir>/train_config.json`。

### 8.6 验证与回归对比
- 训练完成后，可在同一环境中运行 `lerobot-eval` 或自定义推理脚本与旧版 ACT 模型对照，确认性能与行为差异。
- 若需要与旧框架保持一致，可在 `new_lerobot` 中编写桥接脚本读取 v3.0 数据（利用 `lerobot-20251011` 的数据加载模块并封装成旧接口）。

### 8.7 补充考虑
- **数据更新策略**：后续如继续采集新数据，优先直接录制为 v3.0 格式；若仍通过旧流程录制，尽量将转换脚本自动化（例如写入 `Makefile` 或 CI 任务）。
- **资源隔离**：为避免误用旧框架，建议在 shell 中定义两个别名：`act-env`、`smolvla-env`，分别激活不同环境并设置对应的 `PYTHONPATH`。
- **安全回滚**：保留 `grasp_dataset_v21_backup` 至少一个训练周期，确认新流程稳定后再决定是否删除备份以释放空间。
- **潜在扩展**：完成 SmolVLA 流水线后，可评估利用同一数据集触发 PI0 或 Diffusion 等策略的复训，以充分发挥新版框架的统一接口。

以上步骤旨在覆盖从数据转换到训练监控的完整流程，确保即使未提前考虑到的数据兼容、依赖冲突等因素，也能通过预案快速处理。

## 9. 数据结构映射与特性校验（v2.1 → v3.0）
- 路径与文件格式
  - v2.1：
    - 数据：`data/chunk-000/episode_000000.parquet`
    - 视频：`videos/chunk-000/<camera_key>/episode_000000.mp4`
    - 元数据：`meta/info.json`、`meta/episodes.jsonl`、`meta/tasks.jsonl`、`meta/episodes_stats.jsonl`
  - v3.0：
    - 数据：`data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet`
    - 视频：`videos/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4`
    - 元数据：`meta/episodes/chunk-XXX/episodes_YYY.parquet`、`meta/tasks/chunk-XXX/file_YYY.parquet`、`meta/episodes_stats/chunk-XXX/file_YYY.parquet`、更新后的 `meta/info.json`
- 关键字段
  - `codebase_version` 必须为 `v3.0`（否则新加载器直接报错）。
  - `data_path`/`video_path` 模板从 episode_* 变为 file_*，并在 episodes parquet 中记录 chunk/file 映射。
  - `fps` 写入每个非视频特征的 `stats`，视频帧率保留在视频 `info` 中。
- 特征对齐（以当前数据为例）
  - 视觉：`observation.images.laptop`、`observation.images.phone`（3×480×640）→ SmolVLA 内部按 `resize_imgs_with_padding=(512,512)` 统一缩放并 pad。
  - 状态：`observation.state`（6 维）→ 由 SmolVLA 以 `max_state_dim=32` 右侧零填充。
  - 动作：`action`（6 维）→ 以 `max_action_dim=32` 右侧零填充。
  - 语言：通过 `task_index` 在 `meta/tasks` 映射为字符串（`__getitem__` 已组装成 `task`）；若缺失可在训练时注入 `--dataset.single_task`。

> 自检脚本建议
> - 快速列出特征键：`python -c 'from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset(repo_id="grasp_dataset", root="/.../grasp_dataset_v30", episodes=[0]); print(ds.meta.features.keys())'`
> - 抽样一条：`python - <<PY\nfrom lerobot.datasets.lerobot_dataset import LeRobotDataset\nds=LeRobotDataset("grasp_dataset", root="/.../grasp_dataset_v30", episodes=[0])\nitem=ds[0]; print(item.keys())\nPY`

## 10. 训练配置详解与推荐配方
- 基础命令（本地不推 Hub）
  ```bash
  lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
    --dataset.repo_id=None \
    --batch_size=16 \
    --num_workers=8 \
    --steps=20000 \
    --save_freq=5000 \
    --eval_freq=0 \
    --output_dir=outputs/train/smolvla_grasp \
    --job_name=smolvla_grasp \
    --wandb.enable=false
  ```
  - 重要：新版框架若 `--policy.push_to_hub` 为默认 true 且未提供 `--policy.repo_id` 会报错。纯本地训练请显式加 `--policy.push_to_hub=false`。
- 轻量化配方（≤12GB 显存）
  - 降低图像尺寸与序列长度：
    ```bash
    --policy.resize_imgs_with_padding="(384,384)" \
    --policy.chunk_size=10 --policy.n_action_steps=10 \
    --batch_size=8 --policy.use_amp=true
    ```
  - 可选：减少 VLM 与 Expert 深度（需权衡效果）
    ```bash
    --policy.num_vlm_layers=12 \
    --policy.num_expert_layers=6 \
    --policy.expert_width_multiplier=0.5
    ```
- 标准配方（≥24GB 显存）
  - 保持默认 512×512 与 `chunk_size=50`，起步 `batch_size=16~32`，按负载微调。
- 数据管线优化
  - 本地大数据且磁盘/IO 成为瓶颈时尝试：`--dataset.streaming=true`（将使用 StreamingLeRobotDataset）。
  - 图像增强：如非必要，避免强随机裁剪/翻转，推荐仅使用 `resize/pad` 保持时空结构稳定。
- 学习率与调度（SmolVLA 预设）
  - `optimizer_lr=1e-4`、CosineWarmup（`scheduler_warmup_steps=1000`、`scheduler_decay_steps=30000`、`decay_lr=2.5e-6`）。
  - 可通过 CLI 覆写：
    ```bash
    --policy.optimizer_lr=5e-5 \
    --policy.scheduler_warmup_steps=2000 \
    --policy.scheduler_decay_steps=60000
    ```
- 语言输入策略
  - 若数据集包含 task 文本，直接使用；若没有，可统一注入：
    ```bash
    --dataset.single_task="抓取工作台上的物体并放入收纳盒。"
    ```

## 11. 本地转换脚本模板（示意）
> 若你希望完全离线转换，可基于官方脚本制作精简本地版：
```python
# scripts/local_convert_v21_to_v30.py
import argparse, shutil
from pathlib import Path
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_info, convert_tasks, convert_data, convert_videos, convert_episodes_metadata
)

def main(in_dir: str, out_dir: str):
    root = Path(in_dir)
    new_root = Path(out_dir)
    if new_root.exists():
        shutil.rmtree(new_root)
    new_root.mkdir(parents=True, exist_ok=True)

    # 估计目标分片大小（可按磁盘情况调大/调小）
    data_mb, video_mb = 100, 500

    convert_info(root, new_root, data_mb, video_mb)
    convert_tasks(root, new_root)
    episodes_meta = convert_data(root, new_root, data_mb)
    episodes_videos_meta = convert_videos(root, new_root, video_mb)
    convert_episodes_metadata(root, new_root, episodes_meta, episodes_videos_meta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    main(args.input_dir, args.output_dir)
```
- 若遇到依赖 `HfApi`/`snapshot_download` 报错，确认你导入的是 `src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` 的函数，而不是直接调用其 `convert_dataset`（该函数会访问 Hub）。
- 处理大数据时，`data_mb`/`video_mb` 可调节，较小的分片便于断点续传与校验。

## 12. 常见问题与排错
- BackwardCompatibilityError（提示 v2.1 与 v3.0 不兼容）
  - 说明仍在使用旧版数据结构；按第 8/11 节转换即可。
- 训练启动即报错 `policy.repo_id` 缺失
  - 加 `--policy.push_to_hub=false`（本地训练）或设置 `--policy.repo_id=<your_user/smolvla_grasp>`（推 Hub）。
- 模型/分词器下载失败
  - `--policy.path=lerobot/smolvla_base` 需要联网；可先在联网环境下载到本地，再改为本地目录。
- `safetensors` 版本相关告警
  - 确认 `safetensors>=0.4.3`，否则权重加载可能会先落盘 CPU 再复制到 GPU，导致变慢。
- CUDA OOM
  - 降低 `batch_size`、`chunk_size`、图像分辨率（见第 10 节轻量化配方），或开启 `--policy.use_amp=true`。
- HF datasets 缓存权限报错
  - 设置 `HF_DATASETS_CACHE` 指向你有写权限的目录（示例见第 8.2/8.3 节）。

## 13. 性能与资源优化建议
- AMP：`--policy.use_amp=true` 一般对推理与训练均有效果，注意数值稳定性。
- 层数裁剪：`--policy.num_vlm_layers`/`--policy.num_expert_layers` 在保证表达力的前提下降低显存与算力占用。
- Expert 宽度：`--policy.expert_width_multiplier`（如 0.5~0.75）减少 MLP 尺寸。
- 图像尺寸：`--policy.resize_imgs_with_padding=(384,384)` 可显著减负，但需验证对表现的影响。
- 数据流式：`--dataset.streaming=true` 与 `--num_workers` 配合，缓解大数据 IO 压力。

## 14. 与 ACT 并行与迁移
- 并行阶段：保留旧环境运行 ACT，SmolVLA 使用新环境，脚本与 `PYTHONPATH` 严格区分。
- 迁移阶段：当 SmolVLA 稳定后，将 ACT 训练也迁移到新框架（同一 `lerobot-train` 入口，不再维持双仓）。
- 统一数据：新数据采集优先产出 v3.0，若出现 v2.1 数据，纳入第 11 节脚本的自动化流程。

—— 以上为更细致的落地方案与配方，覆盖从数据转换、训练配置、性能优化到排错回滚的关键环节。
