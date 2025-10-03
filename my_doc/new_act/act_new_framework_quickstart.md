# 新框架（lerobot-20251011）ACT 训练与异步推理落地指南

本文档面向当前环境：`/home/chenqingyu/robot/new_lerobot`，目标是在新框架 `lerobot-20251011` 中完成 ACT 的端到端验证（数据加载 → 训练 → 异步推理）。重点覆盖版本差异、关键参数、常见坑点与边界情况。

---

## 1. 总览与推荐路径
- 起步建议：先在新框架上训练 ACT，再上线 SmolVLA。
  - 依赖更轻、贴近你现有流程，能更快验证新框架训练/推理闭环。
  - 成功后你对新框架的配置、日志、产物结构有直观把握，SmolVLA 落地成本会更低。
- 新特性：支持异步推理（policy server + robot client），便于在 GPU 侧运行策略、机器人侧只做采集与执行，降低端侧算力要求。

---

## 2. 版本与环境要求
- Python ≥ 3.10，CUDA 对应的 PyTorch 版本（A 卡/Mac 则对应 MPS/CPU）
- 新框架安装：进入 `lerobot-20251011` 目录，安装为可编辑包
  ```bash
  pip install -e /home/chenqingyu/robot/new_lerobot/lerobot-20251011
  ```
- ACT 不需要安装 `.[smolvla]` 额外依赖（SmolVLA 再安装）。
- 建议隔离虚拟环境，避免与旧仓库依赖互相污染。

> 建议设置 `HF_DATASETS_CACHE=/path/with/permission`，避免默认缓存目录写权限问题。

---

## 3. 数据集兼容性与转换（v2.1 → v3.0）
你的现有数据集位于：`/home/chenqingyu/robot/new_lerobot/grasp_dataset`，其 `meta/info.json` 为 `codebase_version: v2.1`。新框架要求 `v3.0`，因此需要转换：

- 结构变化（核心区别）
  - v2.1：`data/chunk-000/episode_000000.parquet`；`videos/chunk-000/<camera>/episode_*.mp4`；`meta/*.jsonl`
  - v3.0：`data/chunk-XXX/file_YYY.parquet`；`videos/chunk-XXX/file_YYY.mp4`；`meta/episodes/*.parquet`、`meta/tasks/*.parquet`、`meta/episodes_stats/*.parquet`，并更新 `meta/info.json`。
- 转换步骤（本地离线）
  - 复制备份：避免误操作
    ```bash
    cp -r /home/chenqingyu/robot/new_lerobot/grasp_dataset \
          /home/chenqingyu/robot/new_lerobot/grasp_dataset_v21_backup
    ```
  - 运行本地转换（已提供辅助脚本）
    ```bash
    python scripts/local_convert_v21_to_v30.py \
      --input-dir /home/chenqingyu/robot/new_lerobot/grasp_dataset \
      --output-dir /home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
      --data-mb 100 --video-mb 500
    ```
- 转换后校验清单
  - `meta/info.json` 的 `codebase_version` 是否为 `"v3.0"`；`data_path`/`video_path` 是否为 `file_*` 模式。
  - `meta/episodes/*.parquet`、`meta/tasks/*.parquet`、`meta/episodes_stats/*.parquet` 是否存在。
  - 随机抽查 1 个 episode，确认 parquet、视频可读（可用 `lerobot.scripts.lerobot_dataset_viz` 或自写小脚本读取）。
- 边界情况与对策
  - 数据量很大：适度降低 `--data-mb/--video-mb`，减小单文件尺寸，利于断点续处理与调试。
  - JSONL → parquet 过程耗内存：优先在本地 NVMe 上操作；必要时分批转换或在高内存机器上执行。
  - 统计缺失或异常：转换会重算 `episodes_stats` 并聚合成 `stats`，如发现缺值，回到备份重跑，定位具体 episode 文件。

> 注意：新框架严格校验 `codebase_version`。未转换将直接抛 `BackwardCompatibilityError`。

---

## 4. 训练配置详解（ACT）
- 启动入口：`lerobot-train`（若未注册则使用 `python -m lerobot.scripts.lerobot_train`）
- 必填参数
  - `--policy.type=act`：指定策略为 ACT。
  - `--dataset.root=/abs/path/to/grasp_dataset_v30`：本地数据根目录。
  - `--dataset.repo_id=grasp_dataset`：必须为非空字符串。新框架不接受 `None`。
  - `--policy.push_to_hub=false`：本地训练不推 Hub；若为 true 则需提供 `--policy.repo_id`。
- 常用训练参数
  - `--batch_size`：建议从 8 起步冒烟，到 16/24 视显存调节。
  - `--num_workers`：8 起步，视 IO 调整；若 CPU/IO 忙，可尝试 `--dataset.streaming=true`。
  - `--steps`/`--save_freq`/`--eval_freq`：冒烟先 200 步快速确认，正式再拉长。
  - `--output_dir`/`--job_name`：输出目录与命名。
- 可选参数（资源优化）
  - `--policy.device=cuda|cpu|mps`：选择加速设备。
  - `--policy.use_amp=true`：自动混合精度，减少显存占用（视稳定性启用）。
- 输入特征与相机命名
  - 新框架会根据数据集元信息自动推断 `input_features`。你当前相机键：`observation.images.laptop`、`observation.images.phone`。
  - 训练无需手工指定图像键，但推理时相机命名必须与训练时一致（见第 6 节）。

### 4.1 冒烟与正式命令示例
- 冒烟（验证能训能存）：
  ```bash
  lerobot-train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
    --dataset.repo_id=grasp_dataset \
    --batch_size=8 \
    --num_workers=8 \
    --steps=200 \
    --save_freq=200 \
    --eval_freq=0 \
    --output_dir=/home/chenqingyu/robot/outputs/act_grasp_v3_smoke \
    --job_name=act_grasp_v3_smoke \
    --wandb.enable=false
  ```
- 正式训练：
  ```bash
  lerobot-train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset_v30 \
    --dataset.repo_id=grasp_dataset \
    --batch_size=24 \
    --num_workers=8 \
    --steps=100000 \
    --save_freq=5000 \
    --eval_freq=0 \
    --output_dir=/home/chenqingyu/robot/outputs/act_grasp_v3 \
    --job_name=act_grasp_v3 \
    --wandb.enable=false
  ```

### 4.2 产物结构与断点续训
- 模型、配置与处理器会存放在 `--output_dir` 内的 `checkpoints` 路径下，`last` 指向最新检查点。
- 推理使用目录：`<output_dir>/checkpoints/last/pretrained_model`
- 断点续训：
  ```bash
  lerobot-train ... \
    --resume=true \
    --config_path=/home/chenqingyu/robot/outputs/act_grasp_v3/train_config.json
  ```

---

## 5. 性能与资源优化建议
- 显存不够：
  - 降低 `batch_size`；启用 `--policy.use_amp=true`；仅保留 1 路相机先验收流程。
- IO 瓶颈：
  - 提高 `--num_workers`；尝试 `--dataset.streaming=true`；将 `HF_DATASETS_CACHE` 指向快速磁盘且有写权限。
- 训练不稳定：
  - 暂时关闭 AMP；核查数据统计（`meta/episodes_stats`）；检查是否存在空 episode 或损坏视频。

---

## 6. 异步推理（新特性）
异步推理分为两端：GPU 机器运行 `policy_server`，机器人侧运行 `robot_client`。

### 6.1 策略服务端（GPU）
```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8080 \
  --fps=30 --inference_latency=0.02 --obs_queue_timeout=1
```
- server 会在接收 client 下发的策略说明后，加载模型至指定设备。

### 6.2 机器人客户端（Koch 示例）
```bash
python -m lerobot.async_inference.robot_client \
  --robot.type=koch_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, phone: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \
  --task="抓取工作台上的物体并放入收纳盒。" \
  --server_address=127.0.0.1:8080 \
  --policy_type=act \
  --pretrained_name_or_path=/home/chenqingyu/robot/outputs/act_grasp_v3/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --actions_per_chunk=100 \
  --chunk_size_threshold=0.5 \
  --fps=30
```
- 相机键名必须与训练时一致：你的数据集为 `laptop`、`phone`，否则会被拒绝。
- `actions_per_chunk` 与 `chunk_size_threshold` 影响延迟/稳定性：减小 chunk 可降低滞后，但需要更频繁的网络往返。
- 与服务器跨机器部署时，请确保端口开放、延迟可接受。

### 6.3 常见问题
- 客户端提示相机不匹配：对齐相机键名；若只能单相机推理，需用对应单相机训练或在策略期望的图像键上补空（SmolVLA 支持空相机占位，但 ACT 不建议）。
- 延迟导致抖动：降低 `actions_per_chunk`、提高服务器性能或靠近部署；检查网络质量。
- FPS 不一致：客户端 `--fps` 与 server `--fps` 尽量一致；机器人侧采集频率过高/过低都可能影响动作时序。

---

## 7. 常见坑点与边界情况
- `--dataset.repo_id` 不能为空：新框架要求字符串；建议使用 `grasp_dataset`。
- 本地训练别忘了 `--policy.push_to_hub=false`，否则会要求 `--policy.repo_id`。
- `safetensors` 版本：建议 ≥ 0.4.3，避免 CPU 落盘再拷贝到 GPU 的低效加载路径。
- HF 缓存权限：若默认 `~/.cache/huggingface/datasets` 无写权限，设置 `HF_DATASETS_CACHE`。
- 多相机 → 单相机推理：不建议在 ACT 上混用。应保持训练/推理相机集合一致；或重新训练仅单相机的策略。
- 数据损坏/缺帧：转换失败时定位具体 episode 重新生成；必要时从备份恢复再转。
- 断点续训：使用 `--resume=true --config_path=<output_dir>/train_config.json`，确保配置文件存在且路径正确。
- 旧脚本对照：旧仓库常用 `--dataset.repo_id=None`，新框架不可；须改为字符串。

---

## 8. 建议的推进方式
1. 将 `grasp_dataset` 转换到 `grasp_dataset_v30` 并完成校验。
2. 跑一轮 200 步冒烟，确认损失下降、检查点写出、设备与缓存配置正常。
3. 启动 policy server 与 robot client，做一次异步推理“干跑”或低速实机测试；确认动作合理、延迟可控。
4. 拉长训练步数、调参优化；完成后再评估 SmolVLA 方案。

如需，我可以将上述命令整合进你的现有文档体系（例如 `smolvla/new_act/new_act_glm4.6.md`）或提供一个更详细的参数对照表，便于从 `koch_train.sh` 平滑迁移到 `lerobot-train`。
