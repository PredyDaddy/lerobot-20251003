# 创建日志目录（如果不存在）
mkdir -p logs

# 注释掉离线模式，因为DINOv2需要下载模型
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
export PYTHONWARNINGS="ignore::UserWarning:torchvision.io._video_deprecation_warning"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 删除旧的输出目录
# rm -rf outputs/train/grasp



# 训练命令检查: 
# 1. 数据集是否选择对了 --dataset.root
# 2. 输出路径是否正确   --output_dir


# 运行训练脚本，同时将所有输出（包括标准输出和错误输出）通过tee写入日志文件
python lerobot/scripts/train.py \
    --policy.type=act_dinov2 \
    --policy.device=cuda \
    --batch_size=24 \
    --num_workers=8 \
    --save_freq=1000 \
    --eval_freq=0 \
    --steps=5700000 \
    --dataset.root=/home/chenqingyu/robot/new_lerobot/grasp_dataset \
    --dataset.repo_id=None \
    --output_dir=outputs/train/grasp_dinov2 \
    --job_name=grasp \
    --wandb.enable=false \
    2>&1 | tee logs/grasp_dinov2_train_$(date +%Y%m%d_%H%M%S).log