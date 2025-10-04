#!/usr/bin/env python3
"""
Koch机械臂快速推理脚本
简化的推理脚本，用于快速测试模型

使用方法:
    python quick_inference.py act
    python quick_inference.py diffusion
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import logging

from koch_inference import KochInference

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 预定义的模型路径
MODEL_PATHS = {
    "act": "/mnt/data/cqy/workspace/lerobot-20251003/output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model",
    "diffusion": "/mnt/data/cqy/workspace/lerobot-20251003/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model"
}

def main():
    parser = argparse.ArgumentParser(description="Koch机械臂快速推理")
    parser.add_argument("policy_type", choices=["act", "diffusion"],
                       help="策略类型")
    parser.add_argument("--duration", type=float, default=60,
                       help="推理持续时间（秒）")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyUSB0",
                       help="机械臂端口")

    args = parser.parse_args()

    # 获取模型路径
    model_path = MODEL_PATHS[args.policy_type]

    print(f"🤖 启动 {args.policy_type.upper()} 推理...")
    print(f"📁 模型路径: {model_path}")
    print(f"⏱️  持续时间: {args.duration}秒")
    print(f"🔌 机械臂端口: {args.robot_port}")
    print("=" * 50)

    # 创建并运行推理
    try:
        inference = KochInference(
            policy_type=args.policy_type,
            model_path=model_path,
            robot_port=args.robot_port,
            fps=30,
            max_relative_target=10.0,
            device="auto"
        )

        inference.run_inference(
            duration=args.duration,
            save_data=False,
            task="Grasp and place object"
        )

    except Exception as e:
        logger.error(f"推理失败: {e}")
        print("\n❌ 推理失败，请检查:")
        print("1. 机械臂连接是否正常")
        print("2. 相机设备是否可用")
        print("3. 模型文件是否存在")
        print("4. conda环境是否正确激活")

if __name__ == "__main__":
    main()