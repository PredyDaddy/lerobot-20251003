#!/usr/bin/env python3
"""
Koch机械臂异步推理示例
使用LeRobot的async inference框架进行分布式推理

使用方法:
    # 终端1: 启动策略服务器
    python async_inference_example.py --mode server --policy act

    # 终端2: 启动机器人客户端
    python async_inference_example.py --mode client --policy act
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型路径
MODEL_PATHS = {
    "act": "/mnt/data/cqy/workspace/lerobot-20251003/output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model",
    "diffusion": "/mnt/data/cqy/workspace/lerobot-20251003/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model"
}

def start_policy_server(policy_type: str, model_path: str, host: str = "localhost", port: int = 8080):
    """启动策略服务器"""
    logger.info(f"启动{policy_type.upper()}策略服务器...")

    cmd = [
        "python", str(PROJECT_ROOT / "src/lerobot/async_inference/policy_server.py"),
        "--host", host,
        "--port", str(port),
        "--policy_type", policy_type,
        "--pretrained_name_or_path", model_path,
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--fps", "30"
    ]

    logger.info(f"执行命令: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        logger.info(f"✓ 策略服务器已启动 (PID: {process.pid})")
        logger.info(f"🌐 服务器地址: {host}:{port}")

        # 等待服务器启动
        time.sleep(3)

        return process
    except Exception as e:
        logger.error(f"启动策略服务器失败: {e}")
        return None

def start_robot_client(policy_type: str, model_path: str, server_address: str = "localhost:8080"):
    """启动机器人客户端"""
    logger.info(f"启动机器人客户端...")

    # 相机配置JSON
    cameras_config = json.dumps({
        "laptop": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "phone": {
            "type": "opencv",
            "index_or_path": 2,
            "width": 640,
            "height": 480,
            "fps": 30
        }
    })

    cmd = [
        "python", str(PROJECT_ROOT / "src/lerobot/async_inference/robot_client.py"),
        "--robot.type=koch_follower",
        "--robot.port=/dev/ttyUSB0",
        f"--robot.cameras={cameras_config}",
        "--robot.max_relative_target=10.0",
        f"--policy_type={policy_type}",
        f"--pretrained_name_or_path={model_path}",
        f"--server_address={server_address}",
        "--policy_device=cuda",
        "--actions_per_chunk=50",
        "--fps=30",
        "--task=Grasp and place object",
        "--debug_visualize_queue_size=true"
    ]

    logger.info(f"执行命令: {' '.join(cmd[:5])} ...")  # 只显示前几个参数

    try:
        process = subprocess.Popen(cmd)
        logger.info(f"✓ 机器人客户端已启动 (PID: {process.pid})")

        return process
    except Exception as e:
        logger.error(f"启动机器人客户端失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Koch机械臂异步推理示例")
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                       help="运行模式")
    parser.add_argument("--policy", choices=["act", "diffusion"], required=True,
                       help="策略类型")
    parser.add_argument("--host", type=str, default="localhost",
                       help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8080,
                       help="服务器端口")

    args = parser.parse_args()

    # 获取模型路径
    model_path = MODEL_PATHS[args.policy]

    if not Path(model_path).exists():
        logger.error(f"模型路径不存在: {model_path}")
        sys.exit(1)

    print(f"🚀 启动异步推理示例")
    print(f"📦 策略类型: {args.policy.upper()}")
    print(f"🎯 运行模式: {args.mode}")
    print(f"📁 模型路径: {model_path}")
    print("=" * 50)

    try:
        import torch
        if args.mode == "server":
            process = start_policy_server(args.policy, model_path, args.host, args.port)
            if process:
                try:
                    process.wait()
                except KeyboardInterrupt:
                    logger.info("停止策略服务器...")
                    process.terminate()
                    process.wait()
        else:  # client
            server_address = f"{args.host}:{args.port}"
            process = start_robot_client(args.policy, model_path, server_address)
            if process:
                try:
                    process.wait()
                except KeyboardInterrupt:
                    logger.info("停止机器人客户端...")
                    process.terminate()
                    process.wait()

    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请确保已安装 torch: pip install torch")
        sys.exit(1)

if __name__ == "__main__":
    main()