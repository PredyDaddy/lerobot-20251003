#!/usr/bin/env python3
"""
Koch机械臂硬件检查脚本
自动检查所有必要的硬件连接和配置

使用方法:
    python check_hardware.py
"""

import sys
import subprocess
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import serial.tools.list_ports

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def print_status(message, status=True):
    status_symbol = "✓" if status else "✗"
    print(f"{status_symbol} {message}")

def check_usb_ports():
    """检查USB端口"""
    print_section("USB端口检查")

    # 获取所有串口
    ports = serial.tools.list_ports.comports()

    if not ports:
        print_status("未找到任何USB串口设备", False)
        print("\n💡 建议:")
        print("1. 检查USB线是否连接")
        print("2. 检查机械臂电源是否开启")
        print("3. 尝试重新插拔USB线")
        return []

    print_status(f"找到 {len(ports)} 个USB串口设备")

    for i, port in enumerate(ports, 1):
        print(f"\n  端口 {i}:")
        print(f"    设备路径: {port.device}")
        print(f"    描述: {port.description}")
        print(f"    硬件ID: {port.hwid}")

        # 检查是否是Dynamixel设备
        if "0403:6014" in port.hwid or "USB Serial" in port.description:
            print_status(f"    -> 可能是Dynamixel设备", True)

    return [port.device for port in ports]

def check_camera_devices():
    """检查相机设备"""
    print_section("相机设备检查")

    # 检查视频设备
    try:
        result = subprocess.run(['ls', '-la', '/dev/video*'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            video_devices = [line.split()[-1] for line in result.stdout.strip().split('\n')
                           if line.strip()]
            print_status(f"找到 {len(video_devices)} 个视频设备")

            for device in video_devices:
                print(f"    {device}")
        else:
            print_status("未找到视频设备", False)
            return []
    except Exception as e:
        print_status(f"检查视频设备失败: {e}", False)
        return []

    # 测试相机可用性
    print("\n相机可用性测试:")
    working_cameras = []

    for i in range(min(4, len(video_devices))):  # 最多测试4个相机
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w, c = frame.shape
                print_status(f"相机 {i} (/dev/video{i}): {w}x{h}", True)
                working_cameras.append(i)
            else:
                print_status(f"相机 {i}: 无法读取帧", False)
            cap.release()
        else:
            print_status(f"相机 {i}: 无法打开", False)

    if not working_cameras:
        print("\n💡 建议:")
        print("1. 检查相机是否被其他程序占用")
        print("2. 尝试重新插拔USB摄像头")
        print("3. 检查相机驱动是否正确安装")

    return working_cameras

def check_permissions():
    """检查用户权限"""
    print_section("权限检查")

    import os

    # 检查dialout组（串口权限）
    groups = subprocess.run(['groups'], capture_output=True, text=True).stdout
    if 'dialout' in groups:
        print_status("用户在dialout组中（串口权限OK）", True)
    else:
        print_status("用户不在dialout组中", False)
        print("    解决方案: sudo usermod -a -G dialout $USER")
        print("    然后重新登录或重启")

    # 检查video组（相机权限）
    if 'video' in groups:
        print_status("用户在video组中（相机权限OK）", True)
    else:
        print_status("用户不在video组中", False)
        print("    解决方案: sudo usermod -a -G video $USER")
        print("    然后重新登录或重启")

def check_python_environment():
    """检查Python环境"""
    print_section("Python环境检查")

    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print_status(f"当前conda环境: {conda_env}", True)
        if conda_env != 'lerobot_v3':
            print("    ⚠️  建议使用lerobot_v3环境")
    else:
        print_status("未激活conda环境", False)
        print("    解决方案: conda activate lerobot_v3")

    # 检查关键依赖
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'lerobot': 'LeRobot',
        'dynamixel_sdk': 'Dynamixel SDK'
    }

    print("\n依赖检查:")
    for module, name in dependencies.items():
        try:
            __import__(module)
            if module == 'torch':
                import torch
                version = torch.__version__
                cuda_available = torch.cuda.is_available()
                print_status(f"{name}: {version} (CUDA: {'✓' if cuda_available else '✗'})", True)
            elif module == 'cv2':
                import cv2
                print_status(f"{name}: {cv2.__version__}", True)
            else:
                print_status(f"{name}: 已安装", True)
        except ImportError:
            print_status(f"{name}: 未安装", False)

def check_model_files():
    """检查模型文件"""
    print_section("模型文件检查")

    model_paths = {
        'ACT': "/mnt/data/cqy/workspace/lerobot-20251003/output/act_train_20251003_174818_1452/checkpoints/152000/pretrained_model",
        'Diffusion': "/mnt/data/cqy/workspace/lerobot-20251003/output/diffusion_train_20251003_222201_5883/checkpoints/108000/pretrained_model"
    }

    for name, path in model_paths.items():
        print(f"\n{name}模型:")
        path_obj = Path(path)

        if path_obj.exists():
            print_status(f"模型目录存在: {path}", True)

            # 检查必要文件
            config_file = path_obj / "config.json"
            model_file = path_obj / "model.safetensors"

            if config_file.exists():
                print_status("  config.json 存在", True)
                # 读取并显示基本信息
                try:
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    print(f"    类型: {config.get('type', 'unknown')}")
                    print(f"    设备: {config.get('device', 'unknown')}")
                except:
                    print("    ⚠️  配置文件读取失败")
            else:
                print_status("  config.json 不存在", False)

            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print_status(f"  model.safetensors 存在 ({size_mb:.1f} MB)", True)
            else:
                print_status("  model.safetensors 不存在", False)
        else:
            print_status(f"模型目录不存在: {path}", False)

def main():
    print("🤖 Koch机械臂硬件检查工具")
    print("=" * 50)

    # 执行所有检查
    usb_ports = check_usb_ports()
    cameras = check_camera_devices()
    check_permissions()
    check_python_environment()
    check_model_files()

    # 生成总结
    print_section("检查总结")

    issues = []

    if not usb_ports:
        issues.append("未找到USB串口设备")

    if len(cameras) < 2:
        issues.append(f"只找到 {len(cameras)} 个相机（需要2个）")

    # 给出建议
    if issues:
        print("❌ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\n🔧 解决方案:")
        print("1. 检查所有硬件连接")
        print("2. 确保用户权限正确配置")
        print("3. 重新插拔USB设备")
        print("4. 运行: sudo usermod -a -G dialout,video $USER")
        print("5. 重新登录或重启系统")
    else:
        print("✅ 所有硬件检查通过！")
        print("🚀 可以开始运行推理脚本了")

if __name__ == "__main__":
    main()