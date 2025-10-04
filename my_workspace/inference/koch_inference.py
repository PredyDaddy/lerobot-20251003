#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time inference on Koch follower robot with a trained LeRobot policy (ACT or Diffusion).

- Loads a pretrained policy from a local checkpoint folder (must contain config.json, model.safetensors,
  and pre/post-processor configs saved by LeRobot training).
- Connects to Koch follower via Dynamixel bus and configured cameras.
- Runs a control loop at target FPS: read observation -> preprocess -> policy.select_action -> postprocess -> send action.

Example:
  python my_workspace/inference/koch_inference.py \
    --policy.path /mnt/data/.../output/diffusion_run/checkpoints/005000/pretrained_model \
    --device cuda --use_amp true --fps 20 \
    --robot.port /dev/ttyUSB0 \
    --robot.cameras '{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --robot.max_relative_target 4.0

Notes:
- Action names/order are taken from the robot's action_features; this matches datasets recorded with this robot.
- Pre/Post processors are loaded from the policy folder to ensure proper normalization and history buffering.
"""

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.control_utils import predict_action
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.koch_follower.config_koch_follower import KochFollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.utils.robot_utils import busy_wait


def parse_cameras(cameras_json: str) -> Dict[str, Any]:
    """Parse cameras JSON into CameraConfig instances per key.

    Input JSON example:
    {
      "front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
      "wrist": {"type": "intelrealsense", "serial_number_or_name": "0123456789", "width": 640, "height": 480, "fps": 30}
    }
    """
    if not cameras_json:
        return {}
    data = json.loads(cameras_json)
    out: Dict[str, Any] = {}
    for name, cfg in data.items():
        t = cfg.get("type")
        if t == "opencv":
            out[name] = OpenCVCameraConfig(
                index_or_path=cfg["index_or_path"],
                fps=cfg["fps"], width=cfg["width"], height=cfg["height"], color_mode=ColorMode.RGB
            )
        elif t in ("intelrealsense", "realsense"):
            out[name] = RealSenseCameraConfig(
                serial_number_or_name=cfg["serial_number_or_name"],
                fps=cfg.get("fps"), width=cfg.get("width"), height=cfg.get("height"), color_mode=ColorMode.RGB
            )
        else:
            raise ValueError(f"Unsupported camera type: {t} for camera {name}")
    return out


def build_robot(args) -> Any:
    cameras = parse_cameras(args.robot_cameras) if args.robot_cameras else {}
    robot_cfg = KochFollowerConfig(
        port=args.robot_port,
        cameras=cameras,
        id=args.robot_id,
        calibration_dir=Path(args.robot_calibration_dir) if args.robot_calibration_dir else None,
        max_relative_target=args.robot_max_relative_target,
        disable_torque_on_disconnect=not args.keep_torque_on_disconnect,
        use_degrees=args.use_degrees,
    )
    robot = make_robot_from_config(robot_cfg)
    return robot


def load_policy_and_processors(policy_path: str, device_str: str | None, use_amp: bool):
    # Load config from checkpoint
    cfg: PreTrainedConfig = PreTrainedConfig.from_pretrained(policy_path)
    if device_str:
        cfg.device = get_safe_torch_device(device_str).type
    else:
        # Ensure device is valid
        cfg.device = get_safe_torch_device(cfg.device).type
    cfg.use_amp = bool(use_amp)

    # Instantiate policy class directly from checkpoint to avoid needing ds_meta/env_cfg
    PolicyCls = get_policy_class(cfg.type)
    policy = PolicyCls.from_pretrained(policy_path, config=cfg)

    # Load processors from the same folder, override device inside pipelines
    preproc, postproc = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={},
    )

    # Reset internal buffers for clean start
    policy.reset()
    preproc.reset()
    postproc.reset()

    return policy, preproc, postproc, cfg


def map_action_to_robot_dict(action_tensor: torch.Tensor, robot) -> Dict[str, float]:
    # Use robot.action_features order (e.g., 'shoulder_pan.pos', ...)
    keys = list(robot.action_features.keys())
    if action_tensor.ndim != 1:
        action_tensor = action_tensor.view(-1)
    if len(keys) != action_tensor.numel():
        raise ValueError(
            f"Action dim mismatch: robot expects {len(keys)} dims, policy produced {action_tensor.numel()}"
        )
    return {keys[i]: float(action_tensor[i].item()) for i in range(len(keys))}


def run_loop(robot, policy, preproc, postproc, policy_cfg: PreTrainedConfig, fps: int, task: str | None):
    device = get_safe_torch_device(policy_cfg.device)
    use_amp = bool(policy_cfg.use_amp)

    robot.connect()

    stop_flag = {"stop": False}

    def _sigint_handler(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop_flag["stop"]:
            loop_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            loop_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if loop_start is not None:
                loop_start.record()

            obs = robot.get_observation()  # dict[str, Any]

            # predict_action handles: torch.inference_mode + autocast + preprocessing + postprocessing
            action_tensor = predict_action(
                observation=obs,
                policy=policy,
                device=device,
                preprocessor=preproc,
                postprocessor=postproc,
                use_amp=use_amp,
                task=task,
                robot_type=robot.name,
            )

            robot_action = map_action_to_robot_dict(action_tensor, robot)
            _ = robot.send_action(robot_action)

            if loop_start is not None:
                loop_end.record()
                torch.cuda.synchronize()
                dt_ms = loop_start.elapsed_time(loop_end)
                print(f"inference: {dt_ms:.1f} ms ({1000.0/max(dt_ms,1e-6):.0f} Hz)")

            # pace loop
            busy_wait(1.0 / max(fps, 1))
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            logging.warning(f"Error while disconnecting robot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Koch follower real-time inference with a LeRobot policy")
    parser.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu|mps (optional, auto if omitted)")
    parser.add_argument("--use_amp", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--task", type=str, default=None, help="Optional task string passed to processor")

    # Robot config
    parser.add_argument("--robot.port", dest="robot_port", type=str, required=True)
    parser.add_argument("--robot.cameras", dest="robot_cameras", type=str, default=None,
                        help="JSON dict mapping camera name -> config")
    parser.add_argument("--robot.id", dest="robot_id", type=str, default=None)
    parser.add_argument("--robot.calibration_dir", dest="robot_calibration_dir", type=str, default=None)
    parser.add_argument("--robot.max_relative_target", dest="robot_max_relative_target", type=float, default=None)
    parser.add_argument("--robot.keep_torque_on_disconnect", dest="keep_torque_on_disconnect",
                        action="store_true", help="Do NOT disable torque on disconnect")
    parser.add_argument("--robot.use_degrees", dest="use_degrees", action="store_true",
                        help="Use degrees normalization for joints (default is [-100,100] range)")

    args = parser.parse_args()
    init_logging()

    # Sanity checks for policy folder
    p = Path(args.policy_path)
    if not p.exists():
        raise FileNotFoundError(f"Policy path not found: {p}")

    robot = build_robot(args)
    policy, preproc, postproc, policy_cfg = load_policy_and_processors(args.policy_path, args.device, args.use_amp)

    print(f"Loaded policy type={policy_cfg.type} device={policy_cfg.device} amp={policy_cfg.use_amp}")
    print(f"Robot action keys: {list(robot.action_features.keys())}")

    run_loop(robot, policy, preproc, postproc, policy_cfg, fps=args.fps, task=args.task)


if __name__ == "__main__":
    # Ensure this repository is installed (pip install -e .) or PYTHONPATH includes src/
    sys.exit(main())

