"""
!/usr/bin/env python3

Local LeRobot Dataset Converter: v2.1 → v3.0

用法（Usage）
  - 目的：将本地 LeRobot v2.1 数据集转换为 v3.0 结构（目录→目录，无需联网）

  - 依赖（在 lerobot_v3 环境安装）：
      pip install pandas pyarrow datasets jsonlines tqdm av
    注：视频拼接使用 PyAV（av），不依赖系统 ffmpeg 二进制。

  - 运行示例（从仓库根目录执行，或确保本脚本能找到 src/ 路径）：
    python my_workspace/scripts/local_convert_v21_to_v30.py \
    --input-dir grasp_dataset \
    --output-dir grasp_dataset_v30 \
    --data-mb 100 --video-mb 500 \
    --overwrite

  - 参数说明：
      --input-dir    v2.1 源数据集目录（包含 meta/info.json 等）
      --output-dir   v3.0 目标输出目录（建议全新目录；配合 --overwrite 可覆盖已存在目录）
      --data-mb      数据文件分片大小（MB），默认 100，对应 v3.0 建议值
      --video-mb     视频文件分片大小（MB），默认 500，对应 v3.0 建议值
      --overwrite    若目标目录已存在则先删除再生成

  - 建议流程：
      1) 备份源目录 grasp_dataset 至 grasp_dataset_backup_YYYYMMDD_HHMMSS
      2) 确保磁盘容量≥原始体量 2.5–3×（备份+新版+中间文件）
      3) 运行本脚本执行转换
      4) 按文档的验证清单逐项核对输出

"""

import argparse
import logging
import shutil
import sys
from pathlib import Path


def _setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def _ensure_import_paths() -> Path:
    """Ensure the repository's src path is importable and return repo root.

    Assumes this script resides in <repo_root>/my_workspace/scripts/.
    """
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    return repo_root


def main(argv: list[str] | None = None) -> int:
    _setup_logging(verbose=True)
    repo_root = _ensure_import_paths()

    # Lazy imports after sys.path setup
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
        convert_data,
        convert_episodes_metadata,
        convert_info,
        convert_tasks,
        convert_videos,
    )
    from lerobot.datasets.utils import (
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        load_info,
    )

    parser = argparse.ArgumentParser(
        description="Convert local LeRobot dataset from v2.1 to v3.0 (directory → directory).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Path to v2.1 dataset root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to v3.0 output root")
    parser.add_argument(
        "--data-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_IN_MB, help="Target data file size in MB"
    )
    parser.add_argument(
        "--video-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_IN_MB, help="Target video file size in MB"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists")

    args = parser.parse_args(argv)

    src = args.input_dir.resolve()
    dst = args.output_dir.resolve()

    if not src.exists() or not (src / "meta/info.json").exists():
        logging.error(f"Input directory invalid or missing meta/info.json: {src}")
        return 1

    if dst.exists():
        if not args.overwrite:
            logging.error(f"Output directory exists: {dst}. Use --overwrite to replace it.")
            return 1
        logging.warning(f"Removing existing output directory: {dst}")
        shutil.rmtree(dst)

    # Pre-check source version (warn if not v2.1)
    try:
        info = load_info(src)
        v = str(info.get("codebase_version", "")).lower()
        if not v.startswith("v2.1"):
            logging.warning(
                f"Source codebase_version is '{info.get('codebase_version')}'. Expected 'v2.1'. Proceeding anyway."
            )
    except Exception as e:
        logging.warning(f"Failed to read source info.json for version check: {e}")

    # Ensure parent exists
    dst.mkdir(parents=True, exist_ok=True)

    try:
        logging.info(f"Converting info from {src} → {dst}")
        convert_info(src, dst, args.data_mb, args.video_mb)

        logging.info("Converting tasks …")
        convert_tasks(src, dst)

        logging.info("Converting data files …")
        episodes_metadata = convert_data(src, dst, args.data_mb)

        logging.info("Converting videos …")
        episodes_videos_metadata = convert_videos(src, dst, args.video_mb)

        logging.info("Consolidating episodes metadata and stats …")
        convert_episodes_metadata(src, dst, episodes_metadata, episodes_videos_metadata)
    except Exception as exc:
        logging.exception(f"Conversion failed: {exc}")
        return 2

    logging.info("✓ Conversion completed. Please validate outputs per the checklist.")
    logging.info(f"Output root: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

