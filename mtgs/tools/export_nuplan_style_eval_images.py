#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#

#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import yaml


def _str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export eval rendered images in nuPlan-style layout: "
            "{log_name}/{camera_name}/{cam_token}.jpg."
        )
    )
    parser.add_argument(
        "--load-config",
        type=Path,
        required=True,
        help="Path to the trained experiment config.yml.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory where nuPlan-style rendered images will be written.",
    )
    parser.add_argument(
        "--ns-config",
        type=str,
        default="mtgs/config/MTGS.py",
        help="Method config used to register MTGS with nerfstudio.",
    )
    parser.add_argument(
        "--eval-2hz",
        type=_str_to_bool,
        default=True,
        help="Whether to export eval frames at 2 Hz, matching the saved config default.",
    )
    parser.add_argument(
        "--eval-num-rays-per-chunk",
        type=int,
        default=None,
        help="Optional eval chunk size override to reduce GPU memory.",
    )
    parser.add_argument(
        "--disable-heavy-loaders",
        type=_str_to_bool,
        default=True,
        help="Disable masks/depth loaders that are not needed for image export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    module_name = args.ns_config.replace(".py", "").replace("/", ".")
    os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
    os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "mtgs=" + module_name + ":method"

    from nerfstudio.utils.eval_utils import eval_load_checkpoint

    config = yaml.load(args.load_config.read_text(), Loader=yaml.Loader)

    # nuPlan saving mode requires full-resolution cameras because it re-distorts
    # rendered images back to the raw nuPlan camera domain.
    config.pipeline.image_saving_mode = "nuplan"
    config.pipeline.datamanager.camera_res_scale_factor = 1.0
    config.pipeline.datamanager.dataparser.eval_2hz = args.eval_2hz

    if args.eval_num_rays_per_chunk is not None:
        config.pipeline.model.eval_num_rays_per_chunk = args.eval_num_rays_per_chunk

    if args.disable_heavy_loaders:
        config.pipeline.datamanager.load_mask = False
        config.pipeline.datamanager.load_custom_masks = ()
        config.pipeline.datamanager.load_instance_masks = False
        config.pipeline.datamanager.load_semantic_masks_from = False
        config.pipeline.datamanager.load_lidar_depth = False
        config.pipeline.datamanager.load_pseudo_depth = False

    config.pipeline.model.output_depth_during_training = False
    config.pipeline.model.predict_normals = False
    config.pipeline.model.color_corrected_metrics = False
    config.pipeline.model.lpips_metric = False
    config.pipeline.model.dinov2_metric = False

    config.load_dir = config.get_checkpoint_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    pipeline.eval()
    eval_load_checkpoint(config, pipeline)

    args.output_path.mkdir(parents=True, exist_ok=True)
    metrics = pipeline.get_average_eval_image_metrics(output_path=args.output_path, get_std=False)
    print(f"Saved nuPlan-style rendered images to: {args.output_path}")
    print(metrics)


if __name__ == "__main__":
    main()
