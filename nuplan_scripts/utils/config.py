#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import importlib
import yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple, Union


class BaseConfig:

    def save_config(self, path: str):
        path = Path(path)
        path.write_text(yaml.dump(self), "utf8")
    
    @staticmethod
    def load_from(path: str):
        path = Path(path)
        return yaml.load(path.read_text(), Loader=yaml.Loader)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)

@dataclass
class RoadBlockConfig(BaseConfig):
    
    road_block_name: str

    road_block: Tuple

    city: Literal['sg-one-north', 'us-ma-boston', 'us-na-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']

    data_root: str = "./data/MTGS"

    interval: int = 1
    """interval = 1 -> 10Hz """

    expand_buffer: int = 0
    """
    expand the trajectory by expand_buffer meters
    those lidar_pcs will be used for lidar registration and stacking lidar points
    """

    reconstruct_buffer: int = 0
    """
    buffer reconstruct_buffer meters for better in range reconstruction.
    those frames will be used for reconstruction, including caching images, lidar, semantic map, etc.
    """

    selected_videos: Tuple = field(default_factory=tuple)
    """tuple of video_idx or dict of video_idx and start_frame, end_frame """

    split: Literal['trainval', 'test', 'all'] = 'trainval'
    """data source of nuplan split"""

    collect_raw: bool = False
    """
    collect raw data from nuplan sensor root to data_root
    """

    exclude_bad_registration: bool = True
    """
    exclude videos with bad lidar registration result
    """

    use_colmap_ba: bool = False
    """
    use colmap bundle adjustment to refine the registration result
    """

@dataclass
class FrameCentralConfig(RoadBlockConfig):
    central_log: str = ""
    """nuplan log name"""

    central_tokens: List[str] = field(default_factory=list)
    """list of central tokens"""

    multi_traversal_mode: Literal['reconstruction', 'off'] = 'off'

def load_config(config_path: str) -> Union[RoadBlockConfig, FrameCentralConfig]:

    if config_path.endswith('.py'):
        module_name = os.path.splitext(os.path.basename(config_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        config = config.config
    elif config_path.endswith('.yml') or config_path.endswith('.yaml'):
        config = BaseConfig.load_from(config_path)
    else:
        raise NotImplementedError(f"not supported config file suffix: .{config_path.split('.')[-1]}")

    # MTGS_DATA_DIR relocates the repo-relative "data/" prefix of data_root, so
    # configs stay portable while reconstructions live elsewhere.
    data_dir_override = os.environ.get("MTGS_DATA_DIR")
    data_root_parts = Path(config.data_root).parts
    if data_dir_override and data_root_parts and data_root_parts[0] == "data":
        config.data_root = os.fspath(Path(data_dir_override).joinpath(*data_root_parts[1:]))

    output_dir_override = os.environ.get("MTGS_OUTPUT_DIR")
    if output_dir_override:
        output_dir = Path(os.path.normpath(output_dir_override))
        if output_dir.name in ("", "."):
            raise ValueError(f"MTGS_OUTPUT_DIR must name a scene directory: {output_dir_override}")
        # VideoScene writes to <data_root>/<road_block_name>. Override both
        # fields so a launcher's --output-dir is honored exactly.
        config.data_root = str(output_dir.parent)
        config.road_block_name = output_dir.name

    return config
