#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import yaml
import pytz
from functools import lru_cache
from pathlib import Path
from rich.console import Console

CONSOLE = Console(width=120)
os.environ['TQDM_NCOLS'] = '120'

# load nuplan config
# NUPLAN_PATH_CONFIG may point to a machine-local copy of nuplan_path.yml.
nuplan_config_path = os.environ.get(
    'NUPLAN_PATH_CONFIG',
    os.path.join(os.path.dirname(__file__), '..', 'configs', 'common', 'nuplan_path.yml'),
)
with open(nuplan_config_path, 'r') as f:
    nuplan_config = yaml.safe_load(f)
NUPLAN_DATA_ROOT = nuplan_config['NUPLAN_DATA_ROOT']
NUPLAN_DB_FILES = nuplan_config['NUPLAN_DB_FILES']
NUPLAN_SENSOR_ROOT = nuplan_config['NUPLAN_SENSOR_ROOT']
NUPLAN_MAP_VERSION = nuplan_config['NUPLAN_MAP_VERSION']
NUPLAN_MAPS_ROOT = nuplan_config['NUPLAN_MAPS_ROOT']


def _discover_sensor_roots():
    roots = [Path(NUPLAN_SENSOR_ROOT)]
    dataset_root = Path(NUPLAN_DATA_ROOT).parent
    shard_patterns = (
        'nuplan-v1.1_*_camera_*',
        'nuplan-v1.1_*_lidar_*',
    )
    for pattern in shard_patterns:
        roots.extend(sorted(path for path in dataset_root.glob(pattern) if path.is_dir()))

    seen = set()
    unique_roots = []
    for root in roots:
        root_str = os.fspath(root)
        if root_str not in seen:
            seen.add(root_str)
            unique_roots.append(root_str)
    return tuple(unique_roots)


NUPLAN_SENSOR_ROOTS = _discover_sensor_roots()


@lru_cache(maxsize=200000)
def resolve_nuplan_sensor_path(relative_path):
    """Resolve DB-relative nuPlan sensor paths from normalized roots or extracted shards."""
    relative_path = os.fspath(relative_path)
    if os.path.isabs(relative_path):
        return relative_path
    for sensor_root in NUPLAN_SENSOR_ROOTS:
        candidate = os.path.join(sensor_root, relative_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(NUPLAN_SENSOR_ROOT, relative_path)


cityscape_label = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'wall': 3,
    'fence': 4,
    'pole': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person': 11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorcycle': 17,
    'bicycle': 18
}

cityscape_colormap = {
    'person': (220, 20, 60),       # Red
    'car': (0, 0, 142),            # Dark blue
    'truck': (0, 0, 70),           # Darker blue
    'bus': (0, 60, 100),           # Blue
    'rider': (255, 0, 0),          # Bright red
    'motorcycle': (0, 0, 230),     # Bright blue
    'bicycle': (119, 11, 32),      # Dark red
}

NUPLAN_ACCEPTABLE_CITYSCAPE_LABELS = {
    "vehicle": [13, 14, 15], 
    "bicycle": [12, 17, 18], 
    "pedestrian": [11], 
    "traffic cone": [5], 
    "barrier": [], 
    "construction zone sign": [], 
    "generic object": [], 
    "background": []
}

NUPLAN_TIMEZONE = {
    'us-ma-boston': pytz.timezone('America/New_York'),
    'sg-one-north': pytz.timezone('Asia/Singapore'),
    'us-nv-las-vegas-strip': pytz.timezone('America/Los_Angeles'),
    'us-pa-pittsburgh-hazelwood': pytz.timezone('US/Eastern'),
}
