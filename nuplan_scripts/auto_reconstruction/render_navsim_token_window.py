#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
"""Render token-centered NAVSIM windows from an auto-reconstruction checkpoint.

This script is intentionally token/window based. For each selected central token
from a FrameCentralConfig, it slices the NAVSIM scene window around that token
(default: 3 frames before, current frame, 8 frames after at 2 Hz), optionally
modifies the ego trajectory, renders the reconstructed cameras, and writes:

    synthetic_sensor_path/<relative camera jpgs>
    synthetic_scenes_path/<synthetic scene token>.pkl

The generated files can be passed to NAVSIM via the usual
synthetic_sensor_path/synthetic_scenes_path configuration fields.

Package-level NAVSIM split files are intentionally generated separately by
build_navsim_package_split.py after all desired scenes have been rendered.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapy as media
import numpy as np
import torch
from nerfstudio.cameras.cameras import CameraType, Cameras as NSCameras
from pyquaternion import Quaternion
from tqdm import tqdm

os.environ.setdefault("NERFSTUDIO_DATAPARSER_CONFIGS", "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser")
os.environ.setdefault(
    "NERFSTUDIO_METHOD_CONFIGS",
    "mtgs_st=mtgs.config.MTGS_ST:method,"
    "mtgs_deformable=mtgs.config.MTGS_deformable:method",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtgs.tools.render import eval_setup  # noqa: E402
from mtgs.utils.camera_utils import invert_distortion, matrix_from_translation_and_quaternion  # noqa: E402
from nuplan_scripts.utils.config import FrameCentralConfig, load_config  # noqa: E402
from nuplan_scripts.auto_reconstruction.candidate_kinematics import (  # noqa: E402
    candidate_ego_dynamics,
    history_query_seconds,
)


CAMERA_NAMES: Tuple[str, ...] = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render modified NAVSIM token windows from an auto-reconstruction MTGS model."
    )
    parser.add_argument("--config", required=True, type=Path, help="FrameCentralConfig yaml.")
    parser.add_argument(
        "--load-config",
        type=Path,
        default=None,
        help="Optional trained Nerfstudio config.yml. Defaults to data_root/background_reconstruction/road_block_name/config.yml.",
    )
    parser.add_argument(
        "--tokens",
        nargs="*",
        default=None,
        help="Central tokens to render. Defaults to all config.central_tokens.",
    )
    parser.add_argument("--source-travel-id", type=int, default=0, help="Traversal id to slice from video_scene_dict.")
    parser.add_argument(
        "--video-scene-pickle",
        choices=("active", "final", "raw", "registered", "colmap"),
        default="final",
        help="Which video_scene_dict pickle to use for MTGS frame/camera metadata.",
    )
    parser.add_argument("--history-frames", type=int, default=4, help="NAVSIM history frames including current frame.")
    parser.add_argument("--future-frames", type=int, default=0, help="NAVSIM future frames after the synthetic current frame.")
    parser.add_argument("--frame-stride", type=int, default=5, help="10 Hz MTGS frame stride for NAVSIM 2 Hz frames.")
    parser.add_argument(
        "--scene-current-offset-frames",
        type=int,
        default=0,
        help="2 Hz frame offset from the central token to the synthetic current frame. Use 8 for a 4 s second-stage endpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("renders/navsim_token_windows"),
        help="Root for generated NAVSIM synthetic files.",
    )
    parser.add_argument(
        "--output-layout",
        choices=("variant", "package"),
        default="variant",
        help="variant writes output-root/road_block/name; package writes output-root/{sensor_blobs,synthetic_scene_pickles} directly.",
    )
    parser.add_argument(
        "--sensor-layout",
        choices=("scene", "log_camera"),
        default=None,
        help="Relative image layout inside sensor_blobs. Defaults to log_camera for package layout, scene otherwise.",
    )
    parser.add_argument("--name", default=None, help="Variant name used under output-root.")
    parser.add_argument("--synthetic-sensor-path", type=Path, default=None)
    parser.add_argument("--synthetic-scenes-path", type=Path, default=None)
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        default=None,
        help="Optional NAVSIM log pkl directory. When provided, benchmark fields are copied from the original logs.",
    )
    parser.add_argument(
        "--annotation-object-types",
        nargs="+",
        default=("vehicle",),
        help=(
            "Object categories to keep in generated NAVSIM scene-pickle annotations and future tracks. "
            "Default is vehicle, matching MTGS_ST rigid-object rendering. Use 'all' to preserve every log object."
        ),
    )
    parser.add_argument(
        "--annotation-track-token-source",
        choices=("mtgs_rendered", "none"),
        default="none",
        help=(
            "Additional annotation filter. mtgs_rendered keeps only objects with a rigid/deformable "
            "Gaussian submodel in the loaded MTGS checkpoint, so 3D boxes match rendered objects. "
            "Use none to keep every object allowed by --annotation-object-types."
        ),
    )
    parser.add_argument("--downscale-factor", type=float, default=1.0, help="Render resolution downscale.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--redistort-nuplan",
        type=str_to_bool,
        default=True,
        help="Redistort rendered images back toward the raw nuPlan camera domain before saving.",
    )
    parser.add_argument(
        "--write-videos",
        type=str_to_bool,
        default=False,
        help="Also save per-camera and stitched multi-view mp4 videos beside the NAVSIM artifacts (disabled by default).",
    )
    parser.add_argument("--video-frame-rate", type=float, default=2.0, help="Frame rate for saved mp4 previews.")
    parser.add_argument("--video-output-dir", type=Path, default=None, help="Optional mp4 output directory. Defaults to <variant>/videos.")
    parser.add_argument(
        "--scene-format",
        choices=("window", "history", "two_stage"),
        default="window",
        help="window writes all output frames inline; history writes only the 4-frame agent input; two_stage writes history plus extended future objects/lights.",
    )
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=32768)
    parser.add_argument("--dx", type=float, default=0.0, help="Global/local x offset applied to the sliced ego path.")
    parser.add_argument("--dy", type=float, default=0.0, help="Global/local y offset applied to the sliced ego path.")
    parser.add_argument("--dz", type=float, default=0.0, help="z offset applied to the sliced ego path.")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Yaw rotation applied to the sliced ego path.")
    parser.add_argument(
        "--pivot",
        choices=("current", "start", "center", "origin"),
        default="current",
        help="Pivot for --yaw-deg.",
    )
    parser.add_argument(
        "--candidate-trajectory-path",
        type=Path,
        default=None,
        help="Optional candidate trajectory .npy with shape (K,T,3) or (T,3), in ego-frame [x-forward, y-left, yaw].",
    )
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument(
        "--candidate-jobs-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of {'token': str, 'candidate_index': int} jobs. "
            "All jobs must use this --config and --candidate-trajectory-path. "
            "The MTGS pipeline is loaded once and reused across the jobs."
        ),
    )
    parser.add_argument("--candidate-frame-rate", type=float, default=10.0)
    parser.add_argument("--candidate-prepend-origin", type=str_to_bool, default=True)
    parser.add_argument("--candidate-use-yaw", type=str_to_bool, default=True)
    parser.add_argument(
        "--candidate-replace-mode",
        choices=("future", "history_first_half"),
        default="future",
        help="future replaces current/future poses; history_first_half anchors the candidate at the first history frame and replaces the path through current.",
    )
    parser.add_argument(
        "--candidate-history-duration",
        type=float,
        default=1.5,
        help="Candidate duration mapped forward from the first history frame. Default samples 0.0, 0.5, 1.0, 1.5s for 4 history frames.",
    )
    parser.add_argument(
        "--candidate-z-mode",
        choices=("distance", "time", "constant"),
        default="distance",
        help="How z is assigned when candidate trajectory replaces current/future poses.",
    )
    parser.add_argument(
        "--skip-invalid-windows",
        action="store_true",
        help=(
            "Skip jobs whose token cannot provide the full requested history/future window. "
            "Skipped jobs are recorded in the manifest with their failure reason."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing synthetic images/pickles.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write metadata describing windows and synthetic tokens; do not load the MTGS model or render images.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def pickle_name(stage: str) -> str:
    names = {
        "active": "video_scene_dict.pkl",
        "final": "video_scene_dict_final.pkl",
        "raw": "video_scene_dict_raw.pkl",
        "registered": "video_scene_dict_registered.pkl",
        "colmap": "video_scene_dict_colmap.pkl",
    }
    return names[stage]


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def select_camera_calibration(cam_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return the raw image-domain calibration used by MTGS for this camera."""

    if "colmap_param" in cam_info:
        intrinsic = cam_info["colmap_param"]["cam_intrinsic"]
        distortion = cam_info["colmap_param"]["distortion"]
    else:
        intrinsic = cam_info["cam_intrinsic"]
        distortion = cam_info["distortion"]
    return np.asarray(intrinsic, dtype=np.float64), np.asarray(distortion, dtype=np.float64)


def optimal_undistorted_intrinsic(
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    image_size: Tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """Match MTGS CustomDataset undistort_images=optimal camera intrinsics."""

    cv_intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
    cv_intrinsic[0, 2] -= 0.5
    cv_intrinsic[1, 2] -= 0.5
    new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
        cv_intrinsic, np.asarray(distortion, dtype=np.float64), image_size, 1
    )
    new_intrinsic[0, 2] += 0.5
    new_intrinsic[1, 2] += 0.5
    return new_intrinsic.astype(np.float32)


def camera_calibration_for_render(cam_info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return render-domain K plus raw-domain K/D for optional redistortion."""

    raw_intrinsic, raw_distortion = select_camera_calibration(cam_info)
    render_intrinsic = optimal_undistorted_intrinsic(raw_intrinsic, raw_distortion)
    return render_intrinsic, raw_intrinsic.astype(np.float32), raw_distortion.astype(np.float32)


def camera_calibration_for_output(
    cam_info: Dict,
    redistort_nuplan: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the calibration that matches the saved JPEG image domain."""

    render_intrinsic, raw_intrinsic, raw_distortion = camera_calibration_for_render(cam_info)
    if redistort_nuplan:
        return raw_intrinsic.astype(np.float32), raw_distortion.astype(np.float32)
    return render_intrinsic.astype(np.float32), np.zeros_like(raw_distortion, dtype=np.float32)


def normalize_annotation_object_types(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if values is None:
        return ("vehicle",)
    normalized = tuple(str(value).strip().lower() for value in values if str(value).strip())
    if not normalized:
        return tuple()
    if "all" in normalized:
        return None
    return normalized


def latest_nerfstudio_checkpoint(load_config_path: Path) -> Path:
    checkpoint_dir = load_config_path.parent / "nerfstudio_models"
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No Nerfstudio checkpoints found in {checkpoint_dir}. "
            "Pass --annotation-track-token-source none to skip MTGS renderable-object filtering."
        )

    def checkpoint_step(path: Path) -> int:
        match = re.search(r"step-(\d+)\.ckpt$", path.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=checkpoint_step)


def load_renderable_annotation_track_tokens(load_config_path: Path) -> Tuple[Tuple[str, ...], Path]:
    """Load object track tokens that have actual MTGS Gaussian object models."""

    checkpoint_path = latest_nerfstudio_checkpoint(load_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("pipeline", checkpoint.get("state_dict", checkpoint))
    if not hasattr(state_dict, "keys"):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a readable state dict.")

    tokens = set()
    object_prefixes = ("rigid_object_", "deformable_object_")
    for key in state_dict.keys():
        marker = "gaussian_models."
        if marker not in key:
            continue
        model_name = key.split(marker, 1)[1].split(".", 1)[0]
        for prefix in object_prefixes:
            if model_name.startswith(prefix):
                token = model_name[len(prefix):]
                if token:
                    tokens.add(str(token))
                break

    if not tokens:
        raise ValueError(
            f"No rigid/deformable object Gaussian models were found in {checkpoint_path}. "
            "Pass --annotation-track-token-source none to skip MTGS renderable-object filtering."
        )
    return tuple(sorted(tokens)), checkpoint_path


def filter_annotations_by_object_types(
    annotations: Dict,
    allowed_object_types: Optional[Tuple[str, ...]],
    allowed_track_tokens: Optional[Tuple[str, ...]],
) -> Dict:
    """Keep only annotation rows whose category and track token are explicitly renderable."""

    if allowed_object_types is None and allowed_track_tokens is None:
        return annotations

    names = [str(x) for x in annotations.get("names", [])]
    track_tokens = [str(x) for x in annotations.get("track_tokens", [])]
    if allowed_track_tokens is not None and len(track_tokens) != len(names):
        raise ValueError(
            "Cannot filter annotations by renderable MTGS track tokens because "
            f"track_tokens has length {len(track_tokens)} but names has length {len(names)}."
        )

    allowed_names = set(allowed_object_types) if allowed_object_types is not None else None
    allowed_tokens = set(allowed_track_tokens) if allowed_track_tokens is not None else None
    keep_indices = []
    for idx, name in enumerate(names):
        if allowed_names is not None and name.lower() not in allowed_names:
            continue
        if allowed_tokens is not None and track_tokens[idx] not in allowed_tokens:
            continue
        keep_indices.append(idx)

    filtered = dict(annotations)
    filtered["names"] = [names[idx] for idx in keep_indices]

    for key in ("instance_tokens", "track_tokens"):
        values = [str(x) for x in annotations.get(key, [])]
        if len(values) == len(names):
            filtered[key] = [values[idx] for idx in keep_indices]
        else:
            filtered[key] = values

    for key in ("boxes", "velocity_3d"):
        values = np.asarray(annotations.get(key, []))
        if values.shape[:1] == (len(names),):
            filtered[key] = values[keep_indices]
        else:
            filtered[key] = values

    return filtered


def synthetic_token(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf8")).hexdigest()[:17]


def deterministic_frame_tokens(base_token: str, offsets: Sequence[int], current_offset: int) -> List[str]:
    frame_tokens = []
    for offset in offsets:
        relative_offset = int(offset) - int(current_offset)
        if relative_offset == 0:
            frame_tokens.append(base_token)
        else:
            frame_tokens.append(synthetic_token(base_token, "frame", relative_offset))
    if len(set(frame_tokens)) != len(frame_tokens):
        raise ValueError(f"Generated duplicate frame tokens from base token {base_token}: {frame_tokens}")
    return frame_tokens


def quaternion_array(value) -> np.ndarray:
    if isinstance(value, Quaternion):
        return np.asarray(value.q, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def yaw_from_quaternion(value) -> float:
    return float(Quaternion(quaternion_array(value)).yaw_pitch_roll[0])


def yaw_from_matrix(matrix: np.ndarray) -> float:
    return float(Quaternion(matrix=np.asarray(matrix)[:3, :3]).yaw_pitch_roll[0])


def make_pose(translation: np.ndarray, yaw: float, template_quat=None) -> Tuple[np.ndarray, np.ndarray]:
    if template_quat is None:
        quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw))
    else:
        old = Quaternion(quaternion_array(template_quat))
        roll = old.yaw_pitch_roll[2]
        pitch = old.yaw_pitch_roll[1]
        quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw)) * Quaternion(axis=[0.0, 1.0, 0.0], angle=pitch) * Quaternion(axis=[1.0, 0.0, 0.0], angle=roll)
    matrix = quat.transformation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)[:3]
    return matrix, np.asarray(quat.q, dtype=np.float64)


def get_local_translation(frame_info: Dict) -> np.ndarray:
    return np.asarray(frame_info["ego2global_translation"], dtype=np.float64).copy()


def get_local_yaw(frame_info: Dict) -> float:
    return yaw_from_quaternion(frame_info["ego2global_rotation"])


def choose_pivot(translations: np.ndarray, current_idx: int, mode: str) -> np.ndarray:
    if mode == "current":
        return translations[current_idx]
    if mode == "start":
        return translations[0]
    if mode == "center":
        return translations.mean(axis=0)
    if mode == "origin":
        return np.zeros(3, dtype=np.float64)
    raise ValueError(f"Unknown pivot: {mode}")


def rotate_xy(points: np.ndarray, pivot: np.ndarray, yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    out = points.copy()
    out[:, :2] = (rot @ (out[:, :2] - pivot[:2]).T).T + pivot[:2]
    return out


def apply_rigid_offset(
    translations: np.ndarray,
    yaws: np.ndarray,
    current_idx: int,
    dx: float,
    dy: float,
    dz: float,
    yaw_deg: float,
    pivot_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pivot = choose_pivot(translations, current_idx, pivot_mode)
    yaw_rad = np.deg2rad(yaw_deg)
    out_t = rotate_xy(translations, pivot, yaw_rad)
    out_t += np.array([dx, dy, dz], dtype=np.float64)[None]
    out_y = yaws + yaw_rad
    return out_t, out_y


def prepare_candidate(candidate: np.ndarray, args: argparse.Namespace, candidate_path: Path) -> np.ndarray:
    if candidate.ndim != 2 or candidate.shape[1] < 2:
        raise ValueError(f"{candidate_path} must have shape (K,T,2+) or (T,2+), got {candidate.shape}")
    candidate = np.asarray(candidate[:, :3] if candidate.shape[1] >= 3 else candidate[:, :2], dtype=np.float64)
    if args.candidate_prepend_origin:
        candidate = np.concatenate([np.zeros((1, candidate.shape[1]), dtype=np.float64), candidate], axis=0)
    return candidate


def load_candidate_bank(args: argparse.Namespace) -> Tuple[Path, np.ndarray]:
    if args.candidate_trajectory_path is None:
        raise ValueError("--candidate-trajectory-path is required when using trajectory candidates.")
    candidate_path = resolve(args.candidate_trajectory_path)
    return candidate_path, np.load(candidate_path)


def candidate_from_bank(
    candidate_bank: np.ndarray,
    candidate_index: int,
    args: argparse.Namespace,
    candidate_path: Path,
) -> np.ndarray:
    if candidate_bank.ndim == 3:
        if candidate_index < 0 or candidate_index >= candidate_bank.shape[0]:
            raise ValueError(
                f"--candidate-index {candidate_index} out of range for {candidate_path}; "
                f"valid range is [0, {candidate_bank.shape[0] - 1}]"
            )
        return prepare_candidate(candidate_bank[candidate_index], args, candidate_path)
    if candidate_bank.ndim == 2:
        return prepare_candidate(candidate_bank, args, candidate_path)
    raise ValueError(f"{candidate_path} must have shape (K,T,2+) or (T,2+), got {candidate_bank.shape}")


def load_candidate(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.candidate_trajectory_path is None:
        return None
    candidate_path, candidate_bank = load_candidate_bank(args)
    return candidate_from_bank(candidate_bank, args.candidate_index, args, candidate_path)


def load_candidate_jobs(path: Path) -> List[Tuple[str, int]]:
    job_path = resolve(path)
    data = json.loads(job_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{job_path} must contain a JSON list of candidate jobs.")

    jobs: List[Tuple[str, int]] = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid candidate job in {job_path}: {item!r}")
        token = str(item.get("token", "")).strip()
        candidate_index = item.get("candidate_index")
        if not token or isinstance(candidate_index, bool):
            raise ValueError(f"Invalid candidate job in {job_path}: {item!r}")
        try:
            candidate_index = int(candidate_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid candidate index in {job_path}: {item!r}") from exc
        identity = (token, candidate_index)
        if identity in seen:
            raise ValueError(f"Duplicate candidate job in {job_path}: {identity}")
        seen.add(identity)
        jobs.append(identity)
    if not jobs:
        raise ValueError(f"No candidate jobs found in {job_path}")
    return jobs


def candidate_variant_name(args: argparse.Namespace, candidate: Optional[np.ndarray], candidate_index: int) -> str:
    variant_name = args.name
    if variant_name is None:
        if candidate is not None:
            variant_name = f"candidate_{candidate_index}"
        elif any(abs(v) > 1e-9 for v in (args.dx, args.dy, args.dz, args.yaw_deg)):
            variant_name = f"offset_dx{args.dx:g}_dy{args.dy:g}_yaw{args.yaw_deg:g}"
        else:
            variant_name = "original_window"
    return variant_name.replace("/", "_")


def interp_rows(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.repeat(values, len(target_t), axis=0)
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    out = np.empty((len(target_t), values.shape[1]), dtype=np.float64)
    for i, t in enumerate(target_t):
        right = int(np.searchsorted(source_t, t, side="right"))
        right = min(max(right, 1), len(values) - 1)
        left = right - 1
        denom = max(source_t[right] - source_t[left], 1e-6)
        alpha = (t - source_t[left]) / denom
        out[i] = values[left] * (1.0 - alpha) + values[right] * alpha
    return out


def arc_lengths_xy(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    segment_lengths = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=-1)
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)])


def candidate_time_axis(candidate: np.ndarray, candidate_frame_rate: float, candidate_prepend_origin: bool) -> np.ndarray:
    first_sample_index = 0 if candidate_prepend_origin else 1
    return (np.arange(len(candidate), dtype=np.float64) + first_sample_index) / candidate_frame_rate


def apply_candidate(
    translations: np.ndarray,
    yaws: np.ndarray,
    current_idx: int,
    candidate: np.ndarray,
    candidate_frame_rate: float,
    candidate_prepend_origin: bool,
    candidate_use_yaw: bool,
    candidate_z_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_frame_rate <= 0:
        raise ValueError("--candidate-frame-rate must be positive.")

    out_t = translations.copy()
    out_y = yaws.copy()
    current = translations[current_idx]
    current_yaw = yaws[current_idx]
    future_count = len(translations) - current_idx
    target_seconds = np.arange(future_count, dtype=np.float64) * 0.5
    candidate_times = candidate_time_axis(candidate, candidate_frame_rate, candidate_prepend_origin)
    candidate_future = interp_rows(candidate, candidate_times, target_seconds)

    forward = np.array([np.cos(current_yaw), np.sin(current_yaw)], dtype=np.float64)
    left = np.array([-np.sin(current_yaw), np.cos(current_yaw)], dtype=np.float64)
    future_xy = current[:2] + candidate_future[:, 0:1] * forward + candidate_future[:, 1:2] * left
    out_t[current_idx:, :2] = future_xy
    if candidate_z_mode == "distance":
        source_future = translations[current_idx:]
        source_s = arc_lengths_xy(source_future)
        candidate_s = arc_lengths_xy(candidate_future)
        out_t[current_idx:, 2] = interp_rows(source_future[:, 2:3], source_s, candidate_s).squeeze(-1)
    elif candidate_z_mode == "constant":
        out_t[current_idx:, 2] = current[2]
    elif candidate_z_mode == "time":
        out_t[current_idx:, 2] = translations[current_idx:, 2]
    else:
        raise ValueError(f"Unknown candidate z mode: {candidate_z_mode}")

    if candidate_use_yaw:
        if candidate_future.shape[1] >= 3:
            out_y[current_idx:] = current_yaw + candidate_future[:, 2]
        elif len(candidate_future) > 1:
            diffs = np.zeros((len(candidate_future), 2), dtype=np.float64)
            diffs[0] = candidate_future[1, :2] - candidate_future[0, :2]
            diffs[-1] = candidate_future[-1, :2] - candidate_future[-2, :2]
            if len(candidate_future) > 2:
                diffs[1:-1] = candidate_future[2:, :2] - candidate_future[:-2, :2]
            out_y[current_idx:] = current_yaw + np.arctan2(diffs[:, 1], diffs[:, 0])

    # Keep the original history shape relative to the modified current pose.
    delta_xy = out_t[current_idx, :2] - translations[current_idx, :2]
    delta_yaw = out_y[current_idx] - yaws[current_idx]
    if current_idx > 0:
        history = translations[:current_idx].copy()
        rel = history[:, :2] - translations[current_idx, :2]
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        out_t[:current_idx, :2] = out_t[current_idx, :2] + (rot @ rel.T).T
        out_t[:current_idx, 2] = translations[:current_idx, 2]
        out_y[:current_idx] = yaws[:current_idx] + delta_yaw

    return out_t, out_y


def apply_candidate_history_first_half(
    translations: np.ndarray,
    yaws: np.ndarray,
    current_idx: int,
    candidate: np.ndarray,
    candidate_frame_rate: float,
    candidate_prepend_origin: bool,
    candidate_use_yaw: bool,
    candidate_z_mode: str,
    candidate_history_duration: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_frame_rate <= 0:
        raise ValueError("--candidate-frame-rate must be positive.")
    if candidate_history_duration <= 0:
        raise ValueError("--candidate-history-duration must be positive.")

    out_t = translations.copy()
    out_y = yaws.copy()
    history_start = translations[0]
    history_start_yaw = yaws[0]
    history_count = current_idx + 1

    target_seconds = candidate_history_duration - np.arange(history_count - 1, -1, -1, dtype=np.float64) * 0.5
    if target_seconds[0] < -1e-6:
        raise ValueError(
            f"History window needs {-target_seconds[0]:.2f}s more candidate duration; "
            f"increase --candidate-history-duration or reduce --history-frames."
        )
    target_seconds = np.maximum(target_seconds, 0.0)

    candidate_times = candidate_time_axis(candidate, candidate_frame_rate, candidate_prepend_origin)
    candidate_history = interp_rows(candidate, candidate_times, target_seconds)

    if candidate_use_yaw and candidate_history.shape[1] >= 3:
        candidate_yaws = candidate_history[:, 2]
    elif len(candidate_history) > 1:
        diffs = np.zeros((len(candidate_history), 2), dtype=np.float64)
        diffs[0] = candidate_history[1, :2] - candidate_history[0, :2]
        diffs[-1] = candidate_history[-1, :2] - candidate_history[-2, :2]
        if len(candidate_history) > 2:
            diffs[1:-1] = candidate_history[2:, :2] - candidate_history[:-2, :2]
        candidate_yaws = np.arctan2(diffs[:, 1], diffs[:, 0])
    else:
        candidate_yaws = np.zeros(len(candidate_history), dtype=np.float64)

    anchor_xy = candidate_history[0, :2]
    anchor_yaw = candidate_yaws[0]
    world_yaw_from_candidate = history_start_yaw - anchor_yaw
    forward = np.array([np.cos(world_yaw_from_candidate), np.sin(world_yaw_from_candidate)], dtype=np.float64)
    left = np.array([-np.sin(world_yaw_from_candidate), np.cos(world_yaw_from_candidate)], dtype=np.float64)
    rel_xy = candidate_history[:, :2] - anchor_xy[None]
    out_t[:history_count, :2] = history_start[:2] + rel_xy[:, 0:1] * forward + rel_xy[:, 1:2] * left
    out_y[:history_count] = world_yaw_from_candidate + candidate_yaws

    if candidate_z_mode == "distance":
        source_history = translations[:history_count]
        source_s = arc_lengths_xy(source_history)
        candidate_s = arc_lengths_xy(candidate_history)
        out_t[:history_count, 2] = interp_rows(source_history[:, 2:3], source_s, candidate_s).squeeze(-1)
    elif candidate_z_mode == "constant":
        out_t[:history_count, 2] = history_start[2]
    elif candidate_z_mode == "time":
        out_t[:history_count, 2] = translations[:history_count, 2]
    else:
        raise ValueError(f"Unknown candidate z mode: {candidate_z_mode}")

    if history_count < len(translations):
        delta_z = out_t[current_idx, 2] - translations[current_idx, 2]
        delta_yaw = out_y[current_idx] - yaws[current_idx]
        rel_future = translations[history_count:, :2] - translations[current_idx, :2]
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        out_t[history_count:, :2] = out_t[current_idx, :2] + (rot @ rel_future.T).T
        out_t[history_count:, 2] = translations[history_count:, 2] + delta_z
        out_y[history_count:] = yaws[history_count:] + delta_yaw

    return out_t, out_y


def update_frame_pose(frame_info: Dict, translation: np.ndarray, yaw: float) -> None:
    matrix, quat = make_pose(translation, yaw, frame_info.get("ego2global_rotation"))
    frame_info["ego2global_translation"] = translation.astype(np.float64)
    frame_info["ego2global_rotation"] = quat
    frame_info["ego2global"] = matrix
    if "can_bus" in frame_info:
        can_bus = np.asarray(frame_info["can_bus"], dtype=np.float64).copy()
        can_bus[:3] = translation[:3]
        can_bus[3:7] = quat[:4]
        frame_info["can_bus"] = can_bus
    if "lidar2ego" in frame_info and frame_info["lidar2ego"] is not None:
        frame_info["lidar2global"] = matrix @ np.asarray(frame_info["lidar2ego"], dtype=np.float64)


def _segment_gradients(translations: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(translations) < 2:
        zeros = np.zeros((len(translations), 2), dtype=np.float64)
        return zeros, zeros.copy()
    edge_order = 2 if len(translations) >= 3 else 1
    velocities = np.gradient(translations[:, :2], dt, axis=0, edge_order=edge_order)
    accelerations = np.gradient(velocities, dt, axis=0, edge_order=edge_order)
    return velocities, accelerations


def recompute_dynamics(frame_infos: List[Dict], current_idx: int) -> None:
    """Recompute ego velocity/acceleration from poses.

    History frames (up to and including ``current_idx``) and future frames are
    differentiated separately: the current ego state must not depend on frames
    after the current frame, which may be logged poses attached to a replaced path.
    """
    translations = np.asarray([get_local_translation(info) for info in frame_infos], dtype=np.float64)
    yaws = np.asarray([get_local_yaw(info) for info in frame_infos], dtype=np.float64)
    dt = 0.5
    split = current_idx + 1
    history_v, history_a = _segment_gradients(translations[:split], dt)
    future_v, future_a = _segment_gradients(translations[split:], dt)
    velocities_global = np.concatenate([history_v, future_v], axis=0)
    accelerations_global = np.concatenate([history_a, future_a], axis=0)
    for idx, info in enumerate(frame_infos):
        yaw = yaws[idx]
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot_world_to_ego = np.array([[c, -s], [s, c]], dtype=np.float64)
        vel = rot_world_to_ego @ velocities_global[idx]
        acc = rot_world_to_ego @ accelerations_global[idx]
        ego_dynamic_state = np.asarray(info.get("ego_dynamic_state", np.zeros(4)), dtype=np.float64)
        if ego_dynamic_state.shape[0] < 4:
            ego_dynamic_state = np.zeros(4, dtype=np.float64)
        ego_dynamic_state[:2] = vel
        ego_dynamic_state[2:4] = acc
        info["ego_dynamic_state"] = ego_dynamic_state.astype(np.float32)
        if "can_bus" in info:
            can_bus = np.asarray(info["can_bus"], dtype=np.float64).copy()
            if can_bus.shape[0] >= 13:
                can_bus[10:12] = vel
                can_bus[12] = 0.0
            info["can_bus"] = can_bus


def transform_boxes_to_new_ego(original_frame: Dict, modified_frame: Dict) -> None:
    boxes = np.asarray(original_frame.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64)
    if boxes.size == 0:
        modified_frame["gt_boxes"] = boxes.astype(np.float32)
        return
    old_t = np.asarray(original_frame["ego2global_translation"], dtype=np.float64)
    new_t = np.asarray(modified_frame["ego2global_translation"], dtype=np.float64)
    old_yaw = get_local_yaw(original_frame)
    new_yaw = get_local_yaw(modified_frame)
    old_c, old_s = np.cos(old_yaw), np.sin(old_yaw)
    old_rot = np.array([[old_c, -old_s], [old_s, old_c]], dtype=np.float64)
    new_c, new_s = np.cos(-new_yaw), np.sin(-new_yaw)
    new_inv_rot = np.array([[new_c, -new_s], [new_s, new_c]], dtype=np.float64)
    global_xy = old_t[:2] + (old_rot @ boxes[:, :2].T).T
    new_local_xy = (new_inv_rot @ (global_xy - new_t[:2]).T).T
    out = boxes.copy()
    out[:, :2] = new_local_xy
    out[:, 6] = boxes[:, 6] + old_yaw - new_yaw
    modified_frame["gt_boxes"] = out.astype(np.float32)


def load_navsim_log_frames(navsim_log_path: Optional[Path], log_name: str) -> Optional[Dict[str, Dict]]:
    if navsim_log_path is None:
        return None
    path = resolve(navsim_log_path) / f"{log_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"NAVSIM log file not found: {path}")
    frames = load_pickle(path)
    return {frame["token"]: frame for frame in frames}


def build_window_indices(
    token_to_idx: Dict[str, int],
    token: str,
    total_frames: int,
    history_frames: int,
    future_frames: int,
    frame_stride: int,
) -> List[int]:
    if token not in token_to_idx:
        raise KeyError(f"Central token {token} was not found in video_scene_dict.")
    center = token_to_idx[token]
    offsets = range(-(history_frames - 1), future_frames + 1)
    indices = [center + offset * frame_stride for offset in offsets]
    if indices[0] < 0 or indices[-1] >= total_frames:
        raise IndexError(
            f"Window around {token} is out of range: {indices[0]}..{indices[-1]} with {total_frames} frames."
        )
    return indices


def build_offset_indices(
    token_to_idx: Dict[str, int],
    token: str,
    total_frames: int,
    offsets: Sequence[int],
    frame_stride: int,
) -> List[int]:
    if token not in token_to_idx:
        raise KeyError(f"Central token {token} was not found in video_scene_dict.")
    center = token_to_idx[token]
    indices = [center + int(offset) * frame_stride for offset in offsets]
    if min(indices) < 0 or max(indices) >= total_frames:
        raise IndexError(
            f"Offsets around {token} are out of range: {min(indices)}..{max(indices)} with {total_frames} frames."
        )
    return indices


def build_camera_path(
    frame_infos: Sequence[Dict],
    camera_name: str,
    travel_id: int,
    frame_indices: Sequence[int],
    travel_time_bounds: Tuple[float, float],
    redistort_nuplan: bool,
    frame_token_to_model_idx: Optional[Dict[str, int]] = None,
) -> Tuple[NSCameras, List[Tuple[np.ndarray, np.ndarray]], List[Dict]]:
    intrinsics = []
    undistort_params = []
    poses = []
    times = []
    metadata = []
    min_ts, max_ts = travel_time_bounds
    duration = max(max_ts - min_ts, 1.0)
    for info, frame_idx in zip(frame_infos, frame_indices):
        cam_info = info["cams"][camera_name]
        intrinsic, raw_intrinsic, raw_distortion = camera_calibration_for_render(cam_info)
        cam2ego = matrix_from_translation_and_quaternion(
            cam_info["sensor2ego_translation"],
            cam_info["sensor2ego_rotation"],
            opencv2nf=True,
        )
        pose = np.asarray(info["ego2global"], dtype=np.float64) @ cam2ego
        cam_timestamp = float(cam_info.get("timestamp", info["timestamp"]))
        intrinsics.append(intrinsic)
        undistort_params.append((raw_intrinsic, raw_distortion))
        poses.append(pose[:3, :4])
        times.append((cam_timestamp - min_ts) / duration)
        frame_token = info["token"]
        model_frame_idx = None
        if frame_token_to_model_idx is None:
            model_frame_idx = int(frame_idx)
        else:
            mapped_idx = frame_token_to_model_idx.get(frame_token)
            if mapped_idx is not None:
                model_frame_idx = int(mapped_idx)

        metadata.append(
            {
                "travel_id": travel_id,
                # MTGS dynamic nodes expect frame indices from the filtered
                # training dataparser, not the raw video_scene_dict index. If
                # this source frame was filtered out during reconstruction,
                # leave frame_idx unset and let MTGS use the normalized
                # timestamp for object-pose interpolation.
                "frame_idx": model_frame_idx,
                "source_frame_idx": int(frame_idx),
                "frame_token": frame_token,
                "cam_token": Path(cam_info["data_path"]).stem,
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "angular_velocity": np.zeros(3, dtype=np.float32),
            }
        )

    intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
    poses_np = np.asarray(poses, dtype=np.float32)
    norm_times = np.clip(np.asarray(times, dtype=np.float64), 0.0, 1.0)
    cameras = NSCameras(
        fx=torch.from_numpy(intrinsics_np[:, 0, 0]),
        fy=torch.from_numpy(intrinsics_np[:, 1, 1]),
        cx=torch.from_numpy(intrinsics_np[:, 0, 2]),
        cy=torch.from_numpy(intrinsics_np[:, 1, 2]),
        height=1080,
        width=1920,
        camera_to_worlds=torch.from_numpy(poses_np),
        camera_type=CameraType.PERSPECTIVE,
        times=torch.from_numpy(norm_times.astype(np.float32)),
    )
    return cameras, undistort_params, metadata


def render_camera_images(
    pipeline,
    cameras: NSCameras,
    output_paths: Sequence[Path],
    undistort_params: Sequence[Tuple[np.ndarray, np.ndarray]],
    camera_metadata: Sequence[Dict],
    downscale_factor: float,
    jpeg_quality: int,
    redistort_nuplan: bool,
) -> List[np.ndarray]:
    if len(cameras) != len(output_paths):
        raise ValueError("Camera count and output path count differ.")
    cameras.rescale_output_resolution(1.0 / downscale_factor)
    cameras = cameras.to(pipeline.device)
    rendered_images = []
    for camera_idx, output_path in enumerate(output_paths):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        camera = cameras[camera_idx : camera_idx + 1]
        camera.metadata = dict(camera_metadata[camera_idx])
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        image = (outputs["rgb"].detach().cpu().numpy() * 255.0).astype(np.uint8)
        if redistort_nuplan:
            intrinsic, distortion = undistort_params[camera_idx]
            image = invert_distortion(image, intrinsic, distortion)
        media.write_image(output_path, image, fmt="jpeg", quality=jpeg_quality)
        rendered_images.append(image)
    return rendered_images


def write_preview_videos(
    rendered_by_camera: Dict[str, List[np.ndarray]],
    output_dir: Path,
    fps: float,
    filename_suffix: str = "",
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths: Dict[str, str] = {}
    for camera_name, images in rendered_by_camera.items():
        if not images:
            continue
        path = output_dir / f"{camera_name}{filename_suffix}.mp4"
        media.write_video(path, np.stack(images, axis=0), fps=fps)
        video_paths[camera_name] = str(path)

    groups = {
        "concat_front": ("CAM_L0", "CAM_F0", "CAM_R0"),
        "concat_back": ("CAM_R2", "CAM_B0", "CAM_L2"),
    }
    for group_name, cameras in groups.items():
        if not all(camera in rendered_by_camera for camera in cameras):
            continue
        frame_counts = {len(rendered_by_camera[camera]) for camera in cameras}
        if len(frame_counts) != 1:
            raise ValueError(f"Cannot stitch {group_name}; frame counts differ: {sorted(frame_counts)}")
        stitched = []
        for frame_idx in range(next(iter(frame_counts))):
            stitched.append(np.concatenate([rendered_by_camera[camera][frame_idx] for camera in cameras], axis=1))
        path = output_dir / f"{group_name}{filename_suffix}.mp4"
        media.write_video(path, np.stack(stitched, axis=0), fps=fps)
        video_paths[group_name] = str(path)
    return video_paths


def make_sensor2lidar(cam_info: Dict, lidar2ego) -> Tuple[np.ndarray, np.ndarray]:
    sensor2ego = matrix_from_translation_and_quaternion(
        cam_info["sensor2ego_translation"],
        cam_info["sensor2ego_rotation"],
        opencv2nf=False,
    )
    ego2lidar = np.linalg.inv(np.asarray(lidar2ego, dtype=np.float64))
    sensor2lidar = ego2lidar @ sensor2ego
    return sensor2lidar[:3, :3].astype(np.float32), sensor2lidar[:3, 3].astype(np.float32)


def make_scene_frame_data(
    original_frame: Dict,
    modified_frame: Dict,
    synthetic_frame_token: str,
    scene_token: str,
    log_name: str,
    map_name: str,
    global_translation_offset: np.ndarray,
    relative_camera_paths: Dict[str, Path],
    annotation_object_types: Optional[Tuple[str, ...]],
    annotation_track_tokens: Optional[Tuple[str, ...]],
    redistort_nuplan: bool,
) -> Dict:
    local_t = np.asarray(modified_frame["ego2global_translation"], dtype=np.float64)
    local_yaw = get_local_yaw(modified_frame)
    ego_pose = np.array(
        [
            local_t[0] + global_translation_offset[0],
            local_t[1] + global_translation_offset[1],
            local_yaw,
        ],
        dtype=np.float64,
    )
    ego_dynamic_state = np.asarray(modified_frame.get("ego_dynamic_state", np.zeros(4)), dtype=np.float32)
    driving_command = original_frame.get("driving_command", modified_frame.get("driving_command", np.zeros(4, dtype=np.int64)))
    annotations = {
        "boxes": np.asarray(modified_frame.get("gt_boxes", np.zeros((0, 7))), dtype=np.float32),
        "names": [str(x) for x in np.asarray(modified_frame.get("gt_names", []), dtype=object).tolist()],
        "velocity_3d": np.asarray(modified_frame.get("gt_velocity_3d", np.zeros((0, 3))), dtype=np.float32),
        "instance_tokens": [str(x) for x in np.asarray(modified_frame.get("instance_tokens", []), dtype=object).tolist()],
        "track_tokens": [str(x) for x in np.asarray(modified_frame.get("track_tokens", []), dtype=object).tolist()],
    }
    annotations = filter_annotations_by_object_types(
        annotations,
        allowed_object_types=annotation_object_types,
        allowed_track_tokens=annotation_track_tokens,
    )

    camera_dict = {}
    for camera_name in CAMERA_NAMES:
        cam_info = modified_frame["cams"][camera_name]
        sensor2lidar_rotation, sensor2lidar_translation = make_sensor2lidar(cam_info, modified_frame["lidar2ego"])
        output_intrinsic, output_distortion = camera_calibration_for_output(cam_info, redistort_nuplan)
        camera_dict[camera_name.lower()] = {
            "data_path": relative_camera_paths[camera_name],
            "sensor2lidar_rotation": sensor2lidar_rotation,
            "sensor2lidar_translation": sensor2lidar_translation,
            "cam_intrinsic": output_intrinsic,
            "distortion": output_distortion,
        }

    return {
        "token": synthetic_frame_token,
        "timestamp": int(modified_frame["timestamp"]),
        "roadblock_ids": list(original_frame.get("roadblock_ids", [])),
        "traffic_lights": list(original_frame.get("traffic_lights", [])),
        "annotations": annotations,
        "ego_status": {
            "ego_pose": ego_pose,
            "ego_velocity": np.asarray(ego_dynamic_state[:2], dtype=np.float32),
            "ego_acceleration": np.asarray(ego_dynamic_state[2:4], dtype=np.float32),
            "driving_command": np.asarray(driving_command),
            "in_global_frame": True,
        },
        "lidar_path": None,
        "camera_dict": camera_dict,
    }


def load_original_navsim_window(
    navsim_frames_by_token: Optional[Dict[str, Dict]],
    mtgs_frames: Sequence[Dict],
) -> List[Dict]:
    originals = []
    for frame in mtgs_frames:
        if navsim_frames_by_token is not None and frame["token"] in navsim_frames_by_token:
            originals.append(navsim_frames_by_token[frame["token"]])
        else:
            originals.append(frame)
    return originals


def write_manifest(path: Path, entries: List[Dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf8")


def frame_to_detection_tracks(frame_data: Dict):
    from navsim.common.dataclasses import Annotations, EgoStatus
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        annotations_to_detection_tracks,
        ego_status_to_ego_state,
    )
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    annotations = Annotations(**frame_data["annotations"])
    ego_status = EgoStatus(**frame_data["ego_status"])
    ego_state = ego_status_to_ego_state(
        ego_status=ego_status,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(int(frame_data["timestamp"])),
    )
    return annotations_to_detection_tracks(annotations, ego_state)


def serialize_scene_frames(
    scene_frames: List[Dict],
    history_frames: int,
    future_frames: int,
    scene_format: str,
) -> Tuple[List[Dict], int, Optional[List], Optional[List]]:
    if scene_format == "window":
        return scene_frames, future_frames, None, None

    if len(scene_frames) < history_frames:
        raise ValueError(f"Cannot write {scene_format} scene with {len(scene_frames)} frames and {history_frames} history frames.")

    if scene_format == "history":
        return scene_frames[:history_frames], 0, None, None

    future_scene_frames = scene_frames[history_frames:]
    extended_detections_tracks = [frame_to_detection_tracks(frame) for frame in future_scene_frames]
    history_scene_frames = scene_frames[:history_frames]
    for frame in history_scene_frames[:-1]:
        # Routes and traffic lights are not consumed from past agent-input
        # frames. Keep traffic annotations for visualization and debugging.
        frame["roadblock_ids"] = []
        frame["traffic_lights"] = []
    return history_scene_frames, 0, None, extended_detections_tracks


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path.as_posix())
    if not isinstance(config, FrameCentralConfig):
        raise TypeError(f"Expected FrameCentralConfig, got {type(config)}")

    reconstruction_dir = Path(config.data_root) / "background_reconstruction" / config.road_block_name
    load_config_path = resolve(args.load_config) if args.load_config else reconstruction_dir / "config.yml"
    road_block_dir = Path(config.data_root) / config.road_block_name
    video_pickle = road_block_dir / pickle_name(args.video_scene_pickle)
    video_scene_dict = load_pickle(video_pickle)
    video_token = f"{config.road_block_name}-{args.source_travel_id}"
    if video_token not in video_scene_dict:
        raise KeyError(f"{video_token} not found in {video_pickle}. Available keys: {list(video_scene_dict)}")

    video_info = video_scene_dict[video_token]
    source_frames = video_info["frame_infos"]
    token_to_idx = {frame["token"]: idx for idx, frame in enumerate(source_frames)}
    selected_tokens = args.tokens if args.tokens is not None and len(args.tokens) > 0 else list(config.central_tokens)
    candidate_jobs = load_candidate_jobs(args.candidate_jobs_json) if args.candidate_jobs_json is not None else None
    if candidate_jobs is not None:
        if args.tokens:
            raise ValueError("--tokens cannot be combined with --candidate-jobs-json.")
        if args.output_layout != "package":
            raise ValueError("--candidate-jobs-json requires --output-layout package.")
        candidate_path, candidate_bank = load_candidate_bank(args)
        if candidate_bank.ndim != 3:
            raise ValueError(
                f"--candidate-jobs-json requires a candidate bank with shape (K,T,2+), got {candidate_bank.shape}"
            )
        render_jobs = [
            (token, candidate_index, candidate_from_bank(candidate_bank, candidate_index, args, candidate_path))
            for token, candidate_index in candidate_jobs
        ]
    else:
        candidate = load_candidate(args)
        render_jobs = [(token, args.candidate_index, candidate) for token in selected_tokens]
    annotation_object_types = normalize_annotation_object_types(args.annotation_object_types)
    annotation_track_tokens = None
    annotation_track_token_checkpoint = None
    if args.annotation_track_token_source == "mtgs_rendered":
        annotation_track_tokens, annotation_track_token_checkpoint = load_renderable_annotation_track_tokens(load_config_path)
        print(
            "Filtering scene-pickle annotations to "
            f"{len(annotation_track_tokens)} renderable MTGS object track tokens from: "
            f"{annotation_track_token_checkpoint}"
        )
    if args.output_layout == "package":
        output_root = resolve(args.output_root)
    else:
        output_root = resolve(args.output_root) / config.road_block_name / candidate_variant_name(
            args, render_jobs[0][2], render_jobs[0][1]
        )
    sensor_layout = args.sensor_layout or ("log_camera" if args.output_layout == "package" else "scene")
    synthetic_sensor_path = resolve(args.synthetic_sensor_path) if args.synthetic_sensor_path else output_root / "sensor_blobs"
    synthetic_scenes_path = resolve(args.synthetic_scenes_path) if args.synthetic_scenes_path else output_root / "synthetic_scene_pickles"
    video_output_dir = resolve(args.video_output_dir) if args.video_output_dir else output_root / "videos"
    if not args.dry_run:
        synthetic_sensor_path.mkdir(parents=True, exist_ok=True)
    synthetic_scenes_path.mkdir(parents=True, exist_ok=True)
    if args.write_videos and not args.dry_run:
        video_output_dir.mkdir(parents=True, exist_ok=True)

    navsim_frames_by_token = load_navsim_log_frames(args.navsim_log_path, config.central_log)
    global_translation_offset = np.asarray(video_info["global2world_translation"], dtype=np.float64)
    all_timestamps = []
    for info in source_frames:
        all_timestamps.append(float(info["timestamp"]))
        for cam_info in info["cams"].values():
            all_timestamps.append(float(cam_info.get("timestamp", info["timestamp"])))
    travel_time_bounds = (min(all_timestamps), max(all_timestamps))

    validation_output_offsets = list(
        range(
            args.scene_current_offset_frames - (args.history_frames - 1),
            args.scene_current_offset_frames + args.future_frames + 1,
        )
    )
    validation_trajectory_offsets = list(
        range(min(0, validation_output_offsets[0]), max(0, validation_output_offsets[-1]) + 1)
    )
    skipped_window_jobs: List[Dict] = []
    valid_render_jobs = []
    for central_token, candidate_index, candidate in render_jobs:
        try:
            build_offset_indices(
                token_to_idx=token_to_idx,
                token=central_token,
                total_frames=len(source_frames),
                offsets=validation_trajectory_offsets,
                frame_stride=args.frame_stride,
            )
        except (KeyError, IndexError) as exc:
            if not args.skip_invalid_windows:
                raise
            skipped_window_jobs.append(
                {
                    "central_token": central_token,
                    "candidate_index": candidate_index if candidate is not None else None,
                    "skipped": True,
                    "skip_reason": str(exc),
                }
            )
            print(f"Skipping invalid NAVSIM window: {central_token} ({exc})")
        else:
            valid_render_jobs.append((central_token, candidate_index, candidate))
    render_jobs = valid_render_jobs

    pipeline = None
    if render_jobs and not args.dry_run:
        def update_config(ns_config):
            ns_config.pipeline.datamanager.dataparser.train_split_fraction = 1.0
            ns_config.pipeline.datamanager.dataparser.train_scene_travels = (args.source_travel_id,)
            ns_config.pipeline.datamanager.dataparser.eval_scene_travels = (args.source_travel_id,)
            ns_config.pipeline.datamanager.dataparser.cameras = CAMERA_NAMES
            ns_config.pipeline.datamanager.load_mask = False
            ns_config.pipeline.datamanager.load_custom_masks = ()
            ns_config.pipeline.datamanager.load_instance_masks = False
            ns_config.pipeline.datamanager.load_semantic_masks_from = False
            ns_config.pipeline.datamanager.load_lidar_depth = False
            ns_config.pipeline.datamanager.load_pseudo_depth = False
            ns_config.pipeline.model.output_depth_during_training = False
            ns_config.pipeline.model.predict_normals = False
            ns_config.pipeline.model.color_corrected_metrics = False
            ns_config.pipeline.model.lpips_metric = False
            if hasattr(ns_config.pipeline.model, "dinov2_metric"):
                ns_config.pipeline.model.dinov2_metric = False
            return ns_config

        _, pipeline, _, _ = eval_setup(
            load_config_path,
            eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )

    frame_token_to_model_idx = None
    if pipeline is not None:
        frame_token_to_model_idx = getattr(pipeline.model, "frame_token2frame_idx", None)

    manifest = list(skipped_window_jobs)
    for central_token, candidate_index, candidate in tqdm(render_jobs, desc="Rendering NAVSIM token windows"):
        output_offsets = list(
            range(
                args.scene_current_offset_frames - (args.history_frames - 1),
                args.scene_current_offset_frames + args.future_frames + 1,
            )
        )
        trajectory_offsets = list(range(min(0, output_offsets[0]), max(0, output_offsets[-1]) + 1))
        trajectory_indices = build_offset_indices(
            token_to_idx=token_to_idx,
            token=central_token,
            total_frames=len(source_frames),
            offsets=trajectory_offsets,
            frame_stride=args.frame_stride,
        )
        output_index_positions = [trajectory_offsets.index(offset) for offset in output_offsets]
        original_trajectory_frames = [source_frames[idx] for idx in trajectory_indices]
        working_frames = [copy.deepcopy(frame) for frame in original_trajectory_frames]
        current_idx = trajectory_offsets.index(0)

        translations = np.asarray([get_local_translation(frame) for frame in working_frames], dtype=np.float64)
        yaws = np.asarray([get_local_yaw(frame) for frame in working_frames], dtype=np.float64)
        translations, yaws = apply_rigid_offset(
            translations,
            yaws,
            current_idx=current_idx,
            dx=args.dx,
            dy=args.dy,
            dz=args.dz,
            yaw_deg=args.yaw_deg,
            pivot_mode=args.pivot,
        )
        if candidate is not None:
            if args.candidate_replace_mode == "history_first_half":
                translations, yaws = apply_candidate_history_first_half(
                    translations,
                    yaws,
                    current_idx=current_idx,
                    candidate=candidate,
                    candidate_frame_rate=args.candidate_frame_rate,
                    candidate_prepend_origin=args.candidate_prepend_origin,
                    candidate_use_yaw=args.candidate_use_yaw,
                    candidate_z_mode=args.candidate_z_mode,
                    candidate_history_duration=args.candidate_history_duration,
                )
            else:
                translations, yaws = apply_candidate(
                    translations,
                    yaws,
                    current_idx=current_idx,
                    candidate=candidate,
                    candidate_frame_rate=args.candidate_frame_rate,
                    candidate_prepend_origin=args.candidate_prepend_origin,
                    candidate_use_yaw=args.candidate_use_yaw,
                    candidate_z_mode=args.candidate_z_mode,
                )

        for frame, translation, yaw in zip(working_frames, translations, yaws):
            update_frame_pose(frame, translation, yaw)
        recompute_dynamics(working_frames, current_idx)
        if candidate is not None and args.candidate_replace_mode == "history_first_half":
            # The rendered history follows the candidate; take its exact kinematics.
            history_dynamics = candidate_ego_dynamics(
                candidate,
                args.candidate_frame_rate,
                args.candidate_prepend_origin,
                history_query_seconds(current_idx + 1, args.candidate_history_duration),
            )
            for frame, dynamics in zip(working_frames[: current_idx + 1], history_dynamics):
                ego_dynamic_state = np.asarray(frame["ego_dynamic_state"], dtype=np.float64).copy()
                ego_dynamic_state[:4] = dynamics
                frame["ego_dynamic_state"] = ego_dynamic_state.astype(np.float32)
                if "can_bus" in frame:
                    can_bus = np.asarray(frame["can_bus"], dtype=np.float64).copy()
                    if can_bus.shape[0] >= 13:
                        can_bus[10:12] = dynamics[:2]
                    frame["can_bus"] = can_bus
        for original_frame, modified_frame in zip(original_trajectory_frames, working_frames):
            transform_boxes_to_new_ego(original_frame, modified_frame)

        original_mtgs_frames = [original_trajectory_frames[pos] for pos in output_index_positions]
        modified_frames = [working_frames[pos] for pos in output_index_positions]
        window_indices = [trajectory_indices[pos] for pos in output_index_positions]
        output_current_idx = args.history_frames - 1

        variant_name = candidate_variant_name(args, candidate, candidate_index)
        candidate_tail = str(candidate_index) if candidate is not None else None
        scene_identity = [config.road_block_name, central_token, variant_name, args.scene_current_offset_frames]
        if candidate is not None:
            scene_identity.extend(
                [
                    resolve(args.candidate_trajectory_path),
                    candidate_index,
                    args.candidate_replace_mode,
                    args.candidate_history_duration,
                    args.candidate_z_mode,
                ]
            )
        scene_token = synthetic_token(*scene_identity)
        synthetic_frame_tokens = deterministic_frame_tokens(
            scene_token, output_offsets, args.scene_current_offset_frames
        )
        initial_token = synthetic_frame_tokens[output_current_idx]
        navsim_original_frames = load_original_navsim_window(navsim_frames_by_token, original_mtgs_frames)

        relative_camera_paths: List[Dict[str, Path]] = []
        render_frame_count = args.history_frames if args.scene_format == "two_stage" else len(modified_frames)
        for frame_idx, synthetic_frame_token in enumerate(synthetic_frame_tokens):
            per_frame = {}
            for camera_name in CAMERA_NAMES:
                if sensor_layout == "log_camera":
                    rel = Path(config.central_log) / camera_name / f"{synthetic_frame_token}.jpg"
                else:
                    rel = Path(config.central_log) / scene_token / camera_name / f"{synthetic_frame_token}.jpg"
                out = synthetic_sensor_path / rel
                if frame_idx < render_frame_count and out.exists() and not args.overwrite and not args.dry_run:
                    raise FileExistsError(f"{out} already exists. Pass --overwrite to replace outputs.")
                per_frame[camera_name] = rel
            relative_camera_paths.append(per_frame)

        video_paths: Dict[str, str] = {}
        if not args.dry_run:
            rendered_by_camera: Dict[str, List[np.ndarray]] = {}
            for camera_name in CAMERA_NAMES:
                cameras, undistort_params, camera_metadata = build_camera_path(
                    modified_frames[:render_frame_count],
                    camera_name,
                    travel_id=args.source_travel_id,
                    frame_indices=window_indices[:render_frame_count],
                    travel_time_bounds=travel_time_bounds,
                    redistort_nuplan=args.redistort_nuplan,
                    frame_token_to_model_idx=frame_token_to_model_idx,
                )
                output_paths = [
                    synthetic_sensor_path / per_frame[camera_name]
                    for per_frame in relative_camera_paths[:render_frame_count]
                ]
                rendered_by_camera[camera_name] = render_camera_images(
                    pipeline=pipeline,
                    cameras=cameras,
                    output_paths=output_paths,
                    undistort_params=undistort_params,
                    camera_metadata=camera_metadata,
                    downscale_factor=args.downscale_factor,
                    jpeg_quality=args.jpeg_quality,
                    redistort_nuplan=args.redistort_nuplan,
                )
            if args.write_videos:
                token_video_dir = video_output_dir / central_token
                video_suffix = f"_{candidate_tail}" if candidate_tail is not None else ""
                video_paths = write_preview_videos(rendered_by_camera, token_video_dir, args.video_frame_rate, video_suffix)

        scene_frames = []
        for frame_idx, (original_frame, modified_frame) in enumerate(zip(navsim_original_frames, modified_frames)):
            scene_frames.append(
                make_scene_frame_data(
                    original_frame=original_frame,
                    modified_frame=modified_frame,
                    synthetic_frame_token=synthetic_frame_tokens[frame_idx],
                    scene_token=scene_token,
                    log_name=config.central_log,
                    map_name=config.city,
                    global_translation_offset=global_translation_offset,
                    relative_camera_paths=relative_camera_paths[frame_idx],
                    annotation_object_types=annotation_object_types,
                    annotation_track_tokens=annotation_track_tokens,
                    redistort_nuplan=args.redistort_nuplan,
                )
            )

        serialized_frames, metadata_future_frames, extended_traffic_light_data, extended_detections_tracks = serialize_scene_frames(
            scene_frames=scene_frames,
            history_frames=args.history_frames,
            future_frames=args.future_frames,
            scene_format=args.scene_format,
        )

        scene_dict = {
            "scene_metadata": {
                "log_name": config.central_log,
                "scene_token": scene_token,
                "map_name": config.city,
                "initial_token": initial_token,
                "num_history_frames": args.history_frames,
                "num_future_frames": metadata_future_frames,
            },
            "frames": serialized_frames,
            "extended_traffic_light_data": extended_traffic_light_data,
            "extended_detections_tracks": extended_detections_tracks,
        }
        scene_path = synthetic_scenes_path / f"{scene_token}.pkl"
        if scene_path.exists() and not args.overwrite:
            raise FileExistsError(f"{scene_path} already exists. Pass --overwrite to replace outputs.")
        dump_pickle(scene_path, scene_dict)

        manifest.append(
            {
                "central_token": central_token,
                "synthetic_scene_token": scene_token,
                "synthetic_initial_token": initial_token,
                "scene_pickle": str(scene_path),
                "window_source_tokens": [frame["token"] for frame in original_mtgs_frames],
                "output_offsets_2hz": output_offsets,
                "trajectory_offsets_2hz": trajectory_offsets,
                "scene_current_offset_frames": args.scene_current_offset_frames,
                "candidate_index": candidate_index if candidate is not None else None,
                "candidate_replace_mode": args.candidate_replace_mode if candidate is not None else None,
                "candidate_history_duration": args.candidate_history_duration if candidate is not None else None,
                "history_frames": args.history_frames,
                "future_frames": args.future_frames,
                "scene_format": args.scene_format,
                "annotation_object_types": list(annotation_object_types) if annotation_object_types is not None else "all",
                "annotation_track_token_source": args.annotation_track_token_source,
                "annotation_track_token_count": len(annotation_track_tokens) if annotation_track_tokens is not None else None,
                "annotation_track_token_checkpoint": str(annotation_track_token_checkpoint) if annotation_track_token_checkpoint else None,
                "synthetic_sensor_path": str(synthetic_sensor_path),
                "output_layout": args.output_layout,
                "sensor_layout": sensor_layout,
                "video_dir": str(video_output_dir / central_token) if args.write_videos else None,
                "videos": video_paths,
            }
        )

    manifest_path = output_root / "manifest.json"
    write_manifest(manifest_path, manifest)
    if args.dry_run:
        print("Dry run: no sensor images rendered (scene pickles reference their future sensor_blobs paths).")
    else:
        print(f"Wrote synthetic sensor blobs to: {synthetic_sensor_path}")
    print(f"Wrote synthetic scene pickles to: {synthetic_scenes_path}")
    if args.write_videos and not args.dry_run:
        print(f"Wrote preview videos to: {video_output_dir}")
    print(f"Wrote manifest to: {manifest_path}")
    print("Synthetic initial tokens for NAVSIM overrides:")
    for item in manifest:
        if not item.get("skipped", False):
            print(item["synthetic_initial_token"])


if __name__ == "__main__":
    main()
