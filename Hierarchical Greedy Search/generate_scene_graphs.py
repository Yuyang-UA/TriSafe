#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS / NAVSIM safety-critical scene graph generation                            #
#-------------------------------------------------------------------------------#
"""Generate ego-centric danger scene graphs for one trajectory candidate.

The script intentionally mirrors the ego-trajectory replacement semantics used by
``nuplan_scripts/auto_reconstruction/render_navsim_token_window.py`` while avoiding
raw sensor access. It only reads NAVSIM-style traffic log pickles and, for static
boundary risk, the nuPlan map database. For the MTGS vehicle-only rendering
setting, dynamic TTC graphs include vehicle annotations by default. Saved JSON is
transition-only by default, so endpoint physical geometry stays internal.

Example:

    python "Hierarchical Greedy Search/generate_scene_graphs.py" \
      --config data/auto_reconstruction/config/navtest_available_merged_pointcloud_good/2021.05.25.14.16.10_veh-35_00083_00485-4b594a8e80915943.yaml \
      --tokens 4b594a8e80915943 \
      --candidate-trajectory-path trajectory_candidate/human_trajectory_candidate.npy \
      --candidate-index 10000 \
      --candidate-replace-mode history_first_half \
      --candidate-history-duration 1.5 \
      --history-frames 4 \
      --navsim-log-path "${OPENSCENE_DATA_ROOT}/navsim_logs/test" \
      --output-root "Hierarchical Greedy Search/scene_graphs" \
      --overwrite
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nuplan_scripts.utils.config import FrameCentralConfig, load_config  # noqa: E402


DEFAULT_DYNAMIC_OBJECT_TYPES = ("vehicle",)
DEFAULT_EXCLUDED_TYPES = ("background",)
DEFAULT_EGO_LENGTH = 5.176
DEFAULT_EGO_WIDTH = 2.297
DEFAULT_REAR_AXLE_TO_CENTER = 1.461


@dataclass(frozen=True)
class EgoFootprintParams:
    length: float
    width: float
    rear_axle_to_center: float


@dataclass
class ObjectState:
    token: str
    instance_token: str
    category: str
    node_type: str
    center: np.ndarray
    yaw: float
    length: float
    width: float
    height: float
    velocity_xy: np.ndarray
    frame_token: str
    frame_index: int

    def polygon(self) -> Polygon:
        return oriented_box_polygon(self.center, self.yaw, self.length, self.width)


@dataclass
class BoundaryLine:
    token: str
    geometry: LineString
    length: float
    distance_to_window: float
    boundary_role: str


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv_set(value: str) -> Tuple[str, ...]:
    if value.strip() == "":
        return tuple()
    return tuple(item.strip().lower().replace(" ", "_") for item in value.split(",") if item.strip())


def resolve(path: os.PathLike | str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path)))).resolve()


def synthetic_token(*parts: object, length: int = 17) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf8")).hexdigest()[:length]


def finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def round_float(value: Any, ndigits: int = 4) -> Any:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, ndigits)


def round_xy(value: Sequence[float], ndigits: int = 4) -> List[float]:
    return [round_float(v, ndigits) for v in value]


def coords_of_polygon(poly: Polygon, ndigits: int = 4) -> List[List[float]]:
    if poly.is_empty:
        return []
    return [round_xy(xy, ndigits) for xy in list(poly.exterior.coords)]


def coords_of_line(line: LineString, ndigits: int = 4, max_points: int = 80) -> List[List[float]]:
    if line.is_empty:
        return []
    coords = list(line.coords)
    if len(coords) > max_points:
        step = max(1, int(math.ceil(len(coords) / max_points)))
        coords = coords[::step]
        if coords[-1] != list(line.coords)[-1]:
            coords.append(list(line.coords)[-1])
    return [round_xy(xy, ndigits) for xy in coords]


def quaternion_array(value: Any) -> np.ndarray:
    if isinstance(value, Quaternion):
        return np.asarray(value.q, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def yaw_from_quaternion(value: Any) -> float:
    return float(Quaternion(quaternion_array(value)).yaw_pitch_roll[0])


def get_frame_translation(frame: Dict[str, Any]) -> np.ndarray:
    return np.asarray(frame["ego2global_translation"], dtype=np.float64).copy()


def get_frame_yaw(frame: Dict[str, Any]) -> float:
    return yaw_from_quaternion(frame["ego2global_rotation"])


def rot2d(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def oriented_box_polygon(center_xy: np.ndarray, yaw: float, length: float, width: float) -> Polygon:
    half_l = float(length) / 2.0
    half_w = float(width) / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    world = center_xy[None, :] + (rot2d(float(yaw)) @ local.T).T
    return Polygon(world)


def ego_center_from_rear_axle(rear_xy: np.ndarray, yaw: float, footprint: EgoFootprintParams) -> np.ndarray:
    return np.asarray(rear_xy, dtype=np.float64)[:2] + np.array(
        [np.cos(yaw), np.sin(yaw)], dtype=np.float64
    ) * footprint.rear_axle_to_center


def ego_polygon_from_rear_axle(rear_xy: np.ndarray, yaw: float, footprint: EgoFootprintParams) -> Polygon:
    center = ego_center_from_rear_axle(rear_xy, yaw, footprint)
    return oriented_box_polygon(center, yaw, footprint.length, footprint.width)


def load_ego_footprint() -> EgoFootprintParams:
    try:
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

        params = get_pacifica_parameters()
        return EgoFootprintParams(
            length=float(params.length),
            width=float(params.width),
            rear_axle_to_center=float(params.rear_axle_to_center),
        )
    except Exception:
        return EgoFootprintParams(
            length=DEFAULT_EGO_LENGTH,
            width=DEFAULT_EGO_WIDTH,
            rear_axle_to_center=DEFAULT_REAR_AXLE_TO_CENTER,
        )


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
    out = points.copy()
    out[:, :2] = (rot2d(yaw_rad) @ (out[:, :2] - pivot[:2]).T).T + pivot[:2]
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


def load_candidate(path: Path, index: int, prepend_origin: bool) -> np.ndarray:
    candidate = np.load(resolve(path))
    if candidate.ndim == 3:
        if index < 0 or index >= candidate.shape[0]:
            raise ValueError(
                f"--candidate-index {index} out of range for {path}; valid range is [0, {candidate.shape[0] - 1}]"
            )
        candidate = candidate[index]
    if candidate.ndim != 2 or candidate.shape[1] < 2:
        raise ValueError(f"{path} must have shape (K,T,2+) or (T,2+), got {candidate.shape}")
    candidate = np.asarray(candidate[:, :3] if candidate.shape[1] >= 3 else candidate[:, :2], dtype=np.float64)
    if prepend_origin:
        candidate = np.concatenate([np.zeros((1, candidate.shape[1]), dtype=np.float64), candidate], axis=0)
    return candidate


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

    delta_xy = out_t[current_idx, :2] - translations[current_idx, :2]
    delta_yaw = out_y[current_idx] - yaws[current_idx]
    if current_idx > 0:
        rel = translations[:current_idx, :2] - translations[current_idx, :2]
        out_t[:current_idx, :2] = out_t[current_idx, :2] + (rot2d(delta_yaw) @ rel.T).T
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
        out_t[history_count:, :2] = out_t[current_idx, :2] + (rot2d(delta_yaw) @ rel_future.T).T
        out_t[history_count:, 2] = translations[history_count:, 2] + delta_z
        out_y[history_count:] = yaws[history_count:] + delta_yaw

    return out_t, out_y


def build_offset_indices(
    token_to_idx: Dict[str, int],
    token: str,
    total_frames: int,
    offsets: Sequence[int],
    frame_stride: int,
) -> List[int]:
    if token not in token_to_idx:
        raise KeyError(f"Central token {token} was not found in NAVSIM log frames.")
    center = token_to_idx[token]
    indices = [center + int(offset) * frame_stride for offset in offsets]
    if min(indices) < 0 or max(indices) >= total_frames:
        raise IndexError(
            f"Offsets around {token} are out of range: {min(indices)}..{max(indices)} with {total_frames} frames."
        )
    return indices


def frame_annotations(frame: Dict[str, Any]) -> Dict[str, Any]:
    anns = frame.get("anns", None)
    if isinstance(anns, dict):
        return anns
    annotations = frame.get("annotations", None)
    if isinstance(annotations, dict):
        return {
            "gt_boxes": annotations.get("boxes", np.zeros((0, 7))),
            "gt_names": annotations.get("names", []),
            "gt_velocity_3d": annotations.get("velocity_3d", np.zeros((0, 3))),
            "instance_tokens": annotations.get("instance_tokens", []),
            "track_tokens": annotations.get("track_tokens", []),
        }
    return {
        "gt_boxes": frame.get("gt_boxes", np.zeros((0, 7))),
        "gt_names": frame.get("gt_names", []),
        "gt_velocity_3d": frame.get("gt_velocity_3d", np.zeros((0, 3))),
        "instance_tokens": frame.get("instance_tokens", []),
        "track_tokens": frame.get("track_tokens", []),
    }


def load_navsim_log(navsim_log_path: Path, log_name: str) -> List[Dict[str, Any]]:
    path = resolve(navsim_log_path) / f"{log_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"NAVSIM log file not found: {path}")
    with path.open("rb") as f:
        frames = pickle.load(f)
    if isinstance(frames, dict):
        frames = list(frames.values())
    if not isinstance(frames, list):
        raise TypeError(f"Expected list/dict of frames in {path}, got {type(frames)}")
    return frames


def frame_times_seconds(frames: Sequence[Dict[str, Any]]) -> np.ndarray:
    timestamps = [frame.get("timestamp", None) for frame in frames]
    if all(timestamp is not None for timestamp in timestamps):
        arr = np.asarray(timestamps, dtype=np.float64)
        return (arr - arr[0]) / 1e6
    return np.arange(len(frames), dtype=np.float64) * 0.5


def endpoint_velocity_xy(endpoint_pos: int, times_s: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Estimate endpoint ego velocity from the modified four-frame history trajectory."""
    if len(translations) < 2:
        return np.zeros(2, dtype=np.float64)

    if endpoint_pos <= 0:
        other_pos = 1
        dt = float(times_s[other_pos] - times_s[endpoint_pos])
        delta_xy = translations[other_pos, :2] - translations[endpoint_pos, :2]
    elif endpoint_pos >= len(translations) - 1:
        other_pos = endpoint_pos - 1
        dt = float(times_s[endpoint_pos] - times_s[other_pos])
        delta_xy = translations[endpoint_pos, :2] - translations[other_pos, :2]
    else:
        prev_pos = endpoint_pos - 1
        next_pos = endpoint_pos + 1
        dt = float(times_s[next_pos] - times_s[prev_pos])
        delta_xy = translations[next_pos, :2] - translations[prev_pos, :2]

    if abs(dt) < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(delta_xy, dtype=np.float64) / dt


def projected_ego_polygon_from_endpoint(
    endpoint_pos: int,
    delta_t: float,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
) -> Polygon:
    """Project ego forward from one history endpoint using constant velocity and constant heading."""
    rear_xy = translations[endpoint_pos, :2] + endpoint_velocity_xy(endpoint_pos, times_s, translations) * float(delta_t)
    return ego_polygon_from_rear_axle(rear_xy, float(yaws[endpoint_pos]), footprint)


def projected_object_polygon_from_endpoint(object_state: ObjectState, delta_t: float) -> Polygon:
    """Project a traffic element forward using its endpoint world velocity and constant heading."""
    center = object_state.center + object_state.velocity_xy * float(delta_t)
    return oriented_box_polygon(center, object_state.yaw, object_state.length, object_state.width)


def projected_ego_samples_for_map_query(
    translations: np.ndarray,
    yaws: np.ndarray,
    times_s: np.ndarray,
    endpoint_positions: Sequence[int],
    horizon_s: float,
    step_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Include projected ego samples so static boundary queries cover the TLC lookahead path."""
    sampled_translations = [translations]
    sampled_yaws = [yaws]
    deltas = np.arange(0.0, max(float(horizon_s), 0.0) + step_s * 0.5, step_s, dtype=np.float64)

    for endpoint_pos in endpoint_positions:
        velocity_xy = endpoint_velocity_xy(endpoint_pos, times_s, translations)
        projected = np.repeat(translations[endpoint_pos : endpoint_pos + 1], len(deltas), axis=0)
        projected[:, :2] = translations[endpoint_pos, :2] + deltas[:, None] * velocity_xy[None]
        sampled_translations.append(projected)
        sampled_yaws.append(np.full(len(deltas), float(yaws[endpoint_pos]), dtype=np.float64))

    return np.concatenate(sampled_translations, axis=0), np.concatenate(sampled_yaws, axis=0)


def object_states_for_frame(
    frame: Dict[str, Any],
    frame_index: int,
    dynamic_object_types: Tuple[str, ...],
    excluded_types: Tuple[str, ...],
) -> Dict[str, ObjectState]:
    anns = frame_annotations(frame)
    boxes = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64)
    if boxes.size == 0:
        return {}

    names = np.asarray(anns.get("gt_names", []), dtype=object)
    velocities = np.asarray(anns.get("gt_velocity_3d", np.zeros((len(boxes), 3))), dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[0] != len(boxes):
        velocities = np.zeros((len(boxes), 3), dtype=np.float64)
    track_tokens = np.asarray(anns.get("track_tokens", []), dtype=object)
    instance_tokens = np.asarray(anns.get("instance_tokens", []), dtype=object)

    ego_t = get_frame_translation(frame)
    ego_yaw = get_frame_yaw(frame)
    ego_rot = rot2d(ego_yaw)
    frame_token = str(frame.get("token", frame_index))

    states: Dict[str, ObjectState] = {}
    for idx, box in enumerate(boxes):
        category = str(names[idx] if idx < len(names) else "unknown").lower().replace(" ", "_")
        if category in excluded_types or category not in dynamic_object_types:
            continue
        track_token = str(track_tokens[idx]) if idx < len(track_tokens) else ""
        instance_token = str(instance_tokens[idx]) if idx < len(instance_tokens) else ""
        token = track_token or instance_token or f"{frame_token}:{idx}"
        if token in states:
            token = f"{token}:{idx}"

        local_xy = np.asarray(box[:2], dtype=np.float64)
        center = ego_t[:2] + ego_rot @ local_xy
        yaw = float(box[6] + ego_yaw)
        velocity_xy = ego_rot @ np.asarray(velocities[idx, :2], dtype=np.float64)
        node_type = "veh" if category == "vehicle" else category
        states[token] = ObjectState(
            token=token,
            instance_token=instance_token,
            category=category,
            node_type=node_type,
            center=center,
            yaw=yaw,
            length=float(box[3]),
            width=float(box[4]),
            height=float(box[5]),
            velocity_xy=velocity_xy,
            frame_token=frame_token,
            frame_index=frame_index,
        )
    return states


def risk_from_time_to_event(time_s: float, tau_min: float, alpha: float) -> float:
    if not np.isfinite(time_s):
        return 0.0
    if time_s <= tau_min:
        return 1.0
    return float(np.exp(-alpha * (time_s - tau_min)))


def time_to_collision(
    endpoint_pos: int,
    object_state: ObjectState,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    horizon_s: float,
    step_s: float,
) -> float:
    """
    Endpoint TTC: project ego and one traffic element forward from the endpoint state.

    No recorded future frames are used. If no projected polygon intersection occurs
    inside ``horizon_s``, the returned TTC is infinity.
    """
    deltas = np.arange(0.0, max(float(horizon_s), 0.0) + step_s * 0.5, step_s, dtype=np.float64)
    for delta in deltas:
        ego_poly = projected_ego_polygon_from_endpoint(endpoint_pos, float(delta), times_s, translations, yaws, footprint)
        object_poly = projected_object_polygon_from_endpoint(object_state, float(delta))
        if ego_poly.intersects(object_poly):
            return float(delta)
    return math.inf


def time_to_line_crossing(
    endpoint_pos: int,
    line: LineString,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    horizon_s: float,
    step_s: float,
) -> float:
    """
    Endpoint TLC: project ego forward from the endpoint state.

    No recorded future frames are used. If no ego-boundary intersection occurs
    inside ``horizon_s``, the returned TLC is infinity.
    """
    deltas = np.arange(0.0, max(float(horizon_s), 0.0) + step_s * 0.5, step_s, dtype=np.float64)
    for delta in deltas:
        ego_poly = projected_ego_polygon_from_endpoint(endpoint_pos, float(delta), times_s, translations, yaws, footprint)
        if ego_poly.intersects(line):
            return float(delta)
    return math.inf


def history_interval_horizon(pos: int, endpoint_pos: int, times_s: np.ndarray) -> float:
    if pos >= endpoint_pos:
        return 0.0
    next_pos = min(pos + 1, endpoint_pos)
    return max(float(times_s[next_pos] - times_s[pos]), 0.0)


def dynamic_collision_history(
    endpoint_pos: int,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    states_by_pos: Sequence[Dict[str, ObjectState]],
    step_s: float,
) -> Dict[str, Dict[str, Any]]:
    """Detect ego-object collisions that already occur inside the history window."""
    collisions: Dict[str, Dict[str, Any]] = {}
    for pos in range(0, endpoint_pos + 1):
        horizon_s = history_interval_horizon(pos, endpoint_pos, times_s)
        for token, obj in states_by_pos[pos].items():
            ttc = time_to_collision(
                endpoint_pos=pos,
                object_state=obj,
                times_s=times_s,
                translations=translations,
                yaws=yaws,
                footprint=footprint,
                horizon_s=horizon_s,
                step_s=step_s,
            )
            if not np.isfinite(ttc) or ttc > horizon_s + step_s * 0.5:
                continue
            history_time_s = float(times_s[pos] - times_s[0] + ttc)
            if token in collisions and history_time_s >= float(collisions[token]["history_time_s"]):
                continue
            collisions[token] = {
                "reason": "already_collided_in_history",
                "history_frame_index": int(pos),
                "history_source_token": str(obj.frame_token),
                "history_time_s": history_time_s,
                "history_ttc_s": float(ttc),
            }
    return collisions


def static_violation_history(
    endpoint_pos: int,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    boundary_lines: Sequence[BoundaryLine],
    drivable_union: Optional[Any],
    step_s: float,
) -> Dict[str, Dict[str, Any]]:
    """Detect boundary/drivable-area violations already reached inside history."""
    violations: Dict[str, Dict[str, Any]] = {}

    def remember(boundary: BoundaryLine, pos: int, event_time_s: float, reason: str) -> None:
        history_time_s = float(times_s[pos] - times_s[0] + event_time_s)
        if boundary.token in violations and history_time_s >= float(violations[boundary.token]["history_time_s"]):
            return
        violations[boundary.token] = {
            "reason": reason,
            "history_frame_index": int(pos),
            "history_time_s": history_time_s,
            "history_tlc_s": float(event_time_s),
        }

    for pos in range(0, endpoint_pos + 1):
        horizon_s = history_interval_horizon(pos, endpoint_pos, times_s)
        ego_poly = ego_polygon_from_rear_axle(translations[pos, :2], float(yaws[pos]), footprint)
        if drivable_union is not None and not drivable_union.is_empty and not bool(drivable_union.covers(ego_poly)):
            if boundary_lines:
                nearest = min(boundary_lines, key=lambda item: item.geometry.distance(ego_poly))
                remember(nearest, pos, 0.0, "already_outside_drivable_union_in_history")

        for boundary in boundary_lines:
            tlc = time_to_line_crossing(
                endpoint_pos=pos,
                line=boundary.geometry,
                times_s=times_s,
                translations=translations,
                yaws=yaws,
                footprint=footprint,
                horizon_s=horizon_s,
                step_s=step_s,
            )
            if np.isfinite(tlc) and tlc <= horizon_s + step_s * 0.5:
                remember(boundary, pos, float(tlc), "already_crossed_boundary_in_history")
    return violations


def edge_reason_map(graph: Dict[str, Any]) -> Dict[str, str]:
    return {
        str(edge["target"]): str(edge["reason"])
        for edge in graph.get("edges", [])
        if edge.get("reason") is not None
    }


def transition_event_from_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    if "collid" in reason:
        return "collision"
    if "violation" in reason or "outside" in reason or "boundary" in reason:
        return "violation"
    return None


def build_ego_node(
    graph_frame_index: int,
    endpoint_pos: int,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
) -> Dict[str, Any]:
    rear_xy = translations[endpoint_pos, :2]
    yaw = float(yaws[endpoint_pos])
    center_xy = ego_center_from_rear_axle(rear_xy, yaw, footprint)
    poly = ego_polygon_from_rear_axle(rear_xy, yaw, footprint)
    return {
        "id": "ego",
        "node_type": "ego",
        "history_frame_index": int(graph_frame_index),
        "time_s": round_float(times_s[endpoint_pos] - times_s[0]),
        "rear_axle": round_xy(rear_xy),
        "center": round_xy(center_xy),
        "yaw": round_float(yaw, 6),
        "size": {
            "length": round_float(footprint.length),
            "width": round_float(footprint.width),
            "rear_axle_to_center": round_float(footprint.rear_axle_to_center),
        },
        "polygon": coords_of_polygon(poly),
    }


def build_dynamic_graph(
    graph_frame_index: int,
    endpoint_pos: int,
    frames: Sequence[Dict[str, Any]],
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    states_by_pos: Sequence[Dict[str, ObjectState]],
    collision_history: Optional[Dict[str, Dict[str, Any]]],
    object_radius: float,
    ttc_horizon_s: float,
    risk_step_s: float,
    tau_min: float,
    alpha: float,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    ego_node = build_ego_node(graph_frame_index, endpoint_pos, times_s, translations, yaws, footprint)
    ego_center = np.asarray(ego_node["center"], dtype=np.float64)
    endpoint_objects = states_by_pos[endpoint_pos]
    collision_history = collision_history or {}

    objects_by_token = dict(endpoint_objects)
    for token in collision_history:
        if token in objects_by_token:
            continue
        for pos in range(endpoint_pos, -1, -1):
            if token in states_by_pos[pos]:
                objects_by_token[token] = states_by_pos[pos][token]
                break

    nodes = [ego_node]
    edges = []
    risks_by_token: Dict[str, float] = {}

    for token, obj in sorted(objects_by_token.items(), key=lambda kv: np.linalg.norm(kv[1].center - ego_center)):
        distance = float(np.linalg.norm(obj.center - ego_center))
        collision_info = collision_history.get(token)
        if object_radius > 0 and distance > object_radius and collision_info is None:
            continue
        ttc = time_to_collision(
            endpoint_pos=endpoint_pos,
            object_state=obj,
            times_s=times_s,
            translations=translations,
            yaws=yaws,
            footprint=footprint,
            horizon_s=ttc_horizon_s,
            step_s=risk_step_s,
        )
        reason = None
        if collision_info is not None:
            ttc = 0.0
            reason = str(collision_info["reason"])
        risk = risk_from_time_to_event(ttc, tau_min, alpha)
        risks_by_token[token] = risk

        nodes.append(
            {
                "id": token,
                "node_type": obj.node_type,
                "category": obj.category,
                "track_token": obj.token,
                "instance_token": obj.instance_token,
                "center": round_xy(obj.center),
                "yaw": round_float(obj.yaw, 6),
                "size": {
                    "length": round_float(obj.length),
                    "width": round_float(obj.width),
                    "height": round_float(obj.height),
                },
                "velocity_xy": round_xy(obj.velocity_xy),
                "speed": round_float(np.linalg.norm(obj.velocity_xy)),
                "distance_to_ego": round_float(distance),
                "polygon": coords_of_polygon(obj.polygon()),
            }
        )
        edge = {
            "source": "ego",
            "target": token,
            "edge_type": f"dyn_{obj.node_type}",
            "time_to_collision_s": round_float(finite_or_none(ttc), 4),
            "risk": round_float(risk, 6),
            "distance_to_ego": round_float(distance),
        }
        if reason is not None:
            edge["reason"] = reason
            edge["event_type"] = "collision"
            edge["history_collision_frame_index"] = int(collision_info["history_frame_index"])
            edge["history_collision_source_token"] = str(collision_info["history_source_token"])
            edge["history_collision_time_s"] = round_float(collision_info["history_time_s"], 4)
            edge["history_collision_ttc_s"] = round_float(collision_info["history_ttc_s"], 4)
        edges.append(edge)

    return (
        {
            "graph_type": "dynamic",
            "history_frame_index": int(graph_frame_index),
            "source_token": str(frames[endpoint_pos].get("token", endpoint_pos)),
            "nodes": nodes,
            "edges": edges,
            "risk_params": {
                "ttc_horizon_s": float(ttc_horizon_s),
                "risk_step_s": float(risk_step_s),
                "tau_min": float(tau_min),
                "alpha": float(alpha),
                "projection_model": "constant_velocity_constant_heading_from_endpoint",
            },
        },
        risks_by_token,
    )


def resolve_maps_root() -> Path:
    from nuplan_scripts.utils.constants import NUPLAN_MAPS_ROOT

    candidates: List[Path] = []
    if os.environ.get("NUPLAN_MAPS_ROOT"):
        candidates.append(Path(os.environ["NUPLAN_MAPS_ROOT"]))
    if os.environ.get("OPENSCENE_DATA_ROOT"):
        data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
        candidates.append(data_root / "nuplan" / "dataset" / "maps")
        candidates.append(data_root / "maps")
    candidates.extend(
        [
            Path(NUPLAN_MAPS_ROOT),
            PROJECT_ROOT.parent / "dataset" / "nuplan" / "dataset" / "maps",
            PROJECT_ROOT.parent / "dataset" / "maps",
            PROJECT_ROOT / "data" / "nuplan" / "dataset" / "maps",
            Path("data/nuplan/dataset/maps"),
        ]
    )

    checked: List[str] = []
    for candidate in candidates:
        resolved = resolve(candidate)
        checked.append(os.fspath(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "nuPlan maps root was not found. Set NUPLAN_MAPS_ROOT or OPENSCENE_DATA_ROOT, "
        f"or update nuplan_scripts/configs/common/nuplan_path.yml. Checked: {checked}"
    )


def load_map_api(city: str):
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan_scripts.utils.constants import NUPLAN_MAP_VERSION

    return get_maps_api(os.fspath(resolve_maps_root()), NUPLAN_MAP_VERSION, city)


def flatten_lines(geometry: Any) -> List[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if not line.is_empty]
    if isinstance(geometry, Polygon):
        return [LineString(geometry.exterior.coords)] + [LineString(ring.coords) for ring in geometry.interiors]
    if isinstance(geometry, MultiPolygon):
        lines: List[LineString] = []
        for poly in geometry.geoms:
            lines.extend(flatten_lines(poly))
        return lines
    if isinstance(geometry, GeometryCollection):
        lines = []
        for geom in geometry.geoms:
            lines.extend(flatten_lines(geom))
        return lines
    return []


def extract_boundary_lines(
    map_api: Any,
    city: str,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    boundary_radius: float,
    max_boundaries: int,
) -> Tuple[List[BoundaryLine], Optional[Any], Dict[str, Any]]:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    ego_centers = np.asarray(
        [ego_center_from_rear_axle(t[:2], y, footprint) for t, y in zip(translations, yaws)], dtype=np.float64
    )
    query_center = ego_centers.mean(axis=0)
    query_radius = float(boundary_radius + np.linalg.norm(ego_centers - query_center[None], axis=1).max())

    layer_names = ["ROADBLOCK", "ROADBLOCK_CONNECTOR", "INTERSECTION", "CARPARK_AREA"]
    layers = [getattr(SemanticMapLayer, name) for name in layer_names if hasattr(SemanticMapLayer, name)]
    proximal = map_api.get_proximal_map_objects(
        Point2D(float(query_center[0]), float(query_center[1])),
        query_radius,
        layers,
    )

    polygons = []
    counts_by_layer: Dict[str, int] = {}
    for layer in layers:
        objects = proximal.get(layer, [])
        counts_by_layer[str(layer).split(".")[-1]] = len(objects)
        for map_object in objects:
            polygon = getattr(map_object, "polygon", None)
            if polygon is not None and not polygon.is_empty:
                polygons.append(polygon)

    if not polygons:
        return [], None, {
            "city": city,
            "query_center": round_xy(query_center),
            "query_radius": round_float(query_radius),
            "counts_by_layer": counts_by_layer,
            "warning": "No drivable polygons found around ego trajectory.",
        }

    drivable_union = unary_union(polygons)
    if isinstance(drivable_union, Polygon):
        union_polygons = [drivable_union]
    elif isinstance(drivable_union, MultiPolygon):
        union_polygons = list(drivable_union.geoms)
    else:
        union_polygons = []

    # ``unary_union(...).boundary`` has no ring provenance. Preserve it before
    # flattening: an exterior ring is a drivable-road edge, while an interior
    # ring encloses a non-drivable island such as a barrier.
    boundary_roles: Dict[str, str] = {}
    for polygon in union_polygons:
        exterior = LineString(polygon.exterior.coords)
        boundary_roles[hashlib.sha1(exterior.wkb).hexdigest()[:12]] = "drivable_outer_boundary"
        for interior in polygon.interiors:
            hole = LineString(interior.coords)
            boundary_roles[hashlib.sha1(hole.wkb).hexdigest()[:12]] = "drivable_hole_boundary"

    raw_lines = flatten_lines(drivable_union.boundary)
    if not raw_lines:
        return [], drivable_union, {
            "city": city,
            "query_center": round_xy(query_center),
            "query_radius": round_float(query_radius),
            "counts_by_layer": counts_by_layer,
            "warning": "Drivable polygons produced no boundary lines.",
        }

    window_polys = [
        ego_polygon_from_rear_axle(t[:2], y, footprint).buffer(boundary_radius)
        for t, y in zip(translations, yaws)
    ]
    window_geom = unary_union(window_polys)

    boundary_lines: List[BoundaryLine] = []
    seen = set()
    for line in raw_lines:
        if line.length <= 1e-3:
            continue
        dist = float(line.distance(window_geom))
        if dist > boundary_radius:
            continue
        digest = hashlib.sha1(line.wkb).hexdigest()[:12]
        if digest in seen:
            continue
        seen.add(digest)
        boundary_lines.append(
            BoundaryLine(
                token=f"bd_{digest}",
                geometry=line,
                length=float(line.length),
                distance_to_window=dist,
                boundary_role=boundary_roles.get(digest, "unclassified_union_boundary"),
            )
        )

    boundary_lines.sort(key=lambda item: (item.distance_to_window, item.token))
    if max_boundaries > 0:
        boundary_lines = boundary_lines[:max_boundaries]

    return boundary_lines, drivable_union, {
        "city": city,
        "query_center": round_xy(query_center),
        "query_radius": round_float(query_radius),
        "counts_by_layer": counts_by_layer,
        "num_union_boundary_lines": len(raw_lines),
        "num_selected_boundary_lines": len(boundary_lines),
    }


def build_static_graph(
    graph_frame_index: int,
    endpoint_pos: int,
    frames: Sequence[Dict[str, Any]],
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    boundary_lines: Sequence[BoundaryLine],
    drivable_union: Optional[Any],
    violation_history: Optional[Dict[str, Dict[str, Any]]],
    boundary_radius: float,
    tlc_horizon_s: float,
    risk_step_s: float,
    tau_min: float,
    alpha: float,
) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
    ego_node = build_ego_node(graph_frame_index, endpoint_pos, times_s, translations, yaws, footprint)
    ego_rear_xy = translations[endpoint_pos, :2]
    ego_yaw = float(yaws[endpoint_pos])
    ego_poly = ego_polygon_from_rear_axle(ego_rear_xy, ego_yaw, footprint)
    violation_history = violation_history or {}

    instant_violation = False
    if drivable_union is not None and not drivable_union.is_empty:
        instant_violation = not bool(drivable_union.covers(ego_poly))

    if boundary_lines:
        nearest_token = min(boundary_lines, key=lambda item: item.geometry.distance(ego_poly)).token
    else:
        nearest_token = None

    nodes = [ego_node]
    edges = []
    risks_by_token: Dict[str, float] = {}

    for boundary in boundary_lines:
        distance = float(boundary.geometry.distance(ego_poly))
        tlc = time_to_line_crossing(
            endpoint_pos=endpoint_pos,
            line=boundary.geometry,
            times_s=times_s,
            translations=translations,
            yaws=yaws,
            footprint=footprint,
            horizon_s=tlc_horizon_s,
            step_s=risk_step_s,
        )
        reason = None
        violation_info = violation_history.get(boundary.token)
        if violation_info is not None:
            tlc = 0.0
            reason = str(violation_info["reason"])
        elif instant_violation and boundary.token == nearest_token and not np.isfinite(tlc):
            tlc = 0.0
            reason = "already_outside_drivable_union_nearest_boundary"
        if (
            distance > boundary_radius
            and not np.isfinite(tlc)
            and not (instant_violation and boundary.token == nearest_token)
            and violation_info is None
        ):
            continue
        risk = risk_from_time_to_event(tlc, tau_min, alpha)
        risks_by_token[boundary.token] = risk

        nodes.append(
            {
                "id": boundary.token,
                "node_type": "bd",
                "length": round_float(boundary.length),
                "distance_to_ego": round_float(distance),
                "distance_to_window": round_float(boundary.distance_to_window),
                "boundary_role": boundary.boundary_role,
                "line": coords_of_line(boundary.geometry),
            }
        )
        edge = {
            "source": "ego",
            "target": boundary.token,
            "edge_type": "sta_bd",
            "time_to_line_crossing_s": round_float(finite_or_none(tlc), 4),
            "risk": round_float(risk, 6),
            "distance_to_ego": round_float(distance),
        }
        if reason is not None:
            edge["reason"] = reason
            edge["event_type"] = "violation"
            if violation_info is not None:
                edge["history_violation_frame_index"] = int(violation_info["history_frame_index"])
                edge["history_violation_time_s"] = round_float(violation_info["history_time_s"], 4)
                edge["history_violation_tlc_s"] = round_float(violation_info["history_tlc_s"], 4)
        edges.append(edge)

    return (
        {
            "graph_type": "static",
            "history_frame_index": int(graph_frame_index),
            "source_token": str(frames[endpoint_pos].get("token", endpoint_pos)),
            "instant_drivable_violation": bool(instant_violation),
            "nodes": nodes,
            "edges": edges,
            "risk_params": {
                "tlc_horizon_s": float(tlc_horizon_s),
                "risk_step_s": float(risk_step_s),
                "tau_min": float(tau_min),
                "alpha": float(alpha),
                "projection_model": "constant_velocity_constant_heading_from_endpoint",
            },
        },
        risks_by_token,
        instant_violation,
    )


def build_transition_graph(
    graph_type: str,
    risk0: Dict[str, float],
    risk3: Dict[str, float],
    source_node_type: str,
    reason0: Optional[Dict[str, str]] = None,
    reason3: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    reason0 = reason0 or {}
    reason3 = reason3 or {}
    tokens = sorted(set(risk0) | set(risk3))
    edges = []
    positive_deltas: Dict[str, float] = {}
    for token in tokens:
        r0 = float(risk0.get(token, 0.0))
        r3 = float(risk3.get(token, 0.0))
        delta = r3 - r0
        positive_deltas[token] = max(0.0, delta)
        if delta > 1e-6:
            trend = "increases"
        elif delta < -1e-6:
            trend = "decreases"
        else:
            trend = "stable"
        edge = {
            "source": "ego",
            "target": token,
            "edge_type": f"delta_{graph_type}",
            "target_node_type": source_node_type,
            "risk_0": round_float(r0, 6),
            "risk_3": round_float(r3, 6),
            "delta_risk": round_float(delta, 6),
            "positive_delta_risk": round_float(max(0.0, delta), 6),
            "trend": trend,
        }
        if token in reason0:
            edge["reason_0"] = reason0[token]
        if token in reason3:
            edge["reason_3"] = reason3[token]
            event = transition_event_from_reason(reason3[token])
            if event is not None:
                edge["terminal_event"] = event
        edges.append(edge)

    return (
        {
            "graph_type": f"transition_{graph_type}",
            "from_history_frame_index": 0,
            "to_history_frame_index": 3,
            "nodes": [{"id": "ego", "node_type": "ego"}]
            + [{"id": token, "node_type": source_node_type} for token in tokens],
            "edges": edges,
        },
        positive_deltas,
    )



@dataclass
class RiskEvalContext:
    config: FrameCentralConfig
    config_path: Path
    central_token: str
    frames: Sequence[Dict[str, Any]]
    current_idx: int
    start_endpoint_pos: int
    end_endpoint_pos: int
    times_s: np.ndarray
    dynamic_object_types: Tuple[str, ...]
    excluded_types: Tuple[str, ...]
    states_by_pos: Sequence[Dict[str, ObjectState]]
    footprint: EgoFootprintParams
    map_api: Optional[Any]


def build_risk_eval_context(
    config: FrameCentralConfig,
    config_path: Path,
    central_token: str,
    log_frames: Sequence[Dict[str, Any]],
    footprint: EgoFootprintParams,
    map_api: Optional[Any],
    args: argparse.Namespace,
) -> RiskEvalContext:
    token_to_idx = {str(frame["token"]): idx for idx, frame in enumerate(log_frames)}
    history_offsets = list(
        range(
            args.scene_current_offset_frames - (args.history_frames - 1),
            args.scene_current_offset_frames + 1,
        )
    )
    trajectory_offsets = list(range(min(0, history_offsets[0]), max(0, history_offsets[-1]) + 1))
    trajectory_indices = build_offset_indices(
        token_to_idx=token_to_idx,
        token=central_token,
        total_frames=len(log_frames),
        offsets=trajectory_offsets,
        frame_stride=args.frame_stride,
    )
    frames = [log_frames[idx] for idx in trajectory_indices]
    current_idx = trajectory_offsets.index(0)
    start_endpoint_pos = trajectory_offsets.index(history_offsets[0])
    end_endpoint_pos = trajectory_offsets.index(history_offsets[-1])

    if args.history_frames != 4:
        raise ValueError("This risk-only formulation expects --history-frames 4, i.e. frame indices 0..3.")

    dynamic_object_types = parse_csv_set(args.dynamic_object_types)
    excluded_types = parse_csv_set(args.excluded_object_types)
    states_by_pos = [
        object_states_for_frame(
            frame,
            frame_index=idx,
            dynamic_object_types=dynamic_object_types,
            excluded_types=excluded_types,
        )
        for idx, frame in enumerate(frames)
    ]
    return RiskEvalContext(
        config=config,
        config_path=config_path,
        central_token=central_token,
        frames=frames,
        current_idx=current_idx,
        start_endpoint_pos=start_endpoint_pos,
        end_endpoint_pos=end_endpoint_pos,
        times_s=frame_times_seconds(frames),
        dynamic_object_types=dynamic_object_types,
        excluded_types=excluded_types,
        states_by_pos=states_by_pos,
        footprint=footprint,
        map_api=map_api,
    )


def dynamic_risk_summary(
    endpoint_pos: int,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    states_by_pos: Sequence[Dict[str, ObjectState]],
    collision_history: Optional[Dict[str, Dict[str, Any]]],
    object_radius: float,
    ttc_horizon_s: float,
    risk_step_s: float,
    tau_min: float,
    alpha: float,
) -> Tuple[Dict[str, float], Dict[str, str], int]:
    ego_center = ego_center_from_rear_axle(translations[endpoint_pos, :2], float(yaws[endpoint_pos]), footprint)
    endpoint_objects = states_by_pos[endpoint_pos]
    collision_history = collision_history or {}
    objects_by_token = dict(endpoint_objects)
    for token in collision_history:
        if token in objects_by_token:
            continue
        for pos in range(endpoint_pos, -1, -1):
            if token in states_by_pos[pos]:
                objects_by_token[token] = states_by_pos[pos][token]
                break

    risks_by_token: Dict[str, float] = {}
    reasons_by_token: Dict[str, str] = {}
    num_edges = 0
    for token, obj in sorted(objects_by_token.items(), key=lambda kv: np.linalg.norm(kv[1].center - ego_center)):
        distance = float(np.linalg.norm(obj.center - ego_center))
        collision_info = collision_history.get(token)
        if object_radius > 0 and distance > object_radius and collision_info is None:
            continue
        ttc = time_to_collision(
            endpoint_pos=endpoint_pos,
            object_state=obj,
            times_s=times_s,
            translations=translations,
            yaws=yaws,
            footprint=footprint,
            horizon_s=ttc_horizon_s,
            step_s=risk_step_s,
        )
        if collision_info is not None:
            ttc = 0.0
            reasons_by_token[token] = str(collision_info["reason"])
        risks_by_token[token] = risk_from_time_to_event(ttc, tau_min, alpha)
        num_edges += 1
    return risks_by_token, reasons_by_token, num_edges


def static_risk_summary(
    endpoint_pos: int,
    times_s: np.ndarray,
    translations: np.ndarray,
    yaws: np.ndarray,
    footprint: EgoFootprintParams,
    boundary_lines: Sequence[BoundaryLine],
    drivable_union: Optional[Any],
    violation_history: Optional[Dict[str, Dict[str, Any]]],
    boundary_radius: float,
    tlc_horizon_s: float,
    risk_step_s: float,
    tau_min: float,
    alpha: float,
) -> Tuple[Dict[str, float], Dict[str, str], int]:
    ego_poly = ego_polygon_from_rear_axle(translations[endpoint_pos, :2], float(yaws[endpoint_pos]), footprint)
    violation_history = violation_history or {}
    instant_violation = False
    if drivable_union is not None and not drivable_union.is_empty:
        instant_violation = not bool(drivable_union.covers(ego_poly))
    nearest_token = min(boundary_lines, key=lambda item: item.geometry.distance(ego_poly)).token if boundary_lines else None

    risks_by_token: Dict[str, float] = {}
    reasons_by_token: Dict[str, str] = {}
    num_edges = 0
    for boundary in boundary_lines:
        distance = float(boundary.geometry.distance(ego_poly))
        tlc = time_to_line_crossing(
            endpoint_pos=endpoint_pos,
            line=boundary.geometry,
            times_s=times_s,
            translations=translations,
            yaws=yaws,
            footprint=footprint,
            horizon_s=tlc_horizon_s,
            step_s=risk_step_s,
        )
        violation_info = violation_history.get(boundary.token)
        if violation_info is not None:
            tlc = 0.0
            reasons_by_token[boundary.token] = str(violation_info["reason"])
        elif instant_violation and boundary.token == nearest_token and not np.isfinite(tlc):
            tlc = 0.0
            reasons_by_token[boundary.token] = "already_outside_drivable_union_nearest_boundary"
        if (
            distance > boundary_radius
            and not np.isfinite(tlc)
            and not (instant_violation and boundary.token == nearest_token)
            and violation_info is None
        ):
            continue
        risks_by_token[boundary.token] = risk_from_time_to_event(tlc, tau_min, alpha)
        num_edges += 1
    return risks_by_token, reasons_by_token, num_edges


def risk_delta_summary(
    risk0: Dict[str, float],
    risk3: Dict[str, float],
    reason0: Dict[str, str],
    reason3: Dict[str, str],
) -> Dict[str, Any]:
    best_positive = 0.0
    best_target = None
    best_delta = 0.0
    hard_events: List[str] = []
    hard_targets: List[str] = []
    hard_reasons: List[str] = []
    tokens = sorted(set(risk0) | set(risk3))
    for token in tokens:
        delta = float(risk3.get(token, 0.0)) - float(risk0.get(token, 0.0))
        rounded_delta = round_float(delta, 6)
        positive = round_float(max(0.0, delta), 6)
        if positive > best_positive:
            best_positive = float(positive)
            best_target = str(token)
            best_delta = float(rounded_delta)
        if token in reason0:
            event = transition_event_from_reason(reason0[token])
            if event is not None:
                hard_events.append(event)
                hard_targets.append(str(token))
                hard_reasons.append(f"reason_0={reason0[token]}")
        if token in reason3:
            event = transition_event_from_reason(reason3[token])
            if event is not None:
                hard_events.append(event)
                hard_targets.append(str(token))
                hard_reasons.append(f"reason_3={reason3[token]}")
                hard_reasons.append(f"terminal_event={event}")
    return {
        "R": float(best_positive),
        "target": best_target,
        "delta": float(best_delta),
        "num_edges": len(tokens),
        "hard_events": sorted(set(hard_events)),
        "hard_targets": sorted(set(hard_targets)),
        "hard_reasons": sorted(set(hard_reasons)),
    }


def evaluate_risk_for_context(
    context: RiskEvalContext,
    candidate: np.ndarray,
    args: argparse.Namespace,
    candidate_index: int,
) -> Dict[str, Any]:
    translations, yaws = compute_modified_trajectory(context.frames, context.current_idx, candidate, args)
    times_s = context.times_s

    dyn_collision0 = dynamic_collision_history(
        endpoint_pos=context.start_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        states_by_pos=context.states_by_pos,
        step_s=args.risk_step,
    )
    dyn_collision3 = dynamic_collision_history(
        endpoint_pos=context.end_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        states_by_pos=context.states_by_pos,
        step_s=args.risk_step,
    )
    dyn_risk0, dyn_reason0, _ = dynamic_risk_summary(
        endpoint_pos=context.start_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        states_by_pos=context.states_by_pos,
        collision_history=dyn_collision0,
        object_radius=args.object_radius,
        ttc_horizon_s=args.ttc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.ttc_tau_min,
        alpha=args.ttc_alpha,
    )
    dyn_risk3, dyn_reason3, _ = dynamic_risk_summary(
        endpoint_pos=context.end_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        states_by_pos=context.states_by_pos,
        collision_history=dyn_collision3,
        object_radius=args.object_radius,
        ttc_horizon_s=args.ttc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.ttc_tau_min,
        alpha=args.ttc_alpha,
    )

    boundary_lines: List[BoundaryLine] = []
    drivable_union = None
    if not args.skip_static_graph and context.map_api is not None:
        map_query_translations, map_query_yaws = projected_ego_samples_for_map_query(
            translations=translations,
            yaws=yaws,
            times_s=times_s,
            endpoint_positions=[context.start_endpoint_pos, context.end_endpoint_pos],
            horizon_s=args.tlc_horizon,
            step_s=args.risk_step,
        )
        boundary_lines, drivable_union, _ = extract_boundary_lines(
            map_api=context.map_api,
            city=context.config.city,
            translations=map_query_translations,
            yaws=map_query_yaws,
            footprint=context.footprint,
            boundary_radius=args.boundary_radius,
            max_boundaries=args.max_boundaries,
        )

    sta_violation0 = static_violation_history(
        endpoint_pos=context.start_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        step_s=args.risk_step,
    )
    sta_violation3 = static_violation_history(
        endpoint_pos=context.end_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        step_s=args.risk_step,
    )
    sta_risk0, sta_reason0, _ = static_risk_summary(
        endpoint_pos=context.start_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        violation_history=sta_violation0,
        boundary_radius=args.boundary_radius,
        tlc_horizon_s=args.tlc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.tlc_tau_min,
        alpha=args.tlc_alpha,
    )
    sta_risk3, sta_reason3, _ = static_risk_summary(
        endpoint_pos=context.end_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=context.footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        violation_history=sta_violation3,
        boundary_radius=args.boundary_radius,
        tlc_horizon_s=args.tlc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.tlc_tau_min,
        alpha=args.tlc_alpha,
    )

    dyn = risk_delta_summary(dyn_risk0, dyn_risk3, dyn_reason0, dyn_reason3)
    sta = risk_delta_summary(sta_risk0, sta_risk3, sta_reason0, sta_reason3)
    hard_events = sorted(set(dyn["hard_events"]) | set(sta["hard_events"]))
    hard_targets = sorted(set(dyn["hard_targets"]) | set(sta["hard_targets"]))
    hard_reasons = sorted(set(dyn["hard_reasons"]) | set(sta["hard_reasons"]))
    raw_score = max(float(dyn["R"]), float(sta["R"]))
    is_valid = len(hard_events) == 0
    if float(dyn["R"]) >= float(sta["R"]):
        source = "dynamic"
        target = dyn["target"]
        edge_delta = dyn["delta"]
    else:
        source = "static"
        target = sta["target"]
        edge_delta = sta["delta"]

    graph_token = synthetic_token(
        context.config.road_block_name,
        context.central_token,
        resolve(args.candidate_trajectory_path),
        int(candidate_index),
        args.candidate_replace_mode,
        args.candidate_history_duration,
        args.candidate_z_mode,
        args.scene_current_offset_frames,
    )
    return {
        "R_dyn": float(dyn["R"]),
        "R_sta": float(sta["R"]),
        "score_raw": float(raw_score),
        "score_valid": float(raw_score if is_valid else 0.0),
        "is_valid": bool(is_valid),
        "D_hard": 0 if is_valid else 1,
        "hard_event": "+".join(hard_events),
        "hard_targets": hard_targets,
        "hard_reasons": hard_reasons,
        "risk_source": source,
        "risk_target": target,
        "risk_edge_delta": float(edge_delta),
        "num_dynamic_edges": int(dyn["num_edges"]),
        "num_static_edges": int(sta["num_edges"]),
        "graph_token": graph_token,
    }

def load_configs(configs: Sequence[Path], config_glob: Optional[str]) -> List[Path]:
    paths = [resolve(path) for path in configs]
    if config_glob:
        paths.extend(resolve(path) for path in glob.glob(os.path.expandvars(os.path.expanduser(config_glob))))
    unique = []
    seen = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError("No config files were provided/found.")
    return unique


def selected_tokens_for_config(config: FrameCentralConfig, requested_tokens: Optional[Sequence[str]]) -> List[str]:
    if requested_tokens:
        requested = [str(token) for token in requested_tokens]
        available = set(config.central_tokens)
        missing = [token for token in requested if token not in available]
        if missing:
            # Some configs are single-token wrappers whose central token is only
            # encoded in the filename/road_block_name. Allow those tokens if the
            # log contains them; the log lookup will be the final authority.
            pass
        return requested
    return [str(token) for token in config.central_tokens]


def compute_modified_trajectory(
    frames: Sequence[Dict[str, Any]],
    current_idx: int,
    candidate: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    translations = np.asarray([get_frame_translation(frame) for frame in frames], dtype=np.float64)
    yaws = np.asarray([get_frame_yaw(frame) for frame in frames], dtype=np.float64)
    translations, yaws = apply_rigid_offset(
        translations=translations,
        yaws=yaws,
        current_idx=current_idx,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        yaw_deg=args.yaw_deg,
        pivot_mode=args.pivot,
    )
    if args.candidate_replace_mode == "history_first_half":
        translations, yaws = apply_candidate_history_first_half(
            translations=translations,
            yaws=yaws,
            current_idx=current_idx,
            candidate=candidate,
            candidate_frame_rate=args.candidate_frame_rate,
            candidate_prepend_origin=args.candidate_prepend_origin,
            candidate_use_yaw=args.candidate_use_yaw,
            candidate_z_mode=args.candidate_z_mode,
            candidate_history_duration=args.candidate_history_duration,
        )
    elif args.candidate_replace_mode == "future":
        translations, yaws = apply_candidate(
            translations=translations,
            yaws=yaws,
            current_idx=current_idx,
            candidate=candidate,
            candidate_frame_rate=args.candidate_frame_rate,
            candidate_prepend_origin=args.candidate_prepend_origin,
            candidate_use_yaw=args.candidate_use_yaw,
            candidate_z_mode=args.candidate_z_mode,
        )
    else:
        raise ValueError(f"Unknown candidate replacement mode: {args.candidate_replace_mode}")
    return translations, yaws


def build_scene_graph_for_token(
    config: FrameCentralConfig,
    config_path: Path,
    central_token: str,
    log_frames: Sequence[Dict[str, Any]],
    candidate: np.ndarray,
    footprint: EgoFootprintParams,
    map_api: Optional[Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    token_to_idx = {str(frame["token"]): idx for idx, frame in enumerate(log_frames)}
    history_offsets = list(
        range(
            args.scene_current_offset_frames - (args.history_frames - 1),
            args.scene_current_offset_frames + 1,
        )
    )
    trajectory_offsets = list(range(min(0, history_offsets[0]), max(0, history_offsets[-1]) + 1))
    trajectory_indices = build_offset_indices(
        token_to_idx=token_to_idx,
        token=central_token,
        total_frames=len(log_frames),
        offsets=trajectory_offsets,
        frame_stride=args.frame_stride,
    )
    frames = [log_frames[idx] for idx in trajectory_indices]
    current_idx = trajectory_offsets.index(0)
    start_endpoint_pos = trajectory_offsets.index(history_offsets[0])
    end_endpoint_pos = trajectory_offsets.index(history_offsets[-1])

    if args.history_frames != 4:
        raise ValueError("This scene-graph formulation currently expects --history-frames 4, i.e. frame indices 0..3.")

    translations, yaws = compute_modified_trajectory(frames, current_idx, candidate, args)
    times_s = frame_times_seconds(frames)

    dynamic_object_types = parse_csv_set(args.dynamic_object_types)
    excluded_types = parse_csv_set(args.excluded_object_types)
    states_by_pos = [
        object_states_for_frame(
            frame,
            frame_index=idx,
            dynamic_object_types=dynamic_object_types,
            excluded_types=excluded_types,
        )
        for idx, frame in enumerate(frames)
    ]
    dyn_collision0 = dynamic_collision_history(
        endpoint_pos=start_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        states_by_pos=states_by_pos,
        step_s=args.risk_step,
    )
    dyn_collision3 = dynamic_collision_history(
        endpoint_pos=end_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        states_by_pos=states_by_pos,
        step_s=args.risk_step,
    )

    dyn0, dyn_risk0 = build_dynamic_graph(
        graph_frame_index=0,
        endpoint_pos=start_endpoint_pos,
        frames=frames,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        states_by_pos=states_by_pos,
        collision_history=dyn_collision0,
        object_radius=args.object_radius,
        ttc_horizon_s=args.ttc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.ttc_tau_min,
        alpha=args.ttc_alpha,
    )
    dyn3, dyn_risk3 = build_dynamic_graph(
        graph_frame_index=3,
        endpoint_pos=end_endpoint_pos,
        frames=frames,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        states_by_pos=states_by_pos,
        collision_history=dyn_collision3,
        object_radius=args.object_radius,
        ttc_horizon_s=args.ttc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.ttc_tau_min,
        alpha=args.ttc_alpha,
    )

    boundary_lines: List[BoundaryLine] = []
    drivable_union = None
    map_metadata: Dict[str, Any] = {"static_graph_enabled": not args.skip_static_graph}
    if not args.skip_static_graph and map_api is not None:
        map_query_translations, map_query_yaws = projected_ego_samples_for_map_query(
            translations=translations,
            yaws=yaws,
            times_s=times_s,
            endpoint_positions=[start_endpoint_pos, end_endpoint_pos],
            horizon_s=args.tlc_horizon,
            step_s=args.risk_step,
        )
        boundary_lines, drivable_union, map_metadata = extract_boundary_lines(
            map_api=map_api,
            city=config.city,
            translations=map_query_translations,
            yaws=map_query_yaws,
            footprint=footprint,
            boundary_radius=args.boundary_radius,
            max_boundaries=args.max_boundaries,
        )
        map_metadata["static_graph_enabled"] = True

    sta_violation_history0 = static_violation_history(
        endpoint_pos=start_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        step_s=args.risk_step,
    )
    sta_violation_history3 = static_violation_history(
        endpoint_pos=end_endpoint_pos,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        step_s=args.risk_step,
    )

    sta0, sta_risk0, sta_violation0 = build_static_graph(
        graph_frame_index=0,
        endpoint_pos=start_endpoint_pos,
        frames=frames,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        violation_history=sta_violation_history0,
        boundary_radius=args.boundary_radius,
        tlc_horizon_s=args.tlc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.tlc_tau_min,
        alpha=args.tlc_alpha,
    )
    sta3, sta_risk3, _ = build_static_graph(
        graph_frame_index=3,
        endpoint_pos=end_endpoint_pos,
        frames=frames,
        times_s=times_s,
        translations=translations,
        yaws=yaws,
        footprint=footprint,
        boundary_lines=boundary_lines,
        drivable_union=drivable_union,
        violation_history=sta_violation_history3,
        boundary_radius=args.boundary_radius,
        tlc_horizon_s=args.tlc_horizon,
        risk_step_s=args.risk_step,
        tau_min=args.tlc_tau_min,
        alpha=args.tlc_alpha,
    )

    delta_dyn, _ = build_transition_graph(
        "dyn",
        dyn_risk0,
        dyn_risk3,
        "veh",
        reason0=edge_reason_map(dyn0),
        reason3=edge_reason_map(dyn3),
    )
    delta_sta, _ = build_transition_graph(
        "sta",
        sta_risk0,
        sta_risk3,
        "bd",
        reason0=edge_reason_map(sta0),
        reason3=edge_reason_map(sta3),
    )

    graph_token = synthetic_token(
        config.road_block_name,
        central_token,
        resolve(args.candidate_trajectory_path),
        args.candidate_index,
        args.candidate_replace_mode,
        args.candidate_history_duration,
        args.candidate_z_mode,
        args.scene_current_offset_frames,
    )

    dynamic_graph_metadata = {
        "object_types": list(dynamic_object_types),
        "excluded_object_types": list(excluded_types),
        "vehicle_only": tuple(dynamic_object_types) == ("vehicle",),
        "node_type_mapping": {"vehicle": "veh"},
    }
    risk_params = {
        "dynamic_ttc": {
            "horizon_s": float(args.ttc_horizon),
            "tau_min": float(args.ttc_tau_min),
            "alpha": float(args.ttc_alpha),
        },
        "static_tlc": {
            "horizon_s": float(args.tlc_horizon),
            "tau_min": float(args.tlc_tau_min),
            "alpha": float(args.tlc_alpha),
        },
        "risk_step_s": float(args.risk_step),
        "projection_model": "constant_velocity_constant_heading_from_endpoint",
    }
    transition_graphs = {
        "G_delta_dyn_0_to_3": delta_dyn,
        "G_delta_sta_0_to_3": delta_sta,
    }

    if args.output_detail == "full":
        return {
            "schema_version": "safety_scene_graph_v1",
            "graph_token": graph_token,
            "config_path": os.fspath(config_path),
            "road_block_name": config.road_block_name,
            "log_name": config.central_log,
            "map_name": config.city,
            "central_token": central_token,
            "candidate": {
                "trajectory_path": os.fspath(resolve(args.candidate_trajectory_path)),
                "index": int(args.candidate_index),
                "replace_mode": args.candidate_replace_mode,
                "history_duration_s": float(args.candidate_history_duration),
                "frame_rate_hz": float(args.candidate_frame_rate),
                "prepend_origin": bool(args.candidate_prepend_origin),
                "use_yaw": bool(args.candidate_use_yaw),
                "z_mode": args.candidate_z_mode,
            },
            "window": {
                "history_frames": int(args.history_frames),
                "frame_stride": int(args.frame_stride),
                "scene_current_offset_frames": int(args.scene_current_offset_frames),
                "trajectory_offsets": [int(x) for x in trajectory_offsets],
                "trajectory_source_tokens": [str(frame.get("token", "")) for frame in frames],
                "graph_endpoint_positions": {
                    "0": int(start_endpoint_pos),
                    "3": int(end_endpoint_pos),
                },
            },
            "dynamic_graph_metadata": dynamic_graph_metadata,
            "risk_params": risk_params,
            "map_metadata": map_metadata,
            "graphs": {
                "G_dyn_0": dyn0,
                "G_dyn_3": dyn3,
                "G_sta_0": sta0,
                "G_sta_3": sta3,
                **transition_graphs,
            },
        }

    return {
        "schema_version": "safety_scene_graph_transition_v1",
        "graph_token": graph_token,
        "road_block_name": config.road_block_name,
        "log_name": config.central_log,
        "map_name": config.city,
        "central_token": central_token,
        "candidate": {
            "trajectory_file": resolve(args.candidate_trajectory_path).name,
            "index": int(args.candidate_index),
            "replace_mode": args.candidate_replace_mode,
            "history_duration_s": float(args.candidate_history_duration),
            "z_mode": args.candidate_z_mode,
        },
        "window": {
            "history_frames": int(args.history_frames),
            "frame_stride": int(args.frame_stride),
            "scene_current_offset_frames": int(args.scene_current_offset_frames),
            "source_tokens": {
                "0": str(frames[start_endpoint_pos].get("token", "")),
                "3": str(frames[end_endpoint_pos].get("token", "")),
            },
        },
        "dynamic_graph_metadata": dynamic_graph_metadata,
        "risk_params": risk_params,
        "graphs": transition_graphs,
    }


def write_json(path: Path, data: Dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", type=Path, nargs="*", default=[], help="One or more FrameCentralConfig YAML files.")
    parser.add_argument("--config-glob", type=str, default=None, help="Optional glob for many FrameCentralConfig YAML files.")
    parser.add_argument("--tokens", nargs="*", default=None, help="Central tokens. Defaults to each config's central_tokens.")
    parser.add_argument("--navsim-log-path", required=True, type=Path, help="Directory containing NAVSIM log pickles.")

    parser.add_argument("--candidate-trajectory-path", required=True, type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--candidate-frame-rate", type=float, default=10.0)
    parser.add_argument("--candidate-prepend-origin", type=str_to_bool, default=True)
    parser.add_argument("--candidate-use-yaw", type=str_to_bool, default=True)
    parser.add_argument("--candidate-replace-mode", choices=("future", "history_first_half"), default="history_first_half")
    parser.add_argument("--candidate-history-duration", type=float, default=1.5)
    parser.add_argument("--candidate-z-mode", choices=("distance", "constant", "time"), default="time")

    parser.add_argument("--history-frames", type=int, default=4)
    # Deprecated compatibility option. Graph generation is history-only; this value is ignored.
    parser.add_argument("--future-frames", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--scene-current-offset-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)

    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pivot", choices=("current", "start", "center", "origin"), default="current")

    parser.add_argument("--object-radius", type=float, default=80.0)
    parser.add_argument("--boundary-radius", type=float, default=60.0)
    parser.add_argument("--max-boundaries", type=int, default=128)
    parser.add_argument(
        "--dynamic-object-types",
        type=str,
        default=",".join(DEFAULT_DYNAMIC_OBJECT_TYPES),
        help="Comma-separated annotation categories included in dynamic TTC graphs. Defaults to vehicle-only.",
    )
    parser.add_argument(
        "--mobile-object-types",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--excluded-object-types", type=str, default=",".join(DEFAULT_EXCLUDED_TYPES))
    parser.add_argument("--extrapolate-missing-objects", type=str_to_bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--skip-static-graph", type=str_to_bool, default=False)

    parser.add_argument("--ttc-horizon", type=float, default=7.0)
    parser.add_argument("--tlc-horizon", type=float, default=4.0)
    parser.add_argument("--risk-step", type=float, default=0.1)
    parser.add_argument("--ttc-tau-min", type=float, default=1.0)
    parser.add_argument("--tlc-tau-min", type=float, default=1.0)
    parser.add_argument("--ttc-alpha", type=float, default=0.30)
    parser.add_argument("--tlc-alpha", type=float, default=0.30)

    parser.add_argument(
        "--output-detail",
        choices=("transition", "full"),
        default="transition",
        help="Saved JSON detail. 'transition' stores only compact risk-transition graphs; 'full' also stores endpoint geometry graphs.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("Hierarchical Greedy Search/scene_graphs"))
    parser.add_argument("--overwrite", type=str_to_bool, default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.risk_step <= 0:
        raise ValueError("--risk-step must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.mobile_object_types is not None:
        # Backward-compatible alias for older commands. The dynamic graph now
        # only contains categories explicitly listed here; for the vehicle-only
        # setting, leave this unspecified or pass --dynamic-object-types vehicle.
        args.dynamic_object_types = args.mobile_object_types

    config_paths = load_configs(args.config, args.config_glob)
    candidate = load_candidate(
        path=args.candidate_trajectory_path,
        index=args.candidate_index,
        prepend_origin=args.candidate_prepend_origin,
    )
    footprint = load_ego_footprint()

    output_root = resolve(args.output_root)
    num_written = 0
    log_cache: Dict[str, List[Dict[str, Any]]] = {}
    map_cache: Dict[str, Any] = {}

    for config_path in config_paths:
        config = load_config(os.fspath(config_path))
        if not isinstance(config, FrameCentralConfig):
            raise TypeError(f"{config_path} is not a FrameCentralConfig.")

        if config.central_log not in log_cache:
            log_cache[config.central_log] = load_navsim_log(args.navsim_log_path, config.central_log)
        log_frames = log_cache[config.central_log]

        map_api = None
        if not args.skip_static_graph:
            if config.city not in map_cache:
                map_cache[config.city] = load_map_api(config.city)
            map_api = map_cache[config.city]

        for central_token in selected_tokens_for_config(config, args.tokens):
            graph = build_scene_graph_for_token(
                config=config,
                config_path=config_path,
                central_token=central_token,
                log_frames=log_frames,
                candidate=candidate,
                footprint=footprint,
                map_api=map_api,
                args=args,
            )
            out_path = output_root / config.road_block_name / f"{central_token}_candidate_{args.candidate_index}.json"
            write_json(out_path, graph, overwrite=args.overwrite)
            num_written += 1

    print(f"Wrote {num_written} scene graph(s) to {output_root}")


if __name__ == "__main__":
    main()
