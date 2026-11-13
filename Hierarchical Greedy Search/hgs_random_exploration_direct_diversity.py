#!/usr/bin/env python3
"""Direct HGS exploration with canonical traffic risk-case signatures.

This runner preserves the command-line interface and search behaviour of
``hgs_random_exploration_direct.py``.  For every final selected candidate, it
rebuilds a full graph, derives a four-level canonical risk-case signature, and
writes ``risk_case_signatures.json`` under ``--summary-root``.

Only transition edges with ``positive_delta_risk > 1e-6`` are active risks.
Impact topology is measured at the graph endpoint. Dynamic impact is
``[ego_side, actor_side]`` for the interacting object pair; static impact is
``[ego_side, boundary_side]`` for the ego/boundary pair. Impact lists are
sorted but not deduplicated, so list ordering never changes equality while each
active target remains represented.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import nearest_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
HGS_DIR = Path(__file__).resolve().parent
if os.fspath(HGS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(HGS_DIR))

import generate_scene_graphs as gsg  # noqa: E402
import hierarchical_greedy_search as hgs  # noqa: E402
import hgs_random_exploration_direct as direct  # noqa: E402


RISK_CASE_FILENAME = "risk_case_signatures.json"
POSITIVE_DELTA_EPSILON = 1e-6
MOTION_HEADING_EPSILON_M = 0.05
DYNAMIC_HISTORY_DURATION_S = 1.5
DYNAMIC_STRIKER_SPEED_MARGIN_MPS = 1.0
STATIC_TURN_THRESHOLD_DEG = 3.0
SIDE_ORDER = {"Front": 0, "Rear": 1, "Left": 2, "Right": 3}
HISTORY15_DESCRIPTOR_DIR = Path("trajectory_candidate/dictionary_structure")
HISTORY15_DESCRIPTOR_STEM = "coarse_group_9_kmeans_history15_descriptors_sub60_40_40_20_20_20_20_20_10"
HISTORY15_DEFAULT_PATHS = {
    "coarse_labels_path": HISTORY15_DESCRIPTOR_DIR / "labels.npy",
    "coarse_representatives_path": HISTORY15_DESCRIPTOR_DIR / "representatives.npy",
    "sub_labels_path": HISTORY15_DESCRIPTOR_DIR / "sub_labels.npy",
    "sub_representatives_path": HISTORY15_DESCRIPTOR_DIR / "sub_representatives.npy",
}
HISTORY15_REPRESENTATIVE_POINTS = 15
_ORIGINAL_RUN_ONE_DIRECT = direct.run_one_direct
_ORIGINAL_WRITE_SELECTED_CANDIDATES_JSON = direct.write_selected_candidates_json
_ORIGINAL_PARSE_ARGS = direct.parse_args
_ORIGINAL_LOAD_COMMON_INPUTS = direct.load_common_inputs
_ORIGINAL_AGGREGATE_NOTES = direct.aggregate_notes
_LAST_PARSED_ARGS: Optional[argparse.Namespace] = None
_LAST_GROUP_SCHEDULE_METADATA: Dict[str, Any] = {}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def cli_option_present(*names: str) -> bool:
    for item in sys.argv[1:]:
        for name in names:
            if item == name or item.startswith(f"{name}="):
                return True
    return False


def parse_args_with_history15_defaults() -> argparse.Namespace:
    global _LAST_PARSED_ARGS
    args = _ORIGINAL_PARSE_ARGS()
    if not cli_option_present("--coarse-labels-path"):
        args.coarse_labels_path = HISTORY15_DEFAULT_PATHS["coarse_labels_path"]
    if not cli_option_present("--coarse-representatives-path"):
        args.coarse_representatives_path = HISTORY15_DEFAULT_PATHS["coarse_representatives_path"]
    if not cli_option_present("--sub-labels-path"):
        args.sub_labels_path = HISTORY15_DEFAULT_PATHS["sub_labels_path"]
    if not cli_option_present("--sub-representatives-path"):
        args.sub_representatives_path = HISTORY15_DEFAULT_PATHS["sub_representatives_path"]
    if not cli_option_present("--representative-points"):
        args.representative_points = HISTORY15_REPRESENTATIVE_POINTS
    _LAST_PARSED_ARGS = args
    return args


def group_schedule_metadata(common: Mapping[str, Any], search_args: argparse.Namespace) -> Dict[str, Any]:
    sub_ids_by_group = hgs.build_sub_ids_by_group(
        common["coarse_labels"],
        common["sub_labels"],
        common["group_indices"],
    )
    subcluster_counts = {str(int(group)): int(len(sub_ids_by_group[int(group)])) for group in common["group_indices"]}
    total_subclusters = int(sum(subcluster_counts.values()))
    q_max = max(1, int(search_args.q_max))
    representative_blocks = {
        group: int(math.ceil(float(count) / float(q_max))) if count > 0 else 0
        for group, count in subcluster_counts.items()
    }
    requested_saturation_window = finite_float(getattr(search_args, "saturation_window", 0.0), 0.0)
    if requested_saturation_window > 0.0:
        saturation_mode = "fixed"
        saturation_windows = {group: float(requested_saturation_window) for group in subcluster_counts}
    elif total_subclusters > 0:
        saturation_mode = "adaptive_subcluster_share"
        saturation_windows = {
            group: float(search_args.total_iterations) * float(count) / float(total_subclusters)
            for group, count in subcluster_counts.items()
        }
    else:
        saturation_mode = "adaptive_uniform_fallback"
        fallback = float(search_args.total_iterations) / float(max(1, len(subcluster_counts)))
        saturation_windows = {group: fallback for group in subcluster_counts}
    return {
        "clustering_default": "history15_descriptors_k9_sub60_40_40_20_20_20_20_20_10",
        "local_phase_policy": (
            "For each selected coarse group, test all available sub-cluster representatives first. "
            "If subcluster_count < q_max, the same local block continues with nearest-member expansion; "
            "if subcluster_count > q_max, multiple selected local blocks are used for representative testing "
            "before expansion starts."
        ),
        "q_max": int(search_args.q_max),
        "subcluster_counts_by_group": subcluster_counts,
        "total_subcluster_count": total_subclusters,
        "representative_testing_blocks_by_group": representative_blocks,
        "saturation_window_mode": saturation_mode,
        "effective_saturation_window_by_group": saturation_windows,
    }


def load_common_inputs_with_diversity_metadata(search_args: argparse.Namespace) -> Dict[str, Any]:
    global _LAST_GROUP_SCHEDULE_METADATA
    common = _ORIGINAL_LOAD_COMMON_INPUTS(search_args)
    _LAST_GROUP_SCHEDULE_METADATA = group_schedule_metadata(common, search_args)
    common["_diversity_group_schedule"] = _LAST_GROUP_SCHEDULE_METADATA
    return common


def aggregate_notes_with_diversity() -> Dict[str, Any]:
    notes = _ORIGINAL_AGGREGATE_NOTES()
    notes["local_phase_policy"] = (
        "A selected group evaluates its sub-cluster representatives before local expansion. "
        "With q_max=20, a 10-sub-cluster group performs representative testing and expansion in the first block; "
        "40- and 60-sub-cluster groups need two or three representative-testing blocks before expansion."
    )
    notes["metric_notes"]["saturation_window_mode"] = (
        "fixed if --saturation-window > 0; otherwise adaptive_subcluster_share, where "
        "L'_g = total_iterations * subcluster_count_g / total_subcluster_count."
    )
    notes["metric_notes"]["effective_saturation_window_by_group"] = (
        "Per-coarse-group saturation count threshold used by the scheduler when adaptive saturation is enabled."
    )
    return notes


def update_aggregate_summary_with_diversity_metadata() -> None:
    if _LAST_PARSED_ARGS is None:
        return
    summary_root = Path(_LAST_PARSED_ARGS.summary_root)
    path = summary_root / "aggregate_summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf8"))
    if _LAST_GROUP_SCHEDULE_METADATA:
        payload.update(_LAST_GROUP_SCHEDULE_METADATA)

    risk_path = summary_root / RISK_CASE_FILENAME
    if risk_path.exists():
        risk_payload = json.loads(risk_path.read_text(encoding="utf8"))
        risk_summary = risk_payload.get("summary", {})
        payload["risk_case_summary"] = {
            "mean_final_traffic_case_quantity_per_token": risk_summary["mean_final_traffic_case_quantity_per_token"]
        } if "mean_final_traffic_case_quantity_per_token" in risk_summary else {}
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def active_edges(graph: Mapping[str, Any], graph_key: str) -> List[Dict[str, Any]]:
    edges = graph.get("graphs", {}).get(graph_key, {}).get("edges", [])
    return [
        dict(edge)
        for edge in edges
        if isinstance(edge, Mapping) and finite_float(edge.get("positive_delta_risk")) > POSITIVE_DELTA_EPSILON
    ]


def count_bin(count: int) -> str:
    return ("None", "Single", "Multi")[min(max(int(count), 0), 2)]


def risk_change_bin(max_delta: float) -> str:
    """Bin the maximum active risk-score increase for one risk source."""
    if max_delta <= POSITIVE_DELTA_EPSILON:
        return "None"
    if max_delta < 0.30:
        return "Low"
    if max_delta < 0.70:
        return "Medium"
    return "High"


def node_index(graph: Mapping[str, Any], graph_key: str) -> Dict[str, Dict[str, Any]]:
    nodes = graph.get("graphs", {}).get(graph_key, {}).get("nodes", [])
    return {
        str(node["id"]): dict(node)
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") is not None
    }


def as_xy(value: Any, description: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"Expected finite [x, y] for {description}, got {value!r}")
    return point


def side_in_local_frame(origin: np.ndarray, yaw: float, target: np.ndarray) -> str:
    """Return Front/Rear/Left/Right of ``target`` relative to ``origin``."""
    delta = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    forward = math.cos(yaw) * float(delta[0]) + math.sin(yaw) * float(delta[1])
    left = -math.sin(yaw) * float(delta[0]) + math.cos(yaw) * float(delta[1])
    if abs(forward) >= abs(left):
        return "Front" if forward >= 0.0 else "Rear"
    return "Left" if left >= 0.0 else "Right"


def angle_diff_rad(a: float, b: float) -> float:
    return float(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))


def motion_yaw_from_nodes(
    start_node: Optional[Mapping[str, Any]],
    end_node: Mapping[str, Any],
    fallback_yaw: float,
) -> float:
    """Use 0.0s-to-1.5s motion as the local-frame heading, with yaw fallback."""
    if start_node is None:
        return finite_float(fallback_yaw)
    start = as_xy(start_node.get("center"), "motion start centre")
    end = as_xy(end_node.get("center"), "motion end centre")
    delta = end - start
    if float(np.linalg.norm(delta)) >= MOTION_HEADING_EPSILON_M:
        return float(math.atan2(float(delta[1]), float(delta[0])))
    return finite_float(fallback_yaw)


def motion_midpoint(
    start_node: Optional[Mapping[str, Any]],
    end_node: Mapping[str, Any],
    description: str,
) -> np.ndarray:
    """Reference centre for side classification over the replaced motion segment."""
    end = as_xy(end_node.get("center"), f"{description} end centre")
    if start_node is None:
        return end
    start = as_xy(start_node.get("center"), f"{description} start centre")
    return 0.5 * (start + end)


def impact_lateral_side(side: str) -> str:
    """Map lateral box-frame sides to the BEV/traffic-impact convention."""
    return {"Left": "Right", "Right": "Left"}.get(side, side)


def motion_velocity_from_nodes(start_node: Optional[Mapping[str, Any]], end_node: Mapping[str, Any]) -> np.ndarray:
    if start_node is None:
        return np.asarray(end_node.get("velocity_xy", [0.0, 0.0]), dtype=np.float64)
    start = as_xy(start_node.get("center"), "motion velocity start centre")
    end = as_xy(end_node.get("center"), "motion velocity end centre")
    return (end - start) / DYNAMIC_HISTORY_DURATION_S


def side_from_velocity(velocity_xy: np.ndarray, yaw: float, fallback: str) -> str:
    speed = float(np.linalg.norm(velocity_xy))
    if speed < MOTION_HEADING_EPSILON_M:
        return fallback
    forward = math.cos(yaw) * float(velocity_xy[0]) + math.sin(yaw) * float(velocity_xy[1])
    left = -math.sin(yaw) * float(velocity_xy[0]) + math.cos(yaw) * float(velocity_xy[1])
    if abs(forward) >= abs(left):
        return "Front" if forward >= 0.0 else "Rear"
    return impact_lateral_side("Left" if left >= 0.0 else "Right")


def polygon_from_node(node: Mapping[str, Any]) -> Optional[Polygon]:
    coords = node.get("polygon")
    if not coords:
        return None
    polygon = Polygon(coords)
    return polygon if polygon.is_valid and not polygon.is_empty else None


def contact_point_between_nodes(a_node: Mapping[str, Any], b_node: Mapping[str, Any]) -> Optional[np.ndarray]:
    a_poly = polygon_from_node(a_node)
    b_poly = polygon_from_node(b_node)
    if a_poly is None or b_poly is None:
        return None
    intersection = a_poly.intersection(b_poly)
    if not intersection.is_empty:
        return np.asarray([intersection.centroid.x, intersection.centroid.y], dtype=np.float64)
    a_point, b_point = nearest_points(a_poly, b_poly)
    return np.asarray([(a_point.x + b_point.x) * 0.5, (a_point.y + b_point.y) * 0.5], dtype=np.float64)


def node_box_size(node: Mapping[str, Any], default_length: float = 4.5, default_width: float = 2.0) -> Tuple[float, float]:
    size = node.get("size", {})
    if not isinstance(size, Mapping):
        return default_length, default_width
    return finite_float(size.get("length"), default_length), finite_float(size.get("width"), default_width)


def contact_side_on_node(node: Mapping[str, Any], contact_point: Optional[np.ndarray], fallback_target: np.ndarray) -> str:
    center = as_xy(node.get("center"), "contact node centre")
    yaw = finite_float(node.get("yaw"))
    point = np.asarray(contact_point if contact_point is not None else fallback_target, dtype=np.float64)
    local = np.asarray(
        [
            math.cos(yaw) * float(point[0] - center[0]) + math.sin(yaw) * float(point[1] - center[1]),
            -math.sin(yaw) * float(point[0] - center[0]) + math.cos(yaw) * float(point[1] - center[1]),
        ],
        dtype=np.float64,
    )
    length, width = node_box_size(node)
    half_l = max(float(length) * 0.5, 1e-6)
    half_w = max(float(width) * 0.5, 1e-6)
    distances = {
        "Front": half_l - float(local[0]),
        "Rear": float(local[0]) + half_l,
        "Left": half_w - float(local[1]),
        "Right": float(local[1]) + half_w,
    }
    side = min(distances, key=distances.get)
    return impact_lateral_side(side)


def dynamic_impact(
    ego_start: Mapping[str, Any],
    ego_end: Mapping[str, Any],
    actor_start: Optional[Mapping[str, Any]],
    actor_end: Mapping[str, Any],
) -> Tuple[str, str]:
    ego_center = as_xy(ego_end.get("center"), "dynamic ego endpoint centre")
    actor_center = as_xy(actor_end.get("center"), "dynamic actor endpoint centre")
    contact_point = contact_point_between_nodes(ego_end, actor_end)
    ego_contact_side = contact_side_on_node(ego_end, contact_point, actor_center)
    actor_contact_side = contact_side_on_node(actor_end, contact_point, ego_center)

    ego_velocity = motion_velocity_from_nodes(ego_start, ego_end)
    actor_velocity = motion_velocity_from_nodes(actor_start, actor_end)
    ego_motion_side = side_from_velocity(ego_velocity, finite_float(ego_end.get("yaw")), ego_contact_side)
    actor_motion_side = side_from_velocity(actor_velocity, finite_float(actor_end.get("yaw")), actor_contact_side)
    ego_speed = float(np.linalg.norm(ego_velocity))
    actor_speed = float(np.linalg.norm(actor_velocity))

    if ego_speed >= actor_speed + DYNAMIC_STRIKER_SPEED_MARGIN_MPS:
        return ego_motion_side, actor_contact_side
    if actor_speed >= ego_speed + DYNAMIC_STRIKER_SPEED_MARGIN_MPS:
        return ego_contact_side, actor_motion_side
    return ego_contact_side, actor_contact_side


def closest_point_and_tangent_on_line(point: np.ndarray, line: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return the nearest point and its oriented non-degenerate line tangent."""
    points = np.asarray(line, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("An active static target must contain a finite polyline with at least two points.")
    best_point: Optional[np.ndarray] = None
    best_tangent: Optional[np.ndarray] = None
    best_distance_sq = math.inf
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        segment_sq = float(np.dot(segment, segment))
        if segment_sq <= 1e-12:
            continue
        fraction = min(1.0, max(0.0, float(np.dot(point - start, segment) / segment_sq)))
        candidate = start + fraction * segment
        distance_sq = float(np.dot(point - candidate, point - candidate))
        if distance_sq < best_distance_sq:
            best_point = candidate
            best_tangent = segment
            best_distance_sq = distance_sq
    if best_point is None or best_tangent is None:
        raise ValueError("An active static target has no non-degenerate line segments.")
    return best_point, best_tangent


def closest_point_on_line(point: np.ndarray, line: Any) -> np.ndarray:
    """Backward-compatible nearest-point helper for a static boundary line."""
    closest, _ = closest_point_and_tangent_on_line(point, line)
    return closest


def lateral_side_in_local_frame(origin: np.ndarray, yaw: float, target: np.ndarray) -> str:
    """Return Left/Right of ``target`` without letting longitudinal offset dominate."""
    delta = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    left = -math.sin(yaw) * float(delta[0]) + math.cos(yaw) * float(delta[1])
    return "Left" if left >= 0.0 else "Right"


def static_impact(
    _ego_start: Mapping[str, Any],
    ego_end: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> Tuple[str, str]:
    """Return ``(ego_impact_side, boundary_impact_side)`` for a static risk.

    The ego side is the endpoint-footprint side nearest the boundary. The
    boundary side is the opposite side of the oriented boundary polyline from
    the ego centre: it describes the boundary face crossed by the ego.
    """
    ego_endpoint = as_xy(ego_end.get("center"), "static ego endpoint centre")
    closest, tangent = closest_point_and_tangent_on_line(ego_endpoint, boundary.get("line"))
    ego_impact_side = contact_side_on_node(ego_end, closest, closest)
    boundary_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
    ego_side_of_boundary = lateral_side_in_local_frame(closest, boundary_yaw, ego_endpoint)
    boundary_impact_side = impact_lateral_side(ego_side_of_boundary)
    return ego_impact_side, boundary_impact_side


def canonical_dynamic_impacts(impacts: Iterable[Tuple[str, str]]) -> List[List[str]]:
    ordered = sorted(impacts, key=lambda pair: (SIDE_ORDER[pair[0]], SIDE_ORDER[pair[1]]))
    return [[ego_side, actor_side] for ego_side, actor_side in ordered]


def canonical_static_incidents(incidents: Iterable[Tuple[str, str, str]]) -> Tuple[List[List[str]], List[str]]:
    ordered = sorted(incidents, key=lambda item: (SIDE_ORDER[item[0]], SIDE_ORDER[item[1]], item[2]))
    return (
        [[ego_side, boundary_side] for ego_side, boundary_side, _ in ordered],
        [boundary_role for _, _, boundary_role in ordered],
    )


def canonical_risk_case_signature(signature: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact ordered RiskCaseSignature fields used for comparison."""
    dynamic_impacts: List[Tuple[str, str]] = []
    for impact in signature.get("dynamic_impacts", []):
        if not isinstance(impact, Sequence) or isinstance(impact, (str, bytes)) or len(impact) != 2:
            raise ValueError(f"Dynamic impact must be [ego_side, actor_side], got {impact!r}")
        dynamic_impacts.append((str(impact[0]), str(impact[1])))

    raw_static_impacts = signature.get("static_impacts", [])
    raw_static_boundary_roles = signature.get("static_boundary_roles", [])
    if not isinstance(raw_static_impacts, Sequence) or isinstance(raw_static_impacts, (str, bytes)):
        raise ValueError("static_impacts must be a list of [ego_side, boundary_side] pairs")
    if not isinstance(raw_static_boundary_roles, Sequence) or isinstance(raw_static_boundary_roles, (str, bytes)):
        raise ValueError("static_boundary_roles must be a list parallel to static_impacts")
    if len(raw_static_impacts) != len(raw_static_boundary_roles):
        raise ValueError("Static impacts and boundary roles must have equal lengths; regenerate risk signatures.")
    static_incidents: List[Tuple[str, str, str]] = []
    for impact, boundary_role in zip(raw_static_impacts, raw_static_boundary_roles):
        if not isinstance(impact, Sequence) or isinstance(impact, (str, bytes)) or len(impact) != 2:
            raise ValueError(f"Static impact must be [ego_side, boundary_side], got {impact!r}")
        static_incidents.append((str(impact[0]), str(impact[1]), str(boundary_role)))
    canonical_static_impacts, canonical_static_boundary_roles = canonical_static_incidents(static_incidents)
    return {
        "source": str(signature.get("source", "")),
        "dynamic_count": str(signature.get("dynamic_count", "None")),
        "static_count": str(signature.get("static_count", "None")),
        "dynamic_impacts": canonical_dynamic_impacts(dynamic_impacts),
        "static_impacts": canonical_static_impacts,
        "static_boundary_roles": canonical_static_boundary_roles,
        "dynamic_change": str(signature.get("dynamic_change", "None")),
        "static_change": str(signature.get("static_change", "None")),
    }


def normalize_risk_case_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    raw_signature = dict(out["risk_case_signature"])
    signature = canonical_risk_case_signature(raw_signature)
    out["risk_case_signature"] = signature
    out["signature_key"] = signature_key(signature)
    return out


def signature_key(signature: Mapping[str, Any]) -> str:
    canonical = canonical_risk_case_signature(signature)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def same_risk_case(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Decision-tree equality over the four-level RiskCaseSignature."""
    a_sig = canonical_risk_case_signature(a)
    b_sig = canonical_risk_case_signature(b)
    if a_sig["source"] != b_sig["source"]:
        return False
    if a_sig["dynamic_count"] != b_sig["dynamic_count"]:
        return False
    if a_sig["static_count"] != b_sig["static_count"]:
        return False
    if a_sig["dynamic_impacts"] != b_sig["dynamic_impacts"]:
        return False
    if a_sig["static_impacts"] != b_sig["static_impacts"]:
        return False
    if a_sig["static_boundary_roles"] != b_sig["static_boundary_roles"]:
        return False
    if a_sig["dynamic_change"] != b_sig["dynamic_change"]:
        return False
    if a_sig["static_change"] != b_sig["static_change"]:
        return False
    return True


def build_risk_case_signature(
    graph: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a canonical four-level signature and auditable active-risk details."""
    dynamic = active_edges(graph, "G_delta_dyn_0_to_3")
    static = active_edges(graph, "G_delta_sta_0_to_3")
    if not dynamic and not static:
        raise ValueError("A final selected candidate must have at least one positive risk increase.")

    dynamic_start_nodes = node_index(graph, "G_dyn_0")
    dynamic_end_nodes = node_index(graph, "G_dyn_3")
    static_start_nodes = node_index(graph, "G_sta_0")
    static_end_nodes = node_index(graph, "G_sta_3")
    dynamic_details: List[Dict[str, Any]] = []
    static_details: List[Dict[str, Any]] = []
    dynamic_impacts: List[Tuple[str, str]] = []
    static_incidents: List[Tuple[str, str, str]] = []

    dynamic_ego_start = dynamic_start_nodes.get("ego")
    dynamic_ego_end = dynamic_end_nodes.get("ego")
    for edge in dynamic:
        target = str(edge.get("target", ""))
        actor_start = dynamic_start_nodes.get(target)
        actor_end = dynamic_end_nodes.get(target)
        if dynamic_ego_start is None or dynamic_ego_end is None or actor_end is None:
            raise ValueError(f"Active dynamic target {target!r} is missing from the endpoint-3 graph.")
        impact = dynamic_impact(dynamic_ego_start, dynamic_ego_end, actor_start, actor_end)
        dynamic_impacts.append(impact)
        dynamic_details.append(
            {"target": target, "positive_delta_risk": finite_float(edge.get("positive_delta_risk")), "impact": list(impact)}
        )

    static_ego_start = static_start_nodes.get("ego")
    static_ego_end = static_end_nodes.get("ego")
    for edge in static:
        target = str(edge.get("target", ""))
        boundary = static_end_nodes.get(target)
        if static_ego_start is None or static_ego_end is None or boundary is None:
            raise ValueError(f"Active static target {target!r} is missing from the endpoint-0 or endpoint-3 graph.")
        impact = static_impact(static_ego_start, static_ego_end, boundary)
        boundary_role = str(boundary.get("boundary_role", "unclassified_union_boundary"))
        static_incidents.append((impact[0], impact[1], boundary_role))
        static_details.append(
            {"target": target, "positive_delta_risk": finite_float(edge.get("positive_delta_risk")), "impact": list(impact), "boundary_role": boundary_role}
        )

    static_impacts, static_boundary_roles = canonical_static_incidents(static_incidents)
    dynamic_max = max((item["positive_delta_risk"] for item in dynamic_details), default=0.0)
    static_max = max((item["positive_delta_risk"] for item in static_details), default=0.0)
    source = "Mix" if dynamic and static else "Dynamic" if dynamic else "Static"
    signature = {
        "source": source,
        "dynamic_count": count_bin(len(dynamic)),
        "static_count": count_bin(len(static)),
        "dynamic_impacts": canonical_dynamic_impacts(dynamic_impacts),
        "static_impacts": static_impacts,
        "static_boundary_roles": static_boundary_roles,
        "dynamic_change": risk_change_bin(dynamic_max),
        "static_change": risk_change_bin(static_max),
    }
    details = {
        "dynamic_active_risks": sorted(dynamic_details, key=lambda item: (item["impact"], item["target"])),
        "static_active_risks": sorted(static_details, key=lambda item: (item["impact"], item["target"])),
        "dynamic_max_positive_delta_risk": dynamic_max,
        "static_max_positive_delta_risk": static_max,
        "topology_reference": "dynamic_history_motion_and_static_endpoint_contact_boundary_tangent",
        "motion_heading_epsilon_m": MOTION_HEADING_EPSILON_M,
    }
    return signature, details


def signature_records_for_run(
    *,
    row: Dict[str, Any],
    config_path: Path,
    token: str,
    navsim_log_path: Path,
    search_args: argparse.Namespace,
    common: Dict[str, Any],
    log_cache: Dict[str, List[Dict[str, Any]]],
    map_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Rebuild full graphs for one run's final candidates and save their signatures."""
    selections = row.get("_selected_candidates", [])
    if not selections:
        return []
    config = hgs.load_config(os.fspath(config_path))
    if not isinstance(config, hgs.FrameCentralConfig):
        raise TypeError(f"{config_path} is not a FrameCentralConfig.")
    if config.central_log not in log_cache:
        log_cache[config.central_log] = gsg.load_navsim_log(navsim_log_path, config.central_log)
    map_api: Optional[Any] = None
    if not search_args.skip_static_graph:
        if config.city not in map_cache:
            map_cache[config.city] = gsg.load_map_api(config.city)
        map_api = map_cache[config.city]

    records: List[Dict[str, Any]] = []
    candidates = common["candidates"]
    for selection in sorted(selections, key=lambda item: direct.int_value(item.get("coarse_group"))):
        candidate_index = direct.int_value(selection.get("candidate_index"))
        coarse_group = direct.int_value(selection.get("coarse_group"))
        graph_args = argparse.Namespace(**vars(search_args))
        graph_args.candidate_index = candidate_index
        graph_args.output_detail = "full"
        candidate = hgs.graph_candidate(candidates[candidate_index], bool(graph_args.candidate_prepend_origin))
        graph = gsg.build_scene_graph_for_token(
            config=config,
            config_path=config_path,
            central_token=token,
            log_frames=log_cache[config.central_log],
            candidate=candidate,
            footprint=common["footprint"],
            map_api=map_api,
            args=graph_args,
        )
        signature, details = build_risk_case_signature(graph)
        score = hgs.score_graph(graph)
        if not score["is_valid"] or finite_float(score["score_valid"]) <= 0.0:
            raise ValueError(
                "A final selected candidate did not rebuild as a positive valid risk case: "
                f"run_id={row.get('run_id')} candidate={candidate_index}."
            )
        records.append(
            {
                "run_id": direct.int_value(row.get("run_id")),
                "config": config_path.name,
                "road_block_name": str(row["road_block_name"]),
                "token": token,
                "coarse_group": coarse_group,
                "candidate_index": candidate_index,
                "graph_token": graph.get("graph_token"),
                "selected_score_valid": finite_float(score["score_valid"]),
                "risk_case_signature": signature,
                "signature_key": signature_key(signature),
                **details,
            }
        )
    return records


def run_one_direct_diversity(**kwargs: Any) -> Dict[str, Any]:
    """Run the established search, then add signatures for its final candidates."""
    row = _ORIGINAL_RUN_ONE_DIRECT(**kwargs)
    if row.get("status") != "ok":
        return row
    row["_risk_case_records"] = signature_records_for_run(
        row=row,
        config_path=kwargs["config_path"],
        token=str(kwargs["token"]),
        navsim_log_path=kwargs["navsim_log_path"],
        search_args=kwargs["search_args"],
        common=kwargs["common"],
        log_cache=kwargs["log_cache"],
        map_cache=kwargs["map_cache"],
    )
    return row


def group_risk_cases(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    case_groups: List[Dict[str, Any]] = []
    for original_record in records:
        record = normalize_risk_case_record(original_record)
        signature = record["risk_case_signature"]
        matched: Optional[Dict[str, Any]] = None
        for case_group in case_groups:
            if same_risk_case(signature, case_group["risk_case_signature"]):
                matched = case_group
                break
        if matched is None:
            matched = {
                "signature_key": signature_key(signature),
                "risk_case_signature": signature,
                "members": [],
            }
            case_groups.append(matched)
        matched["members"].append(record)
    return case_groups


def token_case_quantity_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_token: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for original_record in records:
        record = normalize_risk_case_record(original_record)
        key = (str(record["config"]), str(record["token"]))
        by_token.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (config, token), token_records in sorted(by_token.items()):
        rows.append(
            {
                "config": config,
                "token": token,
                "selected_trajectory_quantity": len(token_records),
                "final_traffic_case_quantity": len(group_risk_cases(token_records)),
            }
        )
    return rows


def summary_for_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    case_groups = group_risk_cases(records)
    selected_quantity = len(records)
    case_quantity = len(case_groups)
    token_case_rows = token_case_quantity_rows(records)
    token_case_values = [float(row["final_traffic_case_quantity"]) for row in token_case_rows]
    mean_token_case_quantity = float(sum(token_case_values) / len(token_case_values)) if token_case_values else 0.0
    return {
        "selected_trajectory_quantity": selected_quantity,
        "final_traffic_case_quantity": case_quantity,
        "mean_final_traffic_case_quantity_per_token": mean_token_case_quantity,
        "token_quantity": len(token_case_rows),
        "final_traffic_case_quantity_definition": "Global number of distinct RiskCaseSignature cases selected in the summary file, using same_risk_case decision-tree comparison.",
        "mean_final_traffic_case_quantity_per_token_definition": "Average final_traffic_case_quantity over config/token records in this summary file.",
        "duplicate_selected_trajectory_quantity": selected_quantity - case_quantity,
        "token_case_quantities": token_case_rows,
    }


def group_records_by_token(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    ordered_records = sorted(
        records,
        key=lambda item: (
            int(item["run_id"]),
            str(item["config"]),
            str(item["token"]),
            int(item["coarse_group"]),
            int(item["candidate_index"]),
        ),
    )
    for original_record in ordered_records:
        record = normalize_risk_case_record(original_record)
        key = (str(record["config"]), str(record["token"]))
        if key not in grouped:
            grouped[key] = {
                "config": key[0],
                "token": key[1],
                "candidates": {},
            }
        coarse_group = str(int(record["coarse_group"]))
        grouped[key]["candidates"][coarse_group] = {
            "candidate_index": int(record["candidate_index"]),
            "risk_case_signature": record["risk_case_signature"],
            "signature_key": str(record["signature_key"]),
            "selected_score_valid": finite_float(record.get("selected_score_valid")),
        }
    return list(grouped.values())


def risk_case_signature_definition() -> Dict[str, Any]:
    return {
        "format": "RiskCaseSignature(source, dynamic_count, static_count, dynamic_impacts, static_impacts, static_boundary_roles, dynamic_change, static_change)",
        "active_risk": "A risk is active only when positive_delta_risk > 1e-6, i.e. the candidate introduced or increased that risk.",
        "source": (
            "Dynamic if n_dyn > 0 and n_sta = 0; Static if n_dyn = 0 and n_sta > 0; "
            "Mix if n_dyn > 0 and n_sta > 0."
        ),
        "multiplicity": "dynamic_count/static_count use None=0, Single=1, Multi=2+ active risks.",
        "impact_topology": (
            "Four sides are Front, Rear, Left, Right. Dynamic "
            "impacts store one [ego_impact_side, actor_impact_side] pair per active dynamic actor. "
            "For dynamic risks, the faster approaching/striking object uses its motion-facing side, "
            "while the other object uses the contacted footprint side from endpoint polygons. "
            "Static impacts store one [ego_impact_side, boundary_impact_side] pair per active boundary; "
            "static_boundary_roles stores its parallel road-edge/barrier role. "
            "The ego side is computed from the endpoint footprint contact nearest the boundary. "
            "The boundary side is the face crossed by the ego: the side opposite the ego centre in "
            "the local frame of the nearest oriented boundary-line segment."
        ),
        "impact_order": (
            "dynamic_impacts are sorted by the predefined (ego side, actor side) order. Static impacts and "
            "static_boundary_roles are co-sorted by (ego side, boundary side, boundary role), where side order "
            "is Front, Rear, Left, Right. Duplicate topology entries are preserved."
        ),
        "risk_change": (
            "For each source, bin the maximum active risk-score increase: "
            "Low for 0.0 <= delta_r < 0.30, Medium for 0.30 <= delta_r < 0.70, "
            "High for 0.70 <= delta_r <= 1.00. None means that source has no active risk."
        ),
        "same_case": "Decision-tree equality over source, multiplicity, static impact topology and boundary role, and risk-increase severity after canonical sorting.",
    }


def build_risk_case_payload(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_records = [normalize_risk_case_record(record) for record in records]
    return {
        "schema_version": "risk_case_signature_by_token_v5_static_boundary_roles",
        "purpose": "Canonical case diversity over final HGS-selected trajectories, grouped by config/token.",
        "signature_definition": risk_case_signature_definition(),
        "summary": summary_for_records(normalized_records),
        "records": group_records_by_token(normalized_records),
    }


def records_from_risk_case_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload.get("selected_trajectory_records"), list):
        return [normalize_risk_case_record(record) for record in payload["selected_trajectory_records"]]

    records: List[Dict[str, Any]] = []
    for token_record in payload.get("records", []):
        config = str(token_record.get("config", ""))
        token = str(token_record.get("token", ""))
        candidates = token_record.get("candidates", {})
        if not isinstance(candidates, Mapping):
            continue
        for coarse_group, item in candidates.items():
            if not isinstance(item, Mapping):
                continue
            record = {
                "run_id": int(item.get("run_id", 0)),
                "config": config,
                "token": token,
                "coarse_group": int(coarse_group),
                "candidate_index": int(item["candidate_index"]),
                "selected_score_valid": finite_float(item.get("selected_score_valid")),
                "risk_case_signature": item["risk_case_signature"],
            }
            records.append(normalize_risk_case_record(record))
    return records


def rewrite_risk_case_signature_file(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    records = records_from_risk_case_payload(payload)
    new_payload = build_risk_case_payload(records)
    path.write_text(json.dumps(new_payload, indent=2), encoding="utf8")
    return new_payload


def write_risk_case_signatures(summary_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    records = [record for row in direct.sorted_run_rows(rows) for record in row.get("_risk_case_records", [])]
    output_path = summary_root / RISK_CASE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_risk_case_payload(records), indent=2), encoding="utf8")


def selected_candidate_case_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    case_groups = group_risk_cases(records)

    def case_sort_key(case_group: Mapping[str, Any]) -> Tuple[int, int]:
        members = case_group.get("members", [])
        if not members:
            return (10**9, 10**9)
        return min((int(member["coarse_group"]), int(member["candidate_index"])) for member in members)

    case_map: Dict[str, List[int]] = {}
    for case_idx, case_group in enumerate(sorted(case_groups, key=case_sort_key)):
        members = sorted(
            case_group.get("members", []),
            key=lambda member: (int(member["coarse_group"]), int(member["candidate_index"])),
        )
        case_map[f"case{case_idx}"] = [int(member["candidate_index"]) for member in members]
    return {
        "final_traffic_case_quantity": len(case_groups),
        "case": case_map,
    }


def write_selected_candidates_json_with_case_summary(summary_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, Any]] = []
    for row in direct.sorted_run_rows(rows):
        token = str(row.get("token", ""))
        if not token:
            continue
        config = Path(str(row.get("config", ""))).name
        candidate_map = {
            str(direct.int_value(candidate.get("coarse_group"))): direct.int_value(candidate.get("candidate_index"))
            for candidate in sorted(
                row.get("_selected_candidates", []),
                key=lambda item: direct.int_value(item.get("coarse_group")),
            )
        }
        case_summary = selected_candidate_case_summary(row.get("_risk_case_records", []))
        payload.append(
            {
                "config": config,
                "token": token,
                "candidates": candidate_map,
                **case_summary,
            }
        )
    (summary_root / "selected_candidates_by_token.json").write_text(json.dumps(payload, indent=2), encoding="utf8")


def enrich_selected_candidates_payload(
    selected_payload: Sequence[Mapping[str, Any]],
    risk_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_token: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in risk_records:
        normalized = normalize_risk_case_record(record)
        key = (str(normalized["config"]), str(normalized["token"]))
        by_token.setdefault(key, []).append(normalized)

    enriched: List[Dict[str, Any]] = []
    for item in selected_payload:
        config = Path(str(item.get("config", ""))).name
        token = str(item.get("token", ""))
        case_summary = selected_candidate_case_summary(by_token.get((config, token), []))
        enriched.append(
            {
                "config": config,
                "token": token,
                "candidates": dict(item.get("candidates", {})),
                **case_summary,
            }
        )
    return enriched


def rewrite_selected_candidates_with_case_summary(selected_path: Path, risk_path: Path) -> List[Dict[str, Any]]:
    selected_payload = json.loads(selected_path.read_text(encoding="utf8"))
    risk_payload = json.loads(risk_path.read_text(encoding="utf8"))
    risk_records = records_from_risk_case_payload(risk_payload)
    enriched = enrich_selected_candidates_payload(selected_payload, risk_records)
    selected_path.write_text(json.dumps(enriched, indent=2), encoding="utf8")
    return enriched


def write_selected_candidates_json_with_diversity(summary_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_risk_case_signatures(summary_root, rows)
    write_selected_candidates_json_with_case_summary(summary_root, rows)


def main() -> None:
    # ``run_one_worker`` looks up these names in the original module. On the
    # platform's fork-based worker pool, the replacements are inherited by each
    # worker and preserve the original direct runner's multiprocessing logic.
    direct.parse_args = parse_args_with_history15_defaults
    direct.load_common_inputs = load_common_inputs_with_diversity_metadata
    direct.aggregate_notes = aggregate_notes_with_diversity
    direct.run_one_direct = run_one_direct_diversity
    direct.write_selected_candidates_json = write_selected_candidates_json_with_diversity
    direct.main()
    update_aggregate_summary_with_diversity_metadata()


if __name__ == "__main__":
    main()
