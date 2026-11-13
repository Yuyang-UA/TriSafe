#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS / NAVSIM hierarchical greedy trajectory candidate search                   #
#-------------------------------------------------------------------------------#
"""Search trajectory candidates with coarse-group scheduling and local expansion.

The search treats trajectory replacement as a counterfactual safety test:

* reward is the largest *valid* risk increase, max(R_dyn, R_sta);
* already-collision / already-violation outcomes are invalid, not desirable;
* one best valid candidate is retained per coarse group.

The algorithm follows the proposed two-level policy:

1. Bootstrap every coarse group with its representative member, defining B_k^(0).
2. At each global round, select one coarse group by A_k,r.
3. Run up to q_max local evaluations in that group.
4. A group evaluates its own sub-cluster medoids first. If the sub-cluster
   count is below q_max, the same local block continues with nearest-member
   expansion; if it is above q_max, several selected local blocks are used for
   medoid testing before expansion begins.

Example:

    python "Hierarchical Greedy Search/hierarchical_greedy_search.py" \
      --config data/auto_reconstruction/config/navtest_available_merged_pointcloud_good/2021.05.25.14.16.10_veh-35_00083_00485-4b594a8e80915943.yaml \
      --tokens 4b594a8e80915943 \
      --navsim-log-path "${OPENSCENE_DATA_ROOT}/navsim_logs/test" \
      --total-iterations 160 \
      --q-max 20 \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HGS_DIR = Path(__file__).resolve().parent
if os.fspath(HGS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(HGS_DIR))

from nuplan_scripts.utils.config import FrameCentralConfig, load_config  # noqa: E402
import generate_scene_graphs as gsg  # noqa: E402


HARD_EVENTS = {"collision", "violation"}


@dataclass
class GroupState:
    group: int
    initial_candidate_index: int
    initial_score: float
    initial_result: Dict[str, Any]
    n: int = 0
    hard_count: int = 0
    best_score: float = 0.0
    best_candidate_index: Optional[int] = None
    best_result: Optional[Dict[str, Any]] = None
    best_history: List[float] = field(default_factory=list)
    evaluated_candidate_indices: set[int] = field(default_factory=set)
    sub_medoid_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    subcluster_order: List[int] = field(default_factory=list)
    phase1_done: bool = False
    exhausted: bool = False

    def __post_init__(self) -> None:
        self.best_score = float(self.initial_score)
        self.best_history = [float(self.initial_score)]
        self.evaluated_candidate_indices.add(int(self.initial_candidate_index))
        if self.initial_score > 0.0 and self.initial_result.get("is_valid", False):
            self.best_candidate_index = int(self.initial_candidate_index)
            self.best_result = self.initial_result


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def resolve(path: os.PathLike | str) -> Path:
    return gsg.resolve(path)


def write_json(path: Path, data: Dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf8")


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    overwrite: bool,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = list(fieldnames) if fieldnames is not None else []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ["empty"]
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_npy(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_candidate_bank(path: Path) -> np.ndarray:
    candidates = np.load(resolve(path))
    if candidates.ndim != 3 or candidates.shape[2] < 2:
        raise ValueError(f"{path} must have shape (N, T, 2+) or (N, T, 3), got {candidates.shape}")
    return np.asarray(candidates[:, :, :3] if candidates.shape[2] >= 3 else candidates[:, :, :2], dtype=np.float64)


def load_labels(path: Path, expected_count: int, name: str) -> np.ndarray:
    labels = np.load(resolve(path))
    if labels.ndim != 1:
        raise ValueError(f"{path} must have shape (N,), got {labels.shape}")
    if len(labels) != expected_count:
        raise ValueError(f"{name} has {len(labels)} labels but candidate bank has {expected_count} trajectories.")
    return np.asarray(labels, dtype=np.int64)


def load_representatives(path: Path, expected_ndim0: Optional[int], name: str) -> np.ndarray:
    reps = np.load(resolve(path))
    if reps.ndim != 3 or reps.shape[2] < 2:
        raise ValueError(f"{path} must have shape (M, T, 2+) or (M, T, 3), got {reps.shape}")
    if expected_ndim0 is not None and reps.shape[0] != expected_ndim0:
        raise ValueError(f"{name} has {reps.shape[0]} representatives; expected {expected_ndim0}.")
    return np.asarray(reps[:, :, :3] if reps.shape[2] >= 3 else reps[:, :, :2], dtype=np.float64)


def selected_groups(num_groups: int, requested_groups: Optional[Sequence[int]]) -> List[int]:
    all_groups = list(range(num_groups))
    if requested_groups is None:
        return all_groups
    requested = [int(group) for group in requested_groups]
    missing = [group for group in requested if group not in all_groups]
    if missing:
        raise ValueError(f"Requested groups are not available: {missing}. Available: {all_groups}")
    return requested


def representative_feature(rep: np.ndarray, feature_len: int, feature_dim: int) -> np.ndarray:
    return np.asarray(rep[:feature_len, :feature_dim], dtype=np.float64).reshape(1, -1)


def candidate_features(candidates: np.ndarray, feature_len: int, feature_dim: int) -> np.ndarray:
    return np.asarray(candidates[:, :feature_len, :feature_dim], dtype=np.float64).reshape(candidates.shape[0], -1)


def nearest_member_index(
    features: np.ndarray,
    rep: np.ndarray,
    member_indices: np.ndarray,
    feature_len: int,
    feature_dim: int,
) -> Tuple[int, float]:
    if len(member_indices) == 0:
        raise ValueError("Cannot find a representative member in an empty candidate set.")
    rep_feature = representative_feature(rep, feature_len, feature_dim)
    distances = np.linalg.norm(features[member_indices] - rep_feature, axis=1)
    best_pos = int(np.argmin(distances))
    return int(member_indices[best_pos]), float(distances[best_pos])


def build_sub_ids_by_group(
    coarse_labels: np.ndarray,
    sub_labels: np.ndarray,
    group_indices: Sequence[int],
) -> Dict[int, List[int]]:
    group_set = set(int(group) for group in group_indices)
    by_group: Dict[int, List[int]] = {int(group): [] for group in group_indices}
    for sub_id in sorted(int(x) for x in np.unique(sub_labels)):
        member_indices = np.flatnonzero(sub_labels == sub_id)
        if len(member_indices) == 0:
            continue
        groups, counts = np.unique(coarse_labels[member_indices], return_counts=True)
        group = int(groups[int(np.argmax(counts))])
        if group in group_set:
            by_group[group].append(sub_id)
    return by_group


def rank_members_by_distance(
    candidates: np.ndarray,
    rep: np.ndarray,
    member_indices: np.ndarray,
    feature_len: int,
    feature_dim: int,
) -> List[int]:
    if len(member_indices) == 0:
        return []
    features = candidate_features(candidates, feature_len, feature_dim)
    rep_feature = representative_feature(rep, feature_len, feature_dim)
    distances = np.linalg.norm(features[member_indices] - rep_feature, axis=1)
    order = np.argsort(distances, kind="mergesort")
    return [int(member_indices[int(pos)]) for pos in order]


def graph_candidate(candidate: np.ndarray, prepend_origin: bool) -> np.ndarray:
    out = np.asarray(candidate, dtype=np.float64)
    if prepend_origin:
        return np.concatenate([np.zeros((1, out.shape[1]), dtype=np.float64), out], axis=0)
    return out


def hard_event_from_reason(reason: Any) -> Optional[str]:
    if reason is None:
        return None
    return gsg.transition_event_from_reason(str(reason))


def graph_edges(graph: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return list(graph.get("graphs", {}).get(key, {}).get("edges", []))


def edge_hard_events(edge: Dict[str, Any]) -> List[str]:
    events: List[str] = []
    terminal = edge.get("terminal_event")
    if terminal in HARD_EVENTS:
        events.append(str(terminal))
    for key in ("reason_0", "reason_3", "reason"):
        event = hard_event_from_reason(edge.get(key))
        if event in HARD_EVENTS:
            events.append(str(event))
    return sorted(set(events))


def max_positive_delta(edges: Sequence[Dict[str, Any]]) -> Tuple[float, Optional[str], Optional[Dict[str, Any]]]:
    best_value = 0.0
    best_target = None
    best_edge = None
    for edge in edges:
        value = finite_float(edge.get("positive_delta_risk", 0.0))
        if value > best_value:
            best_value = value
            best_target = str(edge.get("target", ""))
            best_edge = edge
    return best_value, best_target, best_edge


def score_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    dyn_edges = graph_edges(graph, "G_delta_dyn_0_to_3")
    sta_edges = graph_edges(graph, "G_delta_sta_0_to_3")
    r_dyn, dyn_target, dyn_edge = max_positive_delta(dyn_edges)
    r_sta, sta_target, sta_edge = max_positive_delta(sta_edges)

    hard_events: List[str] = []
    hard_targets: List[str] = []
    hard_reasons: List[str] = []
    for edge in dyn_edges + sta_edges:
        events = edge_hard_events(edge)
        if not events:
            continue
        hard_events.extend(events)
        hard_targets.append(str(edge.get("target", "")))
        for key in ("terminal_event", "reason_0", "reason_3", "reason"):
            if edge.get(key) is not None:
                hard_reasons.append(f"{key}={edge[key]}")

    hard_events = sorted(set(hard_events))
    hard_targets = sorted(set(target for target in hard_targets if target))
    hard_reasons = sorted(set(hard_reasons))
    raw_score = max(r_dyn, r_sta)
    is_valid = len(hard_events) == 0
    score_valid = raw_score if is_valid else 0.0
    if r_dyn >= r_sta:
        source = "dynamic"
        target = dyn_target
        edge = dyn_edge
    else:
        source = "static"
        target = sta_target
        edge = sta_edge

    return {
        "R_dyn": float(r_dyn),
        "R_sta": float(r_sta),
        "score_raw": float(raw_score),
        "score_valid": float(score_valid),
        "is_valid": bool(is_valid),
        "D_hard": 0 if is_valid else 1,
        "hard_event": "+".join(hard_events),
        "hard_targets": hard_targets,
        "hard_reasons": hard_reasons,
        "risk_source": source,
        "risk_target": target,
        "risk_edge_delta": finite_float(edge.get("delta_risk", 0.0)) if edge else 0.0,
        "num_dynamic_edges": len(dyn_edges),
        "num_static_edges": len(sta_edges),
    }


def group_window_improvement(state: GroupState, local_window: int) -> float:
    if state.n <= 0:
        return 0.0
    ell = min(int(local_window), int(state.n))
    before_index = int(state.n - ell)
    return float(state.best_history[state.n] - state.best_history[before_index])


def group_eta(state: GroupState) -> float:
    if state.n <= 0:
        return 0.0
    return float(state.hard_count) / float(state.n)


def group_saturation(
    state: GroupState,
    local_window: int,
    saturation_window: float,
    epsilon: float,
    ts: float,
    rho: float,
) -> float:
    improvement = group_window_improvement(state, local_window)
    return sigmoid((float(state.n) - float(saturation_window)) / float(ts)) * sigmoid(
        (float(epsilon) - improvement) / float(rho)
    )


def group_arg_value(args: argparse.Namespace, name: str, group: int, default: float = 0.0) -> float:
    values = getattr(args, name, None)
    if isinstance(values, dict):
        if int(group) in values:
            return finite_float(values[int(group)], default)
        if str(int(group)) in values:
            return finite_float(values[str(int(group))], default)
    return finite_float(default, default)


def saturation_window_for_group(args: argparse.Namespace, group: int) -> float:
    return group_arg_value(args, "saturation_windows_by_group", group, finite_float(args.saturation_window))


def group_fitness(state: GroupState, args: argparse.Namespace) -> float:
    if state.exhausted:
        return -math.inf
    improvement = group_window_improvement(state, args.local_window)
    saturation = group_saturation(
        state,
        args.local_window,
        saturation_window_for_group(args, state.group),
        args.epsilon,
        args.saturation_temperature,
        args.improvement_rho,
    )
    eta = group_eta(state)
    return (
        float(args.omega_b) * float(state.best_score)
        + float(args.omega_i) * improvement
        - float(args.omega_s) * saturation
        - float(args.omega_h) * eta
    )


def metrics_row(state: GroupState, args: argparse.Namespace, global_round: int, selected: bool = False) -> Dict[str, Any]:
    improvement = group_window_improvement(state, args.local_window)
    saturation_window = saturation_window_for_group(args, state.group)
    saturation = group_saturation(
        state,
        args.local_window,
        saturation_window,
        args.epsilon,
        args.saturation_temperature,
        args.improvement_rho,
    )
    eta = group_eta(state)
    fitness = group_fitness(state, args)
    subcluster_count = int(group_arg_value(args, "subcluster_counts_by_group", state.group, 0.0))
    representative_blocks = int(group_arg_value(args, "representative_blocks_by_group", state.group, 0.0))
    return {
        "global_round": int(global_round),
        "coarse_group": int(state.group),
        "selected": int(bool(selected)),
        "A": None if not math.isfinite(fitness) else float(fitness),
        "B": float(state.best_score),
        "I": float(improvement),
        "S": float(saturation),
        "eta": float(eta),
        "saturation_window": float(saturation_window),
        "saturation_window_mode": str(getattr(args, "saturation_window_mode", "fixed")),
        "subcluster_count": int(subcluster_count),
        "phase1_remaining": max(0, int(subcluster_count) - len(state.sub_medoid_results)),
        "representative_testing_blocks": int(representative_blocks),
        "n": int(state.n),
        "hard_count": int(state.hard_count),
        "best_candidate_index": state.best_candidate_index,
        "best_score": float(state.best_score),
        "phase1_done": int(state.phase1_done),
        "exhausted": int(state.exhausted),
    }


class HierarchicalGreedySearch:
    def __init__(
        self,
        *,
        config: FrameCentralConfig,
        config_path: Path,
        central_token: str,
        log_frames: Sequence[Dict[str, Any]],
        footprint: gsg.EgoFootprintParams,
        map_api: Optional[Any],
        candidates: np.ndarray,
        coarse_labels: np.ndarray,
        coarse_reps: np.ndarray,
        sub_labels: np.ndarray,
        sub_reps: np.ndarray,
        group_indices: Sequence[int],
        output_root: Path,
        args: argparse.Namespace,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.central_token = str(central_token)
        self.log_frames = log_frames
        self.footprint = footprint
        self.map_api = map_api
        self.candidates = candidates
        self.coarse_labels = coarse_labels
        self.coarse_reps = coarse_reps
        self.sub_labels = sub_labels
        self.sub_reps = sub_reps
        self.group_indices = [int(group) for group in group_indices]
        self.output_root = output_root
        self.args = argparse.Namespace(**vars(args))
        self.requested_saturation_window = finite_float(getattr(self.args, "saturation_window", 0.0), 0.0)

        self.eval_counter = 0
        self.local_eval_counter = 0
        self.evaluation_rows: List[Dict[str, Any]] = []
        self.round_rows: List[Dict[str, Any]] = []
        self.graph_cache: Dict[int, Dict[str, Any]] = {}
        self.group_states: Dict[int, GroupState] = {}

        self.feature_len = int(min(candidates.shape[1], coarse_reps.shape[1], sub_reps.shape[1], self.args.representative_points))
        self.feature_dim = int(min(candidates.shape[2], coarse_reps.shape[2], sub_reps.shape[2]))
        self.features = candidate_features(candidates, self.feature_len, self.feature_dim)

        self.sub_ids_by_group = build_sub_ids_by_group(coarse_labels, sub_labels, self.group_indices)
        self.configure_group_level_schedule()
        self.coarse_rep_indices = self._build_coarse_rep_indices()
        self.sub_rep_indices = self._build_sub_rep_indices()
        self.sorted_sub_members: Dict[int, List[int]] = {}
        self.risk_context = None
        if getattr(self.args, "evaluation_mode", "scene_graph") == "risk_only":
            self.risk_context = gsg.build_risk_eval_context(
                config=config,
                config_path=config_path,
                central_token=central_token,
                log_frames=log_frames,
                footprint=footprint,
                map_api=map_api,
                args=self.args,
            )

    def configure_group_level_schedule(self) -> None:
        subcluster_counts = {
            int(group): int(len(self.sub_ids_by_group.get(int(group), [])))
            for group in self.group_indices
        }
        total_subclusters = int(sum(subcluster_counts.values()))
        q_max = max(1, int(self.args.q_max))
        representative_blocks = {
            int(group): int(math.ceil(float(count) / float(q_max))) if count > 0 else 0
            for group, count in subcluster_counts.items()
        }

        if self.requested_saturation_window > 0.0:
            saturation_windows = {
                int(group): float(self.requested_saturation_window)
                for group in self.group_indices
            }
            saturation_window_mode = "fixed"
            display_window = float(self.requested_saturation_window)
        elif total_subclusters > 0:
            saturation_windows = {
                int(group): float(self.args.total_iterations) * float(count) / float(total_subclusters)
                for group, count in subcluster_counts.items()
            }
            saturation_window_mode = "adaptive_subcluster_share"
            display_window = float(self.args.total_iterations) / float(max(1, len(self.group_indices)))
        else:
            display_window = float(self.args.total_iterations) / float(max(1, len(self.group_indices)))
            saturation_windows = {
                int(group): display_window
                for group in self.group_indices
            }
            saturation_window_mode = "adaptive_uniform_fallback"

        self.subcluster_counts_by_group = subcluster_counts
        self.total_subcluster_count = total_subclusters
        self.representative_blocks_by_group = representative_blocks
        self.saturation_windows_by_group = saturation_windows
        self.args.subcluster_counts_by_group = dict(subcluster_counts)
        self.args.total_subcluster_count = int(total_subclusters)
        self.args.representative_blocks_by_group = dict(representative_blocks)
        self.args.saturation_windows_by_group = dict(saturation_windows)
        self.args.saturation_window_mode = saturation_window_mode
        self.args.saturation_window = float(display_window)

    @property
    def run_root(self) -> Path:
        return self.output_root / self.config.road_block_name / self.central_token

    @property
    def graph_root(self) -> Path:
        return self.run_root / "graphs"

    def _build_coarse_rep_indices(self) -> Dict[int, Tuple[int, float]]:
        out: Dict[int, Tuple[int, float]] = {}
        for group in self.group_indices:
            member_indices = np.flatnonzero(self.coarse_labels == group)
            idx, distance = nearest_member_index(
                self.features,
                self.coarse_reps[group],
                member_indices,
                self.feature_len,
                self.feature_dim,
            )
            out[group] = (idx, distance)
        return out

    def _build_sub_rep_indices(self) -> Dict[int, Tuple[int, float]]:
        out: Dict[int, Tuple[int, float]] = {}
        for group in self.group_indices:
            for sub_id in self.sub_ids_by_group[group]:
                if sub_id < 0 or sub_id >= len(self.sub_reps):
                    raise ValueError(f"Sub-cluster id {sub_id} is out of range for {len(self.sub_reps)} representatives.")
                member_indices = np.flatnonzero(self.sub_labels == sub_id)
                idx, distance = nearest_member_index(
                    self.features,
                    self.sub_reps[sub_id],
                    member_indices,
                    self.feature_len,
                    self.feature_dim,
                )
                out[int(sub_id)] = (idx, distance)
        return out

    def _sub_members_by_distance(self, sub_id: int) -> List[int]:
        sub_id = int(sub_id)
        if sub_id not in self.sorted_sub_members:
            member_indices = np.flatnonzero(self.sub_labels == sub_id)
            self.sorted_sub_members[sub_id] = rank_members_by_distance(
                candidates=self.candidates,
                rep=self.sub_reps[sub_id],
                member_indices=member_indices,
                feature_len=self.feature_len,
                feature_dim=self.feature_dim,
            )
        return self.sorted_sub_members[sub_id]

    def _graph_args(self, candidate_index: int) -> argparse.Namespace:
        graph_args = argparse.Namespace(**vars(self.args))
        graph_args.candidate_trajectory_path = self.args.candidate_trajectory_path
        graph_args.candidate_index = int(candidate_index)
        return graph_args

    def _graph_path(self, candidate_index: int) -> Path:
        return self.graph_root / f"candidate_{int(candidate_index):05d}.json"

    def _evaluate_graph_once(self, candidate_index: int) -> Dict[str, Any]:
        candidate_index = int(candidate_index)
        if candidate_index in self.graph_cache:
            return self.graph_cache[candidate_index]

        candidate = graph_candidate(self.candidates[candidate_index], bool(self.args.candidate_prepend_origin))
        graph_args = self._graph_args(candidate_index)
        if getattr(self.args, "evaluation_mode", "scene_graph") == "risk_only" and not self.args.write_graphs:
            if self.risk_context is None:
                raise RuntimeError("Risk-only evaluation was requested but the risk context was not initialized.")
            score = gsg.evaluate_risk_for_context(
                context=self.risk_context,
                candidate=candidate,
                args=graph_args,
                candidate_index=candidate_index,
            )
            result = {
                **score,
                "candidate_index": int(candidate_index),
                "graph_token": score.get("graph_token"),
                "graph_path": "",
            }
            self.graph_cache[candidate_index] = result
            return result

        graph = gsg.build_scene_graph_for_token(
            config=self.config,
            config_path=self.config_path,
            central_token=self.central_token,
            log_frames=self.log_frames,
            candidate=candidate,
            footprint=self.footprint,
            map_api=self.map_api,
            args=graph_args,
        )
        graph["search_candidate"] = {
            "candidate_index": int(candidate_index),
            "valid_scoring_rule": "score_valid=max(R_dyn,R_sta) only when D_hard=0; hard events are invalid.",
        }
        score = score_graph(graph)
        graph["search_score"] = score
        graph_path = self._graph_path(candidate_index)
        if self.args.write_graphs:
            write_json(graph_path, graph, overwrite=self.args.overwrite)

        result = {
            **score,
            "candidate_index": int(candidate_index),
            "graph_token": graph.get("graph_token"),
            "graph_path": os.fspath(graph_path) if self.args.write_graphs else "",
        }
        self.graph_cache[candidate_index] = result
        return result

    def evaluate_action(
        self,
        *,
        group: int,
        candidate_index: int,
        phase: str,
        sub_id: Optional[int],
        global_round: int,
        counts_toward_budget: bool,
    ) -> Dict[str, Any]:
        self.eval_counter += 1
        base = self._evaluate_graph_once(candidate_index)
        row = {
            **base,
            "evaluation_id": int(self.eval_counter),
            "local_evaluation_id": int(self.local_eval_counter + 1) if counts_toward_budget else None,
            "global_round": int(global_round),
            "coarse_group": int(group),
            "sub_id": None if sub_id is None else int(sub_id),
            "phase": phase,
            "counts_toward_budget": int(bool(counts_toward_budget)),
            "road_block_name": self.config.road_block_name,
            "central_token": self.central_token,
        }
        if counts_toward_budget:
            self.local_eval_counter += 1
            row["local_evaluation_id"] = int(self.local_eval_counter)
        self.evaluation_rows.append(row)
        return row

    def bootstrap(self) -> None:
        for group in self.group_indices:
            candidate_index, distance = self.coarse_rep_indices[group]
            result = self.evaluate_action(
                group=group,
                candidate_index=candidate_index,
                phase="bootstrap_coarse_representative",
                sub_id=None,
                global_round=0,
                counts_toward_budget=False,
            )
            result["representative_distance"] = float(distance)
            state = GroupState(
                group=group,
                initial_candidate_index=candidate_index,
                initial_score=float(result["score_valid"]),
                initial_result=result,
            )
            self.group_states[group] = state

    def finalize_phase1_order(self, state: GroupState) -> None:
        if state.phase1_done:
            return
        expected = set(int(sub_id) for sub_id in self.sub_ids_by_group[state.group])
        if set(state.sub_medoid_results) != expected:
            return
        items = []
        for sub_id, result in state.sub_medoid_results.items():
            items.append(
                (
                    -float(result["score_valid"]),
                    -float(result["score_raw"]),
                    int(result["D_hard"]),
                    int(sub_id),
                )
            )
        state.subcluster_order = [int(item[3]) for item in sorted(items)]
        state.phase1_done = True

    def next_local_action(self, state: GroupState) -> Optional[Tuple[int, Optional[int], str]]:
        group = int(state.group)
        for sub_id in self.sub_ids_by_group[group]:
            if int(sub_id) in state.sub_medoid_results:
                continue
            candidate_index = self.sub_rep_indices[int(sub_id)][0]
            return int(candidate_index), int(sub_id), "phase1_subcluster_medoid"

        self.finalize_phase1_order(state)
        if not state.phase1_done:
            state.phase1_done = True
            state.subcluster_order = [int(sub_id) for sub_id in self.sub_ids_by_group[group]]

        for sub_id in state.subcluster_order:
            for candidate_index in self._sub_members_by_distance(sub_id):
                if int(candidate_index) in state.evaluated_candidate_indices:
                    continue
                return int(candidate_index), int(sub_id), "phase2_local_expansion"

        state.exhausted = True
        return None

    def update_group_state(self, state: GroupState, result: Dict[str, Any]) -> None:
        state.n += 1
        state.evaluated_candidate_indices.add(int(result["candidate_index"]))
        if int(result["D_hard"]) == 1:
            state.hard_count += 1
        if result["is_valid"] and float(result["score_valid"]) > float(state.best_score):
            state.best_score = float(result["score_valid"])
            state.best_candidate_index = int(result["candidate_index"])
            state.best_result = result
        state.best_history.append(float(state.best_score))

    def run_local_block(self, group: int, global_round: int, budget: int) -> int:
        state = self.group_states[int(group)]
        completed = 0
        for _ in range(int(budget)):
            action = self.next_local_action(state)
            if action is None:
                break
            candidate_index, sub_id, phase = action
            result = self.evaluate_action(
                group=group,
                candidate_index=candidate_index,
                phase=phase,
                sub_id=sub_id,
                global_round=global_round,
                counts_toward_budget=True,
            )
            if phase == "phase1_subcluster_medoid" and sub_id is not None:
                state.sub_medoid_results[int(sub_id)] = result
                self.finalize_phase1_order(state)
            self.update_group_state(state, result)
            completed += 1
        return completed

    def selectable_states(self) -> List[GroupState]:
        available = [state for state in self.group_states.values() if not state.exhausted]
        if not available:
            return []

        cap = finite_float(getattr(self.args, "group_eval_share_cap", 0.0), 0.0)
        if cap > 0.0 and int(self.args.total_iterations) > 0:
            max_evals = max(int(self.args.q_max), int(math.ceil(float(self.args.total_iterations) * cap)))
            under_cap = [state for state in available if int(state.n) < max_evals]
            if under_cap:
                return under_cap
        return available

    def select_group(self, global_round: int) -> int:
        available = self.selectable_states()
        if not available:
            raise RuntimeError("All groups are exhausted; no group can be selected.")
        if global_round == 0:
            return int(
                max(
                    available,
                    key=lambda state: (float(state.initial_score), -int(state.group)),
                ).group
            )
        return int(
            max(
                available,
                key=lambda state: (group_fitness(state, self.args), -int(state.group)),
            ).group
        )

    def record_round(self, global_round: int, selected_group: int, stage: str) -> None:
        for group in self.group_indices:
            row = metrics_row(self.group_states[group], self.args, global_round, selected=(group == selected_group))
            row["stage"] = stage
            self.round_rows.append(row)

    def run(self) -> None:
        self.bootstrap()

        remaining = int(self.args.total_iterations)
        global_round = 0
        while remaining > 0:
            selected_group = self.select_group(global_round)
            self.record_round(global_round, selected_group, stage="before_local_block")
            budget = min(int(self.args.q_max), remaining)
            completed = self.run_local_block(selected_group, global_round, budget)
            self.record_round(global_round, selected_group, stage="after_local_block")
            if completed == 0:
                if all(state.exhausted for state in self.group_states.values()):
                    break
            remaining -= completed
            global_round += 1

    def final_candidates(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for group in self.group_indices:
            state = self.group_states[group]
            if state.best_candidate_index is None or state.best_score <= 0.0:
                continue
            result = state.best_result or {}
            rows.append(
                {
                    "coarse_group": int(group),
                    "candidate_index": int(state.best_candidate_index),
                    "score_valid": float(state.best_score),
                    "R_dyn": float(result.get("R_dyn", 0.0)),
                    "R_sta": float(result.get("R_sta", 0.0)),
                    "risk_source": result.get("risk_source", ""),
                    "risk_target": result.get("risk_target", ""),
                    "graph_path": result.get("graph_path", ""),
                    "n": int(state.n),
                    "hard_count": int(state.hard_count),
                    "eta": float(group_eta(state)),
                    "initial_candidate_index": int(state.initial_candidate_index),
                    "initial_score": float(state.initial_score),
                }
            )
        rows.sort(key=lambda row: int(row["coarse_group"]))
        return rows

    def group_summary(self) -> List[Dict[str, Any]]:
        rows = []
        for group in self.group_indices:
            state = self.group_states[group]
            rows.append(metrics_row(state, self.args, global_round=-1, selected=False))
        rows.sort(key=lambda row: int(row["coarse_group"]))
        return rows

    def write_outputs(self) -> None:
        final_rows = self.final_candidates()
        group_rows = self.group_summary()
        write_csv(self.run_root / "evaluations.csv", self.evaluation_rows, overwrite=self.args.overwrite)
        write_csv(self.run_root / "global_rounds.csv", self.round_rows, overwrite=self.args.overwrite)
        write_csv(self.run_root / "group_summary.csv", group_rows, overwrite=self.args.overwrite)
        selected_fields = [
            "coarse_group",
            "candidate_index",
            "score_valid",
            "R_dyn",
            "R_sta",
            "risk_source",
            "risk_target",
            "graph_path",
            "n",
            "hard_count",
            "eta",
            "initial_candidate_index",
            "initial_score",
        ]
        write_csv(
            self.run_root / "selected_candidates.csv",
            final_rows,
            overwrite=self.args.overwrite,
            fieldnames=selected_fields,
        )

        selected_indices = np.asarray([row["candidate_index"] for row in final_rows], dtype=np.int64)
        selected_trajectories = self.candidates[selected_indices] if len(selected_indices) else np.zeros(
            (0, self.candidates.shape[1], self.candidates.shape[2]),
            dtype=self.candidates.dtype,
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        write_npy(self.run_root / "selected_candidate_indices.npy", selected_indices, overwrite=self.args.overwrite)
        write_npy(
            self.run_root / "selected_candidate_trajectories.npy",
            selected_trajectories,
            overwrite=self.args.overwrite,
        )

        payload = {
            "schema_version": "hierarchical_greedy_search_v1",
            "road_block_name": self.config.road_block_name,
            "central_token": self.central_token,
            "search_rule": {
                "F_valid": "max(R_dyn, R_sta) if D_hard == 0, else 0",
                "D_hard": "1 if any transition edge has collision/violation terminal event or hard-event reason.",
                "A": "omega_B*B + omega_I*I - omega_S*S - omega_H*eta",
                "B0": "valid score of the coarse representative member; B0 does not increment n_k,t.",
                "S": "Saturation uses the per-group L_prime in the count term and local_window only for I.",
                "local_phase_policy": "Representative sub-cluster medoids are tested before local expansion; groups with fewer sub-clusters than q_max enter expansion in the same selected local block.",
            },
            "parameters": {
                "total_iterations": int(self.args.total_iterations),
                "q_max": int(self.args.q_max),
                "group_eval_share_cap": float(getattr(self.args, "group_eval_share_cap", 0.0)),
                "local_window": int(self.args.local_window),
                "saturation_window": float(self.args.saturation_window),
                "saturation_window_mode": str(getattr(self.args, "saturation_window_mode", "fixed")),
                "saturation_windows_by_group": {str(group): float(value) for group, value in self.saturation_windows_by_group.items()},
                "subcluster_counts_by_group": {str(group): int(value) for group, value in self.subcluster_counts_by_group.items()},
                "total_subcluster_count": int(self.total_subcluster_count),
                "representative_testing_blocks_by_group": {str(group): int(value) for group, value in self.representative_blocks_by_group.items()},
                "omega_b": float(self.args.omega_b),
                "omega_i": float(self.args.omega_i),
                "omega_s": float(self.args.omega_s),
                "omega_h": float(self.args.omega_h),
                "epsilon": float(self.args.epsilon),
                "saturation_temperature": float(self.args.saturation_temperature),
                "improvement_rho": float(self.args.improvement_rho),
            },
            "selected_candidates": final_rows,
            "group_summary": group_rows,
        }
        write_json(self.run_root / "search_result.json", payload, overwrite=self.args.overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", type=Path, nargs="*", default=[], help="One or more FrameCentralConfig YAML files.")
    parser.add_argument("--config-glob", type=str, default=None, help="Optional glob for many FrameCentralConfig YAML files.")
    parser.add_argument("--tokens", nargs="*", default=None, help="Central tokens. Defaults to each config's central_tokens.")
    parser.add_argument("--navsim-log-path", required=True, type=Path, help="Directory containing NAVSIM log pickles.")

    parser.add_argument("--candidate-trajectory-path", type=Path, default=Path("trajectory_candidate/human_trajectory_candidate.npy"))
    parser.add_argument("--coarse-labels-path", type=Path, default=Path("trajectory_candidate/coarse_group_16_kmeans_first20pts_sub20_labels.npy"))
    parser.add_argument(
        "--coarse-representatives-path",
        type=Path,
        default=Path("trajectory_candidate/coarse_group_16_kmeans_first20pts_sub20_representatives.npy"),
    )
    parser.add_argument(
        "--sub-labels-path",
        type=Path,
        default=Path("trajectory_candidate/coarse_group_16_kmeans_first20pts_sub20_sub_labels.npy"),
    )
    parser.add_argument(
        "--sub-representatives-path",
        type=Path,
        default=Path("trajectory_candidate/coarse_group_16_kmeans_first20pts_sub20_sub_representatives.npy"),
    )
    parser.add_argument("--groups", type=int, nargs="*", default=None, help="Optional coarse group ids to search.")
    parser.add_argument("--representative-points", type=int, default=15, help="Number of points used to match medoids to candidates.")

    parser.add_argument("--total-iterations", type=int, default=160, help="Local action evaluations after the 16-group bootstrap.")
    parser.add_argument("--q-max", type=int, default=20, help="Local evaluations per selected coarse group.")
    parser.add_argument(
        "--group-eval-share-cap",
        type=float,
        default=0.0,
        help="Optional max share of local evaluations for one coarse group. 0 disables the cap.",
    )
    parser.add_argument("--local-window", type=int, default=20, help="L in the group-wise improvement window for I.")
    parser.add_argument(
        "--saturation-window",
        type=float,
        default=0.0,
        help="L_prime in the saturation count term. 0 uses total_iterations / number_of_groups.",
    )
    parser.add_argument("--omega-b", type=float, default=1.0)
    parser.add_argument("--omega-i", type=float, default=1.0)
    parser.add_argument("--omega-s", type=float, default=1.0)
    parser.add_argument("--omega-h", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--saturation-temperature", type=float, default=5.0, help="T_s in the saturation penalty.")
    parser.add_argument("--improvement-rho", type=float, default=0.02, help="rho in the saturation penalty.")

    parser.add_argument("--candidate-frame-rate", type=float, default=10.0)
    parser.add_argument("--candidate-prepend-origin", type=gsg.str_to_bool, default=True)
    parser.add_argument("--candidate-use-yaw", type=gsg.str_to_bool, default=True)
    parser.add_argument("--candidate-replace-mode", choices=("future", "history_first_half"), default="history_first_half")
    parser.add_argument("--candidate-history-duration", type=float, default=1.5)
    parser.add_argument("--candidate-z-mode", choices=("distance", "constant", "time"), default="time")

    parser.add_argument("--history-frames", type=int, default=4)
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
    parser.add_argument("--dynamic-object-types", type=str, default=",".join(gsg.DEFAULT_DYNAMIC_OBJECT_TYPES))
    parser.add_argument("--excluded-object-types", type=str, default=",".join(gsg.DEFAULT_EXCLUDED_TYPES))
    parser.add_argument("--skip-static-graph", type=gsg.str_to_bool, default=False)

    parser.add_argument("--ttc-horizon", type=float, default=7.0)
    parser.add_argument("--tlc-horizon", type=float, default=4.0)
    parser.add_argument("--risk-step", type=float, default=0.1)
    parser.add_argument("--ttc-tau-min", type=float, default=1.0)
    parser.add_argument("--tlc-tau-min", type=float, default=1.0)
    parser.add_argument("--ttc-alpha", type=float, default=0.30)
    parser.add_argument("--tlc-alpha", type=float, default=0.30)
    parser.add_argument("--output-detail", choices=("transition", "full"), default="transition")
    parser.add_argument(
        "--evaluation-mode",
        choices=("scene_graph", "risk_only"),
        default="scene_graph",
        help=(
            "Use the standard scene-graph builder or a compact exact risk-only evaluator. "
            "risk_only is used when --write-graphs false; graph writing falls back to scene_graph."
        ),
    )

    parser.add_argument("--output-root", type=Path, default=Path("Hierarchical Greedy Search/hierarchical_greedy_search"))
    parser.add_argument("--write-graphs", type=gsg.str_to_bool, default=True)
    parser.add_argument("--overwrite", type=gsg.str_to_bool, default=False)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_iterations < 0:
        raise ValueError("--total-iterations must be non-negative.")
    if args.q_max <= 0:
        raise ValueError("--q-max must be positive.")
    if args.group_eval_share_cap < 0.0 or args.group_eval_share_cap > 1.0:
        raise ValueError("--group-eval-share-cap must be in [0, 1].")
    if args.local_window <= 0:
        raise ValueError("--local-window must be positive.")
    if args.saturation_window < 0.0:
        raise ValueError("--saturation-window must be non-negative.")
    if args.saturation_temperature <= 0:
        raise ValueError("--saturation-temperature must be positive.")
    if args.improvement_rho <= 0:
        raise ValueError("--improvement-rho must be positive.")
    if args.representative_points <= 0:
        raise ValueError("--representative-points must be positive.")
    if args.risk_step <= 0:
        raise ValueError("--risk-step must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.history_frames != 4:
        raise ValueError("This search expects --history-frames 4, matching the 0-to-3 transition graph.")


def run_for_config_token(
    *,
    config_path: Path,
    config: FrameCentralConfig,
    central_token: str,
    log_frames: Sequence[Dict[str, Any]],
    footprint: gsg.EgoFootprintParams,
    map_api: Optional[Any],
    candidates: np.ndarray,
    coarse_labels: np.ndarray,
    coarse_reps: np.ndarray,
    sub_labels: np.ndarray,
    sub_reps: np.ndarray,
    group_indices: Sequence[int],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    search = HierarchicalGreedySearch(
        config=config,
        config_path=config_path,
        central_token=central_token,
        log_frames=log_frames,
        footprint=footprint,
        map_api=map_api,
        candidates=candidates,
        coarse_labels=coarse_labels,
        coarse_reps=coarse_reps,
        sub_labels=sub_labels,
        sub_reps=sub_reps,
        group_indices=group_indices,
        output_root=output_root,
        args=args,
    )
    search.run()
    search.write_outputs()
    final_count = len(search.final_candidates())
    print(
        f"{config.road_block_name} | token {central_token}: "
        f"{search.local_eval_counter} local evaluations, {final_count} selected valid candidate(s)."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    candidates = load_candidate_bank(args.candidate_trajectory_path)
    coarse_labels = load_labels(args.coarse_labels_path, candidates.shape[0], "coarse labels")
    sub_labels = load_labels(args.sub_labels_path, candidates.shape[0], "sub labels")
    coarse_reps = load_representatives(args.coarse_representatives_path, expected_ndim0=None, name="coarse representatives")
    sub_reps = load_representatives(args.sub_representatives_path, expected_ndim0=None, name="sub representatives")
    group_indices = selected_groups(coarse_reps.shape[0], args.groups)

    config_paths = gsg.load_configs(args.config, args.config_glob)
    footprint = gsg.load_ego_footprint()
    output_root = resolve(args.output_root)

    log_cache: Dict[str, List[Dict[str, Any]]] = {}
    map_cache: Dict[str, Any] = {}

    for config_path in config_paths:
        config = load_config(os.fspath(config_path))
        if not isinstance(config, FrameCentralConfig):
            raise TypeError(f"{config_path} is not a FrameCentralConfig.")
        if config.central_log not in log_cache:
            log_cache[config.central_log] = gsg.load_navsim_log(args.navsim_log_path, config.central_log)
        log_frames = log_cache[config.central_log]

        map_api = None
        if not args.skip_static_graph:
            if config.city not in map_cache:
                map_cache[config.city] = gsg.load_map_api(config.city)
            map_api = map_cache[config.city]

        for central_token in gsg.selected_tokens_for_config(config, args.tokens):
            run_for_config_token(
                config_path=config_path,
                config=config,
                central_token=central_token,
                log_frames=log_frames,
                footprint=footprint,
                map_api=map_api,
                candidates=candidates,
                coarse_labels=coarse_labels,
                coarse_reps=coarse_reps,
                sub_labels=sub_labels,
                sub_reps=sub_reps,
                group_indices=group_indices,
                output_root=output_root,
                args=args,
            )


if __name__ == "__main__":
    main()
