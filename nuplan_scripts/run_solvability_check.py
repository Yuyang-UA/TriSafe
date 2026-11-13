#!/usr/bin/env python3
"""Reactive-IDM solvability check with adaptive lateral proposals.

This private raw-scene oracle reads synthetic Scene pickles directly and does
not read or write NAVSIM metric caches.  It differs from
raw_reactive_idm_solvability.py in three deliberate ways:

* lateral offsets are derived from the current route-lane geometry;
* the brake fallbacks are the 3 m/s² IDM-consistent trajectory and a
  4.05 m/s² trajectory at the NAVSIM comfort deceleration bound; and
* several ego/traffic best-response rounds are performed when --iterations
  is greater than one.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


# Dataset locations default to <repo>/../dataset; override with OPENSCENE_DATA_ROOT / NUPLAN_MAPS_ROOT.
DEFAULT_DATASET_ROOT = Path(os.environ.get("OPENSCENE_DATA_ROOT", Path(__file__).resolve().parents[1].parent / "dataset"))
DEFAULT_PACKAGE_ROOT = DEFAULT_DATASET_ROOT / "navsim_replaced"
DEFAULT_SCENES_PATH = DEFAULT_PACKAGE_ROOT / "synthetic_scene_pickles"
DEFAULT_MAP_ROOT = Path(os.environ.get("NUPLAN_MAPS_ROOT", DEFAULT_DATASET_ROOT / "maps"))
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_ROOT / "solvability_check_new"

# Scene.load_from_disk resolves its map API during import from this variable.
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(DEFAULT_MAP_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import raw_reactive_idm_solvability as base


LATERAL_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0, 1.0)
MIN_LATERAL_OFFSET_M = 1.0
LATERAL_RAY_LENGTH_M = 40.0
DEFAULT_LATERAL_REACH_M = 3.0
NEIGHBOR_LANE_MAX_GAP_M = 0.5
NEIGHBOR_LANE_PROBE_M = 0.05
DRIVABLE_BOUNDARY_MARGIN_M = 0.5
MAX_LATERAL_OFFSET_M = 6.0
IDM_EGO_MAX_DECEL_MPS2 = 3.0
# NAVSIM comfort lower bound on longitudinal acceleration (pdm_comfort_metrics.MIN_LON_ACCEL).
COMFORT_MAX_DECEL_MPS2 = 4.05
BRAKE_DECELERATIONS_MPS2 = (IDM_EGO_MAX_DECEL_MPS2, COMFORT_MAX_DECEL_MPS2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-scenes-path", type=Path, default=DEFAULT_SCENES_PATH)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Ego/traffic best-response refinement rounds (default: 3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent scene workers. Use 1 to run sequentially (default).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _geometry_coordinates(geometry) -> Iterable[Tuple[float, float]]:
    """Yield endpoint coordinates from an arbitrary Shapely intersection."""
    if geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for member in geometry.geoms:
            yield from _geometry_coordinates(member)
        return
    if hasattr(geometry, "coords"):
        coordinates = list(geometry.coords)
        if coordinates:
            yield tuple(coordinates[0][:2])
            if len(coordinates) > 1:
                yield tuple(coordinates[-1][:2])


def _side_boundary_distance(
    lane_polygon, origin_xy: np.ndarray, left_normal_xy: np.ndarray, side: float
) -> Optional[float]:
    """Return the nearest positive lateral ray distance to one lane boundary."""
    end_xy = origin_xy + side * LATERAL_RAY_LENGTH_M * left_normal_xy
    intersection = LineString([tuple(origin_xy), tuple(end_xy)]).intersection(lane_polygon.boundary)
    distances = []
    for coordinate in _geometry_coordinates(intersection):
        signed_distance = side * float(np.dot(np.asarray(coordinate) - origin_xy, left_normal_xy))
        if signed_distance > 1e-4:
            distances.append(signed_distance)
    return min(distances) if distances else None


def _geometry_components(geometry) -> Iterable[object]:
    """Yield non-empty components of a Shapely geometry."""
    if geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for member in geometry.geoms:
            yield from _geometry_components(member)
        return
    yield geometry


def _connected_drivable_geometry(drivable_area_map, origin_xy: np.ndarray):
    """Return the drivable-area component containing the ego rear axle.

    Unioning the PDM drivable polygons removes internal lane boundaries.  The
    selected component therefore ends at an actual outer drivable boundary,
    while excluding disconnected nearby road fragments.
    """
    polygons = [drivable_area_map[token] for token in drivable_area_map.tokens]
    if not polygons:
        return None
    ego_point = Point(*origin_xy)
    containing_components = [
        component for component in _geometry_components(unary_union(polygons)) if component.covers(ego_point)
    ]
    if not containing_components:
        return None
    return max(containing_components, key=lambda component: component.area)


def _adjacent_lane_width(
    drivable_area_map, current_lane, origin_xy: np.ndarray, boundary_xy: np.ndarray, cross_section: LineString
) -> Optional[float]:
    """Return the local width of the lane immediately beyond one current-lane boundary."""

    current_lane_id = getattr(current_lane, "id", None)
    origin = Point(*origin_xy)
    probe = Point(*boundary_xy)
    lane_types = {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
    candidates = []
    for token, map_type in zip(drivable_area_map.tokens, drivable_area_map.map_types):
        if map_type not in lane_types or token == current_lane_id:
            continue
        polygon = drivable_area_map[token]
        if polygon.covers(origin):
            continue
        distance = polygon.distance(probe)
        if distance > NEIGHBOR_LANE_MAX_GAP_M:
            continue
        width_m = float(polygon.intersection(cross_section).length)
        if width_m > 0.0:
            candidates.append((distance, -width_m, width_m))
    return min(candidates)[2] if candidates else None


def _side_lateral_reach(
    lane, drivable_area_map, origin_xy: np.ndarray, left_normal_xy: np.ndarray, cross_section: LineString, side: float
) -> float:
    """Return current-lane clearance plus adjacent-lane width, or the 3 m fallback."""

    current_lane_clearance_m = _side_boundary_distance(lane.polygon, origin_xy, left_normal_xy, side)
    if current_lane_clearance_m is None:
        return DEFAULT_LATERAL_REACH_M
    boundary_xy = origin_xy + side * current_lane_clearance_m * left_normal_xy
    adjacent_lane_width_m = _adjacent_lane_width(
        drivable_area_map,
        lane,
        origin_xy,
        boundary_xy + side * NEIGHBOR_LANE_PROBE_M * left_normal_xy,
        cross_section,
    )
    if adjacent_lane_width_m is None:
        return DEFAULT_LATERAL_REACH_M
    return current_lane_clearance_m + adjacent_lane_width_m


def _local_lane_geometry(
    lane, drivable_area_map, ego_state
) -> Tuple[Optional[float], float, float, Optional[float], Optional[float]]:
    """Measure lane width and one-neighbour lateral reach on each side.

    Each side is the ego rear axle's clearance to its current-lane boundary
    plus the width of the immediately adjacent lane. This deliberately permits
    a one-lane evasion. If lane geometry is unavailable for a side, that side
    uses the requested 3 m default reach.
    """
    try:
        rear_axle = ego_state.rear_axle
        origin_xy = np.array([float(rear_axle.x), float(rear_axle.y)], dtype=np.float64)
        pose = lane.baseline_path.get_nearest_pose_from_position(rear_axle)
        left_normal_xy = np.array([-np.sin(float(pose.heading)), np.cos(float(pose.heading))], dtype=np.float64)
        centerline_xy = np.array([float(pose.x), float(pose.y)], dtype=np.float64)
        cross_section = LineString(
            [
                tuple(centerline_xy - LATERAL_RAY_LENGTH_M * left_normal_xy),
                tuple(centerline_xy + LATERAL_RAY_LENGTH_M * left_normal_xy),
            ]
        )
        lane_width_m = float(lane.polygon.intersection(cross_section).length)
        lane_width_m = lane_width_m if lane_width_m > 0.0 else None
        left_reach_m = _side_lateral_reach(
            lane, drivable_area_map, origin_xy, left_normal_xy, cross_section, side=1.0
        )

        right_reach_m = _side_lateral_reach(
            lane, drivable_area_map, origin_xy, left_normal_xy, cross_section, side=-1.0
        )
        connected_drivable_area = _connected_drivable_geometry(drivable_area_map, origin_xy)
        left_drivable_boundary_distance_m = None
        right_drivable_boundary_distance_m = None
        if connected_drivable_area is not None:
            left_drivable_boundary_distance_m = _side_boundary_distance(
                connected_drivable_area, origin_xy, left_normal_xy, side=1.0
            )
            right_drivable_boundary_distance_m = _side_boundary_distance(
                connected_drivable_area, origin_xy, left_normal_xy, side=-1.0
            )
        return lane_width_m, left_reach_m, right_reach_m, left_drivable_boundary_distance_m, right_drivable_boundary_distance_m
    except Exception:
        return None, DEFAULT_LATERAL_REACH_M, DEFAULT_LATERAL_REACH_M, None, None


def _side_offsets(
    lateral_reach_m: float, drivable_boundary_distance_m: Optional[float], sign: float
) -> List[float]:
    """Construct one side of the laterally feasible PDM offset set.

    A 1 m candidate minimum is applied before the outer drivable-boundary and
    6 m physical caps. Values that become equal after constraint are emitted
    only once.
    """
    available_m = lateral_reach_m
    boundary_distance_m = drivable_boundary_distance_m
    if (
        not np.isfinite(available_m)
        or boundary_distance_m is None
        or not np.isfinite(boundary_distance_m)
        or available_m < MIN_LATERAL_OFFSET_M
    ):
        return []
    maximum_magnitude_m = min(
        MAX_LATERAL_OFFSET_M,
        max(0.0, boundary_distance_m - DRIVABLE_BOUNDARY_MARGIN_M),
    )
    if maximum_magnitude_m < MIN_LATERAL_OFFSET_M:
        return []
    offsets: List[float] = []
    for alpha in LATERAL_FRACTIONS:
        unconstrained_magnitude_m = max(MIN_LATERAL_OFFSET_M, alpha * available_m)
        magnitude_m = min(unconstrained_magnitude_m, maximum_magnitude_m)
        if not any(abs(magnitude_m - abs(existing)) <= 1e-9 for existing in offsets):
            offsets.append(sign * magnitude_m)
    return offsets


class AdaptiveReactiveIDMSolvabilityPlanner(base.ReactiveIDMSolvabilityPlanner):
    """PDM proposal oracle with adaptive offsets and ego/traffic best responses."""

    def __init__(self) -> None:
        # The parent builds PDMClosed with static offsets.  Replace them before
        # proposal-manager construction in prepare(), once map geometry exists.
        super().__init__()
        self.planner._lateral_offsets = None
        self.lateral_offsets: List[float] = []
        self.lane_width_m: Optional[float] = None
        self.left_lateral_reach_m: float = DEFAULT_LATERAL_REACH_M
        self.right_lateral_reach_m: float = DEFAULT_LATERAL_REACH_M
        self.left_drivable_boundary_distance_m: Optional[float] = None
        self.right_drivable_boundary_distance_m: Optional[float] = None

    def prepare(self, context: base.RawSceneContext) -> None:
        self.planner.initialize(context.planner_initialization)
        self.planner._route_roadblock_correction(context.ego_state)
        self.planner._drivable_area_map = base.PDMDrivableMap.from_simulation(
            self.planner._map_api, context.ego_state, base.MAP_RADIUS_M
        )
        current_lane = self.planner._get_starting_lane(context.ego_state)
        if current_lane is None:
            raise RuntimeError("Unable to identify a route lane for adaptive lateral proposals")

        (
            self.lane_width_m,
            self.left_lateral_reach_m,
            self.right_lateral_reach_m,
            self.left_drivable_boundary_distance_m,
            self.right_drivable_boundary_distance_m,
        ) = _local_lane_geometry(current_lane, self.planner._drivable_area_map, context.ego_state)
        self.lateral_offsets = _side_offsets(
            self.left_lateral_reach_m, self.left_drivable_boundary_distance_m, sign=1.0
        )
        self.lateral_offsets.extend(
            _side_offsets(self.right_lateral_reach_m, self.right_drivable_boundary_distance_m, sign=-1.0)
        )

        # PDM paths are constant-offset paths; evaluate d_L(s) and d_R(s) at
        # the current planning state s, then rebuild the manager for this scene.
        self.planner._lateral_offsets = self.lateral_offsets
        self.planner._update_proposal_manager(context.ego_state)

    def pdm_candidates(
        self, context: base.RawSceneContext, planning_observation, iteration: int
    ) -> List[Tuple[str, np.ndarray]]:
        references = self.planner._generator.generate_proposals(
            context.ego_state, planning_observation, self.planner._proposal_manager
        )
        simulated = self.simulator.simulate_proposals(references, context.ego_state)
        candidates: List[Tuple[str, np.ndarray]] = []
        for index in range(simulated.shape[0]):
            proposal = self.planner._proposal_manager[index]
            if proposal.lateral_idx == 0:
                path_name = "route_center"
            else:
                offset = self.lateral_offsets[proposal.lateral_idx - 1]
                side = "left" if offset > 0.0 else "right"
                path_name = f"adaptive_{side}_{abs(offset):.3f}m"
            candidates.append((f"iter{iteration}_{path_name}_speed_policy_{proposal.longitudinal_idx}", simulated[index]))
        return candidates

    def emergency_candidates(self, context: base.RawSceneContext, iteration: int) -> List[Tuple[str, np.ndarray]]:
        """Straight brake fallbacks: IDM-consistent 3 m/s² and the 4.05 m/s² comfort bound."""
        return [
            (f"iter{iteration}_brake_{deceleration:.2f}mps2", self.brake_candidate(context.ego_state, deceleration))
            for deceleration in BRAKE_DECELERATIONS_MPS2
        ]

    def predict_participant_response(self, states: np.ndarray, context: base.RawSceneContext) -> List[object]:
        """Predict vehicles' IDM response to this exact ego candidate.

        simulate_environment inserts the ego footprint at every simulation step
        before propagating the traffic IDM agents.  Thus an interacting nearby
        participant sees this candidate ego plan as an obstacle/lead agent,
        rather than merely replaying its logged future motion.
        """
        return base.build_traffic_policy().simulate_environment(states, context)

    def score(self, context: base.RawSceneContext, label: str, iteration: int, states: np.ndarray) -> base.Candidate:
        """Score an ego candidate against its own reactive participant forecast."""
        reactive_tracks = self.predict_participant_response(states, context)
        observation = self.new_observation(context, context.interpolated_detection_tracks)
        scorer = base.build_scorer()
        result = scorer.score_proposals(
            states=states[None, ...],
            observation=observation,
            centerline=self.planner._centerline,
            route_lane_ids=list(self.planner._route_lane_dict.keys()),
            drivable_area_map=self.planner._drivable_area_map,
            map_parameters=context.map_parameters,
            simulated_agent_detections_tracks=reactive_tracks,
            human_past_trajectory=context.past_human_trajectory,
        )[0]
        return base.Candidate(
            label=label,
            iteration=iteration,
            states=states,
            reactive_tracks=reactive_tracks,
            nc=base.scalar(result, "no_at_fault_collisions"),
            dac=base.scalar(result, "drivable_area_compliance"),
            ttc=base.scalar(result, "time_to_collision_within_bound"),
            progress_m=float(scorer._progress_raw[0]),
            comfort=base.scalar(result, "history_comfort"),
        )

    def solve(self, context: base.RawSceneContext, iterations: int) -> Tuple[base.Candidate, int]:
        """Alternate ego optimisation and traffic IDM best responses.

        Round k plans against the traffic trajectory produced by the selected
        ego response in round k-1.  Each new ego candidate is then evaluated
        against a fresh participant response to that candidate.  This is a
        finite best-response refinement, not a log-replay prediction loop.
        """
        self.prepare(context)
        prior_tracks: Optional[List[object]] = None
        incumbent: Optional[base.Candidate] = None
        evaluated_count = 0

        for iteration in range(iterations):
            planning_observation = self.new_observation(context, prior_tracks)
            candidates = self.pdm_candidates(context, planning_observation, iteration)
            candidates.extend(self.emergency_candidates(context, iteration))
            if incumbent is not None:
                candidates.append((f"iter{iteration}_carry_forward_{incumbent.label}", incumbent.states))

            evaluated = [self.score(context, label, iteration, states) for label, states in candidates]
            evaluated_count += len(evaluated)
            round_best = max(evaluated, key=lambda candidate: candidate.rank)
            if incumbent is None or round_best.rank > incumbent.rank:
                incumbent = round_best
            # The selected ego plan induces the traffic forecast for the next
            # optimisation round; this is the iterative game coupling.
            prior_tracks = round_best.reactive_tracks

        assert incumbent is not None
        return incumbent, evaluated_count


def audit_scene(
    path: Path, map_root: Path, raw_helper: base.MetricCacheProcessor, iterations: int
) -> base.SceneAuditResult:
    try:
        context = base.make_raw_context(path, map_root, raw_helper)
        best, count = AdaptiveReactiveIDMSolvabilityPlanner().solve(context, iterations)
        row = base.AuditRow(
            token=context.token,
            status="scored",
            category=base.category(best.nc, best.dac),
            selected_candidate=best.label,
            selected_iteration=best.iteration,
            evaluated_candidates=count,
            no_at_fault_collisions=best.nc,
            drivable_area_compliance=best.dac,
            time_to_collision_within_bound=best.ttc,
            raw_route_progress_m=best.progress_m,
            history_comfort=best.comfort,
        )
        solution = base.witness_xyz_heading(best.states) if row.category == "nc_1_dac_1" else None
        return base.SceneAuditResult(row=row, solution_xyz_heading=solution)
    except Exception as exc:
        return base.SceneAuditResult(
            base.AuditRow(path.stem, "error", "unscored_error", error=f"{type(exc).__name__}: {exc}")
        )
    finally:
        gc.collect()


def audit_scene_worker(index: int, scene_path: str, map_root: str, iterations: int) -> Tuple[int, base.SceneAuditResult]:
    """Pickle-friendly worker entry point using one process-local raw helper."""
    if base._WORKER_RAW_HELPER is None:
        raise RuntimeError("Raw-scene worker was not initialized")
    return index, audit_scene(Path(scene_path), Path(map_root), base._WORKER_RAW_HELPER, iterations)


def write_outputs(
    output_dir: Path, results: Sequence[base.SceneAuditResult], args: argparse.Namespace, effective_workers: int
) -> Dict[str, int]:
    """Write the established CSV/token outputs and a truthful revised manifest."""
    counts = base.write_outputs(output_dir, results, args, effective_workers)
    summary_path = output_dir / "raw_reactive_idm_solvability_summary.json"
    manifest = json.loads(summary_path.read_text())
    manifest.update(
        {
            "description": "Private raw-scene solvability check with adaptive lateral proposals and iterative reactive-IDM refinement.",
            "implementation": str(Path(__file__).resolve()),
            "candidate_policy": {
                "lateral_offsets": {
                    "formula": "{alpha * d_L(s)} union {-alpha * d_R(s)}, alpha in {1/3, 2/3, 1}",
                    "constrained_formula": "sign(side) * min(max(1 m, alpha * d_side(s)), D_drivable,side(s) - 0.5 m, 6 m)",
                    "d_side(s)": "rear-axle-to-current-lane-boundary distance at s + width of immediately adjacent lane; 3 m if side lane geometry is unavailable",
                    "drivable_boundary": "outer boundary of the connected PDM drivable-area component containing the rear axle; candidates need at least a 0.5 m margin",
                    "evaluation_state": "current ego planning state; PDM proposal paths use constant lateral offsets",
                    "minimum_candidate_offset_m": MIN_LATERAL_OFFSET_M,
                    "maximum_candidate_offset_m": MAX_LATERAL_OFFSET_M,
                    "below_minimum": "raised to 1 m only when that side has at least 1 m clearance; otherwise omitted",
                    "duplicate_offsets": "equal offset magnitudes are removed after applying the drivable-boundary and 6 m caps",
                    "positive_direction": "left of route heading",
                },
                "braking_decelerations_mps2": list(BRAKE_DECELERATIONS_MPS2),
                "braking_constraint": "3.0 m/s² matches BatchIDMPolicy decel_max; 4.05 m/s² is the NAVSIM comfort bound MIN_LON_ACCEL",
                "brake_reference": "straight along the current heading from the ego-frame forward speed (rear_axle_velocity_2d.x)",
                "traffic": "candidate-conditioned NavsimIDMTrafficAgents; ego occupancy is included in each IDM propagation step",
                "ranking": "joint NC/DAC, then TTC, raw progress, comfort",
                "five_metric_only": True,
            },
            "iterative_refinement": {
                "rounds": args.iterations,
                "scheme": "plan against prior selected traffic response; score each ego candidate against its own fresh traffic IDM response; feed the round winner response into the next round",
                "carry_forward": "the incumbent ego candidate remains available in later rounds",
            },
        }
    )
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return counts


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.iterations <= 0 or args.workers <= 0 or (args.max_scenes is not None and args.max_scenes <= 0):
        logging.error("--iterations, --workers, and --max-scenes must be positive.")
        return 2
    if not args.synthetic_scenes_path.is_dir() or not args.map_root.is_dir():
        logging.error("Missing synthetic scenes or map directory.")
        return 2
    try:
        selected_tokens = base.read_tokens(args.tokens_file)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 2

    paths = sorted(args.synthetic_scenes_path.glob("*.pkl"))
    if selected_tokens is not None:
        paths = [path for path in paths if path.stem in selected_tokens]
    if args.max_scenes is not None:
        paths = paths[: args.max_scenes]
    if not paths:
        logging.error("No synthetic scenes selected.")
        return 2

    worker_count = min(args.workers, len(paths))
    results_by_index: Dict[int, base.SceneAuditResult] = {}
    if worker_count == 1:
        raw_helper = base.MetricCacheProcessor(
            cache_path=None, force_feature_computation=False, proposal_sampling=base.SAMPLING
        )
        for index, path in enumerate(paths):
            results_by_index[index] = audit_scene(path, args.map_root, raw_helper, args.iterations)
            row = results_by_index[index].row
            if args.verbose:
                logging.info("%s: %s (%s)", row.token, row.category, row.selected_candidate)
            logging.info("Processed %d/%d scenes", index + 1, len(paths))
    else:
        logging.info("Starting %d independent scene workers.", worker_count)
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=base.initialize_worker,
            initargs=(str(args.map_root),),
        ) as executor:
            future_to_index = {
                executor.submit(audit_scene_worker, index, str(path), str(args.map_root), args.iterations): index
                for index, path in enumerate(paths)
            }
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    returned_index, result = future.result()
                    if returned_index != index:
                        raise RuntimeError(f"Worker returned index {returned_index}, expected {index}")
                    results_by_index[index] = result
                except Exception as exc:
                    results_by_index[index] = base.SceneAuditResult(
                        base.AuditRow(
                            paths[index].stem,
                            "error",
                            "unscored_error",
                            error=f"WorkerFailure {type(exc).__name__}: {exc}",
                        )
                    )
                completed += 1
                row = results_by_index[index].row
                if args.verbose:
                    logging.info("%s: %s (%s)", row.token, row.category, row.selected_candidate)
                logging.info("Processed %d/%d scenes", completed, len(paths))

    results = [results_by_index[index] for index in range(len(paths))]
    counts = write_outputs(args.output_dir, results, args, worker_count)
    logging.info("Adaptive reactive-IDM solvability check complete: %s", args.output_dir)
    logging.info("Counts: %s", json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
