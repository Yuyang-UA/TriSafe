#!/usr/bin/env python3
"""Private raw-scene solvability oracle for synthetic NAVSIM scenes.

This command is deliberately independent of public NAVSIM metric caching.  It
loads each synthetic Scene pickle directly, constructs tracks/map/traffic
lights in memory, and never opens or writes ``metric_cache.pkl``.  It is only
for deciding whether a constructed test case has an operationally safe witness.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Dataset locations default to <repo>/../dataset; override with OPENSCENE_DATA_ROOT / NUPLAN_MAPS_ROOT.
DEFAULT_DATASET_ROOT = Path(os.environ.get("OPENSCENE_DATA_ROOT", Path(__file__).resolve().parents[1].parent / "dataset"))
DEFAULT_PACKAGE_ROOT = DEFAULT_DATASET_ROOT / "navsim_replaced"
DEFAULT_SCENES_PATH = DEFAULT_PACKAGE_ROOT / "synthetic_scene_pickles"
DEFAULT_MAP_ROOT = Path(os.environ.get("NUPLAN_MAPS_ROOT", DEFAULT_DATASET_ROOT / "maps"))
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_ROOT / "raw_reactive_idm_solvability"

# Scene.load_from_disk resolves its map API during import from this variable.
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(DEFAULT_MAP_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.planning.metric_caching.metric_cache import MapParameters
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents


SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
MAP_RADIUS_M = 100.0
RECOVERY_OFFSETS_M = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
CATEGORY_FILES = {
    "nc_1_dac_1": "synthetic_tokens_nc_1_dac_1.txt",
    "nc_0_or_0_5_dac_1": "synthetic_tokens_nc_0_or_0_5_dac_1.txt",
    "nc_1_dac_0": "synthetic_tokens_nc_1_dac_0.txt",
    "nc_0_or_0_5_dac_0": "synthetic_tokens_nc_0_or_0_5_dac_0.txt",
}

# Each process owns one helper and therefore does not share scenario/map state
# with other workers. The helper is used only for raw-scene interpolation; its
# cache path is always ``None``.
_WORKER_RAW_HELPER: Optional[MetricCacheProcessor] = None


@dataclass
class RawSceneContext:
    token: str
    ego_state: object
    planner_input: PlannerInput
    planner_initialization: PlannerInitialization
    interpolated_detection_tracks: List[object]
    interpolated_traffic_light_status: List[list]
    current_tracked_objects: List[object]
    future_tracked_objects: List[object]
    past_human_trajectory: InterpolatedTrajectory
    map_parameters: MapParameters


@dataclass
class Candidate:
    label: str
    iteration: int
    states: np.ndarray
    reactive_tracks: List[object]
    nc: float
    dac: float
    ttc: float
    progress_m: float
    comfort: float

    @property
    def rank(self) -> Tuple[float, float, float, float, float, float]:
        """Joint NC/DAC first, then the remaining requested five metrics."""
        joint_safety = self.nc * self.dac
        return (
            joint_safety,
            min(self.nc, self.dac),
            self.nc + self.dac,
            self.ttc,
            self.progress_m,
            self.comfort,
        )


@dataclass
class AuditRow:
    token: str
    status: str
    category: str
    selected_candidate: str = ""
    selected_iteration: Optional[int] = None
    evaluated_candidates: Optional[int] = None
    no_at_fault_collisions: Optional[float] = None
    drivable_area_compliance: Optional[float] = None
    time_to_collision_within_bound: Optional[float] = None
    raw_route_progress_m: Optional[float] = None
    history_comfort: Optional[float] = None
    error: str = ""


@dataclass
class SceneAuditResult:
    """One scene result, with a witness trajectory only when it is solvable."""

    row: AuditRow
    solution_xyz_heading: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-scenes-path", type=Path, default=DEFAULT_SCENES_PATH)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent scene workers. Use 1 to run sequentially (default).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_traffic_policy() -> NavsimIDMTrafficAgents:
    return NavsimIDMTrafficAgents(
        SAMPLING,
        NavsimIDMAgents(
            target_velocity=10.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=2.0,
            open_loop_detections_types=[],
            minimum_path_length=20.0,
            planned_trajectory_samples=None,
            planned_trajectory_sample_interval=None,
            radius=100.0,
            add_open_loop_parked_vehicles=True,
            idm_snap_threshold=3.0,
        ),
    )


def build_scorer() -> PDMScorer:
    """The requested five metrics only: NC, DAC, TTC, progress, comfort."""
    return PDMScorer(
        SAMPLING,
        PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            lane_keeping_weight=0.0,
            history_comfort_weight=2.0,
            two_frame_extended_comfort_weight=0.0,
            human_penalty_filter=False,
            five_metric_only=True,
        ),
    )


def make_raw_context(scene_path: Path, map_root: Path, raw_helper: MetricCacheProcessor) -> RawSceneContext:
    """Build raw inputs only; no MetricCache is loaded or dumped."""
    scene = Scene.load_from_disk(scene_path, None, SensorConfig.build_no_sensors())
    scenario = NavSimScenario(scene, map_root=str(map_root), map_version="nuplan-maps-v1.0")
    planner_input, planner_initialization = raw_helper._get_planner_inputs(scenario)
    detections = raw_helper._interpolate_gt_observation(scenario)
    traffic_lights = raw_helper._interpolate_traffic_light_status(scenario)
    past_ego = InterpolatedTrajectory(list(scenario.get_ego_past_trajectory(0, 1.5)))
    return RawSceneContext(
        token=scenario.token,
        ego_state=scenario.initial_ego_state,
        planner_input=planner_input,
        planner_initialization=planner_initialization,
        interpolated_detection_tracks=detections,
        interpolated_traffic_light_status=traffic_lights,
        current_tracked_objects=[scenario.initial_tracked_objects],
        future_tracked_objects=detections[1:],
        past_human_trajectory=past_ego,
        map_parameters=MapParameters(scenario.map_root, scenario.map_version, scenario.map_api.map_name),
    )


def scalar(result, name: str) -> float:
    return float(result[name].iloc[0])


def equals(value: float, expected: float) -> bool:
    return abs(value - expected) <= 1e-9


def category(nc: float, dac: float) -> str:
    if equals(nc, 1.0) and equals(dac, 1.0):
        return "nc_1_dac_1"
    if (equals(nc, 0.0) or equals(nc, 0.5)) and equals(dac, 1.0):
        return "nc_0_or_0_5_dac_1"
    if equals(nc, 1.0) and equals(dac, 0.0):
        return "nc_1_dac_0"
    if (equals(nc, 0.0) or equals(nc, 0.5)) and equals(dac, 0.0):
        return "nc_0_or_0_5_dac_0"
    return "unexpected_metric_values"


class ReactiveIDMSolvabilityPlanner:
    """Private PDM candidate oracle, separate from public PDMClosedPlanner use."""

    def __init__(self) -> None:
        self.simulator = PDMSimulator(SAMPLING)
        self.planner = PDMClosedPlanner(
            trajectory_sampling=SAMPLING,
            proposal_sampling=SAMPLING,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.5,
                decel_max=3.0,
            ),
            lateral_offsets=RECOVERY_OFFSETS_M,
            map_radius=MAP_RADIUS_M,
        )

    def prepare(self, context: RawSceneContext) -> None:
        self.planner.initialize(context.planner_initialization)
        self.planner._route_roadblock_correction(context.ego_state)
        self.planner._drivable_area_map = PDMDrivableMap.from_simulation(
            self.planner._map_api, context.ego_state, MAP_RADIUS_M
        )
        self.planner._update_proposal_manager(context.ego_state)

    def new_observation(self, context: RawSceneContext, tracks: Optional[List[object]]) -> PDMObservation:
        observation = PDMObservation(
            SAMPLING, SAMPLING, MAP_RADIUS_M, observation_sample_res=1, extend_observation_for_ttc=False
        )
        if tracks is None:
            observation.update(
                context.ego_state,
                context.planner_input.history.current_state[1],
                context.planner_input.traffic_light_data,
                self.planner._route_lane_dict,
            )
        else:
            observation.update_detections_tracks(
                tracks,
                context.interpolated_traffic_light_status,
                self.planner._route_lane_dict,
                compute_traffic_light_data=True,
            )
        return observation

    def pdm_candidates(
        self, context: RawSceneContext, planning_observation: PDMObservation, iteration: int
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
                offset = RECOVERY_OFFSETS_M[proposal.lateral_idx - 1]
                path_name = f"recovery_offset_{offset:+.1f}m"
            candidates.append((f"iter{iteration}_{path_name}_speed_policy_{proposal.longitudinal_idx}", simulated[index]))
        return candidates

    def brake_candidate(self, ego_state, deceleration: float) -> np.ndarray:
        """Straight, dynamically simulated brake-to-stop fallback."""
        initial = ego_state_to_state_array(ego_state)
        reference = np.repeat(initial[None, :], SAMPLING.num_poses + 1, axis=0)
        heading = float(initial[StateIndex.HEADING])
        # PDM state velocities/accelerations are in the ego (rear-axle) frame,
        # so the forward speed is VELOCITY_X; it must not be projected onto the
        # global heading.
        speed0 = max(0.0, float(initial[StateIndex.VELOCITY_X]))
        for index in range(1, SAMPLING.num_poses + 1):
            time_s = index * SAMPLING.interval_length
            active_time = min(time_s, speed0 / deceleration)
            distance = speed0 * active_time - 0.5 * deceleration * active_time**2
            speed = max(0.0, speed0 - deceleration * time_s)
            accel = -deceleration if speed > 0.0 else 0.0
            reference[index, StateIndex.X] = initial[StateIndex.X] + distance * np.cos(heading)
            reference[index, StateIndex.Y] = initial[StateIndex.Y] + distance * np.sin(heading)
            reference[index, StateIndex.HEADING] = heading
            reference[index, StateIndex.VELOCITY_X] = speed
            reference[index, StateIndex.VELOCITY_Y] = 0.0
            reference[index, StateIndex.ACCELERATION_X] = accel
            reference[index, StateIndex.ACCELERATION_Y] = 0.0
        return self.simulator.simulate_proposals(reference[None, ...], ego_state)[0]

    def emergency_candidates(self, context: RawSceneContext, iteration: int) -> List[Tuple[str, np.ndarray]]:
        return [
            (f"iter{iteration}_controlled_brake", self.brake_candidate(context.ego_state, 3.0)),
            (f"iter{iteration}_emergency_brake", self.brake_candidate(context.ego_state, 6.0)),
        ]

    def score(self, context: RawSceneContext, label: str, iteration: int, states: np.ndarray) -> Candidate:
        """Candidate-conditioned reactive IDM, then NC/DAC/TTC/progress/comfort."""
        reactive_tracks = build_traffic_policy().simulate_environment(states, context)
        observation = self.new_observation(context, context.interpolated_detection_tracks)
        scorer = build_scorer()
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
        return Candidate(
            label=label,
            iteration=iteration,
            states=states,
            reactive_tracks=reactive_tracks,
            nc=scalar(result, "no_at_fault_collisions"),
            dac=scalar(result, "drivable_area_compliance"),
            ttc=scalar(result, "time_to_collision_within_bound"),
            progress_m=float(scorer._progress_raw[0]),
            comfort=scalar(result, "history_comfort"),
        )

    def solve(self, context: RawSceneContext, iterations: int) -> Tuple[Candidate, int]:
        self.prepare(context)
        prior_tracks: Optional[List[object]] = None
        carry: Optional[Candidate] = None
        best: Optional[Candidate] = None
        count = 0
        for iteration in range(iterations):
            candidates = self.pdm_candidates(context, self.new_observation(context, prior_tracks), iteration)
            candidates.extend(self.emergency_candidates(context, iteration))
            if carry is not None:
                candidates.append((f"iter{iteration}_carry_forward_{carry.label}", carry.states))
            evaluated = [self.score(context, label, iteration, states) for label, states in candidates]
            count += len(evaluated)
            iteration_best = max(evaluated, key=lambda candidate: candidate.rank)
            if best is None or iteration_best.rank > best.rank:
                best = iteration_best
            carry = best
            # Best response traffic becomes the next iteration's forecast.
            prior_tracks = iteration_best.reactive_tracks
        assert best is not None
        return best, count


def witness_xyz_heading(states: np.ndarray) -> np.ndarray:
    """Return the selected 10 Hz ego witness as columns x, y, and heading."""
    return np.column_stack(
        (
            states[:, StateIndex.X],
            states[:, StateIndex.Y],
            states[:, StateIndex.HEADING],
        )
    ).astype(np.float64, copy=False)


def audit_scene(
    path: Path, map_root: Path, raw_helper: MetricCacheProcessor, iterations: int
) -> SceneAuditResult:
    try:
        context = make_raw_context(path, map_root, raw_helper)
        best, count = ReactiveIDMSolvabilityPlanner().solve(context, iterations)
        row = AuditRow(
            token=context.token,
            status="scored",
            category=category(best.nc, best.dac),
            selected_candidate=best.label,
            selected_iteration=best.iteration,
            evaluated_candidates=count,
            no_at_fault_collisions=best.nc,
            drivable_area_compliance=best.dac,
            time_to_collision_within_bound=best.ttc,
            raw_route_progress_m=best.progress_m,
            history_comfort=best.comfort,
        )
        # NC == 1 and DAC == 1 is the requested definition of a solvable case.
        solution = witness_xyz_heading(best.states) if row.category == "nc_1_dac_1" else None
        return SceneAuditResult(row=row, solution_xyz_heading=solution)
    except Exception as exc:
        return SceneAuditResult(
            AuditRow(path.stem, "error", "unscored_error", error=f"{type(exc).__name__}: {exc}")
        )
    finally:
        gc.collect()


def initialize_worker(map_root: str) -> None:
    """Create process-local raw interpolation state once per scene worker."""
    os.environ["NUPLAN_MAPS_ROOT"] = map_root
    global _WORKER_RAW_HELPER
    _WORKER_RAW_HELPER = MetricCacheProcessor(
        cache_path=None,
        force_feature_computation=False,
        proposal_sampling=SAMPLING,
    )


def audit_scene_worker(index: int, scene_path: str, map_root: str, iterations: int) -> Tuple[int, SceneAuditResult]:
    """Pickle-friendly worker entrypoint; all output is merged by the parent."""
    if _WORKER_RAW_HELPER is None:
        raise RuntimeError("Raw-scene worker was not initialized")
    return index, audit_scene(Path(scene_path), Path(map_root), _WORKER_RAW_HELPER, iterations)


def read_tokens(path: Optional[Path]) -> Optional[set]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Token file does not exist: {path}")
    tokens = {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}
    if not tokens:
        raise ValueError(f"Token file is empty: {path}")
    return tokens


def write_list(path: Path, tokens: Iterable[str]) -> None:
    values = sorted(tokens)
    path.write_text("\n".join(values) + ("\n" if values else ""))


def write_solvable_trajectories(output_dir: Path, results: Sequence[SceneAuditResult]) -> int:
    """Write one selected trajectory solution for every NC=1, DAC=1 scene."""
    trajectory_dir = output_dir / "solvable_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    index_rows: List[Dict[str, object]] = []
    for result in results:
        row = result.row
        if row.status != "scored" or row.category != "nc_1_dac_1":
            continue
        if result.solution_xyz_heading is None:
            raise RuntimeError(f"Missing solvable trajectory for token {row.token}")
        output_path = trajectory_dir / f"{row.token}.csv"
        with output_path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["step", "time_s", "x", "y", "heading"])
            writer.writeheader()
            for step, (x, y, heading) in enumerate(result.solution_xyz_heading):
                writer.writerow(
                    {
                        "step": step,
                        "time_s": step * SAMPLING.interval_length,
                        "x": float(x),
                        "y": float(y),
                        "heading": float(heading),
                    }
                )
        index_rows.append(
            {
                "token": row.token,
                "selected_candidate": row.selected_candidate,
                "selected_iteration": row.selected_iteration,
                "trajectory_file": str(output_path.relative_to(output_dir)),
            }
        )

    with (output_dir / "solvable_trajectory_index.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["token", "selected_candidate", "selected_iteration", "trajectory_file"],
        )
        writer.writeheader()
        writer.writerows(index_rows)
    return len(index_rows)


def write_outputs(
    output_dir: Path, results: Sequence[SceneAuditResult], args: argparse.Namespace, effective_workers: int
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result.row for result in results]
    token_groups: Dict[str, List[str]] = {key: [] for key in CATEGORY_FILES}
    for row in rows:
        if row.status == "scored" and row.category in token_groups:
            token_groups[row.category].append(row.token)
    for name, filename in CATEGORY_FILES.items():
        write_list(output_dir / filename, token_groups[name])
    write_list(output_dir / "synthetic_tokens_unscored_errors.txt", [row.token for row in rows if row.status == "error"])

    with (output_dir / "raw_reactive_idm_solvability.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(AuditRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    solvable_trajectory_count = write_solvable_trajectories(output_dir, results)
    counts = {
        "scenes_seen": len(rows),
        "scored": sum(row.status == "scored" for row in rows),
        "unscored_errors": sum(row.status == "error" for row in rows),
        "solvable_trajectory_solutions": solvable_trajectory_count,
        **{name: len(tokens) for name, tokens in token_groups.items()},
    }
    manifest = {
        "description": "Private raw-scene solvability oracle; public metric-cache and benchmark code are unchanged.",
        "metric_cache_read": False,
        "metric_cache_written": False,
        "raw_synthetic_scenes_path": str(args.synthetic_scenes_path),
        "map_root": str(args.map_root),
        "iterations": args.iterations,
        "requested_workers": args.workers,
        "effective_workers": effective_workers,
        "execution": {
            "result_merge": "Parent process sorts worker results by input scene order before writing outputs.",
            "trajectory_format": "solvable_trajectories/<token>.csv: 10 Hz x, y, heading witness",
        },
        "candidate_policy": {
            "recovery_offsets_m": RECOVERY_OFFSETS_M,
            "braking_decelerations_mps2": [3.0, 6.0],
            "traffic": "candidate-conditioned NavsimIDMTrafficAgents",
            "ranking": "joint NC/DAC, then TTC, raw progress, comfort",
            "five_metric_only": True,
        },
        "counts": counts,
    }
    (output_dir / "raw_reactive_idm_solvability_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
        selected_tokens = read_tokens(args.tokens_file)
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
    results_by_index: Dict[int, SceneAuditResult] = {}
    if worker_count == 1:
        # Raw interpolation helper only; it is never asked to compute/dump a cache.
        raw_helper = MetricCacheProcessor(cache_path=None, force_feature_computation=False, proposal_sampling=SAMPLING)
        for index, path in enumerate(paths):
            results_by_index[index] = audit_scene(path, args.map_root, raw_helper, args.iterations)
            row = results_by_index[index].row
            if args.verbose:
                logging.info("%s: %s (%s)", row.token, row.category, row.selected_candidate)
            logging.info("Processed %d/%d scenes", index + 1, len(paths))
    else:
        logging.info("Starting %d independent scene workers.", worker_count)
        # Spawn avoids inheriting parent process state held by map/planning libraries.
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=initialize_worker,
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
                    results_by_index[index] = SceneAuditResult(
                        AuditRow(
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
    logging.info("Raw reactive-IDM audit complete: %s", args.output_dir)
    logging.info("Counts: %s", json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
