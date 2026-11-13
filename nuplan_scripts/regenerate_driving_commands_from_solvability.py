#!/usr/bin/env python3
"""Regenerate synthetic NAVSIM route commands from revised PDM-Closed witnesses.

The renderer initially copies ``driving_command`` from the original OpenScene
frame.  That command can be wrong for a safety-critical synthetic case whose
ego history/current state was changed by a selected trajectory candidate.

This utility uses the selected witness written by
``nuplan_scripts/run_solvability_check.py`` for every scene with ``NC=1`` and
``DAC=1``.
It applies NAVSIM's route-direction rule to the witness: evaluate its lateral
displacement at 20 m of traveled distance (or at the 4 s endpoint when the
witness is shorter), and classify it with the usual +/-2 m threshold.

A straight brake witness carries no route intent: it never moves laterally,
so the rule above would always call it ``straight`` even when the route turns.
For a solvable scene whose selected candidate is a brake, the command is
instead read from the PDM route centerline (the one the checker scores
progress against): from the ego's projection onto it, evaluate the
centerline's lateral displacement at the same look-ahead distance in the
centerline's own frame, with the same threshold.  If less than
``--min-route-length-m`` of centerline remains ahead of the ego, the witness
rule is used and the manifest marks the fallback.

Every other scene, including an unscored scene, receives the fourth one-hot
command, ``unknown``.  The result is written to every history-frame ego status
so a model cannot observe stale original route commands through its input
history.  The detailed manifest records the prior current-frame command and
the geometry used for each replacement.  Applied runs also write a flat
``synthetic_token_driving_commands.json`` mapping from scene tokens to these
same four-element one-hot vectors for evaluator input overrides.

The package may hold a subset of the checked scenes (for example the rendered
package from ``scripts/sensor_data_rendering.sh``, which contains the solvable
scenes only); every package scene must have a solvability row.

Example:

    python nuplan_scripts/regenerate_driving_commands_from_solvability.py \
        --synthetic-scenes-path path/to/rendered_data/synthetic_scene_pickles \
        --solvability-dir path/to/risky_cases/solvability_check \
        --manifest-path path/to/rendered_data/driving_command_regeneration_from_solvability.json \
        --commands-path path/to/rendered_data/synthetic_token_driving_commands.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import pickle
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MAP_ROOT = Path(
    os.environ.get("NUPLAN_MAPS_ROOT")
    or Path(os.environ.get("OPENSCENE_DATA_ROOT", "../dataset")) / "maps"
)


PACKAGE_ROOT = Path("path/to/rendered_data")
DEFAULT_SCENES_PATH = PACKAGE_ROOT / "synthetic_scene_pickles"
DEFAULT_SOLVABILITY_DIR = Path("path/to/risky_cases/solvability_check")
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "driving_command_regeneration_from_solvability.json"
DEFAULT_COMMANDS_PATH = PACKAGE_ROOT / "synthetic_token_driving_commands.json"

COMMAND_VECTORS: Mapping[str, Tuple[int, int, int, int]] = {
    "left": (1, 0, 0, 0),
    "straight": (0, 1, 0, 0),
    "right": (0, 0, 1, 0),
    "unknown": (0, 0, 0, 1),
}
SOLVABLE_CATEGORY = "nc_1_dac_1"
WITNESS_SOURCE = "solvable_pdm_witness"
BRAKE_ROUTE_SOURCE = "solvable_brake_route_centerline"
BRAKE_FALLBACK_SOURCE = "solvable_brake_witness_route_too_short"


@dataclass(frozen=True)
class CommandDecision:
    """The replacement route command and the witness geometry behind it."""

    command: str
    source: str
    classification_distance_m: Optional[float]
    lateral_displacement_m: Optional[float]
    witness_file: Optional[str]
    selected_candidate: Optional[str] = None
    route_ego_offset_m: Optional[float] = None
    route_heading_error_deg: Optional[float] = None
    route_remaining_m: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-scenes-path", type=Path, default=DEFAULT_SCENES_PATH)
    parser.add_argument("--solvability-dir", type=Path, default=DEFAULT_SOLVABILITY_DIR)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--commands-path",
        type=Path,
        default=DEFAULT_COMMANDS_PATH,
        help="Output token-to-one-hot command JSON, written when scene updates are applied.",
    )
    parser.add_argument(
        "--lookahead-distance-m",
        type=float,
        default=20.0,
        help="Distance along a solvable PDM witness at which to classify direction (default: 20 m).",
    )
    parser.add_argument(
        "--lateral-threshold-m",
        type=float,
        default=2.0,
        help="Strict lateral displacement threshold for left/right classification (default: 2 m).",
    )
    parser.add_argument(
        "--min-route-length-m",
        type=float,
        default=5.0,
        help="Minimum route centerline ahead of the ego for a brake witness's route command (default: 5 m).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write the manifest without changing scene pickles or the command JSON.",
    )
    return parser.parse_args()


def read_audit_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Solvability audit CSV not found: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not rows[0].get("token"):
        raise ValueError(f"{path} does not contain tokenized solvability rows")
    by_token = {row["token"]: row for row in rows}
    if len(by_token) != len(rows):
        raise ValueError(f"{path} contains duplicate scene tokens")
    return by_token


def read_witness_index(path: Path) -> Dict[str, Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Solvable witness index not found: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    result: Dict[str, Path] = {}
    for row in rows:
        token = str(row.get("token", "")).strip()
        relative = str(row.get("trajectory_file", "")).strip()
        if not token or not relative:
            raise ValueError(f"Invalid witness-index row: {row}")
        if token in result:
            raise ValueError(f"Duplicate witness token {token} in {path}")
        result[token] = path.parent / relative
    return result


def read_witness(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Selected PDM witness is missing: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 1:
        raise ValueError(f"Witness has no samples: {path}")
    try:
        xy = np.asarray([(float(row["x"]), float(row["y"])) for row in rows], dtype=np.float64)
        headings = np.asarray([float(row["heading"]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid x/y/heading values in witness: {path}") from exc
    if not np.isfinite(xy).all() or not np.isfinite(headings).all():
        raise ValueError(f"Witness contains non-finite values: {path}")
    return xy, headings


def point_at_distance(xy: np.ndarray, distance_m: float) -> Tuple[np.ndarray, float]:
    """Interpolate the witness at path distance, or use its endpoint if shorter."""
    if len(xy) == 1:
        return xy[0], 0.0
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_distance_m = float(cumulative[-1])
    if total_distance_m <= distance_m:
        return xy[-1], total_distance_m
    upper_index = int(np.searchsorted(cumulative, distance_m, side="right"))
    lower_index = upper_index - 1
    segment_distance_m = float(cumulative[upper_index] - cumulative[lower_index])
    if segment_distance_m <= 1e-12:
        return xy[upper_index], distance_m
    fraction = (distance_m - cumulative[lower_index]) / segment_distance_m
    return xy[lower_index] + fraction * (xy[upper_index] - xy[lower_index]), distance_m


def lateral_displacement(origin_xy: np.ndarray, heading: float, target_xy: np.ndarray) -> float:
    """Signed left offset of target from origin in the frame of heading."""
    delta_xy = np.asarray(target_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    return float(-np.sin(heading) * delta_xy[0] + np.cos(heading) * delta_xy[1])


def classify_lateral(lateral_displacement_m: float, lateral_threshold_m: float) -> str:
    if lateral_displacement_m > lateral_threshold_m:
        return "left"
    if lateral_displacement_m < -lateral_threshold_m:
        return "right"
    return "straight"


def is_brake_candidate(selected_candidate: Optional[str]) -> bool:
    return "brake" in str(selected_candidate or "")


def command_from_witness(
    path: Path, lookahead_distance_m: float, lateral_threshold_m: float
) -> CommandDecision:
    xy, headings = read_witness(path)
    classification_xy, classification_distance_m = point_at_distance(xy, lookahead_distance_m)
    lateral_displacement_m = lateral_displacement(xy[0], float(headings[0]), classification_xy)
    command = classify_lateral(lateral_displacement_m, lateral_threshold_m)
    return CommandDecision(
        command=command,
        source=WITNESS_SOURCE,
        classification_distance_m=classification_distance_m,
        lateral_displacement_m=lateral_displacement_m,
        witness_file=str(path),
    )


@dataclass(frozen=True)
class RouteDiagnostics:
    """Where the ego sits relative to the route centerline used for its command."""

    ego_offset_m: float
    heading_error_deg: float
    remaining_m: float


def command_from_route_centerline(
    centerline_xyh: np.ndarray,
    ego_xyh: Sequence[float],
    lookahead_distance_m: float,
    lateral_threshold_m: float,
    min_route_length_m: float,
) -> Tuple[RouteDiagnostics, Optional[CommandDecision]]:
    """Measure the ego against the route and classify the route ahead.

    The decision is None when less than ``min_route_length_m`` of centerline
    remains ahead; the diagnostics are returned either way so a fallback can
    record why it happened.

    ``centerline_xyh`` is the discretized PDM route centerline (x, y, heading).
    The ego is projected onto it and the centerline's own lateral displacement
    over the look-ahead is measured in its frame at the projection, so an ego
    offset or yaw relative to the lane does not bias the route direction.
    """
    from shapely.geometry import LineString, Point

    line = LineString(centerline_xyh[:, :2])
    ego_x, ego_y, ego_heading = (float(value) for value in ego_xyh)
    start_s = float(line.project(Point(ego_x, ego_y)))
    remaining_m = float(line.length) - start_s
    start_point = line.interpolate(start_s)
    start_xy = np.asarray([start_point.x, start_point.y])
    # Route heading at the projection: direction of the centerline segment there.
    ahead_point = line.interpolate(min(start_s + 0.5, line.length))
    behind_point = line.interpolate(max(start_s - 0.5, 0.0))
    route_heading = float(np.arctan2(ahead_point.y - behind_point.y, ahead_point.x - behind_point.x))
    diagnostics = RouteDiagnostics(
        ego_offset_m=float(np.hypot(start_xy[0] - ego_x, start_xy[1] - ego_y)),
        heading_error_deg=float(np.degrees((ego_heading - route_heading + np.pi) % (2.0 * np.pi) - np.pi)),
        remaining_m=remaining_m,
    )
    if remaining_m < min_route_length_m:
        return diagnostics, None
    classification_distance_m = min(lookahead_distance_m, remaining_m)
    target_point = line.interpolate(start_s + classification_distance_m)
    lateral_displacement_m = lateral_displacement(start_xy, route_heading, (target_point.x, target_point.y))
    return diagnostics, CommandDecision(
        command=classify_lateral(lateral_displacement_m, lateral_threshold_m),
        source=BRAKE_ROUTE_SOURCE,
        classification_distance_m=classification_distance_m,
        lateral_displacement_m=lateral_displacement_m,
        witness_file=None,
    )


class RouteCenterlineLoader:
    """Rebuild the checker's PDM route centerline for one scene (no traffic simulation).

    The heavy NAVSIM/nuPlan imports happen on first use, so runs without brake
    witnesses do not need the map stack.
    """

    def __init__(self, map_root: Path) -> None:
        os.environ["NUPLAN_MAPS_ROOT"] = str(map_root)
        repo_root = Path(__file__).resolve().parents[1]
        for path in (str(repo_root), str(Path(__file__).resolve().parent)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import raw_reactive_idm_solvability as solvability

        self._base = solvability
        self._map_root = map_root
        self._helper = solvability.MetricCacheProcessor(
            cache_path=None, force_feature_computation=False, proposal_sampling=solvability.SAMPLING
        )

    def __call__(self, scene_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        base = self._base
        scene = base.Scene.load_from_disk(scene_path, None, base.SensorConfig.build_no_sensors())
        scenario = base.NavSimScenario(scene, map_root=str(self._map_root), map_version="nuplan-maps-v1.0")
        _, planner_initialization = self._helper._get_planner_inputs(scenario)
        ego_state = scenario.initial_ego_state
        # Same steps as the checker's prepare() up to the centerline.
        planner = base.ReactiveIDMSolvabilityPlanner().planner
        planner.initialize(planner_initialization)
        planner._route_roadblock_correction(ego_state)
        planner._drivable_area_map = base.PDMDrivableMap.from_simulation(
            planner._map_api, ego_state, base.MAP_RADIUS_M
        )
        current_lane = planner._get_starting_lane(ego_state)
        if current_lane is None:
            raise RuntimeError(f"No route lane found for the ego in {scene_path}")
        centerline_xyh = np.asarray(
            [(state.x, state.y, state.heading) for state in planner._get_discrete_centerline(current_lane)],
            dtype=np.float64,
        )
        rear_axle = ego_state.rear_axle
        return centerline_xyh, (rear_axle.x, rear_axle.y, rear_axle.heading)


def brake_route_decision(
    scene_path: Path,
    witness_decision: CommandDecision,
    selected_candidate: str,
    load_route,
    lookahead_distance_m: float,
    lateral_threshold_m: float,
    min_route_length_m: float,
) -> CommandDecision:
    centerline_xyh, ego_xyh = load_route(scene_path)
    diagnostics, decision = command_from_route_centerline(
        centerline_xyh, ego_xyh, lookahead_distance_m, lateral_threshold_m, min_route_length_m
    )
    if decision is None:
        decision = dataclasses.replace(witness_decision, source=BRAKE_FALLBACK_SOURCE)
    return dataclasses.replace(
        decision,
        witness_file=witness_decision.witness_file,
        selected_candidate=selected_candidate,
        route_ego_offset_m=diagnostics.ego_offset_m,
        route_heading_error_deg=diagnostics.heading_error_deg,
        route_remaining_m=diagnostics.remaining_m,
    )


def decision_for_scene(
    token: str,
    audit_rows: Mapping[str, Mapping[str, str]],
    witness_paths: Mapping[str, Path],
    lookahead_distance_m: float,
    lateral_threshold_m: float,
) -> CommandDecision:
    row = audit_rows.get(token)
    if row is None:
        raise KeyError(f"Synthetic scene {token} has no revised solvability row")
    if row.get("status") != "scored" or row.get("category") != SOLVABLE_CATEGORY:
        return CommandDecision("unknown", "unsolved_or_unscored", None, None, None)
    witness_path = witness_paths.get(token)
    if witness_path is None:
        raise KeyError(f"Solvable scene {token} has no selected PDM witness")
    decision = command_from_witness(witness_path, lookahead_distance_m, lateral_threshold_m)
    return dataclasses.replace(decision, selected_candidate=row.get("selected_candidate"))


def load_pickle(path: Path):
    with path.open("rb") as source:
        return pickle.load(source)


def history_frames(scene: Mapping[str, object], path: Path) -> Sequence[dict]:
    metadata = scene.get("scene_metadata")
    frames = scene.get("frames")
    if not isinstance(metadata, Mapping) or not isinstance(frames, list):
        raise ValueError(f"Not a serialized NAVSIM scene: {path}")
    history_count = metadata.get("num_history_frames")
    if not isinstance(history_count, int) or history_count <= 0 or len(frames) < history_count:
        raise ValueError(f"Invalid history-frame metadata in {path}")
    selected = frames[:history_count]
    if not all(isinstance(frame, dict) and isinstance(frame.get("ego_status"), dict) for frame in selected):
        raise ValueError(f"Missing serialized ego status in a history frame: {path}")
    return selected


def current_command(scene: Mapping[str, object], path: Path) -> List[int]:
    frames = history_frames(scene, path)
    value = frames[-1]["ego_status"].get("driving_command")
    array = np.asarray(value, dtype=np.int64)
    if array.shape != (4,):
        raise ValueError(f"Current driving_command in {path} must have shape (4,), got {array.shape}")
    return [int(item) for item in array]


def replace_history_commands(scene: Mapping[str, object], path: Path, vector: Tuple[int, int, int, int]) -> None:
    for frame in history_frames(scene, path):
        old_value = frame["ego_status"].get("driving_command")
        old_dtype = np.asarray(old_value).dtype if old_value is not None else np.dtype(np.int64)
        frame["ego_status"]["driving_command"] = np.asarray(vector, dtype=old_dtype)


def atomic_dump_pickle(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        pickle.dump(value, temporary, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> int:
    args = parse_args()
    if args.lookahead_distance_m <= 0.0 or args.lateral_threshold_m < 0.0 or args.min_route_length_m < 0.0:
        raise ValueError(
            "--lookahead-distance-m must be positive; --lateral-threshold-m and --min-route-length-m non-negative"
        )

    scenes_path = args.synthetic_scenes_path.resolve()
    solvability_dir = args.solvability_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    commands_path = args.commands_path.resolve()
    if commands_path == manifest_path:
        raise ValueError("--commands-path and --manifest-path must refer to different files")
    scene_paths = sorted(scenes_path.glob("*.pkl"))
    if not scene_paths:
        raise FileNotFoundError(f"No synthetic scene pickles found in {scenes_path}")

    audit_rows = read_audit_rows(solvability_dir / "raw_reactive_idm_solvability.csv")
    witness_paths = read_witness_index(solvability_dir / "solvable_trajectory_index.csv")
    scene_tokens = {path.stem for path in scene_paths}
    missing_audit = sorted(scene_tokens - set(audit_rows))
    if missing_audit:
        raise ValueError(
            f"{len(missing_audit)} synthetic scenes have no solvability row in {solvability_dir}, "
            f"e.g. {missing_audit[:5]}"
        )
    # A rendered package may contain only a subset (e.g. the solvable scenes).
    audit_rows = {token: row for token, row in audit_rows.items() if token in scene_tokens}

    decisions = {
        path.stem: decision_for_scene(
            path.stem, audit_rows, witness_paths, args.lookahead_distance_m, args.lateral_threshold_m
        )
        for path in scene_paths
    }
    brake_paths = [path for path in scene_paths if is_brake_candidate(decisions[path.stem].selected_candidate)]
    if brake_paths:
        load_route = RouteCenterlineLoader(args.map_root.resolve())
        for path in brake_paths:
            witness_decision = decisions[path.stem]
            decisions[path.stem] = brake_route_decision(
                path,
                witness_decision,
                str(witness_decision.selected_candidate),
                load_route,
                args.lookahead_distance_m,
                args.lateral_threshold_m,
                args.min_route_length_m,
            )
    source_counts = Counter(decision.source for decision in decisions.values())
    command_counts = Counter(decision.command for decision in decisions.values())
    changes: List[Dict[str, object]] = []
    token_commands: Dict[str, List[int]] = {}
    for path in scene_paths:
        scene = load_pickle(path)
        if not isinstance(scene, Mapping):
            raise ValueError(f"Not a serialized NAVSIM scene: {path}")
        before = current_command(scene, path)
        decision = decisions[path.stem]
        vector = COMMAND_VECTORS[decision.command]
        token_commands[path.stem] = list(vector)
        changes.append(
            {
                "token": path.stem,
                "previous_current_command": before,
                "new_command": decision.command,
                "new_current_command": list(vector),
                "source": decision.source,
                "classification_distance_m": decision.classification_distance_m,
                "lateral_displacement_m": decision.lateral_displacement_m,
                "witness_file": decision.witness_file,
                "selected_candidate": decision.selected_candidate,
                "route_ego_offset_m": decision.route_ego_offset_m,
                "route_heading_error_deg": decision.route_heading_error_deg,
                "route_remaining_m": decision.route_remaining_m,
            }
        )
        if not args.dry_run:
            replace_history_commands(scene, path, vector)
            atomic_dump_pickle(path, scene)

    manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "synthetic_scenes_path": str(scenes_path),
        "solvability_dir": str(solvability_dir),
        "classification": {
            "solvable_category": SOLVABLE_CATEGORY,
            "lookahead_distance_m": args.lookahead_distance_m,
            "fallback": "The 4-second witness endpoint is used when less than the lookahead distance is traveled.",
            "lateral_threshold_m": args.lateral_threshold_m,
            "rule": "left if lateral > threshold; right if lateral < -threshold; otherwise straight",
            "frame_updates": "All serialized history-frame ego_status.driving_command values receive the case-level command.",
            "unsolved_or_unscored": "unknown",
            "brake_witnesses": (
                "A solvable scene whose selected candidate is a brake takes its command from the PDM route "
                "centerline: lateral displacement over the look-ahead from the ego's projection, in the "
                "centerline frame, with the same threshold. With less than min_route_length_m of centerline "
                "ahead, the witness rule is used (source solvable_brake_witness_route_too_short)."
            ),
            "min_route_length_m": args.min_route_length_m,
            "map_root": str(args.map_root.resolve()),
        },
        "counts": {
            "scenes": len(scene_paths),
            "left": command_counts["left"],
            "straight": command_counts["straight"],
            "right": command_counts["right"],
            "unknown": command_counts["unknown"],
            "sources": dict(sorted(source_counts.items())),
        },
        "changes": changes,
    }
    write_json(manifest_path, manifest)
    if not args.dry_run:
        write_json(commands_path, token_commands)
    print(json.dumps(manifest["counts"], sort_keys=True))
    fallbacks = sorted(token for token, decision in decisions.items() if decision.source == BRAKE_FALLBACK_SOURCE)
    if fallbacks:
        print(f"Brake witnesses with too little route ahead (witness rule used): {', '.join(fallbacks)}")
    print(f"Wrote driving-command manifest: {manifest_path}")
    if not args.dry_run:
        print(f"Wrote token driving commands: {commands_path}")
    if args.dry_run:
        print("Dry run: scene pickles and command JSON were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
