"""Build one synthetic-only NAVSIM split for all or selected scenes in a package.

The generated split contains no original-scene tokens and no reactive mapping.
It is intended for standalone evaluation of replaced synthetic scenes with
run_pdm_score_synthetic.py.

Example:

    python nuplan_scripts/auto_reconstruction/build_navsim_package_split.py \
        --package-root navsim_token_replaced_metrics \
        --split-name mtgs_generated_all \
        --overwrite

To select solvable scenes, pass --tokens-file pointing to the checker's
synthetic_tokens_nc_1_dac_1.txt. Tokens match scene_metadata.initial_token,
which can differ from the scene pickle filename stem.

Only load pickle files that you trust. Python pickle is not safe for untrusted
data.
"""

from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class SceneRecord:
    """Synthetic scene fields needed for filtering and metric-readiness checks."""

    scene_path: Path
    log_name: str
    synthetic_token: str
    synthetic_previous_token: str
    num_history_frames: int
    num_future_frames: int
    extended_detection_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate generated NAVSIM scenes into one synthetic-only Hydra split."
    )
    parser.add_argument(
        "--package-root",
        required=True,
        type=Path,
        help="Package containing synthetic_scene_pickles/ and usually sensor_blobs/.",
    )
    parser.add_argument(
        "--synthetic-scenes-path",
        type=Path,
        default=None,
        help="Override the scene directory. Defaults to <package-root>/synthetic_scene_pickles.",
    )
    parser.add_argument(
        "--output-config-dir",
        type=Path,
        default=None,
        help="Hydra config root. Defaults to <package-root>/navsim_config.",
    )
    parser.add_argument("--split-name", default="mtgs_generated_all")
    parser.add_argument("--data-split", default="test")
    parser.add_argument(
        "--tokens-file",
        type=Path,
        default=None,
        help=(
            "Optional synthetic initial-frame tokens (scene_metadata.initial_token), "
            "one per line, e.g. synthetic_tokens_nc_1_dac_1.txt from solvability checks."
        ),
    )
    parser.add_argument(
        "--required-future-frames",
        type=int,
        default=8,
        help="Required future 2 Hz detection frames for the default 4-second scorer.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def sanitize_config_name(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_")
    return value or "mtgs_generated_all"


def yaml_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def discover_scene_paths(synthetic_scenes_path: Path) -> List[Path]:
    if not synthetic_scenes_path.is_dir():
        raise FileNotFoundError(f"Synthetic scene directory not found: {synthetic_scenes_path}")
    paths = sorted(path for path in synthetic_scenes_path.iterdir() if path.suffix == ".pkl")
    if not paths:
        raise FileNotFoundError(f"No .pkl scenes found in: {synthetic_scenes_path}")
    return paths


def read_scene_record(scene_path: Path) -> SceneRecord:
    scene = load_pickle(scene_path)
    metadata = scene.get("scene_metadata")
    frames = scene.get("frames")
    if not isinstance(metadata, dict) or not isinstance(frames, list):
        raise ValueError(f"{scene_path} is not a NAVSIM synthetic scene pickle")

    required = ("log_name", "initial_token", "num_history_frames", "num_future_frames")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"{scene_path} is missing scene metadata fields: {missing}")

    synthetic_token = str(metadata["initial_token"])
    num_history_frames = int(metadata["num_history_frames"])
    num_future_frames = int(metadata["num_future_frames"])
    if num_history_frames < 2:
        raise ValueError(f"{scene_path} needs at least two history frames")
    if len(frames) < num_history_frames:
        raise ValueError(
            f"{scene_path} contains {len(frames)} frames but declares "
            f"{num_history_frames} history frames"
        )

    current_frame = frames[num_history_frames - 1]
    if str(current_frame.get("token")) != synthetic_token:
        raise ValueError(
            f"{scene_path} initial_token {synthetic_token!r} does not match "
            f"history endpoint {current_frame.get('token')!r}"
        )

    extended_detections = scene.get("extended_detections_tracks")
    extended_detection_frames = (
        len(extended_detections) if isinstance(extended_detections, list) else 0
    )
    return SceneRecord(
        scene_path=scene_path,
        log_name=str(metadata["log_name"]),
        synthetic_token=synthetic_token,
        synthetic_previous_token=str(frames[num_history_frames - 2]["token"]),
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        extended_detection_frames=extended_detection_frames,
    )


def validate_records(
    records: Sequence[SceneRecord], required_future_frames: int
) -> Tuple[int, int]:
    if required_future_frames < 0:
        raise ValueError("--required-future-frames cannot be negative")

    sampling = {(record.num_history_frames, record.num_future_frames) for record in records}
    if len(sampling) != 1:
        details = ", ".join(
            f"history={history}, future={future}" for history, future in sorted(sampling)
        )
        raise ValueError(
            "One SceneFilter cannot represent mixed frame counts. "
            f"Found: {details}. Separate the scenes into compatible packages."
        )

    seen_synthetic: Dict[str, Path] = {}
    errors: List[str] = []
    for record in records:
        if len(record.synthetic_token) != 17:
            errors.append(
                f"{record.scene_path.name}: initial token {record.synthetic_token!r} "
                "is not 17 characters and NAVSIM will not classify it as synthetic"
            )
        previous_path = seen_synthetic.get(record.synthetic_token)
        if previous_path is not None:
            errors.append(
                f"duplicate initial token {record.synthetic_token!r} in "
                f"{previous_path.name} and {record.scene_path.name}"
            )
        seen_synthetic[record.synthetic_token] = record.scene_path
        if record.extended_detection_frames < required_future_frames:
            errors.append(
                f"{record.scene_path.name}: has {record.extended_detection_frames} future "
                f"detection frames; {required_future_frames} are required"
            )

    if errors:
        raise ValueError("Package is not ready for synthetic metric evaluation:\n- " + "\n- ".join(errors))
    return next(iter(sampling))


def select_records_by_tokens(
    records: Sequence[SceneRecord], tokens_file: Path
) -> List[SceneRecord]:
    tokens = {
        line.strip()
        for line in tokens_file.read_text(encoding="utf8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not tokens:
        raise ValueError(f"Token file contains no synthetic initial-frame tokens: {tokens_file}")

    available_tokens = {record.synthetic_token for record in records}
    missing_tokens = sorted(tokens - available_tokens)
    if missing_tokens:
        sample = ", ".join(missing_tokens[:10])
        raise ValueError(
            f"{len(missing_tokens)} requested initial-frame token(s) are missing "
            f"from the package: {sample}"
        )
    return [record for record in records if record.synthetic_token in tokens]


def render_split_yaml(split_name: str, data_split: str) -> str:
    return "\n".join(
        [
            "defaults:",
            f"- scene_filter: {split_name}",
            f"data_split: {data_split}",
        ]
    ) + "\n"


def render_scene_filter_yaml(
    records: Sequence[SceneRecord],
    num_history_frames: int,
    num_future_frames: int,
) -> str:
    synthetic_tokens = sorted(record.synthetic_token for record in records)
    log_names = sorted({record.log_name for record in records})
    lines = [
        "_target_: navsim.common.dataclasses.SceneFilter",
        "_convert_: all",
        f"num_history_frames: {num_history_frames}",
        f"num_future_frames: {num_future_frames}",
        "frame_interval: 1",
        "has_route: true",
        "max_scenes: null",
        "include_synthetic_scenes: true",
        "log_names:",
        *[f"- {yaml_quote(log_name)}" for log_name in log_names],
        "tokens:",
        *[f"- {yaml_quote(token)}" for token in synthetic_tokens],
    ]
    return "\n".join(lines) + "\n"


def ensure_writable(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(map(str, existing))
        raise FileExistsError(f"Output already exists: {joined}. Pass --overwrite to replace it.")


def main() -> None:
    args = parse_args()
    package_root = resolve(args.package_root)
    synthetic_scenes_path = resolve(
        args.synthetic_scenes_path or package_root / "synthetic_scene_pickles"
    )
    output_config_dir = resolve(
        args.output_config_dir or package_root / "navsim_config"
    )
    split_name = sanitize_config_name(args.split_name)

    scene_paths = discover_scene_paths(synthetic_scenes_path)
    records = [read_scene_record(path) for path in scene_paths]
    if args.tokens_file is not None:
        records = select_records_by_tokens(records, resolve(args.tokens_file))
    num_history_frames, num_future_frames = validate_records(
        records, args.required_future_frames
    )

    split_dir = output_config_dir / "train_test_split"
    split_path = split_dir / f"{split_name}.yaml"
    scene_filter_path = split_dir / "scene_filter" / f"{split_name}.yaml"
    ensure_writable((split_path, scene_filter_path), args.overwrite)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    scene_filter_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        render_split_yaml(split_name, args.data_split), encoding="utf8"
    )
    scene_filter_path.write_text(
        render_scene_filter_yaml(records, num_history_frames, num_future_frames),
        encoding="utf8",
    )

    print(f"Scanned {len(scene_paths)} standalone synthetic scenes from: {synthetic_scenes_path}")
    if args.tokens_file is not None:
        print(f"Selected {len(records)} scenes using initial-frame tokens from: {resolve(args.tokens_file)}")
    print("Original scene tokens and reactive mappings: not used")
    print(
        f"Metric readiness: {args.required_future_frames} future detection frames per scene"
    )
    print(
        f"Agent input configuration: history={num_history_frames}, "
        f"inline future={num_future_frames}"
    )
    print(f"Wrote train_test_split: {split_path}")
    print(f"Wrote scene_filter: {scene_filter_path}")
    print(
        "Hydra overrides: "
        f"hydra.searchpath=[pkg://navsim.planning.script.config.common,"
        f"file://{output_config_dir}] "
        f"train_test_split={split_name} "
        f"synthetic_sensor_path={package_root / 'sensor_blobs'} "
        f"synthetic_scenes_path={synthetic_scenes_path}"
    )


if __name__ == "__main__":
    main()
