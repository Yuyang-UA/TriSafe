#!/usr/bin/env python3
"""Two-stage construction of NAVSIM synthetic scenes from HGS-selected candidates.

Stage ``construct`` (scripts/risky_case_construction.sh)
    Reads the HGS ``selected_candidates_by_token.json`` and calls
    ``render_navsim_token_window.py --dry-run`` once per reconstruction config.
    This writes complete NAVSIM scene pickles (poses, ego dynamics, annotations,
    camera calibration and the relative image paths) without loading the MTGS
    model or rendering images. The pickles are the input of
    ``nuplan_scripts/run_solvability_check.py``.

Stage ``sensors`` (scripts/sensor_data_rendering.sh)
    Reads the stage-1 manifest and a solvable-token list (by default
    ``synthetic_tokens_nc_1_dac_1.txt`` from the solvability check), and renders
    the multi-view camera images for those scenes only into a separate package.
    Scene tokens are deterministic functions of (config, token, candidate path,
    candidate index, render settings); the stage-2 renderer output is checked
    against the stage-1 tokens, so the package contains exactly the solvable
    scenes, each with its images.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDERER = Path(__file__).resolve().parent / "render_navsim_token_window.py"

# Fixed window/candidate settings shared by both stages. Changing any of them
# changes the synthetic scene tokens, so they are recorded in the stage-1
# summary and stage 2 refuses to run if they differ.
RENDER_SETTINGS: Dict[str, str] = {
    "--candidate-replace-mode": "history_first_half",
    "--candidate-history-duration": "1.5",
    "--history-frames": "4",
    "--future-frames": "8",
    "--scene-current-offset-frames": "0",
    "--scene-format": "two_stage",
}

PLAN_NAME = "selected_candidates_render_plan.json"
SUMMARY_NAME = "selected_candidates_render_summary.json"
AGGREGATE_MANIFEST_NAME = "selected_candidates_manifest.json"
TOKEN_MAPPING_NAME = "selected_candidates_token_mapping.json"


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got boolean {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if not isinstance(value, int) and float(value) != parsed:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return parsed


def now() -> str:
    return dt.datetime.now().isoformat()


# --------------------------------------------------------------------------- #
# Task planning
# --------------------------------------------------------------------------- #
def selected_records(data: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return candidate records and source rows skipped for empty mappings.

    Supported formats: the HGS list ``[{"config", "token", "candidates": {group: index}}]``
    and the legacy object ``{"selected_candidates_by_token": {token: {...}}}``.
    """
    records: List[Dict[str, Any]] = []
    skipped_empty: List[Dict[str, Any]] = []

    def add_record(config: Any, token: Any, coarse_group: Any, candidate_index: Any) -> None:
        config_name = Path(str(config)).name
        token_text = str(token).strip()
        if not config_name or config_name == ".":
            raise ValueError(f"Missing config for token {token_text!r}")
        if not token_text:
            raise ValueError(f"Missing token for config {config_name!r}")
        records.append(
            {
                "config": config_name,
                "token": token_text,
                "coarse_group": str(coarse_group),
                "candidate_index": as_int(candidate_index, "candidate_index"),
            }
        )

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            config = item.get("config", "")
            token = item.get("token", "")
            candidates = item.get("candidates")
            if isinstance(candidates, dict):
                if not candidates:
                    skipped_empty.append({"config": Path(str(config)).name, "token": str(token).strip()})
                    continue
                for coarse_group, candidate_index in candidates.items():
                    add_record(config, token, coarse_group, candidate_index)
            elif "coarse_group" in item and "candidate_index" in item:
                add_record(config, token, item["coarse_group"], item["candidate_index"])
            elif candidates in (None, []):
                skipped_empty.append({"config": Path(str(config)).name, "token": str(token).strip()})
            else:
                raise ValueError(f"Unsupported candidates value for token {token!r}: {type(candidates).__name__}")
    elif isinstance(data, dict):
        by_token = data.get("selected_candidates_by_token", {})
        if not isinstance(by_token, dict):
            raise ValueError("selected_candidates_by_token must be an object")
        for token, item in by_token.items():
            if not isinstance(item, dict):
                continue
            config = item.get("config", "")
            candidates = item.get("candidates", [])
            if not candidates:
                skipped_empty.append({"config": Path(str(config)).name, "token": str(token).strip()})
                continue
            if not isinstance(candidates, list):
                raise ValueError(f"Unsupported candidates value for token {token!r}: {type(candidates).__name__}")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f"Invalid candidate record for token {token!r}: {candidate!r}")
                add_record(config, token, candidate.get("coarse_group"), candidate.get("candidate_index"))
    else:
        raise ValueError(f"Expected a JSON list or object, got {type(data).__name__}")

    return records, skipped_empty


def find_config_paths(config_dir: Path, config_names: Iterable[str]) -> Dict[str, Path]:
    available: Dict[str, List[Path]] = defaultdict(list)
    for path in config_dir.rglob("*.yaml"):
        available[path.name].append(path)

    resolved: Dict[str, Path] = {}
    for name in sorted(set(config_names)):
        matches = available.get(name, [])
        if not matches:
            raise FileNotFoundError(f"Could not find config {name} under {config_dir}")
        if len(matches) != 1:
            raise RuntimeError(f"Config name {name} is ambiguous under {config_dir}: {matches}")
        resolved[name] = matches[0].resolve()
    return resolved


def build_tasks(records: List[Dict[str, Any]], config_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Deduplicate identical scene renders while retaining their coarse-group provenance."""
    by_identity: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for record in records:
        identity = (record["config"], record["token"], record["candidate_index"])
        task = by_identity.setdefault(
            identity,
            {
                "config": record["config"],
                "config_path": str(config_paths[record["config"]]),
                "token": record["token"],
                "candidate_index": record["candidate_index"],
                "coarse_groups": [],
            },
        )
        task["coarse_groups"].append(record["coarse_group"])

    tasks = list(by_identity.values())
    for task in tasks:
        task["coarse_groups"].sort(key=lambda value: (len(value), value))
    return sorted(tasks, key=lambda task: (task["config"], task["token"], task["candidate_index"]))


def validate_tasks_against_model_summary(tasks: List[Dict[str, Any]], model_token_summary: Path) -> None:
    summary = read_json(model_token_summary)
    if not isinstance(summary, dict):
        raise ValueError(f"{model_token_summary} must map config names to token lists.")
    for task in tasks:
        tokens = summary.get(task["config"])
        if not isinstance(tokens, list) or task["token"] not in tokens:
            raise ValueError(
                f"Selected token {task['token']} is not mapped to config {task['config']} in {model_token_summary}"
            )


def validate_candidate_indices(tasks: List[Dict[str, Any]], candidate_path: Path) -> List[int]:
    candidate_bank = np.load(candidate_path, mmap_mode="r")
    if candidate_bank.ndim != 3 or candidate_bank.shape[2] < 2:
        raise ValueError(f"{candidate_path} must have shape (N, T, 2+), got {candidate_bank.shape}")
    candidate_count = int(candidate_bank.shape[0])
    invalid = [task for task in tasks if not 0 <= task["candidate_index"] < candidate_count]
    if invalid:
        raise ValueError(
            f"{len(invalid)} selected candidate indices are outside [0, {candidate_count - 1}]; "
            f"first invalid task: {invalid[0]}"
        )
    return list(candidate_bank.shape)


def build_model_groups(tasks: List[Dict[str, Any]], job_dir: Path) -> List[Dict[str, Any]]:
    """One group per reconstruction config, so each MTGS model is loaded at most once."""
    by_config: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        group = by_config.setdefault(
            task["config"],
            {"config": task["config"], "config_path": task["config_path"], "tasks": []},
        )
        group["tasks"].append(task)

    groups = list(by_config.values())
    for group in groups:
        group["tasks"].sort(key=lambda task: (task["token"], task["candidate_index"]))
        digest = hashlib.sha1(group["config"].encode("utf-8")).hexdigest()[:16]
        group["candidate_jobs_path"] = str(job_dir / f"{digest}.json")
    return sorted(groups, key=lambda group: group["config"])


def write_candidate_job_files(model_groups: List[Dict[str, Any]]) -> None:
    for group in model_groups:
        jobs = [{"token": task["token"], "candidate_index": task["candidate_index"]} for task in group["tasks"]]
        write_json(Path(group["candidate_jobs_path"]), jobs)


# --------------------------------------------------------------------------- #
# Renderer invocation
# --------------------------------------------------------------------------- #
def build_command(
    model_group: Dict[str, Any],
    *,
    candidate_path: Path,
    output_root: Path,
    navsim_log_path: Path,
    scene_only: bool,
    overwrite: bool,
) -> List[str]:
    command = [
        sys.executable,
        str(RENDERER),
        "--config",
        model_group["config_path"],
        "--candidate-trajectory-path",
        str(candidate_path),
        "--candidate-jobs-json",
        model_group["candidate_jobs_path"],
    ]
    for flag, value in RENDER_SETTINGS.items():
        command.extend([flag, value])
    command.extend(
        [
            "--skip-invalid-windows",
            "--output-root",
            str(output_root),
            "--output-layout",
            "package",
            "--navsim-log-path",
            str(navsim_log_path),
            "--write-videos",
            "False",
        ]
    )
    if scene_only:
        # Renderer dry run: write complete scene pickles, skip model loading and images.
        command.append("--dry-run")
    if overwrite:
        command.append("--overwrite")
    return command


def load_renderer_manifest(output_root: Path, expected_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Renderer succeeded but did not write {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list) or not all(isinstance(record, dict) for record in manifest):
        raise RuntimeError(f"Expected a list of manifest records in {manifest_path}, got {manifest!r}")
    by_identity = {(str(record.get("central_token")), record.get("candidate_index")): record for record in manifest}
    expected_identities = [(task["token"], task["candidate_index"]) for task in expected_tasks]
    if len(by_identity) != len(manifest) or set(by_identity) != set(expected_identities):
        raise RuntimeError(
            f"Manifest jobs in {manifest_path} do not match the requested model group; "
            f"expected {expected_identities}, got {list(by_identity)}"
        )
    return [by_identity[identity] for identity in expected_identities]


def run_model_groups(
    *,
    model_groups: List[Dict[str, Any]],
    output_root: Path,
    summary: Dict[str, Any],
    command_kwargs: Dict[str, Any],
    continue_on_error: bool,
    checkpoint_every: int,
    expected_scene_tokens: Optional[Dict[Tuple[str, int], str]] = None,
) -> int:
    """Run the renderer per config and aggregate manifests, mappings and progress."""
    summary_path = output_root / SUMMARY_NAME
    aggregate_manifest_path = output_root / AGGREGATE_MANIFEST_NAME
    token_mapping_path = output_root / TOKEN_MAPPING_NAME
    aggregate_manifest: List[Dict[str, Any]] = []
    token_mappings: List[Dict[str, Any]] = []

    def checkpoint() -> None:
        write_json(summary_path, summary)
        write_json(aggregate_manifest_path, aggregate_manifest)
        write_json(token_mapping_path, token_mappings)

    checkpoint()
    for position, render_group in enumerate(model_groups, start=1):
        command = build_command(render_group, output_root=output_root, **command_kwargs)
        print(
            f"\n[{position}/{len(model_groups)}] config={render_group['config']} "
            f"candidate_jobs={len(render_group['tasks'])}",
            flush=True,
        )
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            summary["failures"].append(
                {
                    "config": render_group["config"],
                    "candidate_job_count": len(render_group["tasks"]),
                    "candidate_jobs_path": render_group["candidate_jobs_path"],
                    "returncode": result.returncode,
                }
            )
            summary["status"] = "failed"
            checkpoint()
            if not continue_on_error:
                print(f"Stopping after renderer failure; progress summary: {summary_path}", file=sys.stderr)
                return result.returncode
            continue

        manifest_records = load_renderer_manifest(output_root, render_group["tasks"])
        for task, manifest_record in zip(render_group["tasks"], manifest_records):
            manifest_record["selected_candidate"] = {
                "config": task["config"],
                "token": task["token"],
                "candidate_index": task["candidate_index"],
                "coarse_groups": task["coarse_groups"],
            }
            aggregate_manifest.append(manifest_record)
            if manifest_record.get("skipped", False):
                summary["skipped_invalid_window_jobs"].append(
                    {**task, "reason": manifest_record.get("skip_reason", "unknown reason")}
                )
                continue
            scene_token = manifest_record.get("synthetic_scene_token")
            if expected_scene_tokens is not None:
                expected = expected_scene_tokens[(task["token"], task["candidate_index"])]
                if scene_token != expected:
                    raise RuntimeError(
                        f"Scene token changed between stages for {task}: stage 1 {expected}, now {scene_token}. "
                        "The candidate bank path or render settings differ from the risky-case construction run."
                    )
            token_mappings.append(
                {
                    "new_token": manifest_record["synthetic_initial_token"],
                    "original_token": task["token"],
                    "selected_candidate_index": task["candidate_index"],
                }
            )
            summary["completed"].append({**task, "synthetic_scene_token": scene_token})
        if position % checkpoint_every == 0 or position == len(model_groups):
            checkpoint()

    summary["status"] = (
        "completed_with_failures"
        if summary["failures"]
        else "completed_with_skipped_invalid_windows"
        if summary["skipped_invalid_window_jobs"]
        else "completed"
    )
    summary["finished_at"] = now()
    checkpoint()
    print(
        f"\nCompleted {len(summary['completed'])}/{summary['planned_scene_count']} scenes "
        f"in {len(model_groups)} renderer invocations; "
        f"skipped {len(summary['skipped_invalid_window_jobs'])} invalid windows."
    )
    print(f"Aggregate manifest: {aggregate_manifest_path}")
    print(f"Token mapping: {token_mapping_path}")
    print(f"Summary: {summary_path}")
    return 0 if not summary["failures"] else 1


def base_summary(**fields: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"generated_at": now(), "render_settings": dict(RENDER_SETTINGS)}
    summary.update(fields)
    summary.update({"completed": [], "failures": [], "skipped_invalid_window_jobs": []})
    return summary


def print_dry_run(model_groups: List[Dict[str, Any]], output_root: Path, command_kwargs: Dict[str, Any]) -> None:
    for position, render_group in enumerate(model_groups, start=1):
        command = build_command(render_group, output_root=output_root, **command_kwargs)
        print(f"[{position}/{len(model_groups)}] {shlex.join(command)}")


# --------------------------------------------------------------------------- #
# Stage 1: risky-case construction (scene pickles only)
# --------------------------------------------------------------------------- #
def run_construct(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    candidate_path = args.candidate_trajectory_path.resolve()
    records, skipped_empty = selected_records(read_json(args.selected_candidates_json))
    if not records:
        raise ValueError(f"No non-empty selected candidates found in {args.selected_candidates_json}")

    config_paths = find_config_paths(args.config_dir, (record["config"] for record in records))
    tasks = build_tasks(records, config_paths)
    validate_tasks_against_model_summary(tasks, args.model_token_summary)
    candidate_shape = validate_candidate_indices(tasks, candidate_path)
    model_groups = build_model_groups(tasks, output_root / "candidate_jobs")

    summary = base_summary(
        stage="risky_case_construction",
        status="dry_run" if args.dry_run else "running",
        description="Scene pickles only (renderer --dry-run): no MTGS model load and no sensor images.",
        selected_candidates_json=str(args.selected_candidates_json.resolve()),
        config_dir=str(args.config_dir.resolve()),
        model_token_summary=str(args.model_token_summary.resolve()),
        candidate_trajectory_path=str(candidate_path),
        candidate_trajectory_shape=candidate_shape,
        output_root=str(output_root),
        synthetic_scenes_path=str(output_root / "synthetic_scene_pickles"),
        navsim_log_path=str(args.navsim_log_path.resolve()),
        overwrite=args.overwrite,
        source_candidate_records=len(records),
        skipped_empty_candidate_records=skipped_empty,
        planned_scene_count=len(tasks),
        planned_renderer_invocation_count=len(model_groups),
    )
    print(
        f"Risky-case plan: {len(tasks)} scenes in {len(model_groups)} renderer invocations "
        f"from {len(records)} candidate records; skipping {len(skipped_empty)} empty-candidate tokens.",
        flush=True,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_candidate_job_files(model_groups)
    write_json(
        output_root / PLAN_NAME,
        {
            "generated_at": summary["generated_at"],
            "stage": "risky_case_construction",
            "planned_scene_count": len(tasks),
            "planned_renderer_invocation_count": len(model_groups),
            "skipped_empty_candidate_records": skipped_empty,
            "planned_model_groups": model_groups,
        },
    )
    command_kwargs = dict(
        candidate_path=candidate_path,
        navsim_log_path=args.navsim_log_path.resolve(),
        scene_only=True,
        overwrite=args.overwrite,
    )
    if args.dry_run:
        print_dry_run(model_groups, output_root, command_kwargs)
        write_json(output_root / SUMMARY_NAME, summary)
        print(f"Dry-run summary written to: {output_root / SUMMARY_NAME}")
        return 0

    code = run_model_groups(
        model_groups=model_groups,
        output_root=output_root,
        summary=summary,
        command_kwargs=command_kwargs,
        continue_on_error=args.continue_on_error,
        checkpoint_every=args.checkpoint_every,
    )
    if code != 0:
        return code
    print(
        "\nNext step (solvability check):\n  "
        + shlex.join(["RISKY_CASE_ROOT=" + str(output_root), "bash", "scripts/solvability_verification.sh"])
    )
    return 0


# --------------------------------------------------------------------------- #
# Stage 2: multi-view sensor rendering of solvable scenes
# --------------------------------------------------------------------------- #
def read_token_list(path: Path) -> set:
    if not path.is_file():
        raise FileNotFoundError(f"Solvable token list does not exist: {path}")
    tokens = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not tokens:
        raise ValueError(f"Solvable token list is empty: {path}")
    return tokens


def run_sensors(args: argparse.Namespace) -> int:
    risky_root = args.risky_case_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == risky_root:
        raise ValueError(
            "--output-root must differ from --risky-case-root: the sensor package holds only solvable scenes."
        )
    stage1_summary = read_json(risky_root / SUMMARY_NAME)
    if stage1_summary.get("stage") != "risky_case_construction":
        raise ValueError(f"{risky_root / SUMMARY_NAME} is not a risky-case construction summary.")
    if stage1_summary.get("render_settings") != RENDER_SETTINGS:
        raise ValueError(
            "Render settings differ from the risky-case construction run, so scene tokens would not match: "
            f"stage 1 {stage1_summary.get('render_settings')}, now {RENDER_SETTINGS}"
        )
    if stage1_summary.get("status") not in {"completed", "completed_with_skipped_invalid_windows", "completed_with_failures"}:
        print(f"Warning: risky-case construction status is {stage1_summary.get('status')!r}.", file=sys.stderr)

    # Reuse the exact candidate bank path: it is part of the scene-token hash.
    candidate_path = Path(stage1_summary["candidate_trajectory_path"])
    navsim_log_path = (args.navsim_log_path or Path(stage1_summary["navsim_log_path"])).resolve()
    solvable_tokens_path = args.solvable_tokens or risky_root / "solvability_check" / "synthetic_tokens_nc_1_dac_1.txt"
    solvable_tokens = read_token_list(solvable_tokens_path)

    stage1_manifest = read_json(risky_root / AGGREGATE_MANIFEST_NAME)
    tasks: List[Dict[str, Any]] = []
    expected_scene_tokens: Dict[Tuple[str, int], str] = {}
    matched_tokens = set()
    for record in stage1_manifest:
        if record.get("skipped", False):
            continue
        record_tokens = {record.get("synthetic_scene_token"), record.get("synthetic_initial_token")}
        hit = record_tokens & solvable_tokens
        if not hit:
            continue
        matched_tokens |= hit
        selected = record["selected_candidate"]
        identity = (selected["token"], int(selected["candidate_index"]))
        expected_scene_tokens[identity] = record["synthetic_scene_token"]
        tasks.append({**selected, "candidate_index": int(selected["candidate_index"])})
    unmatched = sorted(solvable_tokens - matched_tokens)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} solvable tokens are not in {risky_root / AGGREGATE_MANIFEST_NAME}; "
            f"first: {unmatched[:5]}. Was the solvability check run on this risky-case package?"
        )
    if not tasks:
        raise ValueError("No solvable scenes selected.")

    config_paths = find_config_paths(Path(stage1_summary["config_dir"]), (task["config"] for task in tasks))
    for task in tasks:
        task["config_path"] = str(config_paths[task["config"]])
    tasks.sort(key=lambda task: (task["config"], task["token"], task["candidate_index"]))
    model_groups = build_model_groups(tasks, output_root / "candidate_jobs")

    summary = base_summary(
        stage="sensor_data_rendering",
        status="dry_run" if args.dry_run else "running",
        description="Multi-view camera images and scene pickles for solvable risky cases only.",
        risky_case_root=str(risky_root),
        solvable_tokens=str(Path(solvable_tokens_path).resolve()),
        solvable_token_count=len(solvable_tokens),
        risky_case_scene_count=stage1_summary.get("planned_scene_count"),
        candidate_trajectory_path=str(candidate_path),
        output_root=str(output_root),
        synthetic_scenes_path=str(output_root / "synthetic_scene_pickles"),
        synthetic_sensor_path=str(output_root / "sensor_blobs"),
        navsim_log_path=str(navsim_log_path),
        overwrite=args.overwrite,
        planned_scene_count=len(tasks),
        planned_model_load_count=len(model_groups),
        planned_renderer_invocation_count=len(model_groups),
    )
    print(
        f"Sensor-rendering plan: {len(tasks)} solvable scenes (of {stage1_summary.get('planned_scene_count')} "
        f"risky cases) in {len(model_groups)} renderer invocations (one model load each).",
        flush=True,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_candidate_job_files(model_groups)
    write_json(
        output_root / PLAN_NAME,
        {
            "generated_at": summary["generated_at"],
            "stage": "sensor_data_rendering",
            "planned_scene_count": len(tasks),
            "planned_model_load_count": len(model_groups),
            "planned_model_groups": model_groups,
        },
    )
    command_kwargs = dict(
        candidate_path=candidate_path,
        navsim_log_path=navsim_log_path,
        scene_only=False,
        overwrite=args.overwrite,
    )
    if args.dry_run:
        print_dry_run(model_groups, output_root, command_kwargs)
        write_json(output_root / SUMMARY_NAME, summary)
        print(f"Dry-run summary written to: {output_root / SUMMARY_NAME}")
        return 0

    return run_model_groups(
        model_groups=model_groups,
        output_root=output_root,
        summary=summary,
        command_kwargs=command_kwargs,
        continue_on_error=args.continue_on_error,
        checkpoint_every=args.checkpoint_every,
        expected_scene_tokens=expected_scene_tokens,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--output-root", type=Path, required=True)
        sub.add_argument("--dry-run", action="store_true", help="Print the planned renderer commands only.")
        sub.add_argument("--overwrite", action="store_true", help="Replace scenes/images from an earlier run.")
        sub.add_argument("--continue-on-error", action="store_true", help="Finish the batch and report all failures.")
        sub.add_argument("--checkpoint-every", type=int, default=25, help="Persist progress after this many configs.")

    construct = subparsers.add_parser("construct", help="Stage 1: scene pickles without sensor data.")
    add_common(construct)
    construct.add_argument("--selected-candidates-json", type=Path, required=True)
    construct.add_argument("--config-dir", type=Path, required=True)
    construct.add_argument("--model-token-summary", type=Path, required=True)
    construct.add_argument("--candidate-trajectory-path", type=Path, required=True)
    construct.add_argument("--navsim-log-path", type=Path, required=True)

    sensors = subparsers.add_parser("sensors", help="Stage 2: multi-view images for solvable scenes.")
    add_common(sensors)
    sensors.add_argument("--risky-case-root", type=Path, required=True)
    sensors.add_argument(
        "--solvable-tokens",
        type=Path,
        default=None,
        help="Token list to render (default: <risky-case-root>/solvability_check/synthetic_tokens_nc_1_dac_1.txt).",
    )
    sensors.add_argument("--navsim-log-path", type=Path, default=None, help="Default: the stage-1 log path.")

    args = parser.parse_args(argv)
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    return args


def main() -> int:
    if not RENDERER.is_file():
        raise FileNotFoundError(f"Renderer not found: {RENDERER}")
    args = parse_args()
    return run_construct(args) if args.stage == "construct" else run_sensors(args)


if __name__ == "__main__":
    raise SystemExit(main())
