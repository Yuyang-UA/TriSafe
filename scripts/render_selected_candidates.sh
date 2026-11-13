#!/usr/bin/env bash
# Render NAVSIM synthetic scenes for trajectory candidates selected by the
# hierarchical search.  The current JSON format is a list of records such as:
#   {"config": "...yaml", "token": "...", "candidates": {"0": 8410, ...}}
# Empty candidate mappings are deliberately skipped.
#
# Example:
#   PYTHON_BIN=/path/to/envs/mtgs/bin/python \
#     OPENSCENE_DATA_ROOT=/path/to/dataset \
#     bash render_selected_candidates.sh
#
# Useful overrides:
#   DRY_RUN=1                         Print the 3,655 planned commands only.
#   OUTPUT_ROOT=path/to/package       Change the output package location.
#   OVERWRITE=1                       Replace scenes from an earlier run.
#   CONTINUE_ON_ERROR=1               Finish the batch and report all failures.
#   CHECKPOINT_EVERY=50               Persist progress after this many renders.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SELECTED_CANDIDATES_JSON="${SELECTED_CANDIDATES_JSON:-trajectory_candidate_selection/K_9/all_1000/selected_candidates_by_token.json}"
CONFIG_DIR="${CONFIG_DIR:-data/auto_reconstruction/config/navtest_available_merged_pointcloud_good}"
MODEL_TOKEN_SUMMARY="${MODEL_TOKEN_SUMMARY:-data/auto_reconstruction/summary.json}"
CANDIDATE_TRAJECTORY_PATH="${CANDIDATE_TRAJECTORY_PATH:-trajectory_candidate/16384.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}/navsim_replaced_new}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}/navsim_logs/test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the Python interpreter for the MTGS/NAVSIM environment." >&2
    exit 1
fi

for required_path in "$SELECTED_CANDIDATES_JSON" "$CONFIG_DIR" "$MODEL_TOKEN_SUMMARY" "$CANDIDATE_TRAJECTORY_PATH" "$NAVSIM_LOG_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        exit 1
    fi
done

export PROJECT_ROOT SELECTED_CANDIDATES_JSON CONFIG_DIR MODEL_TOKEN_SUMMARY CANDIDATE_TRAJECTORY_PATH
export OUTPUT_ROOT NAVSIM_LOG_PATH DRY_RUN OVERWRITE CONTINUE_ON_ERROR CHECKPOINT_EVERY

"$PYTHON_BIN" - <<'PYEOF'
"""Plan and run one NAVSIM render for every selected trajectory candidate."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
SELECTED_JSON = Path(os.environ["SELECTED_CANDIDATES_JSON"]).resolve()
CONFIG_DIR = Path(os.environ["CONFIG_DIR"]).resolve()
MODEL_TOKEN_SUMMARY = Path(os.environ["MODEL_TOKEN_SUMMARY"]).resolve()
CANDIDATE_PATH = Path(os.environ["CANDIDATE_TRAJECTORY_PATH"]).resolve()
OUTPUT_ROOT = Path(os.environ["OUTPUT_ROOT"]).resolve()
NAVSIM_LOG_PATH = Path(os.environ["NAVSIM_LOG_PATH"]).resolve()
DRY_RUN = os.environ["DRY_RUN"] == "1"
OVERWRITE = os.environ["OVERWRITE"] == "1"
CONTINUE_ON_ERROR = os.environ["CONTINUE_ON_ERROR"] == "1"
RENDERER = PROJECT_ROOT / "nuplan_scripts/auto_reconstruction/render_navsim_token_window.py"
SUMMARY_PATH = OUTPUT_ROOT / "selected_candidates_render_summary.json"
AGGREGATE_MANIFEST_PATH = OUTPUT_ROOT / "selected_candidates_manifest.json"
PLAN_PATH = OUTPUT_ROOT / "selected_candidates_render_plan.json"
TOKEN_MAPPING_PATH = OUTPUT_ROOT / "selected_candidates_token_mapping.json"
JOB_DIR = OUTPUT_ROOT / "candidate_jobs"
CHECKPOINT_EVERY = int(os.environ["CHECKPOINT_EVERY"])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got boolean {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        try:
            if float(value) != parsed:
                raise ValueError(f"{field} must be an integer, got {value!r}")
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be an integer, got {value!r}")
    return parsed


def selected_records(data: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return candidate records and source rows skipped for empty mappings."""
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


def find_config_paths(config_names: Iterable[str]) -> Dict[str, Path]:
    available: Dict[str, List[Path]] = defaultdict(list)
    for path in CONFIG_DIR.rglob("*.yaml"):
        available[path.name].append(path)

    resolved: Dict[str, Path] = {}
    for name in sorted(set(config_names)):
        matches = available.get(name, [])
        if not matches:
            raise FileNotFoundError(f"Could not find config {name} under {CONFIG_DIR}")
        if len(matches) != 1:
            raise RuntimeError(f"Config name {name} is ambiguous under {CONFIG_DIR}: {matches}")
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


def validate_tasks_against_model_summary(tasks: List[Dict[str, Any]]) -> None:
    summary = json.loads(MODEL_TOKEN_SUMMARY.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{MODEL_TOKEN_SUMMARY} must map config names to token lists.")
    for task in tasks:
        tokens = summary.get(task["config"])
        if not isinstance(tokens, list) or task["token"] not in tokens:
            raise ValueError(
                f"Selected token {task['token']} is not mapped to config {task['config']} in {MODEL_TOKEN_SUMMARY}"
            )


def build_model_groups(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One group per reconstruction config, so each MTGS model is loaded once."""
    by_config: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        group = by_config.setdefault(
            task["config"],
            {
                "config": task["config"],
                "config_path": task["config_path"],
                "tasks": [],
            },
        )
        group["tasks"].append(task)

    groups = list(by_config.values())
    for group in groups:
        group["tasks"].sort(key=lambda task: (task["token"], task["candidate_index"]))
        digest = hashlib.sha1(group["config"].encode("utf-8")).hexdigest()[:16]
        group["candidate_jobs_path"] = str(JOB_DIR / f"{digest}.json")
    return sorted(groups, key=lambda group: group["config"])


def write_candidate_job_files(model_groups: List[Dict[str, Any]]) -> None:
    for group in model_groups:
        jobs = [
            {"token": task["token"], "candidate_index": task["candidate_index"]}
            for task in group["tasks"]
        ]
        write_json(Path(group["candidate_jobs_path"]), jobs)


def build_command(model_group: Dict[str, Any]) -> List[str]:
    command = [
        sys.executable,
        str(RENDERER),
        "--config",
        model_group["config_path"],
        "--candidate-trajectory-path",
        str(CANDIDATE_PATH),
        "--candidate-jobs-json",
        model_group["candidate_jobs_path"],
        "--candidate-replace-mode",
        "history_first_half",
        "--candidate-history-duration",
        "1.5",
        "--history-frames",
        "4",
        "--future-frames",
        "8",
        "--scene-current-offset-frames",
        "0",
        "--scene-format",
        "two_stage",
        "--skip-invalid-windows",
        "--output-root",
        str(OUTPUT_ROOT),
        "--output-layout",
        "package",
        "--navsim-log-path",
        str(NAVSIM_LOG_PATH),
        "--write-videos",
        "False",
    ]
    if OVERWRITE:
        command.append("--overwrite")
    return command


def load_renderer_manifest(expected_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest_path = OUTPUT_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Renderer succeeded but did not write {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(record, dict) for record in manifest):
        raise RuntimeError(f"Expected a list of manifest records in {manifest_path}, got {manifest!r}")
    by_identity = {
        (str(record.get("central_token")), record.get("candidate_index")): record
        for record in manifest
    }
    expected_identities = [(task["token"], task["candidate_index"]) for task in expected_tasks]
    if len(by_identity) != len(manifest) or set(by_identity) != set(expected_identities):
        raise RuntimeError(
            f"Manifest jobs in {manifest_path} do not match the requested model group; "
            f"expected {expected_identities}, got {list(by_identity)}"
        )
    return [by_identity[identity] for identity in expected_identities]


def token_mapping_payload(mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the concise original-token to generated-token mapping requested by the user."""
    return mappings


def main() -> int:
    if not RENDERER.is_file():
        raise FileNotFoundError(f"Renderer not found: {RENDERER}")

    source_data = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    records, skipped_empty = selected_records(source_data)
    if not records:
        raise ValueError(f"No non-empty selected candidates found in {SELECTED_JSON}")

    candidate_bank = np.load(CANDIDATE_PATH, mmap_mode="r")
    if candidate_bank.ndim != 3 or candidate_bank.shape[2] < 2:
        raise ValueError(f"{CANDIDATE_PATH} must have shape (N, T, 2+), got {candidate_bank.shape}")
    candidate_count = int(candidate_bank.shape[0])

    config_paths = find_config_paths(record["config"] for record in records)
    tasks = build_tasks(records, config_paths)
    validate_tasks_against_model_summary(tasks)
    model_groups = build_model_groups(tasks)
    if CHECKPOINT_EVERY <= 0:
        raise ValueError("CHECKPOINT_EVERY must be positive")
    invalid_indices = [task for task in tasks if task["candidate_index"] < 0 or task["candidate_index"] >= candidate_count]
    if invalid_indices:
        raise ValueError(
            f"{len(invalid_indices)} selected candidate indices are outside [0, {candidate_count - 1}]; "
            f"first invalid task: {invalid_indices[0]}"
        )

    summary: Dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(),
        "status": "dry_run" if DRY_RUN else "running",
        "selected_candidates_json": str(SELECTED_JSON),
        "config_dir": str(CONFIG_DIR),
        "model_token_summary": str(MODEL_TOKEN_SUMMARY),
        "candidate_trajectory_path": str(CANDIDATE_PATH),
        "candidate_trajectory_shape": list(candidate_bank.shape),
        "output_root": str(OUTPUT_ROOT),
        "navsim_log_path": str(NAVSIM_LOG_PATH),
        "overwrite": OVERWRITE,
        "source_candidate_records": len(records),
        "skipped_empty_candidate_records": skipped_empty,
        "skipped_invalid_window_jobs": [],
        "planned_scene_count": len(tasks),
        "planned_model_load_count": len(model_groups),
        "planned_renderer_invocation_count": len(model_groups),
        "plan_path": str(PLAN_PATH),
        "token_mapping_path": str(TOKEN_MAPPING_PATH),
        "checkpoint_every": CHECKPOINT_EVERY,
        "completed": [],
        "failures": [],
    }
    plan = {
        "generated_at": summary["generated_at"],
        "selected_candidates_json": str(SELECTED_JSON),
        "planned_scene_count": len(tasks),
        "planned_model_load_count": len(model_groups),
        "planned_renderer_invocation_count": len(model_groups),
        "skipped_empty_candidate_records": skipped_empty,
        "planned_model_groups": model_groups,
    }
    print(
        f"Selected-candidate plan: {len(tasks)} scenes in {len(model_groups)} renderer invocations "
        f"from {len(records)} candidate records; skipping {len(skipped_empty)} empty-candidate tokens.",
        flush=True,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_candidate_job_files(model_groups)

    if DRY_RUN:
        for position, render_group in enumerate(model_groups, start=1):
            print(f"[{position}/{len(model_groups)}] {shlex.join(build_command(render_group))}")
        write_json(PLAN_PATH, plan)
        write_json(SUMMARY_PATH, summary)
        write_json(TOKEN_MAPPING_PATH, token_mapping_payload([]))
        print(f"Dry-run summary written to: {SUMMARY_PATH}")
        return 0

    write_json(PLAN_PATH, plan)
    write_json(SUMMARY_PATH, summary)
    aggregate_manifest: List[Dict[str, Any]] = []
    token_mappings: List[Dict[str, Any]] = []
    write_json(TOKEN_MAPPING_PATH, token_mapping_payload(token_mappings))
    for position, render_group in enumerate(model_groups, start=1):
        command = build_command(render_group)
        print(
            f"\n[{position}/{len(model_groups)}] config={render_group['config']} "
            f"candidate_jobs={len(render_group['tasks'])} (one model load)"
        )
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            failure = {
                "config": render_group["config"],
                "candidate_job_count": len(render_group["tasks"]),
                "candidate_jobs_path": render_group["candidate_jobs_path"],
                "returncode": result.returncode,
            }
            summary["failures"].append(failure)
            summary["status"] = "failed"
            write_json(SUMMARY_PATH, summary)
            write_json(AGGREGATE_MANIFEST_PATH, aggregate_manifest)
            write_json(TOKEN_MAPPING_PATH, token_mapping_payload(token_mappings))
            if not CONTINUE_ON_ERROR:
                print(f"Stopping after renderer failure; progress summary: {SUMMARY_PATH}", file=sys.stderr)
                return result.returncode
            continue

        manifest_records = load_renderer_manifest(render_group["tasks"])
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
            token_mappings.append(
                {
                    "new_token": manifest_record["synthetic_initial_token"],
                    "original_token": task["token"],
                    "selected_candidate_index": task["candidate_index"],
                }
            )
            summary["completed"].append({**task, "synthetic_scene_token": manifest_record.get("synthetic_scene_token")})
        if position % CHECKPOINT_EVERY == 0 or position == len(model_groups):
            write_json(AGGREGATE_MANIFEST_PATH, aggregate_manifest)
            write_json(SUMMARY_PATH, summary)
            write_json(TOKEN_MAPPING_PATH, token_mapping_payload(token_mappings))

    summary["status"] = (
        "completed_with_failures"
        if summary["failures"]
        else "completed_with_skipped_invalid_windows"
        if summary["skipped_invalid_window_jobs"]
        else "completed"
    )
    write_json(SUMMARY_PATH, summary)
    write_json(AGGREGATE_MANIFEST_PATH, aggregate_manifest)
    write_json(TOKEN_MAPPING_PATH, token_mapping_payload(token_mappings))
    print(
        f"\nCompleted {len(summary['completed'])}/{len(tasks)} scenes "
        f"in {len(model_groups)} renderer invocations; "
        f"skipped {len(summary['skipped_invalid_window_jobs'])} invalid windows."
    )
    print(f"Aggregate manifest: {AGGREGATE_MANIFEST_PATH}")
    print(f"Token mapping: {TOKEN_MAPPING_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF
