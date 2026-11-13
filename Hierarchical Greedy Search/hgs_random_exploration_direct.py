#!/usr/bin/env python3
"""Run random hierarchical-greedy-search explorations directly.

This runner keeps the experiment light on disk: it disables per-candidate graph
JSON, does not call ``search.write_outputs()``, and only writes aggregate CSV/JSON
summaries for the sampled runs. With --workers > 1, the parent process samples
config-token pairs and writes summaries while worker processes evaluate pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
HGS_DIR = Path(__file__).resolve().parent
if os.fspath(HGS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(HGS_DIR))
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "data/auto_reconstruction/config/navtest_available_merged_pointcloud_good"
DEFAULT_NAVSIM_LOG_PATH = (
    Path(os.environ.get("OPENSCENE_DATA_ROOT", PROJECT_ROOT.parent / "dataset")) / "navsim_logs/test"
)

_WORKER_SEARCH_ARGS: Optional[argparse.Namespace] = None
_WORKER_COMMON: Optional[Dict[str, Any]] = None
_WORKER_NAVSIM_LOG_PATH: Optional[Path] = None
_WORKER_OUTPUT_ROOT: Optional[Path] = None
_WORKER_LOG_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_WORKER_MAP_CACHE: Dict[str, Any] = {}


def parse_config_tokens(path: Path) -> Tuple[str, List[str]]:
    road_block_name = path.stem
    tokens: List[str] = []
    in_tokens = False
    for line in path.read_text(encoding="utf8").splitlines():
        if line.startswith("road_block_name:"):
            road_block_name = line.split(":", 1)[1].strip()
        if line.startswith("central_tokens:"):
            in_tokens = True
            continue
        if not in_tokens:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            tokens.append(stripped[2:].strip())
        elif stripped and not stripped.startswith("#"):
            break
    return road_block_name, tokens


def collect_pairs(config_dir: Path) -> List[Tuple[Path, str, str]]:
    pairs: List[Tuple[Path, str, str]] = []
    for path in sorted(config_dir.glob("*.yaml")):
        road_block_name, tokens = parse_config_tokens(path)
        for token in tokens:
            pairs.append((path, road_block_name, token))
    return pairs


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def gini(values: Sequence[float]) -> float:
    values = sorted(float(v) for v in values if float(v) >= 0.0)
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0.0:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return float((2.0 * weighted) / (len(values) * total) - (len(values) + 1.0) / len(values))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def selected_candidate_records(selected_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in selected_rows:
        records.append(
            {
                "coarse_group": int_value(row.get("coarse_group")),
                "candidate_index": int_value(row.get("candidate_index")),
            }
        )
    records.sort(key=lambda row: int_value(row.get("coarse_group")))
    return records


def write_selected_candidates_json(summary_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, Any]] = []
    for row in sorted_run_rows(rows):
        token = str(row.get("token", ""))
        if not token:
            continue
        config = Path(str(row.get("config", ""))).name
        candidate_map = {
            str(int_value(candidate.get("coarse_group"))): int_value(candidate.get("candidate_index"))
            for candidate in sorted(
                row.get("_selected_candidates", []),
                key=lambda item: int_value(item.get("coarse_group")),
            )
        }
        payload.append(
            {
                "config": config,
                "token": token,
                "candidates": candidate_map,
            }
        )
    (summary_root / "selected_candidates_by_token.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf8",
    )


def aggregate_notes() -> Dict[str, Any]:
    return {
        "purpose": (
            "Aggregate statistics over random hierarchical greedy search runs. "
            "Each run samples one config-token pair and performs the requested local search budget."
        ),
        "score_definition": {
            "score_valid": (
                "The valid risk increase score used by the search. It is max(R_dyn, R_sta) "
                "only when the candidate has no hard event; otherwise it is 0."
            ),
            "R_dyn": "Maximum positive dynamic risk increase from history frame 0 to frame 3.",
            "R_sta": "Maximum positive static/boundary risk increase from history frame 0 to frame 3.",
            "important": "The score is a risk increase, not simply the risk at the last frame.",
        },
        "search_formula": {
            "A": "omega_b * B + omega_i * I - omega_s * S - omega_h * eta",
            "B": "Best valid score found so far in this coarse group.",
            "I": "Recent best-score improvement over local_window evaluations.",
            "S": "Saturation penalty. Higher means the group appears less productive.",
            "eta": "Hard-event rate in the group: hard_count / n.",
            "saturation": (
                "S = sigmoid((n - saturation_window) / saturation_temperature) * "
                "sigmoid((epsilon - I) / improvement_rho)"
            ),
        },
        "quality_guide": {
            "good_search": (
                "High best_selected_score and mean_selected_score, broad groups_visited, "
                "low max_group_eval_share, low group_eval_gini, and low hard-event metrics."
            ),
            "over_exploitation_signal": (
                "High max_group_eval_share or high group_eval_gini means the search spent too much "
                "budget in a small number of groups."
            ),
            "unsafe_sampling_signal": (
                "High total_hard_count, mean_eta, or max_eta means many evaluated candidates caused "
                "collision/violation/out-of-drivable-area style hard events."
            ),
        },
        "metric_notes": {
            "seed": "Actual seed used to sample config-token pairs. Recorded so random runs can be reproduced if needed.",
            "seed_source": "user if --seed was provided; system_random if the seed was generated automatically.",
            "runs_requested": "Number of random config-token runs requested.",
            "runs_ok": "Number of runs completed without exception. Higher is better.",
            "runs_failed": "Number of failed runs. Lower is better.",
            "total_iterations": "Local candidate-evaluation budget per run, excluding bootstrap evaluations.",
            "q_max": "Maximum local evaluations before the algorithm re-selects a coarse group.",
            "saturation_window": "L_prime in S. 0 means total_iterations / number_of_groups.",
            "group_eval_share_cap": "Optional max share of local evaluations for one coarse group. 0 disables the cap.",
            "workers": "Number of worker processes used by the direct runner.",
            "multiprocessing_start_method": "Process start method used for multi-worker runs; serial means no pool was used.",
            "batch_elapsed_sec": "Total wall-clock seconds for the whole exploration batch, including loading and summary writes.",
            "sum_run_elapsed_sec": "Sum of per-run wall-clock seconds. In worker mode this can exceed batch_elapsed_sec.",
            "parallel_speedup_estimate": "sum_run_elapsed_sec / batch_elapsed_sec; a coarse estimate of worker-mode speedup.",
            "evaluation_mode": "risk_only skips standard scene-graph dictionary construction; scene_graph uses the original builder.",
            "risk_step": "Time step for TTC/TLC projection. Larger values are faster but less temporally precise.",
            "ttc_horizon": "Dynamic collision lookahead horizon in seconds. Shorter is faster but may miss later collisions.",
            "tlc_horizon": "Static boundary lookahead horizon in seconds. Shorter is faster but may miss later boundary crossings.",
            "object_radius": "Dynamic objects farther than this radius are skipped unless already colliding. Smaller is faster.",
            "boundary_radius": "Boundary lines farther than this radius are skipped. Smaller is faster.",
            "max_boundaries": "Maximum nearby boundary lines evaluated for static risk. Smaller is faster.",
            "skip_static_graph": "If true, disables static/boundary risk R_sta and only uses dynamic risk. Fastest but changes the scoring target.",
            "elapsed_sec": "Average wall-clock seconds per run. Lower is faster.",
            "local_evaluations": "Average local evaluations completed per run. Usually equals total_iterations.",
            "global_rounds": "Average number of global group-selection rounds per run.",
            "groups_visited": "Average number of coarse groups receiving local evaluations. Higher means broader exploration.",
            "phase1_done_groups": "Average number of groups whose sub-cluster medoid phase completed.",
            "selected_count": "Average number of coarse groups with a positive valid selected candidate.",
            "best_selected_score": "Average best valid risk-increase score per run. Higher finds stronger candidates.",
            "mean_selected_score": "Average score of selected candidates, averaged across runs. Higher is better if hard events remain low.",
            "max_group_best_B": "Average maximum group best-score B per run. Higher means at least one strong group was found.",
            "mean_group_best_B": "Average best-score B across all groups and runs. Higher means strong candidates are spread across groups.",
            "mean_recent_improvement_I": "Average recent improvement. Near 0 means most groups stopped improving recently.",
            "groups_with_I_gt_epsilon": "Average number of groups whose recent improvement exceeded epsilon.",
            "mean_saturation_S": "Average saturation penalty. Higher means the scheduler judged many groups as saturated.",
            "groups_saturated_gt_0_8": "Average number of groups with saturation penalty above 0.8.",
            "mean_eta": "Average group hard-event rate. Lower is safer and more efficient.",
            "max_eta": "Worst group hard-event rate per run, averaged across runs. Lower is better.",
            "total_hard_count": "Average number of hard-event evaluations per run. Lower is better.",
            "max_group_eval_share": "Average largest fraction of budget spent in one group. Lower means less over-exploitation.",
            "group_eval_gini": "Inequality of evaluation allocation across groups. 0 is balanced; 1 is highly concentrated.",
            "selected_group_rounds_unique": "Average number of groups selected by the global scheduler. Higher means broader scheduling.",
        },
    }


def run_summary_column_notes() -> List[Dict[str, str]]:
    notes = {
        "run_id": ("1-based run index in this exploration batch.", "identifier"),
        "config": ("Config YAML used for this run, relative to the project root.", "identifier"),
        "road_block_name": ("Road-block/config name loaded from the YAML.", "identifier"),
        "token": ("Central NAVSIM token selected for this run.", "identifier"),
        "elapsed_sec": ("Wall-clock seconds used by this run.", "lower_is_faster"),
        "returncode": ("0 for success; exception/timeout text for failed runs.", "0_is_success"),
        "status": ("ok means the run completed; failed means the run raised an exception.", "ok_is_best"),
        "error": ("Exception text when status is failed.", "empty_is_best"),
        "local_evaluations": ("Number of budgeted local candidate evaluations completed.", "usually_equals_total_iterations"),
        "all_evaluation_rows": ("Bootstrap plus local evaluation rows kept in memory for this run.", "informational"),
        "global_rounds": ("Number of group-selection rounds. With q_max=20 and 1000 iterations, usually about 50.", "informational"),
        "groups": ("Total number of available coarse groups.", "informational"),
        "groups_visited": ("Number of coarse groups that received local evaluations.", "higher_means_more_exploration"),
        "phase1_done_groups": ("Number of groups whose sub-cluster medoid phase finished.", "higher_means_more_phase1_coverage"),
        "exhausted_groups": ("Number of groups with no remaining candidate actions.", "context_dependent"),
        "selected_count": ("Number of groups with a final positive valid selected candidate.", "higher_is_generally_better"),
        "best_selected_score": ("Best valid risk-increase score among selected candidates in this run.", "higher_finds_stronger_candidate"),
        "mean_selected_score": ("Mean valid score across selected candidates in this run.", "higher_is_better_if_hard_events_are_low"),
        "max_group_best_B": ("Largest group best-score B at the end of this run.", "higher_finds_stronger_group"),
        "mean_group_best_B": ("Mean group best-score B across all groups.", "higher_means_good_candidates_are_spread_out"),
        "mean_recent_improvement_I": ("Mean recent improvement I across groups at the end of the run.", "near_zero_means_groups_have_stalled"),
        "groups_with_I_gt_epsilon": ("Number of groups whose recent improvement exceeded epsilon.", "higher_means_more_groups_still_improving"),
        "mean_saturation_S": ("Mean saturation penalty S across groups.", "higher_means_more_groups_look_saturated"),
        "groups_saturated_gt_0_8": ("Number of groups with saturation penalty above 0.8.", "higher_means_many_groups_are_saturated"),
        "mean_eta": ("Mean hard-event rate across groups, where eta = hard_count / n.", "lower_is_safer"),
        "max_eta": ("Worst group hard-event rate in this run.", "lower_is_safer"),
        "total_hard_count": ("Total local evaluations that produced collision/violation/out-of-drivable-area hard events.", "lower_is_safer"),
        "max_group_eval_share": ("Largest fraction of local budget spent in one group.", "lower_means_less_over_exploitation"),
        "group_eval_gini": ("Inequality of evaluations across groups. 0 is balanced, 1 is highly concentrated.", "lower_means_more_balanced"),
        "group_eval_stdev": ("Standard deviation of local evaluations per group.", "lower_means_more_balanced"),
        "selected_group_rounds_max": ("Maximum number of global rounds assigned to any single group.", "lower_means_less_repeated_focus"),
        "selected_group_rounds_unique": ("Number of groups selected by the global scheduler at least once.", "higher_means_broader_scheduling"),
    }
    return [
        {"column": column, "meaning": meaning, "how_to_read": how_to_read}
        for column, (meaning, how_to_read) in notes.items()
    ]


def aggregate_rows(
    *,
    group_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    round_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    improvement_epsilon: float,
) -> Dict[str, Any]:
    n_values = [int_value(row.get("n")) for row in group_rows]
    eta_values = [float_value(row.get("eta")) for row in group_rows]
    b_values = [float_value(row.get("B")) for row in group_rows]
    i_values = [float_value(row.get("I")) for row in group_rows]
    s_values = [float_value(row.get("S")) for row in group_rows]
    selected_scores = [float_value(row.get("score_valid")) for row in selected_rows]
    hard_counts = [int_value(row.get("hard_count")) for row in group_rows]
    selected_before = [
        row
        for row in round_rows
        if row.get("stage") == "before_local_block" and int_value(row.get("selected")) == 1
    ]
    selected_group_counts: Dict[str, int] = {}
    for row in selected_before:
        group = str(row.get("coarse_group", ""))
        selected_group_counts[group] = selected_group_counts.get(group, 0) + 1

    total_local_evals = sum(n_values)
    max_group_evals = max(n_values) if n_values else 0
    return {
        "status": "ok",
        "local_evaluations": total_local_evals,
        "all_evaluation_rows": len(eval_rows),
        "global_rounds": len(selected_before),
        "groups": len(group_rows),
        "groups_visited": sum(1 for value in n_values if value > 0),
        "phase1_done_groups": sum(1 for row in group_rows if int_value(row.get("phase1_done")) == 1),
        "exhausted_groups": sum(1 for row in group_rows if int_value(row.get("exhausted")) == 1),
        "selected_count": len(selected_rows),
        "best_selected_score": max(selected_scores) if selected_scores else 0.0,
        "mean_selected_score": mean(selected_scores),
        "max_group_best_B": max(b_values) if b_values else 0.0,
        "mean_group_best_B": mean(b_values),
        "mean_recent_improvement_I": mean(i_values),
        "groups_with_I_gt_epsilon": sum(1 for value in i_values if value > improvement_epsilon),
        "mean_saturation_S": mean(s_values),
        "groups_saturated_gt_0_8": sum(1 for value in s_values if value > 0.8),
        "mean_eta": mean(eta_values),
        "max_eta": max(eta_values) if eta_values else 0.0,
        "total_hard_count": sum(hard_counts),
        "max_group_eval_share": float(max_group_evals / total_local_evals) if total_local_evals else 0.0,
        "group_eval_gini": gini(n_values),
        "group_eval_stdev": stdev(n_values),
        "selected_group_rounds_max": max(selected_group_counts.values()) if selected_group_counts else 0,
        "selected_group_rounds_unique": len(selected_group_counts),
    }


def make_search_args(args: argparse.Namespace, first_config: Path) -> argparse.Namespace:
    import hierarchical_greedy_search as hgs

    old_argv = sys.argv
    try:
        sys.argv = [
            "hierarchical_greedy_search.py",
            "--config",
            os.fspath(first_config),
            "--navsim-log-path",
            os.fspath(args.navsim_log_path),
            "--candidate-trajectory-path",
            os.fspath(args.candidate_trajectory_path),
            "--coarse-labels-path",
            os.fspath(args.coarse_labels_path),
            "--coarse-representatives-path",
            os.fspath(args.coarse_representatives_path),
            "--sub-labels-path",
            os.fspath(args.sub_labels_path),
            "--sub-representatives-path",
            os.fspath(args.sub_representatives_path),
            "--representative-points",
            str(args.representative_points),
            "--total-iterations",
            str(args.total_iterations),
            "--q-max",
            str(args.q_max),
            "--group-eval-share-cap",
            str(args.group_eval_share_cap),
            "--write-graphs",
            "false",
            "--evaluation-mode",
            str(args.evaluation_mode),
            "--overwrite",
            "true",
            "--omega-b",
            str(args.omega_b),
            "--omega-i",
            str(args.omega_i),
            "--omega-s",
            str(args.omega_s),
            "--omega-h",
            str(args.omega_h),
            "--saturation-window",
            str(args.saturation_window),
            "--epsilon",
            str(args.epsilon),
            "--improvement-rho",
            str(args.improvement_rho),
            "--risk-step",
            str(args.risk_step),
            "--ttc-horizon",
            str(args.ttc_horizon),
            "--tlc-horizon",
            str(args.tlc_horizon),
            "--object-radius",
            str(args.object_radius),
            "--boundary-radius",
            str(args.boundary_radius),
            "--max-boundaries",
            str(args.max_boundaries),
            "--skip-static-graph",
            str(args.skip_static_graph).lower(),
        ]
        search_args = hgs.parse_args()
    finally:
        sys.argv = old_argv
    hgs.validate_args(search_args)
    return search_args


def load_common_inputs(search_args: argparse.Namespace) -> Dict[str, Any]:
    import hierarchical_greedy_search as hgs

    candidates = hgs.load_candidate_bank(search_args.candidate_trajectory_path)
    coarse_labels = hgs.load_labels(search_args.coarse_labels_path, candidates.shape[0], "coarse labels")
    sub_labels = hgs.load_labels(search_args.sub_labels_path, candidates.shape[0], "sub labels")
    coarse_reps = hgs.load_representatives(
        search_args.coarse_representatives_path,
        expected_ndim0=None,
        name="coarse representatives",
    )
    sub_reps = hgs.load_representatives(
        search_args.sub_representatives_path,
        expected_ndim0=None,
        name="sub representatives",
    )
    return {
        "candidates": candidates,
        "coarse_labels": coarse_labels,
        "sub_labels": sub_labels,
        "coarse_reps": coarse_reps,
        "sub_reps": sub_reps,
        "group_indices": hgs.selected_groups(coarse_reps.shape[0], search_args.groups),
        "footprint": hgs.gsg.load_ego_footprint(),
    }


def run_one_direct(
    *,
    config_path: Path,
    road_block_name: str,
    token: str,
    navsim_log_path: Path,
    output_root: Path,
    search_args: argparse.Namespace,
    common: Dict[str, Any],
    log_cache: Dict[str, List[Dict[str, Any]]],
    map_cache: Dict[str, Any],
) -> Dict[str, Any]:
    import hierarchical_greedy_search as hgs

    start = time.time()
    base = {
        "config": os.fspath(config_path.relative_to(PROJECT_ROOT)),
        "road_block_name": road_block_name,
        "token": token,
    }
    try:
        config = hgs.load_config(os.fspath(config_path))
        if not isinstance(config, hgs.FrameCentralConfig):
            raise TypeError(f"{config_path} is not a FrameCentralConfig.")
        if config.central_log not in log_cache:
            log_cache[config.central_log] = hgs.gsg.load_navsim_log(navsim_log_path, config.central_log)
        log_frames = log_cache[config.central_log]

        map_api: Optional[Any] = None
        if not search_args.skip_static_graph:
            if config.city not in map_cache:
                map_cache[config.city] = hgs.gsg.load_map_api(config.city)
            map_api = map_cache[config.city]

        search = hgs.HierarchicalGreedySearch(
            config=config,
            config_path=config_path,
            central_token=token,
            log_frames=log_frames,
            footprint=common["footprint"],
            map_api=map_api,
            candidates=common["candidates"],
            coarse_labels=common["coarse_labels"],
            coarse_reps=common["coarse_reps"],
            sub_labels=common["sub_labels"],
            sub_reps=common["sub_reps"],
            group_indices=common["group_indices"],
            output_root=output_root,
            args=search_args,
        )
        search.run()
        elapsed = time.time() - start
        group_rows = search.group_summary()
        selected_rows = search.final_candidates()
        return {
            **base,
            "elapsed_sec": round(elapsed, 3),
            "returncode": 0,
            "_selected_candidates": selected_candidate_records(selected_rows),
            **aggregate_rows(
                group_rows=group_rows,
                selected_rows=selected_rows,
                round_rows=search.round_rows,
                eval_rows=search.evaluation_rows,
                improvement_epsilon=float(search_args.epsilon),
            ),
        }
    except Exception as exc:
        elapsed = time.time() - start
        return {
            **base,
            "elapsed_sec": round(elapsed, 3),
            "returncode": "exception",
            "status": "failed",
            "error": repr(exc),
        }


def sorted_run_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: int_value(row.get("run_id")))


def write_incremental_run_summary(summary_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_csv(summary_root / "run_summary.csv", sorted_run_rows(rows))


def init_worker(
    search_args: argparse.Namespace,
    common: Dict[str, Any],
    navsim_log_path: Path,
    output_root: Path,
) -> None:
    global _WORKER_SEARCH_ARGS
    global _WORKER_COMMON
    global _WORKER_NAVSIM_LOG_PATH
    global _WORKER_OUTPUT_ROOT
    global _WORKER_LOG_CACHE
    global _WORKER_MAP_CACHE

    _WORKER_SEARCH_ARGS = search_args
    _WORKER_COMMON = common
    _WORKER_NAVSIM_LOG_PATH = navsim_log_path
    _WORKER_OUTPUT_ROOT = output_root
    _WORKER_LOG_CACHE = {}
    _WORKER_MAP_CACHE = {}


def run_one_worker(task: Tuple[int, Path, str, str]) -> Dict[str, Any]:
    run_id, config_path, road_block_name, token = task
    if (
        _WORKER_SEARCH_ARGS is None
        or _WORKER_COMMON is None
        or _WORKER_NAVSIM_LOG_PATH is None
        or _WORKER_OUTPUT_ROOT is None
    ):
        return {
            "config": os.fspath(config_path.relative_to(PROJECT_ROOT)),
            "road_block_name": road_block_name,
            "token": token,
            "elapsed_sec": 0.0,
            "returncode": "worker_init_error",
            "status": "failed",
            "error": "worker globals were not initialized",
            "run_id": run_id,
        }

    row = run_one_direct(
        config_path=config_path,
        road_block_name=road_block_name,
        token=token,
        navsim_log_path=_WORKER_NAVSIM_LOG_PATH,
        output_root=_WORKER_OUTPUT_ROOT,
        search_args=_WORKER_SEARCH_ARGS,
        common=_WORKER_COMMON,
        log_cache=_WORKER_LOG_CACHE,
        map_cache=_WORKER_MAP_CACHE,
    )
    row["run_id"] = run_id
    return row


def multiprocessing_context() -> Any:
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context()


def build_parser(*, include_omega_i: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--navsim-log-path", type=Path, default=DEFAULT_NAVSIM_LOG_PATH)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=PROJECT_ROOT / "Hierarchical Greedy Search/hgs_parameter_exploration",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--candidate-trajectory-path", type=Path, default=Path("trajectory_candidate/human_trajectory_candidate.npy"))
    parser.add_argument(
        "--coarse-labels-path",
        type=Path,
        default=Path("trajectory_candidate/history4/coarse_group_16_kmeans_history4_sub20_labels.npy"),
    )
    parser.add_argument(
        "--coarse-representatives-path",
        type=Path,
        default=Path("trajectory_candidate/history4/coarse_group_16_kmeans_history4_sub20_representatives.npy"),
    )
    parser.add_argument(
        "--sub-labels-path",
        type=Path,
        default=Path("trajectory_candidate/history4/coarse_group_16_kmeans_history4_sub20_sub_labels.npy"),
    )
    parser.add_argument(
        "--sub-representatives-path",
        type=Path,
        default=Path("trajectory_candidate/history4/coarse_group_16_kmeans_history4_sub20_sub_representatives.npy"),
    )
    parser.add_argument("--representative-points", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent worker processes for config-token runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducibility seed. Omit it to sample a fresh random set each run.",
    )
    parser.add_argument("--total-iterations", type=int, default=1000)
    parser.add_argument("--q-max", type=int, default=20)
    parser.add_argument(
        "--group-eval-share-cap",
        type=float,
        default=0.0,
        help="Optional max share of local evaluations for one coarse group. 0 disables the cap.",
    )
    parser.add_argument("--omega-b", type=float, default=1.0)
    if include_omega_i:
        parser.add_argument("--omega-i", type=float, default=1.0)
    parser.add_argument("--omega-s", type=float, default=1.0)
    parser.add_argument("--omega-h", type=float, default=1.0)
    parser.add_argument(
        "--saturation-window",
        type=float,
        default=0.0,
        help="L_prime in the saturation count term. 0 uses total_iterations / number_of_groups.",
    )
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--improvement-rho", type=float, default=0.02)
    parser.add_argument("--risk-step", type=float, default=0.1)
    parser.add_argument("--ttc-horizon", type=float, default=7.0)
    parser.add_argument("--tlc-horizon", type=float, default=4.0)
    parser.add_argument("--object-radius", type=float, default=80.0)
    parser.add_argument("--boundary-radius", type=float, default=60.0)
    parser.add_argument("--max-boundaries", type=int, default=128)
    parser.add_argument("--skip-static-graph", type=str_to_bool, default=False)
    parser.add_argument(
        "--evaluation-mode",
        choices=("risk_only", "scene_graph"),
        default="risk_only",
        help="Use risk_only for faster experiments, or scene_graph to match the standard graph builder.",
    )
    parser.add_argument(
        "--write-run-details",
        action="store_true",
        help="Also write sampled_pairs.csv and per-run run_summary.csv.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    batch_start = time.time()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.group_eval_share_cap < 0.0 or args.group_eval_share_cap > 1.0:
        raise ValueError("--group-eval-share-cap must be in [0, 1].")
    if args.saturation_window < 0.0:
        raise ValueError("--saturation-window must be non-negative.")
    pairs = collect_pairs(args.config_dir)
    if not pairs:
        raise RuntimeError(f"No config-token pairs found in {args.config_dir}")
    if len(pairs) < args.runs:
        raise RuntimeError(f"Requested {args.runs} runs, but only found {len(pairs)} config-token pairs.")

    if args.seed is None:
        sample_seed = random.SystemRandom().randrange(0, 2**32)
        seed_source = "system_random"
    else:
        sample_seed = int(args.seed)
        seed_source = "user"
    sampled = random.Random(sample_seed).sample(pairs, args.runs)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    if args.write_run_details:
        sample_rows = [
            {
                "run_id": idx,
                "config": os.fspath(path.relative_to(PROJECT_ROOT)),
                "road_block_name": road_block_name,
                "token": token,
            }
            for idx, (path, road_block_name, token) in enumerate(sampled, start=1)
        ]
        write_csv(args.summary_root / "sampled_pairs.csv", sample_rows)
        write_csv(args.summary_root / "run_summary_column_notes.csv", run_summary_column_notes())

    search_args = make_search_args(args, sampled[0][0])
    common = load_common_inputs(search_args)
    tasks = [
        (idx, config_path, road_block_name, token)
        for idx, (config_path, road_block_name, token) in enumerate(sampled, start=1)
    ]
    worker_count = min(int(args.workers), len(tasks))
    start_method = "serial"

    rows: List[Dict[str, Any]] = []
    if worker_count == 1:
        log_cache: Dict[str, List[Dict[str, Any]]] = {}
        map_cache: Dict[str, Any] = {}
        for run_id, config_path, road_block_name, token in tasks:
            print(f"[{run_id:02d}/{args.runs}] {config_path.name} token={token}", flush=True)
            row = run_one_direct(
                config_path=config_path,
                road_block_name=road_block_name,
                token=token,
                navsim_log_path=args.navsim_log_path,
                output_root=search_args.output_root,
                search_args=search_args,
                common=common,
                log_cache=log_cache,
                map_cache=map_cache,
            )
            row["run_id"] = run_id
            rows.append(row)
            if args.write_run_details:
                write_incremental_run_summary(args.summary_root, rows)
                write_selected_candidates_json(args.summary_root, rows)
    else:
        context = multiprocessing_context()
        start_method = context.get_start_method()
        print(
            f"Running {len(tasks)} config-token runs with {worker_count} workers "
            f"({start_method} start method).",
            flush=True,
        )
        with context.Pool(
            processes=worker_count,
            initializer=init_worker,
            initargs=(search_args, common, args.navsim_log_path, search_args.output_root),
        ) as pool:
            for row in pool.imap_unordered(run_one_worker, tasks):
                rows.append(row)
                print(
                    f"[{len(rows):02d}/{args.runs}] finished run_id={int_value(row.get('run_id'))} "
                    f"status={row.get('status')} token={row.get('token')}",
                    flush=True,
                )
                if args.write_run_details:
                    write_incremental_run_summary(args.summary_root, rows)
                    write_selected_candidates_json(args.summary_root, rows)
    rows = sorted_run_rows(rows)
    write_selected_candidates_json(args.summary_root, rows)
    batch_elapsed_sec = round(time.time() - batch_start, 3)
    sum_run_elapsed_sec = sum(float_value(row.get("elapsed_sec")) for row in rows)
    parallel_speedup_estimate = (
        float(sum_run_elapsed_sec / batch_elapsed_sec) if batch_elapsed_sec > 0.0 else 0.0
    )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    group_schedule = common.get("_diversity_group_schedule", {})
    aggregate = {
        "seed": sample_seed,
        "seed_source": seed_source,
        "runs_requested": args.runs,
        "runs_ok": len(ok_rows),
        "runs_failed": len(rows) - len(ok_rows),
        "total_iterations": args.total_iterations,
        "q_max": args.q_max,
        "group_eval_share_cap": args.group_eval_share_cap,
        "saturation_window": args.saturation_window,
        "effective_saturation_window": float(args.saturation_window) if args.saturation_window > 0.0 else float(args.total_iterations) / float(len(common["group_indices"])),
        "workers": worker_count,
        "multiprocessing_start_method": start_method,
        "batch_elapsed_sec": batch_elapsed_sec,
        "sum_run_elapsed_sec": sum_run_elapsed_sec,
        "parallel_speedup_estimate": parallel_speedup_estimate,
        "omega_b": args.omega_b,
        "omega_i": args.omega_i,
        "omega_s": args.omega_s,
        "omega_h": args.omega_h,
        "epsilon": args.epsilon,
        "improvement_rho": args.improvement_rho,
        "evaluation_mode": args.evaluation_mode,
        "risk_step": args.risk_step,
        "ttc_horizon": args.ttc_horizon,
        "tlc_horizon": args.tlc_horizon,
        "object_radius": args.object_radius,
        "boundary_radius": args.boundary_radius,
        "max_boundaries": args.max_boundaries,
        "skip_static_graph": args.skip_static_graph,
        "summary_root": os.fspath(args.summary_root),
        "candidate_trajectory_path": os.fspath(args.candidate_trajectory_path),
        "coarse_labels_path": os.fspath(args.coarse_labels_path),
        "coarse_representatives_path": os.fspath(args.coarse_representatives_path),
        "sub_labels_path": os.fspath(args.sub_labels_path),
        "sub_representatives_path": os.fspath(args.sub_representatives_path),
        "representative_points": int(args.representative_points),
        "notes": aggregate_notes(),
        "means": {},
    }
    numeric_keys = [
        "elapsed_sec",
        "local_evaluations",
        "global_rounds",
        "groups_visited",
        "phase1_done_groups",
        "selected_count",
        "best_selected_score",
        "mean_selected_score",
        "max_group_best_B",
        "mean_group_best_B",
        "mean_recent_improvement_I",
        "groups_with_I_gt_epsilon",
        "mean_saturation_S",
        "groups_saturated_gt_0_8",
        "mean_eta",
        "max_eta",
        "total_hard_count",
        "max_group_eval_share",
        "group_eval_gini",
        "selected_group_rounds_unique",
    ]
    for key in numeric_keys:
        values = [float_value(row.get(key)) for row in ok_rows if row.get(key) not in ("", None)]
        aggregate["means"][key] = mean(values)
    if isinstance(group_schedule, dict) and group_schedule:
        aggregate.update(group_schedule)
    (args.summary_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
