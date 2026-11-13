#!/usr/bin/env python3
"""Run summary-driven direct HGS with B/S/hard-rate group scheduling.

This runner evaluates every config/token pair listed in a summary JSON in
deterministic config-name/token-list order. It writes role-aware risk signatures
and traffic-case diversity after each selected-candidate search, including the
v5 distinction between a drivable road edge and a drivable-area hole/barrier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
HGS_DIR = Path(__file__).resolve().parent
if os.fspath(HGS_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(HGS_DIR))

import hierarchical_greedy_search as hgs  # noqa: E402
import hgs_random_exploration_direct as direct  # noqa: E402
import hgs_random_exploration_direct_diversity as diversity  # noqa: E402


DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "data/auto_reconstruction/summary.json"
PAIR_SOURCE_MODE = "summary_json_order"
HISTORY15_DESCRIPTOR_DIR = Path("trajectory_candidate/dictionary_structure")
HISTORY15_DESCRIPTOR_STEM = (
    "coarse_group_9_kmeans_history15_descriptors_sub60_40_40_20_20_20_20_20_10"
)
HISTORY15_DEFAULT_PATHS = {
    "coarse_labels_path": HISTORY15_DESCRIPTOR_DIR / "labels.npy",
    "coarse_representatives_path": HISTORY15_DESCRIPTOR_DIR / "representatives.npy",
    "sub_labels_path": HISTORY15_DESCRIPTOR_DIR / "sub_labels.npy",
    "sub_representatives_path": HISTORY15_DESCRIPTOR_DIR / "sub_representatives.npy",
}
HISTORY15_REPRESENTATIVE_POINTS = 15


def cli_option_present(*names: str) -> bool:
    """Return whether an option was explicitly supplied, including --x=y."""
    for item in sys.argv[1:]:
        for name in names:
            if item == name or item.startswith(f"{name}="):
                return True
    return False


def parse_args() -> argparse.Namespace:
    """Parse summary options and the direct-runner options except --omega-i."""
    summary_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    summary_parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="JSON mapping config YAML filenames to the central tokens to evaluate.",
    )
    summary_args, remaining = summary_parser.parse_known_args()

    if any(item in {"-h", "--help"} for item in sys.argv[1:]):
        print(
            "Summary-runner options:\n"
            "  --summary-json PATH  JSON mapping config YAML names to token lists "
            f"(default: {DEFAULT_SUMMARY_JSON})\n"
            "  --runs N            Deterministic prefix limit; omit to run every JSON pair.\n"
            "  --seed SEED         Accepted for compatibility but unused (there is no sampling).\n",
            flush=True,
        )

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        parser = direct.build_parser(include_omega_i=False)
        parser.description = (
            "Run deterministic summary HGS and write role-aware v5 traffic-case outputs for final candidates."
        )
        parser.set_defaults(
            coarse_labels_path=HISTORY15_DEFAULT_PATHS["coarse_labels_path"],
            coarse_representatives_path=HISTORY15_DEFAULT_PATHS["coarse_representatives_path"],
            sub_labels_path=HISTORY15_DEFAULT_PATHS["sub_labels_path"],
            sub_representatives_path=HISTORY15_DEFAULT_PATHS["sub_representatives_path"],
            representative_points=HISTORY15_REPRESENTATIVE_POINTS,
            q_max=20,
            saturation_window=0.0,
            epsilon=0.08,
            improvement_rho=0.01,
        )
        args = parser.parse_args()
    finally:
        sys.argv = original_argv

    args.summary_json = Path(summary_args.summary_json)
    args.runs_was_explicit = cli_option_present("--runs")
    args.seed_was_explicit = cli_option_present("--seed")
    return args


def validate_runner_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.group_eval_share_cap < 0.0 or args.group_eval_share_cap > 1.0:
        raise ValueError("--group-eval-share-cap must be in [0, 1].")
    if args.saturation_window < 0.0:
        raise ValueError("--saturation-window must be non-negative.")


def config_path_from_summary_entry(config_dir: Path, config_entry: str) -> Path:
    path = Path(config_entry)
    return path if path.is_absolute() else config_dir / path


def collect_summary_pairs(summary_json: Path, config_dir: Path) -> List[Tuple[Path, str, str]]:
    """Load and validate {config_yaml: [token, ...]} summary entries."""
    try:
        payload = json.loads(summary_json.read_text(encoding="utf8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Summary JSON does not exist: {summary_json}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Summary JSON is not valid JSON: {summary_json}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected {summary_json} to contain a JSON object mapping config YAML names to token lists.")

    pairs: List[Tuple[Path, str, str]] = []
    seen_pairs = set()
    for config_entry in sorted(payload):
        if not isinstance(config_entry, str) or not config_entry:
            raise ValueError(f"Summary config key must be a non-empty string, got {config_entry!r}.")
        tokens = payload[config_entry]
        if not isinstance(tokens, list):
            raise ValueError(f"Summary entry {config_entry!r} must be a token list, got {type(tokens).__name__}.")

        config_path = config_path_from_summary_entry(config_dir, config_entry)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config YAML referenced by {summary_json} does not exist: {config_path}")
        road_block_name, _ = direct.parse_config_tokens(config_path)

        for token in tokens:
            if not isinstance(token, str) or not token.strip():
                raise ValueError(f"Summary entry {config_entry!r} contains an invalid token: {token!r}.")
            pair_key = (os.fspath(config_path.resolve()), token)
            if pair_key in seen_pairs:
                raise ValueError(f"Duplicate config-token pair in {summary_json}: {config_entry!r}, token={token!r}.")
            seen_pairs.add(pair_key)
            pairs.append((config_path, road_block_name, token))
    return pairs


def write_pair_details(summary_root: Path, pairs: Sequence[Tuple[Path, str, str]]) -> None:
    rows = [
        {
            "run_id": run_id,
            "config": os.fspath(config_path.relative_to(PROJECT_ROOT)),
            "road_block_name": road_block_name,
            "token": token,
        }
        for run_id, (config_path, road_block_name, token) in enumerate(pairs, start=1)
    ]
    direct.write_csv(summary_root / "summary_pairs.csv", rows)
    direct.write_csv(summary_root / "run_summary_column_notes.csv", direct.run_summary_column_notes())


def group_fitness_without_improvement(state: hgs.GroupState, args: argparse.Namespace) -> float:
    """Schedule by best score, saturation, and hard-event rate only."""
    if state.exhausted:
        return -math.inf
    saturation = hgs.group_saturation(
        state,
        args.local_window,
        hgs.saturation_window_for_group(args, state.group),
        args.epsilon,
        args.saturation_temperature,
        args.improvement_rho,
    )
    return (
        float(args.omega_b) * float(state.best_score)
        - float(args.omega_s) * saturation
        - float(args.omega_h) * hgs.group_eta(state)
    )


def install_b_s_eta_scheduler() -> None:
    """Install the revised scheduler in this process before constructing HGS."""
    hgs.group_fitness = group_fitness_without_improvement


def install_role_aware_case_diversity() -> None:
    """Attach role-aware risk signatures to direct-runner results in this process."""
    direct.run_one_direct = diversity.run_one_direct_diversity
    direct.write_selected_candidates_json = diversity.write_selected_candidates_json_with_diversity


def make_search_args(args: argparse.Namespace, first_config: Path) -> argparse.Namespace:
    """Use the established HGS argument bridge without exposing --omega-i."""
    compatible_args = argparse.Namespace(**vars(args), omega_i=0.0)
    return direct.make_search_args(compatible_args, first_config)


def add_role_aware_case_summary(aggregate: Dict[str, Any], summary_root: Path) -> None:
    """Record the v5 static-boundary-role outputs produced for this summary run."""
    risk_path = summary_root / diversity.RISK_CASE_FILENAME
    if not risk_path.is_file():
        return
    payload = json.loads(risk_path.read_text(encoding="utf8"))
    summary = payload.get("summary", {})
    aggregate["case_diversity_status"] = "computed"
    aggregate["risk_case_schema_version"] = str(payload.get("schema_version", ""))
    aggregate["risk_case_summary"] = {
        "selected_trajectory_quantity": summary.get("selected_trajectory_quantity", 0),
        "final_traffic_case_quantity": summary.get("final_traffic_case_quantity", 0),
        "mean_final_traffic_case_quantity_per_token": summary.get("mean_final_traffic_case_quantity_per_token", 0.0),
    }
    aggregate["case_diversity_outputs"] = {
        "risk_case_signatures": os.fspath(risk_path),
        "selected_candidates_with_cases": os.fspath(summary_root / "selected_candidates_by_token.json"),
    }


def group_schedule_metadata(common: Mapping[str, Any], search_args: argparse.Namespace) -> Dict[str, Any]:
    sub_ids_by_group = hgs.build_sub_ids_by_group(
        common["coarse_labels"],
        common["sub_labels"],
        common["group_indices"],
    )
    subcluster_counts = {
        str(int(group)): int(len(sub_ids_by_group[int(group)]))
        for group in common["group_indices"]
    }
    total_subclusters = int(sum(subcluster_counts.values()))
    q_max = max(1, int(search_args.q_max))
    representative_blocks = {
        group: int(math.ceil(float(count) / float(q_max))) if count > 0 else 0
        for group, count in subcluster_counts.items()
    }
    requested_window = float(search_args.saturation_window)
    if requested_window > 0.0:
        saturation_mode = "fixed"
        saturation_windows = {group: requested_window for group in subcluster_counts}
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


def load_common_inputs(search_args: argparse.Namespace) -> Dict[str, Any]:
    common = direct.load_common_inputs(search_args)
    common["_group_schedule"] = group_schedule_metadata(common, search_args)
    return common


def init_worker(
    search_args: argparse.Namespace,
    common: Dict[str, Any],
    navsim_log_path: Path,
    output_root: Path,
) -> None:
    install_role_aware_case_diversity()
    install_b_s_eta_scheduler()
    direct.init_worker(search_args, common, navsim_log_path, output_root)


def aggregate_notes() -> Dict[str, Any]:
    notes = direct.aggregate_notes()
    notes["purpose"] = (
        "Aggregate statistics over deterministic summary-driven hierarchical greedy search runs. "
        "Role-aware risk signatures and case diversity are computed during this run."
    )
    notes["search_formula"]["A"] = "omega_b * B - omega_s * S - omega_h * eta"
    notes["search_formula"]["I"] = (
        "Recent best-score improvement over local_window evaluations. It is used only inside S, "
        "not as an additive scheduling term."
    )
    notes["metric_notes"]["seed"] = "Not used: summary pairs are evaluated deterministically."
    notes["metric_notes"]["seed_source"] = "not_used_summary_json"
    notes["metric_notes"]["runs_requested"] = "Number of deterministic summary config-token runs."
    notes["metric_notes"]["saturation_window"] = (
        "L_prime in S. 0 selects an adaptive per-group window proportional to subcluster count."
    )
    notes["metric_notes"]["mean_recent_improvement_I"] = (
        "Diagnostic only: recent improvement remains an input to S, not an additive fitness term."
    )
    notes["metric_notes"]["groups_with_I_gt_epsilon"] = (
        "Diagnostic only: groups whose improvement reduces their saturation penalty."
    )
    notes["metric_notes"]["case_diversity"] = (
        "Computed during this stage with v5 static boundary roles (road edge versus hole/barrier)."
    )
    return notes


def build_aggregate(
    *,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    total_summary_pairs: int,
    run_limit: Optional[int],
    worker_count: int,
    start_method: str,
    batch_elapsed_sec: float,
    common: Mapping[str, Any],
) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    aggregate: Dict[str, Any] = {
        "seed": None,
        "seed_source": "not_used_summary_json",
        "pair_source_mode": PAIR_SOURCE_MODE,
        "summary_json": os.fspath(args.summary_json),
        "summary_pairs_available": int(total_summary_pairs),
        "runs_prefix_limit": run_limit,
        "runs_requested": len(rows),
        "runs_ok": len(ok_rows),
        "runs_failed": len(rows) - len(ok_rows),
        "total_iterations": args.total_iterations,
        "q_max": args.q_max,
        "group_eval_share_cap": args.group_eval_share_cap,
        "saturation_window": args.saturation_window,
        "workers": worker_count,
        "multiprocessing_start_method": start_method,
        "batch_elapsed_sec": batch_elapsed_sec,
        "sum_run_elapsed_sec": sum(direct.float_value(row.get("elapsed_sec")) for row in rows),
        "omega_b": args.omega_b,
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
        "config_dir": os.fspath(args.config_dir),
        "navsim_log_path": os.fspath(args.navsim_log_path),
        "candidate_trajectory_path": os.fspath(args.candidate_trajectory_path),
        "coarse_labels_path": os.fspath(args.coarse_labels_path),
        "coarse_representatives_path": os.fspath(args.coarse_representatives_path),
        "sub_labels_path": os.fspath(args.sub_labels_path),
        "sub_representatives_path": os.fspath(args.sub_representatives_path),
        "representative_points": int(args.representative_points),
        "case_diversity_status": "not_computed",
        "notes": aggregate_notes(),
        "means": {},
    }
    aggregate["parallel_speedup_estimate"] = (
        float(aggregate["sum_run_elapsed_sec"] / batch_elapsed_sec) if batch_elapsed_sec > 0.0 else 0.0
    )
    for key in (
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
    ):
        values = [
            direct.float_value(row.get(key))
            for row in ok_rows
            if row.get(key) not in ("", None)
        ]
        aggregate["means"][key] = direct.mean(values)

    group_schedule = common.get("_group_schedule", {})
    if isinstance(group_schedule, Mapping) and group_schedule:
        aggregate.update(group_schedule)
        aggregate["effective_saturation_window"] = float(search_window_display(args, group_schedule))
    return aggregate


def search_window_display(args: argparse.Namespace, group_schedule: Mapping[str, Any]) -> float:
    if args.saturation_window > 0.0:
        return float(args.saturation_window)
    windows = group_schedule.get("effective_saturation_window_by_group", {})
    if isinstance(windows, Mapping) and windows:
        return float(sum(float(value) for value in windows.values()) / len(windows))
    return float(args.total_iterations)


def main() -> None:
    args = parse_args()
    install_role_aware_case_diversity()
    validate_runner_args(args)
    batch_start = time.time()
    all_pairs = collect_summary_pairs(args.summary_json, args.config_dir)
    if not all_pairs:
        raise RuntimeError(f"No config-token pairs found in summary JSON: {args.summary_json}")

    run_limit: Optional[int] = int(args.runs) if args.runs_was_explicit else None
    if run_limit is not None:
        if run_limit <= 0:
            raise ValueError("--runs must be positive when used as a summary-pair prefix limit.")
        if run_limit > len(all_pairs):
            raise ValueError(f"Requested --runs {run_limit}, but summary JSON contains only {len(all_pairs)} pairs.")
        pairs = all_pairs[:run_limit]
    else:
        pairs = all_pairs

    args.runs = len(pairs)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    if args.write_run_details:
        write_pair_details(args.summary_root, pairs)

    install_b_s_eta_scheduler()
    search_args = make_search_args(args, pairs[0][0])
    common = load_common_inputs(search_args)
    tasks = [
        (run_id, config_path, road_block_name, token)
        for run_id, (config_path, road_block_name, token) in enumerate(pairs, start=1)
    ]
    worker_count = min(int(args.workers), len(tasks))
    rows: List[Dict[str, Any]] = []
    start_method = "serial"

    if worker_count == 1:
        log_cache: Dict[str, List[Dict[str, Any]]] = {}
        map_cache: Dict[str, Any] = {}
        for run_id, config_path, road_block_name, token in tasks:
            print(f"[{run_id:02d}/{args.runs}] {config_path.name} token={token}", flush=True)
            row = direct.run_one_direct(
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
                direct.write_incremental_run_summary(args.summary_root, rows)
                direct.write_selected_candidates_json(args.summary_root, rows)
    else:
        context = direct.multiprocessing_context()
        start_method = context.get_start_method()
        print(
            f"Running {len(tasks)} summary config-token runs with {worker_count} workers "
            f"({start_method} start method).",
            flush=True,
        )
        with context.Pool(
            processes=worker_count,
            initializer=init_worker,
            initargs=(search_args, common, args.navsim_log_path, search_args.output_root),
        ) as pool:
            for row in pool.imap_unordered(direct.run_one_worker, tasks):
                rows.append(row)
                print(
                    f"[{len(rows):02d}/{args.runs}] finished run_id={direct.int_value(row.get('run_id'))} "
                    f"status={row.get('status')} token={row.get('token')}",
                    flush=True,
                )
                if args.write_run_details:
                    direct.write_incremental_run_summary(args.summary_root, rows)
                    direct.write_selected_candidates_json(args.summary_root, rows)

    rows = direct.sorted_run_rows(rows)
    direct.write_selected_candidates_json(args.summary_root, rows)
    batch_elapsed_sec = round(time.time() - batch_start, 3)
    aggregate = build_aggregate(
        args=args,
        rows=rows,
        total_summary_pairs=len(all_pairs),
        run_limit=run_limit,
        worker_count=worker_count,
        start_method=start_method,
        batch_elapsed_sec=batch_elapsed_sec,
        common=common,
    )
    add_role_aware_case_summary(aggregate, args.summary_root)
    (args.summary_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()

