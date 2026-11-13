#!/usr/bin/env python3
"""Run seeded random direct HGS sampling from a summary JSON.

The runner samples config/token pairs without replacement from
data/auto_reconstruction/summary.json, then uses the B/S/hard-rate scheduler
defined by hgs_summary_exploration_direct_new.py. It writes role-aware risk
signatures and case diversity, including the v5 distinction between road edges
and drivable-area holes/barriers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

import hgs_random_exploration_direct as direct  # noqa: E402
import hgs_summary_exploration_direct_new as search  # noqa: E402


DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 0
PAIR_SOURCE_MODE = "summary_json_seeded_sample"


def parse_args() -> argparse.Namespace:
    """Parse the search options with seeded 100-pair sampling defaults."""
    summary_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    summary_parser.add_argument(
        "--summary-json",
        type=Path,
        default=search.DEFAULT_SUMMARY_JSON,
        help="JSON mapping config YAML filenames to the central tokens to sample.",
    )
    summary_args, remaining = summary_parser.parse_known_args()

    if any(item in {"-h", "--help"} for item in sys.argv[1:]):
        print(
            "Seeded-summary options:\n"
            "  --summary-json PATH  JSON mapping config YAML names to token lists.\n"
            f"  --runs N            Number of pairs sampled without replacement (default: {DEFAULT_SAMPLE_SIZE}).\n"
            f"  --seed SEED         Reproducible sampling seed (default: {DEFAULT_SEED}).\n",
            flush=True,
        )

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        parser = direct.build_parser(include_omega_i=False)
        parser.description = (
            "Run seeded summary HGS and write role-aware v5 traffic-case outputs for final candidates."
        )
        parser.set_defaults(
            runs=DEFAULT_SAMPLE_SIZE,
            seed=DEFAULT_SEED,
            coarse_labels_path=search.HISTORY15_DEFAULT_PATHS["coarse_labels_path"],
            coarse_representatives_path=search.HISTORY15_DEFAULT_PATHS["coarse_representatives_path"],
            sub_labels_path=search.HISTORY15_DEFAULT_PATHS["sub_labels_path"],
            sub_representatives_path=search.HISTORY15_DEFAULT_PATHS["sub_representatives_path"],
            representative_points=search.HISTORY15_REPRESENTATIVE_POINTS,
            q_max=20,
            saturation_window=0.0,
            epsilon=0.08,
            improvement_rho=0.01,
        )
        args = parser.parse_args()
    finally:
        sys.argv = original_argv

    args.summary_json = Path(summary_args.summary_json)
    args.seed_was_explicit = search.cli_option_present("--seed")
    return args


def write_sample_details(
    summary_root: Path,
    pairs: Sequence[Tuple[Path, str, str]],
    seed: int,
) -> None:
    rows = [
        {
            "run_id": run_id,
            "seed": seed,
            "config": os.fspath(config_path.relative_to(PROJECT_ROOT)),
            "road_block_name": road_block_name,
            "token": token,
        }
        for run_id, (config_path, road_block_name, token) in enumerate(pairs, start=1)
    ]
    direct.write_csv(summary_root / "sampled_summary_pairs.csv", rows)
    direct.write_csv(summary_root / "run_summary_column_notes.csv", direct.run_summary_column_notes())


def build_aggregate(
    *,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    total_summary_pairs: int,
    worker_count: int,
    start_method: str,
    batch_elapsed_sec: float,
    common: Dict[str, Any],
) -> Dict[str, Any]:
    aggregate = search.build_aggregate(
        args=args,
        rows=rows,
        total_summary_pairs=total_summary_pairs,
        run_limit=None,
        worker_count=worker_count,
        start_method=start_method,
        batch_elapsed_sec=batch_elapsed_sec,
        common=common,
    )
    notes = aggregate["notes"]
    notes["purpose"] = (
        "Aggregate statistics over seeded random summary config-token runs using the "
        "B/S/hard-rate hierarchical greedy scheduler. Risk signatures and case diversity "
        "are computed during this run with v5 static boundary roles."
    )
    notes["metric_notes"]["seed"] = (
        "Seed supplied to random.Random for sampling config/token pairs without replacement."
    )
    notes["metric_notes"]["seed_source"] = (
        "user if --seed was supplied; default_zero when the default reproducible seed is used."
    )
    notes["metric_notes"]["runs_requested"] = (
        "Number of config-token pairs sampled without replacement from --summary-json."
    )
    notes["metric_notes"]["pair_source"] = (
        "Pairs are sampled without replacement from deterministic summary-json order using --seed."
    )

    aggregate["seed"] = int(args.seed)
    aggregate["seed_source"] = "user" if args.seed_was_explicit else "default_zero"
    aggregate["pair_source_mode"] = PAIR_SOURCE_MODE
    aggregate["sample_size"] = len(rows)
    aggregate["sampled_pairs_available"] = int(total_summary_pairs)
    aggregate.pop("runs_prefix_limit", None)
    return aggregate


def main() -> None:
    args = parse_args()
    search.validate_runner_args(args)
    search.install_role_aware_case_diversity()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    batch_start = time.time()

    all_pairs = search.collect_summary_pairs(args.summary_json, args.config_dir)
    if len(all_pairs) < args.runs:
        raise ValueError(
            f"Requested --runs {args.runs}, but summary JSON contains only {len(all_pairs)} config-token pairs."
        )
    pairs = random.Random(int(args.seed)).sample(all_pairs, int(args.runs))

    args.summary_root.mkdir(parents=True, exist_ok=True)
    if args.write_run_details:
        write_sample_details(args.summary_root, pairs, int(args.seed))

    search.install_b_s_eta_scheduler()
    search_args = search.make_search_args(args, pairs[0][0])
    common = search.load_common_inputs(search_args)
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
            f"Running {len(tasks)} seeded summary config-token runs with {worker_count} workers "
            f"({start_method} start method, seed={args.seed}).",
            flush=True,
        )
        with context.Pool(
            processes=worker_count,
            initializer=search.init_worker,
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
        worker_count=worker_count,
        start_method=start_method,
        batch_elapsed_sec=batch_elapsed_sec,
        common=common,
    )
    search.add_role_aware_case_summary(aggregate, args.summary_root)
    (args.summary_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()

