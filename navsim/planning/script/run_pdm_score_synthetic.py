"""Evaluate standalone synthetic NAVSIM scenes with five classic PDM metrics.

This runner executes only the reactive synthetic-scene path. It does not load,
score, pair, weight, or aggregate any corresponding original scene.
"""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

METRIC_COLUMNS = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "history_comfort": "C",
    "pdm_score": "score",
}


def _load_synthetic_driving_commands(
    path: Optional[str], tokens: Iterable[str]
) -> Optional[Dict[str, np.ndarray]]:
    """Load four-value one-hot commands and require coverage of evaluated tokens."""
    if path is None:
        return None

    with Path(path).open() as command_file:
        commands = json.load(command_file)
    if not isinstance(commands, dict):
        raise ValueError(f"Driving commands in {path} must map tokens to one-hot vectors")

    missing_tokens = sorted(set(tokens) - commands.keys())
    if missing_tokens:
        raise ValueError(f"Missing driving commands in {path} for tokens: {missing_tokens}")

    validated_commands = {}
    for token, command in commands.items():
        values = np.asarray(command)
        if values.shape != (4,) or not np.isin(values, [0, 1]).all() or values.sum() != 1:
            raise ValueError(
                f"Driving command for {token} in {path} must be a four-value one-hot vector, got {command!r}"
            )
        validated_commands[token] = values.astype(np.int64)
    return validated_commands


def run_synthetic_pdm_score(
    args: List[Dict[str, Union[List[str], DictConfig]]],
) -> List[pd.DataFrame]:
    """Run agent inference and reactive scoring for synthetic tokens only."""

    node_id = int(args[0].get("node_id", 0))
    thread_id = str(uuid.uuid4())
    log_names = [item["log_file"] for item in args]
    tokens = [token for item in args for token in item["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    driving_commands = _load_synthetic_driving_commands(cfg.get("synthetic_driving_commands_path"), tokens)

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    available_tokens = set(metric_cache_loader.tokens)
    tokens_to_evaluate = sorted(set(scene_loader.reactive_tokens_stage_two) & available_tokens)
    results: List[pd.DataFrame] = []
    for index, token in enumerate(tokens_to_evaluate):
        logger.info(
            "Processing standalone synthetic scenario %d / %d in thread_id=%s, node_id=%s",
            index + 1,
            len(tokens_to_evaluate),
            thread_id,
            node_id,
        )
        poses, interval = None, float("nan")
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            if metric_cache.scene_type != SceneFrameType.SYNTHETIC:
                raise ValueError(
                    f"Metric cache for {token} is not marked synthetic. "
                    "Regenerate it after rendering with 17-character tokens."
                )

            agent_input = scene_loader.get_agent_input_from_token(token)
            if driving_commands is not None:
                agent_input.ego_statuses[-1].driving_command = driving_commands[token].copy()
            if agent.requires_scene:
                trajectory = agent.compute_trajectory(
                    agent_input, scene_loader.get_scene_from_token(token)
                )
            else:
                trajectory = agent.compute_trajectory(agent_input)

            score_row, _ = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            poses = np.asarray(trajectory.poses, dtype=np.float32)
            interval = float(trajectory.trajectory_sampling.interval_length)
        except Exception:
            logger.warning("Agent or scorer failed for synthetic token %s", token)
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False

        score_row["token"] = token
        score_row["trajectory_poses"] = pd.Series([poses], index=score_row.index, dtype=object)
        score_row["trajectory_interval"] = interval
        results.append(score_row)
    return results


def _save_trajectories(records: pd.DataFrame, csv_path: Path) -> Path:
    """Save planned trajectories next to the results CSV for failure attribution."""
    records = records[records["trajectory_poses"].map(lambda poses: poses is not None)]
    records = records.sort_values("token")
    intervals = records["trajectory_interval"].unique()
    if len(intervals) != 1:
        raise ValueError(f"Expected one trajectory interval, got {intervals}")
    poses = np.stack(records["trajectory_poses"].to_list()).astype(np.float32)
    path = csv_path.with_name(f"{csv_path.stem}_trajectories.npz")
    np.savez_compressed(
        path,
        tokens=np.asarray(records["token"].to_list()),
        poses=poses,
        num_poses=poses.shape[1],
        interval_length=float(intervals[0]),
    )
    return path


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Run mapping-free evaluation and save per-scene plus mean five-metric scores."""

    if not cfg.scorer.config.five_metric_only:
        raise ValueError(
            "Synthetic-only evaluation requires scorer.config.five_metric_only=True"
        )

    # Ray prepends the driver's current directory to worker sys.path. Start it
    # from the repository root so workers import this NAVSIM package rather
    # than another installed copy.
    os.chdir(Path(__file__).resolve().parents[3])
    worker = build_worker(cfg)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    synthetic_tokens = set(scene_loader.reactive_tokens_stage_two)
    cached_tokens = set(metric_cache_loader.tokens)
    missing_cache_tokens = sorted(synthetic_tokens - cached_tokens)
    tokens_to_evaluate = synthetic_tokens & cached_tokens
    if missing_cache_tokens:
        logger.warning(
            "Skipping %d synthetic scenes without metric cache: %s",
            len(missing_cache_tokens),
            missing_cache_tokens,
        )
    if not tokens_to_evaluate:
        raise RuntimeError(
            "No synthetic scene has a metric cache. Run run_metric_caching.py first."
        )

    data_points = []
    for log_file, tokens in scene_loader.get_tokens_list_per_log().items():
        selected = sorted(set(tokens) & tokens_to_evaluate)
        if selected:
            data_points.append({"cfg": cfg, "log_file": log_file, "tokens": selected})

    score_rows = worker_map(worker, run_synthetic_pdm_score, data_points)
    pdm_score_df = pd.concat(score_rows, ignore_index=True)
    trajectory_records = pdm_score_df[["token", "trajectory_poses", "trajectory_interval"]]
    output_columns = ["token", "valid", *METRIC_COLUMNS]
    pdm_score_df = pdm_score_df[output_columns].rename(columns=METRIC_COLUMNS)

    successful = pdm_score_df[pdm_score_df["valid"]]
    summary = {"token": "synthetic_mean", "valid": len(successful) == len(pdm_score_df)}
    for column in METRIC_COLUMNS.values():
        summary[column] = successful[column].mean() if not successful.empty else float("nan")
    pdm_score_df.loc[len(pdm_score_df)] = summary

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path, index=False)
    trajectories_path = _save_trajectories(trajectory_records, csv_path)
    logger.info("Saved planned trajectories: %s", trajectories_path)

    logger.info(
        "Finished standalone synthetic evaluation: %d successful, %d failed. "
        "Mean five-metric score: %s. Results: %s",
        len(successful),
        len(pdm_score_df) - 1 - len(successful),
        summary["score"],
        csv_path,
    )
    if cfg.verbose:
        logger.info("Detailed synthetic results:\n%s", pdm_score_df)


if __name__ == "__main__":
    main()
