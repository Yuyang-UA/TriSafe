import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
# --- 新增：可视化所需的模块 ---
from navsim.visualization.plots import plot_bev_with_agent  # 核心可视化函数
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


# --- 新增：图像坐标系配置函数 ---
def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)
    ax.invert_xaxis()
    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run Visualization (Modified from PDMS evaluation).
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # 新增：接收主进程传来的输出路径并建立文件夹
    output_dir = Path(args[0]["output_dir"])
    save_path = output_dir / "visualizations"
    save_path.mkdir(parents=True, exist_ok=True)

    # 初始化 Agent
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # 初始化 DataLoaders (保留 metric_cache_loader 以维持原始 token 过滤逻辑)
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

    pdm_results: List[pd.DataFrame] = []

    # ==========================================
    # first stage (Original Frames)
    # ==========================================
    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one
    tokens_to_evaluate_stage_one = list(set(scene_loader_tokens_stage_one) & set(metric_cache_loader.tokens))

    for idx, (token) in enumerate(tokens_to_evaluate_stage_one):
        logger.info(
            f"Processing stage one visualization {idx + 1} / {len(tokens_to_evaluate_stage_one)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)  # 始终加载 Scene 用于可视化

            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            # --- 修改核心：替换打分逻辑为绘制逻辑 ---
            human_trajectory = scene.get_future_trajectory()
            frame_idx = scene.scene_metadata.num_history_frames - 1

            fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
            add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
            add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["agent"])

            configure_bev_ax(ax)
            configure_ax(ax)

            # 保存图片并清理内存
            filename = save_path / f"{token}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 构建简化的结果 DataFrame 以兼容外层架构
            score_row_stage_one = pd.DataFrame([{"token": token, "valid": True}])

        except Exception:
            logger.warning(f"----------- Agent visualization failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([{"token": token, "valid": False}])

        pdm_results.append(score_row_stage_one)

    # ==========================================
    # second stage (Synthetic Frames)
    # ==========================================
    scene_loader_tokens_stage_two = scene_loader.reactive_tokens_stage_two
    tokens_to_evaluate_stage_two = list(set(scene_loader_tokens_stage_two) & set(metric_cache_loader.tokens))

    for idx, (token) in enumerate(tokens_to_evaluate_stage_two):
        logger.info(
            f"Processing stage two visualization {idx + 1} / {len(tokens_to_evaluate_stage_two)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)

            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            # --- 修改核心：替换打分逻辑为绘制逻辑 ---
            human_trajectory = scene.get_future_trajectory()
            frame_idx = scene.scene_metadata.num_history_frames - 1

            fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
            add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
            add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["agent"])

            configure_bev_ax(ax)
            configure_ax(ax)

            filename = save_path / f"{token}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            score_row_stage_two = pd.DataFrame([{"token": token, "valid": True}])

        except Exception:
            logger.warning(f"----------- Agent visualization failed for token {token}:")
            traceback.print_exc()
            score_row_stage_two = pd.DataFrame([{"token": token, "valid": False}])

        pdm_results.append(score_row_stage_two)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running Visualization.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # 初始化主进程 DataLoader 以分配 Token
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))

    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")

    logger.info(f"Starting visualization of {len(tokens_to_evaluate)} scenarios...")

    # 构造任务分配点 (新增 output_dir)
    output_dir = ""
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "output_dir": output_dir,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    # 执行多线程/多进程渲染
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    # 聚合简化后的结果
    pdm_score_df = pd.concat(score_rows)

    # --- 删除原来的复杂分数汇总和 csv 输出逻辑 ---

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    logger.info(
        f"""
        Finished running visualization.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Images are stored in: {Path(cfg.output_dir) / 'visualizations'}.
        """
    )

    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    main()