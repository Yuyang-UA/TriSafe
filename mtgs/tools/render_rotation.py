#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#

#!/usr/bin/env python
from __future__ import annotations

import gzip
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Callable

import mediapy as media
import numpy as np
import torch
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated
import yaml

from nerfstudio.cameras.camera_paths import get_interpolated_poses_many, get_path_from_json
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datasets.base_dataset import Dataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    filenames: Optional[List[str]] = None,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if "images" in output_format:
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if "video" in output_format:
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)
    if filenames is not None:
        assert len(filenames) == len(cameras), "filenames must have the same length as cameras"

    with ExitStack() as stack:
        writer = None
        render_images = []
        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                camera = cameras[camera_idx : camera_idx + 1]
                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            camera, obb_box=obb_box
                        )
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            camera, obb_box=obb_box
                        )

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    is_depth = rendered_output_name.find("depth") != -1
                    if is_depth:
                        output_image = (
                            colormaps.apply_depth_colormap(
                                output_image,
                                accumulation=outputs["accumulation"],
                                near_plane=depth_near_plane,
                                far_plane=depth_far_plane,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                render_image = np.concatenate(render_image, axis=1)
                render_images.append(render_image)
                if "images" in output_format:
                    img_filename = filenames[camera_idx] if filenames is not None else f"{camera_idx:05d}"
                    if image_format == "png":
                        media.write_image(output_image_dir / f"{img_filename}.png", render_image, fmt="png")
                    if image_format == "jpeg":
                        media.write_image(
                            output_image_dir / f"{img_filename}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                        )
                if "video" in output_format:
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))

    return render_images

def find_ckpt_path(config: TrainerConfig) -> Tuple[Path, int]:
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    return load_path

def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    # config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    # influence the gpu usage
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    config.load_dir = config.get_checkpoint_dir()
    checkpoint_path = find_ckpt_path(config)

    if 'camera_optimizer' in config.pipeline.model.__dict__ \
        and config.pipeline.model.camera_optimizer.mode == "SO3xR3":
        config.pipeline.datamanager.dataparser.load_cam_optim_from = checkpoint_path
    # frame rate is changed
    config.pipeline.datamanager.dataparser.eval_2hz = False

    if update_config_callback is not None:
        config = update_config_callback(config)


    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    # pipeline config: python/object:mtgs.scene_model.custom_pipeline.MultiTravelEvalPipielineConfig
    pipeline.eval()

    # load checkpointed information
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

    return config, pipeline, checkpoint_path, loaded_state["step"]

def _get_interpolated_camera_path(cameras: Cameras, steps: int, order_poses: bool) -> Cameras:
    """Generate a camera path between two cameras. Uses the camera type of the first camera

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices()
    poses = cameras.camera_to_worlds
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps+1, order_poses=order_poses)
    camera_timestamps = cameras.times.squeeze()

    timestamps_new = []
    for i in range(len(camera_timestamps) - 1):
        timestamps_new.append(
            torch.linspace(camera_timestamps[i], camera_timestamps[i+1], steps+1, dtype=torch.float)[:-1])
    timestamps_new.append(camera_timestamps[-1][None])
    timestamps_new = torch.cat(timestamps_new, dim=0)

    mask = torch.cat([torch.ones(steps, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)])
    mask = mask.repeat(len(cameras) - 1)
    mask[-1] = True

    poses = poses[mask]
    Ks = Ks[mask]

    cameras = Cameras(
        fx=Ks[:, 0, 0],
        fy=Ks[:, 1, 1],
        cx=Ks[0, 0, 2],
        cy=Ks[0, 1, 2],
        camera_type=cameras.camera_type[0],
        camera_to_worlds=poses,
        times=timestamps_new,
        width=cameras.width[0],
        height=cameras.height[0],
    )
    return cameras

def _choose_turn_pivot(origins: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "start":
        return origins[0]
    if mode == "center":
        return origins.mean(dim=0)
    if mode == "origin":
        return torch.zeros(3, dtype=origins.dtype, device=origins.device)
    raise ValueError(f"Unknown turn pivot: {mode}")


def _yaw_rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(angles)
    ones = torch.ones_like(angles)
    c = torch.cos(angles)
    s = torch.sin(angles)
    return torch.stack(
        [
            torch.stack([c, -s, zeros], dim=-1),
            torch.stack([s, c, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def _trajectory_progress(origins: torch.Tensor) -> torch.Tensor:
    if origins.shape[0] < 2:
        return torch.zeros(origins.shape[0], dtype=origins.dtype, device=origins.device)
    segment_lengths = torch.linalg.norm(origins[1:, :2] - origins[:-1, :2], dim=-1)
    cumulative = torch.cat([segment_lengths.new_zeros(1), segment_lengths.cumsum(dim=0)])
    total = cumulative[-1].clamp_min(1e-6)
    return cumulative / total


def _resample_rows(values: torch.Tensor, num_rows: int) -> torch.Tensor:
    if values.shape[0] == num_rows:
        return values
    if values.shape[0] < 2:
        return values.repeat(num_rows, 1)
    source_t = torch.linspace(0.0, 1.0, values.shape[0], dtype=values.dtype, device=values.device)
    target_t = torch.linspace(0.0, 1.0, num_rows, dtype=values.dtype, device=values.device)
    return _interpolate_rows_by_scalar(values, source_t, target_t)


def _interpolate_rows_by_scalar(values: torch.Tensor, source_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.repeat(target_t.shape[0], 1)
    target_t = target_t.clamp(source_t[0], source_t[-1])
    right = torch.searchsorted(source_t, target_t, right=True).clamp(1, values.shape[0] - 1)
    left = right - 1
    denom = (source_t[right] - source_t[left]).clamp_min(1e-6)
    alpha = ((target_t - source_t[left]) / denom).unsqueeze(-1)
    return values[left] * (1.0 - alpha) + values[right] * alpha


def _interpolate_values_by_scalar(values: torch.Tensor, source_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    return _interpolate_rows_by_scalar(values[:, None], source_t, target_t).squeeze(-1)


def _polyline_length_xy(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] < 2:
        return points.new_tensor(0.0)
    return torch.linalg.norm(points[1:, :2] - points[:-1, :2], dim=-1).sum()


def _arc_lengths_xy(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] < 2:
        return points.new_zeros(points.shape[0])
    segment_lengths = torch.linalg.norm(points[1:, :2] - points[:-1, :2], dim=-1)
    return torch.cat([segment_lengths.new_zeros(1), segment_lengths.cumsum(dim=0)])


def _prepare_candidate_trajectory(
    candidate_trajectory: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    prepend_origin: bool,
) -> Optional[torch.Tensor]:
    if candidate_trajectory is None:
        return None
    candidate = candidate_trajectory.to(dtype=dtype, device=device)
    if candidate.ndim != 2 or candidate.shape[1] < 2:
        raise ValueError("Candidate trajectory must have shape (T, 2+) or (T, 3).")
    candidate = candidate[:, :3] if candidate.shape[1] >= 3 else candidate[:, :2]
    if prepend_origin:
        origin = torch.zeros(1, candidate.shape[1], dtype=dtype, device=device)
        candidate = torch.cat([origin, candidate], dim=0)
    return candidate


def _candidate_at_target_times(
    candidate: torch.Tensor,
    target_times_normalized: Optional[torch.Tensor],
    source_duration_seconds: Optional[float],
    candidate_frame_rate: float,
    candidate_prepend_origin: bool,
    num_poses: int,
) -> torch.Tensor:
    if candidate_frame_rate <= 0:
        raise ValueError("--candidate-frame-rate must be positive.")
    if source_duration_seconds is None or target_times_normalized is None:
        return _resample_rows(candidate, num_poses)
    first_sample_index = 0 if candidate_prepend_origin else 1
    candidate_times = (
        torch.arange(candidate.shape[0], dtype=candidate.dtype, device=candidate.device) + first_sample_index
    ) / candidate_frame_rate
    target_seconds = target_times_normalized.to(dtype=candidate.dtype, device=candidate.device).reshape(-1) * source_duration_seconds
    return _interpolate_rows_by_scalar(candidate, candidate_times, target_seconds)


def _replace_camera_path_with_candidate(
    poses: torch.Tensor,
    candidate_trajectory: torch.Tensor,
    candidate_use_yaw: bool,
    candidate_scale_to_source_length: bool,
    candidate_frame_rate: float,
    candidate_prepend_origin: bool,
    candidate_z_mode: Literal["distance", "time", "constant"],
    target_times_normalized: Optional[torch.Tensor],
    source_duration_seconds: Optional[float],
) -> None:
    """Map a local ego-frame candidate [x-forward, y-left, yaw] onto a camera path."""
    origins = poses[..., :3, 3]
    num_poses = origins.shape[0]
    source_origins = origins.clone()
    source_xy = source_origins[:, :2]
    route_vector = source_xy[-1] - source_xy[0]
    route_norm = torch.linalg.norm(route_vector).clamp_min(1e-6)
    forward = route_vector / route_norm
    left = torch.stack((-forward[1], forward[0]))

    candidate = candidate_trajectory
    if candidate_scale_to_source_length:
        candidate_length = _polyline_length_xy(candidate).clamp_min(1e-6)
        source_length = _polyline_length_xy(source_origins).clamp_min(1e-6)
        candidate = candidate.clone()
        candidate[:, :2] = candidate[:, :2] * (source_length / candidate_length)

    candidate = _candidate_at_target_times(
        candidate,
        target_times_normalized=target_times_normalized,
        source_duration_seconds=source_duration_seconds,
        candidate_frame_rate=candidate_frame_rate,
        candidate_prepend_origin=candidate_prepend_origin,
        num_poses=num_poses,
    )
    base = source_origins[0]
    origins[:, :2] = base[:2] + candidate[:, 0:1] * forward + candidate[:, 1:2] * left

    source_s = _arc_lengths_xy(source_origins)
    candidate_s = _arc_lengths_xy(candidate)
    if candidate_z_mode == "distance":
        origins[:, 2] = _interpolate_values_by_scalar(source_origins[:, 2], source_s, candidate_s)
    elif candidate_z_mode == "time":
        origins[:, 2] = source_origins[:, 2]
    elif candidate_z_mode == "constant":
        origins[:, 2] = source_origins[0, 2]
    else:
        raise ValueError(f"Unknown candidate z mode: {candidate_z_mode}")

    if candidate_use_yaw:
        if candidate.shape[1] >= 3:
            yaw = candidate[:, 2]
        else:
            diffs = torch.zeros(num_poses, 2, dtype=origins.dtype, device=origins.device)
            diffs[0] = candidate[1, :2] - candidate[0, :2]
            diffs[-1] = candidate[-1, :2] - candidate[-2, :2]
            if num_poses > 2:
                diffs[1:-1] = candidate[2:, :2] - candidate[:-2, :2]
            yaw = torch.atan2(diffs[:, 1], diffs[:, 0])
        yaw_rots = _yaw_rotation_matrices(yaw)
        base_rotation = poses[0, :3, :3].expand(num_poses, -1, -1)
        poses[..., :3, :3] = torch.bmm(yaw_rots, base_rotation)


def _get_camera_xy(cameras: Cameras) -> np.ndarray:
    return cameras.camera_to_worlds.detach().cpu().numpy()[:, :2, 3]


def _get_travel_duration_seconds(dataparser_outputs: Any, travel_id: Optional[int]) -> Optional[float]:
    metadata = getattr(dataparser_outputs, "metadata", {}) or {}
    travel_infos = metadata.get("multi_travel_frame_timestamps", {})
    travel_info = travel_infos.get(travel_id) if travel_id is not None else None
    if travel_info is None:
        return None
    min_ts = travel_info.get("min_ts")
    max_ts = travel_info.get("max_ts")
    if min_ts is None or max_ts is None or max_ts <= min_ts:
        return None
    return float(max_ts - min_ts) / 1_000_000.0


def _candidate_duration_seconds(candidate_trajectory: Optional[torch.Tensor], frame_rate: float, prepend_origin: bool) -> Optional[float]:
    if candidate_trajectory is None:
        return None
    num_steps = int(candidate_trajectory.shape[0])
    if num_steps <= 0 or frame_rate <= 0:
        return None
    return num_steps / frame_rate


def _trim_cameras_to_normalized_time(cameras: Cameras, num_cameras: int, max_time: float) -> Cameras:
    if cameras.times is None or max_time >= 1.0:
        return cameras
    total = len(cameras)
    usable = (total // num_cameras) * num_cameras
    if usable == 0:
        return cameras
    grouped_cameras = cameras[:usable]
    times = grouped_cameras.times.reshape(-1)
    frame_times = times.reshape(-1, num_cameras).mean(dim=1)
    frame_mask = frame_times <= (max_time + 1e-6)
    if int(frame_mask.sum()) < 2:
        raise ValueError(
            "Candidate horizon leaves fewer than two source frames to render. "
            "Use --candidate-trim-to-horizon False or choose a longer candidate."
        )
    camera_mask = frame_mask.repeat_interleave(num_cameras)
    return grouped_cameras[camera_mask]


def _draw_path_arrow(ax, xy: np.ndarray, color: str) -> None:
    if xy.shape[0] < 2:
        return
    ax.annotate(
        "",
        xy=(xy[-1, 0], xy[-1, 1]),
        xytext=(xy[-2, 0], xy[-2, 1]),
        arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
    )


def _plot_trajectory_comparison(
    original_cameras: Cameras,
    replaced_cameras: Cameras,
    output_path: Path,
    travel_id: Optional[int],
    candidate_index: int,
    representative_camera: str,
    scene_box: Optional[Any] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_xy = _get_camera_xy(original_cameras)
    replaced_xy = _get_camera_xy(replaced_cameras)
    original_z = original_cameras.camera_to_worlds.detach().cpu().numpy()[:, 2, 3]
    replaced_z = replaced_cameras.camera_to_worlds.detach().cpu().numpy()[:, 2, 3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))

    aabb = getattr(scene_box, "aabb", None) if scene_box is not None else None
    if aabb is not None:
        aabb_np = aabb.detach().cpu().numpy() if hasattr(aabb, "detach") else np.asarray(aabb)
        min_x, min_y = aabb_np[0, 0], aabb_np[0, 1]
        max_x, max_y = aabb_np[1, 0], aabb_np[1, 1]
        rect = plt.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            edgecolor="0.35",
            label="scene box",
        )
        ax.add_patch(rect)

    ax.plot(original_xy[:, 0], original_xy[:, 1], color="0.45", linestyle="--", linewidth=2.0, label="original path")
    ax.plot(replaced_xy[:, 0], replaced_xy[:, 1], color="#1f77b4", linewidth=2.4, label=f"candidate {candidate_index}")
    ax.scatter(original_xy[0, 0], original_xy[0, 1], color="0.25", s=45, marker="o", zorder=5, label="original start")
    ax.scatter(replaced_xy[0, 0], replaced_xy[0, 1], color="#2ca02c", s=45, marker="o", zorder=5, label="candidate start")
    ax.scatter(original_xy[-1, 0], original_xy[-1, 1], color="0.25", s=55, marker="x", zorder=5, label="original end")
    ax.scatter(replaced_xy[-1, 0], replaced_xy[-1, 1], color="#d62728", s=55, marker="x", zorder=5, label="candidate end")
    _draw_path_arrow(ax, original_xy, "0.45")
    _draw_path_arrow(ax, replaced_xy, "#1f77b4")

    all_xy = np.concatenate([original_xy, replaced_xy], axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    pad = np.maximum(span * 0.08, 2.0)
    ax.set_xlim(min_xy[0] - pad[0], max_xy[0] + pad[0])
    ax.set_ylim(min_xy[1] - pad[1], max_xy[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("MTGS local x (m)")
    ax.set_ylabel("MTGS local y (m)")
    z_text = f"orig z [{original_z.min():.2f}, {original_z.max():.2f}] m | repl z [{replaced_z.min():.2f}, {replaced_z.max():.2f}] m"
    ax.set_title(f"Trajectory replacement | travel {travel_id} | {representative_camera}\n{z_text}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    CONSOLE.print(f"Saved 2D trajectory comparison to {output_path}", style="bold green")


def _apply_view_exploration_offsets(
    cameras: Cameras,
    lateral_shift_meters: float = 0.0,
    pose_shift_x: float = 0.0,
    pose_shift_y: float = 0.0,
    pose_shift_z: float = 0.0,
    turn_yaw_deg: float = 0.0,
    turn_mode: Literal["progressive", "global"] = "progressive",
    turn_pivot: Literal["start", "center", "origin"] = "start",
    candidate_trajectory: Optional[torch.Tensor] = None,
    candidate_prepend_origin: bool = True,
    candidate_use_yaw: bool = True,
    candidate_scale_to_source_length: bool = False,
    candidate_frame_rate: float = 10.0,
    candidate_z_mode: Literal["distance", "time", "constant"] = "distance",
    source_duration_seconds: Optional[float] = None,
) -> Cameras:
    """Apply render-time pose transforms while keeping traversal/timestamp metadata intact."""
    has_xyz_shift = pose_shift_x != 0.0 or pose_shift_y != 0.0 or pose_shift_z != 0.0
    has_turn = turn_yaw_deg != 0.0
    has_candidate = candidate_trajectory is not None
    if lateral_shift_meters == 0.0 and not has_xyz_shift and not has_turn and not has_candidate:
        return cameras

    poses = cameras.camera_to_worlds.clone()
    origins = poses[..., :3, 3]

    candidate = _prepare_candidate_trajectory(
        candidate_trajectory,
        dtype=origins.dtype,
        device=origins.device,
        prepend_origin=candidate_prepend_origin,
    )
    if candidate is not None:
        _replace_camera_path_with_candidate(
            poses,
            candidate_trajectory=candidate,
            candidate_use_yaw=candidate_use_yaw,
            candidate_scale_to_source_length=candidate_scale_to_source_length,
            candidate_frame_rate=candidate_frame_rate,
            candidate_prepend_origin=candidate_prepend_origin,
            candidate_z_mode=candidate_z_mode,
            target_times_normalized=cameras.times.reshape(-1) if cameras.times is not None else None,
            source_duration_seconds=source_duration_seconds,
        )
        origins = poses[..., :3, 3]

    if has_turn:
        original_origins = origins.clone()
        pivot = _choose_turn_pivot(original_origins, turn_pivot)
        total_angle = torch.deg2rad(
            torch.tensor(turn_yaw_deg, dtype=origins.dtype, device=origins.device)
        )
        if turn_mode == "progressive":
            angles = _trajectory_progress(original_origins) * total_angle
        elif turn_mode == "global":
            angles = torch.ones(origins.shape[0], dtype=origins.dtype, device=origins.device) * total_angle
        else:
            raise ValueError(f"Unknown turn mode: {turn_mode}")

        yaw_rots = _yaw_rotation_matrices(angles)
        rel = origins - pivot
        origins[:, :3] = torch.bmm(yaw_rots, rel[..., None]).squeeze(-1) + pivot
        poses[..., :3, :3] = torch.bmm(yaw_rots, poses[..., :3, :3])

    if lateral_shift_meters != 0.0:
        if origins.shape[0] < 2:
            raise ValueError("--lateral-shift-meters requires at least two camera poses.")
        xy = origins[:, :2]
        tangents = torch.zeros_like(xy)
        tangents[0] = xy[1] - xy[0]
        tangents[-1] = xy[-1] - xy[-2]
        if xy.shape[0] > 2:
            tangents[1:-1] = xy[2:] - xy[:-2]
        tangent_norm = torch.linalg.norm(tangents, dim=-1, keepdim=True).clamp_min(1e-6)
        left_normals = torch.stack((-tangents[:, 1], tangents[:, 0]), dim=-1) / tangent_norm
        origins[:, :2] = origins[:, :2] + left_normals * lateral_shift_meters

    if has_xyz_shift:
        shift = torch.tensor(
            [pose_shift_x, pose_shift_y, pose_shift_z],
            dtype=origins.dtype,
            device=origins.device,
        )
        origins[:, :3] = origins[:, :3] + shift

    cameras.camera_to_worlds = poses
    return cameras


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    obb: OrientedBox = field(default_factory=lambda: OrientedBox(R=torch.eye(3), T=torch.zeros(3), S=torch.ones(3) * 2))
    """Oriented box representing the crop region"""

    # properties for backwards-compatibility interface
    @property
    def center(self):
        return self.obb.T

    @property
    def scale(self):
        return self.obb.S


def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None
    bg_color = camera_json["crop"]["crop_bg_color"]
    center = camera_json["crop"]["crop_center"]
    scale = camera_json["crop"]["crop_scale"]
    rot = (0.0, 0.0, 0.0) if "crop_rot" not in camera_json["crop"] else tuple(camera_json["crop"]["crop_rot"])
    assert len(center) == 3
    assert len(scale) == 3
    assert len(rot) == 3
    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        obb=OrientedBox.from_params(center, rot, scale),
    )


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 2.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""
    multi_view_camera: Tuple[Literal['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0'], ...] = \
        ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0')


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options
        )


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    render_as_travel_id: Optional[int] = None
    """Override model traversal id metadata while keeping the selected camera poses."""
    only_travel_id: Optional[int] = None
    """Render only one traversal id from the selected pose source."""
    lateral_shift_meters: float = 0.0
    """Shift rendered camera positions sideways relative to the trajectory direction, in meters."""
    pose_shift_x: float = 0.0
    """Additional render-time camera translation in MTGS local x, in meters."""
    pose_shift_y: float = 0.0
    """Additional render-time camera translation in MTGS local y, in meters."""
    pose_shift_z: float = 0.0
    """Additional render-time camera translation in MTGS local z, in meters."""
    turn_yaw_deg: float = 0.0
    """Yaw angle for render-time route rotation/bending. Positive values rotate counter-clockwise in local x-y."""
    turn_mode: Literal["progressive", "global"] = "progressive"
    """progressive bends the path gradually; global rotates the whole path rigidly."""
    turn_pivot: Literal["start", "center", "origin"] = "start"
    """Pivot used for turn_yaw_deg."""
    candidate_trajectory_path: Optional[Path] = None
    """Optional .npy file containing trajectory candidates with shape (K, T, 3) or (T, 3)."""
    candidate_index: int = 0
    """Candidate trajectory index to render when candidate_trajectory_path has shape (K, T, 3)."""
    candidate_prepend_origin: bool = True
    """Prepend a zero pose so candidate trajectories start at the current camera pose."""
    candidate_use_yaw: bool = True
    """Use the candidate third column as yaw in radians for camera orientation."""
    candidate_scale_to_source_length: bool = False
    """Scale candidate xy length to match the selected source traversal length."""
    candidate_frame_rate: float = 10.0
    """Frame rate of candidate trajectories in candidate_trajectory_path."""
    candidate_trim_to_horizon: bool = True
    """Trim rendered source cameras to the candidate trajectory horizon."""
    candidate_z_mode: Literal["distance", "time", "constant"] = "distance"
    """How to assign camera z after candidate xy replacement."""
    output_path: Optional[Path] = None
    """Path to output video for multi-traversal"""
    interpolation_steps: int = 6
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 60
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        def update_config(config: TrainerConfig) -> TrainerConfig:
            config.pipeline.datamanager.dataparser.train_split_fraction = 1.0
            config.pipeline.datamanager.dataparser.cameras = self.multi_view_camera
            CONSOLE.print(f'Using cams {self.multi_view_camera}.', style="bold green")
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback = update_config
        )

        install_checks.check_ffmpeg_installed()

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
            dataparser_outputs = pipeline.datamanager.eval_dataset._dataparser_outputs
        else:
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras
            dataparser_outputs = pipeline.datamanager.train_dataset._dataparser_outputs

        num_cameras = len(self.multi_view_camera)

        candidate_trajectory = None
        candidate_duration_seconds = None
        if self.candidate_trajectory_path is not None:
            candidate_file = Path(self.candidate_trajectory_path).expanduser()
            candidate_array = np.load(candidate_file)
            if candidate_array.ndim == 3:
                if self.candidate_index < 0 or self.candidate_index >= candidate_array.shape[0]:
                    raise ValueError(
                        f"--candidate-index {self.candidate_index} is out of range for "
                        f"{candidate_file}; valid range is [0, {candidate_array.shape[0] - 1}]"
                    )
                candidate_array = candidate_array[self.candidate_index]
            elif candidate_array.ndim != 2:
                raise ValueError(
                    f"{candidate_file} must have shape (K, T, 3) or (T, 3), got {candidate_array.shape}"
                )
            if candidate_array.shape[-1] < 2:
                raise ValueError(f"{candidate_file} candidate must have at least x/y columns, got {candidate_array.shape}")
            candidate_trajectory = torch.from_numpy(np.asarray(candidate_array, dtype=np.float32))
            candidate_duration_seconds = _candidate_duration_seconds(
                candidate_trajectory,
                frame_rate=self.candidate_frame_rate,
                prepend_origin=self.candidate_prepend_origin,
            )
            CONSOLE.print(
                f"Using candidate trajectory {self.candidate_index} from {candidate_file} "
                f"with shape {tuple(candidate_trajectory.shape)}.",
                style="bold green",
            )

        def render_travel(output_path, cameras, num_cameras, travel_id=None):
            source_duration_seconds = _get_travel_duration_seconds(dataparser_outputs, travel_id)
            if (
                candidate_trajectory is not None
                and self.candidate_trim_to_horizon
                and source_duration_seconds is not None
                and candidate_duration_seconds is not None
            ):
                max_normalized_time = min(candidate_duration_seconds / source_duration_seconds, 1.0)
                cameras = _trim_cameras_to_normalized_time(cameras, num_cameras, max_normalized_time)
            cam_split = []
            for i in range(num_cameras):
                cam_split.append(cameras[i::num_cameras])

            if candidate_trajectory is not None and len(cam_split) > 0:
                representative_camera = self.multi_view_camera[0]
                original_camera_path = _get_interpolated_camera_path(
                    cameras=cam_split[0],
                    steps=self.interpolation_steps,
                    order_poses=self.order_poses,
                )
                replaced_camera_path = _get_interpolated_camera_path(
                    cameras=cam_split[0],
                    steps=self.interpolation_steps,
                    order_poses=self.order_poses,
                )
                replaced_camera_path = _apply_view_exploration_offsets(
                    replaced_camera_path,
                    lateral_shift_meters=self.lateral_shift_meters,
                    pose_shift_x=self.pose_shift_x,
                    pose_shift_y=self.pose_shift_y,
                    pose_shift_z=self.pose_shift_z,
                    turn_yaw_deg=self.turn_yaw_deg,
                    turn_mode=self.turn_mode,
                    turn_pivot=self.turn_pivot,
                    candidate_trajectory=candidate_trajectory,
                    candidate_prepend_origin=self.candidate_prepend_origin,
                    candidate_use_yaw=self.candidate_use_yaw,
                    candidate_scale_to_source_length=self.candidate_scale_to_source_length,
                    candidate_frame_rate=self.candidate_frame_rate,
                    candidate_z_mode=self.candidate_z_mode,
                    source_duration_seconds=source_duration_seconds,
                )
                _plot_trajectory_comparison(
                    original_cameras=original_camera_path,
                    replaced_cameras=replaced_camera_path,
                    output_path=output_path / f"trajectory_2d_map_candidate_{self.candidate_index}.png",
                    travel_id=travel_id,
                    candidate_index=self.candidate_index,
                    representative_camera=representative_camera,
                    scene_box=getattr(dataparser_outputs, "scene_box", None),
                )

            rendered = {}
            for cam_id, cameras in enumerate(cam_split):
                cam_name = self.multi_view_camera[cam_id]
                seconds = (self.interpolation_steps * (len(cameras) - 1) + 1) / self.frame_rate
                camera_path = _get_interpolated_camera_path(
                    cameras=cameras,
                    steps=self.interpolation_steps,
                    order_poses=self.order_poses
                )
                camera_path = _apply_view_exploration_offsets(
                    camera_path,
                    lateral_shift_meters=self.lateral_shift_meters,
                    pose_shift_x=self.pose_shift_x,
                    pose_shift_y=self.pose_shift_y,
                    pose_shift_z=self.pose_shift_z,
                    turn_yaw_deg=self.turn_yaw_deg,
                    turn_mode=self.turn_mode,
                    turn_pivot=self.turn_pivot,
                    candidate_trajectory=candidate_trajectory,
                    candidate_prepend_origin=self.candidate_prepend_origin,
                    candidate_use_yaw=self.candidate_use_yaw,
                    candidate_scale_to_source_length=self.candidate_scale_to_source_length,
                    candidate_frame_rate=self.candidate_frame_rate,
                    candidate_z_mode=self.candidate_z_mode,
                    source_duration_seconds=source_duration_seconds,
                )
                # The camera path uses poses from travel_id. The model traversal identity can
                # be overridden to isolate pose changes from unseen traversal-id effects.
                render_travel_id = self.render_as_travel_id if self.render_as_travel_id is not None else travel_id
                camera_path.metadata = {'travel_id': render_travel_id}
                assert self.output_format == "video"
                output_filename = output_path / f"{cam_name}.mp4"

                images = _render_trajectory_video(
                    pipeline,
                    camera_path,
                    output_filename=Path(output_filename),
                    rendered_output_names=self.rendered_output_names,
                    rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                    seconds=seconds,
                    output_format=self.output_format,
                    image_format=self.image_format,
                    depth_near_plane=self.depth_near_plane,
                    depth_far_plane=self.depth_far_plane,
                    colormap_options=self.colormap_options
                )
                rendered[cam_name] = images

            # concat and save videos
            videos = []
            for cam in ('CAM_L0', 'CAM_F0', 'CAM_R0'):
                video = rendered[cam]
                videos.append(video)
            videos = np.concatenate(videos, axis=-2)
            media.write_video(output_path / "concat_front.mp4", videos, fps=self.frame_rate)

            videos = []
            for cam in ('CAM_R2', 'CAM_B0', 'CAM_L2'):
                video = rendered[cam]
                videos.append(video)
            videos = np.concatenate(videos, axis=-2)
            media.write_video(output_path / "concat_back.mp4", videos, fps=self.frame_rate)

        base_output_path = Path()
        if self.output_path is not None:
            base_output_path = self.output_path
        else:
            if hasattr(config, "base_dir"):
                base_output_path = Path(f"renders/scene_videos/{os.path.basename(config.base_dir)}")
            else:
                base_output_path = Path(f"renders/{config.experiment_name}")

        if hasattr(dataparser_outputs, "travel_ids") and dataparser_outputs.travel_ids is not None:
            travel_ids = dataparser_outputs.travel_ids
            travel_id_set = list(set(travel_ids))
            cameras_travels = {k:[] for k in travel_id_set}
            for idx, travel_id in enumerate(travel_ids):
                cameras_travels[travel_id].append(idx)
            if self.only_travel_id is not None:
                if self.only_travel_id not in cameras_travels:
                    raise ValueError(
                        f"--only-travel-id {self.only_travel_id} is not available in "
                        f"pose_source={self.pose_source}. Available ids: {sorted(cameras_travels.keys())}"
                    )
                travel_id_set = [self.only_travel_id]
            for travel_id in travel_id_set:
                output_path = base_output_path / f"travel_{travel_id}"
                travel_indices = torch.tensor(cameras_travels[travel_id], dtype=torch.int64)
                render_travel(output_path, cameras[travel_indices], num_cameras, travel_id)
        else:
            render_travel(base_output_path, cameras, num_cameras)

@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "val", "test", "train+test"] = "test"
    """Split to render."""
    rendered_output_names: Optional[List[str]] = field(default_factory=lambda: None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""

    def main(self):
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))

        for split in self.split.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(camera)

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if rendered_output_names is None:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1
                        image_name = f"{camera_idx:05d}"

                        # Try to get the original filename
                        image_name = dataparser_outputs.image_filenames[camera_idx].relative_to(images_root)

                        output_path = self.output_path / split / rendered_output_name / image_name
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        if is_raw:
                            with gzip.open(output_path.with_suffix(".npy.gz"), "wb") as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality
                            )
                        else:
                            raise ValueError(f"Unknown image format {self.image_format}")

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))

# command to sub task
Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[DatasetRender, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
