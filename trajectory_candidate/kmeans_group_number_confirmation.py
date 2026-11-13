#!/usr/bin/env python3
"""Select a coarse KMeans group count using compact history15 descriptors.

The trajectory segment is constructed as:

    origin at 0.0 s + first 15 points from trajectory_candidate/human_trajectory_candidate.npy

With a 10 Hz candidate frame rate, this covers 0.0 s to 1.5 s.  The default
compact feature vector is:

    [x_progress, y_progress, deviation_ratio, yaw_change, mean_abs_curvature]

The features are z-standardized before KMeans.  This script only performs coarse
K selection with elbow inertia and silhouette score; it does not produce
sub-clusters or final HGS artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


DEFAULT_INPUT_PATH = Path(__file__).parent / "human_trajectory_candidate.npy"
DEFAULT_OUT_DIR = Path(__file__).parent / "group_number_confirmation"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def prepend_origin_segment(trajectories: np.ndarray, num_future_points: int) -> np.ndarray:
    if trajectories.ndim != 3 or trajectories.shape[2] < 3:
        raise ValueError(f"Expected input shape (N,T,3+), got {trajectories.shape}")
    if num_future_points <= 0 or num_future_points > trajectories.shape[1]:
        raise ValueError(f"--num-future-points={num_future_points} is invalid for horizon {trajectories.shape[1]}")
    prefix = np.asarray(trajectories[:, :num_future_points, :3], dtype=np.float64)
    origin = np.zeros((prefix.shape[0], 1, prefix.shape[2]), dtype=np.float64)
    return np.concatenate([origin, prefix], axis=1)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def standardize_features(raw_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw_features.mean(axis=0, keepdims=True)
    std = raw_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (raw_features - mean) / std, mean.reshape(-1), std.reshape(-1)


def compute_descriptor_features(
    segment: np.ndarray,
    frame_rate_hz: float,
    turn_feature: str,
    deviation_clip_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute compact descriptors from origin + first candidate points."""
    xy = segment[:, :, :2]
    yaw = np.unwrap(segment[:, :, 2], axis=1)
    start_xy = xy[:, 0, :]
    end_xy = xy[:, -1, :]
    displacement = end_xy - start_xy
    x_progress = displacement[:, 0]
    y_progress = displacement[:, 1]

    step_xy = np.diff(xy, axis=1)
    step_dist = np.linalg.norm(step_xy, axis=2)
    path_length = step_dist.sum(axis=1)
    straight_len = np.linalg.norm(displacement, axis=1)
    safe_straight_len = np.where(straight_len < 1e-6, 1e-6, straight_len)
    safe_path_length = np.where(path_length < 1e-6, 1e-6, path_length)

    rel = xy - start_xy[:, None, :]
    dir_unit = displacement / safe_straight_len[:, None]
    lateral = np.abs(rel[:, :, 0] * dir_unit[:, 1:2] - rel[:, :, 1] * dir_unit[:, 0:1])
    max_lateral = lateral.max(axis=1)
    deviation_ratio = max_lateral / safe_straight_len
    clip_upper = float(np.percentile(deviation_ratio, float(deviation_clip_percentile)))
    deviation_ratio = np.clip(deviation_ratio, 0.0, clip_upper)

    yaw_change = angle_diff(yaw[:, -1], yaw[:, 0])
    yaw_steps = angle_diff(yaw[:, 1:], yaw[:, :-1])
    total_abs_yaw_change = np.abs(yaw_steps).sum(axis=1)
    duration_s = float(segment.shape[1] - 1) / float(frame_rate_hz)
    duration_s = max(duration_s, 1e-6)

    if turn_feature == "mean_abs_curvature":
        turn_values = total_abs_yaw_change / safe_path_length
        turn_definition = "sum(abs(delta_yaw)) / path_length over origin + first candidate points"
    elif turn_feature == "signed_mean_curvature":
        turn_values = yaw_change / safe_path_length
        turn_definition = "signed yaw_change / path_length over origin + first candidate points"
    elif turn_feature == "mean_yaw_rate":
        turn_values = yaw_change / duration_s
        turn_definition = "signed yaw_change / duration_s over origin + first candidate points"
    elif turn_feature == "mean_abs_yaw_rate":
        turn_values = total_abs_yaw_change / duration_s
        turn_definition = "sum(abs(delta_yaw)) / duration_s over origin + first candidate points"
    else:
        raise ValueError(f"Unsupported turn feature: {turn_feature}")

    raw_features = np.stack(
        [x_progress, y_progress, deviation_ratio, yaw_change, turn_values],
        axis=1,
    ).astype(np.float64)
    features, mean, std = standardize_features(raw_features)
    feature_names = ["x_progress", "y_progress", "deviation_ratio", "yaw_change", turn_feature]
    info = {
        "feature_family": "history15_descriptors",
        "segment_definition": "origin at 0.0s + first 15 candidate points",
        "candidate_frame_rate_hz": float(frame_rate_hz),
        "num_future_points": int(segment.shape[1] - 1),
        "segment_shape": list(segment.shape),
        "feature_names": feature_names,
        "feature_shape": list(raw_features.shape),
        "standardization": "zscore over all candidates per feature dimension",
        "feature_mean": [float(x) for x in mean],
        "feature_std": [float(x) for x in std],
        "definitions": {
            "x_progress": "end_x - start_x on origin + first candidate points",
            "y_progress": "end_y - start_y on origin + first candidate points",
            "deviation_ratio": (
                "max perpendicular xy distance from the start-end line divided by start-end "
                f"distance, clipped at the {float(deviation_clip_percentile):g}th percentile"
            ),
            "yaw_change": "wrapped final_yaw - initial_yaw",
            turn_feature: turn_definition,
        },
    }
    return features, info


def parse_k_values(args: argparse.Namespace) -> list[int]:
    if args.k_values is not None:
        values = [int(k) for k in args.k_values]
    else:
        values = list(range(int(args.k_min), int(args.k_max) + 1, int(args.k_step)))
    values = sorted(set(values))
    if not values:
        raise ValueError("No K values were requested.")
    if values[0] < 2:
        raise ValueError("KMeans model selection needs K >= 2 for silhouette score.")
    return values


def select_representative_indices(features: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    representative_indices = np.full(centers.shape[0], -1, dtype=np.int64)
    for group in range(centers.shape[0]):
        indices = np.flatnonzero(labels == group)
        if len(indices) == 0:
            raise RuntimeError(f"KMeans produced an empty cluster: {group}")
        distances = np.linalg.norm(features[indices] - centers[group], axis=1)
        representative_indices[group] = int(indices[int(np.argmin(distances))])
    return representative_indices


def min_representative_trajectory_distance(segment: np.ndarray, representative_indices: np.ndarray) -> float:
    representatives = np.asarray(segment[representative_indices, :, :2], dtype=np.float64)
    if len(representatives) < 2:
        return 0.0
    best = np.inf
    for i in range(len(representatives)):
        diff = representatives[i + 1 :] - representatives[i]
        if len(diff) == 0:
            continue
        distances = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
        best = min(best, float(distances.min()))
    return float(best)


def run_kmeans_sweep(
    features: np.ndarray,
    segment: np.ndarray,
    k_values: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(k_values)
    for idx, k in enumerate(k_values, start=1):
        print(f"[{idx}/{total}] KMeans K={k} ...", flush=True)
        kmeans = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=int(args.n_init),
            random_state=int(args.random_state),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            verbose=0,
        )
        labels = kmeans.fit_predict(features)
        counts = np.bincount(labels, minlength=int(k))
        representative_indices = select_representative_indices(features, labels, kmeans.cluster_centers_)
        row: dict[str, Any] = {
            "k": int(k),
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "min_group_size": int(counts.min()),
            "max_group_size": int(counts.max()),
            "calinski_harabasz_index": float(calinski_harabasz_score(features, labels)),
            "davies_bouldin_index": float(davies_bouldin_score(features, labels)),
            "min_representative_trajectory_distance": min_representative_trajectory_distance(
                segment, representative_indices
            ),
        }
        try:
            sample_size = min(int(args.silhouette_sample_size), len(labels))
            if sample_size >= len(labels):
                silhouette = silhouette_score(features, labels, metric="euclidean")
                row["silhouette_sample_size"] = int(len(labels))
            else:
                silhouette = silhouette_score(
                    features,
                    labels,
                    metric="euclidean",
                    sample_size=sample_size,
                    random_state=int(args.random_state),
                )
                row["silhouette_sample_size"] = int(sample_size)
            row["silhouette_score"] = float(silhouette)
        except Exception as exc:  # noqa: BLE001
            row["silhouette_score"] = None
            row["silhouette_error"] = str(exc)
        rows.append(row)
        score_text = "nan" if row["silhouette_score"] is None else f"{row['silhouette_score']:.4f}"
        print(
            f"  inertia={row['inertia']:.2f}, silhouette={score_text}, "
            f"CH={row['calinski_harabasz_index']:.2f}, "
            f"DB={row['davies_bouldin_index']:.3f}, "
            f"min_rep_traj={row['min_representative_trajectory_distance']:.3f}",
            flush=True,
        )
    return rows


def finite_scores(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        xs.append(int(row["k"]))
        ys.append(float(value))
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def elbow_k_by_line_distance(rows: list[dict[str, Any]]) -> int | None:
    k_values, inertia = finite_scores(rows, "inertia")
    if len(k_values) < 3:
        return None
    x = (k_values.astype(np.float64) - float(k_values.min())) / max(float(k_values.max() - k_values.min()), 1e-12)
    y = (inertia - float(inertia.min())) / max(float(inertia.max() - inertia.min()), 1e-12)
    points = np.stack([x, y], axis=1)
    start = points[0]
    line = points[-1] - start
    denom = max(float(np.linalg.norm(line)), 1e-12)
    rel = points - start
    distances = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / denom
    return int(k_values[int(np.argmax(distances))])


def best_silhouette_k(rows: list[dict[str, Any]]) -> int | None:
    valid = [row for row in rows if row.get("silhouette_score") is not None]
    if not valid:
        return None
    return int(max(valid, key=lambda row: float(row["silhouette_score"]))["k"])



def plot_metric_overview(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    metric_specs = [
        ("inertia", "Inertia", "lower is better", "blue"),
        ("min_representative_trajectory_distance", "Representative Trajectory Distance [m]", "higher is better", "crimson"),
        ("calinski_harabasz_index", "Calinski-Harabasz Index", "higher is better", "darkorange"),
        ("davies_bouldin_index", "Davies-Bouldin Index", "lower is better", "teal"),
        ("silhouette_score", "Silhouette Score", "higher is better", "green"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 9.0), dpi=220, squeeze=False)
    axes_flat = axes.reshape(-1)
    for ax, (key, ylabel, subtitle, color) in zip(axes_flat, metric_specs):
        k_values, values = finite_scores(rows, key)
        ax.plot(k_values, values, marker="o", linewidth=1.7, markersize=4.0, color=color)
        ax.set_xlabel("Number of Clusters k")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=9.5)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        if len(k_values) <= 15:
            tick_values = k_values
        else:
            tick_step = int(np.ceil(len(k_values) / 15.0))
            tick_values = k_values[::tick_step]
            if tick_values[-1] != k_values[-1]:
                tick_values = np.append(tick_values, k_values[-1])
        ax.set_xticks(tick_values)
    axes_flat[-1].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clustering metrics figure -> {out_path}")



def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics CSV -> {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(f"Saved summary JSON -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Trajectory candidate .npy file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-future-points", type=int, default=15)
    parser.add_argument("--candidate-frame-rate", type=float, default=10.0)
    parser.add_argument(
        "--turn-feature",
        choices=["mean_abs_curvature", "signed_mean_curvature", "mean_yaw_rate", "mean_abs_yaw_rate"],
        default="mean_abs_curvature",
        help="The fifth compact descriptor after x/y progress, deviation ratio, and yaw change.",
    )
    parser.add_argument("--deviation-clip-percentile", type=float, default=99.0)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument("--k-step", type=int, default=1)
    parser.add_argument("--k-values", type=int, nargs="+", default=None, help="Explicit K values; overrides k-min/k-max/k-step.")
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--silhouette-sample-size", type=int, default=4000)
    parser.add_argument("--metrics-figure-name", type=str, default="kmeans_history15_descriptors_final_metrics.png")
    parser.add_argument("--summary-name", type=str, default="kmeans_history15_descriptors_auto_k_summary.json")
    parser.add_argument("--csv-name", type=str, default="kmeans_history15_descriptors_auto_k_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_step <= 0:
        raise ValueError("--k-step must be positive.")
    if args.silhouette_sample_size <= 0:
        raise ValueError("--silhouette-sample-size must be positive.")
    if not (0.0 < args.deviation_clip_percentile <= 100.0):
        raise ValueError("--deviation-clip-percentile must be in (0, 100].")

    k_values = parse_k_values(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory candidates from {args.input} ...")
    trajectories = np.load(args.input).astype(np.float32)
    print(f"Loaded candidate bank: {trajectories.shape}")

    segment = prepend_origin_segment(trajectories, int(args.num_future_points))
    features, feature_info = compute_descriptor_features(
        segment=segment,
        frame_rate_hz=float(args.candidate_frame_rate),
        turn_feature=args.turn_feature,
        deviation_clip_percentile=float(args.deviation_clip_percentile),
    )
    print(f"Built standardized descriptor feature matrix: {features.shape}")
    print(f"Feature names: {', '.join(feature_info['feature_names'])}")

    rows = run_kmeans_sweep(features=features, segment=segment, k_values=k_values, args=args)
    elbow_k = elbow_k_by_line_distance(rows)
    best_k = best_silhouette_k(rows)

    csv_path = args.out_dir / args.csv_name
    json_path = args.out_dir / args.summary_name
    metrics_figure_path = args.out_dir / args.metrics_figure_name
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "schema_version": "trajectory_candidate_history15_descriptor_kmeans_auto_k_v1",
            "purpose": "Coarse KMeans group-count selection using compact descriptors over origin + first 15 points.",
            "input_path": str(args.input),
            "input_shape": list(trajectories.shape),
            "descriptor_feature": feature_info,
            "k_values": k_values,
            "kmeans": {
                "n_init": int(args.n_init),
                "max_iter": int(args.max_iter),
                "tol": float(args.tol),
                "random_state": int(args.random_state),
            },
            "silhouette_sample_size": int(args.silhouette_sample_size),
            "metric_definitions": {
                "inertia": "KMeans within-cluster sum of squared distances in standardized descriptor space; lower is better.",
                "silhouette_score": "Sampled silhouette score in standardized descriptor space; higher is better.",
                "calinski_harabasz_index": "Between-cluster dispersion divided by within-cluster dispersion in standardized descriptor space; higher is better.",
                "davies_bouldin_index": "Average cluster similarity based on within-cluster scatter and between-centroid separation in standardized descriptor space; lower is better.",
                "min_representative_trajectory_distance": "Minimum pairwise RMS XY distance in meters between representative origin+first15 trajectories; higher is better.",
            },
            "elbow_k": elbow_k,
            "best_silhouette_k": best_k,
            "metrics": rows,
        },
    )
    plot_metric_overview(rows, metrics_figure_path, title="KMeans Final Selection Metrics")
    if elbow_k is not None:
        print(f"Estimated elbow K -> {elbow_k}")
    if best_k is not None:
        print(f"Best sampled silhouette K -> {best_k}")


if __name__ == "__main__":
    main()
