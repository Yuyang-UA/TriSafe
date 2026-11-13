#!/usr/bin/env python3
"""Build final history15 descriptor coarse/sub trajectory groups.

This script uses the selected compact descriptor feature over:

    origin at 0.0 s + first 15 points from trajectory_candidate/human_trajectory_candidate.npy

Default feature vector:

    [x_progress, y_progress, deviation_ratio, yaw_change, mean_abs_curvature]

It writes HGS-compatible coarse labels, coarse representatives, global sub-labels,
sub representatives, an info JSON, and two figures:

    1. coarse-group distribution
    2. representative trajectory instance per coarse group
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from kmeans_group_number_confirmation import (
    DEFAULT_INPUT_PATH,
    angle_diff,
    compute_descriptor_features,
    prepend_origin_segment,
    str_to_bool,
)


DEFAULT_OUT_DIR = Path(__file__).parent / "dictionary_structure"


def group_counts(labels: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_groups)[:n_groups]


def expand_sub_group_counts(values: Sequence[int], n_groups: int) -> list[int]:
    counts = [int(value) for value in values]
    if len(counts) == 1:
        counts = counts * int(n_groups)
    if len(counts) != int(n_groups):
        raise ValueError(f"Expected one sub-group count or {n_groups} per-group counts, got {counts}.")
    if any(count <= 0 for count in counts):
        raise ValueError(f"All sub-group counts must be positive, got {counts}.")
    return counts


def sub_group_suffix(counts: Sequence[int]) -> str:
    return "_".join(str(int(count)) for count in counts)


def sub_group_offsets(counts: Sequence[int]) -> list[int]:
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + int(count))
    return offsets


def group_centers(features: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    centers = np.empty((n_groups, features.shape[1]), dtype=np.float64)
    for group in range(n_groups):
        mask = labels == group
        if not np.any(mask):
            raise RuntimeError(f"Empty coarse group {group}.")
        centers[group] = features[mask].mean(axis=0)
    return centers


def select_representative_indices(features: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    representative_indices = np.full(centers.shape[0], -1, dtype=np.int64)
    for group in range(centers.shape[0]):
        indices = np.flatnonzero(labels == group)
        distances = np.linalg.norm(features[indices] - centers[group], axis=1)
        representative_indices[group] = int(indices[int(np.argmin(distances))])
    return representative_indices


def reorder_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    segment: np.ndarray,
    order: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    old_ids = list(range(centers.shape[0]))
    if order == "none":
        ordered = old_ids
    elif order == "size":
        counts = group_counts(labels, centers.shape[0])
        ordered = sorted(old_ids, key=lambda group: (-int(counts[group]), int(group)))
    elif order == "endpoint":
        keys = {}
        for group in old_ids:
            endpoint = segment[labels == group, -1, :3].mean(axis=0)
            keys[group] = (float(endpoint[1]), float(endpoint[0]), float(endpoint[2]), int(group))
        ordered = sorted(old_ids, key=lambda group: keys[group])
    else:
        raise ValueError(f"Unsupported label order: {order}")

    old_to_new = {int(old): int(new) for new, old in enumerate(ordered)}
    new_labels = np.asarray([old_to_new[int(label)] for label in labels], dtype=np.int64)
    new_centers = np.stack([centers[old] for old in ordered], axis=0)
    return new_labels, new_centers, old_to_new


def subgroup_kmeans(
    trajectories: np.ndarray,
    features: np.ndarray,
    coarse_labels: np.ndarray,
    n_groups: int,
    sub_groups_by_group: Sequence[int],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, list[int]]]:
    n_samples = trajectories.shape[0]
    requested_counts = expand_sub_group_counts(sub_groups_by_group, n_groups)
    offsets = sub_group_offsets(requested_counts)
    total_sub_groups = int(sum(requested_counts))
    sub_labels_global = np.full(n_samples, -1, dtype=np.int64)
    sub_representatives = np.zeros((total_sub_groups, *trajectories.shape[1:]), dtype=trajectories.dtype)
    sub_infos: list[dict[str, Any]] = []
    coarse_to_global_subs: dict[str, list[int]] = {}

    for group in range(n_groups):
        member_indices = np.flatnonzero(coarse_labels == group)
        n_members = len(member_indices)
        if n_members == 0:
            raise RuntimeError(f"Cannot subgroup empty coarse group {group}.")
        requested_sub = int(requested_counts[group])
        effective_sub = min(requested_sub, n_members)
        group_features = features[member_indices]

        if effective_sub == 1:
            local_labels = np.zeros(n_members, dtype=np.int64)
            centers = group_features.mean(axis=0, keepdims=True)
            kmeans_info = {"n_iter": 0, "inertia": 0.0}
        else:
            kmeans = KMeans(
                n_clusters=effective_sub,
                init="k-means++",
                n_init="auto",
                random_state=int(random_state) + int(group),
                max_iter=500,
            )
            local_labels = kmeans.fit_predict(group_features)
            centers = kmeans.cluster_centers_
            kmeans_info = {"n_iter": int(kmeans.n_iter_), "inertia": float(kmeans.inertia_)}

        start = int(offsets[group])
        global_ids = list(range(start, start + requested_sub))
        coarse_to_global_subs[str(group)] = global_ids
        representative_indices: list[int] = []
        for local_sub in range(effective_sub):
            local_positions = np.flatnonzero(local_labels == local_sub)
            distances = np.linalg.norm(group_features[local_positions] - centers[local_sub], axis=1)
            best_local_position = int(local_positions[int(np.argmin(distances))])
            rep_idx = int(member_indices[best_local_position])
            global_sub_id = start + local_sub
            representative_indices.append(rep_idx)
            sub_representatives[global_sub_id] = trajectories[rep_idx]
            sub_labels_global[member_indices[local_labels == local_sub]] = global_sub_id

        for pad_idx in range(effective_sub, requested_sub):
            sub_representatives[start + pad_idx] = sub_representatives[start + effective_sub - 1]

        sub_infos.append(
            {
                "coarse_group": int(group),
                "n_members": int(n_members),
                "n_sub_groups": int(effective_sub),
                "requested_sub_groups": int(requested_sub),
                "global_sub_group_ids": global_ids,
                "sub_group_counts": [int((local_labels == sub).sum()) for sub in range(effective_sub)],
                "representative_indices": representative_indices,
                "kmeans": kmeans_info,
                "empty": False,
            }
        )

    if np.any(sub_labels_global < 0):
        raise RuntimeError("Sub-clustering left candidates without a sub-label.")
    return sub_labels_global, sub_representatives, sub_infos, coarse_to_global_subs


def raw_feature_matrix(features: np.ndarray, feature_info: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(feature_info["feature_mean"], dtype=np.float64)
    std = np.asarray(feature_info["feature_std"], dtype=np.float64)
    return features * std[None, :] + mean[None, :]


def group_feature_stats(raw_features: np.ndarray, labels: np.ndarray, feature_names: list[str], n_groups: int) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for group in range(n_groups):
        mask = labels == group
        group_stats: dict[str, Any] = {}
        for dim, name in enumerate(feature_names):
            values = raw_features[mask, dim]
            group_stats[name] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        stats.append(group_stats)
    return stats


def min_representative_trajectory_distance(segment: np.ndarray, representative_indices: np.ndarray) -> float:
    reps = segment[representative_indices, :, :2]
    best = np.inf
    for i in range(len(reps)):
        diff = reps[i + 1 :] - reps[i]
        if len(diff) == 0:
            continue
        distances = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
        best = min(best, float(distances.min()))
    return float(best)


def clustering_quality(features: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "calinski_harabasz_index": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(features, labels)),
    }
    sample_size = min(int(sample_size), len(labels))
    if sample_size >= len(labels):
        out["silhouette_score"] = float(silhouette_score(features, labels))
        out["silhouette_sample_size"] = int(len(labels))
    else:
        out["silhouette_score"] = float(
            silhouette_score(features, labels, sample_size=sample_size, random_state=int(random_state))
        )
        out["silhouette_sample_size"] = int(sample_size)
    return out


def plot_distribution(labels: np.ndarray, n_groups: int, out_path: Path, title: str) -> None:
    counts = group_counts(labels, n_groups)
    total = int(counts.sum())
    cmap = plt.colormaps["tab10"].resampled(n_groups)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=220)
    x = np.arange(n_groups)
    bars = ax.bar(x, counts, color=[cmap(i) for i in range(n_groups)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_groups)])
    ax.set_xlabel("Coarse group")
    ax.set_ylabel("Number of candidates")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    for bar, count in zip(bars, counts):
        pct = 100.0 * float(count) / max(total, 1)
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{int(count)}\n{pct:.1f}%", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution figure -> {out_path}")


def plot_descriptor_space(
    raw_features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    representative_indices: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    n_groups = int(labels.max()) + 1
    cmap = plt.colormaps["tab10"].resampled(n_groups)
    feature_lookup = {name: idx for idx, name in enumerate(feature_names)}
    pairs = [
        ("x_progress", "y_progress"),
        ("x_progress", "deviation_ratio"),
        ("y_progress", "yaw_change"),
        ("yaw_change", feature_names[-1]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), dpi=220, squeeze=False)
    axes_flat = axes.reshape(-1)
    for ax, (x_name, y_name) in zip(axes_flat, pairs):
        x_idx = feature_lookup[x_name]
        y_idx = feature_lookup[y_name]
        for group in range(n_groups):
            mask = labels == group
            ax.scatter(
                raw_features[mask, x_idx],
                raw_features[mask, y_idx],
                s=5,
                alpha=0.28,
                linewidths=0,
                color=cmap(group),
            )
        ax.scatter(
            raw_features[representative_indices, x_idx],
            raw_features[representative_indices, y_idx],
            c="black",
            marker="x",
            s=58,
            linewidths=1.6,
            zorder=5,
        )
        for group, rep_idx in enumerate(representative_indices):
            ax.text(
                raw_features[rep_idx, x_idx],
                raw_features[rep_idx, y_idx],
                str(group),
                fontsize=7.5,
                ha="left",
                va="bottom",
                color="black",
                zorder=6,
            )
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved descriptor-space figure -> {out_path}")


def plot_representatives(
    representative_segments: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot only origin + first candidate points for each coarse representative."""
    n_groups = representative_segments.shape[0]
    counts = group_counts(labels, n_groups)
    cols = 3
    rows = int(np.ceil(n_groups / cols))
    cmap = plt.colormaps["tab10"].resampled(n_groups)

    all_xy = representative_segments[:, :, :2].reshape(-1, 2)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = max(0.8, 0.12 * float(max(x_max - x_min, 1e-6)))
    y_pad = max(0.8, 0.18 * float(max(y_max - y_min, 1e-6)))

    fig, axes = plt.subplots(rows, cols, figsize=(3.65 * cols, 3.2 * rows), dpi=220, squeeze=False)
    fig.suptitle(title, fontsize=13, y=0.995)
    for group in range(n_groups):
        ax = axes[group // cols, group % cols]
        color = cmap(group)
        segment = representative_segments[group]
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2.0, zorder=2)
        ax.scatter(segment[:, 0], segment[:, 1], s=22, facecolor="white", edgecolor=color, linewidth=0.9, zorder=4)
        ax.scatter(segment[0, 0], segment[0, 1], s=30, color="black", marker="o", zorder=5)
        ax.scatter(segment[-1, 0], segment[-1, 1], s=34, color=color, edgecolor="black", marker="X", linewidth=0.5, zorder=5)
        if segment.shape[1] >= 3:
            arrow_len = max(0.25, 0.04 * float(max(x_max - x_min, y_max - y_min, 1.0)))
            step = max(1, segment.shape[0] // 5)
            points = segment[::step]
            ax.quiver(
                points[:, 0],
                points[:, 1],
                np.cos(points[:, 2]) * arrow_len,
                np.sin(points[:, 2]) * arrow_len,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.004,
                alpha=0.75,
                zorder=3,
            )
        ax.set_title(f"Group {group}  n={int(counts[group])}", fontsize=9.4)
        ax.set_xlabel("x [m]", fontsize=8.5)
        ax.set_ylabel("y [m]", fontsize=8.5)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
    for idx in range(n_groups, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.tight_layout(pad=1.15, w_pad=1.0, h_pad=1.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved representative figure -> {out_path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-groups", type=int, default=9)
    parser.add_argument("--sub-groups", type=int, nargs="+", default=[60, 40, 40, 20, 20, 20, 20, 20, 10])
    parser.add_argument("--num-future-points", type=int, default=15)
    parser.add_argument("--candidate-frame-rate", type=float, default=10.0)
    parser.add_argument(
        "--turn-feature",
        choices=["mean_abs_curvature", "signed_mean_curvature", "mean_yaw_rate", "mean_abs_yaw_rate"],
        default="mean_abs_curvature",
    )
    parser.add_argument("--deviation-clip-percentile", type=float, default=99.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sub-group-seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--silhouette-sample-size", type=int, default=4000)
    parser.add_argument("--label-order", choices=["size", "endpoint", "none"], default="size")
    parser.add_argument("--overwrite", type=str_to_bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_groups <= 1:
        raise ValueError("--n-groups must be greater than 1.")
    sub_groups_by_group = expand_sub_group_counts(args.sub_groups, int(args.n_groups))
    total_sub_groups = int(sum(sub_groups_by_group))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory candidates from {args.input} ...")
    trajectories = np.load(args.input).astype(np.float32)
    if trajectories.ndim != 3 or trajectories.shape[2] < 3:
        raise ValueError(f"Expected input shape (N,T,3+), got {trajectories.shape}")
    print(f"Loaded candidate bank: {trajectories.shape}")

    segment = prepend_origin_segment(trajectories, int(args.num_future_points))
    features, feature_info = compute_descriptor_features(
        segment=segment,
        frame_rate_hz=float(args.candidate_frame_rate),
        turn_feature=args.turn_feature,
        deviation_clip_percentile=float(args.deviation_clip_percentile),
    )
    raw_features = raw_feature_matrix(features, feature_info)
    print(f"Built standardized descriptor feature matrix: {features.shape}")

    kmeans = KMeans(
        n_clusters=int(args.n_groups),
        init="k-means++",
        n_init=int(args.n_init),
        random_state=int(args.random_state),
        max_iter=int(args.max_iter),
        verbose=0,
    )
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_
    labels, centers, old_to_new = reorder_clusters(labels, centers, segment, args.label_order)
    counts = group_counts(labels, args.n_groups)
    representative_indices = select_representative_indices(features, labels, centers)
    representatives = trajectories[representative_indices]
    representative_segments = segment[representative_indices]
    quality = clustering_quality(features, labels, int(args.silhouette_sample_size), int(args.random_state))
    rep_distance = min_representative_trajectory_distance(segment, representative_indices)

    sub_labels, sub_representatives, sub_infos, coarse_to_global_subs = subgroup_kmeans(
        trajectories=trajectories,
        features=features,
        coarse_labels=labels,
        n_groups=int(args.n_groups),
        sub_groups_by_group=sub_groups_by_group,
        random_state=int(args.sub_group_seed),
    )

    sub_suffix = sub_group_suffix(sub_groups_by_group)
    base_name = f"coarse_group_{args.n_groups}_kmeans_history15_descriptors_sub{sub_suffix}"
    label_path = args.out_dir / "labels.npy"
    rep_path = args.out_dir / "representatives.npy"
    sub_label_path = args.out_dir / "sub_labels.npy"
    sub_rep_path = args.out_dir / "sub_representatives.npy"
    info_path = args.out_dir / "info.json"
    distribution_path = args.out_dir / "distribution.png"
    descriptor_space_path = args.out_dir / "descriptor_space.png"
    reps_path = args.out_dir / "reps.png"

    output_paths = [
        label_path,
        rep_path,
        sub_label_path,
        sub_rep_path,
        info_path,
        distribution_path,
        descriptor_space_path,
        reps_path,
    ]
    if not args.overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist: {existing}. Pass --overwrite true to replace.")

    np.save(label_path, labels.astype(np.int64))
    np.save(rep_path, representatives)
    np.save(sub_label_path, sub_labels.astype(np.int64))
    np.save(sub_rep_path, sub_representatives)

    feature_names = list(feature_info["feature_names"])
    info = {
        "schema_version": "trajectory_candidate_history15_descriptor_kmeans_v1",
        "purpose": "Final coarse/sub trajectory clustering using compact descriptors over origin + first 15 points.",
        "input_path": str(args.input),
        "input_shape": list(trajectories.shape),
        "method": "kmeans",
        "n_groups": int(args.n_groups),
        "sub_groups": [int(x) for x in sub_groups_by_group],
        "total_sub_groups": int(total_sub_groups),
        "random_state": int(args.random_state),
        "sub_group_seed": int(args.sub_group_seed),
        "label_order": args.label_order,
        "old_to_new_label": {str(k): int(v) for k, v in old_to_new.items()},
        "descriptor_feature": feature_info,
        "kmeans": {
            "n_init": int(args.n_init),
            "max_iter": int(args.max_iter),
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
        },
        "quality_metrics": {
            **quality,
            "min_representative_trajectory_distance": float(rep_distance),
        },
        "group_counts": [int(x) for x in counts],
        "group_size_stats": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
        },
        "group_feature_stats": group_feature_stats(raw_features, labels, feature_names, int(args.n_groups)),
        "representative_indices": [int(x) for x in representative_indices],
        "artifact_shapes": {
            "labels": list(labels.shape),
            "representatives": list(representatives.shape),
            "sub_labels": list(sub_labels.shape),
            "sub_representatives": list(sub_representatives.shape),
        },
        "sub_group_infos": sub_infos,
        "coarse_to_global_sub_group_ids": coarse_to_global_subs,
        "sub_labels_format": (
            "sub_labels.npy has shape (N,) and stores globally unique sub-group ids using "
            "the cumulative per-group offsets recorded in coarse_to_global_sub_group_ids."
        ),
        "output_files": {
            "labels": str(label_path),
            "representatives": str(rep_path),
            "sub_labels": str(sub_label_path),
            "sub_representatives": str(sub_rep_path),
            "info": str(info_path),
            "distribution": str(distribution_path),
            "descriptor_space": str(descriptor_space_path),
            "representatives_figure": str(reps_path),
        },
        "hgs_usage": {
            "candidate_trajectory_path": str(args.input),
            "coarse_labels_path": str(label_path),
            "coarse_representatives_path": str(rep_path),
            "sub_labels_path": str(sub_label_path),
            "sub_representatives_path": str(sub_rep_path),
            "representative_points": int(args.num_future_points),
        },
    }
    write_json(info_path, info)

    plot_distribution(labels, int(args.n_groups), distribution_path, title=f"Coarse-group candidate distribution ({base_name})")
    plot_descriptor_space(
        raw_features=raw_features,
        labels=labels,
        feature_names=feature_names,
        representative_indices=representative_indices,
        out_path=descriptor_space_path,
        title=f"Descriptor-space coarse groups ({base_name})",
    )
    plot_representatives(
        representative_segments=representative_segments,
        labels=labels,
        out_path=reps_path,
        title=f"Representative first-15-point instances ({base_name})",
    )

    print(f"Saved labels       -> {label_path}")
    print(f"Saved reps         -> {rep_path}")
    print(f"Saved sub labels   -> {sub_label_path}")
    print(f"Saved sub reps     -> {sub_rep_path}")
    print(f"Saved info         -> {info_path}")
    print(f"Group counts       -> {[int(x) for x in counts]}")
    print(f"Sub-groups/group   -> {sub_groups_by_group} (total={total_sub_groups})")


if __name__ == "__main__":
    main()
