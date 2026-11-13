"""Ego dynamics of a designed candidate trajectory.

The current ego state of a synthetic scene must describe the motion that the
rendered history shows. It is therefore derived from the candidate trajectory
alone, never from logged frames attached after the current frame.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def candidate_time_axis(candidate: np.ndarray, candidate_frame_rate: float, candidate_prepend_origin: bool) -> np.ndarray:
    first_sample_index = 0 if candidate_prepend_origin else 1
    return (np.arange(len(candidate), dtype=np.float64) + first_sample_index) / candidate_frame_rate


def candidate_ego_dynamics(
    candidate: np.ndarray,
    candidate_frame_rate: float,
    candidate_prepend_origin: bool,
    query_seconds: Sequence[float],
    window_seconds: float = 0.5,
) -> np.ndarray:
    """Return ego-frame ``(vx, vy, ax, ay)`` of a prepared candidate at ``query_seconds``.

    ``candidate`` is the prepared ``(T, 2|3)`` array (x, y[, yaw]) in its own frame, as
    returned by ``prepare_candidate``. The state at time ``t`` is causal: a quadratic
    least-squares fit to the candidate samples in ``[t - window_seconds, t]`` (one NAVSIM
    frame interval by default), so it reflects only motion up to that frame. A frame with
    no earlier samples (t = 0) uses ``[t, t + window_seconds]``. Acceleration is the planar
    acceleration expressed in the ego frame, matching ``recompute_dynamics``.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    times = candidate_time_axis(candidate, candidate_frame_rate, candidate_prepend_origin)
    if len(candidate) < 3:
        raise ValueError(f"Candidate with {len(candidate)} samples is too short for derivative estimation.")
    xy = candidate[:, :2]
    if candidate.shape[1] >= 3:
        yaws = np.unwrap(candidate[:, 2])
    else:
        step = np.gradient(xy, times, axis=0)
        yaws = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))

    eps = 1e-9
    query = np.clip(np.asarray(query_seconds, dtype=np.float64), times[0], times[-1])
    out = np.zeros((len(query), 4), dtype=np.float64)
    for i, t in enumerate(query):
        mask = (times >= t - window_seconds - eps) & (times <= t + eps)
        if mask.sum() < 3:
            mask = (times >= t - eps) & (times <= t + window_seconds + eps)
        tau = times[mask] - t
        coeffs = [np.polyfit(tau, xy[mask, k], 2) for k in range(2)]  # c2 tau^2 + c1 tau + c0
        vel = np.array([coeffs[k][1] for k in range(2)])
        acc = np.array([2.0 * coeffs[k][0] for k in range(2)])
        yaw = np.interp(t, times, yaws)
        c, s = np.cos(-yaw), np.sin(-yaw)
        to_ego = np.array([[c, -s], [s, c]], dtype=np.float64)
        out[i, :2] = to_ego @ vel
        out[i, 2:] = to_ego @ acc
    return out


def history_query_seconds(history_count: int, candidate_history_duration: float) -> np.ndarray:
    """Candidate times sampled by ``history_first_half`` for each history frame."""
    seconds = candidate_history_duration - np.arange(history_count - 1, -1, -1, dtype=np.float64) * 0.5
    return np.maximum(seconds, 0.0)
