"""Comfort checks matching MeanFuser's native NAVSIM navtest scorer.

The input is the simulated proposal from t=0 through t=4 s. The acceleration
checks use the full proposal for smoothing; jerk uses eight poses to smooth
acceleration and the full proposal to estimate its derivative. NAVSIM's yaw
checks use a five-pose derivative window.
"""

import numpy as np
from scipy.signal import savgol_filter

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def _smooth_acceleration(values: np.ndarray, window_length: int) -> np.ndarray:
    return np.round(
        savgol_filter(values, polyorder=2, window_length=min(window_length, values.shape[-1]), axis=-1),
        decimals=8,
    )


def _derivative(values: np.ndarray, time_point_s: np.ndarray, *, order: int, polynomial: int, window: int) -> np.ndarray:
    window = min(window, len(time_point_s))
    if polynomial >= window:
        raise ValueError(f"{polynomial} < {window} does not hold!")
    intervals = np.diff(time_point_s)
    if not (intervals > 0).all():
        raise RuntimeError("Time points must be strictly increasing")
    return savgol_filter(
        values,
        polyorder=polynomial,
        window_length=window,
        deriv=order,
        delta=intervals.mean(),
        axis=-1,
    )


def _within_bounds(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.all((values > lower) & (values < upper), axis=-1)


def ego_is_comfortable_navtest(states: np.ndarray, time_point_s: np.ndarray) -> np.ndarray:
    """Return NAVSIM's six binary comfort checks for each simulated proposal."""
    num_proposals, num_steps, num_states = states.shape
    assert num_steps == len(time_point_s)
    assert num_states == StateIndex.size()

    acceleration_x = states[..., StateIndex.ACCELERATION_X]
    acceleration_y = states[..., StateIndex.ACCELERATION_Y]
    acceleration_magnitude = np.hypot(acceleration_x, acceleration_y)

    def jerk(acceleration: np.ndarray) -> np.ndarray:
        smoothed = _smooth_acceleration(acceleration, window_length=8)
        return np.round(
            _derivative(smoothed, time_point_s, order=1, polynomial=2, window=num_steps),
            decimals=8,
        )

    headings = states[..., StateIndex.HEADING]
    full_turn = 2.0 * np.pi
    turns = np.zeros_like(headings)
    turns[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / full_turn), axis=-1)
    unwrapped_headings = headings - full_turn * turns
    # NAVSIM's yaw helper uses its default five-pose derivative window.
    yaw_acceleration = np.round(
        _derivative(unwrapped_headings, time_point_s, order=2, polynomial=3, window=5), decimals=8
    )
    yaw_rate = np.round(
        _derivative(unwrapped_headings, time_point_s, order=1, polynomial=2, window=5), decimals=8
    )

    return np.stack(
        [
            _within_bounds(_smooth_acceleration(acceleration_x, num_steps), -4.05, 2.40),
            _within_bounds(_smooth_acceleration(acceleration_y, num_steps), -4.89, 4.89),
            _within_bounds(jerk(acceleration_magnitude), -8.37, 8.37),
            _within_bounds(jerk(acceleration_x), -4.13, 4.13),
            _within_bounds(yaw_acceleration, -1.93, 1.93),
            _within_bounds(yaw_rate, -0.95, 0.95),
        ],
        axis=-1,
    )
