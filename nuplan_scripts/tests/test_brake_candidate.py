"""Behavioral checks for the straight brake fallbacks of the solvability checkers."""

import sys
from pathlib import Path

import numpy as np
import pytest
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuplan_scripts import run_solvability_check_new as checker
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

SPEED_MPS = 10.0
HEADINGS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -np.pi / 2]


def make_ego_state(heading: float, speed: float = SPEED_MPS, x: float = 100.0, y: float = -50.0) -> EgoState:
    # rear_axle_velocity_2d is in the ego frame: x is forward speed regardless of heading.
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x, y, heading),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
    )


def local_track(states: np.ndarray) -> np.ndarray:
    """Displacements expressed in the initial heading frame (forward, left)."""
    heading = states[0, StateIndex.HEADING]
    delta = states[:, StateIndex.STATE_SE2][:, :2] - states[0, StateIndex.STATE_SE2][:2]
    rotation = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    return delta @ rotation.T


@pytest.fixture(scope="module")
def planner():
    return checker.AdaptiveReactiveIDMSolvabilityPlanner()


def test_brake_set_is_idm_and_comfort_bound():
    assert checker.BRAKE_DECELERATIONS_MPS2 == (3.0, 4.05)


@pytest.mark.parametrize("deceleration", checker.BRAKE_DECELERATIONS_MPS2)
def test_brake_shape_does_not_depend_on_global_heading(planner, deceleration):
    tracks = [local_track(planner.brake_candidate(make_ego_state(h), deceleration)) for h in HEADINGS]
    for track in tracks[1:]:
        np.testing.assert_allclose(track, tracks[0], atol=1e-3)


@pytest.mark.parametrize("deceleration", checker.BRAKE_DECELERATIONS_MPS2)
@pytest.mark.parametrize("heading", HEADINGS)
def test_brake_reaches_requested_deceleration_without_stopping_early(planner, heading, deceleration):
    states = planner.brake_candidate(make_ego_state(heading), deceleration)
    track = local_track(states)
    horizon_s = checker.base.SAMPLING.time_horizon
    stop_s = min(horizon_s, SPEED_MPS / deceleration)
    ideal_distance = SPEED_MPS * stop_s - 0.5 * deceleration * stop_s**2
    # The LQR tracker ramps into the brake and creeps towards standstill, so it
    # travels somewhat further than the ideal profile; the old heading-projection
    # bug instead stopped short of it (near the start pose at heading pi/2).
    assert track[-1, 0] >= ideal_distance
    assert track[-1, 0] <= 1.3 * ideal_distance
    assert -states[:, StateIndex.ACCELERATION_X].min() == pytest.approx(deceleration, rel=0.1)
    assert np.abs(track[:, 1]).max() < 0.05


def test_stronger_brake_stops_sooner(planner):
    soft, hard = (
        local_track(planner.brake_candidate(make_ego_state(np.pi / 2), d))[-1, 0]
        for d in checker.BRAKE_DECELERATIONS_MPS2
    )
    assert hard < soft


def test_emergency_candidates_emit_one_brake_per_deceleration(planner):
    context = type("Ctx", (), {"ego_state": make_ego_state(np.pi / 2)})()
    labels = [label for label, _ in planner.emergency_candidates(context, 0)]
    assert labels == ["iter0_brake_3.00mps2", "iter0_brake_4.05mps2"]
