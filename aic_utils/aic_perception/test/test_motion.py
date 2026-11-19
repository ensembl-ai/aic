"""Plug placement geometry and spiral search helpers."""

from __future__ import annotations

import math

import numpy as np

from aic_perception.utils.geometry import from_xyz_rpy
from aic_perception.utils.motion import level_rotation, spiral_point, spiral_step, tcp_target


def test_level_rotation_removes_tilt_and_keeps_yaw() -> None:
    """A slightly tilted port frame becomes exactly level with the same heading."""

    for up_axis_rpy in ((0.0, 0.0, 0.7), (math.pi / 2, 0.0, -1.2)):
        exact = from_xyz_rpy((0, 0, 0), up_axis_rpy)[:3, :3]
        tilted = from_xyz_rpy((0, 0, 0), (0.03, -0.02, 0.0))[:3, :3] @ exact
        level = level_rotation(tilted)
        assert np.allclose(level.T @ level, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(level), 1.0)
        assert np.isclose(np.max(np.abs(level[2, :])), 1.0)
        assert np.allclose(level, exact, atol=2e-3)


def test_tcp_target_places_the_plug() -> None:
    """Commanding the returned TCP pose puts the calibrated plug at the goal."""

    tcp_from_plug = from_xyz_rpy((0.0, 0.015, 0.06), (0.4, -0.5, 1.3))
    goal = from_xyz_rpy((-0.4, 0.3, 0.2), (math.pi, 0.0, 0.3))
    assert np.allclose(tcp_target(goal, tcp_from_plug) @ tcp_from_plug, goal)


def test_spiral_turns_are_one_pitch_apart_and_speed_is_bounded() -> None:
    """Consecutive turns are ``pitch`` apart and each step moves about ``speed * period``."""

    pitch, speed, period = 0.001, 0.004, 0.05
    assert np.isclose(np.linalg.norm(spiral_point(2 * math.pi, pitch)), pitch)
    assert np.isclose(np.linalg.norm(spiral_point(4 * math.pi, pitch)), 2 * pitch)
    angle, previous = 0.0, spiral_point(0.0, pitch)
    for _ in range(400):
        angle = spiral_step(angle, pitch, speed, period)
        point = spiral_point(angle, pitch)
        assert np.linalg.norm(point - previous) <= speed * period * 1.05 + 1e-12
        previous = point


def test_port_average_converges_and_rejects_outliers() -> None:
    """Accepted measurements pull the estimate in; outliers are ignored, then a persistent shift resets it."""

    from aic_perception.utils.motion import PortAverage

    tracker = PortAverage(np.zeros(3), alpha=0.2, gate=0.008, reset_after=5)
    for _ in range(40):
        tracker.update(np.array([0.003, 0.0, 0.0]))
    assert np.allclose(tracker.position, [0.003, 0.0, 0.0], atol=1e-5)
    assert not tracker.update(np.array([0.05, 0.0, 0.0]))
    assert np.allclose(tracker.position, [0.003, 0.0, 0.0], atol=1e-5)
    for _ in range(4):
        tracker.update(np.array([0.05, 0.0, 0.0]))
    assert tracker.resets == 1 and np.allclose(tracker.position, [0.05, 0.0, 0.0])


def test_funnel_target_descends_only_near_the_port() -> None:
    """Far to the side the target stays high; directly above it is the final height."""

    from aic_perception.utils.motion import funnel_target

    port = np.array([0.1, 0.2, 0.03])
    assert np.allclose(funnel_target(port, port + [0.0, 0.0, 0.2], 0.01), port + [0.0, 0.0, 0.01])
    assert np.isclose(funnel_target(port, port + [0.05, 0.0, 0.2], 0.01)[2], 0.03 + 0.01 + 0.05)


def test_speed_limited_step_ramps_brakes_and_arrives() -> None:
    """The commanded point accelerates, never exceeds the speed limit, and lands on the target."""

    from aic_perception.utils.motion import speed_limited_step

    position, speed, speeds = np.zeros(3), 0.0, []
    target = np.array([0.3, 0.0, 0.0])
    for _ in range(400):
        position, speed = speed_limited_step(position, target, speed, 0.06, 0.15, 0.05)
        speeds.append(speed)
    assert np.allclose(position, target)
    assert max(speeds) <= 0.06 + 1e-12
    assert all(b - a <= 0.15 * 0.05 + 1e-12 for a, b in zip(speeds, speeds[1:]))
