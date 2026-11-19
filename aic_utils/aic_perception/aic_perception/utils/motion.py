"""Geometry for plug placement and force-guided search, on numpy arrays."""

from __future__ import annotations

import math

import numpy as np

from aic_perception.utils.geometry import invert

UP = np.array([0.0, 0.0, 1.0])


def level_rotation(rotation: np.ndarray) -> np.ndarray:
    """Remove the tilt of a rotation whose frame should be level with the base.

    The AIC board lies flat on the table, so every port frame has one axis
    along base ``z``. The axis closest to vertical is snapped onto ``±z`` and
    the frame re-orthonormalized, keeping the estimated yaw.

    Args:
        rotation: 3x3 rotation with small tilt error.

    Returns:
        3x3 rotation with one axis exactly vertical.
    """

    axis = int(np.argmax(np.abs(rotation[2, :])))
    columns = rotation.copy()
    columns[:, axis] = np.sign(rotation[2, axis]) * UP
    following = (axis + 1) % 3
    projected = columns[:, following] - columns[:, axis] * (
        columns[:, axis] @ columns[:, following]
    )
    columns[:, following] = projected / np.linalg.norm(projected)
    last = (axis + 2) % 3
    columns[:, last] = np.cross(columns[:, axis], columns[:, following])
    return columns


def tcp_target(base_from_plug: np.ndarray, tcp_from_plug: np.ndarray) -> np.ndarray:
    """TCP pose that places the plug at a desired pose.

    Args:
        base_from_plug: Desired plug pose in the base frame.
        tcp_from_plug: Calibrated plug pose in the TCP frame.

    Returns:
        Required TCP pose in the base frame.
    """

    return base_from_plug @ invert(tcp_from_plug)


def spiral_point(angle: float, pitch: float) -> np.ndarray:
    """Point on an Archimedean spiral whose turns are ``pitch`` apart.

    Args:
        angle: Spiral angle in radians, starting at 0.
        pitch: Radial spacing between turns in meters.

    Returns:
        ``(x, y)`` offset in meters.
    """

    radius = pitch * angle / (2.0 * math.pi)
    return np.array([radius * math.cos(angle), radius * math.sin(angle)])


def spiral_step(angle: float, pitch: float, speed: float, period: float) -> float:
    """Advance the spiral angle so the tip moves at a constant path speed.

    Args:
        angle: Current spiral angle.
        pitch: Radial spacing between turns.
        speed: Desired path speed in meters per second.
        period: Control period in seconds.

    Returns:
        Next spiral angle.
    """

    radius = max(pitch * angle / (2.0 * math.pi), pitch)
    return angle + speed * period / radius


class PortAverage:
    """Running-average estimate of a static port position with outlier rejection.

    Each accepted measurement moves the estimate by ``alpha`` of the
    difference (an exponential moving average, i.e. a fixed-gain Kalman filter
    for a target that does not move). A measurement farther than ``gate`` from
    the estimate is rejected; after ``reset_after`` rejections in a row the
    estimate restarts from the latest measurement, so one bad initial
    estimate cannot lock out all later evidence.
    """

    def __init__(
        self, initial: np.ndarray, alpha: float, gate: float, reset_after: int
    ):
        """Start from an initial estimate.

        Args:
            initial: Initial port position.
            alpha: Weight of each new measurement, in (0, 1].
            gate: Largest distance from the estimate that is still accepted.
            reset_after: Consecutive rejections that restart the estimate.
        """

        self.position = np.asarray(initial, dtype=float).copy()
        self.alpha, self.gate, self.reset_after = alpha, gate, reset_after
        self.accepted = self.rejected = self.resets = 0
        self.streak = 0

    def update(self, measurement: np.ndarray) -> bool:
        """Fold in one measurement; returns whether it was accepted."""

        measurement = np.asarray(measurement, dtype=float)
        if np.linalg.norm(measurement - self.position) > self.gate:
            self.rejected += 1
            self.streak += 1
            if self.streak >= self.reset_after:
                self.position = measurement.copy()
                self.resets += 1
                self.streak = 0
            return False
        self.streak = 0
        self.accepted += 1
        self.position += self.alpha * (measurement - self.position)
        return True


def funnel_target(port: np.ndarray, plug: np.ndarray, height: float) -> np.ndarray:
    """Point to steer the plug toward: above the port, lower only as it gets closer sideways.

    The target height is ``height`` above the port plus the plug's current
    sideways distance from it (a 45° funnel), so the plug descends only once
    it is over the port and never sweeps low across neighbouring parts.

    Args:
        port: Port entrance position.
        plug: Current commanded plug position.
        height: Final height above the entrance.

    Returns:
        Target position.
    """

    lateral = float(np.linalg.norm(np.asarray(plug)[:2] - np.asarray(port)[:2]))
    return np.asarray(port, dtype=float) + np.array([0.0, 0.0, height + lateral])


def speed_limited_step(
    position: np.ndarray,
    target: np.ndarray,
    speed: float,
    max_speed: float,
    accel: float,
    period: float,
) -> tuple[np.ndarray, float]:
    """Move a commanded point toward a (possibly moving) target with bounded speed and acceleration.

    Speed ramps up by ``accel`` per second, never exceeds ``max_speed``, and
    is limited to ``sqrt(2 * accel * distance)`` so the point brakes smoothly
    instead of stopping abruptly.

    Returns:
        New position and the speed used.
    """

    delta = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    distance = float(np.linalg.norm(delta))
    speed = min(speed + accel * period, max_speed, math.sqrt(2.0 * accel * distance))
    step = speed * period
    if distance <= step or distance == 0.0:
        return np.asarray(target, dtype=float).copy(), speed
    return np.asarray(position, dtype=float) + delta / distance * step, speed
