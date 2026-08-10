"""Reusable device-side numerical operations for AIC Warp kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp


if TYPE_CHECKING:
    Matrix6 = Any
    Vector6 = Any
else:
    Matrix6 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
    Vector6 = wp.types.vector(length=6, dtype=wp.float32)
CHOLESKY_EPSILON = 1.0e-12
SMALL_ANGLE_EPSILON = 1.0e-6
PI_ANGLE_EPSILON = 1.0e-4


@wp.func
def scale_to_limit(vector: wp.vec3, limit: float) -> wp.vec3:
    """Preserve a vector's direction while limiting its magnitude.

    Args:
        vector: Vector to limit.
        limit: Maximum permitted Euclidean norm.

    Returns:
        The original vector or its magnitude-limited equivalent.
    """

    magnitude = wp.length(vector)
    if magnitude > limit:
        return vector * (limit / magnitude)
    return vector


@wp.func
def rotation_error_world(current: wp.mat33, target: wp.mat33) -> wp.vec3:
    """Compute the world-frame SO(3) logarithm between two rotations.

    The result is an axis multiplied by the exact principal angle in radians.
    A matrix-based near-pi branch avoids the zero-skew singularity of the
    common small-angle approximation.

    Args:
        current: Current world-frame rotation matrix.
        target: Desired world-frame rotation matrix.

    Returns:
        A world-frame rotation vector whose norm is in radians.
    """

    current_x = wp.vec3(current[0, 0], current[1, 0], current[2, 0])
    current_y = wp.vec3(current[0, 1], current[1, 1], current[2, 1])
    current_z = wp.vec3(current[0, 2], current[1, 2], current[2, 2])
    target_x = wp.vec3(target[0, 0], target[1, 0], target[2, 0])
    target_y = wp.vec3(target[0, 1], target[1, 1], target[2, 1])
    target_z = wp.vec3(target[0, 2], target[1, 2], target[2, 2])
    skew_error = 0.5 * (
        wp.cross(current_x, target_x)
        + wp.cross(current_y, target_y)
        + wp.cross(current_z, target_z)
    )
    cosine = 0.5 * (
        wp.dot(current_x, target_x)
        + wp.dot(current_y, target_y)
        + wp.dot(current_z, target_z)
        - 1.0
    )
    cosine = wp.clamp(cosine, -1.0, 1.0)
    angle = wp.acos(cosine)
    if angle < SMALL_ANGLE_EPSILON:
        return skew_error

    sine = wp.sin(angle)
    if wp.abs(sine) > PI_ANGLE_EPSILON:
        return skew_error * (angle / sine)

    current_row_x = wp.vec3(current[0, 0], current[0, 1], current[0, 2])
    current_row_y = wp.vec3(current[1, 0], current[1, 1], current[1, 2])
    current_row_z = wp.vec3(current[2, 0], current[2, 1], current[2, 2])
    target_row_x = wp.vec3(target[0, 0], target[0, 1], target[0, 2])
    target_row_y = wp.vec3(target[1, 0], target[1, 1], target[1, 2])
    target_row_z = wp.vec3(target[2, 0], target[2, 1], target[2, 2])

    error_00 = wp.dot(target_row_x, current_row_x)
    error_01 = wp.dot(target_row_x, current_row_y)
    error_02 = wp.dot(target_row_x, current_row_z)
    error_10 = wp.dot(target_row_y, current_row_x)
    error_11 = wp.dot(target_row_y, current_row_y)
    error_12 = wp.dot(target_row_y, current_row_z)
    error_20 = wp.dot(target_row_z, current_row_x)
    error_21 = wp.dot(target_row_z, current_row_y)
    error_22 = wp.dot(target_row_z, current_row_z)

    axis = wp.vec3(0.0, 0.0, 0.0)
    if error_00 >= error_11 and error_00 >= error_22:
        axis_x = wp.sqrt(wp.max(0.5 * (error_00 + 1.0), SMALL_ANGLE_EPSILON))
        axis = wp.vec3(
            axis_x,
            (error_01 + error_10) / (4.0 * axis_x),
            (error_02 + error_20) / (4.0 * axis_x),
        )
    elif error_11 >= error_22:
        axis_y = wp.sqrt(wp.max(0.5 * (error_11 + 1.0), SMALL_ANGLE_EPSILON))
        axis = wp.vec3(
            (error_01 + error_10) / (4.0 * axis_y),
            axis_y,
            (error_12 + error_21) / (4.0 * axis_y),
        )
    else:
        axis_z = wp.sqrt(wp.max(0.5 * (error_22 + 1.0), SMALL_ANGLE_EPSILON))
        axis = wp.vec3(
            (error_02 + error_20) / (4.0 * axis_z),
            (error_12 + error_21) / (4.0 * axis_z),
            axis_z,
        )
    return wp.normalize(axis) * angle


@wp.func
def damped_least_squares(
    jacobian: Matrix6,
    cartesian_step: Vector6,
    damping: float,
) -> Vector6:
    """Solve one six-dimensional damped least-squares IK system.

    Warp does not expose a per-thread inverse for a 6x6 matrix. This fixed-size
    Cholesky solve keeps every environment on the device and avoids a host or
    PyTorch synchronization in each control step.

    Args:
        jacobian: Six-dimensional geometric Jacobian.
        cartesian_step: Requested translation and rotation error step.
        damping: Positive Tikhonov damping coefficient.

    Returns:
        Six bounded-later joint increments before joint-limit handling.
    """

    system = Matrix6(0.0)
    for row in range(6):
        for column in range(6):
            value = float(0.0)
            for joint in range(6):
                value += jacobian[row, joint] * jacobian[column, joint]
            if row == column:
                value += damping * damping
            system[row, column] = value

    lower = Matrix6(0.0)
    for row in range(6):
        for column in range(row + 1):
            value = system[row, column]
            for inner in range(column):
                value -= lower[row, inner] * lower[column, inner]
            if row == column:
                lower[row, column] = wp.sqrt(wp.max(value, CHOLESKY_EPSILON))
            else:
                lower[row, column] = value / lower[column, column]

    forward = Vector6(0.0)
    for row in range(6):
        value = cartesian_step[row]
        for column in range(row):
            value -= lower[row, column] * forward[column]
        forward[row] = value / lower[row, row]

    solved = Vector6(0.0)
    for reverse_row in range(6):
        row = 5 - reverse_row
        value = forward[row]
        for column in range(row + 1, 6):
            value -= lower[column, row] * solved[column]
        solved[row] = value / lower[row, row]

    joint_step = Vector6(0.0)
    for joint in range(6):
        value = float(0.0)
        for row in range(6):
            value += jacobian[row, joint] * solved[row]
        joint_step[joint] = value
    return joint_step
