"""Validated host-side pose operations backed by MuJoCo's math API."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np


def normalized_quaternion(values: Sequence[float]) -> np.ndarray:
    """Return a normalized MuJoCo WXYZ quaternion.

    This utility exists only at the MJCF boundary, where MuJoCo represents
    orientations as quaternions. Control and dataset APIs use rotation
    matrices.

    Args:
        values: Four quaternion components in MuJoCo WXYZ order.

    Returns:
        A normalized four-element ``float64`` array.

    Raises:
        ValueError: If ``values`` does not contain four finite components or
            has zero length.
    """

    quaternion = np.asarray(values, dtype=np.float64).copy()
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("A MuJoCo quaternion must contain four finite values")
    if mujoco.mju_normalize4(quaternion) == 0.0:
        raise ValueError("Cannot normalize a zero quaternion")
    return quaternion


def rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """Rotate a three-dimensional vector with MuJoCo's math API.

    Args:
        quaternion: Rotation in MuJoCo WXYZ order.
        vector: Three-dimensional vector to rotate.

    Returns:
        The rotated three-element ``float64`` array.

    Raises:
        ValueError: If either input has the wrong shape or invalid values.
    """

    rotation = normalized_quaternion(quaternion)
    source = np.asarray(vector, dtype=np.float64)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("A vector must contain three finite values")
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, source, rotation)
    return result


def rotation_matrix_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an MJCF quaternion into a three-by-three rotation matrix.

    Args:
        quaternion: Rotation in MuJoCo WXYZ order.

    Returns:
        A ``(3, 3)`` ``float64`` SO(3) rotation matrix.

    Raises:
        ValueError: If the quaternion is malformed or has zero length.
    """

    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, normalized_quaternion(quaternion))
    return result.reshape(3, 3)


def compose_pose(
    parent_position: Sequence[float],
    parent_quaternion: Sequence[float],
    child_position: Sequence[float],
    child_quaternion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two MJCF poses using MuJoCo's native pose multiplication.

    Args:
        parent_position: Parent translation with three components.
        parent_quaternion: Parent orientation in MuJoCo WXYZ order.
        child_position: Child translation with three components.
        child_quaternion: Child orientation in MuJoCo WXYZ order.

    Returns:
        A ``(position, quaternion)`` pair in the parent frame.

    Raises:
        ValueError: If a position or quaternion is malformed.
    """

    first_position = np.asarray(parent_position, dtype=np.float64)
    second_position = np.asarray(child_position, dtype=np.float64)
    if first_position.shape != (3,) or not np.all(np.isfinite(first_position)):
        raise ValueError("Parent position must contain three finite values")
    if second_position.shape != (3,) or not np.all(np.isfinite(second_position)):
        raise ValueError("Child position must contain three finite values")

    result_position = np.empty(3, dtype=np.float64)
    result_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mulPose(
        result_position,
        result_quaternion,
        first_position,
        normalized_quaternion(parent_quaternion),
        second_position,
        normalized_quaternion(child_quaternion),
    )
    return result_position, result_quaternion
