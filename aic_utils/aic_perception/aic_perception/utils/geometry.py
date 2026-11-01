"""Rigid-transform and camera-projection helpers on 4x4 numpy matrices.

Quaternion and Euler conversions come from ``transforms3d``, which is
available both in the pixi environment and inside the ``aic_eval`` container.
"""

from __future__ import annotations

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a homogeneous transform.

    Args:
        rotation: 3x3 rotation matrix.
        translation: Length-3 translation.

    Returns:
        4x4 transform.
    """

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def from_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    """Build a transform from a position and URDF-style fixed-axis roll/pitch/yaw.

    Args:
        xyz: Translation.
        rpy: Roll, pitch, yaw in radians.

    Returns:
        4x4 transform.
    """

    return transform(euler2mat(*rpy, axes="sxyz"), np.asarray(xyz, dtype=float))


def from_xyz_quaternion_xyzw(xyz: tuple[float, ...], xyzw: tuple[float, ...]) -> np.ndarray:
    """Build a transform from a ROS-ordered ``(x, y, z, w)`` quaternion.

    Args:
        xyz: Translation.
        xyzw: Unit quaternion in ROS order.

    Returns:
        4x4 transform.
    """

    x, y, z, w = xyzw
    return transform(quat2mat((w, x, y, z)), np.asarray(xyz, dtype=float))


def quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return the rotation of a transform as a ROS-ordered quaternion.

    Args:
        matrix: 4x4 transform or 3x3 rotation.

    Returns:
        ``(x, y, z, w)`` with non-negative ``w``.
    """

    w, x, y, z = mat2quat(np.asarray(matrix)[:3, :3])
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return float(x), float(y), float(z), float(w)


def invert(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid transform.

    Args:
        matrix: 4x4 rigid transform.

    Returns:
        Inverse transform.
    """

    rotation = matrix[:3, :3].T
    return transform(rotation, -rotation @ matrix[:3, 3])


def project(
    intrinsics: np.ndarray, base_from_camera: np.ndarray, points_base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points into an optical-frame pinhole camera.

    Args:
        intrinsics: 3x3 camera matrix.
        base_from_camera: 4x4 pose of the optical frame in the base frame.
        points_base: ``(N, 3)`` points in the base frame.

    Returns:
        ``(N, 2)`` pixel coordinates and ``(N,)`` optical-axis depths.
    """

    camera_from_base = invert(base_from_camera)
    points_camera = points_base @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
    depth = points_camera[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    pixels = (points_camera @ intrinsics.T)[:, :2] / safe_depth[:, None]
    return pixels, depth


def scaled_intrinsics(intrinsics: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Rescale a camera matrix for a resized image.

    Uses the pixel-center convention, so a resize maps continuous pixel
    coordinate ``u`` to ``(u + 0.5) * scale - 0.5``.

    Args:
        intrinsics: 3x3 camera matrix of the source image.
        scale_x: Output width divided by source width.
        scale_y: Output height divided by source height.

    Returns:
        3x3 camera matrix of the resized image.
    """

    scaled = np.array(intrinsics, dtype=float)
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] = (scaled[0, 2] + 0.5) * scale_x - 0.5
    scaled[1, 2] = (scaled[1, 2] + 0.5) * scale_y - 0.5
    return scaled
