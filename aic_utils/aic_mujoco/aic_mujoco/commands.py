"""Device command objects accepted by the AIC simulation controller."""

from __future__ import annotations

from typing import Any

import warp as wp


class HoldPositionCommand:
    """One desired six-joint HOLD position for every MJWarp environment."""

    def __init__(self, num_envs: int, joint_count: int, device: Any):
        """Allocate zero-initialized joint targets on a Warp device.

        Args:
            num_envs: Number of parallel environments.
            joint_count: Number of controlled arm joints.
            device: Warp device that owns the command.
        """

        self.position = wp.zeros((num_envs, joint_count), dtype=float, device=device)


class CartesianPoseCommand:
    """One desired world-frame Cartesian pose per MJWarp environment."""

    def __init__(self, num_envs: int, device: Any):
        """Allocate position, rotation-matrix, and activation tensors.

        Args:
            num_envs: Number of parallel environments.
            device: Warp device that owns the command.
        """

        self.position = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.rotation = wp.zeros(num_envs, dtype=wp.mat33, device=device)
        self.active = wp.zeros(num_envs, dtype=bool, device=device)


class JointDeltaAction:
    """Bounded six-joint target increments emitted by a motion policy."""

    def __init__(self, num_envs: int, joint_count: int, device: Any):
        """Allocate zero-initialized joint increments on a Warp device.

        Args:
            num_envs: Number of parallel environments.
            joint_count: Number of controlled arm joints.
            device: Warp device that owns the action.
        """

        self.position = wp.zeros((num_envs, joint_count), dtype=float, device=device)
