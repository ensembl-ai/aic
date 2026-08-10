"""Device command objects accepted by the AIC simulation controller."""

from __future__ import annotations

from typing import Any

import warp as wp


class HoldPositionCommand:
    """One desired six-joint HOLD position for every MJWarp environment."""

    def __init__(self, num_envs: int, joint_count: int, device: Any):
        self.position = wp.zeros((num_envs, joint_count), dtype=float, device=device)
