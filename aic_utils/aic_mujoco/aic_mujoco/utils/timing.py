"""Wall-clock pacing shared by interactive simulation workflows."""

from __future__ import annotations

import time


def wait_for_realtime(start_time: float, step_index: int, timestep: float) -> None:
    """Wait until a physics step's real-time deadline.

    Args:
        start_time: Wall-clock time immediately before the rollout loop.
        step_index: Number of physics steps completed since ``start_time``.
        timestep: Configured physics duration of one step in seconds.
    """

    deadline = start_time + step_index * timestep
    delay = deadline - time.perf_counter()
    if delay > 0.0:
        time.sleep(delay)
