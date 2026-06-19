"""AIC MJLab observation placeholders.

Do not add observation formulas here casually. Each term should be chosen after
we decide what the policy is allowed to observe for a given task.
"""

from __future__ import annotations


def joint_state_observation(env, asset_cfg=None):
    """Placeholder for selected robot joint positions/velocities."""
    raise NotImplementedError("Select exact joint-state observation contents first.")


def tcp_pose_observation(env, asset_cfg=None, frame_name: str = "gripper_tcp"):
    """Placeholder for TCP pose or task-frame TCP pose."""
    raise NotImplementedError("Select TCP pose frame and representation first.")


def zeroed_force_torque_observation(
    env,
    sensor_name: str = "AtiForceTorqueSensor",
    baseline_name: str = "ati_ft_zero",
):
    """Placeholder for force/torque minus reset-time baseline."""
    raise NotImplementedError(
        "Add sensor indexing and reset-time baseline storage before use."
    )


def task_geometry_observation(env, asset_cfg=None):
    """Placeholder for board, NIC, port, plug, or cable task geometry."""
    raise NotImplementedError("Decide task state exposure before use.")


def camera_health_observation(env, camera_names: tuple[str, ...]):
    """Placeholder for camera availability/diagnostic observations."""
    raise NotImplementedError(
        "Camera observations should be added after proprioceptive training is stable."
    )
