"""AIC MJLab reset/randomization event placeholders."""

from __future__ import annotations


def reset_robot_to_initial_state(env, env_ids=None, **kwargs):
    """Placeholder for robot reset."""
    raise NotImplementedError("Choose reset joint pose/ranges first.")


def randomize_task_board(env, env_ids=None, **kwargs):
    """Placeholder for task board pose randomization."""
    raise NotImplementedError("Use AIC-specified ranges once selected.")


def randomize_nic_or_port(env, env_ids=None, **kwargs):
    """Placeholder for NIC/port randomization."""
    raise NotImplementedError("Choose which task objects are randomized first.")


def zero_force_torque_sensor(
    env,
    env_ids=None,
    sensor_name: str = "AtiForceTorqueSensor",
    baseline_name: str = "ati_ft_zero",
    samples: int = 1,
):
    """Placeholder for reset-time force/torque baseline capture."""
    raise NotImplementedError(
        "Implement after confirming MJLab/MuJoCo Warp sensor data layout."
    )
