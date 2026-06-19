"""MJLab environment-configuration skeleton for AIC insertion training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AicInsertionTaskSpec:
    """Names and paths needed to build the AIC MJLab task.

    These defaults match the generated MuJoCo files in ``aic_utils/aic_mujoco``.
    They are intentionally just identifiers. Reward, observation, and
    randomization terms are not defined here.
    """

    mjcf_dir: Path = Path(__file__).resolve().parents[2] / "mjcf"
    scene_xml: str = "scene.xml"
    robot_entity_name: str = "robot"
    world_entity_name: str = "world"
    tcp_site_name: str = "gripper_tcp"
    force_torque_sensor_name: str = "AtiForceTorqueSensor"
    arm_actuator_patterns: tuple[str, ...] = (
        "shoulder_pan_joint_motor",
        "shoulder_lift_joint_motor",
        "elbow_joint_motor",
        "wrist_1_joint_motor",
        "wrist_2_joint_motor",
        "wrist_3_joint_motor",
    )


def make_aic_insertion_env_cfg(spec: AicInsertionTaskSpec | None = None):
    """Return a future MJLab ``ManagerBasedRlEnvCfg`` for AIC insertion.

    This function is intentionally not implemented yet. The next step is to
    decide whether the first MJLab implementation should:

    1. load the already-composed ``scene.xml`` as one entity, or
    2. split robot/world/task objects into MJLab entities using ``MjSpec``.

    After that, we can wire in MJLab's built-in ``DifferentialIKActionCfg`` for
    Cartesian delta actions and add placeholders from ``observations.py``,
    ``events.py``, ``rewards.py``, and ``terminations.py`` as real terms.
    """

    _ = spec or AicInsertionTaskSpec()
    raise NotImplementedError(
        "AIC MJLab env config skeleton is in place. Choose scene/entity "
        "composition strategy before constructing ManagerBasedRlEnvCfg."
    )
