"""Resolved six-joint arm mapping for the reduced AIC MJCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class ArmJoints:
    """Ordered MuJoCo addresses for the six controlled arm joints."""

    names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_addresses: np.ndarray
    ranges: np.ndarray

    @classmethod
    def resolve(cls, model: mujoco.MjModel, names: dict[str, Any]) -> "ArmJoints":
        """Resolve and validate the configured joint-to-actuator mapping."""

        joint_names = tuple(names["joints"])
        actuator_names = tuple(names["actuators"])
        joint_ids = np.asarray(
            [
                required_model_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=np.int32,
        )
        if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("All six controlled joints must be scalar hinge joints")

        actuator_ids = np.asarray(
            [
                required_model_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in actuator_names
            ],
            dtype=np.int32,
        )
        for actuator_id, joint_id in zip(actuator_ids, joint_ids, strict=True):
            if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("Every configured actuator must use joint transmission")
            if int(model.actuator_trnid[actuator_id, 0]) != int(joint_id):
                raise ValueError("Each configured actuator must drive its corresponding joint")

        return cls(
            names=joint_names,
            joint_ids=joint_ids,
            qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
            dof_addresses=np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
            actuator_addresses=actuator_ids,
            ranges=np.asarray(model.jnt_range[joint_ids], dtype=np.float64).copy(),
        )

    @property
    def count(self) -> int:
        """Return the number of controlled arm joints."""

        return len(self.names)


def required_model_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    """Return a named model ID or fail immediately when the name is absent."""

    identifier = mujoco.mj_name2id(model, object_type, name)
    if identifier < 0:
        raise ValueError(f"Generated MJCF is missing required name: {name}")
    return int(identifier)
