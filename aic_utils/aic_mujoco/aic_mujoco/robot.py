"""Validated robot and named-scene interface for the reduced AIC model."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np

from aic_mujoco.joints import ArmJoints, required_model_id


class AICRobot:
    """Resolve the arm, actuators, cameras, wrench sensors, and fixtures."""

    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]):
        names = config["scene"]["names"]
        self.joints = ArmJoints.resolve(model, names)
        self._validate_control(model, config)

        force_id = required_model_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, names["sensors"]["force"]
        )
        torque_id = required_model_id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, names["sensors"]["torque"]
        )
        self.force_sensor_dimension = int(model.sensor_dim[force_id])
        self.torque_sensor_dimension = int(model.sensor_dim[torque_id])
        if self.force_sensor_dimension != 3 or self.torque_sensor_dimension != 3:
            raise ValueError("Configured force and torque sensors must each have dimension three")
        self.force_sensor_address = int(model.sensor_adr[force_id])
        self.torque_sensor_address = int(model.sensor_adr[torque_id])
        self.wrench_dimension = (
            self.force_sensor_dimension + self.torque_sensor_dimension
        )

        camera_ids = {
            key: required_model_id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for key, name in names["cameras"].items()
        }
        if sorted(camera_ids.values()) != list(range(3)):
            raise ValueError("The reduced scene must contain exactly the three configured cameras")
        self.camera_ids: Mapping[str, int] = MappingProxyType(camera_ids)

        self.board_mocap_id = self._mocap_id(model, names["board_body"])
        self.nic_mocap_id = self._mocap_id(model, names["nic_body"])

    def _validate_control(self, model: mujoco.MjModel, config: dict[str, Any]) -> None:
        control = config["control"]
        torque_limits = np.asarray(control["torque_limits"])
        expected_ctrlrange = np.column_stack((-torque_limits, torque_limits))
        actuators = self.joints.actuator_addresses
        if not np.all(model.actuator_ctrllimited[actuators]) or not np.allclose(
            model.actuator_ctrlrange[actuators], expected_ctrlrange, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Generated MJCF actuator limits do not match configuration")

        reset_lower = np.asarray(control["home"]) + np.asarray(
            control["reset_perturbation_lower"]
        )
        reset_upper = np.asarray(control["home"]) + np.asarray(
            control["reset_perturbation_upper"]
        )
        if np.any(reset_lower < self.joints.ranges[:, 0]) or np.any(
            reset_upper > self.joints.ranges[:, 1]
        ):
            raise ValueError("Configured joint reset envelope exceeds an MJCF joint range")

    @staticmethod
    def _mocap_id(model: mujoco.MjModel, body_name: str) -> int:
        body_id = required_model_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Generated MJCF body is not mocap-controlled: {body_name}")
        return mocap_id
