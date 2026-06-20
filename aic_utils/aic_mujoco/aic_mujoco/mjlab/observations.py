"""Observation helpers for AIC MuJoCo policy-training experiments.

These helpers define the low-dimensional signals used by the AIC insertion
policy stack:

  joint state
  TCP/site pose
  reset-zeroed force/torque
  camera health diagnostics

Task-specific observation formulas should be explicit and testable here before
they are wired as MJLab observation terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class WrenchObservation:
    force: np.ndarray | None
    torque: np.ndarray | None
    raw_force: np.ndarray | None
    raw_torque: np.ndarray | None
    force_bias: np.ndarray | None
    torque_bias: np.ndarray | None


@dataclass
class ContactObservation:
    ncon: int
    max_penetration: float


def read_sensor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_name: str,
) -> np.ndarray | None:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        return None

    start = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[start : start + dim], dtype=float).copy()


def zero_sensor(
    values: np.ndarray | None,
    bias: np.ndarray | None,
) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if bias is None:
        return values.copy()
    return values - np.asarray(bias, dtype=float)


def force_torque_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force_sensor: str = "AtiForceTorqueSensor_force",
    torque_sensor: str = "AtiForceTorqueSensor_torque",
    force_bias: np.ndarray | None = None,
    torque_bias: np.ndarray | None = None,
) -> WrenchObservation:
    raw_force = read_sensor(model, data, force_sensor)
    raw_torque = read_sensor(model, data, torque_sensor)
    return WrenchObservation(
        force=zero_sensor(raw_force, force_bias),
        torque=zero_sensor(raw_torque, torque_bias),
        raw_force=raw_force,
        raw_torque=raw_torque,
        force_bias=force_bias,
        torque_bias=torque_bias,
    )


def contact_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> ContactObservation:
    """Return the maximum active MuJoCo contact penetration.

    Visual overlap is not contact. This reads MuJoCo's active contact array
    after ``mj_step``/``mj_forward`` and returns ``max(0, -contact.dist)``.
    """

    distances: list[float] = []
    ncon = int(getattr(data, "ncon", 0))
    contacts = getattr(data, "contact")

    for contact_id in range(ncon):
        contact = contacts[contact_id]
        distances.append(float(contact.dist))

    min_distance = None if not distances else min(distances)
    max_penetration = 0.0 if min_distance is None else max(0.0, -min_distance)
    return ContactObservation(
        ncon=ncon,
        max_penetration=max_penetration,
    )


class CameraHealthChecker:
    """Lightweight camera render sanity check for demo/debug runs."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_names: list[str],
        width: int = 160,
        height: int = 120,
        min_std: float = 1.0,
    ):
        self.model = model
        self.camera_names = camera_names
        self.min_std = min_std
        self.renderer: mujoco.Renderer | None = None
        self.error: str | None = None

        missing = [
            name
            for name in camera_names
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0
        ]
        if missing:
            self.error = "missing camera(s): " + ", ".join(missing)
            return

        try:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:  # noqa: BLE001
            self.error = f"renderer init failed: {exc}"

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def summary(self, data: mujoco.MjData) -> str:
        if self.error is not None:
            return self.error
        assert self.renderer is not None

        parts: list[str] = []
        for name in self.camera_names:
            try:
                self.renderer.update_scene(data, camera=name)
                image = self.renderer.render()
                finite = bool(np.isfinite(image).all())
                mean = float(np.mean(image))
                std = float(np.std(image))
                ok = finite and image.size > 0 and std >= self.min_std
                status = "ok" if ok else "blank"
                parts.append(f"{name}:{status}(mean={mean:.1f},std={std:.1f})")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"{name}:error({exc})")
        return "; ".join(parts)


def joint_state_observation(env, asset_cfg=None):
    """Return joint-state observations from an installed MJLab environment."""
    raise NotImplementedError("Wire joint-state observation to the installed MJLab API.")


def tcp_pose_observation(env, asset_cfg=None, frame_name: str = "gripper_tcp"):
    """Return TCP pose observations from an installed MJLab environment."""
    raise NotImplementedError("Wire TCP pose observation to the installed MJLab API.")


def zeroed_force_torque_observation(
    env,
    sensor_name: str = "AtiForceTorqueSensor",
    baseline_name: str = "ati_ft_zero",
):
    """Return reset-zeroed force/torque observations from MJLab sensor state."""
    raise NotImplementedError(
        "Wire zeroed force/torque observation to the installed MJLab API."
    )


def task_geometry_observation(env, asset_cfg=None):
    """Return task geometry observations from an installed MJLab environment."""
    raise NotImplementedError("Wire task geometry observation to the installed MJLab API.")


def camera_health_observation(env, camera_names: tuple[str, ...]):
    """Return camera health observations from an installed MJLab environment."""
    raise NotImplementedError(
        "Wire camera observation to the installed MJLab API."
    )
