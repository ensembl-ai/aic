"""Prototype observations for AIC MuJoCo policy-training experiments.

These helpers are the plain MuJoCo version of the observations we expect to
carry into MJLab:

  joint state
  TCP/site pose
  reset-zeroed force/torque
  camera health diagnostics

They are intentionally conservative. Reward/task-specific observation formulas
still belong in explicit task design, not hidden inside this module.
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


def vector_summary(values: np.ndarray | None) -> str:
    if values is None:
        return "missing"
    values = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(values))
    return "norm={:.3f} xyz=[{}]".format(
        norm,
        ", ".join(f"{x:+.3f}" for x in values),
    )


def camera_names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
