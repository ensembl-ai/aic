from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint_path: Path
    entrance_topic: str
    entrance_timeout_s: float
    control_period_s: float
    max_policy_duration_s: float
    pose_command_frame: str
    insertion_axis_entrance: np.ndarray
    position_scale_m: float
    progress_scale_m: float
    force_scale_n: float
    torque_scale_nm: float
    action_translation_scale_m: float
    action_rotation_scale_rad: float
    stiffness: list[float]
    damping: list[float]
    wrench_feedback_gains: list[float]
    force_guard_n: float
    success_depth_m: float
    success_lateral_m: float
    success_angle_rad: float
    feedback_period_s: float


_REQUIRED_KEYS = {
    "checkpoint_path",
    "entrance_topic",
    "entrance_timeout_s",
    "control_period_s",
    "max_policy_duration_s",
    "pose_command_frame",
    "insertion_axis_entrance",
    "position_scale_m",
    "progress_scale_m",
    "force_scale_n",
    "torque_scale_nm",
    "action_translation_scale_m",
    "action_rotation_scale_rad",
    "stiffness",
    "damping",
    "wrench_feedback_gains",
    "force_guard_n",
    "success_depth_m",
    "success_lateral_m",
    "success_angle_rad",
    "feedback_period_s",
}


def _require_number(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if not isinstance(value, int | float):
        raise ValueError(f"'{key}' must be numeric.")
    return float(value)


def _require_vector(data: dict[str, Any], key: str, length: int) -> list[float]:
    value = data[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"'{key}' must be a list of length {length}.")
    vector = [float(item) for item in value]
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"'{key}' contains non-finite values.")
    return vector


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Insertion policy config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Insertion policy config must be a YAML mapping.")

    missing = sorted(_REQUIRED_KEYS - set(data))
    extra = sorted(set(data) - _REQUIRED_KEYS)
    if missing or extra:
        raise ValueError(f"Invalid insertion config. missing={missing}, extra={extra}")

    entrance_topic = str(data["entrance_topic"])
    pose_command_frame = str(data["pose_command_frame"])
    if not entrance_topic or not pose_command_frame:
        raise ValueError("Frame and topic names must be non-empty.")

    axis = np.asarray(_require_vector(data, "insertion_axis_entrance", 3), dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if not np.isclose(axis_norm, 1.0, atol=1.0e-6):
        raise ValueError("'insertion_axis_entrance' must be unit length.")

    stiffness = _require_vector(data, "stiffness", 6)
    damping = _require_vector(data, "damping", 6)
    wrench_feedback_gains = _require_vector(data, "wrench_feedback_gains", 6)
    if any(gain < 0.0 or gain > 0.95 for gain in wrench_feedback_gains):
        raise ValueError("'wrench_feedback_gains' must be in [0, 0.95].")

    positive_keys = [
        "entrance_timeout_s",
        "control_period_s",
        "max_policy_duration_s",
        "position_scale_m",
        "progress_scale_m",
        "force_scale_n",
        "torque_scale_nm",
        "action_translation_scale_m",
        "action_rotation_scale_rad",
        "force_guard_n",
        "success_depth_m",
        "success_lateral_m",
        "success_angle_rad",
        "feedback_period_s",
    ]
    numbers = {key: _require_number(data, key) for key in positive_keys}
    invalid = [key for key, value in numbers.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"Config values must be positive: {invalid}")

    return RuntimeConfig(
        checkpoint_path=Path(str(data["checkpoint_path"])),
        entrance_topic=entrance_topic,
        entrance_timeout_s=numbers["entrance_timeout_s"],
        control_period_s=numbers["control_period_s"],
        max_policy_duration_s=numbers["max_policy_duration_s"],
        pose_command_frame=pose_command_frame,
        insertion_axis_entrance=axis,
        position_scale_m=numbers["position_scale_m"],
        progress_scale_m=numbers["progress_scale_m"],
        force_scale_n=numbers["force_scale_n"],
        torque_scale_nm=numbers["torque_scale_nm"],
        action_translation_scale_m=numbers["action_translation_scale_m"],
        action_rotation_scale_rad=numbers["action_rotation_scale_rad"],
        stiffness=stiffness,
        damping=damping,
        wrench_feedback_gains=wrench_feedback_gains,
        force_guard_n=numbers["force_guard_n"],
        success_depth_m=numbers["success_depth_m"],
        success_lateral_m=numbers["success_lateral_m"],
        success_angle_rad=numbers["success_angle_rad"],
        feedback_period_s=numbers["feedback_period_s"],
    )
