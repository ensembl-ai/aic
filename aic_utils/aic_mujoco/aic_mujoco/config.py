"""Strict two-layer configuration for the AIC MuJoCo-Warp foundation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PACKAGE_DIR / "configs" / "base.json"
DEFAULT_RUN_CONFIG = PACKAGE_DIR / "configs" / "run.json"


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input.

    Nested objects are combined. Every non-object value, including arrays,
    replaces the corresponding base value in full.
    """

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_SCHEMA: dict[str, Any] = {
    "scene": {
        "model_name": str,
        "source_robot": str,
        "source_world": str,
        "output": str,
        "gripper_fixed_position": float,
        "names": {
            "robot_root_body": str,
            "tool_body": str,
            "left_finger_body": str,
            "right_finger_body": str,
            "left_finger_joint": str,
            "right_finger_joint": str,
            "sfp_source_body": str,
            "lc_source_body": str,
            "board_source_body": str,
            "nic_source_body": str,
            "board_body": str,
            "nic_body": str,
            "light": str,
            "joints": list,
            "actuators": list,
            "cameras": {"center": str, "left": str, "right": str},
            "sensors": {"force": str, "torque": str},
        },
    },
    "physics": {
        "device": str,
        "timestep": float,
        "integrator": str,
        "solver": str,
        "iterations": int,
        "tolerance": float,
        "gravity": list,
        "nconmax": int,
        "njmax": int,
        "graph_capture": bool,
    },
    "runtime": {"num_envs": int, "seed": int},
    "control": {
        "home": list,
        "reset_perturbation_lower": list,
        "reset_perturbation_upper": list,
        "stiffness": list,
        "damping": list,
        "torque_limits": list,
    },
    "domain_randomization": {
        "board_position_lower": list,
        "board_position_upper": list,
        "board_yaw_deviation_lower": float,
        "board_yaw_deviation_upper": float,
        "nic_rail_indices": list,
        "nic_rail_x_base": float,
        "nic_rail_y_by_index": list,
        "nic_rail_z": float,
        "nic_translation_lower": float,
        "nic_translation_upper": float,
    },
    "sensors": {
        "physics_sample_hz": int,
        "publication_hz": int,
        "tare_settle_steps": int,
        "tare_sample_count": int,
    },
    "cameras": {
        "width": int,
        "height": int,
        "fps": int,
        "use_textures": bool,
        "use_shadows": bool,
    },
    "visualization": {
        "enabled": bool,
        "host": str,
        "port": int,
        "env_ids": (list, str),
        "jpeg_quality": int,
        "realtime": bool,
        "grid_columns": int,
        "grid_spacing": list,
        "initial_camera_position": list,
        "initial_camera_look_at": list,
    },
    "recording": {
        "enabled": bool,
        "output_directory": str,
        "env_ids": list,
        "codec": str,
    },
}


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be an object: {path}")
    return value


def _matches(value: Any, expected: type | tuple[type, ...]) -> bool:
    if isinstance(expected, tuple):
        return isinstance(value, expected)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _validate_shape(value: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    missing = schema.keys() - value.keys()
    extra = value.keys() - schema.keys()
    if missing:
        raise KeyError(f"Missing configuration keys at {path}: {sorted(missing)}")
    if extra:
        raise KeyError(f"Unknown configuration keys at {path}: {sorted(extra)}")
    for key, expected in schema.items():
        item = value[key]
        item_path = f"{path}.{key}"
        if isinstance(expected, dict):
            if not isinstance(item, dict):
                raise TypeError(f"{item_path} must be an object")
            _validate_shape(item, expected, item_path)
        elif not _matches(item, expected):
            expected_name = (
                " or ".join(item.__name__ for item in expected)
                if isinstance(expected, tuple)
                else expected.__name__
            )
            raise TypeError(f"{item_path} must be {expected_name}")


def _numbers(config: dict[str, Any], path: str, length: int) -> list[float]:
    value: Any = config
    for key in path.split("."):
        value = value[key]
    if len(value) != length:
        raise ValueError(f"{path} must contain exactly {length} values")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in value):
        raise TypeError(f"Every value in {path} must be numeric")
    if any(not math.isfinite(float(x)) for x in value):
        raise ValueError(f"Every value in {path} must be finite")
    return [float(x) for x in value]


def _validate(config: dict[str, Any]) -> None:
    _validate_shape(config, _SCHEMA, "config")

    scene = config["scene"]
    if not scene["model_name"]:
        raise ValueError("scene.model_name cannot be empty")
    for key in ("source_robot", "source_world", "output"):
        if not scene[key]:
            raise ValueError(f"scene.{key} cannot be empty")
    if not math.isfinite(scene["gripper_fixed_position"]):
        raise ValueError("scene.gripper_fixed_position must be finite")
    names = config["scene"]["names"]
    scalar_names = [
        value
        for key, value in names.items()
        if key not in ("joints", "actuators", "cameras", "sensors")
    ]
    if any(not value for value in scalar_names):
        raise ValueError("Every configured MJCF name must be non-empty")
    for section in ("cameras", "sensors"):
        values = list(names[section].values())
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"scene.names.{section} must contain unique non-empty names")
    for key in ("joints", "actuators"):
        values = names[key]
        if len(values) != 6 or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"scene.names.{key} must contain six non-empty names")
        if len(set(values)) != 6:
            raise ValueError(f"scene.names.{key} contains duplicate names")

    vector_paths = {
        "physics.gravity": 3,
        "control.home": 6,
        "control.reset_perturbation_lower": 6,
        "control.reset_perturbation_upper": 6,
        "control.stiffness": 6,
        "control.damping": 6,
        "control.torque_limits": 6,
        "domain_randomization.board_position_lower": 3,
        "domain_randomization.board_position_upper": 3,
        "domain_randomization.nic_rail_y_by_index": 5,
        "visualization.grid_spacing": 2,
        "visualization.initial_camera_position": 3,
        "visualization.initial_camera_look_at": 3,
    }
    vectors = {path: _numbers(config, path, size) for path, size in vector_paths.items()}

    for lower_path, upper_path in (
        ("control.reset_perturbation_lower", "control.reset_perturbation_upper"),
        (
            "domain_randomization.board_position_lower",
            "domain_randomization.board_position_upper",
        ),
    ):
        if any(a > b for a, b in zip(vectors[lower_path], vectors[upper_path], strict=True)):
            raise ValueError(f"{lower_path} must not exceed {upper_path}")

    if any(x < 0.0 for x in vectors["control.stiffness"]):
        raise ValueError("control.stiffness cannot be negative")
    if any(x < 0.0 for x in vectors["control.damping"]):
        raise ValueError("control.damping cannot be negative")
    if any(x <= 0.0 for x in vectors["control.torque_limits"]):
        raise ValueError("control.torque_limits must be positive")

    physics = config["physics"]
    if physics["device"] != "cpu" and not physics["device"].startswith("cuda"):
        raise ValueError("physics.device must be 'cpu' or a CUDA device such as 'cuda:0'")
    if physics["integrator"] != "implicitfast":
        raise ValueError("physics.integrator must be 'implicitfast' for this foundation")
    if physics["solver"] != "Newton":
        raise ValueError("physics.solver must be 'Newton' for this foundation")
    for key in ("timestep", "tolerance"):
        if not math.isfinite(physics[key]) or physics[key] <= 0.0:
            raise ValueError(f"physics.{key} must be positive")
    for key in ("iterations", "nconmax", "njmax"):
        if physics[key] <= 0:
            raise ValueError(f"physics.{key} must be positive")

    runtime = config["runtime"]
    if runtime["num_envs"] <= 0:
        raise ValueError("runtime.num_envs must be positive")
    if runtime["seed"] < 0:
        raise ValueError("runtime.seed cannot be negative")

    randomization = config["domain_randomization"]
    scalar_randomization = (
        "board_yaw_deviation_lower",
        "board_yaw_deviation_upper",
        "nic_rail_x_base",
        "nic_rail_z",
        "nic_translation_lower",
        "nic_translation_upper",
    )
    if any(not math.isfinite(randomization[key]) for key in scalar_randomization):
        raise ValueError("Domain-randomization scalar values must be finite")
    if randomization["board_yaw_deviation_lower"] > randomization["board_yaw_deviation_upper"]:
        raise ValueError("board yaw deviation lower bound exceeds its upper bound")
    if randomization["nic_translation_lower"] > randomization["nic_translation_upper"]:
        raise ValueError("NIC translation lower bound exceeds its upper bound")
    rails = randomization["nic_rail_indices"]
    rail_count = len(vectors["domain_randomization.nic_rail_y_by_index"])
    if not rails or any(type(x) is not int or x < 0 or x >= rail_count for x in rails):
        raise ValueError(
            f"nic_rail_indices must contain integers in [0, {rail_count - 1}]"
        )
    if len(set(rails)) != len(rails):
        raise ValueError("nic_rail_indices contains duplicates")

    step_hz = round(1.0 / physics["timestep"])
    if not math.isclose(step_hz * physics["timestep"], 1.0, abs_tol=1e-12):
        raise ValueError("physics.timestep must divide one second exactly")
    rates = {
        "cameras.fps": config["cameras"]["fps"],
        "sensors.physics_sample_hz": config["sensors"]["physics_sample_hz"],
        "sensors.publication_hz": config["sensors"]["publication_hz"],
    }
    for path, rate in rates.items():
        if rate <= 0 or step_hz % rate:
            raise ValueError(f"{path} must be positive and divide the physics rate {step_hz}")
    if config["sensors"]["tare_settle_steps"] < 0:
        raise ValueError("sensors.tare_settle_steps cannot be negative")
    if config["sensors"]["tare_sample_count"] <= 0:
        raise ValueError("sensors.tare_sample_count must be positive")

    cameras = config["cameras"]
    if cameras["width"] <= 0 or cameras["height"] <= 0:
        raise ValueError("Camera dimensions must be positive")
    visualization = config["visualization"]
    if not visualization["host"]:
        raise ValueError("visualization.host cannot be empty")
    if not 1 <= visualization["port"] <= 65535:
        raise ValueError("visualization.port must be in [1, 65535]")
    if not 1 <= visualization["jpeg_quality"] <= 100:
        raise ValueError("visualization.jpeg_quality must be in [1, 100]")
    if visualization["grid_columns"] <= 0:
        raise ValueError("visualization.grid_columns must be positive")
    if any(value <= 0.0 for value in vectors["visualization.grid_spacing"]):
        raise ValueError("visualization.grid_spacing values must be positive")
    if vectors["visualization.initial_camera_position"] == vectors[
        "visualization.initial_camera_look_at"
    ]:
        raise ValueError(
            "visualization.initial_camera_position and initial_camera_look_at must differ"
        )
    visual_env_ids = visualization["env_ids"]
    if isinstance(visual_env_ids, str) and visual_env_ids != "all":
        raise ValueError("visualization.env_ids string value must be 'all'")
    list_selections = [("recording", config["recording"]["env_ids"])]
    if isinstance(visual_env_ids, list):
        list_selections.append(("visualization", visual_env_ids))
    for section, env_ids in list_selections:
        if any(type(x) is not int or x < 0 or x >= runtime["num_envs"] for x in env_ids):
            raise ValueError(f"{section}.env_ids contains an invalid environment index")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"{section}.env_ids contains duplicates")
        if config[section]["enabled"] and not env_ids:
            raise ValueError(f"{section}.env_ids cannot be empty when enabled")
    if not config["recording"]["output_directory"]:
        raise ValueError("recording.output_directory cannot be empty")
    if not config["recording"]["codec"]:
        raise ValueError("recording.codec cannot be empty")


def load_config(
    base_path: str | Path = DEFAULT_BASE_CONFIG,
    run_path: str | Path = DEFAULT_RUN_CONFIG,
) -> dict[str, Any]:
    """Deep-merge and strictly validate the canonical base and run files."""

    base = Path(base_path).expanduser().resolve()
    run = Path(run_path).expanduser().resolve()
    config = deep_merge(_read_object(base), _read_object(run))
    _validate(config)
    for key in ("source_robot", "source_world", "output"):
        config["scene"][key] = str((base.parent / config["scene"][key]).resolve())
    config["recording"]["output_directory"] = str(
        (base.parent / config["recording"]["output_directory"]).resolve()
    )
    return config
