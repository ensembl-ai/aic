"""Strict two-layer configuration for the AIC MuJoCo-Warp foundation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PACKAGE_DIR / "configs" / "base.json"
DEFAULT_RUN_CONFIG = PACKAGE_DIR / "configs" / "run.json"
DEFAULT_COLLECTION_CONFIG = PACKAGE_DIR / "configs" / "collect.json"


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


CONFIG_SCHEMA: dict[str, Any] = {
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
        "realtime": bool,
        "grid_spacing": list,
        "initial_camera_position": list,
        "initial_camera_look_at": list,
    },
}


COLLECTION_SCHEMA = deep_merge(
    CONFIG_SCHEMA,
    {
        "expert": {
            "controlled_body": str,
            "target_body": str,
            "goal_offset_position": list,
            "goal_offset_rotation_matrix": list,
            "control_hz": int,
            "translation_gain": float,
            "rotation_gain": float,
            "maximum_translation_step": float,
            "maximum_rotation_step": float,
            "maximum_joint_step": list,
            "dls_damping": float,
            "joint_limit_margin": float,
            "position_tolerance": float,
            "orientation_tolerance": float,
            "success_consecutive_steps": int,
            "maximum_episode_steps": int,
        },
        "dataset": {
            "output_directory": str,
            "resume": bool,
            "instruction": str,
            "image_width": int,
            "image_height": int,
            "video_codec": str,
            "video_pixel_format": str,
            "video_crf": int,
            "keep_failed_trajectories": bool,
            "maximum_failed_trajectories": int,
            "splits": {
                "train": int,
                "validation": int,
                "test": int,
            },
        },
    },
)


def read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object without supplying defaults.

    Args:
        path: JSON file to read.

    Returns:
        Parsed top-level object.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        TypeError: If the JSON root is not an object.
        ValueError: If the file is not valid JSON.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be an object: {path}")
    return value


def matches(value: Any, expected: type | tuple[type, ...]) -> bool:
    """Check a configuration value against the strict schema type rules.

    Args:
        value: Parsed configuration value.
        expected: Accepted type or types.

    Returns:
        Whether the value satisfies the schema rule.
    """

    if isinstance(expected, tuple):
        return isinstance(value, expected)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_shape(value: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    """Validate exact recursive configuration keys and value types.

    Args:
        value: Configuration subtree to validate.
        schema: Exact schema for that subtree.
        path: Dotted path used in validation errors.

    Raises:
        KeyError: If a required key is missing or an unknown key is present.
        TypeError: If a value does not have the required type.
    """

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
            validate_shape(item, expected, item_path)
        elif not matches(item, expected):
            expected_name = (
                " or ".join(item.__name__ for item in expected)
                if isinstance(expected, tuple)
                else expected.__name__
            )
            raise TypeError(f"{item_path} must be {expected_name}")


def numbers(config: dict[str, Any], path: str, length: int) -> list[float]:
    """Read a finite fixed-length numeric vector from a configuration.

    Args:
        config: Complete merged configuration.
        path: Dotted path to the vector.
        length: Exact required vector length.

    Returns:
        Values converted to floats.

    Raises:
        TypeError: If any vector member is not numeric.
        ValueError: If length or finiteness validation fails.
    """

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


def validate_config(
    config: dict[str, Any], schema: dict[str, Any] = CONFIG_SCHEMA
) -> None:
    """Validate the complete runtime configuration without fallbacks.

    Args:
        config: Deep-merged configuration to validate.
        schema: Exact accepted schema, normally ``CONFIG_SCHEMA``.

    Raises:
        KeyError: If required or unknown keys are present.
        TypeError: If a configured value has the wrong type.
        ValueError: If a configured value violates a semantic constraint.
    """

    validate_shape(config, schema, "config")

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
    }
    if "visualization" in config:
        vector_paths.update(
            {
                "visualization.grid_spacing": 2,
                "visualization.initial_camera_position": 3,
                "visualization.initial_camera_look_at": 3,
            }
        )
    vectors = {path: numbers(config, path, size) for path, size in vector_paths.items()}

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
    if "visualization" in config:
        visualization = config["visualization"]
        if not visualization["host"]:
            raise ValueError("visualization.host cannot be empty")
        if not 1 <= visualization["port"] <= 65535:
            raise ValueError("visualization.port must be in [1, 65535]")
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
        if isinstance(visual_env_ids, list):
            env_ids = visual_env_ids
            if any(
                type(x) is not int or x < 0 or x >= runtime["num_envs"]
                for x in env_ids
            ):
                raise ValueError(
                    "visualization.env_ids contains an invalid environment index"
                )
            if len(set(env_ids)) != len(env_ids):
                raise ValueError("visualization.env_ids contains duplicates")
            if visualization["enabled"] and not env_ids:
                raise ValueError(
                    "visualization.env_ids cannot be empty when enabled"
                )


def validate_collection_config(config: dict[str, Any]) -> None:
    """Validate every expert and dataset value without implicit defaults."""

    validate_config(config, COLLECTION_SCHEMA)
    expert = config["expert"]
    dataset = config["dataset"]

    if not expert["controlled_body"] or not expert["target_body"]:
        raise ValueError("expert controlled and target body names cannot be empty")
    expert_vectors = {
        "expert.goal_offset_position": 3,
        "expert.goal_offset_rotation_matrix": 9,
        "expert.maximum_joint_step": 6,
    }
    vectors = {
        path: numbers(config, path, length)
        for path, length in expert_vectors.items()
    }
    rotation = np.asarray(
        vectors["expert.goal_offset_rotation_matrix"], dtype=np.float64
    ).reshape(3, 3)
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1.0e-6,
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(
            "expert.goal_offset_rotation_matrix must be a valid SO(3) matrix"
        )
    if any(value <= 0.0 for value in vectors["expert.maximum_joint_step"]):
        raise ValueError("Every expert.maximum_joint_step value must be positive")

    positive_expert_scalars = (
        "translation_gain",
        "rotation_gain",
        "maximum_translation_step",
        "maximum_rotation_step",
        "dls_damping",
        "position_tolerance",
        "orientation_tolerance",
    )
    for key in positive_expert_scalars:
        if not math.isfinite(expert[key]) or expert[key] <= 0.0:
            raise ValueError(f"expert.{key} must be finite and positive")
    if not math.isfinite(expert["joint_limit_margin"]) or expert[
        "joint_limit_margin"
    ] < 0.0:
        raise ValueError("expert.joint_limit_margin must be finite and nonnegative")
    if expert["control_hz"] != config["cameras"]["fps"]:
        raise ValueError(
            "expert.control_hz must equal cameras.fps so every labeled action has RGB"
        )
    if expert["success_consecutive_steps"] <= 0:
        raise ValueError("expert.success_consecutive_steps must be positive")
    if expert["maximum_episode_steps"] < expert["success_consecutive_steps"]:
        raise ValueError(
            "expert.maximum_episode_steps must cover success_consecutive_steps"
        )

    if not dataset["output_directory"]:
        raise ValueError("dataset.output_directory cannot be empty")
    if not dataset["instruction"]:
        raise ValueError("dataset.instruction cannot be empty")
    if not dataset["video_codec"] or not dataset["video_pixel_format"]:
        raise ValueError("Dataset video codec and pixel format cannot be empty")
    for key in ("image_width", "image_height"):
        if dataset[key] <= 0 or dataset[key] % 2:
            raise ValueError(f"dataset.{key} must be a positive even integer")
    if dataset["image_width"] > config["cameras"]["width"] or dataset[
        "image_height"
    ] > config["cameras"]["height"]:
        raise ValueError("Dataset images cannot exceed the native camera resolution")
    if not 0 <= dataset["video_crf"] <= 51:
        raise ValueError("dataset.video_crf must be in [0, 51]")
    if dataset["maximum_failed_trajectories"] < 0:
        raise ValueError("dataset.maximum_failed_trajectories cannot be negative")
    split_counts = dataset["splits"]
    if any(count < 0 for count in split_counts.values()):
        raise ValueError("Dataset split trajectory counts cannot be negative")
    if sum(split_counts.values()) <= 0:
        raise ValueError("At least one dataset trajectory must be requested")


def load_config(
    base_path: str | Path = DEFAULT_BASE_CONFIG,
    run_path: str | Path = DEFAULT_RUN_CONFIG,
) -> dict[str, Any]:
    """Deep-merge and strictly validate the canonical base and run files."""

    base = Path(base_path).expanduser().resolve()
    run = Path(run_path).expanduser().resolve()
    config = deep_merge(read_object(base), read_object(run))
    validate_config(config)
    for key in ("source_robot", "source_world", "output"):
        config["scene"][key] = str((base.parent / config["scene"][key]).resolve())
    return config


def load_collection_config(
    base_path: str | Path = DEFAULT_BASE_CONFIG,
    collection_path: str | Path = DEFAULT_COLLECTION_CONFIG,
) -> dict[str, Any]:
    """Load the strict base plus synthetic-collection execution overlay."""

    base = Path(base_path).expanduser().resolve()
    collection = Path(collection_path).expanduser().resolve()
    config = deep_merge(read_object(base), read_object(collection))
    validate_collection_config(config)
    for key in ("source_robot", "source_world", "output"):
        config["scene"][key] = str((base.parent / config["scene"][key]).resolve())
    config["dataset"]["output_directory"] = str(
        (collection.parent / config["dataset"]["output_directory"]).resolve()
    )
    return config
