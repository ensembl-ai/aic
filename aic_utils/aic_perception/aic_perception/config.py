"""Strict two-layer configuration for the AIC multiview scene estimator.

Every entry point loads ``configs/base.json`` plus exactly one overlay
(``collect.json``, ``train.json``, or ``infer.json``). Unknown or missing keys
are errors; nothing falls back to an implicit default.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / "configs"

TARGET_SCHEMA = {"id": str, "type": str, "frame": str}

BASE_SCHEMA: dict[str, Any] = {
    "schema": {
        "name": str,
        "base_frame": str,
        "cameras": list,
        "target_types": list,
        "targets": list,
    },
    "images": {"width": int, "height": int},
    "dataset": {"directory": str, "validation_fraction": float},
}

RANGE_SCHEMA = {"count": list, "translation": list, "yaw_deg": list}

OVERLAY_SCHEMAS: dict[str, dict[str, Any]] = {
    "collect": {
        "collection": {
            "scenes": int,
            "seed": int,
            "workers": (int, str),
            "jpeg_quality": int,
            "minimum_image_change": float,
            "task_board_xacro": str,
            "progress_interval_s": float,
        },
        "docker": {
            "image": str,
            "container_prefix": str,
            "rendering": str,
            "cpus_per_worker": float,
            "memory_per_worker_gb": float,
            "max_gpu_workers": int,
            "launch_interval_s": float,
            "max_restarts": int,
        },
        "timing": {
            "bringup_timeout_s": float,
            "service_timeout_s": float,
            "observation_timeout_s": float,
            "settle_time_s": float,
        },
        "randomization": {
            "board_x": list,
            "board_y": list,
            "board_z": float,
            "board_yaw": list,
            "nic_cards": RANGE_SCHEMA,
            "sc_ports": {**RANGE_SCHEMA, "minimum_spacing": float},
            "mounts": {"probability": float, "translation": list, "yaw_deg": list},
        },
    },
    "train": {
        "curation": {"preview_count": int},
        "run": {"name": str, "directory": str, "seed": int, "device": str},
        "model": {
            "pretrained_backbone": bool,
            "d_model": int,
            "heads": int,
            "encoder_layers": int,
            "decoder_layers": int,
            "feedforward": int,
            "dropout": float,
        },
        "optimization": {
            "epochs": int,
            "batch_size": int,
            "learning_rate": float,
            "backbone_learning_rate": float,
            "weight_decay": float,
            "warmup_steps": int,
            "gradient_clip": float,
            "bf16": bool,
            "num_workers": int,
        },
        "loss": {
            "presence": float,
            "translation": float,
            "rotation": float,
            "confidence": float,
            "confidence_translation_m": float,
            "confidence_rotation_deg": float,
        },
        "augmentation": {"brightness": float, "contrast": float, "saturation": float},
        "logging": {"wandb_project": str, "wandb_mode": str, "log_every_steps": int},
    },
    "infer": {
        "inference": {
            "checkpoint": str,
            "device": str,
            "observation_topic": str,
            "estimate_topic": str,
            "rate_hz": float,
            "presence_threshold": float,
            "tf_prefix": str,
        },
        "far_approach": {
            "estimate_count": int,
            "estimate_timeout_s": float,
            "hover_height_m": float,
            "approach_duration_s": float,
            "settle_time_s": float,
        },
    },
}

RENDERING_MODES = ("wsl_gpu", "nvidia_gpu", "cpu")
WANDB_MODES = ("online", "offline", "disabled")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of two objects without mutating either.

    Args:
        base: Lower-priority configuration.
        overlay: Higher-priority configuration; non-object values replace.

    Returns:
        Merged configuration.
    """

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    Args:
        path: JSON file.

    Returns:
        Parsed top-level object.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the root is not an object.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be an object: {path}")
    return value


def matches(value: Any, expected: type | tuple[type, ...]) -> bool:
    """Check a value against a schema type, treating ints as valid floats.

    Args:
        value: Parsed value.
        expected: Accepted type or types.

    Returns:
        Whether the value satisfies the rule.
    """

    if isinstance(expected, tuple):
        return any(matches(value, item) for item in expected)
    if isinstance(value, bool):
        return expected is bool
    if expected is float:
        return isinstance(value, (int, float))
    return isinstance(value, expected)


def validate_shape(value: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    """Validate exact keys and value types recursively.

    Args:
        value: Configuration subtree.
        schema: Expected subtree.
        path: Dotted path used in error messages.

    Raises:
        KeyError: On missing or unknown keys.
        TypeError: On wrongly typed values.
    """

    missing = schema.keys() - value.keys()
    extra = value.keys() - schema.keys()
    if missing:
        raise KeyError(f"Missing configuration keys at {path}: {sorted(missing)}")
    if extra:
        raise KeyError(f"Unknown configuration keys at {path}: {sorted(extra)}")
    for key, expected in schema.items():
        item_path = f"{path}.{key}"
        if isinstance(expected, dict):
            if not isinstance(value[key], dict):
                raise TypeError(f"{item_path} must be an object")
            validate_shape(value[key], expected, item_path)
        elif not matches(value[key], expected):
            raise TypeError(f"{item_path} has the wrong type")


def interval(value: list[Any], path: str) -> tuple[float, float]:
    """Validate a finite ``[lower, upper]`` interval.

    Args:
        value: Configured two-element list.
        path: Dotted path used in error messages.

    Returns:
        Interval bounds.

    Raises:
        ValueError: If the interval is malformed.
    """

    if len(value) != 2 or not all(matches(x, float) for x in value):
        raise ValueError(f"{path} must be a numeric [lower, upper] pair")
    lower, upper = float(value[0]), float(value[1])
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        raise ValueError(f"{path} must be finite with lower <= upper")
    return lower, upper


def validate_base(config: dict[str, Any]) -> None:
    """Validate the board schema, image size, and dataset location.

    Args:
        config: Merged configuration.

    Raises:
        ValueError: On semantic violations.
    """

    schema = config["schema"]
    if not schema["name"] or not schema["base_frame"]:
        raise ValueError("schema.name and schema.base_frame cannot be empty")
    if schema["cameras"] != ["left", "center", "right"]:
        raise ValueError("schema.cameras must be the three AIC cameras in order")
    types = schema["target_types"]
    if not types or len(set(types)) != len(types):
        raise ValueError("schema.target_types must be unique and non-empty")
    targets = schema["targets"]
    if not targets:
        raise ValueError("schema.targets cannot be empty")
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError(f"schema.targets[{index}] must be an object")
        validate_shape(target, TARGET_SCHEMA, f"schema.targets[{index}]")
        if target["type"] not in types:
            raise ValueError(f"schema.targets[{index}].type is not in target_types")
    for key in ("id", "frame"):
        values = [target[key] for target in targets]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"schema target {key}s must be unique and non-empty")
    if config["images"]["width"] <= 0 or config["images"]["height"] <= 0:
        raise ValueError("images.width and images.height must be positive")
    if not 0.0 < config["dataset"]["validation_fraction"] < 1.0:
        raise ValueError("dataset.validation_fraction must be in (0, 1)")


def validate_collect(config: dict[str, Any]) -> None:
    """Validate collection, container, timing, and randomization values.

    Args:
        config: Merged collection configuration.

    Raises:
        ValueError: On semantic violations.
    """

    collection = config["collection"]
    if collection["scenes"] <= 0:
        raise ValueError("collection.scenes must be positive")
    workers = collection["workers"]
    if isinstance(workers, str) and workers != "auto":
        raise ValueError("collection.workers must be a positive integer or 'auto'")
    if isinstance(workers, int) and workers <= 0:
        raise ValueError("collection.workers must be positive")
    if not 1 <= collection["jpeg_quality"] <= 100:
        raise ValueError("collection.jpeg_quality must be in [1, 100]")
    if collection["minimum_image_change"] <= 0.0:
        raise ValueError("collection.minimum_image_change must be positive")
    docker = config["docker"]
    if docker["rendering"] not in RENDERING_MODES:
        raise ValueError(f"docker.rendering must be one of {RENDERING_MODES}")
    for key in ("cpus_per_worker", "memory_per_worker_gb"):
        if docker[key] <= 0.0:
            raise ValueError(f"docker.{key} must be positive")
    if docker["max_gpu_workers"] <= 0:
        raise ValueError("docker.max_gpu_workers must be positive")
    if docker["launch_interval_s"] < 0.0 or docker["max_restarts"] < 0:
        raise ValueError("docker.launch_interval_s and docker.max_restarts cannot be negative")
    if any(value <= 0.0 for value in config["timing"].values()):
        raise ValueError("Every timing value must be positive")

    randomization = config["randomization"]
    for key in ("board_x", "board_y", "board_yaw"):
        interval(randomization[key], f"randomization.{key}")
    for group, rail_count in (("nic_cards", 5), ("sc_ports", 5)):
        lower, upper = interval(randomization[group]["count"], f"{group}.count")
        if not (lower.is_integer() and upper.is_integer()) or lower < 0 or upper > rail_count:
            raise ValueError(f"randomization.{group}.count must be integers in [0, {rail_count}]")
        interval(randomization[group]["translation"], f"{group}.translation")
        interval(randomization[group]["yaw_deg"], f"{group}.yaw_deg")
    spacing = randomization["sc_ports"]["minimum_spacing"]
    lower, upper = randomization["sc_ports"]["translation"]
    if spacing < 0.0 or 2.0 * spacing > upper - lower:
        raise ValueError("sc_ports.minimum_spacing must fit three ports on one rail")
    if randomization["nic_cards"]["count"][0] < 1:
        raise ValueError("At least one NIC card must be spawned per scene")
    mounts = randomization["mounts"]
    if not 0.0 <= mounts["probability"] <= 1.0:
        raise ValueError("randomization.mounts.probability must be in [0, 1]")
    interval(mounts["translation"], "mounts.translation")
    interval(mounts["yaw_deg"], "mounts.yaw_deg")


def validate_train(config: dict[str, Any]) -> None:
    """Validate curation, model, optimization, and logging values.

    Args:
        config: Merged training configuration.

    Raises:
        ValueError: On semantic violations.
    """

    if config["curation"]["preview_count"] < 0:
        raise ValueError("curation.preview_count cannot be negative")
    model = config["model"]
    if model["d_model"] % model["heads"]:
        raise ValueError("model.d_model must be divisible by model.heads")
    for key in ("d_model", "heads", "decoder_layers", "feedforward"):
        if model[key] <= 0:
            raise ValueError(f"model.{key} must be positive")
    if model["encoder_layers"] < 0 or not 0.0 <= model["dropout"] < 1.0:
        raise ValueError("model.encoder_layers must be >= 0 and dropout in [0, 1)")
    optimization = config["optimization"]
    for key in ("epochs", "batch_size", "learning_rate", "backbone_learning_rate", "gradient_clip"):
        if optimization[key] <= 0:
            raise ValueError(f"optimization.{key} must be positive")
    if optimization["num_workers"] < 0 or optimization["warmup_steps"] < 0:
        raise ValueError("optimization.num_workers and warmup_steps cannot be negative")
    if any(value < 0.0 for value in config["loss"].values()):
        raise ValueError("Loss weights and tolerances cannot be negative")
    if config["logging"]["wandb_mode"] not in WANDB_MODES:
        raise ValueError(f"logging.wandb_mode must be one of {WANDB_MODES}")
    if config["logging"]["log_every_steps"] <= 0:
        raise ValueError("logging.log_every_steps must be positive")


def validate_infer(config: dict[str, Any]) -> None:
    """Validate inference node values.

    Args:
        config: Merged inference configuration.

    Raises:
        ValueError: On semantic violations.
    """

    inference = config["inference"]
    if inference["rate_hz"] <= 0.0:
        raise ValueError("inference.rate_hz must be positive")
    if not 0.0 < inference["presence_threshold"] < 1.0:
        raise ValueError("inference.presence_threshold must be in (0, 1)")
    approach = config["far_approach"]
    if approach["estimate_count"] <= 0:
        raise ValueError("far_approach.estimate_count must be positive")
    for key in ("estimate_timeout_s", "hover_height_m", "approach_duration_s"):
        if approach[key] <= 0.0:
            raise ValueError(f"far_approach.{key} must be positive")
    if approach["settle_time_s"] < 0.0:
        raise ValueError("far_approach.settle_time_s cannot be negative")


VALIDATORS = {"collect": validate_collect, "train": validate_train, "infer": validate_infer}


def load_config(kind: str, config_dir: str | Path = CONFIG_DIR) -> dict[str, Any]:
    """Load ``base.json`` merged with ``<kind>.json`` and resolve paths.

    Relative paths are resolved against the configuration directory, so the
    same files work on the host and inside a collection container.

    Args:
        kind: One of ``collect``, ``train``, or ``infer``.
        config_dir: Directory containing the JSON files.

    Returns:
        Validated configuration with absolute paths.
    """

    if kind not in OVERLAY_SCHEMAS:
        raise ValueError(f"Unknown configuration kind: {kind}")
    directory = Path(config_dir).expanduser().resolve()
    config = deep_merge(
        read_object(directory / "base.json"), read_object(directory / f"{kind}.json")
    )
    validate_shape(config, {**BASE_SCHEMA, **OVERLAY_SCHEMAS[kind]}, "config")
    validate_base(config)
    VALIDATORS[kind](config)

    def resolve(path: str) -> str:
        """Resolve a configured path relative to the configuration directory."""

        return str((directory / path).resolve())

    config["dataset"]["directory"] = resolve(config["dataset"]["directory"])
    if kind == "collect":
        config["collection"]["task_board_xacro"] = resolve(
            config["collection"]["task_board_xacro"]
        )
    if kind == "train":
        config["run"]["directory"] = resolve(config["run"]["directory"])
    if kind == "infer":
        config["inference"]["checkpoint"] = resolve(config["inference"]["checkpoint"])
    return config
