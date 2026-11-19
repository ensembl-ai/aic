"""Strict two-layer configuration for the AIC multiview scene estimator.

Every entry point loads ``configs/base.json`` plus exactly one overlay
(``collect.json``, ``train.json``, ``refine.json``, or ``infer.json``). Unknown or missing keys
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

OPTIMIZATION_SCHEMA = {
    "epochs": int,
    "batch_size": int,
    "learning_rate": float,
    "backbone_learning_rate": float,
    "weight_decay": float,
    "warmup_steps": int,
    "gradient_clip": float,
    "bf16": bool,
    "num_workers": int,
    "early_stopping_patience": int,
    "early_stopping_min_delta": float,
}

OVERLAY_SCHEMAS: dict[str, dict[str, Any]] = {
    "collect": {
        "collection": {
            "boards": int,
            "views_per_board": int,
            "seed": int,
            "workers": (int, str),
            "jpeg_quality": int,
            "minimum_image_change": float,
            "minimum_view_change": float,
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
        "viewpoints": {
            "height_m": list,
            "lateral_m": float,
            "lateral_per_height": float,
            "lateral_power": float,
            "on_axis_fraction": float,
            "home_view": bool,
            "yaw_deg": float,
            "move_speed_m_s": float,
            "settle_timeout_s": float,
            "position_tolerance_m": float,
            "stiffness": list,
            "damping": list,
            "plug_grasp": dict,
        },
        "cable": {
            "spawn": bool,
            "xacro": str,
            "types": dict,
            "gripper_offsets": dict,
            "rpy": list,
            "settle_s": float,
        },
        "randomization": {
            "sc_task_probability": float,
            "board_x": list,
            "board_y": list,
            "board_z": float,
            "board_yaw": list,
            "nic_cards": RANGE_SCHEMA,
            "sc_ports": {**RANGE_SCHEMA, "minimum_spacing": float},
            "mounts": {"probability": float, "translation": list, "yaw_deg": list},
        },
        "benchmark": {
            "seed": int,
            "trials": int,
            "sc_trials": int,
            "batches": int,
            "template": str,
            "directory": str,
        },
    },
    "train": {
        "curation": {"preview_count": int},
        "run": {"name": str, "directory": str, "seed": int, "device": str},
        "model": {
            "backbone": str,
            "pretrained_backbone": bool,
            "backbone_input": list,
            "d_model": int,
            "heads": int,
            "encoder_layers": int,
            "decoder_layers": int,
            "feedforward": int,
            "dropout": float,
        },
        "optimization": OPTIMIZATION_SCHEMA,
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
    "refine": {
        "datasets": list,
        "run": {"name": str, "directory": str, "seed": int, "device": str},
        "selection": {
            "reference_camera": str,
            "max_height_m": float,
            "max_lateral_m": float,
            "max_rotation_deg": float,
        },
        "crop": {"size": int},
        "model": {
            "backbone": str,
            "pretrained_backbone": bool,
            "backbone_input": list,
            "d_model": int,
            "heads": int,
            "layers": int,
            "feedforward": int,
            "dropout": float,
        },
        "optimization": OPTIMIZATION_SCHEMA,
        "loss": {"offset": float, "plug": float, "rotation": float},
        "augmentation": {"brightness": float, "contrast": float, "saturation": float},
        "logging": {"wandb_project": str, "wandb_mode": str, "log_every_steps": int},
    },
    "infer": {
        "logging": {
            "trial_directory": str,
            "estimate_directory": str,
            "log_estimates": bool,
        },
        "refiner": {
            "enabled": bool,
            "checkpoint": str,
            "device": str,
            "iterations": int,
            "samples": int,
            "tolerance_m": float,
            "max_step_m": float,
            "max_yaw_deg": float,
            "min_yaw_deg": float,
            "move_duration_s": float,
            "settle_time_s": float,
            "still_tolerance_m": float,
            "max_settle_s": float,
            "consistency_m": float,
            "measure_retries": int,
            "max_offset_m": float,
            "observation_timeout_s": float,
            "spiral_max_radius_m": float,
            "update_grasp": bool,
            "max_grasp_change_m": float,
        },
        "sc_rank_for_slot": dict,
        "inference": {
            "checkpoint": str,
            "device": str,
            "observation_topic": str,
            "estimate_topic": str,
            "rate_hz": float,
            "presence_threshold": float,
            "tf_prefix": str,
        },
        "insertion": {
            "estimate_count": int,
            "estimate_timeout_s": float,
            "search_after_s": float,
            "search_timeout_s": float,
            "search_move_s": float,
            "search_snapshot_s": float,
            "search_transit_height_m": float,
            "search_max_offset_m": float,
            "search_viewpoints": list,
            "search_anchors": dict,
            "control_period_s": float,
            "approach_duration_s": float,
            "coarse_height_m": float,
            "tracking_approach": bool,
            "track_alpha": float,
            "track_gate_m": float,
            "track_reset_after": int,
            "track_speed_m_s": float,
            "track_accel_m_s2": float,
            "track_dwell_s": float,
            "track_final_move_s": float,
            "track_timeout_s": float,
            "pre_contact_height_m": float,
            "still_tolerance_m": float,
            "still_timeout_s": float,
            "settle_corrections": int,
            "max_correction_m": float,
            "position_tolerance_m": float,
            "tracking_gain": float,
            "baseline_samples": int,
            "refine_at_pre_contact": bool,
            "refine_timeout_s": float,
            "max_refine_shift_m": float,
            "free_stiffness": list,
            "free_damping": list,
            "free_feedback_gains": list,
            "descent_stiffness": list,
            "descent_damping": list,
            "descent_feedback_gains": list,
            "servo_stiffness": list,
            "servo_damping": list,
            "servo_feedback_gains": list,
            "search_stiffness": list,
            "search_damping": list,
            "search_feedback_gains": list,
            "recovery_stiffness": list,
            "recovery_damping": list,
            "recovery_feedback_gains": list,
            "descent_speed_m_s": float,
            "descent_below_entrance_m": float,
            "inside_depth_m": float,
            "contact_force_n": float,
            "max_force_n": float,
            "stall_window_s": float,
            "stall_epsilon_m": float,
            "stall_grace_s": float,
            "press_margin_m": float,
            "entry_depth_m": float,
            "spiral_pitch_m": float,
            "spiral_max_radius_m": float,
            "spiral_speed_m_s": float,
            "insert_speed_m_s": float,
            "insert_max_duration_s": float,
            "insert_extra_depth_m": dict,
            "min_seat_depth_m": dict,
            "max_seat_depth_m": dict,
            "max_lateral_drift_m": float,
            "recovery_attempts": int,
            "recovery_attempts_before_spiral": int,
            "recovery_duration_s": float,
            "recovery_retract_m": float,
            "unstick_margin_m": float,
            "log_ground_truth": bool,
        },
        "plug_grasp": dict,
    },
}

RENDERING_MODES = ("wsl_gpu", "nvidia_gpu", "cpu")
SCORING_FORCE_LIMIT_N = 20.0
IMPEDANCE_MODES = ("free", "descent", "servo", "search", "recovery")
PLUG_TYPES = ("sfp", "sc")
BACKBONES = ("resnet18", "dinov2_vits14")
SC_SLOTS = ("sc_port_0", "sc_port_1")
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
    if collection["boards"] <= 0:
        raise ValueError("collection.boards must be positive")
    if collection["views_per_board"] < 1 + config["viewpoints"]["home_view"]:
        raise ValueError(
            "collection.views_per_board must leave at least one approach view"
        )
    workers = collection["workers"]
    if isinstance(workers, str) and workers != "auto":
        raise ValueError("collection.workers must be a positive integer or 'auto'")
    if isinstance(workers, int) and workers <= 0:
        raise ValueError("collection.workers must be positive")
    if not 1 <= collection["jpeg_quality"] <= 100:
        raise ValueError("collection.jpeg_quality must be in [1, 100]")
    if (
        not 0.0
        < collection["minimum_view_change"]
        <= collection["minimum_image_change"]
    ):
        raise ValueError(
            "collection.minimum_view_change must be positive and at most minimum_image_change"
        )
    docker = config["docker"]
    if docker["rendering"] not in RENDERING_MODES:
        raise ValueError(f"docker.rendering must be one of {RENDERING_MODES}")
    for key in ("cpus_per_worker", "memory_per_worker_gb"):
        if docker[key] <= 0.0:
            raise ValueError(f"docker.{key} must be positive")
    if docker["max_gpu_workers"] <= 0:
        raise ValueError("docker.max_gpu_workers must be positive")
    if docker["launch_interval_s"] < 0.0 or docker["max_restarts"] < 0:
        raise ValueError(
            "docker.launch_interval_s and docker.max_restarts cannot be negative"
        )
    if any(value <= 0.0 for value in config["timing"].values()):
        raise ValueError("Every timing value must be positive")

    randomization = config["randomization"]
    for key in ("board_x", "board_y", "board_yaw"):
        interval(randomization[key], f"randomization.{key}")
    if not 0.0 <= randomization["sc_task_probability"] <= 1.0:
        raise ValueError("randomization.sc_task_probability must be in [0, 1]")
    for group, rail_count in (("nic_cards", 5), ("sc_ports", 2)):
        lower, upper = interval(randomization[group]["count"], f"{group}.count")
        if (
            not (lower.is_integer() and upper.is_integer())
            or lower < 0
            or upper > rail_count
        ):
            raise ValueError(
                f"randomization.{group}.count must be integers in [0, {rail_count}]"
            )
        interval(randomization[group]["translation"], f"{group}.translation")
        interval(randomization[group]["yaw_deg"], f"{group}.yaw_deg")
    spacing = randomization["sc_ports"]["minimum_spacing"]
    lower, upper = randomization["sc_ports"]["translation"]
    if spacing < 0.0 or spacing > upper - lower:
        raise ValueError("sc_ports.minimum_spacing must fit two ports on the rail")
    viewpoints = config["viewpoints"]
    lower, upper = interval(viewpoints["height_m"], "viewpoints.height_m")
    if lower <= 0.0:
        raise ValueError("viewpoints.height_m must keep the plug above the port")
    for key in (
        "lateral_m",
        "lateral_per_height",
        "yaw_deg",
        "settle_timeout_s",
        "position_tolerance_m",
    ):
        if viewpoints[key] < 0.0:
            raise ValueError(f"viewpoints.{key} cannot be negative")
    if viewpoints["lateral_power"] <= 0.0:
        raise ValueError("viewpoints.lateral_power must be positive")
    if not 0.0 <= viewpoints["on_axis_fraction"] <= 1.0:
        raise ValueError("viewpoints.on_axis_fraction must be in [0, 1]")
    if viewpoints["move_speed_m_s"] <= 0.0:
        raise ValueError("viewpoints.move_speed_m_s must be positive")
    for key in ("stiffness", "damping"):
        if len(viewpoints[key]) != 6 or any(
            not matches(v, float) or v < 0.0 for v in viewpoints[key]
        ):
            raise ValueError(f"viewpoints.{key} must be six non-negative numbers")
    for plug_type in PLUG_TYPES:
        grasp = viewpoints["plug_grasp"].get(plug_type, {})
        if (
            set(grasp) != {"translation", "quaternion_xyzw"}
            or len(grasp["translation"]) != 3
            or len(grasp["quaternion_xyzw"]) != 4
        ):
            raise ValueError(
                f"viewpoints.plug_grasp.{plug_type} needs a 3D translation and a quaternion_xyzw"
            )
    cable = config["cable"]
    if set(cable["types"]) != set(PLUG_TYPES) or set(cable["gripper_offsets"]) != set(
        PLUG_TYPES
    ):
        raise ValueError(
            f"cable.types and cable.gripper_offsets need exactly the plug types {PLUG_TYPES}"
        )
    for plug_type, offsets in cable["gripper_offsets"].items():
        if not offsets or any(len(offset) != 3 for offset in offsets):
            raise ValueError(
                f"cable.gripper_offsets.{plug_type} needs one or more 3D offsets"
            )
    if len(cable["rpy"]) != 3:
        raise ValueError("cable.rpy needs three values")
    mounts = randomization["mounts"]
    if not 0.0 <= mounts["probability"] <= 1.0:
        raise ValueError("randomization.mounts.probability must be in [0, 1]")
    interval(mounts["translation"], "mounts.translation")
    interval(mounts["yaw_deg"], "mounts.yaw_deg")
    benchmark = config["benchmark"]
    if (
        not 0 <= benchmark["sc_trials"] <= benchmark["trials"]
        or not 0 < benchmark["batches"] <= benchmark["trials"]
    ):
        raise ValueError(
            "benchmark needs 0 <= sc_trials <= trials and 0 < batches <= trials"
        )


def validate_backbone(model: dict[str, Any]) -> None:
    """Validate the backbone choice and its input size.

    Raises:
        ValueError: On an unknown backbone or an input size the backbone
            cannot take.
    """

    if model["backbone"] not in BACKBONES:
        raise ValueError(f"model.backbone must be one of {BACKBONES}")
    size = model["backbone_input"]
    if size and (len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size)):
        raise ValueError("model.backbone_input must be [] or [width, height] in pixels")
    if model["backbone"] == "dinov2_vits14" and (not size or any(v % 14 for v in size)):
        raise ValueError(
            "model.backbone_input must be [width, height] in multiples of 14 for DINOv2"
        )
    if model["d_model"] % model["heads"]:
        raise ValueError("model.d_model must be divisible by model.heads")
    if not 0.0 <= model["dropout"] < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")


def validate_optimization(config: dict[str, Any]) -> None:
    """Validate the optimization and logging sections shared by both trainers.

    Raises:
        ValueError: On semantic violations.
    """

    optimization = config["optimization"]
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "backbone_learning_rate",
        "gradient_clip",
    ):
        if optimization[key] <= 0:
            raise ValueError(f"optimization.{key} must be positive")
    if optimization["num_workers"] < 0 or optimization["warmup_steps"] < 0:
        raise ValueError("optimization.num_workers and warmup_steps cannot be negative")
    if (
        optimization["early_stopping_patience"] < 0
        or optimization["early_stopping_min_delta"] < 0.0
    ):
        raise ValueError(
            "optimization.early_stopping_patience and early_stopping_min_delta cannot be negative"
        )
    if any(value < 0.0 for value in config["loss"].values()):
        raise ValueError("Loss weights and tolerances cannot be negative")
    if config["logging"]["wandb_mode"] not in WANDB_MODES:
        raise ValueError(f"logging.wandb_mode must be one of {WANDB_MODES}")
    if config["logging"]["log_every_steps"] <= 0:
        raise ValueError("logging.log_every_steps must be positive")


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
    validate_backbone(model)
    for key in ("d_model", "heads", "decoder_layers", "feedforward"):
        if model[key] <= 0:
            raise ValueError(f"model.{key} must be positive")
    if model["encoder_layers"] < 0:
        raise ValueError("model.encoder_layers must be >= 0")
    validate_optimization(config)


def validate_refine(config: dict[str, Any]) -> None:
    """Validate the close-range refiner training values.

    Args:
        config: Merged refiner training configuration.

    Raises:
        ValueError: On semantic violations.
    """

    if not config["datasets"] or any(
        not isinstance(d, str) for d in config["datasets"]
    ):
        raise ValueError("datasets must list one or more curated dataset directories")
    selection = config["selection"]
    if selection["reference_camera"] not in config["schema"]["cameras"]:
        raise ValueError("selection.reference_camera must be one of schema.cameras")
    for key in ("max_height_m", "max_lateral_m", "max_rotation_deg"):
        if selection[key] <= 0.0:
            raise ValueError(f"selection.{key} must be positive")
    size = config["crop"]["size"]
    images = config["images"]
    if size <= 0 or size > min(images["width"], images["height"]):
        raise ValueError("crop.size must be positive and fit inside the stored images")
    model = config["model"]
    validate_backbone(model)
    for key in ("d_model", "heads", "layers", "feedforward"):
        if model[key] <= 0:
            raise ValueError(f"model.{key} must be positive")
    validate_optimization(config)


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
    insertion = config["insertion"]
    if not 0.0 < insertion["track_alpha"] <= 1.0:
        raise ValueError("insertion.track_alpha must be in (0, 1]")
    for target_id, anchor in insertion["search_anchors"].items():
        if len(anchor) != 3 or any(not matches(v, float) for v in anchor):
            raise ValueError(
                f"insertion.search_anchors.{target_id} must be three numbers"
            )
    viewpoints = insertion["search_viewpoints"]
    if not viewpoints or any(
        len(v) != 2 or any(not matches(x, float) for x in v) for v in viewpoints
    ):
        raise ValueError(
            "insertion.search_viewpoints must be a non-empty list of [shift_m, height_m] pairs"
        )
    for key in (
        "estimate_count",
        "baseline_samples",
        "recovery_attempts",
        "settle_corrections",
        "track_reset_after",
    ):
        if insertion[key] <= 0:
            raise ValueError(f"insertion.{key} must be positive")
    if (
        not 0
        <= insertion["recovery_attempts_before_spiral"]
        <= insertion["recovery_attempts"]
    ):
        raise ValueError(
            "insertion.recovery_attempts_before_spiral must be in [0, recovery_attempts]"
        )
    for key, value in insertion.items():
        if isinstance(value, float) and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"insertion.{key} must be finite and positive")
    for key in (
        f"{mode}_{kind}"
        for mode in IMPEDANCE_MODES
        for kind in ("stiffness", "damping", "feedback_gains")
    ):
        values = insertion[key]
        if len(values) != 6 or any(not matches(v, float) or v < 0.0 for v in values):
            raise ValueError(f"insertion.{key} must be six non-negative numbers")
        if key.endswith("feedback_gains") and any(v > 0.95 for v in values):
            raise ValueError(f"insertion.{key} must be at most 0.95 (controller limit)")
    if not 0.0 < insertion["tracking_gain"] <= 1.0:
        raise ValueError("insertion.tracking_gain must be in (0, 1]")
    if insertion["contact_force_n"] >= insertion["max_force_n"]:
        raise ValueError("insertion.contact_force_n must stay below max_force_n")
    if insertion["max_force_n"] >= SCORING_FORCE_LIMIT_N:
        raise ValueError(
            f"insertion.max_force_n must stay below the {SCORING_FORCE_LIMIT_N} N scoring limit"
        )
    if insertion["inside_depth_m"] >= insertion["descent_below_entrance_m"]:
        raise ValueError(
            "insertion.inside_depth_m must be shallower than descent_below_entrance_m"
        )
    for key in ("insert_extra_depth_m", "min_seat_depth_m", "max_seat_depth_m"):
        if any(not matches(v, float) or v <= 0.0 for v in insertion[key].values()):
            raise ValueError(f"insertion.{key} values must be positive")
    for plug, grasp in config["plug_grasp"].items():
        if (
            set(grasp) != {"translation", "quaternion_xyzw"}
            or len(grasp["translation"]) != 3
            or len(grasp["quaternion_xyzw"]) != 4
        ):
            raise ValueError(
                f"plug_grasp.{plug} needs a 3D translation and a quaternion_xyzw"
            )
    refiner = config["refiner"]
    for key in ("iterations", "samples"):
        if refiner[key] <= 0:
            raise ValueError(f"refiner.{key} must be positive")
    for key, value in refiner.items():
        if isinstance(value, float) and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"refiner.{key} must be finite and non-negative")
    if refiner["spiral_max_radius_m"] <= 0.0 or refiner["max_step_m"] <= 0.0:
        raise ValueError("refiner.spiral_max_radius_m and max_step_m must be positive")
    ranks = config["sc_rank_for_slot"]
    if set(ranks) != set(SC_SLOTS) or sorted(ranks.values()) != [0, 1]:
        raise ValueError(f"sc_rank_for_slot must map {SC_SLOTS} onto the ranks 0 and 1")


VALIDATORS = {
    "collect": validate_collect,
    "train": validate_train,
    "refine": validate_refine,
    "infer": validate_infer,
}


def load_config(kind: str, config_dir: str | Path = CONFIG_DIR) -> dict[str, Any]:
    """Load ``base.json`` merged with ``<kind>.json`` and resolve paths.

    Relative paths are resolved against the configuration directory, so the
    same files work on the host and inside a collection container.

    Args:
        kind: One of ``collect``, ``train``, ``refine``, or ``infer``.
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
        config["cable"]["xacro"] = resolve(config["cable"]["xacro"])
        for key in ("template", "directory"):
            config["benchmark"][key] = resolve(config["benchmark"][key])
    if kind in ("train", "refine"):
        config["run"]["directory"] = resolve(config["run"]["directory"])
    if kind == "refine":
        config["datasets"] = [resolve(directory) for directory in config["datasets"]]
    if kind == "infer":
        config["inference"]["checkpoint"] = resolve(config["inference"]["checkpoint"])
        config["refiner"]["checkpoint"] = resolve(config["refiner"]["checkpoint"])
        for key in ("trial_directory", "estimate_directory"):
            config["logging"][key] = resolve(config["logging"][key])
    return config
