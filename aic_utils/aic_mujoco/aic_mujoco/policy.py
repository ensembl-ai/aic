"""LeRobot policy construction for AIC synthetic demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

AMP_DTYPES = {"bfloat16": torch.bfloat16}


def camera_feature_name(camera_name: str) -> str:
    """Return the LeRobot observation key for one named camera."""

    return f"observation.images.{camera_name}"


def validate_policy_device(device_name: str) -> None:
    """Reject unavailable policy devices without silently falling back.

    Args:
        device_name: Explicit PyTorch device from configuration.

    Raises:
        RuntimeError: If the requested CUDA device cannot execute this build.
    """

    device = torch.device(device_name)
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device_name}")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device_name}")
    major, minor = torch.cuda.get_device_capability(device_index)
    architecture = f"sm_{major}{minor}"
    supported_architectures = torch.cuda.get_arch_list()
    if supported_architectures and architecture not in supported_architectures:
        raise RuntimeError(
            f"PyTorch {torch.__version__} does not support {architecture} on "
            f"{torch.cuda.get_device_name(device_index)}; install the locked Pixi "
            "environment before running the policy"
        )


class PolicyNormalizer:
    """Prepare policy observations and normalize supervised action targets."""

    def __init__(
        self,
        config: dict[str, Any],
        statistics: dict[str, dict[str, np.ndarray]],
    ):
        """Create reusable normalization tensors on the policy device.

        Args:
            config: Strict merged policy configuration.
            statistics: Train-split state and action mean/std arrays.
        """

        self.device = torch.device(config["training"]["device"])
        self.amp_dtype = AMP_DTYPES[config["training"]["amp_dtype"]]
        self.camera_names = config["policy"]["camera_names"]
        self.image_mean = torch.tensor(
            config["policy"]["image_mean"], device=self.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            config["policy"]["image_std"], device=self.device
        ).view(1, 3, 1, 1)
        self.state_mean = torch.from_numpy(
            statistics["observation.state"]["mean"]
        ).to(self.device)
        self.state_std = torch.from_numpy(
            statistics["observation.state"]["std"]
        ).to(self.device)
        self.action_mean = torch.from_numpy(statistics["action"]["mean"]).to(
            self.device
        )
        self.action_std = torch.from_numpy(statistics["action"]["std"]).to(
            self.device
        )

    def prepare_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Transfer and normalize one batch of policy observations.

        Args:
            observation: Named RGB tensors and concatenated robot state.

        Returns:
            Policy-ready tensors on the configured device.
        """

        batch: dict[str, torch.Tensor] = {}
        for camera_name in self.camera_names:
            key = camera_feature_name(camera_name)
            image = observation[key].to(self.device, non_blocking=True).float()
            batch[key] = (image.div(255.0) - self.image_mean) / self.image_std
        state = observation[OBS_STATE].to(self.device, non_blocking=True)
        batch[OBS_STATE] = (state - self.state_mean) / self.state_std
        return batch

    def prepare(self, host_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare a complete supervised training or validation batch.

        Args:
            host_batch: Collated observations, actions, and padding mask.

        Returns:
            Normalized tensors on the configured device.
        """

        batch = self.prepare_observation(host_batch)
        action = host_batch[ACTION].to(self.device, non_blocking=True)
        batch[ACTION] = (action - self.action_mean) / self.action_std
        batch["action_is_pad"] = host_batch["action_is_pad"].to(
            self.device, non_blocking=True
        )
        return batch

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert normalized policy actions to joint increments in radians."""

        return action * self.action_std + self.action_mean

    def action_error_radians(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Convert normalized absolute action error back to joint radians."""

        return torch.abs(predicted - target) * self.action_std


def load_normalization_statistics(
    config: dict[str, Any], checkpoint_directory: Path
) -> dict[str, dict[str, np.ndarray]]:
    """Load and strictly validate train-split normalization statistics.

    Args:
        config: Strict merged evaluation configuration.
        checkpoint_directory: Explicit checkpoint containing normalization JSON.

    Returns:
        State and action mean/std arrays in ``float32``.

    Raises:
        FileNotFoundError: If normalization data is absent.
        ValueError: If keys, shapes, values, or standard deviations are invalid.
    """

    path = checkpoint_directory / "normalization.json"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint normalization does not exist: {path}")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid checkpoint normalization JSON: {path}") from error
    expected_features = {"observation.state", "action"}
    if not isinstance(stored, dict) or set(stored) != expected_features:
        raise ValueError("Checkpoint normalization has unexpected feature keys")
    joint_count = len(config["scene"]["names"]["joints"])
    expected_shapes = {
        "observation.state": (
            joint_count * len(config["policy"]["state_fields"]),
        ),
        "action": (joint_count,),
    }
    statistics: dict[str, dict[str, np.ndarray]] = {}
    minimum_std = config["training"]["normalization_minimum_std"]
    for feature_name, expected_shape in expected_shapes.items():
        feature = stored[feature_name]
        if not isinstance(feature, dict) or set(feature) != {"mean", "std"}:
            raise ValueError(
                f"Checkpoint normalization for {feature_name} must contain mean/std"
            )
        statistics[feature_name] = {}
        for statistic_name in ("mean", "std"):
            values = np.asarray(feature[statistic_name], dtype=np.float32)
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Checkpoint {feature_name}.{statistic_name} must have finite "
                    f"shape {expected_shape}"
                )
            statistics[feature_name][statistic_name] = values
        if np.any(statistics[feature_name]["std"] < minimum_std):
            raise ValueError(
                f"Checkpoint {feature_name}.std violates the configured minimum"
            )
    return statistics


def policy_interface_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return only configuration values required to load policy tensors.

    Task randomization, controller gains, goal tolerances, episode horizons,
    physics settings, and visualization behavior may change for evaluation.
    Model architecture is loaded from the checkpoint itself. The active
    configuration supplies only observation construction, preprocessing, and
    validation target shape.
    """

    policy = config["policy"]
    return {
        "joint_names": config["scene"]["names"]["joints"],
        "image_shape": [
            3,
            config["dataset"]["image_height"],
            config["dataset"]["image_width"],
        ],
        "policy_type": policy["type"],
        "camera_names": policy["camera_names"],
        "state_fields": policy["state_fields"],
        "action_field": policy["action_field"],
        "chunk_size": policy["chunk_size"],
        "image_mean": policy["image_mean"],
        "image_std": policy["image_std"],
    }


def load_policy_checkpoint(
    config: dict[str, Any],
) -> tuple[ACTPolicy, dict[str, dict[str, np.ndarray]]]:
    """Load one explicit compatible checkpoint without remote lookup.

    Args:
        config: Strict merged evaluation configuration.

    Returns:
        Evaluation-mode ACT policy and its train-split normalization values.

    Raises:
        FileNotFoundError: If the checkpoint or saved run contract is absent.
        ValueError: If the checkpoint was trained under a different contract.
    """

    validate_policy_device(config["training"]["device"])
    checkpoint = Path(config["evaluation"]["checkpoint_directory"])
    policy_directory = checkpoint / "policy"
    required_policy_files = (
        policy_directory / "config.json",
        policy_directory / "model.safetensors",
    )
    complete_checkpoint = checkpoint.is_dir() and all(
        path.is_file() for path in required_policy_files
    )
    if not complete_checkpoint:
        raise FileNotFoundError(
            f"Complete policy checkpoint does not exist: {checkpoint}"
        )
    run_config_path = checkpoint.parent.parent / "config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint run configuration does not exist: {run_config_path}"
        )
    try:
        trained_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid checkpoint run configuration: {run_config_path}"
        ) from error
    trained_interface = policy_interface_contract(trained_config)
    current_interface = policy_interface_contract(config)
    mismatches = [
        key
        for key in current_interface
        if trained_interface.get(key) != current_interface[key]
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint tensor interface differs from the current configuration: "
            + ", ".join(mismatches)
        )
    statistics = load_normalization_statistics(config, checkpoint)
    loaded = ACTPolicy.from_pretrained(
        policy_directory,
        local_files_only=True,
        strict=True,
    ).to(config["training"]["device"])
    loaded.eval()
    return loaded, statistics


def create_policy(config: dict[str, Any]) -> ACTPolicy:
    """Create the configured three-camera ACT manipulation policy.

    Args:
        config: Strict merged policy-training configuration.

    Returns:
        Untrained LeRobot ACT policy on the configured device.

    Notes:
        The training pipeline owns normalization and automatic mixed precision,
        so the upstream policy's preprocessing and AMP switches remain off.
    """

    policy = config["policy"]
    training = config["training"]
    image_shape = (
        3,
        config["dataset"]["image_height"],
        config["dataset"]["image_width"],
    )
    joint_count = len(config["scene"]["names"]["joints"])
    state_dimension = joint_count * len(policy["state_fields"])
    input_features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dimension,)),
        **{
            camera_feature_name(name): PolicyFeature(FeatureType.VISUAL, image_shape)
            for name in policy["camera_names"]
        },
    }
    output_features = {
        ACTION: PolicyFeature(FeatureType.ACTION, (joint_count,)),
    }
    act_config = ACTConfig(
        n_obs_steps=1,
        chunk_size=policy["chunk_size"],
        n_action_steps=policy["n_action_steps"],
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        vision_backbone=policy["vision_backbone"],
        pretrained_backbone_weights=policy["pretrained_backbone_weights"],
        replace_final_stride_with_dilation=policy[
            "replace_final_stride_with_dilation"
        ],
        pre_norm=policy["pre_norm"],
        dim_model=policy["dim_model"],
        n_heads=policy["n_heads"],
        dim_feedforward=policy["dim_feedforward"],
        feedforward_activation=policy["feedforward_activation"],
        n_encoder_layers=policy["n_encoder_layers"],
        n_decoder_layers=policy["n_decoder_layers"],
        use_vae=policy["use_vae"],
        latent_dim=policy["latent_dim"],
        n_vae_encoder_layers=policy["n_vae_encoder_layers"],
        dropout=policy["dropout"],
        kl_weight=policy["kl_weight"],
        optimizer_lr=training["learning_rate"],
        optimizer_lr_backbone=training["backbone_learning_rate"],
        optimizer_weight_decay=training["weight_decay"],
        device=training["device"],
        use_amp=False,
        push_to_hub=False,
    )
    return ACTPolicy(act_config).to(training["device"])
