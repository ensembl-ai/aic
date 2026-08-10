"""LeRobot policy construction for AIC synthetic demonstrations."""

from __future__ import annotations

from typing import Any

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def camera_feature_name(camera_name: str) -> str:
    """Return the LeRobot observation key for one named camera."""

    return f"observation.images.{camera_name}"


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
