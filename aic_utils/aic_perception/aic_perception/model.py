"""Multiview scene estimator: shared image encoder plus a geometry-aware transformer.

Each camera image is encoded by the same pretrained ResNet-18. Every feature
token receives a ray embedding computed from the camera intrinsics and the
camera pose in the robot base frame, so calibration is an explicit input
rather than something memorized. A transformer encoder fuses the tokens of
all cameras, and one learned query per schema target decodes presence, a
base-frame pose, and a confidence.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from aic_perception.utils.rotations import matrix_from_6d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_CHANNELS = 512
OUTPUTS_PER_QUERY = 1 + 3 + 6 + 1


def camera_rays(
    intrinsics: torch.Tensor,
    base_from_camera: torch.Tensor,
    height: int,
    width: int,
    stride: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute base-frame ray origins and directions for every feature token.

    Args:
        intrinsics: ``(N, 3, 3)`` camera matrices of the input images.
        base_from_camera: ``(N, 4, 4)`` optical-frame poses.
        height: Feature map height.
        width: Feature map width.
        stride: Input pixels per feature cell.

    Returns:
        ``(N, H*W, 3)`` origins and unit directions.
    """

    device = intrinsics.device
    rows = (torch.arange(height, device=device) + 0.5) * stride - 0.5
    columns = (torch.arange(width, device=device) + 0.5) * stride - 0.5
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=-1).reshape(1, -1, 3)
    directions = pixels @ torch.linalg.inv(intrinsics).transpose(1, 2)
    directions = directions @ base_from_camera[:, :3, :3].transpose(1, 2)
    directions = nn.functional.normalize(directions, dim=-1)
    origins = base_from_camera[:, None, :3, 3].expand_as(directions)
    return origins, directions


class SceneEstimator(nn.Module):
    """Predicts every schema target's presence, pose, and confidence."""

    def __init__(
        self,
        model_config: dict[str, Any],
        num_targets: int,
        translation_mean: list[float],
        translation_std: list[float],
    ):
        """Build the network.

        Args:
            model_config: ``model`` section of the training configuration.
            num_targets: Number of schema targets (output queries).
            translation_mean: Base-frame translation mean from curation.
            translation_std: Base-frame translation std from curation.
        """

        super().__init__()
        d_model = model_config["d_model"]
        weights = ResNet18_Weights.IMAGENET1K_V1 if model_config["pretrained_backbone"] else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]).to(
            memory_format=torch.channels_last
        )
        self.input_projection = nn.Conv2d(BACKBONE_CHANNELS, d_model, kernel_size=1)
        self.ray_embedding = nn.Sequential(
            nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        layer = dict(
            d_model=d_model,
            nhead=model_config["heads"],
            dim_feedforward=model_config["feedforward"],
            dropout=model_config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(**layer),
                num_layers=model_config["encoder_layers"],
                enable_nested_tensor=False,
            )
            if model_config["encoder_layers"]
            else nn.Identity()
        )
        self.queries = nn.Embedding(num_targets, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**layer), num_layers=model_config["decoder_layers"]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, OUTPUTS_PER_QUERY))
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("translation_mean", torch.tensor(translation_mean))
        self.register_buffer("translation_std", torch.tensor(translation_std))

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Parameters of the pretrained image encoder."""

        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters trained from scratch."""

        backbone = {id(parameter) for parameter in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone]

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        base_from_camera: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate the scene.

        Args:
            images: ``(B, C, 3, H, W)`` uint8 RGB images.
            intrinsics: ``(B, C, 3, 3)`` camera matrices for those images.
            base_from_camera: ``(B, C, 4, 4)`` optical-frame poses.

        Returns:
            ``presence_logit`` (B, Q), ``translation`` (B, Q, 3) in meters,
            ``rotation`` (B, Q, 3, 3), and ``confidence_logit`` (B, Q).
        """

        batch, cameras, _, height, _ = images.shape
        pixels = images.flatten(0, 1).float().div(255.0)
        pixels = ((pixels - self.image_mean) / self.image_std).contiguous(
            memory_format=torch.channels_last
        )
        features = self.input_projection(self.backbone(pixels))
        feature_height, feature_width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        origins, directions = camera_rays(
            intrinsics.flatten(0, 1).float(),
            base_from_camera.flatten(0, 1).float(),
            feature_height,
            feature_width,
            height / feature_height,
        )
        origins = (origins - self.translation_mean) / self.translation_std
        tokens = tokens + self.ray_embedding(torch.cat((origins, directions), dim=-1))
        memory = self.encoder(tokens.reshape(batch, cameras * tokens.shape[1], -1))
        queries = self.queries.weight.unsqueeze(0).expand(batch, -1, -1)
        outputs = self.head(self.decoder(queries, memory)).float()
        return {
            "presence_logit": outputs[..., 0],
            "translation": self.translation_mean + outputs[..., 1:4] * self.translation_std,
            "rotation": matrix_from_6d(outputs[..., 4:10]),
            "confidence_logit": outputs[..., 10],
        }
