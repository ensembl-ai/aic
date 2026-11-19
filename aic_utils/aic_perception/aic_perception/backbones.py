"""Image encoders shared by the scene estimator and the close-range refiner.

Every backbone maps normalized RGB images to a dense feature map, so the
models above it only see ``(N, channels, rows, columns)`` and the input-pixel
size of one feature cell.

- ``resnet18``: torchvision ResNet-18 without pooling and classifier
  (stride 32, 512 channels).
- ``dinov2_vits14``: DINOv2 ViT-S/14 with registers from ``timm`` (stride 14,
  384 channels, register and class tokens dropped). Its input is resized to
  ``backbone_input`` (multiples of 14) before the patch embedding.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from aic_perception.config import BACKBONES

DINOV2_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"
DINOV2_PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(nn.Module):
    """Normalizes ``uint8`` images and returns a dense feature map."""

    def __init__(self, name: str, pretrained: bool, input_size: list[int] | None):
        """Build one encoder.

        Args:
            name: One of ``BACKBONES``.
            pretrained: Load ImageNet (ResNet) or LVD-142M (DINOv2) weights.
            input_size: ``[width, height]`` the images are resized to before
                the encoder, or ``None`` to keep them; DINOv2 needs multiples
                of 14.
        """

        super().__init__()
        self.name = name
        self.input_size = input_size
        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-2])
            self.channels = 512
        elif name == "dinov2_vits14":
            self.encoder = timm.create_model(
                DINOV2_NAME, pretrained=pretrained, num_classes=0, dynamic_img_size=True
            )
            self.channels = self.encoder.num_features
        else:
            raise ValueError(f"Unknown backbone {name}; expected one of {BACKBONES}")
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            images: ``(N, 3, H, W)`` uint8 RGB.

        Returns:
            ``(N, channels, rows, columns)`` features.
        """

        pixels = (images.float().div(255.0) - self.mean) / self.std
        if self.input_size is not None:
            width, height = self.input_size
            if pixels.shape[-2:] != (height, width):
                pixels = F.interpolate(pixels, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
        if self.name == "resnet18":
            return self.encoder(pixels.contiguous(memory_format=torch.channels_last))
        tokens = self.encoder.forward_features(pixels)[:, self.encoder.num_prefix_tokens :]
        rows, columns = pixels.shape[-2] // DINOV2_PATCH, pixels.shape[-1] // DINOV2_PATCH
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.channels, rows, columns)


def build_backbone(model_config: dict[str, Any]) -> Backbone:
    """Build the backbone selected by a ``model`` configuration section.

    Args:
        model_config: Section with ``backbone``, ``pretrained_backbone``, and
            ``backbone_input`` (``[width, height]`` or an empty list).

    Returns:
        Backbone module.
    """

    size = model_config["backbone_input"] or None
    return Backbone(model_config["backbone"], model_config["pretrained_backbone"], size)
