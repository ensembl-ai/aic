"""Runtime wrapper that turns raw camera data into named target estimates.

Shared by the ROS node and any offline caller. Images are resized exactly as
the collector stored them (area averaging) and the intrinsics are rescaled
with the same convention, so runtime inputs match training inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from aic_perception.schema import BoardSchema
from aic_perception.training import load_checkpoint
from aic_perception.utils.geometry import scaled_intrinsics, transform


@dataclass
class TargetEstimate:
    """Estimate for one schema target.

    Attributes:
        id: Target ID.
        type: Target type.
        presence_probability: Probability that the target is present and in view.
        confidence: Probability that the pose is within the trained tolerance.
        base_from_target: 4x4 pose in the robot base frame.
    """

    id: str
    type: str
    presence_probability: float
    confidence: float
    base_from_target: np.ndarray


class SceneEstimatorRuntime:
    """Loaded estimator with preprocessing for full-resolution camera images."""

    def __init__(self, checkpoint: str, device: str):
        """Load a checkpoint.

        Args:
            checkpoint: Path to a training checkpoint.
            device: Torch device.
        """

        self.device = torch.device(device)
        self.model, self.config = load_checkpoint(checkpoint, device)
        self.schema = BoardSchema.from_config(self.config)
        self.width = self.config["images"]["width"]
        self.height = self.config["images"]["height"]

    @torch.no_grad()
    def predict(
        self,
        images: list[np.ndarray],
        intrinsics: list[np.ndarray],
        base_from_camera: list[np.ndarray],
    ) -> list[TargetEstimate]:
        """Estimate every schema target from one synchronized observation.

        Args:
            images: ``(H, W, 3)`` uint8 RGB images in schema camera order.
            intrinsics: 3x3 camera matrices of those images.
            base_from_camera: 4x4 optical-frame poses in the base frame.

        Returns:
            One estimate per schema target in schema order.
        """

        tensors, matrices = [], []
        for image, camera in zip(images, intrinsics, strict=True):
            pixels = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
            pixels = pixels.permute(2, 0, 1)[None].float()
            resized = F.interpolate(pixels, size=(self.height, self.width), mode="area")
            tensors.append(resized.round().clamp(0, 255).to(torch.uint8)[0])
            matrices.append(
                scaled_intrinsics(
                    camera, self.width / image.shape[1], self.height / image.shape[0]
                )
            )
        outputs = self.model(
            torch.stack(tensors)[None],
            torch.tensor(np.stack(matrices), dtype=torch.float32, device=self.device)[
                None
            ],
            torch.tensor(
                np.stack(base_from_camera), dtype=torch.float32, device=self.device
            )[None],
        )
        presence = torch.sigmoid(outputs["presence_logit"][0]).cpu().numpy()
        confidence = torch.sigmoid(outputs["confidence_logit"][0]).cpu().numpy()
        translation = outputs["translation"][0].cpu().numpy()
        rotation = outputs["rotation"][0].cpu().numpy()
        return [
            TargetEstimate(
                id=target.id,
                type=target.type,
                presence_probability=float(presence[query]),
                confidence=float(confidence[query]),
                base_from_target=transform(rotation[query], translation[query]),
            )
            for query, target in enumerate(self.schema.targets)
        ]

    def describe(self) -> dict[str, Any]:
        """Summary of the loaded model for logging."""

        return {
            "schema": self.schema.name,
            "targets": len(self.schema.targets),
            "image_size": (self.width, self.height),
        }
