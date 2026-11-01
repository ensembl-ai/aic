"""Training losses and evaluation errors for the scene estimator."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from aic_perception.utils.rotations import angle_between

METERS_TO_CM = 100.0


def pose_errors(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-target translation and rotation errors.

    Args:
        outputs: Model outputs.
        batch: Batch with ``base_from_target`` labels.

    Returns:
        ``(B, Q)`` translation errors in meters and rotation errors in radians.
    """

    target = batch["base_from_target"]
    translation = (outputs["translation"] - target[..., :3, 3]).norm(dim=-1)
    rotation = angle_between(outputs["rotation"], target[..., :3, :3])
    return translation, rotation


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted training objective.

    Presence is supervised on every query. Pose and confidence are supervised
    only on visible targets. The confidence target is whether the current
    pose error is within the configured tolerance.

    Args:
        outputs: Model outputs.
        batch: Training batch.
        settings: ``loss`` section of the training configuration.

    Returns:
        Total loss and detached component values for logging.
    """

    present = batch["present"]
    mask = present > 0.5
    presence = F.binary_cross_entropy_with_logits(outputs["presence_logit"], present)
    target = batch["base_from_target"]
    translation = (
        (outputs["translation"] - target[..., :3, 3]).abs().mean(-1)[mask].mean() * METERS_TO_CM
    )
    rotation = (outputs["rotation"] - target[..., :3, :3]).square().sum((-1, -2))[mask].mean()
    with torch.no_grad():
        translation_error, rotation_error = pose_errors(outputs, batch)
        accurate = (translation_error < settings["confidence_translation_m"]) & (
            rotation_error < math.radians(settings["confidence_rotation_deg"])
        )
    confidence = F.binary_cross_entropy_with_logits(
        outputs["confidence_logit"][mask], accurate[mask].float()
    )
    total = (
        settings["presence"] * presence
        + settings["translation"] * translation
        + settings["rotation"] * rotation
        + settings["confidence"] * confidence
    )
    return total, {
        "loss/total": total.item(),
        "loss/presence": presence.item(),
        "loss/translation_cm": translation.item(),
        "loss/rotation_chordal": rotation.item(),
        "loss/confidence": confidence.item(),
    }


class ErrorAccumulator:
    """Accumulates per-target errors over a split for evaluation reports."""

    def __init__(self, target_ids: list[str], target_types: list[str]):
        """Start an empty accumulation.

        Args:
            target_ids: Target IDs in query order.
            target_types: Target type per query.
        """

        self.target_ids = target_ids
        self.target_types = target_types
        self.translation: list[list[float]] = [[] for _ in target_ids]
        self.rotation: list[list[float]] = [[] for _ in target_ids]
        self.presence_correct = [0 for _ in target_ids]
        self.count = 0

    def add(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        """Add one evaluated batch.

        Args:
            outputs: Model outputs.
            batch: Labeled batch.
        """

        translation, rotation = pose_errors(outputs, batch)
        present = batch["present"] > 0.5
        predicted = outputs["presence_logit"] > 0.0
        correct = (predicted == present).sum(0).tolist()
        for query in range(len(self.target_ids)):
            visible = present[:, query]
            self.translation[query] += translation[visible, query].tolist()
            self.rotation[query] += rotation[visible, query].tolist()
            self.presence_correct[query] += correct[query]
        self.count += present.shape[0]

    def summary(self) -> dict[str, Any]:
        """Return mean/median errors per target and per type, in mm and degrees."""

        def describe(translation: list[float], rotation: list[float]) -> dict[str, float]:
            """Summarize one group of errors."""

            if not translation:
                return {"samples": 0}
            millimeters = torch.tensor(translation) * 1000.0
            degrees = torch.rad2deg(torch.tensor(rotation))
            return {
                "samples": len(translation),
                "translation_mean_mm": millimeters.mean().item(),
                "translation_median_mm": millimeters.median().item(),
                "translation_within_5mm": (millimeters < 5.0).float().mean().item(),
                "rotation_mean_deg": degrees.mean().item(),
                "rotation_median_deg": degrees.median().item(),
            }

        per_target = {
            target_id: {
                **describe(self.translation[query], self.rotation[query]),
                "presence_accuracy": self.presence_correct[query] / max(1, self.count),
            }
            for query, target_id in enumerate(self.target_ids)
        }
        per_type = {}
        for target_type in dict.fromkeys(self.target_types):
            queries = [q for q, t in enumerate(self.target_types) if t == target_type]
            per_type[target_type] = {
                **describe(
                    [e for q in queries for e in self.translation[q]],
                    [e for q in queries for e in self.rotation[q]],
                ),
                "presence_accuracy": sum(self.presence_correct[q] for q in queries)
                / max(1, self.count * len(queries)),
            }
        return {"scenes": self.count, "per_type": per_type, "per_target": per_target}
