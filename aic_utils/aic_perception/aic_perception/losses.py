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
    translation = (outputs["translation"] - target[..., :3, 3]).abs().mean(-1)[
        mask
    ].mean() * METERS_TO_CM
    rotation = (
        (outputs["rotation"] - target[..., :3, :3]).square().sum((-1, -2))[mask].mean()
    )
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


CONFIDENCE_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0001)


def describe(translation: list[float], rotation: list[float]) -> dict[str, float]:
    """Summarize one group of translation (m) and rotation (rad) errors."""

    if not translation:
        return {"samples": 0}
    millimeters = torch.tensor(translation) * 1000.0
    degrees = torch.rad2deg(torch.tensor(rotation))
    return {
        "samples": len(translation),
        "translation_mean_mm": millimeters.mean().item(),
        "translation_median_mm": millimeters.median().item(),
        "translation_p90_mm": millimeters.quantile(0.9).item(),
        "translation_within_2mm": (millimeters < 2.0).float().mean().item(),
        "translation_within_5mm": (millimeters < 5.0).float().mean().item(),
        "rotation_mean_deg": degrees.mean().item(),
        "rotation_median_deg": degrees.median().item(),
    }


class ErrorAccumulator:
    """Accumulates errors over a split for evaluation reports.

    Pose errors are grouped per target, per target type, and per distance
    band of the viewpoint. Presence is scored as precision and recall per
    type, and confidence as a calibration table: within each bin of predicted
    confidence, the fraction of estimates that really were within tolerance.
    """

    def __init__(
        self,
        target_ids: list[str],
        target_types: list[str],
        tolerance: tuple[float, float],
        bands: tuple[str, ...] = ("home", "far", "mid", "close"),
    ):
        """Start an empty accumulation.

        Args:
            target_ids: Target IDs in query order.
            target_types: Target type per query.
            tolerance: Translation (m) and rotation (rad) counted as accurate.
            bands: Distance band names indexed by the batch ``band`` values.
        """

        self.target_ids = target_ids
        self.target_types = target_types
        self.tolerance = tolerance
        self.bands = bands
        self.translation: list[list[float]] = [[] for _ in target_ids]
        self.rotation: list[list[float]] = [[] for _ in target_ids]
        self.band_of: list[list[int]] = [[] for _ in target_ids]
        self.confidence: list[float] = []
        self.accurate: list[bool] = []
        self.presence_correct = [0 for _ in target_ids]
        self.true_positive = [0 for _ in target_ids]
        self.false_positive = [0 for _ in target_ids]
        self.false_negative = [0 for _ in target_ids]
        self.count = 0

    def add(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> None:
        """Add one evaluated batch.

        Args:
            outputs: Model outputs.
            batch: Labeled batch, with ``band`` per scene.
        """

        translation, rotation = pose_errors(outputs, batch)
        present = batch["present"] > 0.5
        predicted = outputs["presence_logit"] > 0.0
        confidence = torch.sigmoid(outputs["confidence_logit"].float())
        accurate = (translation < self.tolerance[0]) & (rotation < self.tolerance[1])
        bands = batch["band"].tolist()
        correct = (predicted == present).sum(0).tolist()
        true_positive = (predicted & present).sum(0).tolist()
        false_positive = (predicted & ~present).sum(0).tolist()
        false_negative = (~predicted & present).sum(0).tolist()
        for query in range(len(self.target_ids)):
            visible = present[:, query]
            self.translation[query] += translation[visible, query].tolist()
            self.rotation[query] += rotation[visible, query].tolist()
            self.band_of[query] += [
                band for band, seen in zip(bands, visible.tolist()) if seen
            ]
            self.presence_correct[query] += correct[query]
            self.true_positive[query] += true_positive[query]
            self.false_positive[query] += false_positive[query]
            self.false_negative[query] += false_negative[query]
        self.confidence += confidence[present].tolist()
        self.accurate += accurate[present].tolist()
        self.count += present.shape[0]

    def calibration(self) -> list[dict[str, float]]:
        """Fraction of estimates within tolerance per predicted-confidence bin."""

        confidence = torch.tensor(self.confidence)
        accurate = torch.tensor(self.accurate, dtype=torch.float32)
        table = []
        for low, high in zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:]):
            inside = (confidence >= low) & (confidence < high)
            table.append(
                {
                    "confidence": f"{low:.2f}-{min(high, 1.0):.2f}",
                    "samples": int(inside.sum()),
                    "accurate_fraction": (
                        accurate[inside].mean().item() if inside.any() else float("nan")
                    ),
                }
            )
        return table

    def summary(self) -> dict[str, Any]:
        """Errors per target, per type, and per distance band, in mm and degrees."""

        def presence(queries: list[int]) -> dict[str, float]:
            """Presence accuracy, precision, and recall over some queries."""

            tp = sum(self.true_positive[q] for q in queries)
            fp = sum(self.false_positive[q] for q in queries)
            fn = sum(self.false_negative[q] for q in queries)
            return {
                "presence_accuracy": sum(self.presence_correct[q] for q in queries)
                / max(1, self.count * len(queries)),
                "presence_precision": tp / max(1, tp + fp),
                "presence_recall": tp / max(1, tp + fn),
            }

        per_target = {
            target_id: {
                **describe(self.translation[query], self.rotation[query]),
                **presence([query]),
            }
            for query, target_id in enumerate(self.target_ids)
        }
        per_type, per_band = {}, {band: {} for band in self.bands}
        for target_type in dict.fromkeys(self.target_types):
            queries = [q for q, t in enumerate(self.target_types) if t == target_type]
            per_type[target_type] = {
                **describe(
                    [e for q in queries for e in self.translation[q]],
                    [e for q in queries for e in self.rotation[q]],
                ),
                **presence(queries),
            }
            for index, band in enumerate(self.bands):
                per_band[band][target_type] = describe(
                    [
                        e
                        for q in queries
                        for e, b in zip(self.translation[q], self.band_of[q])
                        if b == index
                    ],
                    [
                        e
                        for q in queries
                        for e, b in zip(self.rotation[q], self.band_of[q])
                        if b == index
                    ],
                )
        return {
            "scenes": self.count,
            "per_type": per_type,
            "per_band": per_band,
            "confidence_calibration": self.calibration(),
            "per_target": per_target,
        }
