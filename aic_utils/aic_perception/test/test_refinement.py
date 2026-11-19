"""Close-range refiner targets, crops, and model."""

from __future__ import annotations

import math

import numpy as np
import torch

from aic_perception.config import load_config
from aic_perception.refinement import (
    Refiner,
    crop,
    crop_windows,
    label_sample,
    refiner_losses,
)
from aic_perception.utils.geometry import from_xyz_rpy, invert

CONFIG = load_config("refine")
CAMERAS = CONFIG["schema"]["cameras"]


def synthetic_label(plug: np.ndarray, port: np.ndarray, camera: np.ndarray) -> dict:
    """Label with one close view, the same camera pose for every camera."""

    intrinsics = [[500.0, 0.0, 288.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]]
    return {
        "scene": {"viewpoint": {"mode": "close", "height": 0.02}, "target": "port"},
        "held_plug": {"plug_type": "sfp", "base_from_plug": plug.tolist()},
        "targets": {"port": {"spawned": True, "base_from_target": port.tolist()}},
        "cameras": {
            name: {"base_from_camera": camera.tolist(), "intrinsics": intrinsics}
            for name in CAMERAS
        },
    }


def test_targets_are_expressed_in_the_reference_camera() -> None:
    """Offset, plug position, and rotation map back to the base-frame truth."""

    camera = from_xyz_rpy((0.1, -0.2, 0.5), (math.pi, 0.1, 0.3))
    plug = from_xyz_rpy((0.12, -0.18, 0.2), (math.pi, 0.0, 0.2))
    port = from_xyz_rpy((0.125, -0.176, 0.18), (math.pi, 0.0, 0.25))
    sample = label_sample(synthetic_label(plug, port, camera), "center")
    rotation = camera[:3, :3]
    assert np.allclose(rotation @ sample["offset"], port[:3, 3] - plug[:3, 3])
    assert np.allclose(rotation @ sample["plug"] + camera[:3, 3], plug[:3, 3])
    turned = rotation @ sample["rotation"]
    axis = turned / np.linalg.norm(turned)
    angle = np.linalg.norm(turned)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rodrigues = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
    assert np.allclose(rodrigues @ plug[:3, :3], port[:3, :3], atol=1e-6)
    assert math.isclose(sample["height"], 0.02, abs_tol=1e-9)
    assert math.isclose(sample["lateral"], math.hypot(0.005, 0.004), abs_tol=1e-9)
    point = invert(camera) @ np.append(plug[:3, 3], 1.0)
    assert np.allclose(
        sample["pixels"]["center"],
        [500 * point[0] / point[2] + 288, 500 * point[1] / point[2] + 256],
    )


def test_samples_without_a_plug_or_at_home_are_skipped() -> None:
    """HOME views and labels without a plug pose give no refiner sample."""

    label = synthetic_label(np.eye(4), np.eye(4), np.eye(4))
    label["held_plug"]["base_from_plug"] = None
    assert label_sample(label, "center") is None
    label = synthetic_label(np.eye(4), np.eye(4), np.eye(4))
    label["scene"]["viewpoint"] = {"mode": "home"}
    assert label_sample(label, "center") is None


def test_crop_windows_centre_the_plug_and_stay_inside() -> None:
    """Windows centre the median plug pixel, shifted inside the image at the border."""

    samples = [
        {"plug_type": plug_type, "pixels": {name: [300.0, v] for name in CAMERAS}}
        for plug_type in ("sfp", "sc")
        for v in (240.0, 250.0, 260.0, 480.0, 490.0)
    ]
    windows = crop_windows(samples, CAMERAS, 576, 512, 100)
    assert windows["sfp"]["center"] == [250, 210]
    image = torch.zeros(3, 512, 576, dtype=torch.uint8)
    assert crop(image, [476, 412], 100).shape == (3, 100, 100)
    tall = [
        {"plug_type": t, "pixels": {name: [300.0, 500.0] for name in CAMERAS}}
        for t in ("sfp", "sc")
    ]
    assert crop_windows(tall, CAMERAS, 576, 512, 100)["sc"]["left"] == [250, 412]


def test_refiner_forward_and_backward() -> None:
    """The refiner trains end to end and outputs metric targets."""

    settings = dict(
        CONFIG["model"],
        pretrained_backbone=False,
        backbone_input=[56, 56],
        d_model=32,
        heads=4,
        feedforward=64,
    )
    statistics = {
        "offset_mean": [0.0, -0.015, 0.02],
        "offset_std": [0.01] * 3,
        "plug_mean": [0.0, 0.03, 0.27],
        "plug_std": [0.01] * 3,
    }
    model = Refiner(settings, len(CAMERAS), 56, statistics)
    batch = {
        "crops": torch.randint(0, 255, (2, len(CAMERAS), 3, 56, 56), dtype=torch.uint8),
        "plug_index": torch.tensor([0, 1]),
        "offset": torch.randn(2, 3) * 0.01,
        "plug": torch.randn(2, 3) * 0.01,
        "rotation": torch.randn(2, 3) * 0.02,
    }
    outputs = model(batch["crops"], batch["plug_index"])
    assert all(outputs[key].shape == (2, 3) for key in ("offset", "plug", "rotation"))
    loss, values = refiner_losses(outputs, batch, CONFIG["loss"])
    loss.backward()
    assert math.isfinite(values["loss/total"])
    assert all(p.grad is not None for p in model.head_parameters())
