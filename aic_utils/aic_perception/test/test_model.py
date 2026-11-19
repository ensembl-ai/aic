"""Model, rotation representation, and loss behavior."""

from __future__ import annotations

import math

import pytest
import torch

from aic_perception.config import load_config
from aic_perception.losses import ErrorAccumulator, compute_losses
from aic_perception.model import SceneEstimator, camera_rays
from aic_perception.utils.rotations import angle_between, matrix_from_6d

CONFIG = load_config("train")
TARGETS = len(CONFIG["schema"]["targets"])


BACKBONE_INPUTS = {"resnet18": [], "dinov2_vits14": [56, 70]}


def small_model(backbone: str = "resnet18") -> SceneEstimator:
    """A tiny untrained estimator for shape and gradient checks."""

    settings = dict(
        CONFIG["model"],
        backbone=backbone,
        backbone_input=BACKBONE_INPUTS[backbone],
        pretrained_backbone=False,
        d_model=32,
        heads=4,
        feedforward=64,
    )
    return SceneEstimator(settings, TARGETS, [-0.4, 0.3, 0.1], [0.1, 0.1, 0.05])


def batch(size: int = 2) -> dict[str, torch.Tensor]:
    """Random batch with the dataset's tensor layout."""

    intrinsics = torch.tensor([[60.0, 0.0, 31.5], [0.0, 60.0, 31.5], [0.0, 0.0, 1.0]])
    poses = torch.eye(4).repeat(size, TARGETS, 1, 1)
    poses[..., :3, 3] = torch.randn(size, TARGETS, 3) * 0.1
    present = (torch.rand(size, TARGETS) > 0.5).float()
    present[:, 0] = 1.0
    return {
        "images": torch.randint(0, 255, (size, 3, 3, 64, 64), dtype=torch.uint8),
        "intrinsics": intrinsics.repeat(size, 3, 1, 1),
        "base_from_camera": torch.eye(4).repeat(size, 3, 1, 1),
        "present": present,
        "base_from_target": poses,
        "band": torch.arange(size) % 4,
    }


def test_rotation_6d_produces_proper_rotations() -> None:
    """Any 6D input maps to an orthonormal, right-handed matrix."""

    rotation = matrix_from_6d(torch.randn(100, 6))
    identity = torch.eye(3).expand(100, 3, 3)
    assert torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(100), atol=1e-5)
    assert torch.allclose(
        angle_between(rotation, rotation), torch.zeros(100), atol=1e-3
    )


def test_rotation_angle_is_geodesic() -> None:
    """A rotation about z by theta is theta away from identity."""

    theta = 0.7
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert math.isclose(
        angle_between(rotation, torch.eye(3)).item(), theta, rel_tol=1e-4
    )


def test_center_ray_follows_optical_axis() -> None:
    """The ray through the principal point is the optical z axis in the base frame."""

    intrinsics = torch.tensor([[[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]])
    pose = torch.eye(4)[None]
    pose[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    origins, directions = camera_rays(intrinsics, pose, 1, 1, (2.0, 2.0))
    assert torch.allclose(directions[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(origins[0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_rays_use_separate_strides_per_axis() -> None:
    """Cell centres follow the horizontal and vertical stride independently."""

    intrinsics = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    _, directions = camera_rays(intrinsics, torch.eye(4)[None], 1, 2, (4.0, 2.0))
    expected = torch.nn.functional.normalize(
        torch.tensor([[1.5, 0.5, 1.0], [5.5, 0.5, 1.0]]), dim=-1
    )
    assert torch.allclose(directions[0], expected)


@pytest.mark.parametrize("backbone", sorted(BACKBONE_INPUTS))
def test_forward_shapes_and_backward(backbone: str) -> None:
    """Outputs cover every query and the loss trains every parameter group."""

    model = small_model(backbone)
    data = batch()
    outputs = model(data["images"], data["intrinsics"], data["base_from_camera"])
    assert outputs["presence_logit"].shape == (2, TARGETS)
    assert outputs["translation"].shape == (2, TARGETS, 3)
    assert outputs["rotation"].shape == (2, TARGETS, 3, 3)
    loss, values = compute_losses(outputs, data, CONFIG["loss"])
    assert torch.isfinite(loss) and all(math.isfinite(v) for v in values.values())
    loss.backward()
    assert all(p.grad is not None for p in model.head_parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone_parameters()
    )


def test_error_accumulator_reports_every_target() -> None:
    """Evaluation summaries cover every target and type."""

    model = small_model().eval()
    data = batch(4)
    with torch.no_grad():
        outputs = model(data["images"], data["intrinsics"], data["base_from_camera"])
    types = [target["type"] for target in CONFIG["schema"]["targets"]]
    errors = ErrorAccumulator(
        [t["id"] for t in CONFIG["schema"]["targets"]],
        types,
        (0.005, math.radians(5.0)),
    )
    errors.add(outputs, data)
    summary = errors.summary()
    assert summary["scenes"] == 4
    assert set(summary["per_type"]) == set(types)
    assert summary["per_target"]["task_board"]["samples"] == 4
    assert {
        band: summary["per_band"][band]["board"]["samples"]
        for band in summary["per_band"]
    } == {
        "home": 1,
        "far": 1,
        "mid": 1,
        "close": 1,
    }
    calibration = summary["confidence_calibration"]
    assert sum(row["samples"] for row in calibration) == int(data["present"].sum())


def test_old_checkpoints_load_into_the_selectable_backbone(tmp_path) -> None:
    """Checkpoints saved before selectable backbones keep loading as ResNet-18."""

    from aic_perception.training import load_checkpoint

    model = small_model()
    state = {
        key.replace("backbone.encoder.", "backbone.", 1): value
        for key, value in model.state_dict().items()
        if key not in ("backbone.mean", "backbone.std")
    }
    state["image_mean"] = torch.zeros(1, 3, 1, 1)
    state["image_std"] = torch.ones(1, 3, 1, 1)
    old_model = {
        k: v
        for k, v in CONFIG["model"].items()
        if k not in ("backbone", "backbone_input")
    }
    old_model |= {"d_model": 32, "heads": 4, "feedforward": 64}
    torch.save(
        {
            "model": state,
            "config": {**CONFIG, "model": old_model},
            "normalization": {"mean": [-0.4, 0.3, 0.1], "std": [0.1, 0.1, 0.05]},
        },
        tmp_path / "old.pt",
    )
    loaded, config = load_checkpoint(tmp_path / "old.pt", "cpu")
    assert config["model"]["backbone"] == "resnet18"
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_early_stopping_waits_for_a_real_plateau() -> None:
    """Stopping needs `patience` epochs without a `min_delta` gain; 0 disables it."""

    from aic_perception.training import should_stop

    improving = [10.0 - 0.5 * i for i in range(15)]
    assert not should_stop(improving, 10, 0.02)
    plateau = [5.0, 4.0, 3.0] + [3.01] * 10
    assert should_stop(plateau, 10, 0.02)
    assert not should_stop(plateau[:-1], 10, 0.02)
    assert should_stop([3.0] + [2.99] * 10, 10, 0.02)
    assert not should_stop(plateau, 0, 0.02)
