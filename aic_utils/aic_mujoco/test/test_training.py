"""Tests for supervised three-camera manipulation-policy training."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from aic_mujoco.config import load_training_config
from aic_mujoco.policy import camera_feature_name, create_policy
from aic_mujoco.training import train
from aic_mujoco.training_data import (
    TrajectoryDataset,
    compute_normalization_statistics,
    discover_episodes,
    load_episode,
)


BASE = PACKAGE / "configs" / "base.json"
COLLECT = PACKAGE / "configs" / "collect.json"
TRAIN = PACKAGE / "configs" / "train.json"


def write_test_video(path: Path, frames: np.ndarray) -> None:
    """Write a small deterministic MP4 for loader verification."""

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def create_test_episode(config: dict, root: Path, split: str = "train") -> Path:
    """Create one complete episode following the collector's storage contract."""

    steps = 5
    episode = root / split / "episode_000000"
    episode.mkdir(parents=True)
    frame_values = np.arange(steps, dtype=np.uint8)[:, None, None, None]
    frame_shape = (
        steps,
        config["dataset"]["image_height"],
        config["dataset"]["image_width"],
        3,
    )
    frames = np.broadcast_to(frame_values, frame_shape).copy()
    for camera_name in config["policy"]["camera_names"]:
        write_test_video(episode / f"{camera_name}.mp4", frames)
    time_values = np.arange(steps, dtype=np.float32)[:, None]
    joint_offsets = np.arange(6, dtype=np.float32)[None, :] * 0.01
    np.savez_compressed(
        episode / "trajectory.npz",
        qpos=time_values + joint_offsets,
        qvel=(time_values + 1.0) * 0.1 + joint_offsets,
        action_delta_q=(time_values + 1.0) * 0.01 + joint_offsets,
    )
    metadata = {
        "split": split,
        "success": True,
        "instruction": config["dataset"]["instruction"],
        "steps": steps,
    }
    (episode / "episode.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return episode


def test_training_config_is_strict(tmp_path: Path) -> None:
    """Reject unknown policy behavior instead of supplying defaults."""

    training = json.loads(TRAIN.read_text(encoding="utf-8"))
    training["policy"]["camera_names"] = ["left", "center", "right"]
    invalid = tmp_path / "invalid_training.json"
    invalid.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="must list center, left, and right"):
        load_training_config(BASE, COLLECT, invalid)


def test_training_dataset_decodes_aligned_action_chunks(tmp_path: Path) -> None:
    """Decode camera/state data and pad only the end of action chunks."""

    config = copy.deepcopy(load_training_config(BASE, COLLECT, TRAIN))
    config["dataset"]["output_directory"] = str(tmp_path / "dataset")
    config["dataset"]["image_width"] = 16
    config["dataset"]["image_height"] = 16
    config["policy"]["chunk_size"] = 4
    complete = create_test_episode(
        config, Path(config["dataset"]["output_directory"])
    )
    shutil.copytree(complete, complete.parent / "episode_000001.incomplete")

    records = discover_episodes(config, "train", 1)
    assert len(records) == 1
    joint_count = len(config["scene"]["names"]["joints"])
    statistics = compute_normalization_statistics(
        records,
        config["policy"]["state_fields"],
        config["policy"]["action_field"],
        joint_count,
        config["training"]["normalization_minimum_std"],
    )
    assert statistics["observation.state"]["mean"].shape == (12,)
    assert statistics["action"]["std"].shape == (6,)

    images, states, actions = load_episode(
        records[0],
        config["policy"]["camera_names"],
        config["policy"]["state_fields"],
        config["policy"]["action_field"],
        joint_count,
        config["dataset"]["image_height"],
        config["dataset"]["image_width"],
    )
    assert images[camera_feature_name("center")].shape == (5, 3, 16, 16)
    assert states.shape == (5, 12)
    assert actions.shape == (5, 6)

    samples = list(TrajectoryDataset(config, records, epoch=0, shuffle=False))
    assert len(samples) == 5
    torch.testing.assert_close(samples[0]["action"][:4], actions[:4])
    torch.testing.assert_close(samples[-1]["action"][0], actions[-1])
    assert samples[-1]["action_is_pad"].tolist() == [False, True, True, True]


def test_act_policy_computes_a_supervised_loss_on_cpu() -> None:
    """Exercise the upstream ACT forward and backward training contract."""

    config = copy.deepcopy(load_training_config(BASE, COLLECT, TRAIN))
    config["training"].update({"device": "cpu", "use_amp": False})
    config["dataset"].update({"image_height": 64, "image_width": 64})
    config["policy"].update(
        {
            "pretrained_backbone_weights": None,
            "chunk_size": 4,
            "n_action_steps": 1,
            "dim_model": 64,
            "n_heads": 4,
            "dim_feedforward": 128,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "latent_dim": 8,
            "n_vae_encoder_layers": 1,
        }
    )
    policy = create_policy(config)
    batch = {
        camera_feature_name(name): torch.randn(1, 3, 64, 64)
        for name in config["policy"]["camera_names"]
    }
    batch["observation.state"] = torch.randn(1, 12)
    batch["action"] = torch.randn(1, 4, 6)
    batch["action_is_pad"] = torch.zeros(1, 4, dtype=torch.bool)

    loss, components = policy(batch)
    assert torch.isfinite(loss)
    assert set(components) == {"l1_loss", "kld_loss"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_one_step_training_writes_an_atomic_checkpoint(tmp_path: Path) -> None:
    """Exercise the complete train, validate, log, and save lifecycle."""

    config = copy.deepcopy(load_training_config(BASE, COLLECT, TRAIN))
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "run"
    config["dataset"].update(
        {
            "output_directory": str(dataset_root),
            "image_height": 64,
            "image_width": 64,
        }
    )
    config["training"].update(
        {
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "steps": 1,
            "minimum_train_episodes": 1,
            "minimum_validation_episodes": 1,
            "log_every_steps": 1,
            "validate_every_steps": 1,
            "validation_batches": 1,
            "checkpoint_every_steps": 1,
            "output_directory": str(output_root),
            "use_amp": False,
        }
    )
    config["policy"].update(
        {
            "pretrained_backbone_weights": None,
            "chunk_size": 4,
            "n_action_steps": 1,
            "dim_model": 64,
            "n_heads": 4,
            "dim_feedforward": 128,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "latent_dim": 8,
            "n_vae_encoder_layers": 1,
        }
    )
    config["wandb"].update({"mode": "disabled", "watch": "none"})
    create_test_episode(config, dataset_root, "train")
    create_test_episode(config, dataset_root, "validation")

    train(config)

    checkpoint = output_root / "checkpoints" / "step_00000001"
    assert (output_root / "config.json").is_file()
    assert (output_root / "normalization.json").is_file()
    assert (checkpoint / "policy" / "model.safetensors").is_file()
    assert (checkpoint / "normalization.json").is_file()
    assert not checkpoint.with_name(checkpoint.name + ".incomplete").exists()
