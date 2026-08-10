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

from aic_mujoco.config import load_evaluation_config, load_training_config
from aic_mujoco.evaluation import evaluate_action_prediction
from aic_mujoco.policy import (
    PolicyNormalizer,
    camera_feature_name,
    create_policy,
    load_policy_checkpoint,
)
from aic_mujoco.training import train
from aic_mujoco.training_data import (
    TrajectoryDataset,
    compute_normalization_statistics,
    discover_episodes,
    load_episode,
)
from aic_mujoco.utils.metrics import (
    RolloutMetricsRecorder,
    TrainingMetricsRecorder,
    WorkspaceSample,
)

BASE = PACKAGE / "configs" / "base.json"
COLLECT = PACKAGE / "configs" / "collect.json"
TRAIN = PACKAGE / "configs" / "train.json"
EVALUATE = PACKAGE / "configs" / "evaluate.json"


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


def test_evaluation_config_requires_one_explicit_visual_world(
    tmp_path: Path,
) -> None:
    """Reject an inference overlay that silently changes world cardinality."""

    evaluation = json.loads(EVALUATE.read_text(encoding="utf-8"))
    evaluation["runtime"]["num_envs"] = 2
    invalid = tmp_path / "invalid_evaluation.json"
    invalid.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime.num_envs=1"):
        load_evaluation_config(BASE, COLLECT, TRAIN, invalid)


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


def test_local_training_and_workspace_metrics_are_incremental(
    tmp_path: Path,
) -> None:
    """Persist scalar history and compute an exact SFP workspace path length."""

    training = TrainingMetricsRecorder(tmp_path / "training_run")
    training.record(10, "train", {"train/loss": 0.5})
    training.record(10, "validation", {"validation/action_mae_rad": 0.01})
    training.finish({"status": "completed", "final_step": 10})
    history = (
        tmp_path / "training_run" / "metrics" / "training" / "history.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(history) == 2

    config = copy.deepcopy(load_evaluation_config(BASE, COLLECT, TRAIN, EVALUATE))
    checkpoint = tmp_path / "policy_run" / "checkpoints" / "step_00000010"
    config["evaluation"]["checkpoint_directory"] = str(checkpoint)
    rollout = RolloutMetricsRecorder(config)
    rollout.start_episode(1)
    first = WorkspaceSample(
        sfp_position=np.asarray([0.0, 0.0, 0.0]),
        goal_position=np.asarray([1.0, 0.0, 0.0]),
        position_error=1.0,
        orientation_error=0.2,
    )
    second = WorkspaceSample(
        sfp_position=np.asarray([0.03, 0.04, 0.0]),
        goal_position=np.asarray([1.0, 0.0, 0.0]),
        position_error=0.9,
        orientation_error=0.1,
    )
    rollout.record(1, first)
    rollout.record(2, second)
    episode = rollout.finish_episode("timeout", second)
    rollout.close("completed")

    assert episode["workspace_path_length_m"] == pytest.approx(0.05)
    assert episode["position_progress_m"] == pytest.approx(0.1)
    assert episode["minimum_position_error_m"] == pytest.approx(0.9)
    assert episode["final_error_increase_from_closest_m"] == pytest.approx(0.0)
    summary = json.loads(
        (rollout.directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["timeout_count"] == 1
    assert summary["success_rate"] == 0.0
    assert summary["positive_position_progress_rate"] == 1.0
    assert summary["final_at_closest_rate"] == 1.0
    assert len(
        (rollout.directory / "workspace_path.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2


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

    config["evaluation"] = {
        "checkpoint_directory": str(checkpoint),
        "dataset_split": "validation",
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": 1,
        "metrics_output": str(tmp_path / "metrics.json"),
        "maximum_episode_steps": 500,
        "reset_pause_seconds": 0.0,
    }
    config["expert"]["maximum_episode_steps"] = 500
    loaded_policy, loaded_statistics = load_policy_checkpoint(config)
    validation_records = discover_episodes(config, "validation", 1)
    metrics = evaluate_action_prediction(
        config,
        loaded_policy,
        PolicyNormalizer(config, loaded_statistics),
        validation_records,
        batch_size=1,
        num_workers=0,
        prefetch_factor=1,
        maximum_batches=None,
    )
    assert metrics["observations"] == 5
    assert metrics["action_vectors"] > 0
    assert np.isfinite(metrics["action_mae_rad"])
