"""Dataset loading for three-camera AIC policy training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from aic_mujoco.policy import camera_feature_name
from aic_mujoco.utils.videos import decode_rgb_video


@dataclass(frozen=True)
class EpisodeRecord:
    """One complete successful episode available to the trainer."""

    path: Path
    steps: int


def discover_episodes(
    config: dict[str, Any], split: str, minimum_episodes: int
) -> list[EpisodeRecord]:
    """Snapshot complete episodes from one dataset split.

    Args:
        config: Strict merged policy-training configuration.
        split: Dataset split directory to scan.
        minimum_episodes: Required episode count for starting training.

    Returns:
        Sorted immutable episode records.

    Raises:
        FileNotFoundError: If the requested split directory does not exist.
        RuntimeError: If too few complete episodes are available.
        ValueError: If an episode violates the stored dataset contract.
    """

    split_path = Path(config["dataset"]["output_directory"]) / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {split_path}")
    camera_names = config["policy"]["camera_names"]
    instruction = config["dataset"]["instruction"]
    records: list[EpisodeRecord] = []
    for episode_path in sorted(split_path.glob("episode_*")):
        if episode_path.name.endswith(".incomplete"):
            continue
        metadata_path = episode_path / "episode.json"
        trajectory_path = episode_path / "trajectory.npz"
        if not episode_path.is_dir() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_files = [trajectory_path]
        expected_files.extend(episode_path / f"{name}.mp4" for name in camera_names)
        if not all(path.is_file() for path in expected_files):
            raise ValueError(f"Episode is incomplete: {episode_path}")
        if (
            metadata.get("split") != split
            or metadata.get("success") is not True
            or metadata.get("instruction") != instruction
        ):
            raise ValueError(f"Episode metadata is invalid: {metadata_path}")
        steps = metadata.get("steps")
        if type(steps) is not int or steps <= 0:
            raise ValueError(f"Episode step count is invalid: {metadata_path}")
        records.append(EpisodeRecord(episode_path, steps))
    if len(records) < minimum_episodes:
        raise RuntimeError(
            f"{split} has {len(records)} complete episodes; "
            f"training requires {minimum_episodes}"
        )
    return records


def compute_normalization_statistics(
    records: list[EpisodeRecord],
    state_fields: list[str],
    action_field: str,
    action_dimension: int,
    minimum_std: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute train-split state and action mean/std statistics.

    Args:
        records: Complete training episodes.
        state_fields: Ordered trajectory arrays forming robot state.
        action_field: Trajectory array containing expert actions.
        action_dimension: Exact number of controlled joints.
        minimum_std: Strict lower bound for every learned dimension's standard
            deviation.

    Returns:
        Mean and standard deviation arrays for state and action normalization.

    Raises:
        ValueError: If trajectory shapes disagree or a dimension is constant.
    """

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for record in records:
        with np.load(record.path / "trajectory.npz") as trajectory:
            state_parts = [
                np.asarray(trajectory[field], dtype=np.float32)
                for field in state_fields
            ]
            action = np.asarray(trajectory[action_field], dtype=np.float32)
        expected_joint_shape = (record.steps, action_dimension)
        if any(part.shape != expected_joint_shape for part in state_parts):
            raise ValueError(f"State shape is invalid: {record.path}")
        if action.shape != (record.steps, action_dimension):
            raise ValueError(f"Action shape is invalid: {record.path}")
        states.append(np.concatenate(state_parts, axis=1))
        actions.append(action)
    state_values = np.concatenate(states).astype(np.float64)
    action_values = np.concatenate(actions).astype(np.float64)
    statistics = {
        "observation.state": {
            "mean": state_values.mean(axis=0).astype(np.float32),
            "std": state_values.std(axis=0).astype(np.float32),
        },
        "action": {
            "mean": action_values.mean(axis=0).astype(np.float32),
            "std": action_values.std(axis=0).astype(np.float32),
        },
    }
    for feature_name, feature_statistics in statistics.items():
        if np.any(feature_statistics["std"] < minimum_std):
            raise ValueError(
                f"{feature_name} contains a nearly constant dimension below "
                f"training.normalization_minimum_std"
            )
    return statistics


def load_episode(
    record: EpisodeRecord,
    camera_names: list[str],
    state_fields: list[str],
    action_field: str,
    action_dimension: int,
    image_height: int,
    image_width: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Decode one synchronized episode into images, state, and actions.

    Args:
        record: Complete episode to load.
        camera_names: Ordered camera files to decode.
        state_fields: Ordered arrays forming robot state.
        action_field: Expert action array name.
        action_dimension: Exact number of controlled joints.
        image_height: Required stored frame height.
        image_width: Required stored frame width.

    Returns:
        Camera tensors, robot states, and expert actions.

    Raises:
        ValueError: If video and trajectory lengths or shapes disagree.
    """

    images = {
        camera_feature_name(name): decode_rgb_video(record.path / f"{name}.mp4")
        for name in camera_names
    }
    expected_image_shape = (record.steps, 3, image_height, image_width)
    for camera_name, frames in images.items():
        if tuple(frames.shape) != expected_image_shape:
            raise ValueError(
                f"{camera_name} has shape {tuple(frames.shape)}, expected "
                f"{expected_image_shape} in {record.path}"
            )
    with np.load(record.path / "trajectory.npz") as trajectory:
        state_parts = [
            np.asarray(trajectory[field], dtype=np.float32) for field in state_fields
        ]
        action = np.asarray(trajectory[action_field], dtype=np.float32).copy()
    expected_joint_shape = (record.steps, action_dimension)
    if any(part.shape != expected_joint_shape for part in state_parts):
        raise ValueError(f"State shape is invalid: {record.path}")
    if action.shape != (
        record.steps,
        action_dimension,
    ):
        raise ValueError(f"Action shape is invalid: {record.path}")
    state = np.concatenate(state_parts, axis=1).copy()
    return images, torch.from_numpy(state), torch.from_numpy(action)


class TrajectoryDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream shuffled frames while decoding every episode only once per epoch."""

    def __init__(
        self,
        config: dict[str, Any],
        records: list[EpisodeRecord],
        epoch: int,
        shuffle: bool,
    ):
        """Configure one deterministic pass over an episode snapshot.

        Args:
            config: Strict merged policy-training configuration.
            records: Episode snapshot belonging to one split.
            epoch: Epoch number included in deterministic shuffling.
            shuffle: Whether to shuffle episode and frame order.
        """

        super().__init__()
        self.config = config
        self.records = records
        self.epoch = epoch
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Return the number of observation/action samples in this snapshot."""

        return sum(record.steps for record in self.records)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Decode worker-owned episodes and yield aligned ACT samples."""

        training = self.config["training"]
        policy = self.config["policy"]
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rng = np.random.default_rng(training["seed"] + self.epoch)
        order = np.arange(len(self.records))
        if self.shuffle:
            rng.shuffle(order)
        worker_order = order[worker_id::worker_count]
        for record_index in worker_order:
            record = self.records[int(record_index)]
            images, states, actions = load_episode(
                record,
                policy["camera_names"],
                policy["state_fields"],
                policy["action_field"],
                len(self.config["scene"]["names"]["joints"]),
                self.config["dataset"]["image_height"],
                self.config["dataset"]["image_width"],
            )
            frame_order = np.arange(record.steps)
            if self.shuffle:
                rng.shuffle(frame_order)
            for frame_index in frame_order:
                start = int(frame_index)
                stop = min(start + policy["chunk_size"], record.steps)
                valid_steps = stop - start
                action_chunk = torch.zeros(
                    (policy["chunk_size"], actions.shape[1]), dtype=torch.float32
                )
                action_chunk[:valid_steps] = actions[start:stop]
                action_is_pad = torch.ones(policy["chunk_size"], dtype=torch.bool)
                action_is_pad[:valid_steps] = False
                sample = {
                    camera_name: frames[start]
                    for camera_name, frames in images.items()
                }
                sample["observation.state"] = states[start]
                sample["action"] = action_chunk
                sample["action_is_pad"] = action_is_pad
                yield sample
