"""Portable trajectory storage for controller-generated AIC demonstrations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from aic_mujoco.utils.images import resize_rgb_bilinear


SPLIT_NAMES = ("train", "validation", "test")
DATASET_FORMAT_VERSION = 2


class DatasetImageBuffer:
    """Resize all policy images on-device, then download only resized tensors."""

    def __init__(self, config: dict[str, Any], runtime: Any):
        """Allocate one resized RGB tensor for every configured camera.

        Args:
            config: Strict merged collection configuration.
            runtime: Initialized MJWarp runtime containing native RGB tensors.
        """

        native = config["cameras"]
        dataset = config["dataset"]
        self.device = runtime.device
        self.num_envs = runtime.num_envs
        self.source_width = native["width"]
        self.source_height = native["height"]
        self.output_width = dataset["image_width"]
        self.output_height = dataset["image_height"]
        self.images = {
            camera_name: wp.empty(
                (
                    self.num_envs,
                    self.output_height,
                    self.output_width,
                    3,
                ),
                dtype=wp.uint8,
                device=self.device,
            )
            for camera_name in runtime.rgb
        }

    def download(self, source_images: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return one host RGB batch for every named camera."""

        downloaded: dict[str, np.ndarray] = {}
        for camera_name, source in source_images.items():
            output = self.images[camera_name]
            wp.launch(
                resize_rgb_bilinear,
                dim=(self.num_envs, self.output_height, self.output_width),
                inputs=[
                    source,
                    self.source_width,
                    self.source_height,
                    self.output_width,
                    self.output_height,
                ],
                outputs=[output],
                device=self.device,
            )
            downloaded[camera_name] = output.numpy()
        return downloaded


class EpisodeAssignment:
    """One reserved output trajectory and its deterministic scene sample."""

    def __init__(self, split: str, episode_index: int, randomization_id: int):
        """Record a split slot and deterministic randomization identifier.

        Args:
            split: Dataset split name.
            episode_index: Zero-based output episode index.
            randomization_id: Deterministic scene sample identifier.
        """

        self.split = split
        self.episode_index = episode_index
        self.randomization_id = randomization_id


class SyntheticDataset:
    """Own dataset layout, resume state, split allocation, and validation."""

    def __init__(self, config: dict[str, Any]):
        """Prepare or resume the configured dataset directory.

        Args:
            config: Strict merged collection configuration.
        """

        self.config = config
        self.settings = config["dataset"]
        self.root = Path(self.settings["output_directory"])
        self.requested = dict(self.settings["splits"])
        self.completed = {split: set() for split in SPLIT_NAMES}
        self.reserved = {split: set() for split in SPLIT_NAMES}
        self.state_path = self.root / "collection_state.json"
        self.manifest_path = self.root / "manifest.jsonl"
        self.contract_path = self.root / "dataset.json"
        self.next_randomization_id = 0
        self.failed_trajectories = 0
        self.prepare_directory()

    def contract(self) -> dict[str, Any]:
        """Return every configuration value that changes stored demonstrations."""

        dataset = self.config["dataset"]
        return {
            "format_version": DATASET_FORMAT_VERSION,
            "scene": self.config["scene"],
            "physics": self.config["physics"],
            "runtime_seed": self.config["runtime"]["seed"],
            "control": self.config["control"],
            "domain_randomization": self.config["domain_randomization"],
            "sensors": self.config["sensors"],
            "cameras": self.config["cameras"],
            "expert": self.config["expert"],
            "dataset": {
                "instruction": dataset["instruction"],
                "image_width": dataset["image_width"],
                "image_height": dataset["image_height"],
                "video_codec": dataset["video_codec"],
                "video_pixel_format": dataset["video_pixel_format"],
                "video_crf": dataset["video_crf"],
                "splits": dataset["splits"],
            },
        }

    def prepare_directory(self) -> None:
        """Create the dataset layout and recover resumable state.

        Raises:
            FileExistsError: If output exists while resume is disabled.
            ValueError: If an existing dataset has a different contract or
                malformed state.
        """

        if self.root.exists() and not self.settings["resume"]:
            raise FileExistsError(
                f"Dataset directory already exists and resume is disabled: {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        for split in SPLIT_NAMES:
            (self.root / split).mkdir(exist_ok=True)
        (self.root / "failures").mkdir(exist_ok=True)

        expected_contract = self.contract()
        if self.contract_path.exists():
            existing_contract = json.loads(
                self.contract_path.read_text(encoding="utf-8")
            )
            if existing_contract != expected_contract:
                raise ValueError(
                    "Existing dataset contract differs from the collection configuration"
                )
        else:
            self.write_json_atomic(self.contract_path, expected_contract)

        for incomplete in self.root.rglob("*.incomplete"):
            if incomplete.is_dir():
                shutil.rmtree(incomplete)

        self.scan_completed()
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            expected_keys = {"next_randomization_id", "failed_trajectories"}
            if set(state) != expected_keys:
                raise ValueError("Dataset collection_state.json has an invalid shape")
            self.next_randomization_id = int(state["next_randomization_id"])
            self.failed_trajectories = int(state["failed_trajectories"])
        else:
            self.write_state()

    def scan_completed(self) -> None:
        """Discover and validate completed episode directories.

        Raises:
            ValueError: If an episode is malformed, duplicated, or out of its
                configured split range.
        """

        for split in SPLIT_NAMES:
            for episode_directory in (self.root / split).glob("episode_*"):
                if not episode_directory.is_dir():
                    continue
                metadata_path = episode_directory / "episode.json"
                trajectory_path = episode_directory / "trajectory.npz"
                if not metadata_path.is_file() or not trajectory_path.is_file():
                    raise ValueError(
                        f"Completed episode is missing metadata or state: {episode_directory}"
                    )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("split") != split or metadata.get("success") is not True:
                    raise ValueError(f"Invalid completed episode metadata: {metadata_path}")
                episode_index = int(metadata["episode_index"])
                if episode_index < 0 or episode_index >= self.requested[split]:
                    raise ValueError(
                        f"Episode index is outside configured {split} split: {episode_index}"
                    )
                if episode_index in self.completed[split]:
                    raise ValueError(f"Duplicate {split} episode index: {episode_index}")
                self.completed[split].add(episode_index)

    def write_state(self) -> None:
        """Atomically persist resumable collection counters."""

        self.write_json_atomic(
            self.state_path,
            {
                "next_randomization_id": self.next_randomization_id,
                "failed_trajectories": self.failed_trajectories,
            },
        )

    @staticmethod
    def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        """Write one JSON object with atomic replacement.

        Args:
            path: Destination JSON path.
            value: JSON-serializable object to write.
        """

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def reserve_randomization_id(self) -> int:
        """Reserve and persist the next deterministic scene sample ID.

        Returns:
            The reserved nonnegative identifier.
        """

        randomization_id = self.next_randomization_id
        self.next_randomization_id += 1
        self.write_state()
        return randomization_id

    def next_assignment(self) -> EpisodeAssignment | None:
        """Reserve the next missing trajectory in split order.

        Returns:
            A new assignment, or ``None`` when all slots are reserved.
        """

        for split in SPLIT_NAMES:
            unavailable = self.completed[split] | self.reserved[split]
            for episode_index in range(self.requested[split]):
                if episode_index not in unavailable:
                    self.reserved[split].add(episode_index)
                    return EpisodeAssignment(
                        split,
                        episode_index,
                        self.reserve_randomization_id(),
                    )
        return None

    def retry_assignment(self, assignment: EpisodeAssignment) -> EpisodeAssignment:
        """Resample a failed split slot with a new randomization ID.

        Args:
            assignment: Failed assignment whose output slot is retained.

        Returns:
            Replacement assignment for the same split and episode index.
        """

        return EpisodeAssignment(
            assignment.split,
            assignment.episode_index,
            self.reserve_randomization_id(),
        )

    def mark_completed(self, assignment: EpisodeAssignment, metadata: dict[str, Any]) -> None:
        """Record a successfully finalized episode in memory and the manifest.

        Args:
            assignment: Completed dataset assignment.
            metadata: Episode metadata written to the manifest.
        """

        self.completed[assignment.split].add(assignment.episode_index)
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(metadata, sort_keys=True) + "\n")

    def mark_failed(self, metadata: dict[str, Any]) -> int:
        """Record one failed trajectory and enforce the configured limit.

        Args:
            metadata: Failed episode metadata.

        Returns:
            Zero-based failure index used for optional storage.

        Raises:
            RuntimeError: If failures exceed the configured maximum.
        """

        failure_index = self.failed_trajectories
        self.failed_trajectories += 1
        self.write_state()
        failure_manifest = self.root / "failures.jsonl"
        with failure_manifest.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(metadata, sort_keys=True) + "\n")
        if self.failed_trajectories > self.settings["maximum_failed_trajectories"]:
            raise RuntimeError(
                "Expert exceeded dataset.maximum_failed_trajectories; inspect failures"
            )
        return failure_index

    def counts(self) -> dict[str, int]:
        """Return completed trajectory counts by split."""

        return {split: len(self.completed[split]) for split in SPLIT_NAMES}

    def is_complete(self) -> bool:
        """Return whether every requested dataset slot is complete."""

        return all(
            len(self.completed[split]) == self.requested[split]
            for split in SPLIT_NAMES
        )

    def validate(self) -> None:
        """Validate every stored episode against the dataset contract.

        Raises:
            RuntimeError: If collection has not completed.
            ValueError: If arrays, metadata, or videos are inconsistent.
        """

        if not self.is_complete():
            raise RuntimeError(
                f"Dataset is incomplete: completed={self.counts()}, requested={self.requested}"
            )
        expected_cameras = set(self.config["scene"]["names"]["cameras"])
        for split in SPLIT_NAMES:
            for episode_index in range(self.requested[split]):
                path = self.root / split / f"episode_{episode_index:06d}"
                metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
                with np.load(path / "trajectory.npz") as trajectory:
                    keys = set(trajectory.files)
                    required = {
                        "qpos",
                        "qvel",
                        "action_delta_q",
                        "sfp_tip_position",
                        "sfp_tip_rotation_matrix",
                        "goal_position",
                        "goal_rotation_matrix",
                        "position_error_m",
                        "orientation_error_rad",
                        "tared_wrench",
                    }
                    if keys != required:
                        raise ValueError(f"Unexpected trajectory arrays in {path}")
                    sample_count = int(trajectory["qpos"].shape[0])
                    if any(trajectory[key].shape[0] != sample_count for key in required):
                        raise ValueError(f"Misaligned trajectory arrays in {path}")
                if metadata["steps"] != sample_count:
                    raise ValueError(f"Metadata step count differs in {path}")
                videos = {video.stem for video in path.glob("*.mp4")}
                if videos != expected_cameras:
                    raise ValueError(f"Episode camera videos differ in {path}")


class EpisodeWriter:
    """Write one synchronized observation/action trajectory atomically."""

    def __init__(
        self,
        dataset: SyntheticDataset,
        assignment: EpisodeAssignment,
        reset_state: dict[str, Any],
    ):
        """Create one temporary episode directory and its video streams.

        Args:
            dataset: Dataset that owns this episode.
            assignment: Reserved output slot and randomization identifier.
            reset_state: Per-environment reset metadata.
        """

        import imageio.v2 as imageio

        self.dataset = dataset
        self.assignment = assignment
        self.reset_state = reset_state
        self.steps = 0
        self.success_steps = 0
        self.arrays: dict[str, list[np.ndarray | float]] = {
            "qpos": [],
            "qvel": [],
            "action_delta_q": [],
            "sfp_tip_position": [],
            "sfp_tip_rotation_matrix": [],
            "goal_position": [],
            "goal_rotation_matrix": [],
            "position_error_m": [],
            "orientation_error_rad": [],
            "tared_wrench": [],
        }
        split_directory = dataset.root / assignment.split
        self.incomplete_path = (
            split_directory / f"episode_{assignment.episode_index:06d}.incomplete"
        )
        if self.incomplete_path.exists():
            shutil.rmtree(self.incomplete_path)
        self.incomplete_path.mkdir()
        settings = dataset.settings
        self.video_writers = {
            camera_name: imageio.get_writer(
                self.incomplete_path / f"{camera_name}.mp4",
                fps=dataset.config["expert"]["control_hz"],
                codec=settings["video_codec"],
                pixelformat=settings["video_pixel_format"],
                ffmpeg_params=["-crf", str(settings["video_crf"])],
                macro_block_size=None,
            )
            for camera_name in dataset.config["scene"]["names"]["cameras"]
        }

    def append(
        self,
        frames: dict[str, np.ndarray],
        sample: dict[str, np.ndarray | float],
    ) -> None:
        """Append one synchronized observation/action sample.

        Args:
            frames: RGB frame for every configured camera.
            sample: State, action, pose, error, and wrench values.
        """

        for camera_name, writer in self.video_writers.items():
            writer.append_data(frames[camera_name])
        for key in self.arrays:
            self.arrays[key].append(np.asarray(sample[key]).copy())
        self.steps += 1

    def close_videos(self) -> None:
        """Close every open camera video writer."""

        for writer in self.video_writers.values():
            writer.close()
        self.video_writers.clear()

    def metadata(self, success: bool, reason: str) -> dict[str, Any]:
        """Build portable metadata for the current episode.

        Args:
            success: Whether the teacher reached its pose tolerance.
            reason: Explicit termination reason.

        Returns:
            JSON-serializable episode metadata.
        """

        return {
            "split": self.assignment.split,
            "episode_index": self.assignment.episode_index,
            "randomization_id": self.assignment.randomization_id,
            "success": success,
            "termination": reason,
            "steps": self.steps,
            "instruction": self.dataset.settings["instruction"],
            "control_hz": self.dataset.config["expert"]["control_hz"],
            "pose_frame": "MJCF world",
            "reset": json_value(self.reset_state),
            "final_position_error_m": float(self.arrays["position_error_m"][-1]),
            "final_orientation_error_rad": float(
                self.arrays["orientation_error_rad"][-1]
            ),
        }

    def finish(self, success: bool, reason: str) -> None:
        """Atomically finalize, register, or discard this trajectory.

        Args:
            success: Whether the trajectory reached its goal.
            reason: Explicit termination reason.

        Raises:
            FileExistsError: If the final episode path already exists.
            RuntimeError: If no samples were appended.
        """

        self.close_videos()
        if self.steps == 0:
            raise RuntimeError("Cannot finish an empty trajectory")
        arrays = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in self.arrays.items()
        }
        np.savez_compressed(self.incomplete_path / "trajectory.npz", **arrays)
        metadata = self.metadata(success, reason)
        SyntheticDataset.write_json_atomic(
            self.incomplete_path / "episode.json", metadata
        )

        if success:
            completed_path = (
                self.dataset.root
                / self.assignment.split
                / f"episode_{self.assignment.episode_index:06d}"
            )
            if completed_path.exists():
                raise FileExistsError(f"Episode already exists: {completed_path}")
            self.incomplete_path.rename(completed_path)
            self.dataset.mark_completed(self.assignment, metadata)
            return

        failure_index = self.dataset.mark_failed(metadata)
        if self.dataset.settings["keep_failed_trajectories"]:
            failure_path = (
                self.dataset.root / "failures" / f"failure_{failure_index:06d}"
            )
            self.incomplete_path.rename(failure_path)
        else:
            shutil.rmtree(self.incomplete_path)

    def close_incomplete(self) -> None:
        """Close resources while leaving the episode uncommitted."""

        self.close_videos()


def json_value(value: Any) -> Any:
    """Convert nested NumPy values into JSON-serializable Python values.

    Args:
        value: Arbitrarily nested supported value.

    Returns:
        Equivalent value containing only JSON-compatible containers/scalars.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value
