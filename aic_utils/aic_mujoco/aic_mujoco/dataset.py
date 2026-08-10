"""Resumable two-stage storage for synthetic AIC demonstrations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from aic_mujoco.utils.images import resize_rgb_bilinear


SPLIT_NAMES = ("train", "validation", "test")
DATASET_FORMAT_VERSION = 3
TRAJECTORY_SHAPES = {
    "qpos": (6,),
    "qvel": (6,),
    "action_delta_q": (6,),
    "sfp_tip_position": (3,),
    "sfp_tip_rotation_matrix": (3, 3),
    "goal_position": (3,),
    "goal_rotation_matrix": (3, 3),
    "position_error_m": (),
    "orientation_error_rad": (),
    "tared_wrench": (6,),
}


class DatasetImageBuffer:
    """Resize replay images on-device, then download only policy resolution."""

    def __init__(self, config: dict[str, Any], runtime: Any):
        """Allocate one resized RGB tensor for every configured camera.

        Args:
            config: Strict merged collection configuration.
            runtime: Camera-enabled MJWarp replay runtime.

        Raises:
            ValueError: If the supplied runtime has no RGB outputs.
        """

        if not runtime.rgb:
            raise ValueError("Dataset image buffers require a camera-enabled runtime")
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
        """Resize and return one host RGB batch for every named camera."""

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
    """One reserved output episode and its deterministic scene sample."""

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
    """Own rollout staging, RGB export, resume state, and validation."""

    def __init__(self, config: dict[str, Any]):
        """Prepare or resume the configured two-stage dataset directory.

        Args:
            config: Strict merged collection configuration.
        """

        self.config = config
        self.settings = config["dataset"]
        self.root = Path(self.settings["output_directory"])
        self.rollouts_root = self.root / "rollouts"
        self.requested = dict(self.settings["splits"])
        self.completed = {split: set() for split in SPLIT_NAMES}
        self.rollout_ready = {split: set() for split in SPLIT_NAMES}
        self.reserved = {split: set() for split in SPLIT_NAMES}
        self.state_path = self.root / "collection_state.json"
        self.manifest_path = self.root / "manifest.jsonl"
        self.contract_path = self.root / "dataset.json"
        self.next_randomization_id = 0
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
        """Create the two-stage layout and recover resumable state.

        Raises:
            FileExistsError: If output exists while resume is disabled.
            ValueError: If existing data violates the configured contract.
        """

        if self.root.exists() and not self.settings["resume"]:
            raise FileExistsError(
                f"Dataset directory already exists and resume is disabled: {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.rollouts_root.mkdir(exist_ok=True)
        for split in SPLIT_NAMES:
            (self.root / split).mkdir(exist_ok=True)
            (self.rollouts_root / split).mkdir(exist_ok=True)

        expected_contract = self.contract()
        if self.contract_path.exists():
            existing_contract = json.loads(
                self.contract_path.read_text(encoding="utf-8")
            )
            if existing_contract != expected_contract:
                raise ValueError(
                    "Existing dataset contract differs from the collection "
                    "configuration"
                )
        else:
            self.write_json_atomic(self.contract_path, expected_contract)

        for incomplete in self.root.rglob("*.incomplete"):
            if incomplete.is_dir():
                shutil.rmtree(incomplete)

        self.scan_completed()
        self.scan_rollouts()
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            expected_keys = {"next_randomization_id"}
            if set(state) != expected_keys:
                raise ValueError("Dataset collection_state.json has an invalid shape")
            self.next_randomization_id = int(state["next_randomization_id"])
        else:
            self.write_state()

    def validate_episode_files(
        self,
        path: Path,
        split: str,
        episode_index: int,
    ) -> dict[str, Any]:
        """Validate compact trajectory and metadata files for one episode.

        Args:
            path: Episode directory.
            split: Expected split name.
            episode_index: Expected split-local index.

        Returns:
            Parsed metadata.

        Raises:
            ValueError: If metadata or trajectory arrays violate the contract.
        """

        metadata_path = path / "episode.json"
        trajectory_path = path / "trajectory.npz"
        if not metadata_path.is_file() or not trajectory_path.is_file():
            raise ValueError(f"Episode is missing metadata or trajectory: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("split") != split
            or metadata.get("episode_index") != episode_index
            or metadata.get("success") is not True
        ):
            raise ValueError(f"Invalid episode metadata: {metadata_path}")
        with np.load(trajectory_path) as trajectory:
            if set(trajectory.files) != set(TRAJECTORY_SHAPES):
                raise ValueError(f"Unexpected trajectory arrays in {path}")
            sample_count = int(trajectory["qpos"].shape[0])
            for name, shape in TRAJECTORY_SHAPES.items():
                if trajectory[name].shape != (sample_count, *shape):
                    raise ValueError(f"Invalid {name} shape in {path}")
                if trajectory[name].dtype != np.float32:
                    raise ValueError(f"Invalid {name} dtype in {path}")
                if not np.all(np.isfinite(trajectory[name])):
                    raise ValueError(f"Non-finite {name} values in {path}")
        if metadata.get("steps") != sample_count or sample_count <= 0:
            raise ValueError(f"Invalid metadata step count in {path}")
        return metadata

    def scan_completed(self) -> None:
        """Discover and validate fully exported RGB episodes."""

        expected_cameras = set(self.config["scene"]["names"]["cameras"])
        for split in SPLIT_NAMES:
            for path in (self.root / split).glob("episode_*"):
                if not path.is_dir():
                    continue
                try:
                    episode_index = int(path.name.removeprefix("episode_"))
                except ValueError as error:
                    raise ValueError(
                        f"Invalid episode directory name: {path}"
                    ) from error
                if episode_index < 0 or episode_index >= self.requested[split]:
                    raise ValueError(
                        f"Episode index is outside configured {split}: {path}"
                    )
                self.validate_episode_files(path, split, episode_index)
                videos = {video.stem for video in path.glob("*.mp4")}
                if videos != expected_cameras:
                    raise ValueError(f"Episode camera videos differ in {path}")
                if episode_index in self.completed[split]:
                    raise ValueError(
                        f"Duplicate {split} episode index: {episode_index}"
                    )
                self.completed[split].add(episode_index)

    def scan_rollouts(self) -> None:
        """Discover compact successful trajectories awaiting RGB replay."""

        for split in SPLIT_NAMES:
            for path in (self.rollouts_root / split).glob("episode_*"):
                if not path.is_dir():
                    continue
                try:
                    episode_index = int(path.name.removeprefix("episode_"))
                except ValueError as error:
                    raise ValueError(
                        f"Invalid rollout directory name: {path}"
                    ) from error
                if episode_index in self.completed[split]:
                    shutil.rmtree(path)
                    continue
                if episode_index < 0 or episode_index >= self.requested[split]:
                    raise ValueError(
                        f"Rollout index is outside configured {split}: {path}"
                    )
                self.validate_episode_files(path, split, episode_index)
                if any(path.glob("*.mp4")):
                    raise ValueError(
                        f"Compact rollout unexpectedly contains videos: {path}"
                    )
                self.rollout_ready[split].add(episode_index)

    def write_state(self) -> None:
        """Atomically persist resumable collection counters."""

        self.write_json_atomic(
            self.state_path,
            {"next_randomization_id": self.next_randomization_id},
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
        """Reserve and persist the next deterministic scene sample ID."""

        return self.reserve_randomization_ids(1)[0]

    def reserve_randomization_ids(self, count: int) -> list[int]:
        """Reserve and persist a contiguous batch of scene sample IDs."""

        if count <= 0:
            raise ValueError("Randomization ID reservation count must be positive")
        first = self.next_randomization_id
        self.next_randomization_id += count
        self.write_state()
        return list(range(first, self.next_randomization_id))

    def next_assignment(self) -> EpisodeAssignment | None:
        """Reserve the next episode lacking a compact successful rollout."""

        assignments = self.next_assignments(1)
        return assignments[0] if assignments else None

    def next_assignments(self, count: int) -> list[EpisodeAssignment]:
        """Reserve up to ``count`` missing episodes with one state write."""

        if count <= 0:
            raise ValueError("Episode assignment count must be positive")
        slots: list[tuple[str, int]] = []
        for split in SPLIT_NAMES:
            unavailable = (
                self.completed[split]
                | self.rollout_ready[split]
                | self.reserved[split]
            )
            for episode_index in range(self.requested[split]):
                if episode_index not in unavailable:
                    self.reserved[split].add(episode_index)
                    slots.append((split, episode_index))
                    if len(slots) == count:
                        break
            if len(slots) == count:
                break
        if not slots:
            return []
        randomization_ids = self.reserve_randomization_ids(len(slots))
        return [
            EpisodeAssignment(split, episode_index, randomization_id)
            for (split, episode_index), randomization_id in zip(
                slots,
                randomization_ids,
                strict=True,
            )
        ]

    def retry_assignment(self, assignment: EpisodeAssignment) -> EpisodeAssignment:
        """Resample a failed output slot with a new randomization ID."""

        return self.retry_assignments([assignment])[0]

    def retry_assignments(
        self, assignments: list[EpisodeAssignment]
    ) -> list[EpisodeAssignment]:
        """Resample failed output slots with one counter state write."""

        if not assignments:
            return []
        randomization_ids = self.reserve_randomization_ids(len(assignments))
        return [
            EpisodeAssignment(
                assignment.split,
                assignment.episode_index,
                randomization_id,
            )
            for assignment, randomization_id in zip(
                assignments,
                randomization_ids,
                strict=True,
            )
        ]

    def rollout_path(self, assignment: EpisodeAssignment) -> Path:
        """Return the compact staging directory for an assignment."""

        return (
            self.rollouts_root
            / assignment.split
            / f"episode_{assignment.episode_index:06d}"
        )

    def final_path(self, assignment: EpisodeAssignment) -> Path:
        """Return the final RGB episode directory for an assignment."""

        return self.root / assignment.split / f"episode_{assignment.episode_index:06d}"

    def mark_rollout_ready(self, assignment: EpisodeAssignment) -> None:
        """Register a successful compact rollout for later RGB replay."""

        self.rollout_ready[assignment.split].add(assignment.episode_index)

    def pending_replay_assignments(self) -> list[EpisodeAssignment]:
        """Return all staged but unexported assignments in split order."""

        assignments: list[EpisodeAssignment] = []
        for split in SPLIT_NAMES:
            for episode_index in sorted(self.rollout_ready[split]):
                path = self.rollouts_root / split / f"episode_{episode_index:06d}"
                metadata = json.loads(
                    (path / "episode.json").read_text(encoding="utf-8")
                )
                assignments.append(
                    EpisodeAssignment(
                        split,
                        episode_index,
                        metadata["randomization_id"],
                    )
                )
        return assignments

    def mark_completed(
        self,
        assignment: EpisodeAssignment,
        metadata: dict[str, Any],
    ) -> None:
        """Register one fully exported RGB episode and update its manifest."""

        self.completed[assignment.split].add(assignment.episode_index)
        self.rollout_ready[assignment.split].discard(assignment.episode_index)
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(metadata, sort_keys=True) + "\n")

    def counts(self) -> dict[str, int]:
        """Return fully exported episode counts by split."""

        return {split: len(self.completed[split]) for split in SPLIT_NAMES}

    def rollout_counts(self) -> dict[str, int]:
        """Return compact staged rollout counts by split."""

        return {split: len(self.rollout_ready[split]) for split in SPLIT_NAMES}

    def rollouts_complete(self) -> bool:
        """Return whether every requested slot is exported or ready to replay."""

        return all(
            len(self.completed[split] | self.rollout_ready[split])
            == self.requested[split]
            for split in SPLIT_NAMES
        )

    def is_complete(self) -> bool:
        """Return whether every requested dataset slot has RGB and labels."""

        return all(
            len(self.completed[split]) == self.requested[split]
            for split in SPLIT_NAMES
        )

    def validate(self) -> None:
        """Validate every final episode against the dataset contract."""

        if not self.is_complete():
            raise RuntimeError(
                f"Dataset is incomplete: completed={self.counts()}, "
                f"requested={self.requested}"
            )
        expected_cameras = set(self.config["scene"]["names"]["cameras"])
        for split in SPLIT_NAMES:
            for episode_index in range(self.requested[split]):
                path = self.root / split / f"episode_{episode_index:06d}"
                self.validate_episode_files(path, split, episode_index)
                videos = {video.stem for video in path.glob("*.mp4")}
                if videos != expected_cameras:
                    raise ValueError(f"Episode camera videos differ in {path}")


class RolloutWriter:
    """Buffer one compact controller rollout without images or subprocesses."""

    def __init__(
        self,
        dataset: SyntheticDataset,
        assignment: EpisodeAssignment,
        reset_state: dict[str, Any],
    ):
        """Preallocate one bounded trajectory buffer.

        Args:
            dataset: Dataset receiving the compact rollout.
            assignment: Reserved output slot and randomization identifier.
            reset_state: Per-environment deterministic reset metadata.
        """

        self.dataset = dataset
        self.assignment = assignment
        self.reset_state = reset_state
        self.steps = 0
        self.success_steps = 0
        maximum_steps = dataset.config["expert"]["maximum_episode_steps"]
        self.arrays = {
            name: np.empty((maximum_steps, *shape), dtype=np.float32)
            for name, shape in TRAJECTORY_SHAPES.items()
        }

    def append(self, samples: dict[str, np.ndarray], env_id: int) -> None:
        """Append one environment from synchronized batched host samples.

        Args:
            samples: Batched state/action arrays.
            env_id: Environment row to append.

        Raises:
            RuntimeError: If the configured trajectory capacity is exceeded.
        """

        if self.steps >= self.arrays["qpos"].shape[0]:
            raise RuntimeError("Rollout exceeds expert.maximum_episode_steps")
        for name, output in self.arrays.items():
            output[self.steps] = samples[name][env_id]
        self.steps += 1

    def metadata(self, reason: str) -> dict[str, Any]:
        """Build portable metadata for the current compact rollout."""

        if self.steps == 0:
            raise RuntimeError("Cannot describe an empty rollout")
        return {
            "split": self.assignment.split,
            "episode_index": self.assignment.episode_index,
            "randomization_id": self.assignment.randomization_id,
            "success": True,
            "termination": reason,
            "steps": self.steps,
            "instruction": self.dataset.settings["instruction"],
            "control_hz": self.dataset.config["expert"]["control_hz"],
            "pose_frame": "MJCF world",
            "reset": json_value(self.reset_state),
            "final_position_error_m": float(
                self.arrays["position_error_m"][self.steps - 1]
            ),
            "final_orientation_error_rad": float(
                self.arrays["orientation_error_rad"][self.steps - 1]
            ),
        }

    def write(self, path: Path, metadata: dict[str, Any]) -> None:
        """Atomically write compact arrays and metadata into a new directory."""

        incomplete = path.with_name(path.name + ".incomplete")
        if incomplete.exists():
            shutil.rmtree(incomplete)
        if path.exists():
            raise FileExistsError(f"Episode already exists: {path}")
        incomplete.mkdir(parents=True)
        np.savez_compressed(
            incomplete / "trajectory.npz",
            qpos=self.arrays["qpos"][: self.steps],
            qvel=self.arrays["qvel"][: self.steps],
            action_delta_q=self.arrays["action_delta_q"][: self.steps],
            sfp_tip_position=self.arrays["sfp_tip_position"][: self.steps],
            sfp_tip_rotation_matrix=self.arrays["sfp_tip_rotation_matrix"][
                : self.steps
            ],
            goal_position=self.arrays["goal_position"][: self.steps],
            goal_rotation_matrix=self.arrays["goal_rotation_matrix"][: self.steps],
            position_error_m=self.arrays["position_error_m"][: self.steps],
            orientation_error_rad=self.arrays["orientation_error_rad"][: self.steps],
            tared_wrench=self.arrays["tared_wrench"][: self.steps],
        )
        SyntheticDataset.write_json_atomic(incomplete / "episode.json", metadata)
        incomplete.rename(path)

    def finish(self, reason: str) -> None:
        """Stage one successful compact rollout for RGB replay."""

        metadata = self.metadata(reason)
        path = self.dataset.rollout_path(self.assignment)
        self.write(path, metadata)
        self.dataset.mark_rollout_ready(self.assignment)


class ReplayWriter:
    """Encode RGB for one staged rollout within a bounded replay batch."""

    def __init__(self, dataset: SyntheticDataset, assignment: EpisodeAssignment):
        """Create one final temporary directory and three lazy video writers.

        Args:
            dataset: Dataset containing the staged rollout.
            assignment: Staged episode being exported.
        """

        import imageio.v2 as imageio

        self.dataset = dataset
        self.assignment = assignment
        self.source_path = dataset.rollout_path(assignment)
        self.final_path = dataset.final_path(assignment)
        self.incomplete_path = self.final_path.with_name(
            self.final_path.name + ".incomplete"
        )
        if self.incomplete_path.exists():
            shutil.rmtree(self.incomplete_path)
        if self.final_path.exists():
            raise FileExistsError(f"Episode already exists: {self.final_path}")
        self.incomplete_path.mkdir()
        self.metadata = json.loads(
            (self.source_path / "episode.json").read_text(encoding="utf-8")
        )
        self.expected_steps = int(self.metadata["steps"])
        self.steps = 0
        settings = dataset.settings
        self.video_writers = {
            camera_name: imageio.get_writer(
                self.incomplete_path / f"{camera_name}.mp4",
                fps=dataset.config["expert"]["control_hz"],
                codec=settings["video_codec"],
                pixelformat=settings["video_pixel_format"],
                ffmpeg_params=[
                    "-crf",
                    str(settings["video_crf"]),
                    "-threads",
                    str(settings["video_threads"]),
                ],
                macro_block_size=None,
            )
            for camera_name in dataset.config["scene"]["names"]["cameras"]
        }

    def append(self, frames: dict[str, np.ndarray]) -> None:
        """Append one synchronized frame from every configured camera."""

        if self.steps >= self.expected_steps:
            raise RuntimeError("Replay produced more frames than the staged rollout")
        for camera_name, writer in self.video_writers.items():
            writer.append_data(frames[camera_name])
        self.steps += 1

    def close(self) -> None:
        """Close all active video encoders."""

        for writer in self.video_writers.values():
            writer.close()
        self.video_writers.clear()

    def finish(self) -> None:
        """Atomically publish videos and compact labels as a final episode."""

        self.close()
        if self.steps != self.expected_steps:
            raise RuntimeError(
                f"Replay frame count {self.steps} differs from expected "
                f"{self.expected_steps}"
            )
        shutil.copy2(self.source_path / "trajectory.npz", self.incomplete_path)
        shutil.copy2(self.source_path / "episode.json", self.incomplete_path)
        self.incomplete_path.rename(self.final_path)
        self.dataset.mark_completed(self.assignment, self.metadata)
        shutil.rmtree(self.source_path)


def json_value(value: Any) -> Any:
    """Convert nested NumPy values into JSON-serializable values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value
