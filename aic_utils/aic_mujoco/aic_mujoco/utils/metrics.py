"""Local scalar-training and workspace-rollout evidence artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRICS_FORMAT_VERSION = 1


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp with microsecond uniqueness."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a formatted JSON object.

    Args:
        path: Destination JSON path.
        value: JSON-serializable object.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class JsonlWriter:
    """Append durable JSON records to one line-oriented metrics file."""

    def __init__(self, path: Path):
        """Create a new line-buffered JSONL file.

        Args:
            path: New output path. Existing files are rejected.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("x", encoding="utf-8", buffering=1)

    def append(self, value: dict[str, Any]) -> None:
        """Write and flush one JSON object."""

        self.file.write(json.dumps(value, sort_keys=True) + "\n")
        self.file.flush()

    def close(self) -> None:
        """Close the metrics file."""

        self.file.close()


class TrainingMetricsRecorder:
    """Keep a complete local scalar copy of metrics sent to W&B."""

    def __init__(self, output_directory: Path):
        """Create the Git-trackable training evidence directory.

        Args:
            output_directory: Training run root.
        """

        self.directory = output_directory / "metrics" / "training"
        self.writer = JsonlWriter(self.directory / "history.jsonl")
        self.latest: dict[str, Any] = {}

    def record(
        self, step: int, phase: str, metrics: dict[str, float | int]
    ) -> None:
        """Persist one scalar metric event.

        Args:
            step: Completed optimizer step.
            phase: Event category such as ``train`` or ``validation``.
            metrics: Exact scalar values also sent to W&B.
        """

        record: dict[str, Any] = {"step": step, "phase": phase}
        record.update(metrics)
        self.writer.append(record)
        self.latest.update(metrics)

    def finish(self, summary: dict[str, Any]) -> None:
        """Write final run summary and close the history file.

        Args:
            summary: Dataset, model, checkpoint, and completion metadata.
        """

        complete_summary = {
            "format_version": METRICS_FORMAT_VERSION,
            **summary,
            "latest_metrics": self.latest,
        }
        write_json_atomic(self.directory / "summary.json", complete_summary)
        self.writer.close()


@dataclass(frozen=True)
class WorkspaceSample:
    """One policy-rate SFP and Cartesian-goal observation."""

    sfp_position: np.ndarray
    goal_position: np.ndarray
    position_error: float
    orientation_error: float


class RolloutMetricsRecorder:
    """Persist incremental SFP paths and comparable episode summaries."""

    def __init__(self, config: dict[str, Any]):
        """Create one timestamped evidence session for the explicit checkpoint.

        Args:
            config: Strict merged evaluation configuration.
        """

        checkpoint = Path(config["evaluation"]["checkpoint_directory"])
        run_directory = checkpoint.parent.parent
        self.session_id = utc_timestamp()
        self.directory = (
            run_directory
            / "metrics"
            / "rollouts"
            / checkpoint.name
            / self.session_id
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.samples = JsonlWriter(self.directory / "workspace_path.jsonl")
        self.episodes = JsonlWriter(self.directory / "episodes.jsonl")
        self.control_hz = config["expert"]["control_hz"]
        self.completed: list[dict[str, Any]] = []
        self.active_episode: int | None = None
        self.active_steps = 0
        self.initial_sample: WorkspaceSample | None = None
        self.previous_position: np.ndarray | None = None
        self.path_length = 0.0
        self.minimum_position_error = math.inf
        self.minimum_orientation_error = math.inf
        self.closest_goal_step = 0
        metadata = {
            "format_version": METRICS_FORMAT_VERSION,
            "session_id": self.session_id,
            "checkpoint": checkpoint.name,
            "policy_type": config["policy"]["type"],
            "control_hz": self.control_hz,
            "maximum_episode_steps": config["evaluation"][
                "maximum_episode_steps"
            ],
            "rollout_episodes": config["evaluation"]["rollout_episodes"],
            "position_tolerance_m": config["expert"]["position_tolerance"],
            "orientation_tolerance_rad": config["expert"][
                "orientation_tolerance"
            ],
            "success_consecutive_steps": config["expert"][
                "success_consecutive_steps"
            ],
            "runtime_seed": config["runtime"]["seed"],
            "domain_randomization": config["domain_randomization"],
        }
        write_json_atomic(self.directory / "session.json", metadata)

    @property
    def has_active_samples(self) -> bool:
        """Return whether the current episode contains workspace samples."""

        return self.active_episode is not None and self.active_steps > 0

    def start_episode(self, episode_index: int) -> None:
        """Reset running path statistics for one newly randomized episode.

        Args:
            episode_index: One-based session-local episode number.
        """

        if self.active_episode is not None:
            raise RuntimeError("Cannot start an episode while another is active")
        self.active_episode = episode_index
        self.active_steps = 0
        self.initial_sample = None
        self.previous_position = None
        self.path_length = 0.0
        self.minimum_position_error = math.inf
        self.minimum_orientation_error = math.inf
        self.closest_goal_step = 0

    def record(self, policy_step: int, sample: WorkspaceSample) -> None:
        """Append one SFP point and update online path metrics.

        Args:
            policy_step: One-based policy step within the active episode.
            sample: Current SFP/goal positions and pose errors.
        """

        if self.active_episode is None:
            raise RuntimeError("Cannot record workspace data without an episode")
        position = np.asarray(sample.sfp_position, dtype=np.float64)
        goal = np.asarray(sample.goal_position, dtype=np.float64)
        if position.shape != (3,) or goal.shape != (3,):
            raise ValueError("Workspace positions must contain three values")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(goal)):
            raise ValueError("Workspace positions must be finite")
        scalar_values = (sample.position_error, sample.orientation_error)
        if any(not math.isfinite(value) or value < 0.0 for value in scalar_values):
            raise ValueError("Workspace pose errors must be finite and nonnegative")
        if policy_step != self.active_steps + 1:
            raise ValueError("Policy steps must be recorded consecutively")

        step_distance = 0.0
        if self.previous_position is not None:
            step_distance = float(np.linalg.norm(position - self.previous_position))
            self.path_length += step_distance
        if self.initial_sample is None:
            self.initial_sample = sample
        if sample.position_error < self.minimum_position_error:
            self.minimum_position_error = sample.position_error
            self.closest_goal_step = policy_step
        self.minimum_orientation_error = min(
            self.minimum_orientation_error, sample.orientation_error
        )
        initial_error = self.initial_sample.position_error
        self.active_steps = policy_step
        self.previous_position = position.copy()
        self.samples.append(
            {
                "episode": self.active_episode,
                "policy_step": policy_step,
                "simulation_time_s": policy_step / self.control_hz,
                "sfp_position_m": position.tolist(),
                "goal_position_m": goal.tolist(),
                "position_error_m": sample.position_error,
                "orientation_error_rad": sample.orientation_error,
                "step_distance_m": step_distance,
                "workspace_path_length_m": self.path_length,
                "goal_progress_m": initial_error - sample.position_error,
            }
        )

    def finish_episode(
        self, termination: str, final_sample: WorkspaceSample
    ) -> dict[str, Any]:
        """Finalize and persist one success, timeout, or interrupted episode.

        Args:
            termination: Explicit terminal reason.
            final_sample: Most recently recorded workspace observation.

        Returns:
            Serializable episode summary.
        """

        if not self.has_active_samples or self.initial_sample is None:
            raise RuntimeError("Cannot finish an episode without workspace samples")
        if self.previous_position is None:
            raise RuntimeError("Active episode has no previous SFP position")
        initial_position = np.asarray(
            self.initial_sample.sfp_position, dtype=np.float64
        )
        displacement = float(np.linalg.norm(self.previous_position - initial_position))
        straightness = (
            displacement / self.path_length if self.path_length > 0.0 else 0.0
        )
        episode = {
            "episode": self.active_episode,
            "termination": termination,
            "success": termination == "success",
            "policy_steps": self.active_steps,
            "simulation_duration_s": self.active_steps / self.control_hz,
            "initial_position_error_m": self.initial_sample.position_error,
            "final_position_error_m": final_sample.position_error,
            "minimum_position_error_m": self.minimum_position_error,
            "position_progress_m": (
                self.initial_sample.position_error - final_sample.position_error
            ),
            "final_error_increase_from_closest_m": (
                final_sample.position_error - self.minimum_position_error
            ),
            "initial_orientation_error_rad": self.initial_sample.orientation_error,
            "final_orientation_error_rad": final_sample.orientation_error,
            "minimum_orientation_error_rad": self.minimum_orientation_error,
            "workspace_path_length_m": self.path_length,
            "workspace_displacement_m": displacement,
            "path_straightness": straightness,
            "closest_goal_step": self.closest_goal_step,
        }
        self.episodes.append(episode)
        self.completed.append(episode)
        self.active_episode = None
        self.write_summary("running")
        return episode

    def aggregate(self, status: str) -> dict[str, Any]:
        """Build session-level metrics from every finalized episode.

        Args:
            status: Current session lifecycle state.

        Returns:
            Aggregate scalar metrics and counts.
        """

        evaluated = [
            episode
            for episode in self.completed
            if episode["termination"] in ("success", "timeout")
        ]
        successes = sum(episode["success"] for episode in evaluated)
        timeouts = sum(episode["termination"] == "timeout" for episode in evaluated)
        interrupted = sum(
            episode["termination"] == "interrupted" for episode in self.completed
        )
        summary: dict[str, Any] = {
            "format_version": METRICS_FORMAT_VERSION,
            "session_id": self.session_id,
            "status": status,
            "episode_count": len(self.completed),
            "evaluated_episode_count": len(evaluated),
            "success_count": successes,
            "timeout_count": timeouts,
            "interrupted_count": interrupted,
            "success_rate": successes / len(evaluated) if evaluated else 0.0,
        }
        if evaluated:
            positive_progress = sum(
                float(episode["position_progress_m"]) > 0.0
                for episode in evaluated
            )
            final_at_closest = sum(
                int(episode["closest_goal_step"]) == int(episode["policy_steps"])
                for episode in evaluated
            )
            summary["positive_position_progress_count"] = positive_progress
            summary["positive_position_progress_rate"] = (
                positive_progress / len(evaluated)
            )
            summary["final_at_closest_count"] = final_at_closest
            summary["final_at_closest_rate"] = final_at_closest / len(evaluated)
            metric_names = (
                "initial_position_error_m",
                "final_position_error_m",
                "minimum_position_error_m",
                "position_progress_m",
                "final_error_increase_from_closest_m",
                "initial_orientation_error_rad",
                "final_orientation_error_rad",
                "minimum_orientation_error_rad",
                "workspace_path_length_m",
                "path_straightness",
                "simulation_duration_s",
            )
            for name in metric_names:
                values = [float(episode[name]) for episode in evaluated]
                summary[f"mean_{name}"] = float(np.mean(values))
                summary[f"median_{name}"] = float(np.median(values))
            summary["best_minimum_position_error_m"] = min(
                float(episode["minimum_position_error_m"])
                for episode in evaluated
            )
        return summary

    def write_summary(self, status: str) -> None:
        """Atomically publish the current aggregate session summary."""

        write_json_atomic(self.directory / "summary.json", self.aggregate(status))

    def close(self, status: str) -> None:
        """Finalize the session summary and close incremental files.

        Args:
            status: Final session lifecycle state.
        """

        self.write_summary(status)
        self.samples.close()
        self.episodes.close()
