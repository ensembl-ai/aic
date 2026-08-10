"""Orchestration for privileged-teacher demonstration collection."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np

from aic_mujoco.config import load_collection_config
from aic_mujoco.controllers import CartesianMoveController
from aic_mujoco.dataset import (
    DatasetImageBuffer,
    EpisodeAssignment,
    EpisodeWriter,
    SyntheticDataset,
)
from aic_mujoco.joints import required_model_id
from aic_mujoco.outputs import RuntimeOutputs
from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.scene import prepare_scene
from aic_mujoco.utils.mujoco_math import rotation_matrix_from_quaternion
from aic_mujoco.utils.timing import wait_for_realtime


def download_reset_state(runtime: AICWarpRuntime) -> dict[str, np.ndarray]:
    """Download compact reset metadata and expose orientations as matrices.

    Args:
        runtime: Initialized MJWarp runtime.

    Returns:
        Host arrays describing every environment's current reset sample.
    """

    state = {
        name: tensor.numpy()
        for name, tensor in runtime.reset_state().items()
        if name != "episode_steps"
    }
    for body_name in ("board", "nic"):
        quaternion_key = f"{body_name}_quaternion"
        quaternions = state.pop(quaternion_key)
        state[f"{body_name}_rotation_matrix"] = np.stack(
            [rotation_matrix_from_quaternion(value) for value in quaternions]
        )
    return state


def reset_state_for_environment(
    reset_state: dict[str, np.ndarray], env_id: int
) -> dict[str, Any]:
    """Select one environment from downloaded reset metadata.

    Args:
        reset_state: Batched host reset metadata.
        env_id: Environment index to select.

    Returns:
        Independent copies of the selected environment values.
    """

    return {name: values[env_id].copy() for name, values in reset_state.items()}


def start_pending_episodes(
    runtime: AICWarpRuntime,
    dataset: SyntheticDataset,
    pending: dict[int, EpisodeAssignment],
    active: dict[int, EpisodeWriter],
) -> None:
    """Start pending episodes whose per-environment F/T tare is ready.

    Args:
        runtime: Active MJWarp runtime.
        dataset: Destination synthetic dataset.
        pending: Assignments waiting for taring, keyed by environment ID.
        active: Episode writers currently collecting labeled samples.
    """

    tare_ready = runtime.tare_ready.numpy()
    ready_ids = [env_id for env_id in pending if tare_ready[env_id]]
    if not ready_ids:
        return
    reset_state = download_reset_state(runtime)
    for env_id in ready_ids:
        assignment = pending.pop(env_id)
        active[env_id] = EpisodeWriter(
            dataset,
            assignment,
            reset_state_for_environment(reset_state, env_id),
        )


def assign_episode(
    runtime: AICWarpRuntime,
    env_id: int,
    assignment: EpisodeAssignment | None,
    pending: dict[int, EpisodeAssignment],
) -> None:
    """Reset one environment for a deterministic dataset assignment.

    Args:
        runtime: Active MJWarp runtime.
        env_id: Environment receiving the assignment.
        assignment: Reserved trajectory, or ``None`` when work is exhausted.
        pending: Assignments waiting for F/T taring.
    """

    if assignment is None:
        return
    pending[env_id] = assignment
    runtime.reset([env_id], [assignment.randomization_id])


def current_samples(
    runtime: AICWarpRuntime,
    teacher: CartesianMoveController,
) -> dict[str, np.ndarray]:
    """Download the current synchronized state/action labels.

    Args:
        runtime: Active MJWarp runtime.
        teacher: Privileged Cartesian teacher after its current action update.

    Returns:
        Batched host arrays for every trajectory field.
    """

    return {
        "qpos": runtime.data.qpos.numpy(),
        "qvel": runtime.data.qvel.numpy(),
        "action_delta_q": teacher.action.position.numpy(),
        "sfp_tip_position": teacher.current_position.numpy(),
        "sfp_tip_rotation_matrix": teacher.current_rotation.numpy(),
        "goal_position": teacher.command.position.numpy(),
        "goal_rotation_matrix": teacher.command.rotation.numpy(),
        "position_error_m": teacher.position_error.numpy(),
        "orientation_error_rad": teacher.orientation_error.numpy(),
        "tared_wrench": runtime.tared_wrench.numpy(),
    }


def append_active_samples(
    active: dict[int, EpisodeWriter],
    images: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
) -> None:
    """Append one synchronized sample to every active environment writer.

    Args:
        active: Episode writers keyed by environment ID.
        images: Batched named RGB images.
        samples: Batched state/action labels.
    """

    for env_id, writer in active.items():
        frames = {
            camera_name: camera_batch[env_id]
            for camera_name, camera_batch in images.items()
        }
        sample = {name: values[env_id] for name, values in samples.items()}
        writer.append(frames, sample)


def completed_environments(
    config: dict[str, Any],
    active: dict[int, EpisodeWriter],
    samples: dict[str, np.ndarray],
) -> list[tuple[int, bool, str]]:
    """Evaluate success and timeout termination for active episodes.

    Args:
        config: Strict merged collection configuration.
        active: Episode writers keyed by environment ID.
        samples: Most recently downloaded state/action labels.

    Returns:
        ``(env_id, success, reason)`` tuples for finished environments.
    """

    expert = config["expert"]
    completed: list[tuple[int, bool, str]] = []
    for env_id, writer in active.items():
        position_error = float(samples["position_error_m"][env_id])
        orientation_error = float(samples["orientation_error_rad"][env_id])
        within_tolerance = (
            position_error <= expert["position_tolerance"]
            and orientation_error <= expert["orientation_tolerance"]
        )
        if within_tolerance:
            writer.success_steps += 1
        else:
            writer.success_steps = 0
        if writer.success_steps >= expert["success_consecutive_steps"]:
            completed.append((env_id, True, "pose_tolerance"))
        elif writer.steps >= expert["maximum_episode_steps"]:
            completed.append((env_id, False, "step_limit"))
    return completed


def run_collection(config: dict[str, Any], dataset: SyntheticDataset) -> None:
    """Collect all missing trajectories with independently resetting worlds.

    Args:
        config: Strict merged collection configuration.
        dataset: Prepared resumable dataset destination.
    """

    runtime = AICWarpRuntime(config)
    controlled_body_id = required_model_id(
        runtime.host_model,
        mujoco.mjtObj.mjOBJ_BODY,
        config["expert"]["controlled_body"],
    )
    target_body_id = required_model_id(
        runtime.host_model,
        mujoco.mjtObj.mjOBJ_BODY,
        config["expert"]["target_body"],
    )
    teacher = CartesianMoveController(
        config,
        runtime.robot.joints,
        runtime.host_model,
        controlled_body_id,
        target_body_id,
        runtime.num_envs,
        runtime.device,
    )
    image_buffer = DatasetImageBuffer(config, runtime)
    outputs = RuntimeOutputs(config, runtime)
    pending: dict[int, EpisodeAssignment] = {}
    active: dict[int, EpisodeWriter] = {}

    for env_id in range(runtime.num_envs):
        assign_episode(runtime, env_id, dataset.next_assignment(), pending)

    print(f"Scene: {config['scene']['output']}")
    print(f"Worlds: {runtime.num_envs} on {runtime.device}")
    print(f"Dataset: {dataset.root}")
    print(f"Requested: {dataset.requested}; already complete: {dataset.counts()}")

    start_time = time.perf_counter()
    step_index = 0
    realtime = config["visualization"]["enabled"] and config["visualization"][
        "realtime"
    ]
    timestep = config["physics"]["timestep"]
    try:
        while not dataset.is_complete():
            events = runtime.step()
            step_index += 1
            if realtime:
                wait_for_realtime(start_time, step_index, timestep)
            if not events.camera:
                continue

            outputs.update(runtime)
            start_pending_episodes(runtime, dataset, pending, active)
            active_mask = [env_id in active for env_id in range(runtime.num_envs)]
            teacher.set_active(active_mask)
            if not active:
                continue

            teacher.update_goal_from_target_body(runtime.data)
            teacher.move(runtime.data, runtime.hold_command)
            images = image_buffer.download(runtime.rgb)
            samples = current_samples(runtime, teacher)
            append_active_samples(active, images, samples)

            finished = completed_environments(config, active, samples)
            for env_id, success, reason in finished:
                writer = active.pop(env_id)
                writer.finish(success, reason)
                if success:
                    next_assignment = dataset.next_assignment()
                else:
                    next_assignment = dataset.retry_assignment(writer.assignment)
                assign_episode(runtime, env_id, next_assignment, pending)
                status = "saved" if success else "failed and resampled"
                print(
                    f"{writer.assignment.split} "
                    f"episode {writer.assignment.episode_index}: {status}; "
                    f"completed {dataset.counts()}"
                )
    except KeyboardInterrupt:
        print("\nCollection interrupted; completed episodes are safe and resumable.")
    finally:
        for writer in active.values():
            writer.close_incomplete()
        outputs.close()


def main() -> None:
    """Run the no-CLI synthetic demonstration collector."""

    config = load_collection_config()
    prepare_scene(config)
    dataset = SyntheticDataset(config)
    if dataset.is_complete():
        dataset.validate()
        print(f"Dataset is already complete and valid: {dataset.root}")
        return
    run_collection(config, dataset)
    if dataset.is_complete():
        dataset.validate()
        print(f"Dataset collection complete and validated: {dataset.root}")
