"""Two-stage privileged rollout collection and bounded RGB replay."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

import mujoco
import numpy as np

from aic_mujoco.config import load_collection_config
from aic_mujoco.controllers import CartesianMoveController
from aic_mujoco.dataset import (
    DatasetImageBuffer,
    EpisodeAssignment,
    ReplayWriter,
    RolloutWriter,
    SyntheticDataset,
)
from aic_mujoco.joints import required_model_id
from aic_mujoco.outputs import RuntimeOutputs
from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.scene import prepare_scene
from aic_mujoco.utils.mujoco_math import (
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
)
from aic_mujoco.utils.timing import wait_for_realtime


def download_reset_state(runtime: AICWarpRuntime) -> dict[str, np.ndarray]:
    """Download compact reset metadata and expose orientations as matrices.

    Args:
        runtime: Initialized MJWarp rollout runtime.

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
    """Select independent reset metadata for one environment."""

    return {name: values[env_id].copy() for name, values in reset_state.items()}


def start_pending_rollouts(
    runtime: AICWarpRuntime,
    dataset: SyntheticDataset,
    pending: dict[int, EpisodeAssignment],
    active: dict[int, RolloutWriter],
) -> None:
    """Start pending rollouts whose independent F/T tare is ready."""

    tare_ready = runtime.tare_ready.numpy()
    ready_ids = [env_id for env_id in pending if tare_ready[env_id]]
    if not ready_ids:
        return
    reset_state = download_reset_state(runtime)
    for env_id in ready_ids:
        assignment = pending.pop(env_id)
        active[env_id] = RolloutWriter(
            dataset,
            assignment,
            reset_state_for_environment(reset_state, env_id),
        )


def assign_rollouts(
    runtime: AICWarpRuntime,
    assignments: list[tuple[int, EpisodeAssignment | None]],
    pending: dict[int, EpisodeAssignment],
) -> None:
    """Reset all newly assigned worlds in one batched device operation."""

    selected = [item for item in assignments if item[1] is not None]
    if not selected:
        return
    env_ids: list[int] = []
    randomization_ids: list[int] = []
    for env_id, possible_assignment in selected:
        if possible_assignment is None:
            continue
        pending[env_id] = possible_assignment
        env_ids.append(env_id)
        randomization_ids.append(possible_assignment.randomization_id)
    runtime.reset(env_ids, randomization_ids)


def current_samples(
    runtime: AICWarpRuntime,
    teacher: CartesianMoveController,
) -> dict[str, np.ndarray]:
    """Download synchronized state and action labels for every rollout world."""

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


def append_rollout_samples(
    active: dict[int, RolloutWriter],
    samples: dict[str, np.ndarray],
) -> None:
    """Append one synchronized label sample to every active rollout."""

    for env_id, writer in active.items():
        writer.append(samples, env_id)


def completed_rollout_environments(
    config: dict[str, Any],
    active: dict[int, RolloutWriter],
    samples: dict[str, np.ndarray],
) -> list[tuple[int, bool, str]]:
    """Evaluate pose success and timeout termination for active rollouts."""

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


def create_teacher(
    config: dict[str, Any], runtime: AICWarpRuntime
) -> CartesianMoveController:
    """Construct the privileged Cartesian teacher for one runtime."""

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
    return CartesianMoveController(
        config,
        runtime.robot.joints,
        runtime.host_model,
        controlled_body_id,
        target_body_id,
        runtime.num_envs,
        runtime.device,
    )


def run_rollout_collection(
    config: dict[str, Any], dataset: SyntheticDataset
) -> bool:
    """Collect compact trajectories across all configured MJWarp worlds.

    Returns:
        Whether all requested successful rollouts are staged.
    """

    runtime = AICWarpRuntime(config, render_cameras=False)
    teacher = create_teacher(config, runtime)
    outputs = RuntimeOutputs(config, runtime)
    pending: dict[int, EpisodeAssignment] = {}
    active: dict[int, RolloutWriter] = {}

    initial_assignments = dataset.next_assignments(runtime.num_envs)
    assign_rollouts(
        runtime,
        list(enumerate(initial_assignments)),
        pending,
    )

    print(f"Scene: {config['scene']['output']}")
    print(f"Rollout worlds: {runtime.num_envs} on {runtime.device}; RGB disabled")
    print(f"Dataset: {dataset.root}")
    print(
        f"Requested: {dataset.requested}; staged: {dataset.rollout_counts()}; "
        f"exported: {dataset.counts()}"
    )

    start_time = time.perf_counter()
    step_index = 0
    realtime = config["visualization"]["enabled"] and config["visualization"][
        "realtime"
    ]
    timestep = config["physics"]["timestep"]
    try:
        while not dataset.rollouts_complete():
            events = runtime.step()
            step_index += 1
            if realtime:
                wait_for_realtime(start_time, step_index, timestep)
            if not events.camera:
                continue

            outputs.update(runtime)
            start_pending_rollouts(runtime, dataset, pending, active)
            teacher.set_active(
                [env_id in active for env_id in range(runtime.num_envs)]
            )
            if not active:
                continue

            teacher.update_goal_from_target_body(runtime.data)
            teacher.move(runtime.data, runtime.hold_command)
            samples = current_samples(runtime, teacher)
            append_rollout_samples(active, samples)

            finished = completed_rollout_environments(config, active, samples)
            replacements: list[tuple[int, EpisodeAssignment | None]] = []
            successful_envs: list[int] = []
            failed_envs: list[int] = []
            failed_assignments: list[EpisodeAssignment] = []
            for env_id, success, reason in finished:
                writer = active.pop(env_id)
                if success:
                    writer.finish(reason)
                    successful_envs.append(env_id)
                else:
                    failed_envs.append(env_id)
                    failed_assignments.append(writer.assignment)
            if finished:
                print(
                    f"Staged {dataset.rollout_counts()}; "
                    f"discarded and resampling {len(failed_envs)} attempts"
                )
            next_assignments = dataset.next_assignments(len(successful_envs)) if (
                successful_envs
            ) else []
            replacements.extend(zip(successful_envs, next_assignments, strict=False))
            retry_assignments = dataset.retry_assignments(failed_assignments)
            replacements.extend(zip(failed_envs, retry_assignments, strict=True))
            assign_rollouts(runtime, replacements, pending)
    except KeyboardInterrupt:
        print("\nRollout collection interrupted; staged rollouts are safe.")
        return False
    finally:
        outputs.close()
    return True


def replay_qpos(dataset: SyntheticDataset, assignment: EpisodeAssignment) -> np.ndarray:
    """Load the compact joint trajectory required for visual replay."""

    with np.load(dataset.rollout_path(assignment) / "trajectory.npz") as trajectory:
        return np.asarray(trajectory["qpos"], dtype=np.float32).copy()


def restore_replay_scenes(
    runtime: AICWarpRuntime,
    dataset: SyntheticDataset,
    assignments: list[EpisodeAssignment],
) -> None:
    """Restore exact saved board and NIC poses for a replay batch."""

    board_positions = runtime.board_position.numpy()
    board_quaternions = runtime.board_quaternion.numpy()
    nic_positions = runtime.nic_position.numpy()
    nic_quaternions = runtime.nic_quaternion.numpy()
    for env_id, assignment in enumerate(assignments):
        metadata = json.loads(
            (dataset.rollout_path(assignment) / "episode.json").read_text(
                encoding="utf-8"
            )
        )
        reset = metadata["reset"]
        board_positions[env_id] = reset["board_position"]
        board_quaternions[env_id] = quaternion_from_rotation_matrix(
            reset["board_rotation_matrix"]
        )
        nic_positions[env_id] = reset["nic_position"]
        nic_quaternions[env_id] = quaternion_from_rotation_matrix(
            reset["nic_rotation_matrix"]
        )
    runtime.set_scene_poses(
        board_positions,
        board_quaternions,
        nic_positions,
        nic_quaternions,
    )


def export_replay_batch(
    config: dict[str, Any],
    dataset: SyntheticDataset,
    runtime: AICWarpRuntime,
    image_buffer: DatasetImageBuffer,
    assignments: list[EpisodeAssignment],
) -> None:
    """Replay and export one bounded batch of staged trajectories."""

    env_ids = list(range(len(assignments)))
    runtime.reset(
        env_ids,
        [assignment.randomization_id for assignment in assignments],
    )
    restore_replay_scenes(runtime, dataset, assignments)
    trajectories = [replay_qpos(dataset, assignment) for assignment in assignments]
    writers = [ReplayWriter(dataset, assignment) for assignment in assignments]
    positions = np.tile(
        np.asarray(config["control"]["home"], dtype=np.float32),
        (runtime.num_envs, 1),
    )
    try:
        maximum_steps = max(trajectory.shape[0] for trajectory in trajectories)
        for step in range(maximum_steps):
            for env_id, trajectory in enumerate(trajectories):
                if step < trajectory.shape[0]:
                    positions[env_id] = trajectory[step]
            runtime.set_joint_positions(positions)
            runtime.render()
            images = image_buffer.download(runtime.rgb)
            for env_id, writer in enumerate(writers):
                if step >= trajectories[env_id].shape[0]:
                    continue
                frames = {
                    camera_name: camera_batch[env_id]
                    for camera_name, camera_batch in images.items()
                }
                writer.append(frames)
        for writer in writers:
            writer.finish()
    finally:
        for writer in writers:
            writer.close()


def run_rgb_replay(config: dict[str, Any], dataset: SyntheticDataset) -> bool:
    """Render all staged trajectories with bounded GPU and encoder concurrency.

    Returns:
        Whether every staged trajectory was fully exported.
    """

    assignments = dataset.pending_replay_assignments()
    if not assignments:
        return dataset.is_complete()
    replay_batch_size = config["dataset"]["replay_batch_size"]
    replay_config = copy.deepcopy(config)
    replay_config["runtime"]["num_envs"] = replay_batch_size
    replay_config["visualization"]["enabled"] = False
    runtime = AICWarpRuntime(replay_config, render_cameras=True)
    image_buffer = DatasetImageBuffer(replay_config, runtime)
    camera_count = len(config["scene"]["names"]["cameras"])
    print(
        f"RGB replay: {replay_batch_size} worlds, at most "
        f"{replay_batch_size * camera_count} video encoders"
    )

    try:
        for start in range(0, len(assignments), replay_batch_size):
            batch = assignments[start : start + replay_batch_size]
            export_replay_batch(
                replay_config,
                dataset,
                runtime,
                image_buffer,
                batch,
            )
            print(f"Exported {dataset.counts()}; staged {dataset.rollout_counts()}")
    except KeyboardInterrupt:
        print("\nRGB replay interrupted; staged rollouts are safe.")
        return False
    return dataset.is_complete()


def main() -> None:
    """Run large-scale rollouts followed by bounded RGB replay."""

    config = load_collection_config()
    prepare_scene(config)
    dataset = SyntheticDataset(config)
    if dataset.is_complete():
        dataset.validate()
        print(f"Dataset is already complete and valid: {dataset.root}")
        return
    if not dataset.rollouts_complete() and not run_rollout_collection(config, dataset):
        return
    if not run_rgb_replay(config, dataset):
        return
    dataset.validate()
    print(f"Dataset collection complete and validated: {dataset.root}")
