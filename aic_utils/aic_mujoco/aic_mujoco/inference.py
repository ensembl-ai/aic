"""Closed-loop ACT inference in one continuously reset MJWarp environment."""

from __future__ import annotations

import time
from typing import Any

import mujoco
import torch
import warp as wp
from aic_mujoco.commands import JointDeltaAction
from aic_mujoco.config import load_evaluation_config
from aic_mujoco.dataset import DatasetImageBuffer
from aic_mujoco.joints import required_model_id
from aic_mujoco.outputs import RuntimeOutputs
from aic_mujoco.policy import (
    ACTPolicy,
    PolicyNormalizer,
    camera_feature_name,
    load_policy_checkpoint,
)
from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.scene import prepare_scene
from aic_mujoco.utils.metrics import RolloutMetricsRecorder, WorkspaceSample
from aic_mujoco.utils.timing import wait_for_realtime
from aic_mujoco.utils.warp_math import rotation_error_world


@wp.kernel
def apply_joint_delta_action(
    qpos: wp.array2d[float],
    predicted_delta: wp.array2d[float],
    qpos_addresses: wp.array[int],
    joint_lower: wp.array[float],
    joint_upper: wp.array[float],
    maximum_joint_step: wp.array[float],
    joint_limit_margin: float,
    action_delta: wp.array2d[float],
    hold_position: wp.array2d[float],
):
    """Bound a policy action and update the existing joint HOLD command.

    Args:
        qpos: Batched generalized positions.
        predicted_delta: Policy-predicted joint increments in radians.
        qpos_addresses: Ordered arm generalized-position addresses.
        joint_lower: Arm joint lower limits.
        joint_upper: Arm joint upper limits.
        maximum_joint_step: Per-joint increment limits.
        joint_limit_margin: Clearance maintained from compiled joint limits.
        action_delta: Bounded joint action command.
        hold_position: Mutable target consumed by the impedance controller.
    """

    world, joint = wp.tid()
    q_index = qpos_addresses[joint]
    bounded_delta = wp.clamp(
        predicted_delta[world, joint],
        -maximum_joint_step[joint],
        maximum_joint_step[joint],
    )
    target = wp.clamp(
        qpos[world, q_index] + bounded_delta,
        joint_lower[joint] + joint_limit_margin,
        joint_upper[joint] - joint_limit_margin,
    )
    action_delta[world, joint] = target - qpos[world, q_index]
    hold_position[world, joint] = target


@wp.kernel
def measure_cartesian_goal_error(
    body_position: wp.array2d[wp.vec3],
    body_rotation: wp.array2d[wp.mat33],
    controlled_body_id: int,
    target_body_id: int,
    offset_position: wp.vec3,
    offset_rotation: wp.mat33,
    current_position: wp.array[wp.vec3],
    goal_position: wp.array[wp.vec3],
    position_error: wp.array[float],
    orientation_error: wp.array[float],
):
    """Measure privileged Cartesian goal error without controlling the robot.

    Args:
        body_position: Batched MJWarp body positions.
        body_rotation: Batched MJWarp body rotation matrices.
        controlled_body_id: Body moved by the learned policy.
        target_body_id: Body defining the local goal frame.
        offset_position: Goal translation in the target body's local frame.
        offset_rotation: Goal rotation in the target body's local frame.
        current_position: Output SFP positions in the MJCF world frame.
        goal_position: Output Cartesian goal positions in the MJCF world frame.
        position_error: Output translation-error magnitudes in meters.
        orientation_error: Output principal rotation-error magnitudes in radians.
    """

    world = wp.tid()
    target_parent_rotation = body_rotation[world, target_body_id]
    target_position = (
        body_position[world, target_body_id]
        + target_parent_rotation * offset_position
    )
    target_rotation = target_parent_rotation * offset_rotation
    linear_error = target_position - body_position[world, controlled_body_id]
    angular_error = rotation_error_world(
        body_rotation[world, controlled_body_id], target_rotation
    )
    current_position[world] = body_position[world, controlled_body_id]
    goal_position[world] = target_position
    position_error[world] = wp.length(linear_error)
    orientation_error[world] = wp.length(angular_error)


class PolicyController:
    """Convert MJWarp camera/state observations into bounded HOLD targets."""

    def __init__(
        self,
        config: dict[str, Any],
        runtime: AICWarpRuntime,
        policy: ACTPolicy,
        normalizer: PolicyNormalizer,
    ):
        """Bind the trained policy to one camera-enabled MJWarp runtime.

        Args:
            config: Strict merged evaluation configuration.
            runtime: Single-world camera-enabled MJWarp runtime.
            policy: Loaded evaluation-mode ACT policy.
            normalizer: Checkpoint normalization applied to observations/actions.
        """

        self.config = config
        self.runtime = runtime
        self.policy = policy
        self.normalizer = normalizer
        self.image_buffer = DatasetImageBuffer(config, runtime)
        self.action = JointDeltaAction(
            runtime.num_envs, runtime.robot.joints.count, runtime.device
        )
        joints = runtime.robot.joints
        torch_device = normalizer.device
        self.qpos = wp.to_torch(runtime.data.qpos)
        self.qvel = wp.to_torch(runtime.data.qvel)
        self.qpos_indices = torch.tensor(
            joints.qpos_addresses, dtype=torch.long, device=torch_device
        )
        self.qvel_indices = torch.tensor(
            joints.dof_addresses, dtype=torch.long, device=torch_device
        )
        self.qpos_addresses = wp.array(
            joints.qpos_addresses, dtype=int, device=runtime.device
        )
        self.joint_lower = wp.array(
            joints.ranges[:, 0], dtype=float, device=runtime.device
        )
        self.joint_upper = wp.array(
            joints.ranges[:, 1], dtype=float, device=runtime.device
        )
        self.maximum_joint_step = wp.array(
            config["expert"]["maximum_joint_step"],
            dtype=float,
            device=runtime.device,
        )

    def state_observation(self) -> torch.Tensor:
        """Return configured arm-state fields directly from MJWarp memory."""

        fields: list[torch.Tensor] = []
        for field_name in self.config["policy"]["state_fields"]:
            if field_name == "qpos":
                fields.append(self.qpos.index_select(1, self.qpos_indices))
            elif field_name == "qvel":
                fields.append(self.qvel.index_select(1, self.qvel_indices))
            else:
                raise ValueError(f"Unsupported policy state field: {field_name}")
        return torch.cat(fields, dim=1)

    @torch.no_grad()
    def act(self) -> JointDeltaAction:
        """Infer and apply the first action from a newly predicted ACT chunk."""

        resized_images = self.image_buffer.resize(self.runtime.rgb)
        wp.synchronize_device(self.runtime.device)
        observation = {
            camera_feature_name(name): wp.to_torch(image).permute(0, 3, 1, 2)
            for name, image in resized_images.items()
        }
        observation["observation.state"] = self.state_observation()
        batch = self.normalizer.prepare_observation(observation)
        with torch.autocast(
            device_type=self.normalizer.device.type,
            dtype=self.normalizer.amp_dtype,
            enabled=self.config["training"]["use_amp"],
        ):
            action_chunk = self.policy.predict_action_chunk(batch)
        predicted_delta = self.normalizer.denormalize_action(
            action_chunk[:, 0, :]
        ).float().contiguous()
        if not torch.isfinite(predicted_delta).all():
            raise RuntimeError("Policy produced a non-finite joint action")
        if self.normalizer.device.type == "cuda":
            torch.cuda.synchronize(self.normalizer.device)
        wp.launch(
            apply_joint_delta_action,
            dim=(self.runtime.num_envs, self.runtime.robot.joints.count),
            inputs=[
                self.runtime.data.qpos,
                wp.from_torch(predicted_delta, dtype=wp.float32),
                self.qpos_addresses,
                self.joint_lower,
                self.joint_upper,
                self.maximum_joint_step,
                self.config["expert"]["joint_limit_margin"],
            ],
            outputs=[self.action.position, self.runtime.hold_command.position],
            device=self.runtime.device,
        )
        return self.action


class PolicyGoalMonitor:
    """Use privileged simulation state only to terminate policy episodes."""

    def __init__(self, config: dict[str, Any], runtime: AICWarpRuntime):
        """Resolve configured bodies and allocate device error outputs.

        Args:
            config: Strict merged evaluation configuration.
            runtime: Runtime containing compiled body topology and transforms.
        """

        expert = config["expert"]
        self.runtime = runtime
        self.controlled_body_id = required_model_id(
            runtime.host_model,
            mujoco.mjtObj.mjOBJ_BODY,
            expert["controlled_body"],
        )
        self.target_body_id = required_model_id(
            runtime.host_model,
            mujoco.mjtObj.mjOBJ_BODY,
            expert["target_body"],
        )
        self.offset_position = wp.vec3(expert["goal_offset_position"])
        self.offset_rotation = wp.mat33(*expert["goal_offset_rotation_matrix"])
        self.current_position = wp.zeros(
            runtime.num_envs, dtype=wp.vec3, device=runtime.device
        )
        self.goal_position = wp.zeros(
            runtime.num_envs, dtype=wp.vec3, device=runtime.device
        )
        self.position_error = wp.zeros(
            runtime.num_envs, dtype=float, device=runtime.device
        )
        self.orientation_error = wp.zeros(
            runtime.num_envs, dtype=float, device=runtime.device
        )

    def measure(self) -> WorkspaceSample:
        """Return the current SFP/goal workspace observation for environment zero."""

        wp.launch(
            measure_cartesian_goal_error,
            dim=self.runtime.num_envs,
            inputs=[
                self.runtime.data.xpos,
                self.runtime.data.xmat,
                self.controlled_body_id,
                self.target_body_id,
                self.offset_position,
                self.offset_rotation,
            ],
            outputs=[
                self.current_position,
                self.goal_position,
                self.position_error,
                self.orientation_error,
            ],
            device=self.runtime.device,
        )
        return WorkspaceSample(
            sfp_position=self.current_position.numpy()[0].copy(),
            goal_position=self.goal_position.numpy()[0].copy(),
            position_error=float(self.position_error.numpy()[0]),
            orientation_error=float(self.orientation_error.numpy()[0]),
        )


def terminal_reason(
    config: dict[str, Any],
    camera_steps: int,
    success_streak: int,
) -> str | None:
    """Return the configured episode terminal reason, if any.

    Args:
        config: Strict merged evaluation configuration.
        camera_steps: Policy observations processed in the current episode.
        success_streak: Consecutive in-tolerance observations.

    Returns:
        ``success``, ``timeout``, or ``None`` while the episode remains active.
    """

    expert = config["expert"]
    if success_streak >= expert["success_consecutive_steps"]:
        return "success"
    if camera_steps >= config["evaluation"]["maximum_episode_steps"]:
        return "timeout"
    return None


def run_inference(config: dict[str, Any]) -> None:
    """Run the configured number of closed-loop policy episodes.

    Args:
        config: Strict merged evaluation configuration.
    """

    scene_path = prepare_scene(config)
    policy, statistics = load_policy_checkpoint(config)
    runtime = AICWarpRuntime(config, render_cameras=True)
    controller = PolicyController(
        config, runtime, policy, PolicyNormalizer(config, statistics)
    )
    monitor = PolicyGoalMonitor(config, runtime)
    metrics = RolloutMetricsRecorder(config)
    outputs = RuntimeOutputs(config, runtime)
    expert = config["expert"]
    realtime = config["visualization"]["realtime"]
    timestep = config["physics"]["timestep"]
    pause_seconds = config["evaluation"]["reset_pause_seconds"]
    rollout_episodes = config["evaluation"]["rollout_episodes"]
    episode_index = 1
    camera_steps = 0
    success_streak = 0
    physics_steps = 0
    clock_start = time.perf_counter()
    last_sample: WorkspaceSample | None = None
    metrics.start_episode(episode_index)
    session_status = "failed"
    print(f"Scene: {scene_path}")
    print(f"Policy: {config['evaluation']['checkpoint_directory']}")
    print(f"Rollout metrics: {metrics.directory}")
    print(
        f"Running {rollout_episodes} closed-loop episodes; "
        "press Ctrl+C to stop early."
    )
    print(f"Episode {episode_index} started")
    try:
        while True:
            events = runtime.step()
            physics_steps += 1
            if realtime:
                wait_for_realtime(clock_start, physics_steps, timestep)
            if not events.camera:
                continue
            outputs.update(runtime)
            if not bool(runtime.tare_ready.numpy()[0]):
                continue

            camera_steps += 1
            last_sample = monitor.measure()
            metrics.record(camera_steps, last_sample)
            within_tolerance = (
                last_sample.position_error <= expert["position_tolerance"]
                and last_sample.orientation_error
                <= expert["orientation_tolerance"]
            )
            success_streak = success_streak + 1 if within_tolerance else 0
            reason = terminal_reason(config, camera_steps, success_streak)
            if reason is None:
                controller.act()
                continue

            print(
                f"Episode {episode_index} {reason}: {camera_steps} policy steps, "
                f"position={last_sample.position_error:.6f} m, "
                f"orientation={last_sample.orientation_error:.6f} rad"
            )
            episode_metrics = metrics.finish_episode(reason, last_sample)
            print(
                f"  path={episode_metrics['workspace_path_length_m']:.6f} m, "
                f"progress={episode_metrics['position_progress_m']:.6f} m, "
                f"closest={episode_metrics['minimum_position_error_m']:.6f} m"
            )
            if episode_index >= rollout_episodes:
                session_status = "completed"
                print(f"Completed {rollout_episodes} configured rollout episodes.")
                break
            if pause_seconds > 0.0:
                time.sleep(pause_seconds)
            runtime.reset([0])
            episode_index += 1
            camera_steps = 0
            success_streak = 0
            physics_steps = 0
            clock_start = time.perf_counter()
            last_sample = None
            metrics.start_episode(episode_index)
            print(f"Episode {episode_index} started")
    except KeyboardInterrupt:
        if metrics.has_active_samples and last_sample is not None:
            metrics.finish_episode("interrupted", last_sample)
        session_status = "stopped"
        print("\nStopping policy inference cleanly.")
    finally:
        metrics.close(session_status)
        outputs.close()


def main() -> None:
    """Load canonical overlays and run closed-loop policy evaluation."""

    run_inference(load_evaluation_config())
