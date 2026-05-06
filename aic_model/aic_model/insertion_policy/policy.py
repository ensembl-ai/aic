from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Wrench
from std_msgs.msg import Header
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat

from aic_model.insertion_policy.config import RuntimeConfig, load_runtime_config
from aic_model.insertion_policy.training.observation import (
    ObservationScales,
    encode_actor_observation,
    insertion_metrics,
    pose_msg_to_matrix,
)


def _matrix_to_pose(transform: np.ndarray) -> Pose:
    quat_wxyz = mat2quat(transform[:3, :3])
    pose = Pose()
    pose.position.x = float(transform[0, 3])
    pose.position.y = float(transform[1, 3])
    pose.position.z = float(transform[2, 3])
    pose.orientation.w = float(quat_wxyz[0])
    pose.orientation.x = float(quat_wxyz[1])
    pose.orientation.y = float(quat_wxyz[2])
    pose.orientation.z = float(quat_wxyz[3])
    return pose


def _delta_transform(action: np.ndarray, config: RuntimeConfig) -> np.ndarray:
    delta = np.eye(4, dtype=np.float64)
    delta[:3, 3] = action[:3] * config.action_translation_scale_m
    rotvec = action[3:] * config.action_rotation_scale_rad
    delta[:3, :3] = euler2mat(float(rotvec[0]), float(rotvec[1]), float(rotvec[2]), axes="sxyz")
    return delta


class InsertionPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._parent_node.declare_parameter("insertion_policy.config_path", "")
        config_path = (
            self._parent_node.get_parameter("insertion_policy.config_path")
            .get_parameter_value()
            .string_value
        )
        if not config_path:
            raise ValueError("Required parameter 'insertion_policy.config_path' is empty.")
        self._config = load_runtime_config(config_path)
        self._entrance_pose: PoseStamped | None = None
        self._entrance_sub = self._parent_node.create_subscription(
            PoseStamped,
            self._config.entrance_topic,
            self._entrance_callback,
            1,
        )

        import torch

        from aic_model.insertion_policy.training.actor_critic import load_actor_checkpoint

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._actor, self._obs_mean, self._obs_std = load_actor_checkpoint(
            self._config.checkpoint_path,
            self._device,
        )
        self._observation_scales = ObservationScales(
            position_scale_m=self._config.position_scale_m,
            progress_scale_m=self._config.progress_scale_m,
            force_scale_n=self._config.force_scale_n,
            torque_scale_nm=self._config.torque_scale_nm,
            insertion_axis_entrance=self._config.insertion_axis_entrance,
        )

    def _entrance_callback(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != self._config.pose_command_frame:
            raise ValueError(
                f"Entrance pose frame must be {self._config.pose_command_frame}, got {msg.header.frame_id}."
            )
        self._entrance_pose = msg

    def _wait_for_inputs(self, get_observation: GetObservationCallback) -> tuple[object, PoseStamped]:
        start = time.monotonic()
        while time.monotonic() - start < self._config.entrance_timeout_s:
            obs = get_observation()
            if obs is not None and self._entrance_pose is not None:
                return obs, self._entrance_pose
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for observation and insertion entrance pose.")

    def _motion_update(self, desired_pose: np.ndarray) -> MotionUpdate:
        return MotionUpdate(
            header=Header(
                frame_id=self._config.pose_command_frame,
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=_matrix_to_pose(desired_pose),
            target_stiffness=np.diag(self._config.stiffness).flatten(),
            target_damping=np.diag(self._config.damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=self._config.wrench_feedback_gains,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

    def _force_norm(self, observation) -> float:
        force = observation.wrist_wrench.wrench.force
        return float(np.linalg.norm([force.x, force.y, force.z]))

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        observation, entrance = self._wait_for_inputs(get_observation)
        entrance_pose_base = pose_msg_to_matrix(entrance.pose)
        desired_pose = pose_msg_to_matrix(observation.controller_state.tcp_pose)
        previous_action = np.zeros(6, dtype=np.float32)

        duration_s = min(float(task.time_limit), self._config.max_policy_duration_s)
        start_time = self.time_now()
        next_feedback_time = 0.0

        while True:
            elapsed_s = (self.time_now() - start_time).nanoseconds * 1.0e-9
            if elapsed_s >= duration_s:
                send_feedback("insertion policy timeout")
                return False

            observation = get_observation()
            if observation is None:
                self.sleep_for(self._config.control_period_s)
                continue

            if self._force_norm(observation) > self._config.force_guard_n:
                send_feedback("force guard exceeded")
                return False

            actor_obs = encode_actor_observation(
                observation,
                entrance_pose_base,
                previous_action,
                self._observation_scales,
            )
            with self._torch.inference_mode():
                obs_tensor = self._torch.as_tensor(actor_obs, dtype=self._torch.float32, device=self._device).reshape(1, -1)
                obs_tensor = (obs_tensor - self._obs_mean) / self._obs_std
                action = self._actor(obs_tensor).detach().cpu().numpy().reshape(-1)
            action = np.clip(action.astype(np.float64), -1.0, 1.0)
            desired_pose = desired_pose @ _delta_transform(action, self._config)

            move_robot(motion_update=self._motion_update(desired_pose))
            previous_action = action.astype(np.float32)

            tcp_pose_base = pose_msg_to_matrix(observation.controller_state.tcp_pose)
            lateral, angle, depth = insertion_metrics(
                tcp_pose_base,
                entrance_pose_base,
                self._config.insertion_axis_entrance,
            )
            if (
                lateral <= self._config.success_lateral_m
                and angle <= self._config.success_angle_rad
                and depth >= self._config.success_depth_m
            ):
                send_feedback("insertion policy success")
                return True

            if elapsed_s >= next_feedback_time:
                send_feedback(
                    f"depth={depth:.4f} lateral={lateral:.4f} angle={angle:.4f}"
                )
                next_feedback_time = elapsed_s + self._config.feedback_period_s
            self.sleep_for(self._config.control_period_s)
