from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Pose
from transforms3d.quaternions import quat2mat

from aic_model_interfaces.msg import Observation


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
OBSERVATION_DIM = 28


@dataclass(frozen=True)
class ObservationScales:
    position_scale_m: float
    progress_scale_m: float
    force_scale_n: float
    torque_scale_nm: float
    insertion_axis_entrance: np.ndarray


def pose_msg_to_matrix(pose: Pose) -> np.ndarray:
    quat_wxyz = np.array(
        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat2mat(quat_wxyz)
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def _joint_positions(observation: Observation) -> np.ndarray:
    observed = dict(zip(observation.joint_states.name, observation.joint_states.position))
    missing = [name for name in ARM_JOINT_NAMES if name not in observed]
    if missing:
        raise ValueError(f"Observation missing arm joints: {missing}")
    return np.asarray([observed[name] for name in ARM_JOINT_NAMES], dtype=np.float32)


def _wrench_vector(observation: Observation) -> np.ndarray:
    wrench = observation.wrist_wrench.wrench
    return np.asarray(
        [
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
        ],
        dtype=np.float32,
    )


def insertion_metrics(
    tcp_pose_base: np.ndarray,
    entrance_pose_base: np.ndarray,
    insertion_axis_entrance: np.ndarray,
) -> tuple[float, float, float]:
    entrance_axis_base = entrance_pose_base[:3, :3] @ insertion_axis_entrance
    tcp_delta_base = tcp_pose_base[:3, 3] - entrance_pose_base[:3, 3]
    depth = float(np.dot(tcp_delta_base, entrance_axis_base))

    tcp_delta_entrance = entrance_pose_base[:3, :3].T @ tcp_delta_base
    lateral = float(np.linalg.norm(tcp_delta_entrance[:2]))

    rot_entrance_tcp = entrance_pose_base[:3, :3].T @ tcp_pose_base[:3, :3]
    trace = float(np.clip((np.trace(rot_entrance_tcp) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(trace))
    return lateral, angle, depth


def encode_actor_observation(
    observation: Observation,
    entrance_pose_base: np.ndarray,
    previous_action: np.ndarray,
    scales: ObservationScales,
) -> np.ndarray:
    previous_action = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous_action.size != 6:
        raise ValueError("previous_action must have length 6.")

    tcp_pose_base = pose_msg_to_matrix(observation.controller_state.tcp_pose)
    transform_tcp_entrance = np.linalg.inv(tcp_pose_base) @ entrance_pose_base
    rel_pos_tcp = transform_tcp_entrance[:3, 3] / scales.position_scale_m
    rel_rot_6d = transform_tcp_entrance[:3, :2].reshape(-1)

    _, _, depth = insertion_metrics(
        tcp_pose_base,
        entrance_pose_base,
        scales.insertion_axis_entrance,
    )
    progress = np.asarray([depth / scales.progress_scale_m], dtype=np.float32)

    wrench = _wrench_vector(observation)
    wrench[:3] /= scales.force_scale_n
    wrench[3:] /= scales.torque_scale_nm

    encoded = np.concatenate(
        [
            _joint_positions(observation) / np.pi,
            rel_pos_tcp.astype(np.float32),
            rel_rot_6d.astype(np.float32),
            wrench.astype(np.float32),
            progress,
            previous_action.astype(np.float32),
        ]
    ).astype(np.float32)
    if encoded.size != OBSERVATION_DIM:
        raise RuntimeError(f"Internal observation size error: {encoded.size} != {OBSERVATION_DIM}")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("Actor observation contains non-finite values.")
    return encoded
