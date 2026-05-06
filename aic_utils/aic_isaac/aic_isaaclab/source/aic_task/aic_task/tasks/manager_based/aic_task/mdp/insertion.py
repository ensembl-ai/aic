from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
)


@dataclass
class InsertionBuffers:
    true_entrance_pos_w: torch.Tensor
    true_entrance_quat_w: torch.Tensor
    noisy_entrance_pos_w: torch.Tensor
    noisy_entrance_quat_w: torch.Tensor
    gripper_plug_pos_tcp: torch.Tensor
    gripper_plug_quat_tcp: torch.Tensor
    position_curriculum_m: torch.Tensor
    orientation_curriculum_rad: torch.Tensor


def _buffers(env) -> InsertionBuffers:
    if hasattr(env, "_aic_insertion_buffers"):
        return env._aic_insertion_buffers
    device = env.device
    zeros3 = torch.zeros(env.num_envs, 3, device=device)
    identity_quat = torch.zeros(env.num_envs, 4, device=device)
    identity_quat[:, 0] = 1.0
    env._aic_insertion_buffers = InsertionBuffers(
        true_entrance_pos_w=zeros3.clone(),
        true_entrance_quat_w=identity_quat.clone(),
        noisy_entrance_pos_w=zeros3.clone(),
        noisy_entrance_quat_w=identity_quat.clone(),
        gripper_plug_pos_tcp=zeros3.clone(),
        gripper_plug_quat_tcp=identity_quat.clone(),
        position_curriculum_m=torch.full((1,), 0.005, device=device),
        orientation_curriculum_rad=torch.zeros(1, device=device),
    )
    return env._aic_insertion_buffers


def _uniform(
    n: int,
    ranges: dict[str, tuple[float, float]],
    device: torch.device,
    names: tuple[str, str, str],
) -> torch.Tensor:
    values = torch.zeros(n, 3, device=device)
    for index, name in enumerate(names):
        lo, hi = ranges[name]
        values[:, index] = torch.empty(n, device=device).uniform_(lo, hi)
    return values


def sample_insertion_episode(
    env,
    env_ids: torch.Tensor,
    port_cfg: SceneEntityCfg,
    entrance_offset_port: tuple[float, float, float],
    entrance_rpy_port: tuple[float, float, float],
    entrance_position_noise: dict[str, tuple[float, float]],
    entrance_rpy_noise: dict[str, tuple[float, float]],
    gripper_plug_translation: dict[str, tuple[float, float]],
    gripper_plug_rpy: dict[str, tuple[float, float]],
) -> None:
    buffers = _buffers(env)
    port: RigidObject = env.scene[port_cfg.name]
    ids = env_ids.to(device=env.device)
    n = len(ids)

    port_pos = port.data.root_pos_w[ids]
    port_quat = port.data.root_quat_w[ids]
    entrance_offset = torch.tensor(entrance_offset_port, dtype=torch.float32, device=env.device).repeat(n, 1)
    entrance_pos = port_pos + quat_apply(port_quat, entrance_offset)
    entrance_rpy = torch.tensor(entrance_rpy_port, dtype=torch.float32, device=env.device).repeat(n, 1)
    entrance_quat = quat_mul(
        port_quat,
        quat_from_euler_xyz(entrance_rpy[:, 0], entrance_rpy[:, 1], entrance_rpy[:, 2]),
    )

    pos_noise = _uniform(n, entrance_position_noise, env.device, ("x", "y", "z"))
    rpy_noise = _uniform(n, entrance_rpy_noise, env.device, ("roll", "pitch", "yaw"))
    noise_quat = quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])

    buffers.true_entrance_pos_w[ids] = entrance_pos
    buffers.true_entrance_quat_w[ids] = entrance_quat
    buffers.noisy_entrance_pos_w[ids] = entrance_pos + pos_noise
    buffers.noisy_entrance_quat_w[ids] = quat_mul(entrance_quat, noise_quat)
    buffers.gripper_plug_pos_tcp[ids] = _uniform(
        n,
        gripper_plug_translation,
        env.device,
        ("x", "y", "z"),
    )
    plug_rpy = _uniform(n, gripper_plug_rpy, env.device, ("roll", "pitch", "yaw"))
    buffers.gripper_plug_quat_tcp[ids] = quat_from_euler_xyz(
        plug_rpy[:, 0],
        plug_rpy[:, 1],
        plug_rpy[:, 2],
    )


def _tcp_pose(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    return (
        _first_body_tensor(robot.data.body_pos_w, asset_cfg),
        _first_body_tensor(robot.data.body_quat_w, asset_cfg),
    )


def _first_body_tensor(data: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    body_ids = asset_cfg.body_ids
    body_id = 0 if isinstance(body_ids, slice) else body_ids[0]
    return data[:, body_id, :]


def _plug_tip_pose(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    buffers = _buffers(env)
    tcp_pos, tcp_quat = _tcp_pose(env, asset_cfg)
    plug_pos = tcp_pos + quat_apply(tcp_quat, buffers.gripper_plug_pos_tcp)
    plug_quat = quat_mul(tcp_quat, buffers.gripper_plug_quat_tcp)
    return plug_pos, plug_quat


def _relative_to_entrance(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    entrance_pos_w: torch.Tensor,
    entrance_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_pos = quat_apply_inverse(entrance_quat_w, pos_w - entrance_pos_w)
    rel_quat = quat_mul(entrance_quat_w * torch.tensor([1.0, -1.0, -1.0, -1.0], device=pos_w.device), quat_w)
    return rel_pos, rel_quat


def insertion_actor_observation(
    env,
    asset_cfg: SceneEntityCfg,
    insertion_axis_entrance: tuple[float, float, float],
    position_scale_m: float,
    progress_scale_m: float,
    force_scale_n: float,
    torque_scale_nm: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    robot: Articulation = env.scene[asset_cfg.name]
    tcp_pos, tcp_quat = _tcp_pose(env, asset_cfg)
    rel_pos_tcp = quat_apply_inverse(tcp_quat, buffers.noisy_entrance_pos_w - tcp_pos) / position_scale_m
    rel_quat_tcp = quat_mul(tcp_quat * torch.tensor([1.0, -1.0, -1.0, -1.0], device=env.device), buffers.noisy_entrance_quat_w)
    rel_rot = matrix_from_quat(rel_quat_tcp)[:, :, :2].reshape(env.num_envs, 6)

    axis = torch.tensor(insertion_axis_entrance, dtype=torch.float32, device=env.device)
    entrance_axis_w = quat_apply(buffers.noisy_entrance_quat_w, axis.repeat(env.num_envs, 1))
    progress = torch.sum((tcp_pos - buffers.noisy_entrance_pos_w) * entrance_axis_w, dim=1, keepdim=True)
    progress = progress / progress_scale_m

    wrench = _first_body_tensor(robot.data.body_incoming_wrench_w, asset_cfg).clone()
    wrench[:, :3] /= force_scale_n
    wrench[:, 3:] /= torque_scale_nm

    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids] / torch.pi
    return torch.cat([joint_pos, rel_pos_tcp, rel_rot, wrench, progress, env.action_manager.action], dim=1)


def insertion_privileged_observation(
    env,
    asset_cfg: SceneEntityCfg,
    insertion_axis_entrance: tuple[float, float, float],
) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, plug_quat = _plug_tip_pose(env, asset_cfg)
    rel_true_pos, rel_true_quat = _relative_to_entrance(
        plug_pos,
        plug_quat,
        buffers.true_entrance_pos_w,
        buffers.true_entrance_quat_w,
    )
    rel_noisy_pos = quat_apply_inverse(
        buffers.true_entrance_quat_w,
        buffers.noisy_entrance_pos_w - buffers.true_entrance_pos_w,
    )
    angle_error = quat_error_magnitude(plug_quat, buffers.true_entrance_quat_w).unsqueeze(1)
    axis = torch.tensor(insertion_axis_entrance, dtype=torch.float32, device=env.device)
    entrance_axis_w = quat_apply(buffers.true_entrance_quat_w, axis.repeat(env.num_envs, 1))
    depth = torch.sum((plug_pos - buffers.true_entrance_pos_w) * entrance_axis_w, dim=1, keepdim=True)
    return torch.cat([rel_true_pos, rel_true_quat, rel_noisy_pos, angle_error, depth], dim=1)


def insertion_lateral_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sigma_m: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, plug_quat = _plug_tip_pose(env, asset_cfg)
    rel_pos, _ = _relative_to_entrance(plug_pos, plug_quat, buffers.true_entrance_pos_w, buffers.true_entrance_quat_w)
    lateral_sq = torch.sum(rel_pos[:, :2] ** 2, dim=1)
    return torch.exp(-lateral_sq / (sigma_m**2))


def insertion_angle_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sigma_rad: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    _, plug_quat = _plug_tip_pose(env, asset_cfg)
    angle = quat_error_magnitude(plug_quat, buffers.true_entrance_quat_w)
    return torch.exp(-(angle**2) / (sigma_rad**2))


def insertion_depth_reward(
    env,
    asset_cfg: SceneEntityCfg,
    insertion_axis_entrance: tuple[float, float, float],
    depth_scale_m: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, _ = _plug_tip_pose(env, asset_cfg)
    axis = torch.tensor(insertion_axis_entrance, dtype=torch.float32, device=env.device)
    entrance_axis_w = quat_apply(buffers.true_entrance_quat_w, axis.repeat(env.num_envs, 1))
    depth = torch.sum((plug_pos - buffers.true_entrance_pos_w) * entrance_axis_w, dim=1)
    return torch.clamp(depth / depth_scale_m, -1.0, 1.0)


def insertion_success(
    env,
    asset_cfg: SceneEntityCfg,
    insertion_axis_entrance: tuple[float, float, float],
    lateral_threshold_m: float,
    angle_threshold_rad: float,
    depth_threshold_m: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, plug_quat = _plug_tip_pose(env, asset_cfg)
    rel_pos, _ = _relative_to_entrance(plug_pos, plug_quat, buffers.true_entrance_pos_w, buffers.true_entrance_quat_w)
    lateral = torch.linalg.norm(rel_pos[:, :2], dim=1)
    angle = quat_error_magnitude(plug_quat, buffers.true_entrance_quat_w)
    axis = torch.tensor(insertion_axis_entrance, dtype=torch.float32, device=env.device)
    entrance_axis_w = quat_apply(buffers.true_entrance_quat_w, axis.repeat(env.num_envs, 1))
    depth = torch.sum((plug_pos - buffers.true_entrance_pos_w) * entrance_axis_w, dim=1)
    return ((lateral <= lateral_threshold_m) & (angle <= angle_threshold_rad) & (depth >= depth_threshold_m)).float()


def force_guard(
    env,
    asset_cfg: SceneEntityCfg,
    threshold_n: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    forces = _first_body_tensor(robot.data.body_incoming_wrench_w, asset_cfg)[:, :3]
    return torch.linalg.norm(forces, dim=1) > threshold_n


def force_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    forces = _first_body_tensor(robot.data.body_incoming_wrench_w, asset_cfg)[:, :3]
    return torch.linalg.norm(forces, dim=1)


def insertion_workspace_guard(
    env,
    asset_cfg: SceneEntityCfg,
    max_lateral_m: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, plug_quat = _plug_tip_pose(env, asset_cfg)
    rel_pos, _ = _relative_to_entrance(plug_pos, plug_quat, buffers.true_entrance_pos_w, buffers.true_entrance_quat_w)
    return torch.linalg.norm(rel_pos[:, :2], dim=1) > max_lateral_m


def sapu_interpenetration_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    threshold_m: float,
) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, plug_quat = _plug_tip_pose(env, asset_cfg)
    rel_pos, _ = _relative_to_entrance(plug_pos, plug_quat, buffers.true_entrance_pos_w, buffers.true_entrance_quat_w)
    penetration = torch.clamp(-rel_pos[:, 2] - threshold_m, min=0.0)
    return penetration / threshold_m


def sdf_query_reward(env, asset_cfg: SceneEntityCfg, sigma_m: float) -> torch.Tensor:
    buffers = _buffers(env)
    plug_pos, plug_quat = _plug_tip_pose(env, asset_cfg)
    rel_pos, rel_quat = _relative_to_entrance(plug_pos, plug_quat, buffers.true_entrance_pos_w, buffers.true_entrance_quat_w)
    pos_error = torch.linalg.norm(rel_pos, dim=1)
    angle_error = quat_error_magnitude(rel_quat, torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    return torch.exp(-(pos_error**2) / (sigma_m**2)) * torch.exp(-(angle_error**2) / (0.08**2))


def insertion_sampling_curriculum(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    insertion_axis_entrance: tuple[float, float, float],
    success_advance_threshold: float,
    success_revert_threshold: float,
    position_step_m: float,
    position_max_m: float,
    orientation_step_rad: float,
    orientation_max_rad: float,
    lateral_threshold_m: float,
    angle_threshold_rad: float,
    depth_threshold_m: float,
) -> dict[str, torch.Tensor]:
    buffers = _buffers(env)
    success = insertion_success(
        env,
        asset_cfg,
        insertion_axis_entrance,
        lateral_threshold_m,
        angle_threshold_rad,
        depth_threshold_m,
    ).mean()
    if success > success_advance_threshold:
        buffers.position_curriculum_m[:] = torch.clamp(
            buffers.position_curriculum_m + position_step_m,
            max=position_max_m,
        )
        buffers.orientation_curriculum_rad[:] = torch.clamp(
            buffers.orientation_curriculum_rad + orientation_step_rad,
            max=orientation_max_rad,
        )
    elif success < success_revert_threshold:
        buffers.position_curriculum_m[:] = torch.clamp(
            buffers.position_curriculum_m - position_step_m,
            min=position_step_m,
        )
        buffers.orientation_curriculum_rad[:] = torch.clamp(
            buffers.orientation_curriculum_rad - orientation_step_rad,
            min=0.0,
        )
    return {
        "position_m": buffers.position_curriculum_m[0],
        "orientation_rad": buffers.orientation_curriculum_rad[0],
        "success_rate": success,
    }
