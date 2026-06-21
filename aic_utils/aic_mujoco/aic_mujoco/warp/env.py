"""Direct MuJoCo vector environment for AIC insertion R&D.

The environment keeps the action/observation/reward contract we want before
full GPU batching:

  reset:
    use the same SFP-tip pre-insertion transform chain as
    ``hold_fixed_target.py`` and ``demo_joint_target_control.py``.

  action:
    3D Cartesian delta in world coordinates at ``gripper_tcp``.

  step:
    Cartesian delta -> differential IK from the TCP Jacobian -> joint
    impedance torques -> ``mj_step``.

  observation:
    joint positions, TCP/plug/port positions, zeroed force/torque, contact max
    penetration, previous action.

  reward:
    progress toward the port, lateral error, zeroed force penalty, action
    penalty, and max-penetration penalty.

This module intentionally has no ROS and no MJLab dependency. It is the simple,
inspectable baseline that later maps to a batched MuJoCo Warp/RSL-RL wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from aic_mujoco.config import load_json_config
from aic_mujoco.controllers import JointImpedanceConfig, JointImpedanceController
from aic_mujoco.joints import JointGroup, PassiveJointGroup, TorqueMode
from aic_mujoco.mjlab.observations import (
    ContactObservation,
    WrenchObservation,
    contact_observation,
    force_torque_observation,
)
from aic_mujoco.mjlab.reset import (
    PreinsertResetConfig,
    WrenchZeroingConfig,
    reset_preinsert_episode,
)
from aic_mujoco.mjlab.rewards import (
    ForcePenaltyTerms,
    InsertionProgressTerms,
    RewardComposition,
    compose_reward,
    insertion_axis_progress,
    zeroed_wrench_penalty,
)
from aic_mujoco.mjlab.step import (
    CartesianStepConfig,
    CartesianStepResult,
)
from aic_mujoco.utils import body_transform
from aic_mujoco.commands import JointTarget


@dataclass(frozen=True)
class AicInsertionVecEnvConfig:
    """Construction inputs for the direct prototype vector env.

    Args:
        xml_path: MuJoCo scene to load. Training/debug currently use
            ``scene.xml`` for the full sensor/contact scene.
        task_config: Merged JSON config with joints, reset frames, gains,
            rewards, termination thresholds, and sensor names.
        num_envs: Number of independent ``MjData`` copies to step.
        device: Torch device used for tensors exposed to policy/RSL-RL.
    """

    xml_path: Path
    task_config: dict[str, Any]
    num_envs: int = 32
    device: str = "cuda"

    @classmethod
    def from_files(
        cls,
        xml_path: str | Path,
        config_path: str | Path,
        num_envs: int | None = None,
        device: str = "cuda",
    ) -> "AicInsertionVecEnvConfig":
        """Load JSON config and normalize file paths.

        ``num_envs`` overrides the config value when provided. This keeps
        command-line experiments cheap without editing the experiment JSON.
        """

        task_config = load_json_config(config_path)
        return cls(
            xml_path=Path(xml_path).expanduser().resolve(),
            task_config=task_config,
            num_envs=int(num_envs if num_envs is not None else task_config["num_envs"]),
            device=device,
        )


@dataclass(frozen=True)
class EnvStepMetrics:
    """Per-env diagnostics computed after each control step.

    These values are intentionally explicit rather than buried in ``extras``:
    rewards, Viser/debug logging, and termination all depend on them. They also
    reveal where time/quality is going: IK error, contact penetration, force
    norm, progress, and lateral alignment.
    """

    reward: RewardComposition
    progress: InsertionProgressTerms
    wrench: WrenchObservation
    force_terms: ForcePenaltyTerms
    contact: ContactObservation
    lateral_error: float
    action_norm: float
    ik_ok: bool
    ik_error: float
    tcp_error: float


def _array(values: Any) -> np.ndarray:
    """Convert config lists/scalars into float NumPy arrays."""

    return np.asarray(values, dtype=float)


def _torch_tensor(values: np.ndarray, device: str) -> torch.Tensor:
    """Move a NumPy float array onto the policy device as ``float32``."""

    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _torch_bool(values: np.ndarray, device: str) -> torch.Tensor:
    """Move a NumPy boolean array onto the policy device."""

    return torch.as_tensor(values, dtype=torch.bool, device=device)


def _numpy_actions(actions: Any) -> np.ndarray:
    """Accept Torch or NumPy actions and return CPU NumPy for MuJoCo stepping."""

    if isinstance(actions, torch.Tensor):
        return actions.detach().cpu().numpy()
    return np.asarray(actions, dtype=float)


def _object_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    """Resolve a required MuJoCo object name and fail loudly if absent."""

    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name!r}")
    return int(obj_id)


def _lateral_error_to_axis(
    world_T_port_bottom: np.ndarray,
    world_T_port_entrance: np.ndarray,
    world_T_plug: np.ndarray,
) -> float:
    """Return plug lateral distance from the port insertion axis.

    Args:
        world_T_port_bottom: Pose of the nominal inserted/end frame.
        world_T_port_entrance: Pose of the entrance frame.
        world_T_plug: Current pose of the SFP tip.

    The insertion axis is entrance -> bottom. The reward uses this as an
    alignment penalty separate from axial progress.
    """

    bottom = world_T_port_bottom[:3, 3]
    entrance = world_T_port_entrance[:3, 3]
    plug = world_T_plug[:3, 3]
    axis = bottom - entrance
    depth = float(np.linalg.norm(axis))
    if depth <= 1e-12:
        return 0.0
    unit_axis = axis / depth
    closest = entrance + np.dot(plug - entrance, unit_axis) * unit_axis
    return float(np.linalg.norm(plug - closest))


class AicInsertionVecEnv:
    """Small RSL-style vector env using plain MuJoCo data copies.

    This is deliberately conservative: one shared ``MjModel`` and one ``MjData``
    per environment. Reset solves IK per environment so future randomized
    board/port poses get their own valid pre-insertion state. Runtime actions
    do not call the global IK solver; they use the local TCP Jacobian and a
    joint impedance controller.
    """

    num_actions = 3

    def __init__(self, cfg: AicInsertionVecEnvConfig):
        """Load model/config and allocate one ``MjData`` per env.

        This is where most prototype startup cost lives: MuJoCo model loading,
        controller construction, object-name validation, and per-env buffers.
        Reset IK is not cached because future domain randomization must solve a
        fresh pre-insertion pose for each randomized board/port state.
        """

        self.cfg = cfg
        self.task_cfg = cfg.task_config
        self.device = cfg.device
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(cfg.num_envs)]
        self.num_envs = int(cfg.num_envs)

        self.controlled = JointGroup(
            self.model,
            self.task_cfg["controlled_joints"],
            torque_mode=TorqueMode(self.task_cfg["torque_mode"]),
        )
        self.passive = PassiveJointGroup(
            self.model,
            self.task_cfg["passive_joints"],
            mode=self.task_cfg["passive_mode"],
        )
        controller_cfg = JointImpedanceConfig(
            use_bias_compensation=bool(self.task_cfg["use_bias_compensation"]),
            torque_limits=_array(self.task_cfg["torque_limits"]),
            torque_rate_limits=_array(self.task_cfg["torque_rate_limits"]),
            clamp_to_joint_limits=True,
        )
        self.controllers = [
            JointImpedanceController(self.controlled, controller_cfg)
            for _ in range(self.num_envs)
        ]

        self.kp = _array(self.task_cfg["kp"])
        self.kd = _array(self.task_cfg["kd"])
        self.tcp_site_name = str(self.task_cfg["preinsert_tcp_site"])
        self.sfp_tip_body = str(self.task_cfg["preinsert_sfp_tip_body"])
        self.port_entrance_body = str(self.task_cfg["preinsert_port_body"])
        self.port_bottom_body = str(self.task_cfg["reward_port_bottom_body"])
        self.tcp_site_id = _object_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.tcp_site_name,
        )
        _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.sfp_tip_body)
        _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.port_entrance_body)
        _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.port_bottom_body)

        self.max_episode_length = int(self.task_cfg["max_episode_steps"])
        self.decimation = int(self.task_cfg["decimation"])
        self.action_scale = float(self.task_cfg["action_scale"])
        self.success_progress = float(self.task_cfg["success_progress"])
        self.terminate_force_limit = float(self.task_cfg["terminate_force_limit"])
        self.terminate_penetration_limit = float(
            self.task_cfg["terminate_penetration_limit"]
        )
        self.reward_progress_weight = float(self.task_cfg["reward_progress_weight"])
        self.reward_lateral_weight = float(self.task_cfg["reward_lateral_weight"])
        self.reward_force_weight = float(self.task_cfg["reward_force_weight"])
        self.reward_action_weight = float(self.task_cfg["reward_action_weight"])
        self.reward_penetration_weight = float(
            self.task_cfg["reward_penetration_weight"]
        )
        self.reward_force_limit = float(self.task_cfg["reward_force_limit"])

        self.step_cfg = CartesianStepConfig(
            ik_iters=int(self.task_cfg["ik_iters"]),
            ik_tolerance=float(self.task_cfg["ik_tolerance"]),
            ik_damping=float(self.task_cfg["ik_damping"]),
            ik_max_dq=float(self.task_cfg["ik_max_dq"]),
        )
        self.reset_cfg = PreinsertResetConfig(
            home_q=tuple(float(x) for x in self.task_cfg["ik_home_q"]),
            port_body=str(self.task_cfg["preinsert_port_body"]),
            tcp_site=str(self.task_cfg["preinsert_tcp_site"]),
            sfp_tip_body=str(self.task_cfg["preinsert_sfp_tip_body"]),
            tool_body=str(self.task_cfg["preinsert_tool_body"]),
            weld_child_body=str(self.task_cfg["preinsert_weld_child_body"]),
            payload_root_body=str(self.task_cfg["preinsert_payload_root_body"]),
            payload_root_freejoint=str(self.task_cfg["preinsert_payload_root_freejoint"]),
            height=float(self.task_cfg["preinsert_height"]),
        )
        self.wrench_cfg = WrenchZeroingConfig(
            force_sensor=str(self.task_cfg["force_sensor"]),
            torque_sensor=str(self.task_cfg["torque_sensor"]),
            settle_steps=int(self.task_cfg["sensor_zero_settle_steps"]),
            bias_samples=int(self.task_cfg["sensor_zero_samples"]),
            enabled=True,
        )

        self.q_start = np.zeros((self.num_envs, self.controlled.n), dtype=float)
        self.force_biases: list[np.ndarray | None] = [None] * self.num_envs
        self.torque_biases: list[np.ndarray | None] = [None] * self.num_envs
        self.target_pos = np.zeros((self.num_envs, 3), dtype=float)
        self.prev_action = np.zeros((self.num_envs, self.num_actions), dtype=float)
        self.episode_length_buf = np.zeros(self.num_envs, dtype=np.int64)
        self.reset_buf = np.ones(self.num_envs, dtype=bool)
        self.last_metrics: list[EnvStepMetrics | None] = [None] * self.num_envs
        self._last_obs = np.zeros((self.num_envs, 31), dtype=np.float32)
        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)
        self._dof_cols = np.array([h.qvel_addr for h in self.controlled.handles], dtype=int)
        self._q_lo, self._q_hi = self.controlled.joint_limits()

        self.reset()

    @property
    def num_obs(self) -> int:
        """Observation width exposed to RSL-RL and debug scripts."""

        return int(self._last_obs.shape[1])

    @property
    def num_privileged_obs(self) -> int:
        """Critic observation width.

        For the first prototype, actor and critic see the same observation.
        Teacher/student or privileged observations can split this later without
        changing the action interface.
        """

        return self.num_obs

    @property
    def physics_dt(self) -> float:
        """MuJoCo physics timestep in seconds."""

        return float(self.model.opt.timestep)

    @property
    def control_dt(self) -> float:
        """Policy/control interval after decimation."""

        return self.physics_dt * self.decimation

    def reset(self) -> torch.Tensor:
        """Reset all envs and return the first observation tensor."""

        self.reset_idx(np.arange(self.num_envs, dtype=np.int64))
        return self.get_observations()

    def reset_idx(self, env_ids: np.ndarray | list[int]) -> None:
        """Reset selected envs through the pre-insertion IK routine.

        Args:
            env_ids: Integer env indices to reset.

        Each reset computes the SFP-tip pre-insertion pose, solves robot IK,
        places the free payload so the held plug matches the kinematic target,
        settles the scene, and zeros the F/T sensor bias. This is intentionally
        per-env work so future randomized task-board poses are handled
        correctly.
        """

        for env_id in np.asarray(env_ids, dtype=np.int64):
            i = int(env_id)
            data = self.datas[i]
            controller = self.controllers[i]
            result = reset_preinsert_episode(
                model=self.model,
                data=data,
                controlled=self.controlled,
                passive=self.passive,
                controller=controller,
                kp=self.kp,
                kd=self.kd,
                reset_cfg=self.reset_cfg,
                wrench_cfg=self.wrench_cfg,
            )
            self.q_start[i] = result.q_start.copy()
            self.force_biases[i] = result.force_bias
            self.torque_biases[i] = result.torque_bias
            self.target_pos[i] = data.site_xpos[self.tcp_site_id].copy()
            self.prev_action[i] = 0.0
            self.episode_length_buf[i] = 0
            self.reset_buf[i] = False
            controller.reset(self.q_start[i].tolist())
            self.last_metrics[i] = self._compute_metrics(
                env_id=i,
                action=np.zeros(self.num_actions),
                step_result=None,
            )
        self._last_obs = self._collect_obs()

    def get_observations(self) -> torch.Tensor:
        """Return the cached observation tensor on ``self.device``."""

        return _torch_tensor(self._last_obs, self.device)

    def get_privileged_observations(self) -> torch.Tensor:
        """Return critic observations; currently identical to policy obs."""

        return self.get_observations()

    def step(self, actions: Any):
        """Advance every env by one policy step.

        Args:
            actions: Shape ``(num_envs, 3)`` Cartesian delta commands in
                normalized policy units. Values are clipped to ``[-1, 1]`` and
                scaled by ``action_scale`` before stepping.

        Returns:
            ``(obs, rewards, dones, extras)`` in the RSL-RL-style contract.
            Internally this still loops over ``MjData`` objects; replacing that
            loop with true MuJoCo Warp batched state is the planned speed path.
        """

        action_np = _numpy_actions(actions).astype(float)
        if action_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {self.num_actions}), "
                f"got {action_np.shape}"
            )
        action_np = np.clip(action_np, -1.0, 1.0)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)

        for env_id, data in enumerate(self.datas):
            if self.reset_buf[env_id]:
                self.reset_idx([env_id])

            delta_world = action_np[env_id] * self.action_scale
            step_result: CartesianStepResult | None = None
            for _ in range(self.decimation):
                step_result = self._step_cartesian_delta(env_id, delta_world)

            self.episode_length_buf[env_id] += 1
            self.prev_action[env_id] = action_np[env_id]
            metrics = self._compute_metrics(env_id, action_np[env_id], step_result)
            self.last_metrics[env_id] = metrics
            rewards[env_id] = float(metrics.reward.total)
            dones[env_id] = self._is_done(env_id, metrics)
            self.reset_buf[env_id] = dones[env_id]

        self._last_obs = self._collect_obs()
        return (
            self.get_observations(),
            _torch_tensor(rewards, self.device),
            _torch_bool(dones, self.device),
            self._extras(rewards, dones),
        )

    def _step_cartesian_delta(
        self,
        env_id: int,
        delta_world: np.ndarray,
    ) -> CartesianStepResult:
        """Apply one Cartesian delta through differential IK and impedance.

        Args:
            env_id: Environment index to step.
            delta_world: Desired TCP translation in world coordinates for this
                physics substep.

        The method uses MuJoCo's site Jacobian at ``gripper_tcp`` and a damped
        least-squares translation solve. It deliberately does not call the full
        EnsemblRobot IK solver during rollout; global IK is reset-only.
        """

        data = self.datas[env_id]
        dt = float(self.model.opt.timestep)
        data.qfrc_applied[:] = 0.0
        self.passive.enforce(data)
        mujoco.mj_forward(self.model, data)

        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.tcp_site_id)
        j_pos = self._jacp[:, self._dof_cols]
        damping = float(self.step_cfg.ik_damping)
        lhs = j_pos @ j_pos.T + (damping * damping) * np.eye(3)
        dq = j_pos.T @ np.linalg.solve(lhs, np.asarray(delta_world, dtype=float))
        dq_norm = float(np.linalg.norm(dq))
        max_dq = float(self.step_cfg.ik_max_dq)
        if dq_norm > max_dq:
            dq *= max_dq / dq_norm

        q_des = np.clip(self.controlled.q(data) + dq, self._q_lo, self._q_hi)
        target = JointTarget.position(
            q_des=q_des.tolist(),
            qd_des=np.zeros(self.controlled.n).tolist(),
            kp=self.kp.tolist(),
            kd=self.kd.tolist(),
            tau_ff=np.zeros(self.controlled.n).tolist(),
        )
        tau = self.controllers[env_id].compute(data, target, dt)
        self.controlled.apply_torque(data, tau.tolist())
        mujoco.mj_step(self.model, data)
        self.passive.enforce(data)
        mujoco.mj_forward(self.model, data)

        ee_pos = data.site_xpos[self.tcp_site_id].copy()
        return CartesianStepResult(
            q_des=q_des,
            tau=tau,
            ee_pos=ee_pos,
            ee_error=float(np.linalg.norm(delta_world)),
            ik_error=float(np.linalg.norm(j_pos @ dq - delta_world)),
            ik_ok=True,
        )

    def _collect_obs(self) -> np.ndarray:
        """Build the policy observation matrix for all envs.

        Observation layout is fixed at 31 floats: controlled joint positions,
        TCP position, SFP-tip position, port entrance, port bottom,
        tip-to-bottom vector, zeroed force, zeroed torque, max penetration, and
        previous action. Keep this stable while training checkpoints exist.
        """

        observations = np.zeros((self.num_envs, 31), dtype=np.float32)
        for env_id, data in enumerate(self.datas):
            world_T_tcp = np.eye(4, dtype=float)
            world_T_tcp[:3, :3] = data.site_xmat[self.tcp_site_id].reshape(3, 3)
            world_T_tcp[:3, 3] = data.site_xpos[self.tcp_site_id]
            world_T_plug = body_transform(self.model, data, self.sfp_tip_body)
            world_T_entrance = body_transform(self.model, data, self.port_entrance_body)
            world_T_bottom = body_transform(self.model, data, self.port_bottom_body)
            wrench = force_torque_observation(
                self.model,
                data,
                force_sensor=self.wrench_cfg.force_sensor,
                torque_sensor=self.wrench_cfg.torque_sensor,
                force_bias=self.force_biases[env_id],
                torque_bias=self.torque_biases[env_id],
            )
            contact = contact_observation(self.model, data)
            force = np.zeros(3) if wrench.force is None else wrench.force[:3]
            torque = np.zeros(3) if wrench.torque is None else wrench.torque[:3]
            parts = [
                self.controlled.q(data),
                world_T_tcp[:3, 3],
                world_T_plug[:3, 3],
                world_T_entrance[:3, 3],
                world_T_bottom[:3, 3],
                world_T_plug[:3, 3] - world_T_bottom[:3, 3],
                force,
                torque,
                np.array([contact.max_penetration], dtype=float),
                self.prev_action[env_id],
            ]
            observations[env_id] = np.concatenate(parts).astype(np.float32)
        return observations

    def _compute_metrics(
        self,
        env_id: int,
        action: np.ndarray,
        step_result: CartesianStepResult | None,
    ) -> EnvStepMetrics:
        """Compute reward ingredients and diagnostics for one env."""

        data = self.datas[env_id]
        world_T_plug = body_transform(self.model, data, self.sfp_tip_body)
        world_T_entrance = body_transform(self.model, data, self.port_entrance_body)
        world_T_bottom = body_transform(self.model, data, self.port_bottom_body)
        progress = insertion_axis_progress(
            world_T_port_bottom=world_T_bottom,
            world_T_port_entrance=world_T_entrance,
            world_T_plug=world_T_plug,
        )
        lateral_error = _lateral_error_to_axis(
            world_T_port_bottom=world_T_bottom,
            world_T_port_entrance=world_T_entrance,
            world_T_plug=world_T_plug,
        )
        wrench = force_torque_observation(
            self.model,
            data,
            force_sensor=self.wrench_cfg.force_sensor,
            torque_sensor=self.wrench_cfg.torque_sensor,
            force_bias=self.force_biases[env_id],
            torque_bias=self.torque_biases[env_id],
        )
        force_terms = zeroed_wrench_penalty(
            wrench.force,
            wrench.torque,
            force_limit=self.reward_force_limit,
        )
        contact = contact_observation(self.model, data)
        action_norm = float(np.linalg.norm(action))
        reward = compose_reward(
            [
                ("progress", progress.normalized_progress, self.reward_progress_weight),
                ("lateral", -lateral_error, self.reward_lateral_weight),
                ("force", force_terms.penalty, self.reward_force_weight),
                ("action", -(action_norm * action_norm), self.reward_action_weight),
                (
                    "penetration",
                    -contact.max_penetration,
                    self.reward_penetration_weight,
                ),
            ]
        )
        return EnvStepMetrics(
            reward=reward,
            progress=progress,
            wrench=wrench,
            force_terms=force_terms,
            contact=contact,
            lateral_error=lateral_error,
            action_norm=action_norm,
            ik_ok=True if step_result is None else step_result.ik_ok,
            ik_error=0.0 if step_result is None else step_result.ik_error,
            tcp_error=0.0 if step_result is None else step_result.ee_error,
        )

    def _is_done(self, env_id: int, metrics: EnvStepMetrics) -> bool:
        """Return whether an env should reset after the current step."""

        if self.episode_length_buf[env_id] >= self.max_episode_length:
            return True
        if metrics.force_terms.force_norm >= self.terminate_force_limit:
            return True
        if metrics.contact.max_penetration >= self.terminate_penetration_limit:
            return True
        if metrics.progress.normalized_progress >= self.success_progress:
            return True
        return False

    def _extras(self, rewards: np.ndarray, dones: np.ndarray) -> dict[str, Any]:
        """Aggregate scalar episode diagnostics for RSL-RL logging."""

        metrics = [m for m in self.last_metrics if m is not None]
        return {
            "episode": {
                "reward_mean": float(np.mean(rewards)),
                "done_count": int(np.count_nonzero(dones)),
                "progress_mean": float(
                    np.mean([m.progress.normalized_progress for m in metrics])
                ),
                "force_norm_max": float(
                    np.max([m.force_terms.force_norm for m in metrics])
                ),
                "max_penetration": float(
                    np.max([m.contact.max_penetration for m in metrics])
                ),
                "lateral_error_mean": float(
                    np.mean([m.lateral_error for m in metrics])
                ),
                "ik_error_mean": float(np.mean([m.ik_error for m in metrics])),
            }
        }
