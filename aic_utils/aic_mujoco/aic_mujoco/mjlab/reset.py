"""Prototype reset utilities for AIC MuJoCo policy-training experiments.

These functions are intentionally plain MuJoCo/Python utilities, not final
MJLab manager terms. They capture the reset behavior we want before wiring it
into a full MJLab environment:

  reset MuJoCo state
  place the robot/payload at the pre-insertion pose
  snapshot passive joints
  hold the reset pose for settling
  estimate force/torque bias from raw sensor samples
  start the episode with zeroed wrench observations
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aic_mujoco.commands import JointTarget
from aic_mujoco.controllers import JointImpedanceController
from aic_mujoco.joints import JointGroup, PassiveJointGroup
from aic_mujoco.utils import compute_preinsert_joint_target
from aic_mujoco.mjlab.observations import read_sensor


@dataclass(frozen=True)
class PreinsertResetConfig:
    """Frame names and home pose for SFP-tip pre-insertion reset."""

    home_q: tuple[float, ...] = (
        -0.1597,
        -1.3542,
        -1.6648,
        -1.6933,
        1.5710,
        1.4110,
    )
    port_body: str = "sfp_port_0_link_entrance"
    tcp_site: str = "gripper_tcp"
    sfp_tip_body: str = "sfp_tip_link"
    tool_body: str = "ati/tool_link"
    weld_child_body: str = "lc_plug_link"
    payload_root_body: str = "cable_end_0"
    payload_root_freejoint: str = "cable_end_0_free"
    height: float = 0.05


@dataclass(frozen=True)
class WrenchZeroingConfig:
    """Settings for reset-time force/torque bias estimation."""

    force_sensor: str = "AtiForceTorqueSensor_force"
    torque_sensor: str = "AtiForceTorqueSensor_torque"
    settle_steps: int = 100
    bias_samples: int = 50
    enabled: bool = True


@dataclass
class ResetResult:
    """State returned by a pre-insertion reset."""

    q_start: np.ndarray
    preinsert_diagnostics: dict[str, np.ndarray | str | float | int | None]
    force_bias: np.ndarray | None
    torque_bias: np.ndarray | None


def hold_joint_pose_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: JointGroup,
    passive: PassiveJointGroup,
    controller: JointImpedanceController,
    q_hold: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Step physics once while holding a joint pose with impedance control."""

    dt = float(model.opt.timestep)
    data.qfrc_applied[:] = 0.0
    passive.enforce(data)

    target = JointTarget.position(
        q_des=np.asarray(q_hold, dtype=float).tolist(),
        qd_des=np.zeros(controlled.n).tolist(),
        kp=np.asarray(kp, dtype=float).tolist(),
        kd=np.asarray(kd, dtype=float).tolist(),
        tau_ff=np.zeros(controlled.n).tolist(),
    )
    tau = controller.compute(data, target, dt)
    controlled.apply_torque(data, tau.tolist())

    mujoco.mj_step(model, data)
    passive.enforce(data)
    mujoco.mj_forward(model, data)
    return tau


def zero_force_torque_after_settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: JointGroup,
    passive: PassiveJointGroup,
    controller: JointImpedanceController,
    q_hold: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    cfg: WrenchZeroingConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Hold the reset pose, then average raw F/T samples into a bias."""

    if not cfg.enabled:
        return None, None

    controller.reset(np.asarray(q_hold, dtype=float).tolist())
    for _ in range(max(0, cfg.settle_steps)):
        hold_joint_pose_step(
            model, data, controlled, passive, controller, q_hold, kp, kd
        )

    force_values: list[np.ndarray] = []
    torque_values: list[np.ndarray] = []
    for _ in range(max(1, cfg.bias_samples)):
        hold_joint_pose_step(
            model, data, controlled, passive, controller, q_hold, kp, kd
        )
        force = read_sensor(model, data, cfg.force_sensor)
        torque = read_sensor(model, data, cfg.torque_sensor)
        if force is not None:
            force_values.append(force)
        if torque is not None:
            torque_values.append(torque)

    force_bias = None if not force_values else np.mean(np.asarray(force_values), axis=0)
    torque_bias = (
        None if not torque_values else np.mean(np.asarray(torque_values), axis=0)
    )
    passive.enforce(data)
    mujoco.mj_forward(model, data)
    return force_bias, torque_bias


def reset_preinsert_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: JointGroup,
    passive: PassiveJointGroup,
    controller: JointImpedanceController,
    kp: np.ndarray,
    kd: np.ndarray,
    reset_cfg: PreinsertResetConfig,
    wrench_cfg: WrenchZeroingConfig,
) -> ResetResult:
    """Reset into the SFP-tip pre-insertion pose and estimate F/T bias."""

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_start, diagnostics = compute_preinsert_joint_target(
        model=model,
        data=data,
        controlled=controlled,
        home_q=np.asarray(reset_cfg.home_q, dtype=float),
        port_body=reset_cfg.port_body,
        tcp_site=reset_cfg.tcp_site,
        sfp_tip_body=reset_cfg.sfp_tip_body,
        tool_body=reset_cfg.tool_body,
        weld_child_body=reset_cfg.weld_child_body,
        height=reset_cfg.height,
        payload_root_body=reset_cfg.payload_root_body,
        payload_root_freejoint=reset_cfg.payload_root_freejoint,
    )

    passive.snapshot(data)
    controller.reset(q_start.tolist())
    force_bias, torque_bias = zero_force_torque_after_settle(
        model=model,
        data=data,
        controlled=controlled,
        passive=passive,
        controller=controller,
        q_hold=q_start,
        kp=kp,
        kd=kd,
        cfg=wrench_cfg,
    )
    controller.reset(controlled.q(data).tolist())

    return ResetResult(
        q_start=q_start,
        preinsert_diagnostics=diagnostics,
        force_bias=force_bias,
        torque_bias=torque_bias,
    )
