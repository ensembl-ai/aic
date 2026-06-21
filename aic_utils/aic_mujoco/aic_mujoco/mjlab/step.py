"""Prototype stepping utilities for AIC MuJoCo policy-training experiments."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aic_mujoco.commands import JointTarget
from aic_mujoco.controllers import JointImpedanceController
from aic_mujoco.joints import JointGroup, PassiveJointGroup


@dataclass(frozen=True)
class CartesianStepConfig:
    """Numerical settings for translation-only Cartesian IK."""

    ik_iters: int = 20
    ik_tolerance: float = 0.00025
    ik_damping: float = 0.03
    ik_max_dq: float = 0.03


@dataclass
class CartesianStepResult:
    """Diagnostics returned after one Cartesian target step."""

    q_des: np.ndarray
    tau: np.ndarray
    ee_pos: np.ndarray
    ee_error: float
    ik_error: float
    ik_ok: bool


def make_downward_path(distance: float, step: float) -> np.ndarray:
    """Return world-frame offsets for a simple policy moving straight down."""

    distance = max(0.0, float(distance))
    step = max(np.finfo(float).eps, float(step))
    n = max(1, int(np.ceil(distance / step)))
    offsets = np.zeros((n + 1, 3), dtype=float)
    offsets[:, 2] = -np.linspace(0.0, distance, n + 1)
    return offsets


def solve_position_ik(
    model: mujoco.MjModel,
    seed_data: mujoco.MjData,
    controlled: JointGroup,
    site_id: int,
    target_pos: np.ndarray,
    q_seed: np.ndarray,
    cfg: CartesianStepConfig,
) -> tuple[np.ndarray, float, bool]:
    """Translation-only damped least-squares IK for the controlled joint group."""

    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = seed_data.qpos
    ik_data.qvel[:] = 0.0

    q = np.asarray(q_seed, dtype=float).copy()
    lo, hi = controlled.joint_limits()

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    cols = np.array([h.qvel_addr for h in controlled.handles], dtype=int)

    last_err_norm = np.inf

    for _ in range(cfg.ik_iters):
        controlled.set_q(ik_data, q.tolist(), zero_velocity=True)
        mujoco.mj_forward(model, ik_data)

        current_pos = ik_data.site_xpos[site_id].copy()
        err = np.asarray(target_pos, dtype=float) - current_pos
        err_norm = float(np.linalg.norm(err))
        last_err_norm = err_norm

        if err_norm <= cfg.ik_tolerance:
            return q, err_norm, True

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = jacp[:, cols]
        A = J @ J.T + (cfg.ik_damping * cfg.ik_damping) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > cfg.ik_max_dq:
            dq *= cfg.ik_max_dq / dq_norm

        q = q + dq
        q = np.minimum(np.maximum(q, lo), hi)

    return q, last_err_norm, False


def step_cartesian_position_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: JointGroup,
    passive: PassiveJointGroup,
    controller: JointImpedanceController,
    site_id: int,
    target_pos: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    cfg: CartesianStepConfig,
) -> CartesianStepResult:
    """Apply one Cartesian-position policy step through IK and joint impedance."""

    dt = float(model.opt.timestep)
    data.qfrc_applied[:] = 0.0
    passive.enforce(data)

    q_des, ik_error, ik_ok = solve_position_ik(
        model=model,
        seed_data=data,
        controlled=controlled,
        site_id=site_id,
        target_pos=target_pos,
        q_seed=controlled.q(data),
        cfg=cfg,
    )

    joint_target = JointTarget.position(
        q_des=q_des.tolist(),
        qd_des=np.zeros(controlled.n).tolist(),
        kp=np.asarray(kp, dtype=float).tolist(),
        kd=np.asarray(kd, dtype=float).tolist(),
        tau_ff=np.zeros(controlled.n).tolist(),
    )

    tau = controller.compute(data, joint_target, dt)
    controlled.apply_torque(data, tau.tolist())
    mujoco.mj_step(model, data)
    passive.enforce(data)
    mujoco.mj_forward(model, data)

    ee_pos = data.site_xpos[site_id].copy()
    ee_error = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - ee_pos))
    return CartesianStepResult(
        q_des=q_des,
        tau=tau,
        ee_pos=ee_pos,
        ee_error=ee_error,
        ik_error=ik_error,
        ik_ok=ik_ok,
    )
