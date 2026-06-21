"""Joint-space impedance controllers for MuJoCo prototype policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aic_mujoco.commands import JointControlMode, JointTarget
from aic_mujoco.joints import JointGroup


@dataclass
class JointImpedanceConfig:
    """Tunable limits/options for ``JointImpedanceController``.

    Args:
        use_bias_compensation: Add MuJoCo ``qfrc_bias`` to compensate gravity
            and velocity-dependent terms.
        clamp_to_joint_limits: Clamp position targets to XML joint limits.
        torque_limits: Optional per-joint absolute torque limits.
        torque_rate_limits: Optional per-joint torque slew limits.
        target_rate_limits: Optional per-joint target-position slew limits.
    """

    use_bias_compensation: bool = True
    clamp_to_joint_limits: bool = True
    torque_limits: np.ndarray | None = None
    torque_rate_limits: np.ndarray | None = None
    target_rate_limits: np.ndarray | None = None


class JointImpedanceController:
    """Generic joint target -> torque controller.

    Control law:

      tau = Kp * (q_des - q) + Kd * (qd_des - qd) + tau_ff + bias

    where `bias` is optional MuJoCo qfrc_bias compensation.

    This class is intentionally generic over any JointGroup. It does not know
    about UR5e, Hand-E, SFP, cables, ROS, or AIC-specific package paths.
    """

    def __init__(self, joint_group: JointGroup, config: JointImpedanceConfig):
        """Bind the controller to a MuJoCo joint group and config."""

        self.group = joint_group
        self.config = config
        self._internal_q_des: np.ndarray | None = None
        self._last_tau: np.ndarray | None = None
        self._lo, self._hi = self.group.joint_limits()

    def reset(self, q_current: Sequence[float]) -> None:
        """Reset internal integrated target and torque history."""

        q = np.asarray(q_current, dtype=float)
        if q.shape != (self.group.n,):
            raise ValueError(
                f"q_current must have shape ({self.group.n},), got {q.shape}"
            )
        self._internal_q_des = q.copy()
        self._last_tau = None

    def compute(self, data, target: JointTarget, dt: float) -> np.ndarray:
        """Convert a joint target into torques for the current MuJoCo state.

        Args:
            data: MuJoCo data containing current ``q/qdot/qfrc_bias``.
            target: Position or velocity-mode joint command.
            dt: Physics timestep used for velocity target integration and slew
                limits.
        """

        target.validate(self.group.n)

        q = self.group.q(data)
        qd = self.group.qd(data)

        if target.mode == JointControlMode.POSITION:
            assert target.q_des is not None
            q_des = target.q_des.copy()
            qd_des = target.qd_des.copy()
            self._internal_q_des = q_des.copy()

        elif target.mode == JointControlMode.VELOCITY:
            qd_des = target.qd_des.copy()
            if self._internal_q_des is None:
                self._internal_q_des = q.copy()
            q_des = self._internal_q_des + qd_des * dt
            self._internal_q_des = q_des.copy()

        else:
            raise ValueError(f"Unsupported mode: {target.mode}")

        if self.config.target_rate_limits is not None:
            limits = np.asarray(self.config.target_rate_limits, dtype=float)
            max_delta = limits * dt
            prev = q if self._internal_q_des is None else self._internal_q_des
            q_des = prev + np.clip(q_des - prev, -max_delta, max_delta)
            self._internal_q_des = q_des.copy()

        if self.config.clamp_to_joint_limits:
            q_des = np.minimum(np.maximum(q_des, self._lo), self._hi)

        tau = target.kp * (q_des - q) + target.kd * (qd_des - qd) + target.tau_ff

        if self.config.use_bias_compensation:
            tau = tau + self.group.bias_forces(data)

        if self.config.torque_limits is not None:
            limits = np.asarray(self.config.torque_limits, dtype=float)
            tau = np.clip(tau, -limits, limits)

        if self.config.torque_rate_limits is not None and self._last_tau is not None:
            rate_limits = np.asarray(self.config.torque_rate_limits, dtype=float)
            max_delta = rate_limits * dt
            tau = self._last_tau + np.clip(tau - self._last_tau, -max_delta, max_delta)

        self._last_tau = tau.copy()
        return tau
