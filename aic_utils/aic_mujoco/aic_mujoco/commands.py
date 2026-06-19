from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class JointControlMode(str, Enum):
    POSITION = "position"
    VELOCITY = "velocity"


@dataclass(frozen=True)
class JointTarget:
    """Generic joint target command.

    This intentionally mirrors the command semantics of AIC's JointMotionUpdate
    without importing ROS message types.

    POSITION mode:
      q_des is the desired joint position vector.
      qd_des is the desired joint velocity vector, usually zeros.

    VELOCITY mode:
      qd_des is desired joint velocity.
      q_des can be None; the controller integrates qd_des into an internal
      position target.
    """

    mode: JointControlMode
    q_des: np.ndarray | None
    qd_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray

    @staticmethod
    def position(
        q_des: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        qd_des: Sequence[float] | None = None,
        tau_ff: Sequence[float] | None = None,
    ) -> "JointTarget":
        q = np.asarray(q_des, dtype=float)
        n = q.shape[0]
        return JointTarget(
            mode=JointControlMode.POSITION,
            q_des=q,
            qd_des=np.zeros(n) if qd_des is None else np.asarray(qd_des, dtype=float),
            kp=np.asarray(kp, dtype=float),
            kd=np.asarray(kd, dtype=float),
            tau_ff=np.zeros(n) if tau_ff is None else np.asarray(tau_ff, dtype=float),
        )

    @staticmethod
    def velocity(
        qd_des: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        tau_ff: Sequence[float] | None = None,
    ) -> "JointTarget":
        qd = np.asarray(qd_des, dtype=float)
        n = qd.shape[0]
        return JointTarget(
            mode=JointControlMode.VELOCITY,
            q_des=None,
            qd_des=qd,
            kp=np.asarray(kp, dtype=float),
            kd=np.asarray(kd, dtype=float),
            tau_ff=np.zeros(n) if tau_ff is None else np.asarray(tau_ff, dtype=float),
        )

    def validate(self, n: int) -> None:
        fields = {
            "qd_des": self.qd_des,
            "kp": self.kp,
            "kd": self.kd,
            "tau_ff": self.tau_ff,
        }
        if self.q_des is not None:
            fields["q_des"] = self.q_des

        for name, value in fields.items():
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
