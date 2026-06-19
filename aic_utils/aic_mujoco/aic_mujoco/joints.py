from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import mujoco
import numpy as np


class TorqueMode(str, Enum):
    """How joint torques are applied in MuJoCo."""

    ACTUATOR_CTRL = "actuator_ctrl"
    QFRC_APPLIED = "qfrc_applied"


@dataclass(frozen=True)
class JointHandle:
    name: str
    joint_id: int
    qpos_addr: int
    qvel_addr: int
    actuator_id: int | None
    actuator_gear: float


class JointGroup:
    """Generic named joint group.

    Use this for the controlled arm, a gripper, a neck, a mobile base, etc.
    It is not UR5e-specific and not gripper-specific.

    If `torque_mode == actuator_ctrl`, each joint must have a direct MuJoCo
    joint actuator. If not, use qfrc_applied for debugging or add actuators
    to the XML.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: Sequence[str],
        torque_mode: TorqueMode = TorqueMode.ACTUATOR_CTRL,
        allow_missing_actuators: bool = False,
    ):
        self.model = model
        self.joint_names = list(joint_names)
        self.torque_mode = TorqueMode(torque_mode)
        self.handles = [self._make_handle(name) for name in self.joint_names]

        if self.torque_mode == TorqueMode.ACTUATOR_CTRL and not allow_missing_actuators:
            missing = [h.name for h in self.handles if h.actuator_id is None]
            if missing:
                raise RuntimeError(
                    "No direct MuJoCo actuator found for controlled joints: "
                    + ", ".join(missing)
                    + ". Use --torque-mode qfrc_applied for debugging or add actuators."
                )

    @property
    def n(self) -> int:
        return len(self.handles)

    def _make_handle(self, joint_name: str) -> JointHandle:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {joint_name!r}")

        jtype = self.model.jnt_type[jid]
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise RuntimeError(
                f"Joint {joint_name!r} must be hinge or slide for this controller; "
                f"got MuJoCo type {int(jtype)}."
            )

        actuator_id, gear = self._find_direct_joint_actuator(jid)
        return JointHandle(
            name=joint_name,
            joint_id=int(jid),
            qpos_addr=int(self.model.jnt_qposadr[jid]),
            qvel_addr=int(self.model.jnt_dofadr[jid]),
            actuator_id=actuator_id,
            actuator_gear=gear,
        )

    def _find_direct_joint_actuator(self, joint_id: int) -> tuple[int | None, float]:
        for aid in range(self.model.nu):
            if self.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            if int(self.model.actuator_trnid[aid, 0]) != int(joint_id):
                continue

            gear = float(self.model.actuator_gear[aid, 0])
            if abs(gear) < 1e-12:
                gear = 1.0
            return int(aid), gear

        return None, 1.0

    def q(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.qpos[h.qpos_addr] for h in self.handles], dtype=float)

    def qd(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.qvel[h.qvel_addr] for h in self.handles], dtype=float)

    def set_q(self, data: mujoco.MjData, q: Sequence[float], zero_velocity: bool = True) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n,):
            raise ValueError(f"q must have shape ({self.n},), got {q.shape}")

        for value, h in zip(q, self.handles):
            data.qpos[h.qpos_addr] = float(value)
            if zero_velocity:
                data.qvel[h.qvel_addr] = 0.0

        mujoco.mj_forward(self.model, data)

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(self.n, dtype=float)
        hi = np.empty(self.n, dtype=float)
        for i, h in enumerate(self.handles):
            if self.model.jnt_limited[h.joint_id]:
                lo[i] = self.model.jnt_range[h.joint_id, 0]
                hi[i] = self.model.jnt_range[h.joint_id, 1]
            else:
                lo[i] = -np.inf
                hi[i] = np.inf
        return lo, hi

    def bias_forces(self, data: mujoco.MjData) -> np.ndarray:
        """Return MuJoCo generalized bias forces for this group.

        `qfrc_bias` contains gravity/Coriolis/centrifugal terms. Adding this
        term is the MuJoCo-side analog of the gravity/bias compensation stage
        in an impedance controller.
        """

        return np.asarray([data.qfrc_bias[h.qvel_addr] for h in self.handles], dtype=float)

    def apply_torque(self, data: mujoco.MjData, tau: Sequence[float]) -> None:
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.n,):
            raise ValueError(f"tau must have shape ({self.n},), got {tau.shape}")

        if self.torque_mode == TorqueMode.ACTUATOR_CTRL:
            for h, effort in zip(self.handles, tau):
                if h.actuator_id is None:
                    continue
                data.ctrl[h.actuator_id] = float(effort) / h.actuator_gear
        else:
            for h, effort in zip(self.handles, tau):
                data.qfrc_applied[h.qvel_addr] = float(effort)

    def zero_owned_actuators(self, data: mujoco.MjData) -> None:
        for h in self.handles:
            if h.actuator_id is not None:
                data.ctrl[h.actuator_id] = 0.0

    def print_mapping(self, title: str = "JointGroup") -> None:
        print(f"{title}:")
        for h in self.handles:
            actuator_name = "-"
            if h.actuator_id is not None:
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, h.actuator_id) or str(h.actuator_id)
            print(
                f"  {h.name:35s} qpos={h.qpos_addr:2d} qvel={h.qvel_addr:2d} "
                f"actuator={actuator_name} gear={h.actuator_gear:g}"
            )


class PassiveJointGroup:
    """Generic passive joint manager.

    Use this for joints you do NOT want controlled by the policy/controller.

    mode='free':
      Do nothing. Joint evolves passively under physics.

    mode='freeze':
      Snapshot q/qdot and re-enforce the same values every step.
      This is a kinematic lock, useful when your XML uses a welded payload and
      you do not want the gripper to move or participate as an active system.

    This is generic; it is not gripper-specific.
    """

    def __init__(self, model: mujoco.MjModel, joint_names: Sequence[str], mode: str = "free"):
        self.model = model
        self.joint_names = list(joint_names)
        self.mode = mode
        if self.mode not in ("free", "freeze"):
            raise ValueError("PassiveJointGroup mode must be 'free' or 'freeze'.")

        self.handles = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Passive joint not found: {name!r}")
            self.handles.append(
                JointHandle(
                    name=name,
                    joint_id=int(jid),
                    qpos_addr=int(model.jnt_qposadr[jid]),
                    qvel_addr=int(model.jnt_dofadr[jid]),
                    actuator_id=None,
                    actuator_gear=1.0,
                )
            )

        self._q_frozen: np.ndarray | None = None

    def snapshot(self, data: mujoco.MjData) -> None:
        if not self.handles:
            self._q_frozen = np.zeros(0)
            return
        self._q_frozen = np.asarray([data.qpos[h.qpos_addr] for h in self.handles], dtype=float)

    def enforce(self, data: mujoco.MjData) -> None:
        if self.mode != "freeze" or self._q_frozen is None:
            return
        for q, h in zip(self._q_frozen, self.handles):
            data.qpos[h.qpos_addr] = float(q)
            data.qvel[h.qvel_addr] = 0.0

    def print_mapping(self, title: str = "PassiveJointGroup") -> None:
        print(f"{title} mode={self.mode}:")
        if not self.handles:
            print("  <none>")
            return
        for h in self.handles:
            print(f"  {h.name:35s} qpos={h.qpos_addr:2d} qvel={h.qvel_addr:2d}")


def print_model_summary(model: mujoco.MjModel) -> None:
    print("\nJoints:")
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        print(
            f"  {jid:3d} {name:45s} "
            f"type={int(model.jnt_type[jid])} "
            f"qpos={int(model.jnt_qposadr[jid])} dof={int(model.jnt_dofadr[jid])}"
        )

    print("\nActuators:")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        trntype = int(model.actuator_trntype[aid])
        trnid = tuple(int(x) for x in model.actuator_trnid[aid])
        gear0 = float(model.actuator_gear[aid, 0])
        print(f"  {aid:3d} {name:45s} trntype={trntype} trnid={trnid} gear0={gear0:g}")

    print("\nEquality constraints:")
    for eid in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eid)
        print(
            f"  {eid:3d} {name or '<unnamed>':45s} "
            f"type={int(model.eq_type[eid])} obj1={int(model.eq_obj1id[eid])} obj2={int(model.eq_obj2id[eid])}"
        )
    print()
