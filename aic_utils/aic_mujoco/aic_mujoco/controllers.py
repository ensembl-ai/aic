"""Batched joint HOLD controller implemented entirely as a Warp kernel."""

from __future__ import annotations

from typing import Any

import warp as wp

from aic_mujoco.commands import HoldPositionCommand
from aic_mujoco.joints import ArmJoints


@wp.kernel
def hold_impedance(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qfrc_bias: wp.array2d(dtype=float),
    target_position: wp.array2d(dtype=float),
    qpos_addresses: wp.array(dtype=int),
    dof_addresses: wp.array(dtype=int),
    actuator_addresses: wp.array(dtype=int),
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    torque_limits: wp.array(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    world, joint = wp.tid()
    q_index = qpos_addresses[joint]
    v_index = dof_addresses[joint]
    torque = (
        stiffness[joint] * (target_position[world, joint] - qpos[world, q_index])
        - damping[joint] * qvel[world, v_index]
        + qfrc_bias[world, v_index]
    )
    limit = torque_limits[joint]
    ctrl[world, actuator_addresses[joint]] = wp.clamp(torque, -limit, limit)


class JointHoldController:
    """Own controller parameters and write six actuator torques per world."""

    def __init__(
        self,
        config: dict[str, Any],
        joints: ArmJoints,
        num_envs: int,
        device: Any,
    ):
        control = config["control"]
        self.num_envs = num_envs
        self.joint_count = joints.count
        self.device = device
        self.qpos_addresses = wp.array(
            joints.qpos_addresses, dtype=int, device=device
        )
        self.dof_addresses = wp.array(
            joints.dof_addresses, dtype=int, device=device
        )
        self.actuator_addresses = wp.array(
            joints.actuator_addresses, dtype=int, device=device
        )
        self.stiffness = wp.array(control["stiffness"], dtype=float, device=device)
        self.damping = wp.array(control["damping"], dtype=float, device=device)
        self.torque_limits = wp.array(
            control["torque_limits"], dtype=float, device=device
        )

    def apply(self, data: Any, command: HoldPositionCommand) -> None:
        """Evaluate the configured impedance law on all environments."""

        wp.launch(
            hold_impedance,
            dim=(self.num_envs, self.joint_count),
            inputs=[
                data.qpos,
                data.qvel,
                data.qfrc_bias,
                command.position,
                self.qpos_addresses,
                self.dof_addresses,
                self.actuator_addresses,
                self.stiffness,
                self.damping,
                self.torque_limits,
            ],
            outputs=[data.ctrl],
            device=self.device,
        )
