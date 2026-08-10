"""Batched joint impedance and Cartesian motion controllers on Warp devices."""

from __future__ import annotations

from typing import Any

import warp as wp

from aic_mujoco.commands import (
    CartesianPoseCommand,
    HoldPositionCommand,
    JointDeltaAction,
)
from aic_mujoco.joints import ArmJoints
from aic_mujoco.utils.warp_math import (
    Matrix6,
    Vector6,
    damped_least_squares,
    rotation_error_world,
    scale_to_limit,
)


@wp.kernel
def target_pose_from_body(
    body_position: wp.array2d[wp.vec3],
    body_rotation: wp.array2d[wp.mat33],
    target_body_id: int,
    offset_position: wp.vec3,
    offset_rotation: wp.mat33,
    active: wp.array[bool],
    target_position: wp.array[wp.vec3],
    target_rotation: wp.array[wp.mat33],
):
    """Compose one target pose from a body pose and local matrix offset.

    Args:
        body_position: Batched MJWarp body positions.
        body_rotation: Batched MJWarp body rotation matrices.
        target_body_id: Body that defines the local goal frame.
        offset_position: Goal translation in the target body's local frame.
        offset_rotation: Goal rotation in the target body's local frame.
        active: Per-environment controller activation mask.
        target_position: Output world-frame goal positions.
        target_rotation: Output world-frame goal rotation matrices.
    """

    world = wp.tid()
    if not active[world]:
        return
    parent_rotation = body_rotation[world, target_body_id]
    target_position[world] = (
        body_position[world, target_body_id] + parent_rotation * offset_position
    )
    target_rotation[world] = parent_rotation * offset_rotation


@wp.kernel
def cartesian_delta_move(
    qpos: wp.array2d[float],
    body_position: wp.array2d[wp.vec3],
    body_rotation: wp.array2d[wp.mat33],
    joint_anchor: wp.array2d[wp.vec3],
    joint_axis: wp.array2d[wp.vec3],
    target_position: wp.array[wp.vec3],
    target_rotation: wp.array[wp.mat33],
    active: wp.array[bool],
    controlled_body_id: int,
    joint_ids: wp.array[int],
    qpos_addresses: wp.array[int],
    joint_lower: wp.array[float],
    joint_upper: wp.array[float],
    maximum_joint_step: wp.array[float],
    maximum_translation_step: float,
    maximum_rotation_step: float,
    translation_gain: float,
    rotation_gain: float,
    damping: float,
    joint_limit_margin: float,
    position_tolerance: float,
    orientation_tolerance: float,
    hold_position: wp.array2d[float],
    action_position: wp.array2d[float],
    current_position: wp.array[wp.vec3],
    current_rotation: wp.array[wp.mat33],
    position_error: wp.array[float],
    orientation_error: wp.array[float],
):
    """Convert Cartesian pose errors into bounded joint target increments.

    Args:
        qpos: Batched generalized positions.
        body_position: Batched world-frame body positions.
        body_rotation: Batched world-frame body rotation matrices.
        joint_anchor: Batched world-frame joint anchors.
        joint_axis: Batched world-frame joint axes.
        target_position: Per-environment Cartesian goal positions.
        target_rotation: Per-environment Cartesian goal rotations.
        active: Per-environment controller activation mask.
        controlled_body_id: Body moved by the six-joint arm.
        joint_ids: Ordered MuJoCo arm joint identifiers.
        qpos_addresses: Ordered generalized-position addresses.
        joint_lower: Lower joint limits.
        joint_upper: Upper joint limits.
        maximum_joint_step: Per-joint action limits.
        maximum_translation_step: Cartesian translation-step limit.
        maximum_rotation_step: Cartesian orientation-step limit.
        translation_gain: Proportional translation gain.
        rotation_gain: Proportional rotation gain.
        damping: Damped-least-squares coefficient.
        joint_limit_margin: Clearance maintained from every joint limit.
        position_tolerance: Successful translation-error threshold.
        orientation_tolerance: Successful orientation-error threshold.
        hold_position: Mutable low-level joint position command.
        action_position: Output labeled joint increments.
        current_position: Output current tool positions.
        current_rotation: Output current tool rotation matrices.
        position_error: Output translation-error magnitudes.
        orientation_error: Output orientation-error magnitudes.
    """

    world = wp.tid()
    if not active[world]:
        for joint in range(6):
            action_position[world, joint] = 0.0
        return

    tip_position = body_position[world, controlled_body_id]
    tip_rotation = body_rotation[world, controlled_body_id]
    linear_error = target_position[world] - tip_position
    angular_error = rotation_error_world(tip_rotation, target_rotation[world])
    linear_error_norm = wp.length(linear_error)
    angular_error_norm = wp.length(angular_error)

    current_position[world] = tip_position
    current_rotation[world] = tip_rotation
    position_error[world] = linear_error_norm
    orientation_error[world] = angular_error_norm

    if (
        linear_error_norm <= position_tolerance
        and angular_error_norm <= orientation_tolerance
    ):
        for joint in range(6):
            action_position[world, joint] = 0.0
            hold_position[world, joint] = qpos[world, qpos_addresses[joint]]
        return

    linear_step = scale_to_limit(
        linear_error * translation_gain, maximum_translation_step
    )
    angular_step = scale_to_limit(
        angular_error * rotation_gain, maximum_rotation_step
    )
    cartesian_step = Vector6(
        linear_step[0],
        linear_step[1],
        linear_step[2],
        angular_step[0],
        angular_step[1],
        angular_step[2],
    )

    jacobian = Matrix6(0.0)
    for joint in range(6):
        joint_id = joint_ids[joint]
        axis = joint_axis[world, joint_id]
        linear = wp.cross(axis, tip_position - joint_anchor[world, joint_id])
        jacobian[0, joint] = linear[0]
        jacobian[1, joint] = linear[1]
        jacobian[2, joint] = linear[2]
        jacobian[3, joint] = axis[0]
        jacobian[4, joint] = axis[1]
        jacobian[5, joint] = axis[2]

    joint_step = damped_least_squares(jacobian, cartesian_step, damping)
    for joint in range(6):
        q_index = qpos_addresses[joint]
        bounded_step = wp.clamp(
            joint_step[joint],
            -maximum_joint_step[joint],
            maximum_joint_step[joint],
        )
        lower = joint_lower[joint] + joint_limit_margin
        upper = joint_upper[joint] - joint_limit_margin
        target = wp.clamp(qpos[world, q_index] + bounded_step, lower, upper)
        action_position[world, joint] = target - qpos[world, q_index]
        hold_position[world, joint] = target


@wp.kernel
def hold_impedance(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    qfrc_bias: wp.array2d[float],
    target_position: wp.array2d[float],
    qpos_addresses: wp.array[int],
    dof_addresses: wp.array[int],
    actuator_addresses: wp.array[int],
    stiffness: wp.array[float],
    damping: wp.array[float],
    torque_limits: wp.array[float],
    ctrl: wp.array2d[float],
):
    """Write clipped impedance torques for every world and arm joint.

    Args:
        qpos: Batched generalized positions.
        qvel: Batched generalized velocities.
        qfrc_bias: Batched MuJoCo dynamics-bias forces.
        target_position: Batched six-joint HOLD targets.
        qpos_addresses: Ordered generalized-position addresses.
        dof_addresses: Ordered generalized-velocity addresses.
        actuator_addresses: Ordered actuator control addresses.
        stiffness: Per-joint proportional gains.
        damping: Per-joint derivative gains.
        torque_limits: Per-joint torque limits.
        ctrl: Batched actuator control output.
    """

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
        """Initialize immutable device parameters for joint HOLD control.

        Args:
            config: Strict merged runtime configuration.
            joints: Resolved six-joint model mapping.
            num_envs: Number of parallel MJWarp environments.
            device: Warp device that owns controller arrays.
        """

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


class CartesianMoveController:
    """Convert a Cartesian pose error into bounded six-joint target increments.

    This controller is a privileged demonstration teacher. It reads exact
    MJWarp body poses and joint kinematics, computes one damped-least-squares
    differential IK step, and updates the existing joint HOLD command. The
    low-level impedance controller remains responsible for producing torque.
    """

    def __init__(
        self,
        config: dict[str, Any],
        joints: ArmJoints,
        model: Any,
        controlled_body_id: int,
        target_body_id: int,
        num_envs: int,
        device: Any,
    ):
        """Initialize the privileged Cartesian demonstration teacher.

        Args:
            config: Strict merged collection configuration.
            joints: Resolved six-joint model mapping.
            model: Compiled host MuJoCo model used for topology validation.
            controlled_body_id: Body moved by the Cartesian controller.
            target_body_id: Body that defines the goal frame.
            num_envs: Number of parallel MJWarp environments.
            device: Warp device that owns controller arrays.
        """

        expert = config["expert"]
        self.num_envs = num_envs
        self.device = device
        self.controlled_body_id = controlled_body_id
        self.target_body_id = target_body_id
        self.joint_ids = wp.array(joints.joint_ids, dtype=int, device=device)
        self.qpos_addresses = wp.array(
            joints.qpos_addresses, dtype=int, device=device
        )
        self.joint_lower = wp.array(joints.ranges[:, 0], dtype=float, device=device)
        self.joint_upper = wp.array(joints.ranges[:, 1], dtype=float, device=device)
        self.maximum_joint_step = wp.array(
            expert["maximum_joint_step"], dtype=float, device=device
        )
        self.offset_position = wp.vec3(expert["goal_offset_position"])
        self.offset_rotation = wp.mat33(
            *expert["goal_offset_rotation_matrix"]
        )
        self.maximum_translation_step = expert["maximum_translation_step"]
        self.maximum_rotation_step = expert["maximum_rotation_step"]
        self.translation_gain = expert["translation_gain"]
        self.rotation_gain = expert["rotation_gain"]
        self.damping = expert["dls_damping"]
        self.joint_limit_margin = expert["joint_limit_margin"]
        self.position_tolerance = expert["position_tolerance"]
        self.orientation_tolerance = expert["orientation_tolerance"]
        self.validate_kinematic_chain(model, joints)

        self.command = CartesianPoseCommand(num_envs, device)
        self.action = JointDeltaAction(num_envs, joints.count, device)
        self.current_position = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.current_rotation = wp.zeros(
            num_envs, dtype=wp.mat33, device=device
        )
        self.position_error = wp.zeros(num_envs, dtype=float, device=device)
        self.orientation_error = wp.zeros(num_envs, dtype=float, device=device)

    def validate_kinematic_chain(self, model: Any, joints: ArmJoints) -> None:
        """Fail if the configured arm does not drive the controlled body."""

        ancestor_bodies: set[int] = set()
        body_id = self.controlled_body_id
        while body_id > 0:
            ancestor_bodies.add(body_id)
            body_id = int(model.body_parentid[body_id])
        joint_bodies = {int(model.jnt_bodyid[joint_id]) for joint_id in joints.joint_ids}
        if not joint_bodies.issubset(ancestor_bodies):
            raise ValueError(
                "Every configured arm joint must be an ancestor of the controlled body"
            )
        margin = self.joint_limit_margin
        for lower, upper in joints.ranges:
            if float(lower) + margin >= float(upper) - margin:
                raise ValueError("expert.joint_limit_margin leaves an empty joint range")

    def set_active(self, active: list[bool]) -> None:
        """Select environments controlled by the Cartesian teacher."""

        if len(active) != self.num_envs:
            raise ValueError("Cartesian controller active mask has the wrong length")
        wp.copy(
            self.command.active,
            wp.array(active, dtype=bool, device=self.device),
        )

    def update_goal_from_target_body(self, data: Any) -> None:
        """Compose each active goal from the configured target body and offset."""

        wp.launch(
            target_pose_from_body,
            dim=self.num_envs,
            inputs=[
                data.xpos,
                data.xmat,
                self.target_body_id,
                self.offset_position,
                self.offset_rotation,
                self.command.active,
            ],
            outputs=[
                self.command.position,
                self.command.rotation,
            ],
            device=self.device,
        )

    def move(self, data: Any, hold_command: HoldPositionCommand) -> JointDeltaAction:
        """Apply one closed-loop Cartesian step and return the labeled action."""

        wp.launch(
            cartesian_delta_move,
            dim=self.num_envs,
            inputs=[
                data.qpos,
                data.xpos,
                data.xmat,
                data.xanchor,
                data.xaxis,
                self.command.position,
                self.command.rotation,
                self.command.active,
                self.controlled_body_id,
                self.joint_ids,
                self.qpos_addresses,
                self.joint_lower,
                self.joint_upper,
                self.maximum_joint_step,
                self.maximum_translation_step,
                self.maximum_rotation_step,
                self.translation_gain,
                self.rotation_gain,
                self.damping,
                self.joint_limit_margin,
                self.position_tolerance,
                self.orientation_tolerance,
            ],
            outputs=[
                hold_command.position,
                self.action.position,
                self.current_position,
                self.current_rotation,
                self.position_error,
                self.orientation_error,
            ],
            device=self.device,
        )
        return self.action
