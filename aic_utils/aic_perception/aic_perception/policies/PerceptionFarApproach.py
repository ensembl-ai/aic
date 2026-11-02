"""Far approach to a perceived SFP port, loadable by ``aic_model``.

The target port pose comes from the multiview scene estimator
(``SceneEstimate`` published by ``scripts/run_inference.py``). The gripper is
driven so that the plug tip ends ``hover_height_m`` above the perceived port
entrance, with the plug orientation aligned to the port by the orientation
code of AIC's ``CheatCode`` example. The policy stops there without inserting.

Placing the plug tip uses the plug's ground-truth pose in the gripper, so the
simulation must run with ``ground_truth:=true``. The same ground truth is
used to log how far the perceived port and the final hover pose are from the
true port entrance.

SC tasks are rejected: the engine names SC ports by spawn slot while the
estimator names them by rail order, and a task message alone cannot map one
onto the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_perception_interfaces.msg import SceneEstimate
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Transform, Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from tf2_ros import TransformException

from aic_perception.config import load_config

CONTROL_PERIOD_S = 0.05


def stamp_seconds(stamp: Any) -> float:
    """Convert a ``builtin_interfaces/Time`` to seconds."""

    return stamp.sec + stamp.nanosec * 1e-9


class PerceptionFarApproach(CheatCode):
    """Hover above an SFP port whose pose comes from the scene estimator."""

    def __init__(self, parent_node: Any):
        """Load the far-approach settings and subscribe to scene estimates.

        Args:
            parent_node: The ``aic_model`` node hosting the policy.
        """

        super().__init__(parent_node)
        config = load_config("infer")
        self.settings = config["far_approach"]
        self.estimates: list[SceneEstimate] = []
        parent_node.create_subscription(
            SceneEstimate,
            config["inference"]["estimate_topic"],
            self.estimates.append,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

    def perceived_port(self, target_id: str) -> Transform | None:
        """Wait for fresh estimates of one target and combine them.

        Only estimates computed after the task arrived are used, so the robot
        is at its start pose and the board of this trial is in view. The
        position is the per-axis median of ``estimate_count`` present
        estimates; the orientation is taken from the newest one.

        Args:
            target_id: Schema target ID, e.g. ``nic_card_mount_2/sfp_port_0``.

        Returns:
            Port entrance transform in ``base_link``, or ``None`` on timeout.
        """

        started = stamp_seconds(self.time_now().to_msg())
        self.estimates.clear()
        poses = []
        waited = 0.0
        while len(poses) < self.settings["estimate_count"]:
            if waited > self.settings["estimate_timeout_s"]:
                return None
            while self.estimates:
                message = self.estimates.pop(0)
                if stamp_seconds(message.header.stamp) < started:
                    continue
                for target in message.targets:
                    if target.id == target_id and target.present:
                        poses.append(target)
                        self.get_logger().info(
                            f"Estimate {len(poses)}: presence {target.presence_probability:.3f}, "
                            f"confidence {target.confidence:.3f}"
                        )
            self.sleep_for(0.1)
            waited += 0.1
        positions = np.array(
            [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in poses]
        )
        median = np.median(positions, axis=0)
        return Transform(
            translation=Vector3(
                x=float(median[0]), y=float(median[1]), z=float(median[2])
            ),
            rotation=poses[-1].pose.orientation,
        )

    def true_position(self, frame: str) -> np.ndarray | None:
        """Ground-truth position of a frame in ``base_link``, if published."""

        try:
            translation = self._parent_node._tf_buffer.lookup_transform(
                "base_link", frame, Time()
            ).transform.translation
        except TransformException:
            return None
        return np.array([translation.x, translation.y, translation.z])

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Perform the far approach for one task.

        Args:
            task: AIC task naming the cable, plug, module, and port.
            get_observation: Unused; perception arrives as ``SceneEstimate``.
            move_robot: Callback that sends motion targets to the controller.
            send_feedback: Unused.

        Returns:
            Whether the hover pose was reached.
        """

        self.get_logger().info(f"PerceptionFarApproach task: {task}")
        if task.port_type != "sfp":
            self.get_logger().error(
                f"Port type '{task.port_type}' is not supported: SC port names in the "
                "task cannot be mapped onto the estimator's rail-order names"
            )
            return False
        self._task = task
        target_id = f"{task.target_module_name}/{task.port_name}"
        port = self.perceived_port(target_id)
        if port is None:
            self.get_logger().error(
                f"No present estimate of {target_id}; is run_inference.py running?"
            )
            return False
        perceived = np.array(
            [port.translation.x, port.translation.y, port.translation.z]
        )
        entrance_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        )
        truth = self.true_position(entrance_frame)
        if truth is not None:
            error = (perceived - truth) * 1000.0
            self.get_logger().info(
                f"Perceived {target_id} entrance error: {np.linalg.norm(error):.2f} mm "
                f"(x {error[0]:.2f}, y {error[1]:.2f}, z {error[2]:.2f})"
            )

        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        if not self._wait_for_tf("base_link", plug_frame):
            return False
        hover = self.settings["hover_height_m"]
        plug_goal = perceived + np.array([0.0, 0.0, hover])
        start = self.true_position("gripper/tcp")
        steps = max(1, round(self.settings["approach_duration_s"] / CONTROL_PERIOD_S))
        for step in range(1, steps + 1):
            fraction = step / steps
            gripper = self.true_position("gripper/tcp")
            plug = self.true_position(plug_frame)
            if start is None or gripper is None or plug is None:
                self.get_logger().warn(
                    "Gripper or plug pose unavailable during approach"
                )
                self.sleep_for(CONTROL_PERIOD_S)
                continue
            # Orientation alignment comes from CheatCode; the position puts the
            # plug tip, not the TCP, at the hover point above the entrance.
            try:
                pose = self.calc_gripper_pose(
                    port,
                    slerp_fraction=fraction,
                    position_fraction=fraction,
                    z_offset=hover,
                    reset_xy_integrator=True,
                )
            except TransformException as error:
                self.get_logger().warn(f"TF lookup failed during approach: {error}")
                self.sleep_for(CONTROL_PERIOD_S)
                continue
            target = fraction * (plug_goal + gripper - plug) + (1.0 - fraction) * start
            pose.position.x, pose.position.y, pose.position.z = map(float, target)
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(CONTROL_PERIOD_S)
        self.sleep_for(self.settings["settle_time_s"])

        plug = self.true_position(plug_frame)
        if plug is not None and truth is not None:
            offset = (plug - (truth + np.array([0.0, 0.0, hover]))) * 1000.0
            self.get_logger().info(
                f"Hover reached: plug is {np.linalg.norm(offset[:2]):.2f} mm off the port axis "
                f"and {offset[2]:+.2f} mm from the {hover * 1000:.0f} mm hover height"
            )
        self.get_logger().info("PerceptionFarApproach finished (no insertion)")
        return True
