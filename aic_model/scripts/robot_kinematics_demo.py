#!/usr/bin/env python3

import math
import time
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from aic_model.robot import EnsemblRobot


class RobotKinematicsDemo(Node):
    def __init__(self) -> None:
        super().__init__("robot_kinematics_demo")
        self.robot = EnsemblRobot()
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.rng = np.random.default_rng(7)
        self.publish_static_frames()

    def publish_static_frames(self) -> None:
        transforms = []
        for parent_frame, child_frame in [
            ("world", "map"),
            ("map", self.robot.manipulator_base_frame),
        ]:
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent_frame
            transform.child_frame_id = child_frame
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self.static_tf_broadcaster.sendTransform(transforms)

    def publish_joint_state(self, joint_values: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot._active_joint_names)
        msg.position = joint_values.tolist()
        self.joint_state_pub.publish(msg)

    def publish_axis_markers(
        self,
        transform: np.ndarray,
        namespace: str,
        marker_id_offset: int,
        scale: float = 0.08,
    ) -> None:
        origin = transform[:3, 3]
        rotation = transform[:3, :3]
        colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        markers = MarkerArray()
        for axis_idx in range(3):
            marker = Marker()
            marker.header.frame_id = self.robot.manipulator_base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = marker_id_offset + axis_idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.008
            marker.scale.y = 0.014
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = colors[axis_idx][0]
            marker.color.g = colors[axis_idx][1]
            marker.color.b = colors[axis_idx][2]
            marker.lifetime = Duration(sec=0, nanosec=0)
            end = origin + scale * rotation[:, axis_idx]
            marker.points = [
                Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
            ]
            markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def sleep_with_spin(self, duration_sec: float) -> None:
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)

    def sample_joint_configuration(self) -> np.ndarray:
        arm_low = np.array([-math.pi, -2.0, -2.0, -math.pi, -math.pi, -math.pi])
        arm_high = np.array([math.pi, 0.0, 2.0, math.pi, math.pi, math.pi])
        if len(self.robot._manipulator_joint_indices) != arm_low.size:
            raise RuntimeError(
                "The kinematics demo sampling limits expect a 6-DOF manipulator, "
                f"got {len(self.robot._manipulator_joint_indices)} joints."
            )

        joint_values = np.zeros(len(self.robot._active_joint_names), dtype=np.float64)
        joint_values[self.robot._manipulator_joint_indices] = self.rng.uniform(
            low=arm_low,
            high=arm_high,
        )

        manipulator_indices = set(self.robot._manipulator_joint_indices)
        non_manipulator_indices = [
            index
            for index in range(len(self.robot._active_joint_names))
            if index not in manipulator_indices
        ]
        if non_manipulator_indices:
            joint_values[non_manipulator_indices] = self.rng.uniform(
                low=0.0,
                high=0.02,
                size=len(non_manipulator_indices),
            )
        return joint_values

    def sample_collision_free_configuration(
        self,
        label: str,
        max_attempts: int = 200,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for attempt in range(1, max_attempts + 1):
            joint_values = self.sample_joint_configuration()
            self.robot.SetActiveDOFValues(joint_values)
            in_collision, report = self.robot.CheckCollision(report=True)
            if not in_collision:
                return joint_values, {
                    "label": label,
                    "attempt": attempt,
                    "joint_values": joint_values.tolist(),
                    "collision_free": True,
                    "collision_report": report,
                }
        raise RuntimeError(
            f"Failed to sample a collision-free configuration for {label}."
        )

    def report_collision_state(self, label: str) -> dict[str, Any]:
        in_collision, report = self.robot.CheckCollision(report=True)
        manipulator_joints = self.robot.GetActiveDOFValues()
        fk = self.robot.ComputeFK(self.robot.manipulator_tip_frame)
        return {
            "label": label,
            "manipulator_joint_values": manipulator_joints.tolist(),
            "fk_translation": fk[:3, 3].tolist(),
            "in_collision": in_collision,
            "collision_report": report,
        }

    def run(self) -> None:
        reports: dict[str, Any] = {
            "samples": [],
            "ik_solutions": [],
            "motion_steps": [],
        }

        start_config, start_sample = self.sample_collision_free_configuration("start")
        reports["samples"].append(start_sample)
        self.robot.SetActiveDOFValues(start_config)
        self.publish_joint_state(start_config)
        start_fk = self.robot.ComputeFK(self.robot.manipulator_tip_frame)
        self.publish_axis_markers(start_fk, namespace="start_fk", marker_id_offset=0)
        self.get_logger().info(f"Start FK:\n{start_fk}")
        self.sleep_with_spin(0.5)

        target_config, target_sample = self.sample_collision_free_configuration("target")
        reports["samples"].append(target_sample)
        self.robot.SetActiveDOFValues(target_config)
        self.publish_joint_state(target_config)
        target_fk = self.robot.ComputeFK(self.robot.manipulator_tip_frame)
        self.publish_axis_markers(target_fk, namespace="target_fk", marker_id_offset=10)
        self.get_logger().info(f"Target FK:\n{target_fk}")
        self.sleep_with_spin(0.5)

        self.robot.SetActiveDOFValues(start_config)
        self.publish_joint_state(start_config)
        self.sleep_with_spin(0.5)

        ik_solutions = self.robot.ComputeIK(
            target_fk,
            return_all=True,
            check_collision=False,
        )
        if ik_solutions is None:
            raise RuntimeError("No IK solutions found for the sampled target pose.")
        self.get_logger().info(f"Found {len(ik_solutions)} IK solution(s).")

        chosen_solution = None
        for index, solution in enumerate(ik_solutions):
            candidate_active = start_config.copy()
            candidate_active[self.robot._manipulator_joint_indices] = solution
            self.robot.SetActiveDOFValues(candidate_active)
            solution_report = self.report_collision_state(f"ik_solution_{index}")
            solution_report["ik_solution_index"] = index
            solution_report["ik_solution_joint_values"] = solution.tolist()
            reports["ik_solutions"].append(solution_report)
            if chosen_solution is None and not solution_report["in_collision"]:
                chosen_solution = solution.copy()

        if chosen_solution is None:
            raise RuntimeError(
                "No collision-free IK solution found for the sampled target pose."
            )

        chosen_active = start_config.copy()
        chosen_active[self.robot._manipulator_joint_indices] = chosen_solution

        num_steps = 75
        for step_index, alpha in enumerate(np.linspace(0.0, 1.0, num_steps)):
            joint_values = (1.0 - alpha) * start_config + alpha * chosen_active
            self.robot.SetActiveDOFValues(joint_values)
            self.publish_joint_state(joint_values)
            if step_index in (0, num_steps - 1):
                self.publish_axis_markers(
                    self.robot.ComputeFK(self.robot.manipulator_tip_frame),
                    namespace="motion_fk",
                    marker_id_offset=20 + 10 * step_index,
                )
            motion_report = self.report_collision_state(f"motion_step_{step_index}")
            motion_report["step_index"] = step_index
            reports["motion_steps"].append(motion_report)
            self.sleep_with_spin(0.04)

        jacobian = self.robot.ComputeJacobianGeometric(self.robot.manipulator_tip_frame)
        self.get_logger().info(f"Geometric Jacobian:\n{jacobian}")

        print("\n=== Collision / Kinematics Report ===")
        print(f"Start sample: {reports['samples'][0]}")
        print(f"Target sample: {reports['samples'][1]}")
        print("IK solution reports:")
        for solution_report in reports["ik_solutions"]:
            print(solution_report)
        print("Motion step reports:")
        for motion_report in reports["motion_steps"]:
            print(motion_report)
        print("Jacobian:")
        print(jacobian)


def main() -> None:
    rclpy.init()
    node = RobotKinematicsDemo()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
