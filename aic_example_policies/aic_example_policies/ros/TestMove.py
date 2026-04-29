import numpy as np
from typing import Any, cast

from transforms3d.euler import mat2euler

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model.robot import EnsemblRobot
from aic_task_interfaces.msg import Task
from rclpy.duration import Duration


POSITION_ROI_OFFSETS_METERS = np.array(
    [
        [-0.2, 0.2],
        [0.4, -0.4],
        [0.2, -0.3],
    ],
    dtype=np.float64,
)
TCP_Z_MAX_DEVIATION_RADIANS = np.deg2rad(20.0)


def transform_to_position_euler(transform):
    return (
        transform[:3, 3],
        np.array(mat2euler(transform[:3, :3])),
    )


def rotation_matrix_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )


def sample_tcp_z_cone_rotation(rng, reference_rotation):
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    tilt_angle = np.arccos(
        rng.uniform(np.cos(TCP_Z_MAX_DEVIATION_RADIANS), 1.0),
    )
    tilt_axis = (
        np.cos(azimuth) * reference_rotation[:, 0]
        + np.sin(azimuth) * reference_rotation[:, 1]
    )
    return rotation_matrix_from_axis_angle(tilt_axis, tilt_angle) @ reference_rotation


def sample_roi_transform(rng, anchor_transform):
    position_low = np.min(POSITION_ROI_OFFSETS_METERS, axis=1)
    position_high = np.max(POSITION_ROI_OFFSETS_METERS, axis=1)
    target_transform = np.eye(4, dtype=np.float64)
    target_transform[:3, :3] = sample_tcp_z_cone_rotation(
        rng,
        anchor_transform[:3, :3],
    )
    target_transform[:3, 3] = anchor_transform[:3, 3] + rng.uniform(
        position_low,
        position_high,
    )
    return target_transform


class TestMove(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("TestMove.__init__()")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"TestMove.insert_cable() enter. Task: {task}")
        send_feedback("Sampling workspace targets and executing planned trajectories")

        robot = cast(Any, EnsemblRobot)(
            get_observation=get_observation,
            execute_joint_motion=lambda update: move_robot(joint_motion_update=update),
        )

        rng = np.random.default_rng()
        reference_transform = robot.ComputeFK(robot.manipulator_tip_frame)

        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        executed_trajectories = 0

        while (self.time_now() - start_time) < timeout:
            current_transform = robot.ComputeFK(robot.manipulator_tip_frame)
            target_transform = sample_roi_transform(rng, reference_transform)
            target_position, target_euler = transform_to_position_euler(
                target_transform
            )

            self.get_logger().info(
                f"Sampled target position {np.array2string(target_position, precision=3)}"
            )
            self.get_logger().info(
                f"Sampled target euler_xyz {np.array2string(target_euler, precision=3)}"
            )
            self.get_logger().info(
                f"Current tcp position {np.array2string(current_transform[:3, 3], precision=3)}"
            )

            plan = robot.PlanToTarget(target_transform)
            if plan is None:
                self.get_logger().info("Skipping sample because planning failed")
                continue

            trajectory = robot.Retime(plan.results)
            if trajectory is None:
                self.get_logger().info("Skipping sample because retiming failed")
                continue

            if not robot.ExecuteTrajectory(trajectory):
                self.get_logger().info("Skipping sample because execution failed")
                continue

            executed_trajectories += 1
            send_feedback(f"Executed sampled trajectory {executed_trajectories}")

        self.get_logger().info("TestMove.insert_cable() exiting...")
        return executed_trajectories > 0
