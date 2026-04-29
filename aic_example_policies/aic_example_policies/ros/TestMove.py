import numpy as np
from typing import Any, cast

from transforms3d.euler import euler2mat, mat2euler

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
        [-0.4, 0.1],
        [0.4, -0.4],
        [0.2, -0.3],
    ],
    dtype=np.float64,
)
ORIENTATION_ROI_HALF_WIDTH_RADIANS = np.deg2rad(20.0)


def transform_to_position_euler(transform):
    return (
        transform[:3, 3],
        np.array(mat2euler(transform[:3, :3])),
    )


def position_euler_to_transform(position, euler):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler2mat(*euler)
    transform[:3, 3] = position
    return transform


def sample_roi_transform(rng, anchor_transform):
    anchor_position, anchor_euler = transform_to_position_euler(anchor_transform)
    position_low = np.min(POSITION_ROI_OFFSETS_METERS, axis=1)
    position_high = np.max(POSITION_ROI_OFFSETS_METERS, axis=1)
    return position_euler_to_transform(
        anchor_position + rng.uniform(position_low, position_high),
        anchor_euler
        + rng.uniform(
            -ORIENTATION_ROI_HALF_WIDTH_RADIANS,
            ORIENTATION_ROI_HALF_WIDTH_RADIANS,
            size=3,
        ),
    )


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
            sleep_for=self.sleep_for,
        )

        rng = np.random.default_rng()

        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        executed_trajectories = 0

        while (self.time_now() - start_time) < timeout:
            current_transform = robot.ComputeFK(robot.manipulator_tip_frame)
            target_transform = sample_roi_transform(rng, current_transform)
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
