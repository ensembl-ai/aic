import numpy as np
from typing import Any, cast

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model.robot import EnsemblRobot
from aic_task_interfaces.msg import Task
from rclpy.duration import Duration


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
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
        )

        rng = np.random.default_rng()
        workspace_low = np.array([-0.6, 0.1, 0.2], dtype=np.float64)
        workspace_high = np.array([0.6, 0.4, 0.5], dtype=np.float64)

        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        executed_trajectories = 0

        while (self.time_now() - start_time) < timeout:
            current_transform = robot.ComputeFK(robot.manipulator_tip_frame)
            target_position = rng.uniform(workspace_low, workspace_high)
            target_transform = current_transform.copy()
            target_transform[:3, 3] = target_position

            self.get_logger().info(
                "Sampled target position "
                f"{np.array2string(target_position, precision=3)}"
            )
            self.get_logger().info(
                "Current tcp position "
                f"{np.array2string(current_transform[:3, 3], precision=3)}"
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
            send_feedback(
                f"Executed sampled trajectory {executed_trajectories}"
            )

        self.get_logger().info("TestMove.insert_cable() exiting...")
        return executed_trajectories > 0
