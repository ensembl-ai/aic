import numpy as np

from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model.robot import EnsemblRobot
from aic_task_interfaces.msg import Task


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
        send_feedback("Testing waypoint joint motion with EnsemblRobot")

        robot = EnsemblRobot(get_observation)

        current_transform = robot.ComputeFK(robot.manipulator_tip_frame)
        target_position = np.array(
            [-0.63381193, 0.2899951, 0.05814897],
            dtype=np.float64,
        )

        waypoint_positions = np.linspace(
            current_transform[:3, 3],
            target_position,
            num=10,
            dtype=np.float64,
        )

        joint_motion_update = JointMotionUpdate(
            target_stiffness=[150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
            target_damping=[90.0, 90.0, 90.0, 45.0, 45.0, 45.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )

        for index, position in enumerate(waypoint_positions[1:], start=1):
            self.get_logger().info(
                f"Commanding waypoint {index}/{len(waypoint_positions) - 1}"
            )

            waypoint_transform = current_transform.copy()
            waypoint_transform[:3, 3] = position
            joint_positions = robot.ComputeIK(waypoint_transform)
            if joint_positions is None:
                self.get_logger().error(f"IK failed for waypoint {index}")
                return False

            joint_motion_update.target_state.positions = joint_positions.tolist()
            move_robot(joint_motion_update=joint_motion_update)
            self.sleep_for(1.0)

        self.get_logger().info("TestMove.insert_cable() exiting...")
        return True
