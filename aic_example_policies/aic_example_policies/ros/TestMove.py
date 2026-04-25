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
        target_transform = current_transform.copy()
        target_transform[:3, 3] = np.array(
            [-0.63381193, 0.2899951, 0.05814897],
            dtype=np.float64,
        )

        waypoint_positions = np.linspace(
            current_transform[:3, 3],
            target_transform[:3, 3],
            num=11,
            dtype=np.float64,
        )[1:]
        planner_env = robot.GetEnv()
        planner_joint_names = list(
            planner_env.getJointGroup(robot.manipulator_group_name).getJointNames()
        )

        joint_waypoints = []
        for position in waypoint_positions:
            waypoint_transform = target_transform.copy()
            waypoint_transform[:3, 3] = position
            joint_positions = robot.ComputeIK(
                waypoint_transform,
                check_collision=True,
            )
            joint_waypoints.append(joint_positions.tolist())
            planner_env.setState(planner_joint_names, joint_positions)

        joint_motion_update = JointMotionUpdate(
            target_stiffness=[120.0, 120.0, 120.0, 50.0, 50.0, 50.0],
            target_damping=[40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )

        for index, joint_positions in enumerate(joint_waypoints, start=1):
            self.get_logger().info(
                f"Commanding waypoint {index}/{len(joint_waypoints)}"
            )
            joint_motion_update.target_state.positions = joint_positions
            move_robot(joint_motion_update=joint_motion_update)

        self.get_logger().info("TestMove.insert_cable() exiting...")
        return True
