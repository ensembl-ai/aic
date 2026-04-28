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
from rclpy.duration import Duration
from tesseract_robotics.tesseract_command_language import (
    InstructionPoly_as_MoveInstructionPoly,
    WaypointPoly_as_StateWaypointPoly,
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
        send_feedback("Testing waypoint joint motion with EnsemblRobot")

        robot = EnsemblRobot(get_observation)

        current_transform = robot.ComputeFK(robot.manipulator_tip_frame)
        target_position = np.array(
            [-0.63381193, 0.2899951, 0.05814897],
            dtype=np.float64,
        )
        target_transform = current_transform.copy()
        target_transform[:3, 3] = target_position

        plan = robot.PlanToTarget(target_transform)
        if plan is None:
            self.get_logger().error("Planning failed: PlanToTarget returned None")
            return False

        trajectory = robot.Retime(plan.results)
        if trajectory is None:
            self.get_logger().error("Planning failed: Retime returned None")
            return False

        self.get_logger().info("Tesseract trajectory planned and retimed")

        joint_motion_update = JointMotionUpdate(
            target_stiffness=[150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
            target_damping=[90.0, 90.0, 90.0, 45.0, 45.0, 45.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )

        previous_time = 0.0
        waypoint_index = 0
        for instruction in trajectory.flatten():
            move = InstructionPoly_as_MoveInstructionPoly(instruction)
            waypoint = move.getWaypoint()
            if not waypoint.isStateWaypoint():
                continue

            state_waypoint = WaypointPoly_as_StateWaypointPoly(waypoint)
            waypoint_time = float(state_waypoint.getTime())
            if waypoint_index == 0:
                previous_time = waypoint_time
                waypoint_index += 1
                continue

            positions = np.asarray(
                state_waypoint.getPosition(),
                dtype=np.float64,
            ).reshape(-1)
            velocities = np.asarray(
                state_waypoint.getVelocity(),
                dtype=np.float64,
            ).reshape(-1)
            accelerations = np.asarray(
                state_waypoint.getAcceleration(),
                dtype=np.float64,
            ).reshape(-1)

            joint_motion_update.target_state.positions = positions.tolist()
            joint_motion_update.target_state.velocities = (
                velocities.tolist() if velocities.size == positions.size else []
            )
            joint_motion_update.target_state.accelerations = (
                accelerations.tolist()
                if accelerations.size == positions.size
                else []
            )
            joint_motion_update.target_state.time_from_start = Duration(
                seconds=waypoint_time,
            ).to_msg()
            self.get_logger().info(f"Commanding waypoint {waypoint_index}")
            move_robot(joint_motion_update=joint_motion_update)
            self.sleep_for(max(waypoint_time - previous_time, 0.0))

            previous_time = waypoint_time
            waypoint_index += 1

        if waypoint_index < 2:
            self.get_logger().error("Planned trajectory did not contain waypoints")
            return False

        self.get_logger().info("TestMove.insert_cable() exiting...")
        return True
