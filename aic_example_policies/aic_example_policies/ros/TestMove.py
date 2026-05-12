import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model.robot import EnsemblRobot
from aic_model.utils import lookup_transform_matrix
from aic_task_interfaces.msg import Task


FAR_APPROACH_Z_OFFSET_METERS = 0.005
APPROACH_MAX_JOINT_DELTA = np.deg2rad(90.0)
INSERTION_OVERSHOOT_METERS = 0.015
INSERTION_DIRECTION_BASE = np.array([0.0, 0.0, -1.0], dtype=np.float64)
INSERTION_FORCE_THRESHOLDS_N = np.array([0.0, 0.0, 15.0], dtype=np.float64)
INSERTION_VELOCITY = 0.01
INSERTION_TIMEOUT_SEC = 30.0
INSERTION_COMMAND_PERIOD_SEC = 0.05
INSERTION_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
INSERTION_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
INSERTION_WRENCH_FEEDBACK_GAINS = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
TF_TIMEOUT_SEC = 5.0


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
        send_feedback("Planning plug approach to target port")

        robot = EnsemblRobot(
            get_observation=get_observation,
            execute_motion=lambda update: move_robot(motion_update=update),
            execute_joint_motion=lambda update: move_robot(joint_motion_update=update),
        )

        base_frame = robot.manipulator_base_frame
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        ## FAR APPROACH ESTIMATIONS ##
        # base_T_port is estimated by perception.
        base_T_port = lookup_transform_matrix(
            self._parent_node._tf_buffer,
            base_frame,
            port_frame,
            TF_TIMEOUT_SEC,
        )
        # base_T_plug is estimated by perception.
        base_T_plug = lookup_transform_matrix(
            self._parent_node._tf_buffer,
            base_frame,
            plug_frame,
            TF_TIMEOUT_SEC,
        )
        base_T_tip = robot.ComputeFK()
        plug_T_tip = np.linalg.inv(base_T_plug) @ base_T_tip
        # FAR APPROACH TRANSFORM
        base_T_plug_far_approach = base_T_port.copy()
        base_T_plug_far_approach[:3, 3] += np.array(
            [0.0, 0.0, FAR_APPROACH_Z_OFFSET_METERS],
            dtype=np.float64,
        )
        base_T_tip_target = base_T_plug_far_approach @ plug_T_tip
        self.get_logger().info(
            f"Planning so plug frame '{plug_frame}' reaches "
            f"base_T_plug_far_approach above '{port_frame}' at "
            f"{np.array2string(base_T_plug_far_approach[:3, 3], precision=3)} "
            f"and reaches base_T_tip_target at "
            f"{np.array2string(base_T_tip_target[:3, 3], precision=3)}"
        )

        plan = robot.PlanToTarget(
            base_T_tip_target,
            max_joint_delta=APPROACH_MAX_JOINT_DELTA,
        )
        if plan is None:
            self.get_logger().info(robot._planner.last_failure_reason)
            return False

        trajectory = robot.Retime(plan.results)
        if trajectory is None:
            self.get_logger().info("Plug approach retiming failed")
            return False

        if not robot.ExecuteTrajectory(trajectory):
            self.get_logger().info("Plug approach execution failed")
            return False

        send_feedback("Executing force-limited insertion")
        inserted = robot.executor.insert_part(
            direction=INSERTION_DIRECTION_BASE,
            force_thresholds=INSERTION_FORCE_THRESHOLDS_N,
            velocity=INSERTION_VELOCITY,
            max_distance=FAR_APPROACH_Z_OFFSET_METERS + INSERTION_OVERSHOOT_METERS,
            timeout=INSERTION_TIMEOUT_SEC,
            command_period=INSERTION_COMMAND_PERIOD_SEC,
            frame_id=base_frame,
            stiffness=INSERTION_STIFFNESS,
            damping=INSERTION_DAMPING,
            wrench_feedback_gains=INSERTION_WRENCH_FEEDBACK_GAINS,
            get_observation=get_observation,
            sleep_for=self.sleep_for,
            stamp_message=lambda: self.get_clock().now().to_msg(),
        )
        if inserted:
            send_feedback("Insertion motion completed")
        else:
            self.get_logger().info(robot._planner.last_failure_reason)

        self.get_logger().info("TestMove.insert_cable() exiting...")
        return inserted
