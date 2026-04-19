import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException


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
        self.get_logger().info(f"WaveArm.insert_cable() enter. Task: {task}")
        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        send_feedback("Testing the arm motion")
        while (self.time_now() - start_time) < timeout:
            self.sleep_for(0.25)
            observation = get_observation()
            if observation is None:
                self.get_logger().info("No observation received.")
                continue
            self.get_logger().info(f"observation wrench: {observation.wrist_wrench}")
            t = (
                observation.wrist_wrench.header.stamp.sec
                + observation.wrist_wrench.header.stamp.nanosec / 1e9
            )
            self.get_logger().info(f"observation wrench time: {t}")

            try:
                tcp_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    "gripper/tcp",
                    Time(),
                )
            except TransformException as ex:
                self.get_logger().error(
                    f"Failed to get gripper/tcp orientation in base_link: {ex}"
                )
                continue

            pose_in_base_link = Pose(
                position=Point(x=-0.63381193, y=0.2899951, z=0.05814897),
                orientation=Quaternion(
                    x=tcp_tf.transform.rotation.x,
                    y=tcp_tf.transform.rotation.y,
                    z=tcp_tf.transform.rotation.z,
                    w=tcp_tf.transform.rotation.w,
                ),
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose_in_base_link,
                frame_id="base_link",
            )
        self.get_logger().info("TestMove.insert_cable() exiting...")
        return True
