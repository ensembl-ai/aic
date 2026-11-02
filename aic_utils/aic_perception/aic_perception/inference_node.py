"""ROS node that publishes ``SceneEstimate`` messages from AIC observations.

Subscribes to the synchronized ``/observations`` topic published by
``aic_adapter``, reads each camera pose from TF at the image stamp, and
publishes every schema target in the robot base frame. Present targets are
also broadcast as TF frames ``<tf_prefix>/<target id>`` for RViz and
TF-based planners.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tf2_ros
from aic_model_interfaces.msg import Observation
from aic_perception_interfaces.msg import SceneEstimate, TargetEstimate
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from aic_perception.estimator import SceneEstimatorRuntime
from aic_perception.utils.geometry import from_xyz_quaternion_xyzw, quaternion_xyzw


def matrix_from_transform(message: Any) -> np.ndarray:
    """Convert a ``geometry_msgs/Transform`` into a 4x4 matrix."""

    translation, rotation = message.translation, message.rotation
    return from_xyz_quaternion_xyzw(
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


def pose_message(matrix: np.ndarray) -> Pose:
    """Convert a 4x4 matrix into a ``geometry_msgs/Pose``."""

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, matrix[:3, 3])
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion_xyzw(matrix)
    return pose


class SceneEstimatorNode(Node):
    """Runs the multiview estimator on the newest observation at a fixed rate."""

    def __init__(self, config: dict[str, Any]):
        """Load the model and create ROS interfaces.

        Args:
            config: Validated inference configuration.
        """

        super().__init__("aic_perception")
        settings = config["inference"]
        self.settings = settings
        self.runtime = SceneEstimatorRuntime(settings["checkpoint"], settings["device"])
        self.schema = self.runtime.schema
        self.observation: Observation | None = None
        self.processed_stamp: Any = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.publisher = self.create_publisher(
            SceneEstimate, settings["estimate_topic"], 10
        )
        self.create_subscription(
            Observation, settings["observation_topic"], self.store, 2
        )
        self.create_timer(1.0 / settings["rate_hz"], self.estimate)
        self.get_logger().info(
            f"Loaded {settings['checkpoint']}: {self.runtime.describe()}"
        )

    def store(self, message: Observation) -> None:
        """Keep the newest observation."""

        self.observation = message

    def estimate(self) -> None:
        """Estimate the scene from the newest unprocessed observation."""

        observation = self.observation
        if (
            observation is None
            or observation.center_image.header.stamp == self.processed_stamp
        ):
            return
        stamp = observation.center_image.header.stamp
        images, intrinsics, poses = [], [], []
        try:
            for name in self.schema.cameras:
                image = getattr(observation, f"{name}_image")
                info = getattr(observation, f"{name}_camera_info")
                images.append(
                    np.frombuffer(image.data, np.uint8).reshape(
                        image.height, image.width, 3
                    )
                )
                intrinsics.append(np.asarray(info.k, dtype=float).reshape(3, 3))
                lookup = self.tf_buffer.lookup_transform(
                    self.schema.base_frame,
                    image.header.frame_id,
                    Time.from_msg(stamp),
                    Duration(seconds=0.2),
                )
                poses.append(matrix_from_transform(lookup.transform))
        except tf2_ros.TransformException as error:
            self.get_logger().warning(
                f"Camera pose unavailable: {error}", throttle_duration_sec=5.0
            )
            return
        self.processed_stamp = stamp
        estimates = self.runtime.predict(images, intrinsics, poses)

        message = SceneEstimate(schema=self.schema.name)
        message.header.stamp = stamp
        message.header.frame_id = self.schema.base_frame
        frames = []
        threshold = self.settings["presence_threshold"]
        for estimate in estimates:
            present = estimate.presence_probability >= threshold
            message.targets.append(
                TargetEstimate(
                    id=estimate.id,
                    type=estimate.type,
                    present=present,
                    presence_probability=estimate.presence_probability,
                    confidence=estimate.confidence,
                    pose=pose_message(estimate.base_from_target),
                )
            )
            if present:
                frame = TransformStamped()
                frame.header = message.header
                frame.child_frame_id = f"{self.settings['tf_prefix']}/{estimate.id}"
                pose = message.targets[-1].pose
                frame.transform.translation.x = pose.position.x
                frame.transform.translation.y = pose.position.y
                frame.transform.translation.z = pose.position.z
                frame.transform.rotation = pose.orientation
                frames.append(frame)
        self.publisher.publish(message)
        if frames:
            self.tf_broadcaster.sendTransform(frames)
