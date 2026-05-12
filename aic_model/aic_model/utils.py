import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from transforms3d.affines import compose
from transforms3d.quaternions import quat2mat


def transform_to_matrix(transform_msg):
    """Convert a geometry_msgs Transform into a 4x4 homogeneous matrix."""
    return compose(
        [
            transform_msg.translation.x,
            transform_msg.translation.y,
            transform_msg.translation.z,
        ],
        quat2mat(
            [
                transform_msg.rotation.w,
                transform_msg.rotation.x,
                transform_msg.rotation.y,
                transform_msg.rotation.z,
            ]
        ),
        np.ones(3, dtype=np.float64),
    )


def lookup_transform_matrix(
    tf_buffer,
    target_frame: str,
    source_frame: str,
    timeout_sec: float,
):
    """Look up TF and return target_frame_T_source_frame as a 4x4 matrix."""
    target_T_source_stamped = tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        Time(),
        timeout=Duration(seconds=timeout_sec),
    )
    return transform_to_matrix(target_T_source_stamped.transform)
