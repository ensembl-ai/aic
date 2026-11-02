"""Geometry helpers and ground-truth labeling."""

from __future__ import annotations

import numpy as np

from aic_perception.config import load_config
from aic_perception.labels import build_label, world_pose
from aic_perception.schema import BoardSchema
from aic_perception.utils.geometry import (
    from_xyz_quaternion_xyzw,
    from_xyz_rpy,
    invert,
    project,
    quaternion_xyzw,
    scaled_intrinsics,
)

INTRINSICS = np.array([[1236.63, 0.0, 576.0], [0.0, 1236.63, 512.0], [0.0, 0.0, 1.0]])


def test_quaternion_round_trip() -> None:
    """ROS-ordered quaternions survive a matrix round trip."""

    pose = from_xyz_rpy((0.1, -0.2, 0.3), (0.4, -0.5, 2.0))
    rebuilt = from_xyz_quaternion_xyzw(tuple(pose[:3, 3]), quaternion_xyzw(pose))
    assert np.allclose(pose, rebuilt)
    assert np.allclose(invert(pose) @ pose, np.eye(4))


def test_half_resolution_intrinsics_follow_pixel_centers() -> None:
    """The image center maps to the image center after a 2x downscale."""

    scaled = scaled_intrinsics(INTRINSICS, 0.5, 0.5)
    assert np.isclose(scaled[0, 0], INTRINSICS[0, 0] / 2)
    assert np.isclose(scaled[0, 2], 287.75)
    assert np.isclose(scaled[1, 2], 255.75)


def test_projection_of_point_on_optical_axis() -> None:
    """A point straight ahead of an optical frame lands on the principal point."""

    camera = from_xyz_rpy((1.0, 2.0, 3.0), (0.3, 0.2, 0.1))
    point = (camera @ np.array([0.0, 0.0, 0.5, 1.0]))[:3]
    pixels, depth = project(INTRINSICS, camera, point[None])
    assert np.allclose(pixels[0], [576.0, 512.0])
    assert np.isclose(depth[0], 0.5)


def test_world_pose_composes_nested_edges() -> None:
    """Parent-relative ground-truth edges compose up to the world."""

    board = from_xyz_rpy((0.15, -0.2, 1.14), (0.0, 0.0, 3.1))
    mount = from_xyz_rpy((-0.08, -0.17, 0.012), (0.0, 0.0, 0.1))
    port = from_xyz_rpy((0.011, -0.012, 0.167), (0.0, 0.0, 0.0))
    edges = {
        "task_board": ("aic_world", board),
        "task_board/m": ("task_board", mount),
        "task_board/m/port": ("task_board/m", port),
    }
    assert np.allclose(world_pose("task_board/m/port", edges), board @ mount @ port)
    assert world_pose("task_board/missing", edges) is None


def test_label_marks_spawned_and_visible_targets() -> None:
    """Published frames are spawned; only frames in view are visible."""

    config = load_config("collect")
    schema = BoardSchema.from_config(config)
    world_from_base = from_xyz_rpy((-0.2, 0.2, 1.14), (0.0, 0.0, np.pi))
    board_world = from_xyz_rpy((0.15, -0.2, 1.14), (0.0, 0.0, 0.0))
    port = schema.targets[1]
    hidden = schema.targets[2]
    edges = {
        "task_board": ("aic_world", board_world),
        "task_board/task_board_base_link": ("task_board", np.eye(4)),
        port.frame: ("task_board", from_xyz_rpy((0.0, 0.0, 0.1), (0.0, 0.0, 0.0))),
        hidden.frame: ("task_board", from_xyz_rpy((3.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    }
    look_down = from_xyz_rpy((0.15, -0.2, 1.74), (np.pi, 0.0, 0.0))
    base_from_camera = invert(world_from_base) @ look_down
    camera = {
        "frame": "camera/optical",
        "width": 576,
        "height": 512,
        "intrinsics": scaled_intrinsics(INTRINSICS, 0.5, 0.5),
        "base_from_camera": base_from_camera,
    }
    label = build_label(
        schema, {"scene_id": 0}, edges, world_from_base, {"center": camera}, 1.0
    )
    targets = label["targets"]
    assert targets["task_board"]["spawned"] and targets["task_board"]["visible"]
    assert targets[port.id]["visible"]
    assert targets[hidden.id]["spawned"] and not targets[hidden.id]["visible"]
    assert not targets[schema.targets[-1].id]["spawned"]
    expected = invert(world_from_base) @ board_world @ edges[port.frame][1]
    assert np.allclose(targets[port.id]["base_from_target"], expected)
