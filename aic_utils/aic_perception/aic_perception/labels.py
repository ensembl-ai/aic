"""Ground-truth labels for one captured scene.

Inputs are plain numpy transforms, so labeling is testable without ROS.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aic_perception.schema import BoardSchema
from aic_perception.utils.geometry import invert, project

WORLD_FRAMES = ("world", "aic_world")
MINIMUM_DEPTH_M = 0.05


def world_pose(
    frame: str, edges: dict[str, tuple[str, np.ndarray]]
) -> np.ndarray | None:
    """Compose a frame's world pose from parent-relative ground-truth edges.

    Args:
        frame: Child frame to resolve.
        edges: Child frame mapped to ``(parent, parent_from_child)``.

    Returns:
        ``world_from_frame``, or ``None`` if the frame was not published.
    """

    pose = np.eye(4)
    visited: set[str] = set()
    while frame not in WORLD_FRAMES:
        if frame not in edges or frame in visited:
            return None
        visited.add(frame)
        parent, parent_from_child = edges[frame]
        pose = parent_from_child @ pose
        frame = parent
    return pose


def build_label(
    schema: BoardSchema,
    scene: dict[str, Any],
    edges: dict[str, tuple[str, np.ndarray]],
    world_from_base: np.ndarray,
    cameras: dict[str, dict[str, Any]],
    stamp: float,
) -> dict[str, Any]:
    """Create the JSON label for one scene capture.

    A target is ``spawned`` when the simulator published its frame and
    ``visible`` when its origin projects inside at least one stored image in
    front of the camera. The board counts as visible when its origin or any
    of its ports is visible.

    Args:
        schema: Board schema.
        scene: Sampled scene description.
        edges: Ground-truth TF edges keyed by child frame.
        world_from_base: Pose of the robot base frame in the world.
        cameras: Camera name mapped to ``frame``, ``width``, ``height``,
            ``intrinsics`` (3x3), and ``base_from_camera`` (4x4).
        stamp: Simulation time of the observation in seconds.

    Returns:
        JSON-serializable label.
    """

    base_from_world = invert(world_from_base)
    targets: dict[str, Any] = {}
    for target in schema.targets:
        world_from_target = world_pose(target.frame, edges)
        if world_from_target is None:
            targets[target.id] = {
                "spawned": False,
                "visible": False,
                "base_from_target": None,
                "pixels": {},
            }
            continue
        base_from_target = base_from_world @ world_from_target
        pixels = {}
        visible = False
        for name, camera in cameras.items():
            uv, depth = project(
                np.asarray(camera["intrinsics"]),
                np.asarray(camera["base_from_camera"]),
                base_from_target[None, :3, 3],
            )
            u, v, z = float(uv[0, 0]), float(uv[0, 1]), float(depth[0])
            pixels[name] = [u, v, z]
            inside = 0.0 <= u < camera["width"] and 0.0 <= v < camera["height"]
            visible |= bool(inside and z > MINIMUM_DEPTH_M)
        targets[target.id] = {
            "spawned": True,
            "visible": visible,
            "base_from_target": base_from_target.tolist(),
            "pixels": pixels,
        }
    board = next(target for target in schema.targets if target.type == "board")
    targets[board.id]["visible"] = targets[board.id]["spawned"] and any(
        item["visible"] for item in targets.values()
    )
    return {
        "schema": schema.name,
        "stamp": stamp,
        "scene": scene,
        "cameras": {
            name: {
                "frame": camera["frame"],
                "width": camera["width"],
                "height": camera["height"],
                "intrinsics": np.asarray(camera["intrinsics"]).tolist(),
                "base_from_camera": np.asarray(camera["base_from_camera"]).tolist(),
            }
            for name, camera in cameras.items()
        },
        "targets": targets,
    }
