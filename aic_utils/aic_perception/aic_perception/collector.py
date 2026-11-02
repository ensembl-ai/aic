"""Scene collection worker that runs inside one ``aic_eval`` container.

For each assigned scene the worker spawns a randomized task board through the
same ``/gz_server`` services ``aic_engine`` uses, waits for a synchronized
``/observations`` message rendered after the spawn, reads privileged poses
from ``/scoring/tf``, and writes three JPEG images plus ``label.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Callable

import cv2
import numpy as np
import rclpy
import tf2_ros
from aic_model_interfaces.msg import Observation
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import DeleteEntity, SpawnEntity
from tf2_msgs.msg import TFMessage

from aic_perception.labels import build_label
from aic_perception.layout import image_path, label_path, split_of
from aic_perception.scene_sampler import sample_scene, xacro_arguments
from aic_perception.schema import BoardSchema
from aic_perception.utils.geometry import (
    from_xyz_quaternion_xyzw,
    scaled_intrinsics,
)

ENTITY_NAME = "task_board"
MAXIMUM_CONSECUTIVE_FAILURES = 5
FROZEN_RENDERER_EXIT_CODE = 2


class FrozenRendererError(RuntimeError):
    """Raised when a newly spawned board does not show up in the camera images."""


def thumbnail(image: np.ndarray) -> np.ndarray:
    """Return a small grayscale copy used to compare consecutive captures."""

    return image[::8, ::8].mean(axis=-1)


def seconds(stamp: Any) -> float:
    """Convert a ``builtin_interfaces/Time`` to seconds.

    Args:
        stamp: ROS time message.

    Returns:
        Time in seconds.
    """

    return stamp.sec + stamp.nanosec * 1e-9


def matrix_from_transform(message: Any) -> np.ndarray:
    """Convert a ``geometry_msgs/Transform`` into a 4x4 matrix.

    Args:
        message: ROS transform.

    Returns:
        4x4 transform.
    """

    translation, rotation = message.translation, message.rotation
    return from_xyz_quaternion_xyzw(
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
    )


class SceneCollector(Node):
    """ROS node that turns sampled scenes into labeled multiview samples."""

    def __init__(self, config: dict[str, Any], worker_index: int, worker_count: int):
        """Create subscriptions, service clients, and the TF buffer.

        Args:
            config: Validated collection configuration.
            worker_index: Index of this worker.
            worker_count: Total number of workers sharing the scene range.
        """

        super().__init__(
            f"aic_perception_collector_{worker_index}",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.config = config
        self.schema = BoardSchema.from_config(config)
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.timing = config["timing"]
        self.previous_thumbnails: dict[str, np.ndarray] = {}
        self.empty_thumbnails: dict[str, np.ndarray] = {}
        self.observation: Observation | None = None
        self.edges: dict[str, tuple[str, np.ndarray]] = {}
        self.edges_after = float("inf")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(
            Observation, "/observations", self.store_observation, 2
        )
        self.create_subscription(TFMessage, "/scoring/tf", self.store_ground_truth, 20)
        self.spawn_client = self.create_client(SpawnEntity, "/gz_server/spawn_entity")
        self.delete_client = self.create_client(
            DeleteEntity, "/gz_server/delete_entity"
        )

    def store_observation(self, message: Observation) -> None:
        """Keep the newest synchronized observation."""

        self.observation = message

    def store_ground_truth(self, message: TFMessage) -> None:
        """Record ground-truth edges published after the current spawn."""

        for item in message.transforms:
            if seconds(item.header.stamp) >= self.edges_after:
                self.edges[item.child_frame_id] = (
                    item.header.frame_id,
                    matrix_from_transform(item.transform),
                )

    def sim_time(self) -> float:
        """Return the current simulation time in seconds."""

        return self.get_clock().now().nanoseconds * 1e-9

    def wait_until(
        self, condition: Callable[[], bool], timeout: float, what: str
    ) -> None:
        """Spin until a condition holds.

        Args:
            condition: Predicate evaluated after every spin.
            timeout: Wall-clock limit in seconds.
            what: Description used in the timeout error.

        Raises:
            TimeoutError: If the condition does not hold in time.
        """

        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {what}")
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, client: Any, request: Any) -> Any:
        """Call a service and wait for its response.

        Args:
            client: Service client.
            request: Request message.

        Returns:
            Service response.
        """

        future = client.call_async(request)
        self.wait_until(future.done, self.timing["service_timeout_s"], client.srv_name)
        return future.result()

    def wait_for_simulation(self) -> None:
        """Block until the AIC bringup serves spawning and publishes observations."""

        timeout = self.timing["bringup_timeout_s"]
        for client in (self.spawn_client, self.delete_client):
            if not client.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f"Service {client.srv_name} is unavailable")
        self.wait_until(
            lambda: self.observation is not None, timeout, "first observation"
        )
        self.call(self.delete_client, DeleteEntity.Request(entity=ENTITY_NAME))
        ready_after = self.sim_time() + self.timing["settle_time_s"]
        self.wait_until(
            lambda: seconds(self.observation.center_image.header.stamp) >= ready_after,
            self.timing["observation_timeout_s"],
            "an observation of the empty scene",
        )
        self.empty_thumbnails = {
            name: thumbnail(self.resize(getattr(self.observation, f"{name}_image")))
            for name in self.schema.cameras
        }

    def resize(self, image: Any) -> np.ndarray:
        """Convert a ROS ``rgb8`` image to the stored dataset resolution.

        Args:
            image: ``sensor_msgs/Image``.

        Returns:
            ``(height, width, 3)`` uint8 RGB array.
        """

        if image.encoding != "rgb8":
            raise RuntimeError(f"Unexpected image encoding {image.encoding}")
        pixels = np.frombuffer(image.data, np.uint8).reshape(
            image.height, image.width, 3
        )
        size = (self.config["images"]["width"], self.config["images"]["height"])
        return cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)

    def spawn_board(self, scene: dict[str, Any]) -> float:
        """Replace the task board with a newly sampled one.

        Args:
            scene: Sampled scene description.

        Returns:
            Simulation time just before the spawn request.

        Raises:
            RuntimeError: If xacro or spawning fails.
        """

        self.call(self.delete_client, DeleteEntity.Request(entity=ENTITY_NAME))
        urdf = subprocess.run(
            [
                "xacro",
                self.config["collection"]["task_board_xacro"],
                *xacro_arguments(scene),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        board = scene["board"]
        request = SpawnEntity.Request(
            name=ENTITY_NAME, allow_renaming=False, resource_string=urdf
        )
        request.initial_pose.header.frame_id = "world"
        request.initial_pose.pose.position.x = board["x"]
        request.initial_pose.pose.position.y = board["y"]
        request.initial_pose.pose.position.z = board["z"]
        request.initial_pose.pose.orientation.z = float(np.sin(board["yaw"] / 2.0))
        request.initial_pose.pose.orientation.w = float(np.cos(board["yaw"] / 2.0))
        self.edges = {}
        spawn_time = self.sim_time()
        self.edges_after = spawn_time
        response = self.call(self.spawn_client, request)
        if response.result.result != Result.RESULT_OK:
            raise RuntimeError(f"Spawn failed: {response.result.error_message}")
        return spawn_time

    def lookup(self, target: str, source: str, stamp: Any) -> np.ndarray:
        """Look up ``target_from_source`` from TF.

        Args:
            target: Target frame.
            source: Source frame.
            stamp: ROS time message for the lookup.

        Returns:
            4x4 transform.
        """

        message = self.tf_buffer.lookup_transform(
            target, source, Time.from_msg(stamp), Duration(seconds=1.0)
        )
        return matrix_from_transform(message.transform)

    def capture(
        self, scene: dict[str, Any], spawn_time: float
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Wait for a fresh observation and label it.

        Args:
            scene: Sampled scene description.
            spawn_time: Simulation time of the spawn request.

        Returns:
            Label and resized RGB images keyed by camera name.
        """

        ready_after = spawn_time + self.timing["settle_time_s"]
        board_frame = next(t.frame for t in self.schema.targets if t.type == "board")
        self.wait_until(
            lambda: self.observation is not None
            and seconds(self.observation.center_image.header.stamp) >= ready_after
            and board_frame in self.edges,
            self.timing["observation_timeout_s"],
            "a post-spawn observation and ground truth",
        )
        observation = self.observation
        assert observation is not None
        stamp = observation.center_image.header.stamp
        width, height = self.config["images"]["width"], self.config["images"]["height"]
        cameras, images = {}, {}
        for name in self.schema.cameras:
            image = getattr(observation, f"{name}_image")
            info = getattr(observation, f"{name}_camera_info")
            images[name] = self.resize(image)
            cameras[name] = {
                "frame": image.header.frame_id,
                "width": width,
                "height": height,
                "intrinsics": scaled_intrinsics(
                    np.asarray(info.k).reshape(3, 3),
                    width / image.width,
                    height / image.height,
                ),
                "base_from_camera": self.lookup(
                    self.schema.base_frame, image.header.frame_id, stamp
                ),
            }
        world_from_base = self.lookup("world", self.schema.base_frame, stamp)
        label = build_label(
            self.schema,
            scene,
            dict(self.edges),
            world_from_base,
            cameras,
            seconds(stamp),
        )
        label["joint_positions"] = dict(
            zip(
                observation.joint_states.name,
                map(float, observation.joint_states.position),
            )
        )
        return label, images

    def save(
        self, scene_id: int, label: dict[str, Any], images: dict[str, np.ndarray]
    ) -> None:
        """Write one scene's images, then its label as the completion marker.

        Every file is written under a temporary name and renamed, so readers
        never see partial files and a scene counts as done only once its
        label exists.

        Args:
            scene_id: Global scene index.
            label: Scene label.
            images: RGB images keyed by camera name.
        """

        root = self.config["dataset"]["directory"]
        split = split_of(self.config, scene_id)
        quality = [cv2.IMWRITE_JPEG_QUALITY, self.config["collection"]["jpeg_quality"]]
        for name, image in images.items():
            final = image_path(root, split, scene_id, name)
            staging = final.with_name(f".{final.stem}.tmp.jpg")
            if not cv2.imwrite(
                str(staging), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), quality
            ):
                raise RuntimeError(f"Failed to write {final}")
            os.replace(staging, final)
        final = label_path(root, split, scene_id)
        staging = final.with_name(f".{final.name}.tmp")
        staging.write_text(json.dumps({**label, "split": split}), encoding="utf-8")
        os.replace(staging, final)

    def assigned_scenes(self) -> list[int]:
        """Return this worker's scene IDs whose label is not yet on disk."""

        root = self.config["dataset"]["directory"]
        total = self.config["collection"]["scenes"]
        return [
            scene_id
            for scene_id in range(self.worker_index, total, self.worker_count)
            if not label_path(root, split_of(self.config, scene_id), scene_id).is_file()
        ]

    def check_rendering(
        self, images: dict[str, np.ndarray], board_visible: bool
    ) -> None:
        """Reject a capture in which the newly spawned board was not rendered.

        Every scene spawns a differently placed board, so a capture whose
        cameras all match the previous scene, or, when the board should be in
        view, all match the empty scene recorded at startup, means the renderer
        stopped drawing while the simulator kept publishing fresh timestamps.

        Args:
            images: Resized RGB images keyed by camera name.
            board_visible: Whether the label places the board in view.

        Raises:
            FrozenRendererError: If the capture matches a reference image.
        """

        current = {name: thumbnail(image) for name, image in images.items()}
        threshold = self.config["collection"]["minimum_image_change"]

        def unchanged(reference: dict[str, np.ndarray]) -> bool:
            """Whether every camera matches the reference within the threshold."""

            return bool(reference) and all(
                np.abs(current[name] - reference[name]).mean() < threshold
                for name in current
            )

        if unchanged(self.previous_thumbnails):
            raise FrozenRendererError("Camera images did not change after a new spawn")
        if board_visible and unchanged(self.empty_thumbnails):
            raise FrozenRendererError(
                "Camera images show the empty scene after a spawn"
            )
        self.previous_thumbnails = current

    def run(self) -> int:
        """Collect every assigned scene.

        Returns:
            Process exit code: 0 on success, 1 after repeated failures, 2 when
            the renderer froze (the orchestrator then restarts the container).
        """

        self.wait_for_simulation()
        pending = self.assigned_scenes()
        self.get_logger().info(
            f"Worker {self.worker_index}: {len(pending)} scenes to collect"
        )
        failures = 0
        for scene_id in pending:
            scene = sample_scene(self.config, scene_id)
            try:
                spawn_time = self.spawn_board(scene)
                label, images = self.capture(scene, spawn_time)
                board = next(t.id for t in self.schema.targets if t.type == "board")
                self.check_rendering(images, label["targets"][board]["visible"])
                self.save(scene_id, label, images)
                failures = 0
            except FrozenRendererError as error:
                self.get_logger().error(
                    f"Scene {scene_id} not saved: {error}; exiting for a restart"
                )
                return FROZEN_RENDERER_EXIT_CODE
            except (
                TimeoutError,
                RuntimeError,
                subprocess.CalledProcessError,
                tf2_ros.TransformException,
            ) as error:
                failures += 1
                self.get_logger().error(f"Scene {scene_id} failed: {error}")
                if failures >= MAXIMUM_CONSECUTIVE_FAILURES:
                    return 1
        return 0
