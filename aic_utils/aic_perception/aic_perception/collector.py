"""Scene collection worker that runs inside one ``aic_eval`` container.

Every board is one AIC-style trial. Before a board, the worker makes sure the
gripper holds the task's cable exactly as ``aic_engine`` attaches it
(``sfp_sc_cable`` with the SFP plug in hand for SFP tasks,
``sfp_sc_cable_reversed`` with the SC plug in hand for SC tasks), then returns
the arm to HOME, spawns the randomized task board through the same
``/gz_server`` services ``aic_engine`` uses, and captures ``views_per_board``
scenes along an approach: HOME, then the calibrated plug tip at descending
heights above the task's target port. Each scene waits for a synchronized ``/observations`` message rendered at
its viewpoint, reads privileged poses from ``/scoring/tf``, and writes three
JPEG images plus a label.
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
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from transforms3d.euler import euler2quat

from aic_perception.labels import build_label, world_pose
from aic_perception.layout import image_path, label_path, split_of
from aic_perception.scene_sampler import sample_approach, sample_scene, xacro_arguments
from aic_perception.schema import BoardSchema
from aic_perception.utils.geometry import (
    from_xyz_quaternion_xyzw,
    from_xyz_rpy,
    invert,
    quaternion_xyzw,
    scaled_intrinsics,
    transform,
)
from aic_perception.utils.motion import level_rotation, tcp_target

ENTITY_NAME = "task_board"
CABLE_NAME = "cable_0"
MINIMUM_MOVE_DURATION_S = 0.3
MAXIMUM_CONSECUTIVE_FAILURES = 5
COLLECTION_PASSES = 3
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
        self.motion_publisher = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        self.tcp_from_plug = {
            plug_type: from_xyz_quaternion_xyzw(tuple(grasp["translation"]), tuple(grasp["quaternion_xyzw"]))
            for plug_type, grasp in config["viewpoints"]["plug_grasp"].items()
        }
        self.held: dict[str, Any] = {}
        self.home_tcp = np.eye(4)
        self.away_from_home = False
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
        self.home_tcp = self.current_tcp()

    def hold(self, plug_type: str) -> None:
        """Make the gripper hold the cable of a task's plug type.

        Clears the board, returns HOME, attaches the cable as ``aic_engine``
        does, and records the empty scene that the rendering check compares
        against, since the held cable is part of every image.

        Args:
            plug_type: ``sfp`` or ``sc``.
        """

        self.call(self.delete_client, DeleteEntity.Request(entity=ENTITY_NAME))
        self.go_home()
        cable = self.config["cable"]
        self.held = {}
        held = {"plug_type": plug_type, "cable_type": None, "gripper_offset": None}
        if cable["spawn"]:
            offsets = cable["gripper_offsets"][plug_type]
            offset = offsets[int(np.random.default_rng().integers(len(offsets)))]
            self.spawn_cable(cable["types"][plug_type], offset)
            held |= {"cable_type": cable["types"][plug_type], "gripper_offset": offset}
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
        self.previous_thumbnails = {}
        self.held = held

    def current_tcp(self) -> np.ndarray:
        """Latest ``gripper/tcp`` pose in the base frame."""

        message = self.tf_buffer.lookup_transform(
            self.schema.base_frame, "gripper/tcp", Time(), Duration(seconds=5.0)
        )
        return matrix_from_transform(message.transform)

    def spawn_cable(self, cable_type: str, gripper_offset: list[float]) -> None:
        """Attach a cable to the gripper the way ``aic_engine`` does.

        The cable is placed at the TCP position plus a world-frame offset with
        the configured orientation, then given time to settle.

        Args:
            cable_type: ``sfp_sc_cable`` or ``sfp_sc_cable_reversed``.
            gripper_offset: World-frame offset from the TCP position.
        """

        cable = self.config["cable"]
        world_from_tcp = matrix_from_transform(
            self.tf_buffer.lookup_transform("world", "gripper/tcp", Time(), Duration(seconds=5.0)).transform
        )
        self.call(self.delete_client, DeleteEntity.Request(entity=CABLE_NAME))
        sdf = subprocess.run(
            ["xacro", cable["xacro"], "attach_cable_to_gripper:=true", f"cable_type:={cable_type}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        request = SpawnEntity.Request(name=CABLE_NAME, allow_renaming=False, resource_string=sdf)
        request.initial_pose.header.frame_id = "world"
        position = world_from_tcp[:3, 3] + np.asarray(gripper_offset)
        request.initial_pose.pose.position.x, request.initial_pose.pose.position.y, request.initial_pose.pose.position.z = map(float, position)
        w, x, y, z = euler2quat(*cable["rpy"], axes="sxyz")
        request.initial_pose.pose.orientation.x = float(x)
        request.initial_pose.pose.orientation.y = float(y)
        request.initial_pose.pose.orientation.z = float(z)
        request.initial_pose.pose.orientation.w = float(w)
        response = self.call(self.spawn_client, request)
        if response.result.result != Result.RESULT_OK:
            raise RuntimeError(f"Cable spawn failed: {response.result.error_message}")
        settle_until = self.sim_time() + cable["settle_s"]
        self.wait_until(lambda: self.sim_time() >= settle_until, self.timing["observation_timeout_s"], "cable to settle")

    def command_tcp(self, goal: np.ndarray) -> None:
        """Publish one Cartesian TCP target to the AIC controller."""

        viewpoints = self.config["viewpoints"]
        x, y, z, w = quaternion_xyzw(goal)
        self.motion_publisher.publish(
            MotionUpdate(
                header=Header(frame_id=self.schema.base_frame, stamp=self.get_clock().now().to_msg()),
                pose=Pose(
                    position=Point(x=float(goal[0, 3]), y=float(goal[1, 3]), z=float(goal[2, 3])),
                    orientation=Quaternion(x=x, y=y, z=z, w=w),
                ),
                target_stiffness=np.diag(viewpoints["stiffness"]).flatten().tolist(),
                target_damping=np.diag(viewpoints["damping"]).flatten().tolist(),
                feedforward_wrench_at_tip=Wrench(force=Vector3(), torque=Vector3()),
                wrench_feedback_gains_at_tip=[0.0] * 6,
                trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION),
            )
        )

    def move_tcp(self, goal: np.ndarray) -> None:
        """Move the TCP to ``goal`` along a straight line, then wait to settle.

        Labels always use the TF pose at the image stamp, so an imperfect
        settle only changes the viewpoint, never the label's correctness.
        """

        viewpoints = self.config["viewpoints"]
        start = self.current_tcp()
        period = 0.1
        distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        duration = max(MINIMUM_MOVE_DURATION_S, distance / viewpoints["move_speed_m_s"])
        steps = max(1, round(duration / period))
        for index in range(1, steps + 1):
            fraction = index / steps
            target = goal.copy()
            target[:3, 3] = (1.0 - fraction) * start[:3, 3] + fraction * goal[:3, 3]
            self.command_tcp(target)
            until = self.sim_time() + period
            self.wait_until(lambda: self.sim_time() >= until, self.timing["observation_timeout_s"], "the arm to move")
        deadline = self.sim_time() + viewpoints["settle_timeout_s"]
        while self.sim_time() < deadline:
            self.command_tcp(goal)
            until = self.sim_time() + period
            self.wait_until(lambda: self.sim_time() >= until, self.timing["observation_timeout_s"], "the arm to settle")
            if np.linalg.norm(self.current_tcp()[:3, 3] - goal[:3, 3]) < viewpoints["position_tolerance_m"]:
                break

    def go_home(self) -> None:
        """Return the arm to the recorded start pose before a new board spawns."""

        if self.away_from_home:
            self.move_tcp(self.home_tcp)
            self.away_from_home = False

    def task_target(self, task: dict[str, Any]) -> Any:
        """Return the schema target whose ground-truth frame is the task's port.

        Args:
            task: The board's task (``target_module_name``, ``port_name``).

        Returns:
            Schema target.

        Raises:
            RuntimeError: If the schema has no target for the port.
        """

        frame = f"task_board/{task['target_module_name']}/{task['port_name']}_link_entrance"
        for target in self.schema.targets:
            if target.frame == frame:
                return target
        raise RuntimeError(f"No schema target has frame {frame}")

    def move_to_viewpoint(self, viewpoint: dict[str, Any], target: Any) -> None:
        """Move the cameras to a sampled viewpoint.

        For ``close`` viewpoints the calibrated tip of the held plug is placed
        above the task's target port, aligned with the port as the insertion
        policy aligns it, plus the sampled height, lateral offset, and yaw
        error.

        Args:
            viewpoint: Sampled viewpoint.
            target: Schema target of the task's port.

        Raises:
            RuntimeError: If the target port was not spawned.
        """

        if viewpoint["mode"] == "home":
            return
        board_frame = next(t.frame for t in self.schema.targets if t.type == "board")
        self.wait_until(lambda: board_frame in self.edges, self.timing["observation_timeout_s"], "ground truth")
        world_from_port = world_pose(target.frame, self.edges)
        if world_from_port is None:
            raise RuntimeError(f"Target {target.id} was not spawned")
        world_from_base = matrix_from_transform(
            self.tf_buffer.lookup_transform("world", self.schema.base_frame, Time(), Duration(seconds=1.0)).transform
        )
        entrance = invert(world_from_base) @ world_from_port
        rotation = level_rotation(entrance[:3, :3]) @ from_xyz_rpy((0, 0, 0), (0.0, 0.0, viewpoint["yaw"]))[:3, :3]
        position = entrance[:3, 3] + np.array([viewpoint["offset"][0], viewpoint["offset"][1], viewpoint["height"]])
        tcp_from_plug = self.tcp_from_plug[self.held["plug_type"]]
        self.move_tcp(tcp_target(transform(rotation, position), tcp_from_plug))
        self.away_from_home = True

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
        self, scene: dict[str, Any], ready_time: float
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Wait for a fresh observation and label it.

        Args:
            scene: Sampled scene description.
            ready_time: Simulation time after which the scene is in place.

        Returns:
            Label and resized RGB images keyed by camera name.
        """

        ready_after = ready_time + self.timing["settle_time_s"]
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
        plug_frame = f"{CABLE_NAME}/{self.held['plug_type']}_tip_link"
        world_from_plug = world_pose(plug_frame, self.edges)
        label["held_plug"] = {
            **self.held,
            "frame": plug_frame,
            "base_from_plug": None if world_from_plug is None else (invert(world_from_base) @ world_from_plug).tolist(),
        }
        return label, images

    def save(
        self,
        scene_id: int,
        split: str,
        label: dict[str, Any],
        images: dict[str, np.ndarray],
    ) -> None:
        """Write one scene's images, then its label as the completion marker.

        Every file is written under a temporary name and renamed, so readers
        never see partial files and a scene counts as done only once its
        label exists.

        Args:
            scene_id: Global scene index.
            split: Split of the scene's board.
            label: Scene label.
            images: RGB images keyed by camera name.
        """

        root = self.config["dataset"]["directory"]
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

    def assigned_boards(self) -> dict[int, list[int]]:
        """Return this worker's board IDs mapped to views whose label is not yet on disk."""

        root = self.config["dataset"]["directory"]
        collection = self.config["collection"]
        views = collection["views_per_board"]
        pending = {}
        for board_id in range(self.worker_index, collection["boards"], self.worker_count):
            split = split_of(self.config, board_id)
            missing = [
                view
                for view in range(views)
                if not label_path(root, split, board_id * views + view).is_file()
            ]
            if missing:
                pending[board_id] = missing
        return pending

    def check_rendering(
        self, images: dict[str, np.ndarray], board_visible: bool, after_spawn: bool
    ) -> None:
        """Reject a capture in which the renderer stopped drawing.

        A frozen renderer keeps publishing fresh timestamps with stale
        pixels. The first capture after a spawn shows a differently placed
        board, so it must differ from the previous capture by
        ``minimum_image_change``; later captures of the same board follow a
        camera move that can be only millimetres, so they must merely not be
        identical (``minimum_view_change``). When the board should be in
        view, no capture may match the empty scene recorded when the cable
        was attached.

        Args:
            images: Resized RGB images keyed by camera name.
            board_visible: Whether the label places the board in view.
            after_spawn: Whether this is the first capture of a new board.

        Raises:
            FrozenRendererError: If the capture matches a reference image.
        """

        current = {name: thumbnail(image) for name, image in images.items()}
        collection = self.config["collection"]
        threshold = collection["minimum_image_change"]

        def unchanged(reference: dict[str, np.ndarray], limit: float) -> bool:
            """Whether every camera matches the reference within ``limit``."""

            return bool(reference) and all(
                np.abs(current[name] - reference[name]).mean() < limit
                for name in current
            )

        limit = threshold if after_spawn else collection["minimum_view_change"]
        if unchanged(self.previous_thumbnails, limit):
            raise FrozenRendererError("Camera images did not change after a new spawn or camera move")
        if board_visible and unchanged(self.empty_thumbnails, threshold):
            raise FrozenRendererError(
                "Camera images show the empty scene after a spawn"
            )
        self.previous_thumbnails = current

    def run(self) -> int:
        """Collect every assigned board.

        A board whose views fail is skipped, and up to ``COLLECTION_PASSES``
        passes revisit every board with missing views, so a transient timeout
        does not leave a board incomplete.

        Returns:
            Process exit code: 0 on success, 1 after repeated failures, 2 when
            the renderer froze (the orchestrator then restarts the container).
        """

        self.wait_for_simulation()
        views_per_board = self.config["collection"]["views_per_board"]
        board_target = next(t.id for t in self.schema.targets if t.type == "board")
        for collection_pass in range(COLLECTION_PASSES):
            pending = self.assigned_boards()
            if not pending:
                break
            self.get_logger().info(
                f"Worker {self.worker_index}, pass {collection_pass + 1}: {len(pending)} boards, "
                f"{sum(map(len, pending.values()))} scenes to collect"
            )
            code = self.collect(pending, views_per_board, board_target)
            if code:
                return code
        return 0

    def collect(self, pending: dict[int, list[int]], views_per_board: int, board_target: str) -> int:
        """Collect the missing views of each pending board.

        Args:
            pending: Board IDs mapped to their missing views.
            views_per_board: Views captured per board.
            board_target: Schema ID of the board target.

        Returns:
            Process exit code, or 0 when the pass completed.
        """

        failures = 0
        boards = {board_id: sample_scene(self.config, board_id) for board_id in pending}
        # SFP tasks first, then SC tasks, so the cable changes at most once per pass.
        for board_id in sorted(pending, key=lambda board_id: (boards[board_id]["task"]["port_type"] != "sfp", board_id)):
            board = boards[board_id]
            views = pending[board_id]
            approach = sample_approach(self.config, board_id)
            split = split_of(self.config, board_id)
            try:
                if self.held.get("plug_type") != board["task"]["port_type"]:
                    self.hold(board["task"]["port_type"])
                target = self.task_target(board["task"])
                self.go_home()
                spawn_time = self.spawn_board(board)
                for view in views:
                    scene_id = board_id * views_per_board + view
                    self.move_to_viewpoint(approach[view], target)
                    viewpoint = approach[view] | ({"target": target.id} if approach[view]["mode"] == "close" else {})
                    ready_time = spawn_time if viewpoint["mode"] == "home" else self.sim_time()
                    scene = {**board, "scene_id": scene_id, "view": view, "viewpoint": viewpoint, "target": target.id}
                    label, images = self.capture(scene, ready_time)
                    self.check_rendering(images, label["targets"][board_target]["visible"], view == views[0])
                    self.save(scene_id, split, label, images)
                failures = 0
            except FrozenRendererError as error:
                self.get_logger().error(
                    f"Board {board_id} not saved: {error}; exiting for a restart"
                )
                return FROZEN_RENDERER_EXIT_CODE
            except (
                TimeoutError,
                RuntimeError,
                subprocess.CalledProcessError,
                tf2_ros.TransformException,
            ) as error:
                failures += 1
                self.get_logger().error(f"Board {board_id} failed: {error}")
                if failures >= MAXIMUM_CONSECUTIVE_FAILURES:
                    return 1
        return 0
