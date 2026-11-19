"""Perception-guided, force-guided port insertion for ``aic_model`` policies.

Control uses only information a competition policy receives: the task
message, the robot's own TF (``gripper/tcp``), the wrist force/torque reading
in the observation, the multiview ``SceneEstimate``, CAD port depths, and the
offline plug-in-gripper calibration in ``configs/infer.json``. Simulator
ground truth is never used for control; with ``log_ground_truth`` it is only
read, when published, to log errors for evaluation.

Phases, all along the vertical port axis of the level task board, following
the force strategy of a top AIC phase 1 solution
(github.com/BrenzingerLuca/AI-for-Industry-Challenge):

1. approach: plug to ``pre_contact_height_m`` above the perceived entrance,
   aligned to it, either tracking the estimate continuously
   (``tracking_approach``) or in stages with a re-estimate on the way;
2. servo: the close-range refiner measures the port relative to the held
   plug in the wrist images; the plug is moved by that offset and measured
   again, ``refiner.iterations`` times or until within ``refiner.tolerance_m``.
   This also cancels the grasp calibration error and the arm settling short
   under the cable load, since the measurement is relative and repeated. A
   force baseline is then averaged because the sensor cannot be tared;
3. descent at constant speed until contact force or the plug stops moving;
   the plug is inside when it got ``inside_depth_m`` below the entrance;
4. otherwise a spiral search with soft axial and stiff rotational impedance,
   pressing ``press_margin_m`` below the contact height, until the plug drops
   ``entry_depth_m``; after a converged servo the spiral is limited to
   ``refiner.spiral_max_radius_m``;
5. press-in with the soft axial impedance until the plug stalls; a stall at
   least ``min_seat_depth_m`` below the reference height is seated, a shallower
   one is a snag, retried with soft rotation and, after some attempts,
   retracted and searched again.

SC tasks name the engine's slot (``sc_port_0`` or ``sc_port_1``), which
cannot be seen; the estimator reports SC ports by rank along their rail. With
one SC port present it is the target; with two, ``sc_rank_for_slot`` decides.

Every task appends one JSON line to ``logging.trial_directory`` with the task,
each phase's measurements, ground-truth errors when published, and the
outcome; ``scripts/summarize_trials.py`` turns these into a report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback
from aic_perception_interfaces.msg import SceneEstimate
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_slerp
from transforms3d.quaternions import mat2quat, quat2mat

from aic_perception.config import load_config
from aic_perception.utils.geometry import from_xyz_quaternion_xyzw, from_xyz_rpy, invert, quaternion_xyzw, transform
from aic_perception.utils.motion import (
    PortAverage,
    funnel_target,
    level_rotation,
    speed_limited_step,
    spiral_point,
    spiral_step,
    tcp_target,
)

DEFAULT_SENSOR_FRAME = "ati/tool_link"
TIME_MARGIN_S = 5.0
SC_RANK_IDS = ("sc_port_rank_0", "sc_port_rank_1")
CAMERAS = ("left", "center", "right")


def rounded(value: Any) -> Any:
    """JSON-friendly copy of a value with numpy arrays as rounded lists."""

    if isinstance(value, np.ndarray):
        return np.round(value, 6).tolist()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return value


def stamp_seconds(stamp: Any) -> float:
    """Convert a ``builtin_interfaces/Time`` to seconds."""

    return stamp.sec + stamp.nanosec * 1e-9


def diagonal(values: list[float]) -> list[float]:
    """Row-major 6x6 diagonal matrix for impedance parameters."""

    return np.diag(values).flatten().tolist()


class InsertionFailed(RuntimeError):
    """Raised when a phase cannot complete; the message says why."""


class PortInsertion(Policy):
    """Perception-guided, force-guided insertion engine used by ``PerceptionInsert``."""

    def __init__(self, parent_node: Any):
        """Load settings and subscribe to scene estimates.

        Args:
            parent_node: The ``aic_model`` node hosting the policy.
        """

        super().__init__(parent_node)
        config = load_config("infer")
        self.settings = config["insertion"]
        self.servo_settings = config["refiner"]
        self.sc_ranks = config["sc_rank_for_slot"]
        self.grasps = config["plug_grasp"]
        self.trial_log = Path(config["logging"]["trial_directory"]) / time.strftime("trials_%Y%m%d_%H%M%S.jsonl")
        self.record: dict[str, Any] = {}
        self.started = 0.0
        self.spiral_radius = self.settings["spiral_max_radius_m"]
        self.target_id = ""
        self.last_image_stamp = 0.0
        self.refiner = None
        checkpoint = Path(config["refiner"]["checkpoint"])
        if not config["refiner"]["enabled"]:
            self.get_logger().info("Close-range refiner disabled (refiner.enabled); using the scene estimator alone")
        elif checkpoint.is_file():
            from aic_perception.refinement import RefinerRuntime

            self.refiner = RefinerRuntime(checkpoint, config["refiner"]["device"])
            self.get_logger().info(f"Close-range refiner loaded from {checkpoint}")
        else:
            self.get_logger().warning(
                f"No refiner checkpoint at {checkpoint}; falling back to re-estimating with the scene estimator"
            )
        self.estimates: list[SceneEstimate] = []
        self.tcp_from_plug = np.eye(4)
        self.correction = np.zeros(3)
        self.deadline = float("inf")
        self.bias = np.zeros(3)
        self.port_xy = np.zeros(2)
        parent_node.create_subscription(
            SceneEstimate,
            config["inference"]["estimate_topic"],
            self.estimates.append,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

    # Sensing ---------------------------------------------------------------

    def now(self) -> float:
        """Simulation time in seconds."""

        return stamp_seconds(self.time_now().to_msg())

    def lookup(self, target: str, source: str, stamp: Any = None) -> np.ndarray | None:
        """``target_from_source`` from TF (latest, or at ``stamp``), or ``None`` when unavailable."""

        try:
            if stamp is None:
                message = self._parent_node._tf_buffer.lookup_transform(target, source, Time())
            else:
                message = self._parent_node._tf_buffer.lookup_transform(
                    target, source, Time.from_msg(stamp), Duration(seconds=0.5)
                )
        except TransformException:
            return None
        t, r = message.transform.translation, message.transform.rotation
        return from_xyz_quaternion_xyzw((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def plug_pose(self) -> np.ndarray:
        """Plug pose in ``base_link`` from the robot TCP and the grasp calibration."""

        tcp = self.lookup("base_link", "gripper/tcp")
        if tcp is None:
            raise InsertionFailed("gripper/tcp is not in TF")
        return tcp @ self.tcp_from_plug

    def force(self, get_observation: GetObservationCallback) -> np.ndarray:
        """Wrist force in ``base_link`` from the latest observation.

        Args:
            get_observation: Observation callback from ``aic_model``.

        Returns:
            Force vector in newtons.
        """

        observation = get_observation()
        if observation is None:
            raise InsertionFailed("No observation available")
        wrench = observation.wrist_wrench
        frame = wrench.header.frame_id or DEFAULT_SENSOR_FRAME
        rotation = self.lookup("base_link", frame)
        if rotation is None:
            raise InsertionFailed(f"{frame} is not in TF")
        f = wrench.wrench.force
        return rotation[:3, :3] @ np.array([f.x, f.y, f.z])

    def baseline(self, get_observation: GetObservationCallback) -> np.ndarray:
        """Average force while not in contact; the sensor cannot be tared in evaluation."""

        samples = []
        for _ in range(self.settings["baseline_samples"]):
            samples.append(self.force(get_observation))
            self.sleep_for(self.settings["control_period_s"])
        return np.mean(samples, axis=0)

    def perceived_port(self, target_id: str, timeout: float | None = None) -> np.ndarray:
        """Median of fresh present estimates of one target, with level orientation.

        Args:
            target_id: Schema target ID.
            timeout: Seconds to wait; defaults to ``estimate_timeout_s``.

        Returns:
            Entrance pose in ``base_link``.
        """

        started = self.now()
        self.estimates.clear()
        poses = []
        waited = 0.0
        while len(poses) < self.settings["estimate_count"]:
            if waited > (self.settings["estimate_timeout_s"] if timeout is None else timeout):
                raise InsertionFailed(f"No present estimate of {target_id}; is run_inference.py running?")
            while self.estimates:
                message = self.estimates.pop(0)
                if stamp_seconds(message.header.stamp) < started:
                    continue
                for target in message.targets:
                    if target.id == target_id and target.present:
                        p, q = target.pose.position, target.pose.orientation
                        poses.append(from_xyz_quaternion_xyzw((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
                        self.get_logger().info(
                            f"Estimate {len(poses)}: presence {target.presence_probability:.3f}, "
                            f"confidence {target.confidence:.3f}"
                        )
            self.sleep_for(0.1)
            waited += 0.1
        position = np.median([pose[:3, 3] for pose in poses], axis=0)
        return transform(level_rotation(poses[-1][:3, :3]), position)

    def visible_targets(self, duration: float) -> dict[str, np.ndarray]:
        """Every target present in the fresh estimates of the next ``duration`` seconds.

        Returns:
            Target ID mapped to its pose in ``base_link`` (median position,
            level orientation of the latest estimate).
        """

        started = self.now()
        self.estimates.clear()
        poses: dict[str, list[np.ndarray]] = {}
        while self.now() - started < duration:
            while self.estimates:
                message = self.estimates.pop(0)
                if stamp_seconds(message.header.stamp) < started:
                    continue
                for target in message.targets:
                    if target.present:
                        p, q = target.pose.position, target.pose.orientation
                        pose = from_xyz_quaternion_xyzw((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
                        poses.setdefault(target.id, []).append(pose)
            self.sleep_for(0.1)
        return {
            target_id: transform(level_rotation(found[-1][:3, :3]), np.median([pose[:3, 3] for pose in found], axis=0))
            for target_id, found in poses.items()
        }

    def expected_port(self, target_id: str, seen: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        """Where the unseen target port should be, from what the start pose does see.

        Position: from the sibling port on the same card when it is seen
        (the two differ by a fixed offset on the card), else from the board
        estimate and the target's ``search_anchors`` entry. Orientation: that
        of the nearest seen SFP port (all SFP ports face the same way up to the
        card yaw), so the viewpoint looks at the card as the hover pose would,
        else the plug's current orientation.

        Returns:
            Expected entrance pose in ``base_link`` and the source of its position.
        """

        anchors = self.settings["search_anchors"]
        anchor = np.array(anchors[target_id])
        board = seen["task_board"] if "task_board" in seen else self.perceived_port("task_board")
        module, port_name = target_id.split("/")
        sibling = f"{module}/{'sfp_port_1' if port_name == 'sfp_port_0' else 'sfp_port_0'}"
        if sibling in seen and sibling in anchors:
            position = seen[sibling][:3, 3] + board[:3, :3] @ (anchor - np.array(anchors[sibling]))
            source = "sibling port"
        else:
            position = (board @ np.append(anchor, 1.0))[:3]
            source = "board"
        ports = [pose for name, pose in seen.items() if name in anchors]
        if ports:
            nearest = min(ports, key=lambda pose: float(np.linalg.norm(pose[:2, 3] - position[:2])))
            rotation = nearest[:3, :3]
        else:
            rotation = self.plug_pose()[:3, :3]
        return transform(rotation, position), source

    def find_port(self, move_robot: MoveRobotCallback, task: Task, target_id: str) -> np.ndarray:
        """Perceive the target port, moving closer over the board when it is not seen at first.

        If the start pose yields no present estimate within ``search_after_s``,
        the port's expected pose is inferred from what is seen (``expected_port``)
        and the plug visits ``search_viewpoints`` (shift along the board's slot
        axis, height above the expected entrance). Each viewpoint is reached
        through ``search_transit_height_m`` so the plug turns and crosses the
        board high, then drops straight down; it faces the card as at the
        hover pose. A detection farther than ``search_max_offset_m`` from the
        expected position is rejected as a wrong match.

        Returns:
            Entrance pose in ``base_link``.

        Raises:
            InsertionFailed: If no viewpoint yields a plausible detection.
        """

        settings = self.settings
        anchor = settings["search_anchors"].get(target_id)
        try:
            return self.perceived_port(target_id, settings["search_after_s"] if anchor else None)
        except InsertionFailed:
            if anchor is None:
                raise
        seen = self.visible_targets(settings["search_snapshot_s"])
        expected, source = self.expected_port(target_id, seen)
        board_x = (seen["task_board"] if "task_board" in seen else self.perceived_port("task_board"))[:3, 0]
        self.get_logger().info(f"{target_id} not seen; expected from the {source}, searching")
        self.log_truth(f"search expectation ({source})", task, port=expected)
        for index, (shift, height) in enumerate(settings["search_viewpoints"], start=1):
            point = expected[:3, 3] + shift * board_x
            transit = transform(expected[:3, :3], point + np.array([0.0, 0.0, settings["search_transit_height_m"]]))
            self.move_plug(move_robot, transit, settings["search_move_s"], mode="descent")
            goal = transform(expected[:3, :3], point + np.array([0.0, 0.0, height]))
            self.move_plug(move_robot, goal, settings["search_move_s"], mode="descent")
            try:
                port = self.perceived_port(target_id, settings["search_timeout_s"])
            except InsertionFailed:
                self.get_logger().info(f"Search viewpoint {index}: not seen")
                continue
            offset = float(np.linalg.norm(port[:2, 3] - expected[:2, 3]))
            if offset > settings["search_max_offset_m"]:
                self.get_logger().info(f"Search viewpoint {index}: rejected a detection {1000 * offset:.1f} mm from expected")
                continue
            self.note("search", viewpoint=index, source=source, expected=expected[:3, 3], port=port[:3, 3])
            self.log_truth(f"found at search viewpoint {index}", task, port=port)
            return port
        raise InsertionFailed(f"{target_id} not seen from the start pose or {index} search viewpoints")

    def resolve_target(self, task: Task) -> str:
        """Schema target ID of the task's port.

        SFP ports are named as in the task. SC ports are reported by rank
        along their rail: with one present it is the target; with two,
        ``sc_rank_for_slot`` maps the task's slot onto a rank.

        Raises:
            InsertionFailed: If no SC port is present in the estimates.
        """

        if task.port_type != "sc":
            return f"{task.target_module_name}/{task.port_name}"
        started = self.now()
        self.estimates.clear()
        votes = dict.fromkeys(SC_RANK_IDS, 0)
        messages = 0
        deadline = started + self.settings["estimate_timeout_s"]
        while messages < self.settings["estimate_count"]:
            if self.now() > deadline:
                raise InsertionFailed("No scene estimates; is run_inference.py running?")
            while self.estimates:
                message = self.estimates.pop(0)
                if stamp_seconds(message.header.stamp) < started:
                    continue
                messages += 1
                for target in message.targets:
                    if target.id in votes and target.present:
                        votes[target.id] += 1
            self.sleep_for(0.1)
        present = [target_id for target_id, count in votes.items() if 2 * count > messages]
        if not present:
            raise InsertionFailed(f"No SC port is present in the estimates (votes {votes})")
        if len(present) == 1:
            target_id = present[0]
        else:
            target_id = SC_RANK_IDS[self.sc_ranks[task.target_module_name]]
            self.get_logger().warning(
                f"Two SC ports present; {task.target_module_name} assumed to be {target_id} (sc_rank_for_slot)"
            )
        self.note("target", target_id=target_id, sc_votes=votes, sc_present=len(present))
        return target_id

    def measure(self, get_observation: GetObservationCallback, plug_type: str) -> dict[str, np.ndarray]:
        """Run the refiner on the next fresh observation.

        Returns:
            Refiner output in ``base_link`` plus ``tcp``, the TCP pose at the
            image stamp.

        Raises:
            InsertionFailed: If no fresh observation or camera pose arrives.
        """

        deadline = time.monotonic() + self.servo_settings["observation_timeout_s"]
        while True:
            observation = get_observation()
            if observation is not None:
                stamp = observation.center_image.header.stamp
                if stamp_seconds(stamp) > self.last_image_stamp:
                    break
            if time.monotonic() > deadline:
                raise InsertionFailed("No fresh observation for the refiner")
            self.sleep_for(0.02)
        self.last_image_stamp = stamp_seconds(stamp)
        images, poses = [], []
        for name in CAMERAS:
            image = getattr(observation, f"{name}_image")
            images.append(np.frombuffer(image.data, np.uint8).reshape(image.height, image.width, 3))
            pose = self.lookup("base_link", image.header.frame_id, stamp)
            if pose is None:
                raise InsertionFailed(f"No camera pose for {image.header.frame_id} at the image stamp")
            poses.append(pose)
        tcp = self.lookup("base_link", "gripper/tcp", stamp)
        if tcp is None:
            raise InsertionFailed("No TCP pose at the image stamp")
        return {**self.refiner.predict(images, poses, plug_type), "tcp": tcp}

    def wait_until_still(self, timeout: float, dwell: float) -> float:
        """Wait until the TCP moves less than ``still_tolerance_m`` per 0.1 s, then ``dwell`` more.

        Args:
            timeout: Longest wait for the arm to stop, in seconds.
            dwell: Extra wait after it stopped, for the plug to stop swinging
                in the gripper.

        Returns:
            TCP motion over the last check window in meters.
        """

        deadline = self.now() + timeout
        previous = self.plug_pose()[:3, 3]
        motion = float("inf")
        while self.now() < deadline:
            self.sleep_for(0.1)
            current = self.plug_pose()[:3, 3]
            motion = float(np.linalg.norm(current - previous))
            previous = current
            if motion < self.settings["still_tolerance_m"]:
                break
        if dwell > 0.0:
            self.sleep_for(dwell)
        return motion

    def servo_measurement(self, get_observation: GetObservationCallback, plug_type: str) -> dict[str, Any]:
        """Median of ``refiner.samples`` refiner readings that agree with each other.

        Each reading gives the port from the image alone (the refiner's plug
        estimate plus its offset), so a plug that moved in the gripper does
        not shift the port estimate. When the readings spread more than
        ``consistency_m``, the policy waits for the scene to settle and
        measures again, up to ``measure_retries`` times.

        Returns:
            Median ``port``, ``offset``, ``plug``, and yaw, the TCP pose at
            the last image, and the spread in meters.
        """

        settings = self.servo_settings
        for attempt in range(settings["measure_retries"] + 1):
            self.wait_until_still(settings["max_settle_s"], settings["settle_time_s"])
            samples = [self.measure(get_observation, plug_type) for _ in range(settings["samples"])]
            ports = np.asarray([m["port"] for m in samples])
            port = np.median(ports, axis=0)
            spread = float(np.max(np.linalg.norm(ports[:, :2] - port[:2], axis=1)))
            if spread <= settings["consistency_m"]:
                break
            self.get_logger().info(f"Servo: readings spread {1000 * spread:.2f} mm; measuring again")
        return {
            "port": port,
            "offset": np.median([m["offset"] for m in samples], axis=0),
            "plug": np.median([m["plug"] for m in samples], axis=0),
            "yaw": float(np.median([m["rotation"][2] for m in samples])),
            "tcp": samples[-1]["tcp"],
            "spread": spread,
        }

    def servo(
        self, move_robot: MoveRobotCallback, get_observation: GetObservationCallback, task: Task, port: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Align the plug over the port with the close-range refiner.

        Each iteration measures the port from the wrist images (see
        ``servo_measurement``), then moves the plug onto it at the
        pre-contact height under the well-damped ``servo`` impedance and turns
        it by the measured yaw. The first measurement also updates the grasp
        translation (``update_grasp``), so plug heights and seat depths use
        the plug actually held. Offsets larger than ``max_offset_m`` lie
        outside what the refiner was trained on; the policy then keeps the
        scene estimate instead of acting on them.

        Args:
            port: Current entrance estimate (scene estimator).

        Returns:
            Entrance pose, the pre-contact plug goal, and whether the plug
            ended within ``tolerance_m`` laterally.
        """

        settings = self.servo_settings
        height = self.settings["pre_contact_height_m"]
        goal = port.copy()
        goal[2, 3] += height
        refined, lateral = port, float("inf")
        for iteration in range(settings["iterations"] + 1):
            reading = self.servo_measurement(get_observation, task.plug_type)
            offset, yaw, tcp = reading["offset"], reading["yaw"], reading["tcp"]
            lateral = float(np.linalg.norm(offset[:2]))
            if lateral > settings["max_offset_m"]:
                self.get_logger().warning(
                    f"Servo {iteration}: measured offset {1000 * lateral:.1f} mm is beyond the refiner's range; "
                    "keeping the scene estimate"
                )
                self.note("servo", iteration=iteration, offset_mm=1000 * offset, rejected=True)
                return refined, goal, False
            if iteration == 0 and settings["update_grasp"]:
                measured = invert(tcp) @ np.append(reading["plug"], 1.0)
                change = np.linalg.norm(measured[:3] - self.tcp_from_plug[:3, 3])
                if change <= settings["max_grasp_change_m"]:
                    self.tcp_from_plug[:3, 3] = measured[:3]
                    self.note("grasp_update", change_mm=1000 * change, tcp_from_plug=measured[:3])
                else:
                    self.get_logger().warning(f"Servo: ignored an implausible {1000 * change:.1f} mm grasp change")
            refined = port.copy()
            refined[:3, 3] = reading["port"]
            self.get_logger().info(
                f"Servo {iteration}: port {1000 * offset[0]:+.2f}, {1000 * offset[1]:+.2f} mm sideways, "
                f"{1000 * offset[2]:+.2f} mm vertically, yaw {np.degrees(yaw):+.2f} deg from the plug "
                f"(readings within {1000 * reading['spread']:.2f} mm)"
            )
            truth = self.log_truth(f"servo {iteration}", task, plug=tcp @ self.tcp_from_plug, port=refined)
            self.note(
                "servo", iteration=iteration, offset_mm=1000 * offset, yaw_deg=np.degrees(yaw),
                spread_mm=1000 * reading["spread"], truth=truth,
            )
            if lateral < settings["tolerance_m"] and abs(np.degrees(yaw)) < settings["min_yaw_deg"]:
                return refined, goal, True
            if iteration == settings["iterations"]:
                break
            plug = reading["plug"]
            step = offset[:2] * min(1.0, settings["max_step_m"] / max(lateral, 1e-9))
            turn = np.clip(np.degrees(yaw), -settings["max_yaw_deg"], settings["max_yaw_deg"])
            turn = turn if abs(turn) >= settings["min_yaw_deg"] else 0.0
            current = tcp @ self.tcp_from_plug
            rotation = level_rotation(from_xyz_rpy((0, 0, 0), (0.0, 0.0, np.radians(turn)))[:3, :3] @ current[:3, :3])
            goal = transform(rotation, np.array([plug[0] + step[0], plug[1] + step[1], reading["port"][2] + height]))
            self.move_plug(move_robot, goal, settings["move_duration_s"], mode="servo")
        return refined, goal, lateral < settings["tolerance_m"]

    # Trial record ---------------------------------------------------------------

    def note(self, phase: str, **values: Any) -> None:
        """Append one phase entry to the current trial record."""

        self.record.setdefault("phases", []).append({"phase": phase, "time_s": round(self.now() - self.started, 3), **rounded(values)})

    def write_record(self) -> None:
        """Append the finished trial record to the trial log."""

        try:
            self.trial_log.parent.mkdir(parents=True, exist_ok=True)
            with self.trial_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rounded(self.record)) + "\n")
            self.get_logger().info(f"Trial record appended to {self.trial_log}")
        except (OSError, TypeError, ValueError) as error:
            self.get_logger().error(f"Could not write the trial record: {error}")

    # Commanding --------------------------------------------------------------

    def send(self, move_robot: MoveRobotCallback, plug_goal: np.ndarray, mode: str) -> None:
        """Command the TCP so the plug reaches ``plug_goal`` (plus tracking correction).

        Args:
            move_robot: Motion callback from ``aic_model``.
            plug_goal: Desired plug pose in ``base_link``.
            mode: Impedance set: ``free``, ``descent``, ``search``, or
                ``recovery`` (soft rotation for snags).
        """

        goal = plug_goal.copy()
        goal[:3, 3] += self.correction
        tcp = tcp_target(goal, self.tcp_from_plug)
        x, y, z, w = quaternion_xyzw(tcp)
        move_robot(
            motion_update=MotionUpdate(
                header=Header(frame_id="base_link", stamp=self._parent_node.get_clock().now().to_msg()),
                pose=Pose(
                    position=Point(x=float(tcp[0, 3]), y=float(tcp[1, 3]), z=float(tcp[2, 3])),
                    orientation=Quaternion(x=x, y=y, z=z, w=w),
                ),
                target_stiffness=diagonal(self.settings[f"{mode}_stiffness"]),
                target_damping=diagonal(self.settings[f"{mode}_damping"]),
                feedforward_wrench_at_tip=Wrench(force=Vector3(), torque=Vector3()),
                wrench_feedback_gains_at_tip=list(self.settings[f"{mode}_feedback_gains"]),
                trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION),
            )
        )

    def step(self) -> None:
        """Wait one control period and enforce the task time limit."""

        self.sleep_for(self.settings["control_period_s"])
        if self.now() > self.deadline:
            raise InsertionFailed("Task time limit reached")

    def move_plug(
        self, move_robot: MoveRobotCallback, goal: np.ndarray, duration: float, mode: str = "free"
    ) -> None:
        """Interpolate the plug to ``goal``, then settle on it.

        Position is interpolated linearly and orientation by slerp. The
        compliant controller stops short of its goal under the cable load, so
        after the move the remaining error is corrected, but only once the arm
        has come to rest: up to ``settle_corrections`` times, the commanded
        offset grows by ``tracking_gain`` times the resting error, bounded by
        ``max_correction_m``. Correcting while the arm still moves (an
        integrator running against a lagging arm) winds up and swings the plug
        past the goal and back.

        Args:
            move_robot: Motion callback.
            goal: Desired plug pose in ``base_link``.
            duration: Interpolation time in seconds.
            mode: Impedance set; settling under the descent set makes the
                learned correction match the stiffness of the contact phases.
        """

        start = self.plug_pose()
        start_q, goal_q = mat2quat(start[:3, :3]), mat2quat(goal[:3, :3])
        steps = max(1, round(duration / self.settings["control_period_s"]))
        for index in range(1, steps + 1):
            fraction = index / steps
            rotation = quat2mat(quaternion_slerp(start_q, goal_q, fraction))
            position = (1.0 - fraction) * start[:3, 3] + fraction * goal[:3, 3]
            self.send(move_robot, transform(rotation, position), mode)
            self.step()
        settings = self.settings
        for _ in range(settings["settle_corrections"]):
            self.wait_until_still(settings["still_timeout_s"], 0.0)
            error = goal[:3, 3] - self.plug_pose()[:3, 3]
            if np.linalg.norm(error) < settings["position_tolerance_m"]:
                return
            correction = self.correction + settings["tracking_gain"] * error
            self.correction = correction * min(1.0, settings["max_correction_m"] / max(float(np.linalg.norm(correction)), 1e-9))
            self.send(move_robot, goal, mode)
        self.wait_until_still(settings["still_timeout_s"], 0.0)

    # Phases ------------------------------------------------------------------

    def track_approach(self, move_robot: MoveRobotCallback, task: Task, port: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approach the port continuously while folding in every new estimate of it.

        The port estimate is a running average of the scene estimator's
        stream (``track_alpha``), with estimates farther than ``track_gate_m``
        rejected as outliers and a reset after ``track_reset_after``
        rejections in a row. Every control period the commanded plug moves
        toward a point above the current estimate with bounded speed and
        acceleration; the point lies on a 45 degree funnel, so the plug only
        descends once it is over the port. After arriving it keeps following
        for ``track_dwell_s`` to absorb the closest estimates, then settles at
        rest on the pre-contact pose.

        Args:
            port: Start-pose estimate of the entrance.

        Returns:
            Tracked entrance pose and the pre-contact plug goal.
        """

        settings = self.settings
        height = settings["pre_contact_height_m"]
        tracker = PortAverage(port[:3, 3], settings["track_alpha"], settings["track_gate_m"], settings["track_reset_after"])
        rotation = port[:3, :3]
        start = self.plug_pose()
        command = start[:3, 3].copy()
        start_q, goal_q = mat2quat(start[:3, :3]), mat2quat(rotation)
        initial = max(float(np.linalg.norm(funnel_target(tracker.position, command, height) - command)), 1e-6)
        period = settings["control_period_s"]
        speed, arrived_at = 0.0, None
        started = self.now()
        self.estimates.clear()
        while self.now() - started < settings["track_timeout_s"]:
            while self.estimates:
                message = self.estimates.pop(0)
                if stamp_seconds(message.header.stamp) < started:
                    continue
                for target in message.targets:
                    if target.id == self.target_id and target.present:
                        p = target.pose.position
                        tracker.update(np.array([p.x, p.y, p.z]))
            goal = funnel_target(tracker.position, command, height)
            command, speed = speed_limited_step(
                command, goal, speed, settings["track_speed_m_s"], settings["track_accel_m_s2"], period
            )
            remaining = float(np.linalg.norm(goal - command))
            fraction = min(1.0, max(0.0, 1.0 - remaining / initial))
            self.send(move_robot, transform(quat2mat(quaternion_slerp(start_q, goal_q, fraction)), command), "descent")
            self.step()
            if remaining == 0.0 and arrived_at is None:
                arrived_at = self.now()
            if arrived_at is not None and self.now() - arrived_at >= settings["track_dwell_s"]:
                break
        port = transform(rotation, tracker.position)
        pre_contact = port.copy()
        pre_contact[2, 3] += height
        self.get_logger().info(
            f"Tracking approach: {tracker.accepted} estimates used, {tracker.rejected} rejected, "
            f"{tracker.resets} resets, {self.now() - started:.1f} s"
        )
        self.note(
            "track_approach", accepted=tracker.accepted, rejected=tracker.rejected, resets=tracker.resets,
            port=tracker.position, duration_s=self.now() - started, arrived=arrived_at is not None,
        )
        self.log_truth("tracked port", task, port=port)
        self.move_plug(move_robot, pre_contact, settings["track_final_move_s"], mode="descent")
        return port, pre_contact

    def refine(
        self, move_robot: MoveRobotCallback, task: Task, port: np.ndarray, pre_contact: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-estimate the port from the coarse hover and move to pre-contact over it.

        The estimator runs continuously, so fresh estimates at
        ``coarse_height_m`` see the port from much closer than the start pose
        (validation median about 1.6 mm up close versus 2 mm from HOME, and
        far fewer outliers). Shifts larger than ``max_refine_shift_m`` are
        treated as implausible and ignored.

        Returns:
            Port entrance pose and pre-contact plug goal to continue with.
        """

        try:
            refined = self.perceived_port(self.target_id, self.settings["refine_timeout_s"])
        except InsertionFailed:
            self.get_logger().info("Refine: no present estimate at the coarse hover; keeping the start estimate")
            self.move_plug(move_robot, pre_contact, 2.0, mode="descent")
            return port, pre_contact
        self.log_truth("re-estimate at coarse hover", task, port=refined)
        shift = np.linalg.norm(refined[:2, 3] - port[:2, 3])
        if shift > self.settings["max_refine_shift_m"]:
            self.get_logger().info(f"Refine: rejected a {1000 * shift:.1f} mm shift as implausible")
            self.move_plug(move_robot, pre_contact, 2.0, mode="descent")
            return port, pre_contact
        self.get_logger().info(
            f"Refine: shifting {1000 * shift:.1f} mm sideways, {1000 * (refined[2, 3] - port[2, 3]):+.1f} mm in height"
        )
        goal = refined.copy()
        goal[2, 3] += self.settings["pre_contact_height_m"]
        self.move_plug(move_robot, goal, 2.0, mode="descent")
        self.log_truth("refined pre-contact", task, plug=self.plug_pose())
        return refined, goal

    def ramp_descend(
        self,
        move_robot: MoveRobotCallback,
        get_observation: GetObservationCallback,
        start: np.ndarray,
        depth: float,
        speed: float,
        mode: str,
        duration: float,
        stop_force: float | None,
    ) -> tuple[np.ndarray, str]:
        """Lower the commanded plug at constant speed until something stops it.

        Args:
            start: Commanded plug pose to descend from.
            depth: Maximum commanded descent in meters.
            speed: Descent speed in meters per second.
            mode: Impedance set.
            duration: Time limit in seconds.
            stop_force: Contact force that ends the descent, or ``None``.

        Returns:
            Measured plug pose and the reason: ``force``, ``stall``, or ``timeout``.
        """

        period = self.settings["control_period_s"]
        window = max(2, round(self.settings["stall_window_s"] / period))
        grace = round(self.settings["stall_grace_s"] / period)
        heights: list[float] = []
        goal = start.copy()
        for index in range(1, round(duration / period) + 1):
            goal[2, 3] = start[2, 3] - min(speed * period * index, depth)
            self.send(move_robot, goal, mode)
            self.step()
            plug = self.plug_pose()
            load = self.load(get_observation)
            if stop_force is not None and load >= stop_force:
                return plug, "force"
            if load >= self.settings["max_force_n"]:
                return plug, "force"
            self.check_drift(plug)
            heights.append(plug[2, 3])
            recent = heights[-window:]
            if index > grace and len(recent) == window and max(recent) - min(recent) < self.settings["stall_epsilon_m"]:
                return plug, "stall"
        return self.plug_pose(), "timeout"

    def check_drift(self, plug: np.ndarray) -> None:
        """Abort when the plug has left the neighbourhood of the perceived port.

        Falling off the edge of a card looks like a deep insertion in height
        alone, so every contact phase also bounds the lateral distance.

        Raises:
            InsertionFailed: If the plug is farther than ``max_lateral_drift_m``.
        """

        drift = np.linalg.norm(plug[:2, 3] - self.port_xy)
        if drift > self.settings["max_lateral_drift_m"]:
            raise InsertionFailed(f"Plug drifted {1000 * drift:.1f} mm sideways from the perceived port")

    def load(self, get_observation: GetObservationCallback) -> float:
        """Contact force magnitude relative to the pre-contact baseline."""

        return float(np.linalg.norm(self.force(get_observation) - self.bias))

    def spiral_search(
        self, move_robot: MoveRobotCallback, get_observation: GetObservationCallback, contact: np.ndarray, entry_z: float
    ) -> tuple[bool, np.ndarray]:
        """Archimedean spiral pressed into the surface until the plug drops.

        Args:
            contact: Plug pose on the surface where the search starts.
            entry_z: Plug height that proves entry.

        Returns:
            Whether the plug entered, and its final pose.
        """

        center = contact.copy()
        center[2, 3] -= self.settings["press_margin_m"]
        angle = 0.0
        plug = self.plug_pose()
        while True:
            offset = spiral_point(angle, self.settings["spiral_pitch_m"])
            if np.linalg.norm(offset) > self.spiral_radius:
                self.get_logger().info("Spiral: reached the maximum radius without entry")
                self.note("spiral", entered=False, radius_mm=1000 * self.spiral_radius, turns=angle / (2 * np.pi))
                return False, plug
            goal = center.copy()
            goal[:2, 3] += offset
            self.send(move_robot, goal, "search")
            self.step()
            plug = self.plug_pose()
            self.check_drift(plug)
            if plug[2, 3] < entry_z:
                self.get_logger().info(
                    f"Spiral: entered at radius {1000 * np.linalg.norm(offset):.1f} mm after "
                    f"{angle / (2 * np.pi):.1f} turns"
                )
                self.note("spiral", entered=True, radius_mm=1000 * np.linalg.norm(offset), turns=angle / (2 * np.pi))
                return True, plug
            angle = spiral_step(
                angle, self.settings["spiral_pitch_m"], self.settings["spiral_speed_m_s"], self.settings["control_period_s"]
            )

    def press_until_seated(
        self, move_robot: MoveRobotCallback, get_observation: GetObservationCallback, entry: np.ndarray, reference_z: float, plug_type: str
    ) -> float:
        """Press with soft axial impedance until seated, recovering from snags.

        Args:
            entry: Plug pose at the start of the press.
            reference_z: Height the seat depth is measured from.
            plug_type: Selects the seat and extra depth.

        Returns:
            Final depth below ``reference_z`` in meters.
        """

        seat = self.settings["min_seat_depth_m"][plug_type]
        extra = self.settings["insert_extra_depth_m"][plug_type]
        start = entry.copy()
        mode, duration = "search", self.settings["insert_max_duration_s"]
        for attempt in range(self.settings["recovery_attempts"] + 1):
            plug, reason = self.ramp_descend(
                move_robot, get_observation, start, extra, self.settings["insert_speed_m_s"], mode, duration, None
            )
            depth = reference_z - plug[2, 3]
            if depth > self.settings["max_seat_depth_m"][plug_type]:
                raise InsertionFailed(f"Plug went {1000 * depth:.1f} mm deep, deeper than the port: not in the port")
            if depth >= seat:
                self.get_logger().info(f"Seated at {1000 * depth:.1f} mm ({reason})")
                return depth
            if attempt == self.settings["recovery_attempts"]:
                break
            self.get_logger().info(f"Snag {attempt + 1} at {1000 * depth:.1f} mm ({reason}); retrying with soft rotation")
            mode, duration = "recovery", self.settings["recovery_duration_s"]
            start = plug.copy()
            if attempt + 1 == self.settings["recovery_attempts_before_spiral"]:
                lifted = plug.copy()
                lifted[2, 3] += self.settings["recovery_retract_m"]
                self.move_plug(move_robot, lifted, 1.0, mode="descent")
                surface, _ = self.ramp_descend(
                    move_robot, get_observation, lifted, 2 * self.settings["recovery_retract_m"],
                    self.settings["descent_speed_m_s"], "descent", self.settings["insert_max_duration_s"],
                    self.settings["contact_force_n"],
                )
                _, start = self.spiral_search(
                    move_robot, get_observation, surface, plug[2, 3] - self.settings["unstick_margin_m"]
                )
        self.get_logger().info(f"Not seated: {1000 * depth:.1f} mm of {1000 * seat:.1f} mm")
        return depth

    # Evaluation logging ------------------------------------------------------------

    def log_truth(
        self, label: str, task: Task, plug: np.ndarray | None = None, port: np.ndarray | None = None
    ) -> dict[str, float]:
        """Log errors against simulator ground truth when it is published.

        Never feeds back into control; skipped when ground truth is absent.

        Args:
            label: Phase name for the log line.
            task: Current task (names the true port and plug frames).
            plug: The policy's own plug pose estimate, if any.
            port: The policy's own port entrance estimate, if any.

        Returns:
            Errors in mm and degrees (empty without ground truth); also
            stored in the trial record.
        """

        if not self.settings["log_ground_truth"]:
            return {}
        entrance = self.lookup("base_link", f"task_board/{task.target_module_name}/{task.port_name}_link_entrance")
        if entrance is None:
            return {}
        values: dict[str, float] = {}
        if port is not None:
            values["port_error_mm"] = 1000 * float(np.linalg.norm(port[:3, 3] - entrance[:3, 3]))
            values["port_lateral_error_mm"] = 1000 * float(np.linalg.norm(port[:2, 3] - entrance[:2, 3]))
        true_plug = self.lookup("base_link", f"{task.cable_name}/{task.plug_name}_link")
        if true_plug is not None:
            offset = true_plug[:3, 3] - entrance[:3, 3]
            values["plug_off_axis_mm"] = 1000 * float(np.linalg.norm(offset[:2]))
            values["plug_above_entrance_mm"] = 1000 * float(offset[2])
            relative = entrance[:3, :3].T @ true_plug[:3, :3]
            values["plug_rotation_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
            if plug is not None:
                values["plug_estimate_error_mm"] = 1000 * float(np.linalg.norm(plug[:3, 3] - true_plug[:3, 3]))
        text = {
            "port_error_mm": "perceived port error {:.2f} mm",
            "port_lateral_error_mm": "({:.2f} mm sideways)",
            "plug_off_axis_mm": "true plug {:.2f} mm off axis,",
            "plug_above_entrance_mm": "{:+.2f} mm from entrance;",
            "plug_rotation_deg": "true plug rotation {:.2f} deg from port;",
            "plug_estimate_error_mm": "plug estimate error {:.2f} mm",
        }
        self.get_logger().info(f"[ground truth] {label}: " + " ".join(text[k].format(v) for k, v in values.items()))
        self.note(f"truth {label}", **values)
        return values

    # Task ---------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Run the phases for one task and append its trial record.

        Args:
            task: AIC task naming the plug, module, and port.
            get_observation: Observation callback (images and wrist wrench).
            move_robot: Motion callback.
            send_feedback: Unused.

        Returns:
            Whether the plug was seated (always a plain ``bool``; ``aic_model``
            copies it into a ROS field that rejects ``numpy.bool_``).
        """

        self.get_logger().info(f"{type(self).__name__} task: {task}")
        self.started = self.now()
        self.deadline = self.started + task.time_limit - TIME_MARGIN_S
        self.correction = np.zeros(3)
        self.spiral_radius = self.settings["spiral_max_radius_m"]
        self.record = {
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "policy": type(self).__name__,
            "refiner": self.refiner is not None,
            "task": {
                key: getattr(task, key)
                for key in ("cable_name", "plug_type", "plug_name", "port_type", "port_name", "target_module_name", "time_limit")
            },
            "phases": [],
        }
        seated, failure = False, None
        try:
            seated = self.run_phases(task, get_observation, move_robot)
        except InsertionFailed as error:
            failure = str(error)
            self.get_logger().error(f"{type(self).__name__} stopped: {error}")
        finally:
            self.record |= {"success": seated, "failure": failure, "duration_s": round(self.now() - self.started, 2)}
            self.write_record()
        return bool(seated)

    def run_phases(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback) -> bool:
        """Approach, servo, descend, search, and press; raises ``InsertionFailed`` on failure.

        Returns:
            Whether the plug was seated (or, for the far approach, reached
            the hover pose).
        """

        grasp = self.grasps.get(task.plug_type)
        if grasp is None:
            raise InsertionFailed(f"No plug_grasp calibration for '{task.plug_type}'; run CalibratePlugGrasp")
        self.tcp_from_plug = from_xyz_quaternion_xyzw(tuple(grasp["translation"]), tuple(grasp["quaternion_xyzw"]))
        settings = self.settings
        target_id = self.resolve_target(task)
        self.target_id = target_id
        port = self.find_port(move_robot, task, target_id)
        self.note("perception", target_id=target_id, port=port[:3, 3])
        self.log_truth("perception", task, port=port)
        self.port_xy = port[:2, 3].copy()

        if settings["tracking_approach"]:
            port, pre_contact = self.track_approach(move_robot, task, port)
        else:
            # Staged: straight to a coarse hover above the port (the entrance is
            # the highest point of its card, so the path stays clear),
            # re-estimate there, then down to pre-contact over the new estimate.
            coarse = port.copy()
            coarse[2, 3] += settings["coarse_height_m"]
            self.move_plug(move_robot, coarse, settings["approach_duration_s"], mode="descent")
            self.log_truth("coarse hover", task, plug=self.plug_pose())
            pre_contact = port.copy()
            pre_contact[2, 3] += settings["pre_contact_height_m"]
            if settings["refine_at_pre_contact"]:
                port, pre_contact = self.refine(move_robot, task, port, pre_contact)
            else:
                self.move_plug(move_robot, pre_contact, 2.0, mode="descent")
        self.log_truth("pre-contact", task, plug=self.plug_pose())
        if self.refiner is not None:
            port, pre_contact, converged = self.servo(move_robot, get_observation, task, port)
            if converged:
                self.spiral_radius = self.servo_settings["spiral_max_radius_m"]
            self.note("servo_result", converged=converged, spiral_radius_mm=1000 * self.spiral_radius)
        entrance_z = port[2, 3]
        self.port_xy = port[:2, 3].copy()
        self.log_truth("aligned", task, plug=self.plug_pose(), port=port)
        self.bias = self.baseline(get_observation)
        self.note("baseline", force_n=self.bias)

        depth = settings["pre_contact_height_m"] + settings["descent_below_entrance_m"]
        contact, reason = self.ramp_descend(
            move_robot, get_observation, pre_contact, depth, settings["descent_speed_m_s"], "descent",
            depth / settings["descent_speed_m_s"] + 5.0, settings["contact_force_n"],
        )
        inside = contact[2, 3] <= entrance_z - settings["inside_depth_m"]
        self.get_logger().info(
            f"Descent stopped ({reason}) {1000 * (contact[2, 3] - entrance_z):+.1f} mm from the entrance: "
            f"{'inside' if inside else 'on the surface'}"
        )
        self.note("descent", reason=reason, height_above_entrance_mm=1000 * (contact[2, 3] - entrance_z), inside=inside)
        self.log_truth("contact", task, plug=contact)
        reference_z = entrance_z
        entry = contact
        if not inside:
            reference_z = contact[2, 3]
            entered, entry = self.spiral_search(
                move_robot, get_observation, contact, contact[2, 3] - settings["entry_depth_m"]
            )
            self.log_truth("search", task, plug=entry)
            if not entered:
                self.get_logger().info("Pressing from the last spiral position")
        depth = self.press_until_seated(move_robot, get_observation, entry, reference_z, task.plug_type)
        self.note("press", depth_mm=1000 * depth, seat_mm=1000 * settings["min_seat_depth_m"][task.plug_type])
        self.log_truth("final", task, plug=self.plug_pose())
        return bool(depth >= settings["min_seat_depth_m"][task.plug_type])
