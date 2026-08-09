"""Independent batched AIC HOLD worlds implemented directly with MuJoCo-Warp."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from aic_mujoco.commands import HoldPositionCommand
from aic_mujoco.controllers import JointHoldController
from aic_mujoco.joints import required_model_id
from aic_mujoco.robot import AICRobot


@wp.kernel
def _apply_reset_samples(
    reset: wp.array(dtype=bool),
    q_samples: wp.array2d(dtype=float),
    board_pos_samples: wp.array(dtype=wp.vec3),
    board_quat_samples: wp.array(dtype=wp.quat),
    nic_pos_samples: wp.array(dtype=wp.vec3),
    nic_quat_samples: wp.array(dtype=wp.quat),
    rail_samples: wp.array(dtype=int),
    translation_samples: wp.array(dtype=float),
    qpos_adr: wp.array(dtype=int),
    joint_count: int,
    wrench_dimension: int,
    board_mocap_id: int,
    nic_mocap_id: int,
    tare_settle_steps: int,
    tare_sample_interval: int,
    qpos: wp.array2d(dtype=float),
    q_hold: wp.array2d(dtype=float),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
    board_position: wp.array(dtype=wp.vec3),
    board_quaternion: wp.array(dtype=wp.quat),
    nic_position: wp.array(dtype=wp.vec3),
    nic_quaternion: wp.array(dtype=wp.quat),
    nic_rail_index: wp.array(dtype=int),
    nic_translation: wp.array(dtype=float),
    tare_sum: wp.array2d(dtype=float),
    tare_baseline: wp.array2d(dtype=float),
    tare_settle_remaining: wp.array(dtype=int),
    tare_interval_remaining: wp.array(dtype=int),
    tare_samples: wp.array(dtype=int),
    tare_ready: wp.array(dtype=bool),
    episode_steps: wp.array(dtype=int),
):
    world = wp.tid()
    if not reset[world]:
        return
    for joint in range(joint_count):
        value = q_samples[world, joint]
        qpos[world, qpos_adr[joint]] = value
        q_hold[world, joint] = value
    for axis in range(wrench_dimension):
        tare_sum[world, axis] = 0.0
        tare_baseline[world, axis] = 0.0
    board_position[world] = board_pos_samples[world]
    board_quaternion[world] = board_quat_samples[world]
    nic_position[world] = nic_pos_samples[world]
    nic_quaternion[world] = nic_quat_samples[world]
    nic_rail_index[world] = rail_samples[world]
    nic_translation[world] = translation_samples[world]
    mocap_pos[world, board_mocap_id] = board_pos_samples[world]
    mocap_quat[world, board_mocap_id] = board_quat_samples[world]
    mocap_pos[world, nic_mocap_id] = nic_pos_samples[world]
    mocap_quat[world, nic_mocap_id] = nic_quat_samples[world]
    tare_settle_remaining[world] = tare_settle_steps
    tare_interval_remaining[world] = tare_sample_interval
    tare_samples[world] = 0
    tare_ready[world] = False
    episode_steps[world] = 0


@wp.kernel
def _advance_episode_steps(
    tare_ready: wp.array(dtype=bool),
    episode_steps: wp.array(dtype=int),
):
    world = wp.tid()
    if tare_ready[world]:
        episode_steps[world] = episode_steps[world] + 1


@wp.kernel
def _update_tare(
    sensordata: wp.array2d(dtype=float),
    force_adr: int,
    torque_adr: int,
    force_dimension: int,
    wrench_dimension: int,
    sample_interval: int,
    sample_target: int,
    tare_sum: wp.array2d(dtype=float),
    tare_baseline: wp.array2d(dtype=float),
    settle_remaining: wp.array(dtype=int),
    interval_remaining: wp.array(dtype=int),
    sample_count: wp.array(dtype=int),
    ready: wp.array(dtype=bool),
    time: wp.array(dtype=float),
):
    world = wp.tid()
    if ready[world]:
        return
    if settle_remaining[world] > 0:
        settle_remaining[world] = settle_remaining[world] - 1
        return
    if interval_remaining[world] > 1:
        interval_remaining[world] = interval_remaining[world] - 1
        return
    interval_remaining[world] = sample_interval
    count = sample_count[world] + 1
    sample_count[world] = count
    for axis in range(wrench_dimension):
        sensor_index = force_adr + axis
        if axis >= force_dimension:
            sensor_index = torque_adr + axis - force_dimension
        total = tare_sum[world, axis] + sensordata[world, sensor_index]
        tare_sum[world, axis] = total
        if count == sample_target:
            tare_baseline[world, axis] = total / float(sample_target)
    if count == sample_target:
        ready[world] = True
        time[world] = 0.0


@wp.kernel
def _sample_wrench(
    sensordata: wp.array2d(dtype=float),
    force_adr: int,
    torque_adr: int,
    force_dimension: int,
    tare_baseline: wp.array2d(dtype=float),
    tare_ready: wp.array(dtype=bool),
    raw: wp.array2d(dtype=float),
    tared: wp.array2d(dtype=float),
):
    world, axis = wp.tid()
    sensor_index = force_adr + axis
    if axis >= force_dimension:
        sensor_index = torque_adr + axis - force_dimension
    value = sensordata[world, sensor_index]
    raw[world, axis] = value
    if tare_ready[world]:
        tared[world, axis] = value - tare_baseline[world, axis]
    else:
        tared[world, axis] = 0.0


@wp.kernel
def _unpack_rgb(
    packed: wp.array2d(dtype=wp.uint32),
    address: int,
    width: int,
    output: wp.array4d(dtype=wp.uint8),
):
    world, row, column = wp.tid()
    pixel = packed[world, address + row * width + column]
    output[world, row, column, 0] = wp.uint8((pixel >> wp.uint32(16)) & wp.uint32(255))
    output[world, row, column, 1] = wp.uint8((pixel >> wp.uint32(8)) & wp.uint32(255))
    output[world, row, column, 2] = wp.uint8(pixel & wp.uint32(255))


@dataclass(frozen=True)
class StepEvents:
    """Observation clocks that fired after one physics step."""

    camera: bool
    wrench_sample: bool
    wrench_publication: bool


class AICWarpRuntime:
    """One shared MJCF model with independently randomized MJWarp worlds."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.num_envs = config["runtime"]["num_envs"]
        self.device = wp.get_device(config["physics"]["device"])
        if config["physics"]["graph_capture"] and not self.device.is_cuda:
            raise ValueError("physics.graph_capture requires a CUDA device")

        with wp.ScopedDevice(self.device):
            self.host_model = mujoco.MjModel.from_xml_path(config["scene"]["output"])
            self._validate_compiled_physics()
            self.robot = AICRobot(self.host_model, config)
            self.model = mjw.put_model(self.host_model)
            self.data = mjw.make_data(
                self.host_model,
                nworld=self.num_envs,
                nconmax=config["physics"]["nconmax"],
                njmax=config["physics"]["njmax"],
            )
            self._allocate_state()
            self._create_renderer()
            self._compile_physics_step()
            self._reset_counts = np.zeros(self.num_envs, dtype=np.uint64)
            self.episode_step = 0
            self.reset()
            self._finish_initial_tare()
            self.render()
            self._check_capacities()
            self._check_internal_grasp_contacts()

    def _validate_compiled_physics(self) -> None:
        physics = self.config["physics"]
        expected_integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        expected_solver = mujoco.mjtSolver.mjSOL_NEWTON
        if self.host_model.opt.integrator != expected_integrator:
            raise ValueError("Generated MJCF does not use configured implicitfast integration")
        if self.host_model.opt.solver != expected_solver:
            raise ValueError("Generated MJCF does not use configured Newton solving")
        scalar_values = (
            ("timestep", self.host_model.opt.timestep),
            ("tolerance", self.host_model.opt.tolerance),
        )
        for key, actual in scalar_values:
            if not math.isclose(actual, physics[key], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Generated MJCF physics.{key} does not match configuration")
        if self.host_model.opt.iterations != physics["iterations"]:
            raise ValueError("Generated MJCF physics.iterations does not match configuration")
        if not np.allclose(
            self.host_model.opt.gravity, physics["gravity"], rtol=0.0, atol=1e-12
        ):
            raise ValueError("Generated MJCF physics.gravity does not match configuration")

    def _allocate_state(self) -> None:
        joint_count = self.robot.joints.count
        self.hold_command = HoldPositionCommand(
            self.num_envs, joint_count, self.device
        )
        self.controller = JointHoldController(
            self.config, self.robot.joints, self.num_envs, self.device
        )
        self.board_position = wp.zeros(self.num_envs, dtype=wp.vec3, device=self.device)
        self.board_quaternion = wp.zeros(self.num_envs, dtype=wp.quat, device=self.device)
        self.nic_position = wp.zeros(self.num_envs, dtype=wp.vec3, device=self.device)
        self.nic_quaternion = wp.zeros(self.num_envs, dtype=wp.quat, device=self.device)
        self.nic_rail_index = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.nic_translation = wp.zeros(self.num_envs, dtype=float, device=self.device)

        wrench_shape = (self.num_envs, self.robot.wrench_dimension)
        self._tare_sum = wp.zeros(wrench_shape, dtype=float, device=self.device)
        self.tare_baseline = wp.zeros(wrench_shape, dtype=float, device=self.device)
        self._tare_settle_remaining = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self._tare_interval_remaining = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self._tare_samples = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.tare_ready = wp.zeros(self.num_envs, dtype=bool, device=self.device)
        self.episode_steps = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.raw_wrench = wp.zeros(wrench_shape, dtype=float, device=self.device)
        self.tared_wrench = wp.zeros(wrench_shape, dtype=float, device=self.device)

        step_hz = round(1.0 / self.config["physics"]["timestep"])
        self._camera_interval = step_hz // self.config["cameras"]["fps"]
        self._wrench_interval = step_hz // self.config["sensors"]["physics_sample_hz"]
        self._publication_interval = step_hz // self.config["sensors"]["publication_hz"]

    def _create_renderer(self) -> None:
        camera = self.config["cameras"]
        resolution = (camera["width"], camera["height"])
        self.render_context = mjw.create_render_context(
            self.host_model,
            nworld=self.num_envs,
            cam_res=[resolution] * self.host_model.ncam,
            render_rgb=[True] * self.host_model.ncam,
            render_depth=[False] * self.host_model.ncam,
            use_textures=camera["use_textures"],
            use_shadows=camera["use_shadows"],
        )
        addresses = self.render_context.rgb_adr.numpy().tolist()
        self.rgb = {
            key: wp.empty(
                (self.num_envs, camera["height"], camera["width"], 3),
                dtype=wp.uint8,
                device=self.device,
            )
            for key in self.robot.camera_ids
        }
        self._rgb_addresses = {
            key: int(addresses[camera_id])
            for key, camera_id in self.robot.camera_ids.items()
        }

    def _physics_step(self) -> None:
        mjw.step1(self.model, self.data)
        self.controller.apply(self.data, self.hold_command)
        mjw.step2(self.model, self.data)
        wp.launch(
            _advance_episode_steps,
            dim=self.num_envs,
            inputs=[self.tare_ready],
            outputs=[self.episode_steps],
            device=self.device,
        )
        wp.launch(
            _update_tare,
            dim=self.num_envs,
            inputs=[
                self.data.sensordata,
                self.robot.force_sensor_address,
                self.robot.torque_sensor_address,
                self.robot.force_sensor_dimension,
                self.robot.wrench_dimension,
                self._wrench_interval,
                self.config["sensors"]["tare_sample_count"],
            ],
            outputs=[
                self._tare_sum,
                self.tare_baseline,
                self._tare_settle_remaining,
                self._tare_interval_remaining,
                self._tare_samples,
                self.tare_ready,
                self.data.time,
            ],
            device=self.device,
        )

    def _compile_physics_step(self) -> None:
        self._physics_step()
        wp.synchronize_device(self.device)
        self._graph = None
        if self.config["physics"]["graph_capture"]:
            with wp.ScopedCapture(device=self.device) as capture:
                self._physics_step()
            self._graph = capture.graph

    def _sample_reset_batch(self, reset_ids: list[int]) -> dict[str, np.ndarray]:
        control = self.config["control"]
        randomization = self.config["domain_randomization"]
        batch = {
            "q": np.zeros(
                (self.num_envs, self.robot.joints.count), dtype=np.float32
            ),
            "board_pos": np.zeros((self.num_envs, 3), dtype=np.float32),
            "board_quat": np.zeros((self.num_envs, 4), dtype=np.float32),
            "nic_pos": np.zeros((self.num_envs, 3), dtype=np.float32),
            "nic_quat": np.zeros((self.num_envs, 4), dtype=np.float32),
            "rail": np.zeros(self.num_envs, dtype=np.int32),
            "translation": np.zeros(self.num_envs, dtype=np.float32),
        }
        for world in reset_ids:
            seed = np.random.SeedSequence(
                [self.config["runtime"]["seed"], world, int(self._reset_counts[world])]
            )
            rng = np.random.default_rng(seed)
            self._reset_counts[world] += 1
            batch["q"][world] = np.asarray(control["home"]) + rng.uniform(
                control["reset_perturbation_lower"],
                control["reset_perturbation_upper"],
            )
            board_pos = rng.uniform(
                randomization["board_position_lower"],
                randomization["board_position_upper"],
            )
            deviation = rng.uniform(
                randomization["board_yaw_deviation_lower"],
                randomization["board_yaw_deviation_upper"],
            )
            yaw = (math.pi + deviation + math.pi) % (2.0 * math.pi) - math.pi
            board_quat = np.asarray(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
            )
            rail = int(rng.choice(randomization["nic_rail_indices"]))
            translation = float(
                rng.uniform(
                    randomization["nic_translation_lower"],
                    randomization["nic_translation_upper"],
                )
            )
            local = np.asarray(
                [
                    randomization["nic_rail_x_base"] + translation,
                    randomization["nic_rail_y_by_index"][rail],
                    randomization["nic_rail_z"],
                ]
            )
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            nic_pos = board_pos + np.asarray(
                [
                    cosine * local[0] - sine * local[1],
                    sine * local[0] + cosine * local[1],
                    local[2],
                ]
            )
            batch["board_pos"][world] = board_pos
            batch["board_quat"][world] = board_quat
            batch["nic_pos"][world] = nic_pos
            batch["nic_quat"][world] = board_quat
            batch["rail"][world] = rail
            batch["translation"][world] = translation
        return batch

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Reset selected worlds independently; taring then progresses asynchronously."""

        reset_ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if not reset_ids:
            raise ValueError("reset requires at least one environment ID")
        if len(set(reset_ids)) != len(reset_ids):
            raise ValueError("reset environment IDs contain duplicates")
        if any(
            type(index) is not int or index < 0 or index >= self.num_envs
            for index in reset_ids
        ):
            raise ValueError("reset contains an invalid environment ID")
        batch = self._sample_reset_batch(reset_ids)
        reset_mask = np.zeros(self.num_envs, dtype=bool)
        reset_mask[reset_ids] = True
        with wp.ScopedDevice(self.device):
            mask = wp.array(reset_mask, dtype=bool, device=self.device)
            mjw.reset_data(self.model, self.data, mask)
            wp.launch(
                _apply_reset_samples,
                dim=self.num_envs,
                inputs=[
                    mask,
                    wp.array(batch["q"], dtype=float, device=self.device),
                    wp.array(batch["board_pos"], dtype=wp.vec3, device=self.device),
                    wp.array(batch["board_quat"], dtype=wp.quat, device=self.device),
                    wp.array(batch["nic_pos"], dtype=wp.vec3, device=self.device),
                    wp.array(batch["nic_quat"], dtype=wp.quat, device=self.device),
                    wp.array(batch["rail"], dtype=int, device=self.device),
                    wp.array(batch["translation"], dtype=float, device=self.device),
                    self.controller.qpos_addresses,
                    self.robot.joints.count,
                    self.robot.wrench_dimension,
                    self.robot.board_mocap_id,
                    self.robot.nic_mocap_id,
                    self.config["sensors"]["tare_settle_steps"],
                    self._wrench_interval,
                ],
                outputs=[
                    self.data.qpos,
                    self.hold_command.position,
                    self.data.mocap_pos,
                    self.data.mocap_quat,
                    self.board_position,
                    self.board_quaternion,
                    self.nic_position,
                    self.nic_quaternion,
                    self.nic_rail_index,
                    self.nic_translation,
                    self._tare_sum,
                    self.tare_baseline,
                    self._tare_settle_remaining,
                    self._tare_interval_remaining,
                    self._tare_samples,
                    self.tare_ready,
                    self.episode_steps,
                ],
                device=self.device,
            )
        if len(reset_ids) == self.num_envs:
            self.episode_step = 0

    def _advance_physics(self) -> None:
        if self._graph is None:
            self._physics_step()
        else:
            wp.capture_launch(self._graph)
        self.episode_step += 1

    def _finish_initial_tare(self) -> None:
        steps = self.config["sensors"]["tare_settle_steps"] + (
            self.config["sensors"]["tare_sample_count"] * self._wrench_interval
        )
        for _ in range(steps):
            self._advance_physics()
        wp.synchronize_device(self.device)
        if not np.all(self.tare_ready.numpy()):
            raise RuntimeError("Initial per-environment F/T taring did not complete")
        self._sample_wrench_now()
        self.episode_steps.zero_()
        self.episode_step = 0

    def _sample_wrench_now(self) -> None:
        wp.launch(
            _sample_wrench,
            dim=(self.num_envs, self.robot.wrench_dimension),
            inputs=[
                self.data.sensordata,
                self.robot.force_sensor_address,
                self.robot.torque_sensor_address,
                self.robot.force_sensor_dimension,
                self.tare_baseline,
                self.tare_ready,
            ],
            outputs=[self.raw_wrench, self.tared_wrench],
            device=self.device,
        )

    def render(self) -> None:
        """Update all three RGB tensors from MJWarp's batched ray renderer."""

        mjw.fwd_position(self.model, self.data)
        mjw.render(self.model, self.data, self.render_context)
        camera = self.config["cameras"]
        for key, output in self.rgb.items():
            wp.launch(
                _unpack_rgb,
                dim=(self.num_envs, camera["height"], camera["width"]),
                inputs=[
                    self.render_context.rgb_data,
                    self._rgb_addresses[key],
                    camera["width"],
                ],
                outputs=[output],
                device=self.device,
            )

    def step(self) -> StepEvents:
        """Advance every independent world once and update due observations."""

        with wp.ScopedDevice(self.device):
            self._advance_physics()
            wrench_sample = self.episode_step % self._wrench_interval == 0
            camera = self.episode_step % self._camera_interval == 0
            publication = self.episode_step % self._publication_interval == 0
            if wrench_sample:
                self._sample_wrench_now()
            if camera:
                self.render()
        return StepEvents(camera, wrench_sample, publication)

    def observations(self) -> dict[str, Any]:
        """Return named device tensors; no packed renderer storage leaks out."""

        return {
            "rgb": self.rgb,
            "wrench": {
                "raw": self.raw_wrench,
                "tared": self.tared_wrench,
                "tare_ready": self.tare_ready,
            },
        }

    def reset_state(self) -> dict[str, wp.array]:
        """Return the current independently sampled reset parameters."""

        return {
            "joint_hold": self.hold_command.position,
            "board_position": self.board_position,
            "board_quaternion": self.board_quaternion,
            "nic_position": self.nic_position,
            "nic_quaternion": self.nic_quaternion,
            "nic_rail_index": self.nic_rail_index,
            "nic_translation": self.nic_translation,
            "episode_steps": self.episode_steps,
        }

    def _check_capacities(self) -> None:
        wp.synchronize_device(self.device)
        max_constraints = int(np.max(self.data.nefc.numpy()))
        actual_contacts = int(self.data.nacon.numpy()[0])
        broadphase_contacts = int(self.data.ncollision.numpy()[0])
        if max_constraints > self.data.njmax:
            raise RuntimeError(
                f"MJWarp constraint capacity overflow: {max_constraints} > {self.data.njmax}"
            )
        if actual_contacts > self.data.naconmax or broadphase_contacts > self.data.naconmax:
            raise RuntimeError(
                "MJWarp contact capacity overflow; increase physics.nconmax explicitly"
            )

    def _check_internal_grasp_contacts(self) -> None:
        count = min(int(self.data.nacon.numpy()[0]), self.data.naconmax)
        if count == 0:
            return
        names = self.config["scene"]["names"]
        finger_ids = {
            required_model_id(
                self.host_model,
                mujoco.mjtObj.mjOBJ_BODY,
                names["left_finger_body"],
            ),
            required_model_id(
                self.host_model,
                mujoco.mjtObj.mjOBJ_BODY,
                names["right_finger_body"],
            ),
        }
        sfp_root = required_model_id(
            self.host_model, mujoco.mjtObj.mjOBJ_BODY, names["sfp_source_body"]
        )
        sfp_ids = {sfp_root}
        for body in range(self.host_model.nbody):
            parent = body
            while parent > 0:
                parent = int(self.host_model.body_parentid[parent])
                if parent == sfp_root:
                    sfp_ids.add(body)
                    break
        geom_pairs = self.data.contact.geom.numpy()[:count]
        for first, second in geom_pairs:
            bodies = {
                int(self.host_model.geom_bodyid[int(first)]),
                int(self.host_model.geom_bodyid[int(second)]),
            }
            if bodies & finger_ids and bodies & sfp_ids:
                raise RuntimeError("Internal gripper/SFP contact reached the F/T sensor")
