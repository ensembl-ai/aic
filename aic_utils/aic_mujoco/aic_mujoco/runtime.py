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
from aic_mujoco.utils.images import unpack_rgb


@wp.kernel
def apply_reset_samples(
    reset: wp.array[bool],
    q_samples: wp.array2d[float],
    board_pos_samples: wp.array[wp.vec3],
    board_quat_samples: wp.array[wp.quat],
    nic_pos_samples: wp.array[wp.vec3],
    nic_quat_samples: wp.array[wp.quat],
    rail_samples: wp.array[int],
    translation_samples: wp.array[float],
    qpos_adr: wp.array[int],
    joint_count: int,
    wrench_dimension: int,
    board_mocap_id: int,
    nic_mocap_id: int,
    tare_settle_steps: int,
    tare_sample_interval: int,
    qpos: wp.array2d[float],
    q_hold: wp.array2d[float],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
    board_position: wp.array[wp.vec3],
    board_quaternion: wp.array[wp.quat],
    nic_position: wp.array[wp.vec3],
    nic_quaternion: wp.array[wp.quat],
    nic_rail_index: wp.array[int],
    nic_translation: wp.array[float],
    tare_sum: wp.array2d[float],
    tare_baseline: wp.array2d[float],
    tare_settle_remaining: wp.array[int],
    tare_interval_remaining: wp.array[int],
    tare_samples: wp.array[int],
    tare_ready: wp.array[bool],
    episode_steps: wp.array[int],
):
    """Apply selected host reset samples and clear per-episode state.

    Args:
        reset: Per-environment reset mask.
        q_samples: Sampled arm joint positions.
        board_pos_samples: Sampled board positions.
        board_quat_samples: Board orientations required by MJWarp mocap.
        nic_pos_samples: Sampled NIC positions.
        nic_quat_samples: NIC orientations required by MJWarp mocap.
        rail_samples: Sampled discrete NIC rail indices.
        translation_samples: Sampled NIC rail translations.
        qpos_adr: Ordered arm generalized-position addresses.
        joint_count: Number of controlled arm joints.
        wrench_dimension: Combined force/torque dimension.
        board_mocap_id: Board mocap index.
        nic_mocap_id: NIC mocap index.
        tare_settle_steps: Physics steps before F/T tare sampling.
        tare_sample_interval: Physics steps between tare samples.
        qpos: Batched generalized positions.
        q_hold: Batched HOLD targets.
        mocap_pos: Batched MJWarp mocap positions.
        mocap_quat: Batched MJWarp mocap orientations.
        board_position: Stored board reset positions.
        board_quaternion: Stored board reset orientations.
        nic_position: Stored NIC reset positions.
        nic_quaternion: Stored NIC reset orientations.
        nic_rail_index: Stored NIC rail indices.
        nic_translation: Stored NIC translations.
        tare_sum: Per-environment wrench accumulation.
        tare_baseline: Per-environment wrench baseline.
        tare_settle_remaining: Remaining settling steps.
        tare_interval_remaining: Remaining steps before the next tare sample.
        tare_samples: Collected tare sample counts.
        tare_ready: Per-environment tare readiness.
        episode_steps: Per-environment post-tare episode step counts.
    """

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
def advance_episode_steps(
    tare_ready: wp.array[bool],
    episode_steps: wp.array[int],
):
    """Advance episode clocks only for environments with a valid F/T tare.

    Args:
        tare_ready: Per-environment tare readiness.
        episode_steps: Mutable per-environment episode step counts.
    """

    world = wp.tid()
    if tare_ready[world]:
        episode_steps[world] = episode_steps[world] + 1


@wp.kernel
def update_tare(
    sensordata: wp.array2d[float],
    force_adr: int,
    torque_adr: int,
    force_dimension: int,
    wrench_dimension: int,
    sample_interval: int,
    sample_target: int,
    tare_sum: wp.array2d[float],
    tare_baseline: wp.array2d[float],
    settle_remaining: wp.array[int],
    interval_remaining: wp.array[int],
    sample_count: wp.array[int],
    ready: wp.array[bool],
    time: wp.array[float],
):
    """Advance independent settle, sample, and average F/T tare state.

    Args:
        sensordata: Batched MJWarp sensor values.
        force_adr: Force sensor start address.
        torque_adr: Torque sensor start address.
        force_dimension: Number of force components.
        wrench_dimension: Combined force/torque dimension.
        sample_interval: Physics steps between samples.
        sample_target: Samples averaged into the baseline.
        tare_sum: Per-environment running sums.
        tare_baseline: Completed per-environment baselines.
        settle_remaining: Remaining settling steps.
        interval_remaining: Remaining steps before the next sample.
        sample_count: Collected sample counts.
        ready: Per-environment tare readiness.
        time: MJWarp simulation time, reset when taring completes.
    """

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
def sample_wrench(
    sensordata: wp.array2d[float],
    force_adr: int,
    torque_adr: int,
    force_dimension: int,
    tare_baseline: wp.array2d[float],
    tare_ready: wp.array[bool],
    raw: wp.array2d[float],
    tared: wp.array2d[float],
):
    """Read raw and independently tared six-axis wrench observations.

    Args:
        sensordata: Batched MJWarp sensor values.
        force_adr: Force sensor start address.
        torque_adr: Torque sensor start address.
        force_dimension: Number of force components.
        tare_baseline: Per-environment tare values.
        tare_ready: Per-environment tare readiness.
        raw: Batched raw wrench output.
        tared: Batched baseline-subtracted wrench output.
    """

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


@dataclass(frozen=True)
class StepEvents:
    """Observation clocks that fired after one physics step."""

    camera: bool
    wrench_sample: bool
    wrench_publication: bool


class AICWarpRuntime:
    """One shared MJCF model with independently randomized MJWarp worlds."""

    def __init__(self, config: dict[str, Any]):
        """Compile, upload, reset, tare, and render all configured worlds.

        Args:
            config: Strict merged runtime or collection configuration.

        Raises:
            ValueError: If graph capture is requested on a non-CUDA device or
                the compiled model violates the runtime contract.
        """

        self.config = config
        self.num_envs = config["runtime"]["num_envs"]
        self.device = wp.get_device(config["physics"]["device"])
        if config["physics"]["graph_capture"] and not self.device.is_cuda:
            raise ValueError("physics.graph_capture requires a CUDA device")

        with wp.ScopedDevice(self.device):
            self.host_model = mujoco.MjModel.from_xml_path(config["scene"]["output"])
            self.validate_compiled_physics()
            self.robot = AICRobot(self.host_model, config)
            self.model = mjw.put_model(self.host_model)
            self.data = mjw.make_data(
                self.host_model,
                nworld=self.num_envs,
                nconmax=config["physics"]["nconmax"],
                njmax=config["physics"]["njmax"],
            )
            self.allocate_state()
            self.create_renderer()
            self.compile_physics_step()
            self.reset_counts = np.zeros(self.num_envs, dtype=np.uint64)
            self.episode_step = 0
            self.reset()
            self.finish_initial_tare()
            self.render()
            self.check_capacities()
            self.check_internal_grasp_contacts()

    def validate_compiled_physics(self) -> None:
        """Validate compiled physics options against the configuration.

        Raises:
            ValueError: If any compiled physics option differs.
        """

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

    def allocate_state(self) -> None:
        """Allocate controller, randomization, clock, and wrench state."""

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
        self.tare_sum = wp.zeros(wrench_shape, dtype=float, device=self.device)
        self.tare_baseline = wp.zeros(wrench_shape, dtype=float, device=self.device)
        self.tare_settle_remaining = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.tare_interval_remaining = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.tare_samples = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.tare_ready = wp.zeros(self.num_envs, dtype=bool, device=self.device)
        self.episode_steps = wp.zeros(self.num_envs, dtype=int, device=self.device)
        self.raw_wrench = wp.zeros(wrench_shape, dtype=float, device=self.device)
        self.tared_wrench = wp.zeros(wrench_shape, dtype=float, device=self.device)

        step_hz = round(1.0 / self.config["physics"]["timestep"])
        self.camera_interval = step_hz // self.config["cameras"]["fps"]
        self.wrench_interval = step_hz // self.config["sensors"]["physics_sample_hz"]
        self.publication_interval = step_hz // self.config["sensors"]["publication_hz"]

    def create_renderer(self) -> None:
        """Create MJWarp's batched RGB-only renderer and named outputs."""

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
        self.rgb_addresses = {
            key: int(addresses[camera_id])
            for key, camera_id in self.robot.camera_ids.items()
        }

    def physics_step(self) -> None:
        """Advance one split MJWarp step with impedance control and taring."""

        mjw.step1(self.model, self.data)
        self.controller.apply(self.data, self.hold_command)
        mjw.step2(self.model, self.data)
        wp.launch(
            advance_episode_steps,
            dim=self.num_envs,
            inputs=[self.tare_ready],
            outputs=[self.episode_steps],
            device=self.device,
        )
        wp.launch(
            update_tare,
            dim=self.num_envs,
            inputs=[
                self.data.sensordata,
                self.robot.force_sensor_address,
                self.robot.torque_sensor_address,
                self.robot.force_sensor_dimension,
                self.robot.wrench_dimension,
                self.wrench_interval,
                self.config["sensors"]["tare_sample_count"],
            ],
            outputs=[
                self.tare_sum,
                self.tare_baseline,
                self.tare_settle_remaining,
                self.tare_interval_remaining,
                self.tare_samples,
                self.tare_ready,
                self.data.time,
            ],
            device=self.device,
        )

    def compile_physics_step(self) -> None:
        """Warm up the physics step and optionally capture a CUDA graph."""

        self.physics_step()
        wp.synchronize_device(self.device)
        self.graph = None
        if self.config["physics"]["graph_capture"]:
            with wp.ScopedCapture(device=self.device) as capture:
                self.physics_step()
            self.graph = capture.graph

    def sample_reset_batch(
        self,
        reset_ids: list[int],
        randomization_ids: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample deterministic reset values for selected environments.

        Args:
            reset_ids: Environment indices to sample.
            randomization_ids: Optional explicit deterministic sample IDs.

        Returns:
            Full-size host arrays containing samples at selected indices.
        """

        control = self.config["control"]
        randomization = self.config["domain_randomization"]
        explicit_ids = (
            dict(zip(reset_ids, randomization_ids, strict=True))
            if randomization_ids is not None
            else {}
        )
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
            if randomization_ids is None:
                seed_values = [
                    self.config["runtime"]["seed"],
                    world,
                    int(self.reset_counts[world]),
                ]
            else:
                seed_values = [
                    self.config["runtime"]["seed"],
                    explicit_ids[world],
                ]
            seed = np.random.SeedSequence(seed_values)
            rng = np.random.default_rng(seed)
            self.reset_counts[world] += 1
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
            board_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_axisAngle2Quat(
                board_quat,
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                yaw,
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

    def reset(
        self,
        env_ids: list[int] | None = None,
        randomization_ids: list[int] | None = None,
    ) -> None:
        """Reset worlds independently and optionally select deterministic samples.

        Normal rollouts derive randomness from environment ID and reset count.
        Dataset collection supplies one nonnegative ``randomization_id`` per
        reset environment. That makes a trajectory's scene sample independent
        of which parallel environment happens to execute it.
        """

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
        if randomization_ids is not None:
            if len(randomization_ids) != len(reset_ids):
                raise ValueError(
                    "reset randomization IDs must match the environment ID count"
                )
            if any(type(index) is not int or index < 0 for index in randomization_ids):
                raise ValueError("reset randomization IDs must be nonnegative integers")
            if len(set(randomization_ids)) != len(randomization_ids):
                raise ValueError("reset randomization IDs contain duplicates")
        batch = self.sample_reset_batch(reset_ids, randomization_ids)
        reset_mask = np.zeros(self.num_envs, dtype=bool)
        reset_mask[reset_ids] = True
        with wp.ScopedDevice(self.device):
            mask = wp.array(reset_mask, dtype=bool, device=self.device)
            mjw.reset_data(self.model, self.data, mask)
            wp.launch(
                apply_reset_samples,
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
                    self.wrench_interval,
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
                    self.tare_sum,
                    self.tare_baseline,
                    self.tare_settle_remaining,
                    self.tare_interval_remaining,
                    self.tare_samples,
                    self.tare_ready,
                    self.episode_steps,
                ],
                device=self.device,
            )
        if len(reset_ids) == self.num_envs:
            self.episode_step = 0

    def advance_physics(self) -> None:
        """Execute one eager or CUDA-graph physics step."""

        if self.graph is None:
            self.physics_step()
        else:
            wp.capture_launch(self.graph)
        self.episode_step += 1

    def finish_initial_tare(self) -> None:
        """Run initialization until every environment has a valid F/T tare.

        Raises:
            RuntimeError: If configured taring does not complete.
        """

        steps = self.config["sensors"]["tare_settle_steps"] + (
            self.config["sensors"]["tare_sample_count"] * self.wrench_interval
        )
        for _ in range(steps):
            self.advance_physics()
        wp.synchronize_device(self.device)
        if not np.all(self.tare_ready.numpy()):
            raise RuntimeError("Initial per-environment F/T taring did not complete")
        self.sample_wrench_now()
        self.episode_steps.zero_()
        self.episode_step = 0

    def sample_wrench_now(self) -> None:
        """Update raw and tared wrench output tensors immediately."""

        wp.launch(
            sample_wrench,
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
                unpack_rgb,
                dim=(self.num_envs, camera["height"], camera["width"]),
                inputs=[
                    self.render_context.rgb_data,
                    self.rgb_addresses[key],
                    camera["width"],
                ],
                outputs=[output],
                device=self.device,
            )

    def step(self) -> StepEvents:
        """Advance every independent world once and update due observations."""

        with wp.ScopedDevice(self.device):
            self.advance_physics()
            wrench_sample = self.episode_step % self.wrench_interval == 0
            camera = self.episode_step % self.camera_interval == 0
            publication = self.episode_step % self.publication_interval == 0
            if wrench_sample:
                self.sample_wrench_now()
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

    def check_capacities(self) -> None:
        """Fail if MJWarp contact or constraint buffers overflowed.

        Raises:
            RuntimeError: If configured device capacities are insufficient.
        """

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

    def check_internal_grasp_contacts(self) -> None:
        """Verify excluded gripper/SFP contacts cannot contaminate wrist F/T.

        Raises:
            RuntimeError: If a gripper-finger/SFP contact is present.
        """

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
