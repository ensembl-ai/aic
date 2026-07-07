"""MuJoCo Warp batched physics helpers for the AIC training scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

from aic_mujoco.config import load_json_config


@dataclass(frozen=True)
class TrainingPhysicsConfig:
    """Inputs for a direct MuJoCo Warp compile/step run.

    Args:
        xml_path: Warp-compatible scene, normally ``mjcf/scene_warp.xml``.
        num_envs: Number of batched Warp worlds to allocate.
        steps: Number of Warp physics steps to run.
        device: Warp device string. Use ``cuda`` for the intended training
            path; this module does not silently choose a fallback device.
        log_interval: Print throughput every N Warp steps. Use ``0`` to print
            only the final summary.
        config_path: Optional task config. When provided, ``reset_q`` is used
            to seed robot joint positions before uploading data to Warp.
        sample_envs: Number of envs to download at log points for observation
            diagnostics. This is intentionally small so logging does not become
            the training bottleneck.
        motion_scale: Multiplier for the built-in debug policy. The policy is
            a small per-env joint target perturbation around ``reset_q`` plus
            PD/bias torque control, so the arm is held up instead of falling.
    """

    xml_path: Path
    num_envs: int = 32
    steps: int = 1
    device: str = "cuda"
    log_interval: int = 100
    config_path: Path | None = None
    sample_envs: int = 16
    motion_scale: float = 1.0


@dataclass(frozen=True)
class DebugJointPolicy:
    """GPU-resident target/PD constants for the temporary training policy.

    Args:
        base_q: Reset joint position for the six controlled UR joints.
        kp: Joint stiffness used to convert target error to motor torque.
        kd: Joint damping used to suppress velocity.
        torque_limits: Per-joint torque clipping limits.

    This is not the final learned policy. It is a deliberately small GPU-side
    policy/controller so the XML -> Warp -> action -> observation -> reward
    pipeline is exercised with different env trajectories instead of passive
    zero-control physics.
    """

    base_q: wp.array
    kp: wp.array
    kd: wp.array
    torque_limits: wp.array


@dataclass(frozen=True)
class SampleObservation:
    """Observation and reward summary from a small sample of Warp worlds."""

    reward_mean: float
    progress_mean: float
    lateral_error_mean: float
    force_norm_max: float
    contact_force_sum_mean: float
    max_penetration: float
    action_norm_mean: float
    action_norm_max: float
    qpos_std_mean: float
    sample_envs: int


@wp.kernel
def _apply_debug_joint_policy(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    qfrc_bias: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    base_q: wp.array(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    torque_limits: wp.array(dtype=wp.float32),
    step_idx: int,
    motion_scale: float,
) -> None:
    """Write per-env motor torques from a simple phase-varied joint policy."""

    env_id = wp.tid()
    phase = wp.float32(env_id) * wp.float32(0.731)
    t = wp.float32(step_idx) * wp.float32(0.002)
    amp = wp.float32(0.015) * wp.float32(motion_scale)

    target0 = base_q[0] + amp * wp.sin(t * wp.float32(1.3) + phase)
    target1 = base_q[1] + amp * wp.sin(t * wp.float32(0.9) + phase * wp.float32(1.7))
    target2 = base_q[2] + amp * wp.sin(t * wp.float32(1.1) + phase * wp.float32(2.3))
    target3 = base_q[3] + amp * wp.sin(t * wp.float32(1.6) + phase * wp.float32(0.5))
    target4 = base_q[4] + amp * wp.sin(t * wp.float32(1.4) + phase * wp.float32(1.1))
    target5 = base_q[5] + amp * wp.sin(t * wp.float32(1.0) + phase * wp.float32(1.9))

    tau0 = kp[0] * (target0 - qpos[env_id, 0]) - kd[0] * qvel[env_id, 0] + qfrc_bias[env_id, 0]
    tau1 = kp[1] * (target1 - qpos[env_id, 1]) - kd[1] * qvel[env_id, 1] + qfrc_bias[env_id, 1]
    tau2 = kp[2] * (target2 - qpos[env_id, 2]) - kd[2] * qvel[env_id, 2] + qfrc_bias[env_id, 2]
    tau3 = kp[3] * (target3 - qpos[env_id, 3]) - kd[3] * qvel[env_id, 3] + qfrc_bias[env_id, 3]
    tau4 = kp[4] * (target4 - qpos[env_id, 4]) - kd[4] * qvel[env_id, 4] + qfrc_bias[env_id, 4]
    tau5 = kp[5] * (target5 - qpos[env_id, 5]) - kd[5] * qvel[env_id, 5] + qfrc_bias[env_id, 5]

    ctrl[env_id, 0] = wp.clamp(tau0, -torque_limits[0], torque_limits[0])
    ctrl[env_id, 1] = wp.clamp(tau1, -torque_limits[1], torque_limits[1])
    ctrl[env_id, 2] = wp.clamp(tau2, -torque_limits[2], torque_limits[2])
    ctrl[env_id, 3] = wp.clamp(tau3, -torque_limits[3], torque_limits[3])
    ctrl[env_id, 4] = wp.clamp(tau4, -torque_limits[4], torque_limits[4])
    ctrl[env_id, 5] = wp.clamp(tau5, -torque_limits[5], torque_limits[5])
    ctrl[env_id, 6] = wp.float32(0.0)


def run_training_physics(
    cfg: TrainingPhysicsConfig,
) -> dict[str, float | int | str]:
    """Compile a MuJoCo model into Warp data and run zero-control steps.

    Args:
        cfg: XML path, batch size, and step count.

    Returns:
        Model and throughput summary, including ``nq/nv/nu``, total physics
        steps, aggregate simulated seconds, and env-0 sim time.

    This is intentionally direct: no MJLab, no manager layer, no hidden scene
    composition. If this fails, the failure is in the MuJoCo XML, MuJoCo Warp,
    CUDA/Warp cache, or the local driver/runtime setup.
    """

    wp.set_device(str(cfg.device))
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    data = mujoco.MjData(model)
    if cfg.config_path is not None:
        initialize_data_from_config(model, data, cfg.config_path)
    else:
        mujoco.mj_forward(model, data)
    policy = make_debug_joint_policy(cfg.config_path, cfg.device)
    sample_datas = [mujoco.MjData(model) for _ in range(min(cfg.sample_envs, cfg.num_envs))]
    task_ids = task_entity_ids(model)
    print_training_header(cfg, model)
    t_compile0 = perf_counter()
    warp_model = mujoco_warp.put_model(model)
    warp_data = mujoco_warp.put_data(model, data, nworld=int(cfg.num_envs))
    wp.synchronize()
    compile_wall_s = perf_counter() - t_compile0
    t_step0 = perf_counter()
    last_log_step = 0
    last_log_time = t_step0
    for step_idx in range(1, int(cfg.steps) + 1):
        apply_debug_joint_policy(
            warp_data=warp_data,
            policy=policy,
            num_envs=int(cfg.num_envs),
            step_idx=step_idx,
            motion_scale=float(cfg.motion_scale),
        )
        mujoco_warp.step(warp_model, warp_data)
        if int(cfg.log_interval) > 0 and (
            step_idx % int(cfg.log_interval) == 0 or step_idx == int(cfg.steps)
        ):
            wp.synchronize()
            now = perf_counter()
            interval_steps = step_idx - last_log_step
            interval_wall_s = max(now - last_log_time, 1e-12)
            interval_physics_steps = interval_steps * int(cfg.num_envs)
            interval_sim_s = interval_physics_steps * float(model.opt.timestep)
            total_wall_s = max(now - t_step0, 1e-12)
            total_physics_steps = step_idx * int(cfg.num_envs)
            total_sim_s = total_physics_steps * float(model.opt.timestep)
            obs = sample_observations(
                model=model,
                warp_data=warp_data,
                sample_datas=sample_datas,
                task_ids=task_ids,
            )
            print_training_progress(
                step_idx=step_idx,
                total_steps=int(cfg.steps),
                num_envs=int(cfg.num_envs),
                physics_dt=float(model.opt.timestep),
                interval_steps=interval_steps,
                interval_wall_s=interval_wall_s,
                total_wall_s=total_wall_s,
                total_physics_steps=total_physics_steps,
                total_sim_s=total_sim_s,
                obs=obs,
            )
            last_log_step = step_idx
            last_log_time = now
    wp.synchronize()
    step_wall_s = perf_counter() - t_step0
    out = mujoco.MjData(model)
    mujoco_warp.get_data_into(out, model, warp_data, world_id=0)
    physics_steps = int(cfg.num_envs) * int(cfg.steps)
    aggregate_sim_seconds = physics_steps * float(model.opt.timestep)
    summary = {
        "xml": str(cfg.xml_path),
        "device": str(cfg.device),
        "num_envs": int(cfg.num_envs),
        "steps": int(cfg.steps),
        "physics_steps": physics_steps,
        "compile_wall_s": float(compile_wall_s),
        "step_wall_s": float(step_wall_s),
        "physics_steps_per_sec": float(physics_steps / max(step_wall_s, 1e-12)),
        "aggregate_sim_seconds": float(aggregate_sim_seconds),
        "aggregate_sim_seconds_per_wall_second": float(
            aggregate_sim_seconds / max(step_wall_s, 1e-12)
        ),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "sim_time_env0": float(out.time),
    }
    final_obs = sample_observations(
        model=model,
        warp_data=warp_data,
        sample_datas=sample_datas,
        task_ids=task_ids,
    )
    print_training_summary(summary, final_obs)
    return summary


def initialize_data_from_config(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config_path: Path,
) -> None:
    """Seed MuJoCo data from the task reset configuration.

    Args:
        model: Compiled training model.
        data: Host data object that will later be copied to Warp.
        config_path: JSON config containing ``controlled_joints`` and
            ``reset_q`` or ``ik_home_q``.

    The Warp-compatible scene has no cable freejoint. For the first training
    state seed we therefore set the robot arm joints and let the rigidly held
    plug follow the gripper through the model topology.
    """

    cfg = load_json_config(config_path)
    joint_names = [str(name) for name in cfg["controlled_joints"]]
    q_values = cfg.get("reset_q", cfg.get("ik_home_q"))
    if q_values is None:
        raise RuntimeError(f"Config has no reset_q or ik_home_q: {config_path}")
    if len(q_values) != len(joint_names):
        raise RuntimeError(
            f"reset_q length {len(q_values)} does not match controlled_joints "
            f"length {len(joint_names)}"
        )
    mujoco.mj_resetData(model, data)
    for joint_name, q_value in zip(joint_names, q_values, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in training scene: {joint_name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(q_value)
    mujoco.mj_forward(model, data)
    data.ctrl[: min(6, model.nu)] = data.qfrc_bias[: min(6, model.nu)]


def make_debug_joint_policy(config_path: Path | None, device: str) -> DebugJointPolicy:
    """Create GPU-resident constants for the temporary joint policy.

    Args:
        config_path: JSON config containing ``reset_q``, ``kp``, ``kd``, and
            ``torque_limits``.
        device: Warp device string such as ``cuda``.

    Returns:
        A ``DebugJointPolicy`` with arrays allocated on ``device``.
    """

    if config_path is None:
        raise RuntimeError("Training requires --config for reset/control constants.")
    cfg = load_json_config(config_path)
    base_q = np.asarray(cfg["reset_q"], dtype=np.float32)
    kp = np.asarray(cfg["kp"], dtype=np.float32)
    kd = np.asarray(cfg["kd"], dtype=np.float32)
    torque_limits = np.asarray(cfg["torque_limits"], dtype=np.float32)
    return DebugJointPolicy(
        base_q=wp.array(base_q, dtype=wp.float32, device=device),
        kp=wp.array(kp, dtype=wp.float32, device=device),
        kd=wp.array(kd, dtype=wp.float32, device=device),
        torque_limits=wp.array(torque_limits, dtype=wp.float32, device=device),
    )


def apply_debug_joint_policy(
    *,
    warp_data,
    policy: DebugJointPolicy,
    num_envs: int,
    step_idx: int,
    motion_scale: float,
) -> None:
    """Apply the temporary per-env joint policy directly on Warp tensors."""

    wp.launch(
        _apply_debug_joint_policy,
        dim=int(num_envs),
        inputs=[
            warp_data.qpos,
            warp_data.qvel,
            warp_data.qfrc_bias,
            warp_data.ctrl,
            policy.base_q,
            policy.kp,
            policy.kd,
            policy.torque_limits,
            int(step_idx),
            float(motion_scale),
        ],
    )


def task_entity_ids(model: mujoco.MjModel) -> dict[str, int]:
    """Resolve body ids used by sampled observation/reward diagnostics."""

    names = {
        "tip": "sfp_tip_link",
        "port_entrance": "sfp_port_0_link_entrance",
        "port_bottom": "sfp_port_0_link",
    }
    ids: dict[str, int] = {}
    for key, name in names.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Body not found in training scene: {name}")
        ids[key] = int(body_id)
    return ids


def sample_observations(
    *,
    model: mujoco.MjModel,
    warp_data,
    sample_datas: list[mujoco.MjData],
    task_ids: dict[str, int],
) -> SampleObservation:
    """Download a small env sample and compute training-style diagnostics.

    Args:
        model: Compiled MuJoCo model.
        warp_data: Batched MuJoCo Warp data.
        sample_datas: Reusable host ``MjData`` buffers.
        task_ids: Body ids for SFP tip and port reference bodies.

    Returns:
        Mean reward/progress/lateral/action statistics plus contact summaries.

    This samples a few worlds for logging only. The fast path remains batched
    Warp stepping; full all-env reductions belong in the next learned policy
    implementation.
    """

    rewards: list[float] = []
    progresses: list[float] = []
    lateral_errors: list[float] = []
    force_sums: list[float] = []
    penetrations: list[float] = []
    action_norms: list[float] = []
    qpos_values: list[np.ndarray] = []
    for env_id, data in enumerate(sample_datas):
        mujoco_warp.get_data_into(data, model, warp_data, world_id=env_id)
        tip = np.asarray(data.xpos[task_ids["tip"]], dtype=float)
        port_entrance = np.asarray(data.xpos[task_ids["port_entrance"]], dtype=float)
        port_bottom = np.asarray(data.xpos[task_ids["port_bottom"]], dtype=float)
        lateral = float(np.linalg.norm((tip - port_entrance)[:2]))
        insertion_axis = port_bottom - port_entrance
        axis_norm = float(np.linalg.norm(insertion_axis))
        if axis_norm > 1e-9:
            progress = float(np.dot(tip - port_entrance, insertion_axis / axis_norm))
        else:
            progress = 0.0
        penetration, normal_force_sum = contact_diagnostics(model, data)
        action_norm = float(np.linalg.norm(data.ctrl[: min(6, model.nu)]))
        reward = progress - lateral - 5.0 * penetration - 1e-4 * action_norm
        rewards.append(reward)
        progresses.append(progress)
        lateral_errors.append(lateral)
        penetrations.append(penetration)
        force_sums.append(normal_force_sum)
        action_norms.append(action_norm)
        qpos_values.append(np.asarray(data.qpos[: min(6, model.nq)], dtype=float).copy())

    qpos_stack = np.asarray(qpos_values, dtype=float)
    return SampleObservation(
        reward_mean=float(np.mean(rewards)),
        progress_mean=float(np.mean(progresses)),
        lateral_error_mean=float(np.mean(lateral_errors)),
        force_norm_max=float(np.max(force_sums)),
        contact_force_sum_mean=float(np.mean(force_sums)),
        max_penetration=float(np.max(penetrations)),
        action_norm_mean=float(np.mean(action_norms)),
        action_norm_max=float(np.max(action_norms)),
        qpos_std_mean=float(np.mean(np.std(qpos_stack, axis=0))),
        sample_envs=len(sample_datas),
    )


def contact_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[float, float]:
    """Return max penetration and summed normal contact force for one env."""

    max_penetration = 0.0
    normal_force_sum = 0.0
    wrench = np.zeros(6, dtype=float)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        normal_force_sum += abs(float(wrench[0]))
    return max_penetration, normal_force_sum


def print_training_header(cfg: TrainingPhysicsConfig, model: mujoco.MjModel) -> None:
    """Print static training-scene and runtime metadata before stepping."""

    print()
    print("################################################################################")
    print("AIC insertion training physics")
    print("################################################################################")
    print(f"  backend:                 MuJoCo Warp")
    print(f"  device:                  {cfg.device}")
    print(f"  xml:                     {cfg.xml_path}")
    print(f"  worlds/envs:             {int(cfg.num_envs)}")
    print(f"  rollout physics steps:   {int(cfg.steps)}")
    print(f"  sampled log envs:        {int(min(cfg.sample_envs, cfg.num_envs))}")
    print(f"  debug motion scale:      {float(cfg.motion_scale):.3f}")
    print(f"  physics dt:              {float(model.opt.timestep):.6f} s")
    print(f"  aggregate sim horizon:   {int(cfg.num_envs) * int(cfg.steps) * float(model.opt.timestep):.3f} s")
    print(f"  model:                   nbody={model.nbody} nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom} nmesh={model.nmesh}")
    print()


def print_training_progress(
    *,
    step_idx: int,
    total_steps: int,
    num_envs: int,
    physics_dt: float,
    interval_steps: int,
    interval_wall_s: float,
    total_wall_s: float,
    total_physics_steps: int,
    total_sim_s: float,
    obs: SampleObservation,
) -> None:
    """Print one periodic training-style progress block.

    Observation fields are sampled from a small number of downloaded worlds.
    They are real diagnostics from the batched physics state, not placeholders.
    """

    interval_physics_steps = interval_steps * num_envs
    interval_sim_s = interval_physics_steps * physics_dt
    interval_physics_hz = interval_physics_steps / max(interval_wall_s, 1e-12)
    total_physics_hz = total_physics_steps / max(total_wall_s, 1e-12)
    interval_rtf = interval_sim_s / max(interval_wall_s, 1e-12)
    total_rtf = total_sim_s / max(total_wall_s, 1e-12)
    progress_pct = 100.0 * step_idx / max(total_steps, 1)
    eta_s = (total_steps - step_idx) * (total_wall_s / max(step_idx, 1))

    print()
    print("--------------------------------------------------------------------------------")
    print(f"Training physics progress {step_idx}/{total_steps} ({progress_pct:5.1f}%)")
    print("--------------------------------------------------------------------------------")
    print(f"  envs:                         {num_envs}")
    print(f"  physics steps total:          {total_physics_steps}")
    print(f"  aggregate sim time:           {total_sim_s:.3f} s")
    print(f"  wall time:                    {total_wall_s:.3f} s")
    print(f"  ETA:                          {eta_s:.3f} s")
    print(f"  interval physics steps/sec:   {interval_physics_hz:,.1f}")
    print(f"  total physics steps/sec:      {total_physics_hz:,.1f}")
    print(f"  interval sim seconds/wall:    {interval_rtf:,.3f}")
    print(f"  total sim seconds/wall:       {total_rtf:,.3f}")
    print(f"  sample envs:                  {obs.sample_envs}")
    print(f"  reward mean:                  {obs.reward_mean:+.5f}")
    print(f"  progress mean:                {obs.progress_mean:+.5f} m")
    print(f"  lateral error mean:           {obs.lateral_error_mean:.5f} m")
    print(f"  contact normal force mean:    {obs.contact_force_sum_mean:.5f} N")
    print(f"  contact normal force max:     {obs.force_norm_max:.5f} N")
    print(f"  max penetration:              {obs.max_penetration:.6f} m")
    print(f"  action norm mean:             {obs.action_norm_mean:.5f}")
    print(f"  action norm max:              {obs.action_norm_max:.5f}")
    print(f"  qpos std mean across sample:  {obs.qpos_std_mean:.6f}")
    print(flush=True)


def print_training_summary(
    summary: dict[str, float | int | str],
    obs: SampleObservation,
) -> None:
    """Print final throughput and model summary in the same readable format."""

    print()
    print("################################################################################")
    print("Training physics summary")
    print("################################################################################")
    for key in (
        "device",
        "num_envs",
        "steps",
        "physics_steps",
        "compile_wall_s",
        "step_wall_s",
        "physics_steps_per_sec",
        "aggregate_sim_seconds",
        "aggregate_sim_seconds_per_wall_second",
        "sim_time_env0",
    ):
        print(f"  {key}: {summary[key]}")
    print(f"  final_reward_mean: {obs.reward_mean}")
    print(f"  final_progress_mean: {obs.progress_mean}")
    print(f"  final_lateral_error_mean: {obs.lateral_error_mean}")
    print(f"  final_contact_normal_force_mean: {obs.contact_force_sum_mean}")
    print(f"  final_max_penetration: {obs.max_penetration}")
    print(f"  final_action_norm_mean: {obs.action_norm_mean}")
    print(f"  final_qpos_std_mean: {obs.qpos_std_mean}")
    print()
