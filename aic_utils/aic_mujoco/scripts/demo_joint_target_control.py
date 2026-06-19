#!/usr/bin/env python3
"""
Pre-insertion Cartesian policy demo using the generic MuJoCo joint-impedance
layer.

This is still "target -> joint torque" control:

  start from the same SFP-tip pre-insertion pose as hold_fixed_target.py
    -> vanilla policy emits small Cartesian deltas in world -Z
    -> damped least-squares IK gives q_des
    -> JointImpedanceController converts q_des -> joint torques
    -> MuJoCo steps physics

No ROS.
No gripper controller.
Gripper joints are passive/frozen by default.
End-effector orientation is not commanded; this is translation-only IK.

Run:
  PYTHONPATH=. python scripts/demo_joint_target_control.py \
    --xml /path/to/scene.xml \
    --ee-site gripper_tcp

If your EE site name is different:
  PYTHONPATH=. python scripts/demo_joint_target_control.py --xml scene.xml --print-sites
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from aic_mujoco.joints import (
    JointGroup,
    PassiveJointGroup,
    TorqueMode,
    print_model_summary,
)
from aic_mujoco.controllers import (
    JointImpedanceConfig,
    JointImpedanceController,
    JointTarget,
)
from aic_mujoco.utils import compute_preinsert_joint_target

DEFAULT_CONFIG = {
    "controlled_joints": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
    "passive_joints": [
        "gripper/left_finger_joint",
        "gripper/right_finger_joint",
    ],
    "passive_mode": "freeze",
    "torque_mode": "actuator_ctrl",
    "kp": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
    "kd": [40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
    "torque_limits": [120.0, 120.0, 120.0, 60.0, 60.0, 60.0],
    "torque_rate_limits": [1200.0, 1200.0, 1200.0, 600.0, 600.0, 600.0],
    "use_bias_compensation": True,
    "ik_home_q": [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110],
    "preinsert_port_body": "sfp_port_0_link_entrance",
    "preinsert_tcp_site": "gripper_tcp",
    "preinsert_sfp_tip_body": "sfp_tip_link",
    "preinsert_tool_body": "ati/tool_link",
    "preinsert_weld_child_body": "lc_plug_link",
    "preinsert_payload_root_body": "cable_end_0",
    "preinsert_payload_root_freejoint": "cable_end_0_free",
    "preinsert_height": 0.05,
}

DEFAULT_CAMERA_NAMES = ["center_camera", "left_camera", "right_camera"]
DEFAULT_FORCE_SENSOR = "AtiForceTorqueSensor_force"
DEFAULT_TORQUE_SENSOR = "AtiForceTorqueSensor_torque"


def load_config(path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        cfg.update(json.load(f))
    return cfg


def csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.replace(" ", ",").split(",") if x.strip()]


def print_sites_and_bodies(model: mujoco.MjModel) -> None:
    print("\nSites:")
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        print(f"  {sid:3d}: {name}")

    print("\nBodies:")
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  {bid:3d}: {name}")
    print()


def sensor_values(
    model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
) -> np.ndarray | None:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        return None

    start = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return np.asarray(data.sensordata[start : start + dim], dtype=float).copy()


def vector_summary(
    values: np.ndarray | None, baseline: np.ndarray | None = None
) -> str:
    if values is None:
        return "missing"
    if baseline is not None:
        values = values - baseline
    norm = float(np.linalg.norm(values))
    return "norm={:.3f} xyz=[{}]".format(
        norm,
        ", ".join(f"{x:+.3f}" for x in values),
    )


def delta_vector_summary(
    current: np.ndarray | None,
    previous: np.ndarray | None,
) -> str:
    if current is None:
        return "missing"
    if previous is None:
        return "n/a"
    return vector_summary(current - previous)


def zeroed_values(
    values: np.ndarray | None,
    baseline: np.ndarray | None,
) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if baseline is None:
        return values.copy()
    return values - baseline


def settle_and_zero_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: JointGroup,
    passive: PassiveJointGroup,
    controller: JointImpedanceController,
    q_hold: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    force_sensor: str,
    torque_sensor: str,
    settle_steps: int,
    bias_samples: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Settle after reset, then estimate raw force/torque sensor bias."""

    dt = float(model.opt.timestep)
    q_hold = np.asarray(q_hold, dtype=float)
    qd_hold = np.zeros(controlled.n)
    controller.reset(q_hold)

    def step_hold_pose() -> None:
        data.qfrc_applied[:] = 0.0
        passive.enforce(data)
        target = JointTarget.position(
            q_des=q_hold,
            qd_des=qd_hold,
            kp=kp,
            kd=kd,
            tau_ff=np.zeros(controlled.n),
        )
        tau = controller.compute(data, target, dt)
        controlled.apply_torque(data, tau)
        mujoco.mj_step(model, data)
        passive.enforce(data)
        mujoco.mj_forward(model, data)

    for _ in range(max(0, settle_steps)):
        step_hold_pose()

    force_values: list[np.ndarray] = []
    torque_values: list[np.ndarray] = []
    for _ in range(max(1, bias_samples)):
        step_hold_pose()
        force = sensor_values(model, data, force_sensor)
        torque = sensor_values(model, data, torque_sensor)
        if force is not None:
            force_values.append(force)
        if torque is not None:
            torque_values.append(torque)

    force_bias = None if not force_values else np.mean(np.asarray(force_values), axis=0)
    torque_bias = (
        None if not torque_values else np.mean(np.asarray(torque_values), axis=0)
    )
    passive.enforce(data)
    mujoco.mj_forward(model, data)
    return force_bias, torque_bias


def camera_names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


class CameraChecker:
    def __init__(
        self,
        model: mujoco.MjModel,
        camera_names: list[str],
        width: int,
        height: int,
        min_std: float,
    ):
        self.model = model
        self.camera_names = camera_names
        self.min_std = min_std
        self.renderer: mujoco.Renderer | None = None
        self.error: str | None = None

        missing = [
            name
            for name in camera_names
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0
        ]
        if missing:
            self.error = "missing camera(s): " + ", ".join(missing)
            return

        try:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:  # noqa: BLE001
            self.error = f"renderer init failed: {exc}"

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def summary(self, data: mujoco.MjData) -> str:
        if self.error is not None:
            return self.error
        assert self.renderer is not None

        parts: list[str] = []
        for name in self.camera_names:
            try:
                self.renderer.update_scene(data, camera=name)
                image = self.renderer.render()
                finite = bool(np.isfinite(image).all())
                mean = float(np.mean(image))
                std = float(np.std(image))
                ok = finite and image.size > 0 and std >= self.min_std
                status = "ok" if ok else "blank"
                parts.append(f"{name}:{status}(mean={mean:.1f},std={std:.1f})")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"{name}:error({exc})")
        return "; ".join(parts)


def make_downward_path(distance: float, step: float) -> np.ndarray:
    """Return world-frame offsets for a simple policy moving straight down."""

    distance = max(0.0, float(distance))
    step = max(np.finfo(float).eps, float(step))
    n = max(1, int(np.ceil(distance / step)))
    offsets = np.zeros((n + 1, 3), dtype=float)
    offsets[:, 2] = -np.linspace(0.0, distance, n + 1)
    return offsets


def solve_position_ik(
    model: mujoco.MjModel,
    seed_data: mujoco.MjData,
    controlled: JointGroup,
    site_id: int,
    target_pos: np.ndarray,
    q_seed: np.ndarray,
    max_iters: int,
    tolerance: float,
    damping: float,
    max_dq: float,
) -> tuple[np.ndarray, float, bool]:
    """Translation-only damped least-squares IK for the controlled joint group."""

    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = seed_data.qpos
    ik_data.qvel[:] = 0.0

    q = np.asarray(q_seed, dtype=float).copy()
    lo, hi = controlled.joint_limits()

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    cols = np.array([h.qvel_addr for h in controlled.handles], dtype=int)

    last_err_norm = np.inf

    for _ in range(max_iters):
        controlled.set_q(ik_data, q, zero_velocity=True)
        mujoco.mj_forward(model, ik_data)

        current_pos = ik_data.site_xpos[site_id].copy()
        err = target_pos - current_pos
        err_norm = float(np.linalg.norm(err))
        last_err_norm = err_norm

        if err_norm <= tolerance:
            return q, err_norm, True

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = jacp[:, cols]

        # Damped least-squares:
        # dq = J.T (J J.T + lambda^2 I)^-1 dx
        A = J @ J.T + (damping * damping) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_dq:
            dq *= max_dq / dq_norm

        q = q + dq
        q = np.minimum(np.maximum(q, lo), hi)

    return q, last_err_norm, False


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True)
    p.add_argument("--config", default=None)

    p.add_argument("--ee-site", default="gripper_tcp")
    p.add_argument(
        "--down-distance",
        type=float,
        default=0.08,
        help="World-Z insertion distance for the vanilla Cartesian policy.",
    )
    p.add_argument(
        "--step-size",
        type=float,
        default=0.001,
        help="Cartesian waypoint spacing in meters.",
    )
    p.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=0.0015,
        help="Advance when EE is this close.",
    )
    p.add_argument("--max-steps-per-waypoint", type=int, default=20)

    p.add_argument("--controlled-joints", type=csv_list, default=None)
    p.add_argument("--passive-joints", type=csv_list, default=None)
    p.add_argument("--passive-mode", choices=["free", "freeze"], default=None)
    p.add_argument("--torque-mode", choices=[m.value for m in TorqueMode], default=None)
    p.add_argument(
        "--home", type=csv_floats, default=None, help="Optional 6 joint start pose."
    )
    p.add_argument(
        "--start-mode",
        choices=["preinsert", "home", "config", "xml"],
        default="preinsert",
        help="Initial robot state before the Cartesian Z-down policy starts.",
    )
    p.add_argument("--ik-home-q", type=csv_floats, default=None)
    p.add_argument("--preinsert-port-body", default=None)
    p.add_argument("--preinsert-tcp-site", default=None)
    p.add_argument("--preinsert-sfp-tip-body", default=None)
    p.add_argument("--preinsert-tool-body", default=None)
    p.add_argument("--preinsert-weld-child-body", default=None)
    p.add_argument("--preinsert-payload-root-body", default=None)
    p.add_argument("--preinsert-payload-root-freejoint", default=None)
    p.add_argument(
        "--preinsert-height",
        type=float,
        default=None,
        help="World-Z height above the port entrance for the SFP tip start pose.",
    )
    p.add_argument("--no-bias-comp", action="store_true")
    p.add_argument("--torque-mode-fallback-qfrc", action="store_true")

    p.add_argument("--ik-iters", type=int, default=20)
    p.add_argument("--ik-tolerance", type=float, default=0.00025)
    p.add_argument("--ik-damping", type=float, default=0.03)
    p.add_argument("--ik-max-dq", type=float, default=0.03)

    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--log-period", type=float, default=1.0)
    p.add_argument("--force-sensor", default=DEFAULT_FORCE_SENSOR)
    p.add_argument("--torque-sensor", default=DEFAULT_TORQUE_SENSOR)
    p.add_argument(
        "--sensor-zero-settle-steps",
        type=int,
        default=100,
        help="Hold the reset pose for this many physics steps before estimating F/T bias.",
    )
    p.add_argument(
        "--sensor-zero-samples",
        type=int,
        default=50,
        help="Average this many startup readings as force/torque zero offsets.",
    )
    p.add_argument(
        "--no-sensor-zero",
        action="store_true",
        help="Print raw force/torque sensor values without subtracting startup offsets.",
    )
    p.add_argument(
        "--camera-names",
        type=camera_names,
        default=DEFAULT_CAMERA_NAMES,
        help="Comma-separated MuJoCo camera names to sanity-check.",
    )
    p.add_argument("--camera-check-width", type=int, default=160)
    p.add_argument("--camera-check-height", type=int, default=120)
    p.add_argument(
        "--camera-min-std",
        type=float,
        default=1.0,
        help="Minimum rendered image standard deviation to count as non-blank.",
    )
    p.add_argument("--no-camera-check", action="store_true")
    p.add_argument("--print-model", action="store_true")
    p.add_argument("--print-sites", action="store_true")
    return p


def main() -> int:
    args = make_parser().parse_args()
    cfg = load_config(args.config)

    if args.controlled_joints is not None:
        cfg["controlled_joints"] = args.controlled_joints
    if args.passive_joints is not None:
        cfg["passive_joints"] = args.passive_joints
    if args.passive_mode is not None:
        cfg["passive_mode"] = args.passive_mode
    if args.torque_mode is not None:
        cfg["torque_mode"] = args.torque_mode
    if args.ik_home_q is not None:
        cfg["ik_home_q"] = args.ik_home_q
    if args.preinsert_port_body is not None:
        cfg["preinsert_port_body"] = args.preinsert_port_body
    if args.preinsert_tcp_site is not None:
        cfg["preinsert_tcp_site"] = args.preinsert_tcp_site
    if args.preinsert_sfp_tip_body is not None:
        cfg["preinsert_sfp_tip_body"] = args.preinsert_sfp_tip_body
    if args.preinsert_tool_body is not None:
        cfg["preinsert_tool_body"] = args.preinsert_tool_body
    if args.preinsert_weld_child_body is not None:
        cfg["preinsert_weld_child_body"] = args.preinsert_weld_child_body
    if args.preinsert_payload_root_body is not None:
        cfg["preinsert_payload_root_body"] = args.preinsert_payload_root_body
    if args.preinsert_payload_root_freejoint is not None:
        cfg["preinsert_payload_root_freejoint"] = args.preinsert_payload_root_freejoint
    if args.preinsert_height is not None:
        cfg["preinsert_height"] = args.preinsert_height
    if args.no_bias_comp:
        cfg["use_bias_compensation"] = False

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml).expanduser().resolve()))
    data = mujoco.MjData(model)

    if args.print_model:
        print_model_summary(model)
        return 0
    if args.print_sites:
        print_sites_and_bodies(model)
        return 0

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.ee_site)
    if site_id < 0:
        raise RuntimeError(
            f"EE site {args.ee_site!r} not found. "
            "Run with --print-sites and pass the correct --ee-site."
        )

    torque_mode = TorqueMode(cfg["torque_mode"])
    try:
        controlled = JointGroup(
            model, cfg["controlled_joints"], torque_mode=torque_mode
        )
    except RuntimeError:
        if not args.torque_mode_fallback_qfrc:
            raise
        print("WARNING: actuator_ctrl mapping failed; falling back to qfrc_applied.")
        torque_mode = TorqueMode.QFRC_APPLIED
        controlled = JointGroup(
            model, cfg["controlled_joints"], torque_mode=torque_mode
        )

    passive = PassiveJointGroup(
        model,
        cfg.get("passive_joints", []),
        mode=cfg.get("passive_mode", "freeze"),
    )

    controller = JointImpedanceController(
        controlled,
        JointImpedanceConfig(
            use_bias_compensation=bool(cfg["use_bias_compensation"]),
            torque_limits=np.asarray(cfg["torque_limits"], dtype=float),
            torque_rate_limits=np.asarray(cfg["torque_rate_limits"], dtype=float),
            clamp_to_joint_limits=True,
        ),
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    if args.start_mode == "preinsert":
        q_start, preinsert_diagnostics = compute_preinsert_joint_target(
            model=model,
            data=data,
            controlled=controlled,
            home_q=np.asarray(cfg["ik_home_q"], dtype=float),
            port_body=cfg["preinsert_port_body"],
            tcp_site=cfg["preinsert_tcp_site"],
            sfp_tip_body=cfg["preinsert_sfp_tip_body"],
            tool_body=cfg["preinsert_tool_body"],
            weld_child_body=cfg["preinsert_weld_child_body"],
            height=float(cfg.get("preinsert_height", 0.05)),
            payload_root_body=cfg.get("preinsert_payload_root_body"),
            payload_root_freejoint=cfg.get("preinsert_payload_root_freejoint"),
        )
        desired_world_T_sfp_tip = preinsert_diagnostics["desired_world_T_sfp_tip"]
        actual_world_T_sfp_tip = preinsert_diagnostics[
            "actual_world_T_sfp_tip_after_payload_set"
        ]
        assert isinstance(desired_world_T_sfp_tip, np.ndarray)
        assert isinstance(actual_world_T_sfp_tip, np.ndarray)
        sfp_tip_error = float(
            np.linalg.norm(
                desired_world_T_sfp_tip[:3, 3] - actual_world_T_sfp_tip[:3, 3]
            )
        )
        print("Preinsert start:")
        print(
            f"  tcp_T_sfp_tip_source: {preinsert_diagnostics['tcp_T_sfp_tip_source']}"
        )
        print(f"  ik_solutions:         {preinsert_diagnostics['solution_count']}")
        print(f"  desired_sfp_tip:      {desired_world_T_sfp_tip[:3, 3].tolist()}")
        print(f"  actual_sfp_tip:       {actual_world_T_sfp_tip[:3, 3].tolist()}")
        print(f"  sfp_tip_error_m:      {sfp_tip_error:.6g}")
    elif args.start_mode == "home":
        if args.home is None:
            raise ValueError("--start-mode home requires --home")
        q_start = np.asarray(args.home, dtype=float)
        if q_start.shape != (controlled.n,):
            raise ValueError(
                f"start q must have shape ({controlled.n},), got {q_start.shape}"
            )
        controlled.set_q(data, q_start, zero_velocity=True)
        mujoco.mj_forward(model, data)
    elif args.start_mode == "config":
        if cfg.get("target_q") is None:
            raise ValueError("--start-mode config requires target_q in config")
        q_start = np.asarray(cfg["target_q"], dtype=float)
        if q_start.shape != (controlled.n,):
            raise ValueError(
                f"start q must have shape ({controlled.n},), got {q_start.shape}"
            )
        controlled.set_q(data, q_start, zero_velocity=True)
        mujoco.mj_forward(model, data)
    else:
        q_start = controlled.q(data).copy()

    # Snapshot passive joints after the robot is placed.
    # This freezes gripper/passive joints at their current XML/reset values.
    passive.snapshot(data)

    q_current = controlled.q(data).copy()
    controller.reset(q_current)

    print(f"Initial controlled q: {q_current.tolist()}")

    kp = np.asarray(cfg["kp"], dtype=float)
    kd = np.asarray(cfg["kd"], dtype=float)

    start_pos = data.site_xpos[site_id].copy()
    offsets = make_downward_path(args.down_distance, args.step_size)
    target_positions = start_pos[None, :] + offsets

    waypoint_i = 0
    steps_on_waypoint = 0
    last_log = -1e9
    t0 = time.time()
    camera_checker = None
    if not args.no_camera_check:
        camera_checker = CameraChecker(
            model=model,
            camera_names=args.camera_names,
            width=args.camera_check_width,
            height=args.camera_check_height,
            min_std=args.camera_min_std,
        )
    force_baseline = None
    torque_baseline = None
    if not args.no_sensor_zero:
        force_baseline, torque_baseline = settle_and_zero_wrench(
            model=model,
            data=data,
            controlled=controlled,
            passive=passive,
            controller=controller,
            q_hold=q_current,
            kp=kp,
            kd=kd,
            force_sensor=args.force_sensor,
            torque_sensor=args.torque_sensor,
            settle_steps=args.sensor_zero_settle_steps,
            bias_samples=args.sensor_zero_samples,
        )
        q_current = controlled.q(data).copy()
        controller.reset(q_current)

    previous_observed_force = None
    previous_observed_torque = None

    controlled.print_mapping("Controlled joints")
    passive.print_mapping("Passive joints")
    print()
    print("Cartesian Z-down policy demo")
    print(f"  ee_site:              {args.ee_site}")
    print(f"  start_mode:           {args.start_mode}")
    print(f"  start_pos:            {start_pos.tolist()}")
    print(f"  policy_action:        delta_world_z_down")
    print(f"  down_distance:        {args.down_distance:.4f} m")
    print(f"  step_size:            {args.step_size:.4f} m")
    print(f"  waypoints per cycle:  {len(target_positions)}")
    print(f"  torque_mode:          {torque_mode.value}")
    print(f"  passive_mode:         {passive.mode}")
    print(f"  gripper controller:   none")
    print(f"  force_sensor:         {args.force_sensor}")
    print(f"  torque_sensor:        {args.torque_sensor}")
    print(
        f"  sensor_zeroing:       {'disabled' if args.no_sensor_zero else f'{args.sensor_zero_settle_steps} settle steps, {args.sensor_zero_samples} bias samples'}"
    )
    print(
        f"  force_zero:           {None if force_baseline is None else force_baseline.tolist()}"
    )
    print(
        f"  torque_zero:          {None if torque_baseline is None else torque_baseline.tolist()}"
    )
    if camera_checker is not None:
        print(f"  camera_check:         {', '.join(args.camera_names)}")
    else:
        print("  camera_check:         disabled")
    print()
    print("Close viewer to stop.")
    print()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t = time.time() - t0
                if args.duration > 0.0 and t >= args.duration:
                    break

                dt = float(model.opt.timestep)

                data.qfrc_applied[:] = 0.0
                passive.enforce(data)

                cart_target = target_positions[waypoint_i]

                q_seed = controlled.q(data)
                q_des, ik_err, ik_ok = solve_position_ik(
                    model=model,
                    seed_data=data,
                    controlled=controlled,
                    site_id=site_id,
                    target_pos=cart_target,
                    q_seed=q_seed,
                    max_iters=args.ik_iters,
                    tolerance=args.ik_tolerance,
                    damping=args.ik_damping,
                    max_dq=args.ik_max_dq,
                )

                joint_target = JointTarget.position(
                    q_des=q_des,
                    qd_des=np.zeros(controlled.n),
                    kp=kp,
                    kd=kd,
                    tau_ff=np.zeros(controlled.n),
                )

                tau = controller.compute(data, joint_target, dt)
                controlled.apply_torque(data, tau)

                mujoco.mj_step(model, data)

                passive.enforce(data)
                mujoco.mj_forward(model, data)
                viewer.sync()

                ee_pos = data.site_xpos[site_id].copy()
                ee_err = float(np.linalg.norm(cart_target - ee_pos))

                steps_on_waypoint += 1
                if (
                    ee_err <= args.waypoint_tolerance
                    or steps_on_waypoint >= args.max_steps_per_waypoint
                ):
                    waypoint_i += 1
                    steps_on_waypoint = 0
                    if waypoint_i >= len(target_positions):
                        waypoint_i = 0

                if t - last_log >= args.log_period:
                    last_log = t
                    force = sensor_values(model, data, args.force_sensor)
                    torque = sensor_values(model, data, args.torque_sensor)
                    observed_force = zeroed_values(force, force_baseline)
                    observed_torque = zeroed_values(torque, torque_baseline)
                    camera_summary = (
                        camera_checker.summary(data)
                        if camera_checker is not None
                        else "disabled"
                    )
                    print(
                        "t={:7.2f} | wp={:5d}/{:5d} | ee_err={:8.5f} m | ik_err={:8.5f} m | ik_ok={} | max_tau={:8.2f}".format(
                            t,
                            waypoint_i,
                            len(target_positions),
                            ee_err,
                            ik_err,
                            str(ik_ok),
                            float(np.max(np.abs(tau))),
                        )
                    )
                    print(f"  obs.force:  {vector_summary(observed_force)}")
                    print(f"  obs.torque: {vector_summary(observed_torque)}")
                    print(
                        f"  obs.delta_force:  {delta_vector_summary(observed_force, previous_observed_force)}"
                    )
                    print(
                        f"  obs.delta_torque: {delta_vector_summary(observed_torque, previous_observed_torque)}"
                    )
                    print(f"  obs.cameras: {camera_summary}")
                    previous_observed_force = (
                        None
                        if observed_force is None
                        else np.asarray(observed_force, dtype=float).copy()
                    )
                    previous_observed_torque = (
                        None
                        if observed_torque is None
                        else np.asarray(observed_torque, dtype=float).copy()
                    )

                time.sleep(dt)
    finally:
        if camera_checker is not None:
            camera_checker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
