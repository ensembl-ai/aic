#!/usr/bin/env python3
"""
Cartesian cube-vertex traversal demo using the existing generic MuJoCo joint
impedance layer.

This is still "target -> joint torque" control:

  Cartesian 1 mm waypoint
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
    --ee-site AtiForceTorqueSensor

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
}


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


def make_cube_vertex_path(side: float, step: float) -> np.ndarray:
    """Return 1 mm-ish waypoints around cube vertices and back to start.

    Sequence visits all 8 cube vertices:

      bottom square:
        (0,0,0) -> (L,0,0) -> (L,L,0) -> (0,L,0) -> (0,0,0)

      go up:
        (0,0,L)

      top square:
        (L,0,L) -> (L,L,L) -> (0,L,L) -> (0,0,L)

      return down:
        (0,0,0)

    Each edge is discretized at `step`, e.g. 0.001 m.
    """

    L = float(side)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [L, 0.0, 0.0],
            [L, L, 0.0],
            [0.0, L, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, L],
            [L, 0.0, L],
            [L, L, L],
            [0.0, L, L],
            [0.0, 0.0, L],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    waypoints: list[np.ndarray] = []
    for a, b in zip(vertices[:-1], vertices[1:]):
        delta = b - a
        dist = float(np.linalg.norm(delta))
        n = max(1, int(round(dist / step)))

        # endpoint=False avoids duplicating vertices between segments.
        for i in range(n):
            alpha = i / n
            waypoints.append(a + alpha * delta)

    waypoints.append(vertices[-1])
    return np.asarray(waypoints, dtype=float)


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

    p.add_argument("--ee-site", default="AtiForceTorqueSensor")
    p.add_argument(
        "--side-length", type=float, default=0.10, help="Cube side length in meters."
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
    p.add_argument("--max-steps-per-waypoint", type=int, default=80)

    p.add_argument("--controlled-joints", type=csv_list, default=None)
    p.add_argument("--passive-joints", type=csv_list, default=None)
    p.add_argument("--passive-mode", choices=["free", "freeze"], default=None)
    p.add_argument("--torque-mode", choices=[m.value for m in TorqueMode], default=None)
    p.add_argument(
        "--home", type=csv_floats, default=None, help="Optional 6 joint start pose."
    )
    p.add_argument("--no-bias-comp", action="store_true")
    p.add_argument("--torque-mode-fallback-qfrc", action="store_true")

    p.add_argument("--ik-iters", type=int, default=20)
    p.add_argument("--ik-tolerance", type=float, default=0.00025)
    p.add_argument("--ik-damping", type=float, default=0.03)
    p.add_argument("--ik-max-dq", type=float, default=0.03)

    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--log-period", type=float, default=1.0)
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

    # Choose initial robot configuration.
    # Priority:
    #   1. --home CLI argument
    #   2. config target_mode == "explicit" and target_q
    #   3. XML/MuJoCo qpos0
    if args.home is not None:
        q_start = np.asarray(args.home, dtype=float)
    elif cfg.get("target_mode", "initial") == "explicit":
        if cfg.get("target_q") is None:
            raise ValueError('config has target_mode="explicit" but target_q is null')
        q_start = np.asarray(cfg["target_q"], dtype=float)
    else:
        q_start = None

    if q_start is not None:
        if q_start.shape != (controlled.n,):
            raise ValueError(
                f"start q must have shape ({controlled.n},), got {q_start.shape}"
            )
        controlled.set_q(data, q_start, zero_velocity=True)
        mujoco.mj_forward(model, data)

    # Snapshot passive joints after the robot is placed.
    # This freezes gripper/passive joints at their current XML/reset values.
    passive.snapshot(data)

    q_current = controlled.q(data).copy()
    controller.reset(q_current)

    print(f"Initial controlled q: {q_current.tolist()}")

    kp = np.asarray(cfg["kp"], dtype=float)
    kd = np.asarray(cfg["kd"], dtype=float)

    start_pos = data.site_xpos[site_id].copy()
    offsets = make_cube_vertex_path(args.side_length, args.step_size)
    target_positions = start_pos[None, :] + offsets

    waypoint_i = 0
    steps_on_waypoint = 0
    last_log = -1e9
    t0 = time.time()

    controlled.print_mapping("Controlled joints")
    passive.print_mapping("Passive joints")
    print()
    print("Cartesian cube traversal demo")
    print(f"  ee_site:              {args.ee_site}")
    print(f"  start_pos:            {start_pos.tolist()}")
    print(f"  side_length:          {args.side_length:.4f} m")
    print(f"  step_size:            {args.step_size:.4f} m")
    print(f"  waypoints per cycle:  {len(target_positions)}")
    print(f"  torque_mode:          {torque_mode.value}")
    print(f"  passive_mode:         {passive.mode}")
    print(f"  gripper controller:   none")
    print()
    print("Close viewer to stop.")
    print()

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

            time.sleep(dt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
