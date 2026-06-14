#!/usr/bin/env python3
"""
Generic fixed-target hold demo.

Purpose:
  Test whether your MuJoCo-side AIC-style controller can hold a fixed arm pose
  under gravity.

Properties:
  - No ROS.
  - No gripper controller.
  - Gripper can be passive/free OR frozen via config.
  - Target is fixed to initial q unless specified.
  - All robot-specific choices live in config/CLI, not source code.

Run:
  PYTHONPATH=. python scripts/hold_fixed_target.py --xml /path/to/scene.xml

With config:
  PYTHONPATH=. python scripts/hold_fixed_target.py \
    --xml /path/to/scene.xml \
    --config configs/aic_ur5e_hold.json
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
    "passive_joints": ["gripper/left_finger_joint", "gripper/right_finger_joint"],
    "passive_mode": "freeze",
    "torque_mode": "actuator_ctrl",
    "target_mode": "initial",
    "target_q": None,
    "kp": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
    "kd": [40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
    "torque_limits": [120.0, 120.0, 120.0, 60.0, 60.0, 60.0],
    "torque_rate_limits": [1200.0, 1200.0, 1200.0, 600.0, 600.0, 600.0],
    "use_bias_compensation": True,
}


def load_config(path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg


def csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.replace(" ", ",").split(",") if x.strip()]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True)
    p.add_argument("--config", default=None, help="Optional JSON config file.")

    # CLI overrides. Use these only when you don't want to edit JSON.
    p.add_argument("--controlled-joints", type=csv_list, default=None)
    p.add_argument("--passive-joints", type=csv_list, default=None)
    p.add_argument("--passive-mode", choices=["free", "freeze"], default=None)
    p.add_argument("--torque-mode", choices=[m.value for m in TorqueMode], default=None)
    p.add_argument("--target-q", type=csv_floats, default=None)
    p.add_argument("--no-bias-comp", action="store_true")
    p.add_argument("--torque-mode-fallback-qfrc", action="store_true")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--log-period", type=float, default=1.0)
    p.add_argument("--print-model", action="store_true")
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
    if args.target_q is not None:
        cfg["target_q"] = args.target_q
        cfg["target_mode"] = "explicit"
    if args.no_bias_comp:
        cfg["use_bias_compensation"] = False

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml).expanduser().resolve()))
    data = mujoco.MjData(model)

    if args.print_model:
        print_model_summary(model)
        return 0

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
        mode=cfg.get("passive_mode", "free"),
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

    # Snapshot passive joints first. This preserves the XML initial gripper state.
    # If passive_mode='freeze', these joints will stay exactly where the XML put them.
    passive.snapshot(data)

    if cfg.get("target_mode", "initial") == "explicit":
        q_target = np.asarray(cfg["target_q"], dtype=float)
        controlled.set_q(data, q_target, zero_velocity=True)
    else:
        q_target = controlled.q(data).copy()

    qd_target = np.zeros(controlled.n)
    controller.reset(q_target)

    kp = np.asarray(cfg["kp"], dtype=float)
    kd = np.asarray(cfg["kd"], dtype=float)

    controlled.print_mapping("Controlled joints")
    passive.print_mapping("Passive joints")
    print()
    print("Fixed-target hold:")
    print(f"  torque_mode:           {torque_mode.value}")
    print(f"  q_target:              {q_target.tolist()}")
    print(f"  passive_mode:          {passive.mode}")
    print(f"  bias_compensation:     {cfg['use_bias_compensation']}")
    print("  gripper controller:    none")
    print()
    print("Close viewer to stop.")
    print()

    t0 = time.time()
    last_log = -1e9

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            if args.duration > 0.0 and t >= args.duration:
                break

            dt = float(model.opt.timestep)

            # Clear external forces. We only write controlled torques below.
            data.qfrc_applied[:] = 0.0

            # Keep passive joints frozen if configured. This is intentionally not
            # a gripper controller; it is a generic kinematic freeze.
            passive.enforce(data)

            target = JointTarget.position(
                q_des=q_target,
                qd_des=qd_target,
                kp=kp,
                kd=kd,
                tau_ff=np.zeros(controlled.n),
            )
            tau = controller.compute(data, target, dt)
            controlled.apply_torque(data, tau)

            mujoco.mj_step(model, data)

            # Enforce after integration too, so passive joints do not visibly drift.
            passive.enforce(data)
            mujoco.mj_forward(model, data)

            viewer.sync()

            if t - last_log >= args.log_period:
                last_log = t
                err = q_target - controlled.q(data)
                print(
                    "t={:7.2f} | q_err_norm={:10.6f} | max_abs_tau={:9.3f}".format(
                        t,
                        float(np.linalg.norm(err)),
                        float(np.max(np.abs(tau))),
                    )
                )

            time.sleep(dt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
