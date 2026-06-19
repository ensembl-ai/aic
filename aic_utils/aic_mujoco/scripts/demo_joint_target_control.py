#!/usr/bin/env python3
"""Lean pre-insertion Cartesian policy demo.

This script is intentionally a thin demo over the prototype MJLab-facing
modules:

  mjlab.reset:
    reset to the SFP-tip pre-insertion pose and zero the F/T sensor baseline

  mjlab.step:
    apply a Cartesian TCP target through translation-only IK and joint
    impedance

  mjlab.observations:
    read reset-zeroed force/torque and camera health diagnostics

The demo policy is deliberately boring: start at pre-insertion, then walk the
TCP down in world Z. That gives us the same behavior we were testing before,
but with code organized closer to the reset/step/observation split we need for
policy training.

Fresh run:

cd /home/rmalhan/Software/ws_aic/src/aic
pixi shell

PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
python3 aic_utils/aic_mujoco/scripts/demo_joint_target_control.py \
--xml /home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf/scene.xml \
--config /home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco/configs/aic_ur5e_hold.json

Useful demo knobs:

  --down-distance 0.03
  --step-size 0.001
  --duration 10
  --no-camera-check

The config supplies the robot joints, controller gains, pre-insertion frame
names, F/T sensor names, and reset-time zeroing settings. Do not source ROS for
this script; it uses MuJoCo directly.
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

from aic_mujoco.controllers import JointImpedanceConfig, JointImpedanceController
from aic_mujoco.joints import (
    JointGroup,
    PassiveJointGroup,
    TorqueMode,
    print_model_summary,
)
from aic_mujoco.mjlab.observations import (
    CameraHealthChecker,
    force_torque_observation,
    vector_summary,
)
from aic_mujoco.mjlab.reset import (
    PreinsertResetConfig,
    WrenchZeroingConfig,
    reset_preinsert_episode,
)
from aic_mujoco.mjlab.step import (
    CartesianStepConfig,
    make_downward_path,
    step_cartesian_position_target,
)

DEFAULT_CONFIG: dict[str, Any] = {
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
    "force_sensor": "AtiForceTorqueSensor_force",
    "torque_sensor": "AtiForceTorqueSensor_torque",
    "sensor_zero_settle_steps": 100,
    "sensor_zero_samples": 50,
    "camera_names": ["center_camera", "left_camera", "right_camera"],
}


def load_config(path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


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


def make_reset_config(cfg: dict[str, Any]) -> PreinsertResetConfig:
    return PreinsertResetConfig(
        home_q=tuple(float(x) for x in cfg["ik_home_q"]),
        port_body=str(cfg["preinsert_port_body"]),
        tcp_site=str(cfg["preinsert_tcp_site"]),
        sfp_tip_body=str(cfg["preinsert_sfp_tip_body"]),
        tool_body=str(cfg["preinsert_tool_body"]),
        weld_child_body=str(cfg["preinsert_weld_child_body"]),
        payload_root_body=str(cfg["preinsert_payload_root_body"]),
        payload_root_freejoint=str(cfg["preinsert_payload_root_freejoint"]),
        height=float(cfg["preinsert_height"]),
    )


def make_wrench_config(cfg: dict[str, Any]) -> WrenchZeroingConfig:
    return WrenchZeroingConfig(
        force_sensor=str(cfg["force_sensor"]),
        torque_sensor=str(cfg["torque_sensor"]),
        settle_steps=int(cfg["sensor_zero_settle_steps"]),
        bias_samples=int(cfg["sensor_zero_samples"]),
        enabled=True,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--down-distance", type=float, default=0.08)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--log-period", type=float, default=1.0)
    parser.add_argument("--no-camera-check", action="store_true")
    parser.add_argument("--print-model", action="store_true")
    parser.add_argument("--print-sites", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    cfg = load_config(args.config)

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml).expanduser().resolve()))
    data = mujoco.MjData(model)

    if args.print_model:
        print_model_summary(model)
        return 0
    if args.print_sites:
        print_sites_and_bodies(model)
        return 0

    tcp_site = str(cfg["preinsert_tcp_site"])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
    if site_id < 0:
        raise RuntimeError(f"TCP site not found in MuJoCo model: {tcp_site!r}")

    torque_mode = TorqueMode(cfg["torque_mode"])
    controlled = JointGroup(
        model,
        cfg["controlled_joints"],
        torque_mode=torque_mode,
    )
    passive = PassiveJointGroup(
        model,
        cfg["passive_joints"],
        mode=str(cfg["passive_mode"]),
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

    kp = np.asarray(cfg["kp"], dtype=float)
    kd = np.asarray(cfg["kd"], dtype=float)
    reset_result = reset_preinsert_episode(
        model=model,
        data=data,
        controlled=controlled,
        passive=passive,
        controller=controller,
        kp=kp,
        kd=kd,
        reset_cfg=make_reset_config(cfg),
        wrench_cfg=make_wrench_config(cfg),
    )

    start_pos = data.site_xpos[site_id].copy()
    target_positions = start_pos[None, :] + make_downward_path(
        args.down_distance,
        args.step_size,
    )

    camera_checker = None
    if not args.no_camera_check:
        camera_checker = CameraHealthChecker(
            model=model,
            camera_names=list(cfg["camera_names"]),
        )

    diagnostics = reset_result.preinsert_diagnostics
    desired_sfp_tip = diagnostics["desired_world_T_sfp_tip"]
    actual_sfp_tip = diagnostics["actual_world_T_sfp_tip_after_payload_set"]
    assert isinstance(desired_sfp_tip, np.ndarray)
    assert isinstance(actual_sfp_tip, np.ndarray)

    controlled.print_mapping("Controlled joints")
    passive.print_mapping("Passive joints")
    print()
    print("Cartesian Z-down policy demo")
    print(f"  tcp_site:             {tcp_site}")
    print(f"  start_pos:            {start_pos.tolist()}")
    print(f"  policy_action:        delta_world_z_down")
    print(f"  down_distance:        {args.down_distance:.4f} m")
    print(f"  step_size:            {args.step_size:.4f} m")
    print(f"  waypoints per cycle:  {len(target_positions)}")
    print(f"  torque_mode:          {torque_mode.value}")
    print(f"  passive_mode:         {passive.mode}")
    print(f"  q_start:              {reset_result.q_start.tolist()}")
    print(
        "  sfp_tip_error_m:      "
        f"{float(np.linalg.norm(desired_sfp_tip[:3, 3] - actual_sfp_tip[:3, 3])):.6g}"
    )
    print(
        "  sensor_zeroing:       "
        f"{cfg['sensor_zero_settle_steps']} settle steps, {cfg['sensor_zero_samples']} bias samples"
    )
    print(
        f"  force_zero:           {None if reset_result.force_bias is None else reset_result.force_bias.tolist()}"
    )
    print(
        f"  torque_zero:          {None if reset_result.torque_bias is None else reset_result.torque_bias.tolist()}"
    )
    print("  gripper controller:   none")
    print()
    print("Close viewer to stop.")
    print()

    waypoint_i = 0
    steps_on_waypoint = 0
    last_log = -1e9
    t0 = time.time()
    step_cfg = CartesianStepConfig()
    waypoint_tolerance = 0.0015
    max_steps_per_waypoint = 20

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t = time.time() - t0
                if args.duration > 0.0 and t >= args.duration:
                    break

                result = step_cartesian_position_target(
                    model=model,
                    data=data,
                    controlled=controlled,
                    passive=passive,
                    controller=controller,
                    site_id=site_id,
                    target_pos=target_positions[waypoint_i],
                    kp=kp,
                    kd=kd,
                    cfg=step_cfg,
                )
                viewer.sync()

                steps_on_waypoint += 1
                if (
                    result.ee_error <= waypoint_tolerance
                    or steps_on_waypoint >= max_steps_per_waypoint
                ):
                    waypoint_i += 1
                    steps_on_waypoint = 0
                    if waypoint_i >= len(target_positions):
                        waypoint_i = 0

                if t - last_log >= args.log_period:
                    last_log = t
                    obs = force_torque_observation(
                        model=model,
                        data=data,
                        force_sensor=str(cfg["force_sensor"]),
                        torque_sensor=str(cfg["torque_sensor"]),
                        force_bias=reset_result.force_bias,
                        torque_bias=reset_result.torque_bias,
                    )
                    camera_summary = (
                        "disabled"
                        if camera_checker is None
                        else camera_checker.summary(data)
                    )
                    print(
                        "t={:7.2f} | wp={:5d}/{:5d} | ee_err={:8.5f} m | ik_err={:8.5f} m | ik_ok={} | max_tau={:8.2f}".format(
                            t,
                            waypoint_i,
                            len(target_positions),
                            result.ee_error,
                            result.ik_error,
                            str(result.ik_ok),
                            float(np.max(np.abs(result.tau))),
                        )
                    )
                    print(f"  obs.force:   {vector_summary(obs.force)}")
                    print(f"  obs.torque:  {vector_summary(obs.torque)}")
                    print(f"  obs.cameras: {camera_summary}")

                time.sleep(float(model.opt.timestep))
    finally:
        if camera_checker is not None:
            camera_checker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
