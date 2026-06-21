#!/usr/bin/env python3
"""Lean pre-insertion Cartesian policy demo.

This script is intentionally a thin demo over the local prototype modules:

  reset:
    reset to the SFP-tip pre-insertion pose and zero the F/T sensor baseline

  step:
    apply a Cartesian TCP target through translation-only IK and joint
    impedance

  observations:
    read reset-zeroed force/torque and camera health diagnostics

The demo policy is deliberately boring: start at pre-insertion, then walk the
TCP down in world Z. That gives us the same behavior we were testing before,
but with code organized closer to the reset/step/observation split we need for
policy training.

Fresh run from a new ``aic_eval`` distrobox terminal:

cd /home/rmalhan/Software/ws_aic/src/aic
pixi shell
source /opt/ros/kilted/setup.bash
source /home/rmalhan/Software/ws_aic/install/setup.bash
export PYTHONNOUSERSITE=1
export MUJOCO_PLUGIN_PATH=/home/rmalhan/Software/ws_aic/install/opt/mujoco_vendor/lib

PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
python3 aic_utils/aic_mujoco/scripts/demo_joint_target_control.py \
--xml /home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf/scene.xml

Useful demo knobs:

  --down-distance 0.15
  --step-size 0.001
  --duration 10
  --no-camera-check

The config supplies the robot joints, controller gains, pre-insertion frame
names, F/T sensor names, and reset-time zeroing settings. ROS is sourced only
so EnsemblRobot can find robot description packages for reset IK; no ROS nodes
are launched.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from aic_mujoco.config import load_json_config
from aic_mujoco.controllers import JointImpedanceConfig, JointImpedanceController
from aic_mujoco.joints import (
    JointGroup,
    PassiveJointGroup,
    TorqueMode,
    print_model_summary,
)
from aic_mujoco.mjlab.logging import DemoLogRecord, format_demo_log
from aic_mujoco.mjlab.observations import (
    CameraHealthChecker,
    contact_observation,
    force_torque_observation,
)
from aic_mujoco.mjlab.rewards import (
    compose_reward,
    insertion_axis_progress,
    zeroed_wrench_penalty,
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
from aic_mujoco.utils import body_transform

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    SCRIPT_DIR.parent / "configs" / "experiments" / "demo_cartesian_down.json"
)


def print_sites_and_bodies(model: mujoco.MjModel) -> None:
    """Print MuJoCo site and body names for config/debug alignment.

    The policy/reset config names must match the compiled MJCF names exactly.
    This helper gives a compact inventory when a frame name changes after a new
    SDF-to-MJCF conversion.
    """
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
    """Build the pre-insertion reset config from merged JSON settings.

    The reset layer needs only frame names, HOME joint state, payload root, and
    pre-insertion height. Keeping this translation here lets the demo override
    CLI values without leaking raw dictionaries into the reusable reset module.
    """
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
    """Build F/T zeroing settings for the reset-time bias estimator.

    The demo reports force/torque after subtracting a bias measured at reset.
    That mirrors how insertion policies usually consume wrist wrench feedback:
    relative to the current episode's unloaded/held-object baseline.
    """
    return WrenchZeroingConfig(
        force_sensor=str(cfg["force_sensor"]),
        torque_sensor=str(cfg["torque_sensor"]),
        settle_steps=int(cfg["sensor_zero_settle_steps"]),
        bias_samples=int(cfg["sensor_zero_samples"]),
        enabled=True,
    )


def make_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the lean Cartesian down-motion demo.

    The XML path is explicit because this demo is meant to be run against
    regenerated scenes. Most behavior comes from config so the command stays
    short and comparable to the training environment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--down-distance", type=float, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--log-period", type=float, default=None)
    parser.add_argument("--no-camera-check", action="store_true")
    parser.add_argument("--print-model", action="store_true")
    parser.add_argument("--print-sites", action="store_true")
    return parser


def main() -> int:
    """Run a visible single-environment reset, observation, and action loop.

    The script is the human-facing smoke test for the same building blocks used
    by training: pre-insertion IK reset, zeroed wrench observation, Cartesian
    delta stepping through differential IK, impedance torque control, reward
    reporting, and MuJoCo viewer synchronization.
    """
    args = make_parser().parse_args()
    cfg = load_json_config(args.config)
    if args.down_distance is not None:
        cfg["down_distance"] = args.down_distance
    if args.step_size is not None:
        cfg["step_size"] = args.step_size
    if args.log_period is not None:
        cfg["log_period"] = args.log_period

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
        float(cfg["down_distance"]),
        float(cfg["step_size"]),
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
    print(f"  config:               {Path(args.config).expanduser().resolve()}")
    print(f"  down_distance:        {float(cfg['down_distance']):.4f} m")
    print(f"  step_size:            {float(cfg['step_size']):.4f} m")
    print(f"  waypoints per cycle:  {len(target_positions)}")
    print(f"  torque_mode:          {torque_mode.value}")
    print(f"  passive_mode:         {passive.mode}")
    print(f"  sim_timestep:         {float(model.opt.timestep):.6f} s")
    print(f"  reward_port_bottom:   {cfg['reward_port_bottom_body']}")
    print(
        "  reward_weights:       "
        f"progress={float(cfg['reward_progress_weight']):.3g}, "
        f"force={float(cfg['reward_force_weight']):.3g}, "
        f"action={float(cfg['reward_action_weight']):.3g}"
    )
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
    physics_step = 0
    step_cfg = CartesianStepConfig()
    waypoint_tolerance = float(cfg["waypoint_tolerance"])
    max_steps_per_waypoint = int(cfg["max_steps_per_waypoint"])

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t = time.time() - t0
                if args.duration > 0.0 and t >= args.duration:
                    break

                target_pos = target_positions[waypoint_i]
                result = step_cartesian_position_target(
                    model=model,
                    data=data,
                    controlled=controlled,
                    passive=passive,
                    controller=controller,
                    site_id=site_id,
                    target_pos=target_pos,
                    kp=kp,
                    kd=kd,
                    cfg=step_cfg,
                )
                physics_step += 1
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

                if t - last_log >= float(cfg["log_period"]):
                    last_log = t
                    target_delta = target_pos - start_pos
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
                    contacts = contact_observation(model=model, data=data)

                    progress = insertion_axis_progress(
                        world_T_port_bottom=body_transform(
                            model, data, str(cfg["reward_port_bottom_body"])
                        ),
                        world_T_port_entrance=body_transform(
                            model, data, str(cfg["preinsert_port_body"])
                        ),
                        world_T_plug=body_transform(
                            model, data, str(cfg["preinsert_sfp_tip_body"])
                        ),
                    )
                    force_terms = zeroed_wrench_penalty(
                        force=obs.force,
                        torque=obs.torque,
                        force_limit=float(cfg["reward_force_limit"]),
                        torque_limit=None,
                        force_weight=1.0,
                        torque_weight=1.0,
                    )
                    action_penalty = -float(np.dot(target_delta, target_delta))
                    reward = compose_reward(
                        [
                            (
                                "progress",
                                progress.normalized_progress,
                                float(cfg["reward_progress_weight"]),
                            ),
                            (
                                "force",
                                force_terms.penalty,
                                float(cfg["reward_force_weight"]),
                            ),
                            (
                                "action",
                                action_penalty,
                                float(cfg["reward_action_weight"]),
                            ),
                        ]
                    )
                    excessive_force = force_terms.force_norm > float(
                        cfg["terminate_force_limit"]
                    )
                    print(
                        format_demo_log(
                            DemoLogRecord(
                                step=physics_step,
                                sim_time=float(getattr(data, "time", 0.0)),
                                wall_time=t,
                                waypoint_index=waypoint_i,
                                waypoint_count=len(target_positions),
                                target_delta_world=target_delta,
                                tcp_pos=result.ee_pos,
                                tcp_error=result.ee_error,
                                ik_error=result.ik_error,
                                ik_ok=result.ik_ok,
                                max_tau=float(np.max(np.abs(result.tau))),
                                force=obs.force,
                                torque=obs.torque,
                                camera_summary=camera_summary,
                                contacts=contacts,
                                reward_total=reward.total,
                                reward_terms=reward.terms,
                                progress=progress.progress,
                                normalized_progress=progress.normalized_progress,
                                remaining=progress.remaining,
                                force_norm=force_terms.force_norm,
                                torque_norm=force_terms.torque_norm,
                                excessive_force=excessive_force,
                            )
                        )
                    )

                time.sleep(float(model.opt.timestep))
    finally:
        if camera_checker is not None:
            camera_checker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
