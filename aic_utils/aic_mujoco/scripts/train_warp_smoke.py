#!/usr/bin/env python3
"""Run the direct AIC MuJoCo/Warp prototype smoke loop.

Fresh run from a new ``aic_eval`` distrobox terminal:

  cd /home/rmalhan/Software/ws_aic/src/aic
  pixi shell
  source /opt/ros/kilted/setup.bash
  source /home/rmalhan/Software/ws_aic/install/setup.bash
  export PYTHONNOUSERSITE=1
  export MUJOCO_PLUGIN_PATH=/home/rmalhan/Software/ws_aic/install/opt/mujoco_vendor/lib
  export XDG_CACHE_HOME=/tmp/$USER-cache
  mkdir -p "$XDG_CACHE_HOME"

  python3 aic_utils/aic_mujoco/scripts/prepare_warp_scene.py

  PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
  python3 aic_utils/aic_mujoco/scripts/train_warp_smoke.py \
    --num-envs 4 \
    --steps 20 \
    --warp-steps 1

The control env defaults to ``scene.xml`` because that is the known-good
controller/viewer scene with cable and sensors. The Warp smoke defaults to
``scene_warp.xml`` because Warp does not support the cable body plugin. The
Warp scene keeps the robot joints and actuators and rigidly attaches the
SFP/LC plug to the gripper.
The live pre-insertion reset uses ``aic_model.robot.EnsemblRobot`` for IK, so
the ROS package index must be sourced even though no ROS nodes are launched.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from aic_mujoco.warp import (
    AicInsertionVecEnv,
    AicInsertionVecEnvConfig,
    WarpSmokeConfig,
    run_warp_smoke,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_XML = PACKAGE_ROOT / "mjcf" / "scene.xml"
DEFAULT_WARP_XML = PACKAGE_ROOT / "mjcf" / "scene_warp.xml"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiments" / "train_warp_smoke.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--warp-xml", default=str(DEFAULT_WARP_XML))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warp-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-z", type=float, default=-0.25)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    control_xml = Path(args.xml).expanduser().resolve()
    warp_xml = Path(args.warp_xml).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    print("[warp-smoke] direct MuJoCo Warp preflight")
    warp_stats = run_warp_smoke(
        WarpSmokeConfig(
            xml_path=warp_xml,
            num_envs=int(args.num_envs),
            steps=int(args.warp_steps),
        )
    )
    for key, value in warp_stats.items():
        print(f"  {key}: {value}")

    print()
    print("[prototype-env] direct MuJoCo vector control smoke")
    env = AicInsertionVecEnv(
        AicInsertionVecEnvConfig.from_files(
            xml_path=control_xml,
            config_path=config_path,
            num_envs=int(args.num_envs),
            device=str(args.device),
        )
    )
    obs = env.reset()
    print(
        f"  obs_shape={tuple(obs.shape)} num_actions={env.num_actions} "
        f"physics_dt={env.physics_dt:.6f}s control_dt={env.control_dt:.6f}s"
    )

    action = torch.zeros(
        (env.num_envs, env.num_actions),
        dtype=torch.float32,
        device=env.device,
    )
    action[:, 2] = float(args.action_z)

    t0 = time.perf_counter()
    last_extras = {}
    for _ in range(int(args.steps)):
        obs, rewards, dones, last_extras = env.step(action)
    wall = time.perf_counter() - t0
    env_steps = int(args.steps) * env.num_envs
    physics_steps = env_steps * env.decimation

    print()
    print("[prototype-env] summary")
    print(f"  env_steps:        {env_steps}")
    print(f"  physics_steps:    {physics_steps}")
    print(f"  wall_time_s:      {wall:.4f}")
    print(f"  env_steps_per_s:  {env_steps / max(wall, 1e-9):.1f}")
    print(f"  phys_steps_per_s: {physics_steps / max(wall, 1e-9):.1f}")
    print(f"  reward_mean:      {float(rewards.mean().detach().cpu()):+.5f}")
    print(f"  done_count:       {int(dones.detach().cpu().numpy().sum())}")
    print(f"  obs_mean:         {float(obs.mean().detach().cpu()):+.5f}")
    print(f"  obs_std:          {float(obs.std().detach().cpu()):+.5f}")
    print(f"  extras:           {last_extras}")

    metrics = env.last_metrics[0]
    if metrics is not None:
        force_norm = metrics.force_terms.force_norm
        max_pen = metrics.contact.max_penetration
        progress = metrics.progress.normalized_progress
        print()
        print("[env0]")
        print(f"  progress:         {progress:+.5f}")
        print(f"  lateral_error_m:  {metrics.lateral_error:.6f}")
        print(f"  force_norm:       {force_norm:.5f}")
        print(f"  max_penetration:  {max_pen:.6f}")
        print(f"  ik_error_m:       {metrics.ik_error:.6f}")
        print(f"  tcp_error_m:      {metrics.tcp_error:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
