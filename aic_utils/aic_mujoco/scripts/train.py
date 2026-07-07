#!/usr/bin/env python3
"""Run the AIC insertion training physics loop.

Fresh run from a new ``aic_eval`` distrobox terminal:

cd /home/rmalhan/Software/ws_aic/src/aic
pixi shell
export PYTHONNOUSERSITE=1
export MUJOCO_PLUGIN_PATH=/home/rmalhan/Software/ws_aic/install/opt/mujoco_vendor/lib
export XDG_CACHE_HOME=/tmp/$USER-cache
mkdir -p "$XDG_CACHE_HOME"

python3 aic_utils/aic_mujoco/scripts/prepare_training_scene.py

PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
python3 aic_utils/aic_mujoco/scripts/train.py \
--config /home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco/configs/experiments/train.json \
--num-envs 4096 \
--steps 1000 \
--log-interval 100 \
--device cuda

This command is intentionally only the real batched training-physics path:

  MuJoCo XML -> mjModel -> mujoco_warp.put_model/make_data -> mujoco_warp.step

It does not run the older CPU ``MjData`` vector prototype and does not fall
back to CPU. If ``--device cuda`` cannot initialize, the run should fail. The
host seed data is initialized from ``reset_q`` in ``--config`` before it is
uploaded to Warp, so headless training and Viser start from the same posture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aic_mujoco.warp import (
    TrainingPhysicsConfig,
    run_training_physics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_WARP_XML = PACKAGE_ROOT / "mjcf" / "scene_warp.xml"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiments" / "train.json"


def make_parser() -> argparse.ArgumentParser:
    """Build CLI for the AIC insertion training physics command.

    The command measures batched physics stepping and prints periodic
    throughput. It is the first training command before policy/control kernels
    are added on top of the same batched state.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--warp-xml", default=str(DEFAULT_WARP_XML))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sample-envs", type=int, default=16)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    """Run batched training physics and print periodic throughput logs."""

    args = make_parser().parse_args()
    warp_xml = Path(args.warp_xml).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    print("[training-physics] batched MuJoCo Warp physics")
    run_training_physics(
        TrainingPhysicsConfig(
            xml_path=warp_xml,
            num_envs=int(args.num_envs),
            steps=int(args.steps),
            device=str(args.device),
            log_interval=int(args.log_interval),
            config_path=config_path,
            sample_envs=int(args.sample_envs),
            motion_scale=float(args.motion_scale),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
