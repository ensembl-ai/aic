#!/usr/bin/env python3
"""Train the direct AIC MuJoCo prototype env with RSL-RL PPO.

Fresh run from a new ``aic_eval`` distrobox terminal:

cd /home/rmalhan/Software/ws_aic/src/aic
pixi shell
source /opt/ros/kilted/setup.bash
source /home/rmalhan/Software/ws_aic/install/setup.bash
export PYTHONNOUSERSITE=1
export MUJOCO_PLUGIN_PATH=/home/rmalhan/Software/ws_aic/install/opt/mujoco_vendor/lib
export XDG_CACHE_HOME=/tmp/$USER-cache
mkdir -p "$XDG_CACHE_HOME"

PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
python3 aic_utils/aic_mujoco/scripts/train_rsl_rl_direct.py \
--num-envs 4 \
--max-iterations 2 \
--num-steps-per-env 4 \
--device cuda

This is real PPO training: RSL-RL builds an actor/critic model, rollout
storage, optimizer, TensorBoard logs, and checkpoints. The simulator side uses
the direct MuJoCo prototype env: reset solves pre-insertion IK per environment,
then actions are Cartesian deltas stepped through TCP-Jacobian differential IK
and joint impedance. The next performance step is replacing the Python
``MjData`` loop with batched MuJoCo Warp state.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from aic_mujoco.warp import AicInsertionVecEnv, AicInsertionVecEnvConfig
from aic_mujoco.warp.rsl_rl_cfg import make_rsl_rl_direct_ppo_cfg
from aic_mujoco.warp.rsl_rl_wrapper import RslRlDirectWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_XML = PACKAGE_ROOT / "mjcf" / "scene.xml"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiments" / "train_warp_smoke.json"
DEFAULT_RUNS_DIR = PACKAGE_ROOT / "runs"


def make_parser() -> argparse.ArgumentParser:
    """Build CLI for the direct RSL-RL PPO prototype run."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--num-steps-per-env", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", default="aic_direct_ppo")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    return parser


def main() -> int:
    """Create env, wrap it for RSL-RL, train PPO, and save final checkpoint."""

    args = make_parser().parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = (
        Path(args.runs_dir).expanduser().resolve() / f"{run_stamp}_{args.run_name}"
    )
    log_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(int(args.seed))
    print("[rsl-rl-direct] creating env")
    env = AicInsertionVecEnv(
        AicInsertionVecEnvConfig.from_files(
            xml_path=Path(args.xml).expanduser().resolve(),
            config_path=Path(args.config).expanduser().resolve(),
            num_envs=int(args.num_envs),
            device=str(args.device),
        )
    )
    wrapped = RslRlDirectWrapper(env)
    train_cfg = make_rsl_rl_direct_ppo_cfg(
        num_steps_per_env=int(args.num_steps_per_env),
        max_iterations=int(args.max_iterations),
        save_interval=int(args.save_interval),
        run_name=str(args.run_name),
        seed=int(args.seed),
    )

    print("[rsl-rl-direct] dimensions")
    print(f"  obs:              {wrapped.num_obs}")
    print(f"  actions:          {wrapped.num_actions}")
    print(f"  envs:             {wrapped.num_envs}")
    print(f"  device:           {wrapped.device}")
    print(f"  physics_dt:       {env.physics_dt:.6f}")
    print(f"  control_dt:       {env.control_dt:.6f}")
    print(f"  log_dir:          {log_dir}")

    print("[rsl-rl-direct] constructing actor/critic and PPO runner")
    runner = OnPolicyRunner(
        env=wrapped,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device=str(args.device),
    )

    print("[rsl-rl-direct] learning")
    runner.learn(num_learning_iterations=int(args.max_iterations))
    final_path = log_dir / "final.pt"
    runner.save(str(final_path))
    print(f"[rsl-rl-direct] saved {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
