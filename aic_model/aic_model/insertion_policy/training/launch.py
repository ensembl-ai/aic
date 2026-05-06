from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch AIC insertion PPO training.")
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--max_iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    args, extra = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[4]
    train_script = repo_root / "aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Isaac Lab training script not found: {train_script}")

    sys.argv = [
        str(train_script),
        "--task",
        "AIC-Insertion-v0",
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--seed",
        str(args.seed),
        "--run_name",
        args.run_name,
        *extra,
    ]
    if args.headless:
        sys.argv.append("--headless")
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
