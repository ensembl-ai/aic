"""Minimal RSL-RL PPO config for the direct AIC MuJoCo prototype env."""

from __future__ import annotations

from copy import deepcopy


def make_rsl_rl_direct_ppo_cfg(
    *,
    num_steps_per_env: int = 16,
    max_iterations: int = 100,
    save_interval: int = 25,
    run_name: str = "aic_direct_ppo",
    seed: int = 1,
) -> dict:
    """Return a small, explicit PPO config for the installed RSL-RL API.

    Args:
        num_steps_per_env: Rollout horizon collected before each PPO update.
        max_iterations: Number of PPO updates.
        save_interval: Checkpoint interval in learning iterations.
        run_name: Name shown in logs/checkpoint directories.
        seed: Torch/RSL-RL seed passed through the runner config.

    The config is deliberately plain Python data. That makes version mismatch
    errors easier to read and keeps this prototype independent of MJLab manager
    config classes.
    """

    return deepcopy(
        {
            "seed": int(seed),
            "run_name": run_name,
            "num_steps_per_env": int(num_steps_per_env),
            "max_iterations": int(max_iterations),
            "save_interval": int(save_interval),
            "logger": "tensorboard",
            "check_for_nan": True,
            "obs_groups": {
                "actor": ["policy"],
                "critic": ["policy"],
            },
            "actor": {
                "class_name": "rsl_rl.models:MLPModel",
                "hidden_dims": [256, 256],
                "activation": "elu",
                "obs_normalization": True,
                "distribution_cfg": {
                    "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                    "init_std": 0.5,
                    "std_type": "scalar",
                },
            },
            "critic": {
                "class_name": "rsl_rl.models:MLPModel",
                "hidden_dims": [256, 256],
                "activation": "elu",
                "obs_normalization": True,
            },
            "algorithm": {
                "class_name": "rsl_rl.algorithms:PPO",
                "num_learning_epochs": 4,
                "num_mini_batches": 4,
                "clip_param": 0.2,
                "gamma": 0.99,
                "lam": 0.95,
                "value_loss_coef": 1.0,
                "entropy_coef": 0.01,
                "learning_rate": 1.0e-3,
                "max_grad_norm": 1.0,
                "optimizer": "adam",
                "use_clipped_value_loss": True,
                "schedule": "adaptive",
                "desired_kl": 0.01,
                "normalize_advantage_per_mini_batch": False,
                "rnd_cfg": None,
                "symmetry_cfg": None,
            },
        }
    )
