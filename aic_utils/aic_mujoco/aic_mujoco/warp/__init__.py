"""Direct MuJoCo/MuJoCo Warp prototype stack for AIC insertion.

This package is the default R&D path:

  scene.xml / scene_warp.xml -> MuJoCo model -> reset/action/observation/reward

The plain MuJoCo vector env preserves the known-good Cartesian pre-insertion
and impedance-control semantics. The Warp helpers validate the GPU-facing model
directly with ``mujoco_warp`` without using MJLab manager/entity composition.
"""

from .env import AicInsertionVecEnv, AicInsertionVecEnvConfig
from .rsl_rl_cfg import make_rsl_rl_direct_ppo_cfg
from .rsl_rl_wrapper import RslRlDirectWrapper
from .warp_smoke import WarpSmokeConfig, run_warp_smoke

__all__ = [
    "AicInsertionVecEnv",
    "AicInsertionVecEnvConfig",
    "RslRlDirectWrapper",
    "WarpSmokeConfig",
    "make_rsl_rl_direct_ppo_cfg",
    "run_warp_smoke",
]
