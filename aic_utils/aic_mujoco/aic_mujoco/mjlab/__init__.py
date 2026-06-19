"""MJLab-facing skeleton for AIC MuJoCo policy training.

This package intentionally contains placeholders. The goal is to organize the
AIC task into MJLab concepts before committing to reward, observation, and
randomization formulas.
"""

from . import actions, events, observations, rewards, terminations
from .env_cfg import AicInsertionTaskSpec, make_aic_insertion_env_cfg

__all__ = [
    "AicInsertionTaskSpec",
    "actions",
    "events",
    "make_aic_insertion_env_cfg",
    "observations",
    "rewards",
    "terminations",
]
