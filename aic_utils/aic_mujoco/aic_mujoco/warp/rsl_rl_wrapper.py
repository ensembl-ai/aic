"""RSL-RL adapter for the direct AIC MuJoCo prototype environment."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from aic_mujoco.warp.env import AicInsertionVecEnv


class RslRlDirectWrapper(VecEnv):
    """Expose ``AicInsertionVecEnv`` through RSL-RL's ``VecEnv`` contract.

    The wrapper is intentionally thin: it does not change actions, rewards, or
    resets. Its job is only to adapt observations to TensorDict format and add
    the logging/time-out fields RSL-RL expects.
    """

    def __init__(self, env: AicInsertionVecEnv):
        """Store env dimensions and serialize config for RSL-RL logs."""

        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs
        self.cfg = _plain_cfg(env.cfg)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return episode lengths as a Torch tensor on the env device."""

        return torch.as_tensor(
            self.env.episode_length_buf,
            dtype=torch.long,
            device=self.device,
        )

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Allow RSL-RL to write episode lengths back into the env buffer."""

        self.env.episode_length_buf[:] = value.detach().cpu().numpy()

    def get_observations(self) -> TensorDict:
        """Return policy observations in the TensorDict key RSL-RL expects."""

        return _as_tensordict(self.env.get_observations())

    def step(self, actions: torch.Tensor):
        """Step the wrapped env and attach RSL-RL logging extras."""

        obs, rewards, dones, extras = self.env.step(actions)
        extras = dict(extras)
        extras["time_outs"] = (
            torch.as_tensor(
                self.env.episode_length_buf >= self.env.max_episode_length,
                dtype=torch.float32,
                device=self.device,
            )
            * dones.float()
        )
        extras["log"] = _log_dict(extras)
        return _as_tensordict(obs), rewards, dones, extras


def _as_tensordict(obs: torch.Tensor) -> TensorDict:
    """Wrap a policy observation tensor in ``TensorDict({'policy': obs})``."""

    return TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=obs.device)


def _plain_cfg(cfg: Any) -> dict[str, Any]:
    """Convert dataclasses/paths into logging-safe plain Python values."""

    if is_dataclass(cfg):
        cfg = asdict(cfg)
    if isinstance(cfg, dict):
        return {
            str(k): _plain_cfg(v)
            for k, v in cfg.items()
        }
    if isinstance(cfg, list):
        return [_plain_cfg(v) for v in cfg]
    if isinstance(cfg, tuple):
        return tuple(_plain_cfg(v) for v in cfg)
    if hasattr(cfg, "__fspath__"):
        return str(cfg)
    return cfg


def _log_dict(extras: dict[str, Any]) -> dict[str, float]:
    """Extract scalar episode metrics for RSL-RL console/TensorBoard logs."""

    episode = extras.get("episode", {})
    return {
        str(key): float(value)
        for key, value in episode.items()
        if isinstance(value, int | float)
    }
