from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


OBSERVATION_DIM = 28
ACTION_DIM = 6
HIDDEN_DIMS = (512, 256, 128)


class InsertionActor(nn.Module):
    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class InsertionCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def load_actor_checkpoint(path: str | Path, device: str | torch.device) -> tuple[InsertionActor, torch.Tensor, torch.Tensor]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Insertion actor checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"actor", "obs_mean", "obs_std", "observation_dim", "action_dim"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Actor checkpoint missing keys: {sorted(missing)}")

    if int(checkpoint["observation_dim"]) != OBSERVATION_DIM:
        raise ValueError(
            f"Checkpoint observation_dim={checkpoint['observation_dim']} does not match {OBSERVATION_DIM}."
        )
    if int(checkpoint["action_dim"]) != ACTION_DIM:
        raise ValueError(f"Checkpoint action_dim={checkpoint['action_dim']} does not match {ACTION_DIM}.")

    actor = InsertionActor().to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    obs_mean = torch.as_tensor(checkpoint["obs_mean"], dtype=torch.float32, device=device).reshape(1, -1)
    obs_std = torch.as_tensor(checkpoint["obs_std"], dtype=torch.float32, device=device).reshape(1, -1)
    if obs_mean.shape[1] != OBSERVATION_DIM or obs_std.shape[1] != OBSERVATION_DIM:
        raise ValueError("Checkpoint observation normalization tensors have invalid shape.")
    if torch.any(obs_std <= 0.0):
        raise ValueError("Checkpoint obs_std must be strictly positive.")
    return actor, obs_mean, obs_std
