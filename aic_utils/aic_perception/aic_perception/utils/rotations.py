"""Batched rotation representations for learning, in torch.

The network regresses the continuous 6D representation of Zhou et al.,
"On the Continuity of Rotation Representations in Neural Networks" (CVPR
2019): the first two columns of a rotation matrix, re-orthonormalized with
Gram-Schmidt.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def matrix_from_6d(values: torch.Tensor) -> torch.Tensor:
    """Map ``(..., 6)`` network outputs to ``(..., 3, 3)`` rotation matrices.

    Args:
        values: Two stacked, unnormalized column vectors.

    Returns:
        Proper rotation matrices.
    """

    first = F.normalize(values[..., 0:3], dim=-1)
    second = values[..., 3:6]
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def angle_between(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two batches of rotation matrices.

    Args:
        first: ``(..., 3, 3)`` rotations.
        second: ``(..., 3, 3)`` rotations.

    Returns:
        ``(...)`` angles in radians.
    """

    trace = (first.transpose(-1, -2) @ second).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
