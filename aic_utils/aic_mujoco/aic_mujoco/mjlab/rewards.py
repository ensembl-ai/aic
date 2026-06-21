"""Reward utilities for AIC MuJoCo policy-training experiments.

The first reward direction mirrors the useful part of IndustReal's insertion
reward: query a signed-distance field at sampled plug surface points. This is
more informative than a handful of keypoints and less ambiguous than a raw
Chamfer distance to a cavity.

This module deliberately does not choose task-specific weights. It provides
small, reusable terms:

  SDF pose/alignment distance
  insertion-axis progress
  zeroed force/torque penalty
  weighted reward composition

Heavy geometry work is delegated to external libraries:

  trimesh: mesh loading, surface sampling, proximity/SDF queries
  scipy: KD-tree/proximity support used by trimesh
  rtree: acceleration structure used by trimesh proximity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


@dataclass(frozen=True)
class SdfRewardConfig:
    """Configuration for an IndustReal-style sampled SDF query reward."""

    num_surface_points: int = 1000
    epsilon: float = 1e-6
    log_scale: float = 1.0
    clamp_distance: float | None = 0.05
    use_absolute_distance: bool = True


@dataclass(frozen=True)
class SdfRewardTerms:
    """Distance and reward values produced by an SDF query."""

    reward: float
    mean_distance: float
    rms_distance: float
    max_distance: float
    signed_distances: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class InsertionProgressTerms:
    """Axial insertion progress relative to the port entrance/bottom line."""

    progress: float
    normalized_progress: float
    remaining: float


@dataclass(frozen=True)
class ForcePenaltyTerms:
    """Force/torque penalty values after reset-time wrench zeroing."""

    penalty: float
    force_norm: float
    torque_norm: float
    excess_force: float
    excess_torque: float


@dataclass(frozen=True)
class RewardComposition:
    """Weighted reward total plus named contribution dictionary."""

    total: float
    terms: dict[str, float]


def transform_points(frame_T_points: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an ``N x 3`` point cloud."""

    points = np.asarray(points, dtype=float)
    frame_T_points = np.asarray(frame_T_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if frame_T_points.shape != (4, 4):
        raise ValueError(
            f"frame_T_points must have shape (4, 4), got {frame_T_points.shape}"
        )

    return points @ frame_T_points[:3, :3].T + frame_T_points[:3, 3]


def load_mesh(mesh_path: str | Path):
    """Load a mesh with trimesh and require an actual mesh object."""

    mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Loaded empty mesh: {mesh_path}")
    return mesh


def sample_surface_points(mesh, count: int, seed: int | None = None) -> np.ndarray:
    """Sample approximately even surface points from a mesh."""

    if seed is not None:
        np.random.seed(seed)
    sampled = trimesh.sample.sample_surface_even(mesh, int(count))
    points = np.asarray(sampled[0], dtype=float)
    if len(points) < count:
        sampled = trimesh.sample.sample_surface(mesh, int(count))
        points = np.asarray(sampled[0], dtype=float)
    return points


class MeshSdfQuery:
    """Thin wrapper around ``trimesh.proximity.ProximityQuery``.

    Trimesh's signed-distance convention is positive inside a watertight mesh
    and negative outside. For an alignment reward we normally use absolute
    distance to the target surface. For a collision/interpenetration term
    against a port/wall solid, the positive part can be interpreted as
    penetration depth when the mesh is watertight and oriented consistently.
    """

    def __init__(self, mesh):
        """Create a proximity query around a target mesh."""

        self.mesh = mesh
        self.query = trimesh.proximity.ProximityQuery(mesh)

    @classmethod
    def from_mesh_path(cls, mesh_path: str | Path) -> "MeshSdfQuery":
        """Load a mesh from disk and create a signed-distance query object."""

        return cls(load_mesh(mesh_path))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Evaluate signed distance for ``N x 3`` world/query points."""

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        return np.asarray(self.query.signed_distance(points), dtype=float)


class SampledSdfPoseReward:
    """Reward current plug pose against a target plug occupancy SDF.

    Typical usage:

      plug_points = sample_surface_points(plug_mesh, 1000)
      target_sdf = MeshSdfQuery(target_plug_mesh_at_nominal_pose)
      reward = SampledSdfPoseReward(plug_points, target_sdf)

      terms = reward.evaluate(current_world_T_plug)

    The target SDF can be built from the same plug mesh transformed into the
    nominal inserted pose. Then sampled points from the current plug should land
    on that target surface when insertion/alignment is correct.
    """

    def __init__(
        self,
        plug_surface_points_in_plug: np.ndarray,
        target_sdf: MeshSdfQuery,
        cfg: SdfRewardConfig | None = None,
    ):
        """Store plug samples and target SDF query for repeated evaluation."""

        self.plug_surface_points_in_plug = np.asarray(
            plug_surface_points_in_plug,
            dtype=float,
        )
        self.target_sdf = target_sdf
        self.cfg = cfg or SdfRewardConfig()

    def evaluate(self, world_T_plug: np.ndarray) -> SdfRewardTerms:
        """Evaluate the sampled plug points at the current plug pose."""

        world_points = transform_points(
            world_T_plug,
            self.plug_surface_points_in_plug,
        )
        signed = self.target_sdf.signed_distance(world_points)
        distances = np.abs(signed) if self.cfg.use_absolute_distance else signed
        if self.cfg.clamp_distance is not None:
            distances = np.clip(
                distances,
                -self.cfg.clamp_distance,
                self.cfg.clamp_distance,
            )

        abs_distances = np.abs(distances)
        mean_distance = float(np.mean(abs_distances))
        rms_distance = float(np.sqrt(np.mean(abs_distances * abs_distances)))
        max_distance = float(np.max(abs_distances))
        reward = -self.cfg.log_scale * float(
            np.log(rms_distance + self.cfg.epsilon)
        )
        return SdfRewardTerms(
            reward=reward,
            mean_distance=mean_distance,
            rms_distance=rms_distance,
            max_distance=max_distance,
            signed_distances=signed,
        )


def insertion_axis_progress(
    world_T_port_bottom: np.ndarray,
    world_T_port_entrance: np.ndarray,
    world_T_plug: np.ndarray,
) -> InsertionProgressTerms:
    """Compute progress from port entrance toward port bottom.

    AIC scoring uses the port frame as the deeper endpoint and the
    ``*_link_entrance`` frame as the entrance. This helper projects the plug
    position onto the entrance->bottom axis and normalizes by that segment
    length.
    """

    bottom = np.asarray(world_T_port_bottom, dtype=float)[:3, 3]
    entrance = np.asarray(world_T_port_entrance, dtype=float)[:3, 3]
    plug = np.asarray(world_T_plug, dtype=float)[:3, 3]

    axis = bottom - entrance
    depth = float(np.linalg.norm(axis))
    if depth <= 1e-12:
        return InsertionProgressTerms(progress=0.0, normalized_progress=0.0, remaining=0.0)

    unit_axis = axis / depth
    progress = float(np.dot(plug - entrance, unit_axis))
    normalized = float(np.clip(progress / depth, 0.0, 1.0))
    remaining = float(max(0.0, depth - progress))
    return InsertionProgressTerms(
        progress=progress,
        normalized_progress=normalized,
        remaining=remaining,
    )


def zeroed_wrench_penalty(
    force: np.ndarray | None,
    torque: np.ndarray | None = None,
    force_limit: float = 20.0,
    torque_limit: float | None = None,
    force_weight: float = 1.0,
    torque_weight: float = 1.0,
) -> ForcePenaltyTerms:
    """Penalty for exceeding zeroed force/torque limits."""

    force_norm = 0.0 if force is None else float(np.linalg.norm(force))
    torque_norm = 0.0 if torque is None else float(np.linalg.norm(torque))
    excess_force = max(0.0, force_norm - float(force_limit))
    if torque_limit is None:
        excess_torque = 0.0
    else:
        excess_torque = max(0.0, torque_norm - float(torque_limit))
    penalty = -(force_weight * excess_force + torque_weight * excess_torque)
    return ForcePenaltyTerms(
        penalty=penalty,
        force_norm=force_norm,
        torque_norm=torque_norm,
        excess_force=excess_force,
        excess_torque=excess_torque,
    )


def compose_reward(weighted_terms: Iterable[tuple[str, float, float]]) -> RewardComposition:
    """Combine ``(name, value, weight)`` reward terms."""

    terms: dict[str, float] = {}
    total = 0.0
    for name, value, weight in weighted_terms:
        contribution = float(value) * float(weight)
        terms[name] = contribution
        total += contribution
    return RewardComposition(total=total, terms=terms)


def action_regularization(action: np.ndarray) -> float:
    """Return ``-||action||^2`` for policy action magnitude regularization."""

    action = np.asarray(action, dtype=float)
    return -float(np.dot(action, action))


def penetration_penalty(max_penetration: float) -> float:
    """Return a negative reward contribution from maximum contact penetration."""

    return -float(max(0.0, max_penetration))
