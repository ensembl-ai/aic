"""Pretty console logging helpers for prototype MuJoCo RL loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from aic_mujoco.mjlab.observations import ContactObservation


@dataclass(frozen=True)
class DemoLogRecord:
    """Structured values printed by the Cartesian debug demo."""

    step: int
    sim_time: float
    wall_time: float
    waypoint_index: int
    waypoint_count: int
    target_delta_world: np.ndarray
    tcp_pos: np.ndarray
    tcp_error: float
    ik_error: float
    ik_ok: bool
    max_tau: float
    force: np.ndarray | None
    torque: np.ndarray | None
    camera_summary: str
    contacts: ContactObservation
    reward_total: float
    reward_terms: Mapping[str, float]
    progress: float
    normalized_progress: float
    remaining: float
    force_norm: float
    torque_norm: float
    excessive_force: bool


def format_vector(values: np.ndarray | None, precision: int = 3) -> str:
    """Format a vector with signs and fixed precision."""

    if values is None:
        return "missing"
    values = np.asarray(values, dtype=float)
    body = ", ".join(f"{x:+.{precision}f}" for x in values)
    return f"[{body}]"


def format_norm_vector(values: np.ndarray | None, precision: int = 3) -> str:
    """Format a vector plus its Euclidean norm."""

    if values is None:
        return "missing"
    values = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(values))
    return f"norm={norm:.{precision}f} value={format_vector(values, precision)}"


def format_reward_terms(terms: Mapping[str, float]) -> str:
    """Format named reward terms as compact scalar text."""

    if not terms:
        return "none"
    return ", ".join(f"{name}={value:+.5f}" for name, value in terms.items())


def format_demo_log(record: DemoLogRecord) -> str:
    """Format one compact multi-line RL-style step summary."""

    lines = [
        "",
        (
            f"step {record.step:06d}  sim {record.sim_time:7.3f}s  "
            f"wall {record.wall_time:6.2f}s  wp {record.waypoint_index:03d}/{record.waypoint_count:03d}"
        ),
        "─" * 88,
        (
            f"action       d_world={format_vector(record.target_delta_world)}"
        ),
        (
            f"control      tcp={format_vector(record.tcp_pos)}  "
            f"tcp_err={record.tcp_error:.4f}m  ik_err={record.ik_error:.4f}m  "
            f"ik={str(record.ik_ok):5s}  max_tau={record.max_tau:6.2f}Nm"
        ),
        (
            f"wrench       F={format_norm_vector(record.force)}  "
            f"T={format_norm_vector(record.torque)}"
        ),
        (
            f"progress     depth={record.progress:+.4f}m  "
            f"norm={record.normalized_progress:.3f}  remain={record.remaining:.4f}m"
        ),
        (
            f"reward       total={record.reward_total:+.5f}  "
            f"{format_reward_terms(record.reward_terms)}"
        ),
        (
            f"contact      max_penetration={record.contacts.max_penetration:.5f}m  "
            f"excessive_force={record.excessive_force}"
        ),
    ]
    lines.append(f"cameras      {record.camera_summary}")
    return "\n".join(lines)
