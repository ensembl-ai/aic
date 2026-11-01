"""Random AIC task-board scenes expressed as ``task_board.urdf.xacro`` arguments.

Sampling is a pure function of ``(seed, scene_id)``, so any worker can
regenerate any scene and an interrupted collection resumes deterministically.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

NIC_MOUNTS = tuple(f"nic_card_mount_{index}" for index in range(5))
SC_PORTS = tuple(f"sc_port_{index}" for index in range(5))
SC_RAILS = (SC_PORTS[:3], SC_PORTS[3:])
FIXTURE_MOUNTS = tuple(
    f"{kind}_mount_rail_{rail}" for rail in (0, 1) for kind in ("lc", "sfp", "sc")
)


def scene_rng(seed: int, scene_id: int) -> np.random.Generator:
    """Create the independent random stream for one scene.

    Args:
        seed: Collection seed.
        scene_id: Global scene index.

    Returns:
        Seeded generator.
    """

    return np.random.default_rng([seed, scene_id])


def sample_components(
    rng: np.random.Generator, names: tuple[str, ...], settings: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Choose a random subset of rail slots and their offsets.

    Args:
        rng: Scene random stream.
        names: Candidate slot names.
        settings: ``count``, ``translation``, and ``yaw_deg`` ranges.

    Returns:
        Present slots mapped to translation and yaw.
    """

    count = int(rng.integers(settings["count"][0], settings["count"][1] + 1))
    chosen = sorted(rng.choice(len(names), size=count, replace=False).tolist())
    return {
        names[index]: {
            "translation": float(rng.uniform(*settings["translation"])),
            "yaw": math.radians(float(rng.uniform(*settings["yaw_deg"]))),
        }
        for index in chosen
    }


def separate_on_rails(
    rng: np.random.Generator,
    ports: dict[str, dict[str, float]],
    settings: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Place the SC ports of each rail without overlap, named in rail order.

    SC ports are visually identical, so a port's name must follow from where
    it sits: the ``k``-th present port along a rail takes that rail's ``k``-th
    name (``sc_port_0..2`` on rail 0, ``sc_port_3..4`` on rail 1). Neighbours
    keep at least ``minimum_spacing``, with uniformly distributed slack.

    Args:
        rng: Scene random stream.
        ports: Present SC ports with sampled offsets.
        settings: SC port ``translation`` range and ``minimum_spacing``.

    Returns:
        Present SC ports keyed by their rail-order names.
    """

    lower, upper = settings["translation"]
    spacing = settings["minimum_spacing"]
    ordered: dict[str, dict[str, float]] = {}
    for rail in SC_RAILS:
        present = [ports[name] for name in rail if name in ports]
        if not present:
            continue
        slack = upper - lower - spacing * (len(present) - 1)
        gaps = np.sort(rng.uniform(0.0, slack, size=len(present)))
        for rank, port in enumerate(present):
            ordered[rail[rank]] = {
                "translation": float(lower + gaps[rank] + spacing * rank),
                "yaw": port["yaw"],
            }
    return ordered


def rail_order_names(components: dict[str, dict[str, float]]) -> dict[str, str]:
    """Map each present SC port name to its rail-order name.

    Args:
        components: Present components with ``translation`` offsets.

    Returns:
        Current name mapped to the name of its rank along its rail; the
        identity for scenes that already follow rail order.
    """

    mapping = {}
    for rail in SC_RAILS:
        present = sorted(
            (name for name in rail if name in components),
            key=lambda name: components[name]["translation"],
        )
        mapping |= {name: rail[rank] for rank, name in enumerate(present)}
    return mapping


def sample_scene(config: dict[str, Any], scene_id: int) -> dict[str, Any]:
    """Sample one task-board layout.

    Args:
        config: Validated collection configuration.
        scene_id: Global scene index.

    Returns:
        Board pose in the world frame and the present components.
    """

    randomization = config["randomization"]
    rng = scene_rng(config["collection"]["seed"], scene_id)
    board = {
        "x": float(rng.uniform(*randomization["board_x"])),
        "y": float(rng.uniform(*randomization["board_y"])),
        "z": float(randomization["board_z"]),
        "yaw": float(rng.uniform(*randomization["board_yaw"])),
    }
    components = sample_components(rng, NIC_MOUNTS, randomization["nic_cards"])
    components |= separate_on_rails(
        rng,
        sample_components(rng, SC_PORTS, randomization["sc_ports"]),
        randomization["sc_ports"],
    )
    mounts = randomization["mounts"]
    for name in FIXTURE_MOUNTS:
        if rng.uniform() < mounts["probability"]:
            components[name] = {
                "translation": float(rng.uniform(*mounts["translation"])),
                "yaw": math.radians(float(rng.uniform(*mounts["yaw_deg"]))),
            }
    return {"scene_id": scene_id, "board": board, "components": components}


def xacro_arguments(scene: dict[str, Any]) -> list[str]:
    """Convert a sampled scene into ``task_board.urdf.xacro`` arguments.

    Every slot is listed explicitly so the result never depends on xacro
    defaults. Ground-truth pose publishing is always enabled.

    Args:
        scene: Output of ``sample_scene``.

    Returns:
        ``name:=value`` arguments.
    """

    arguments = ["ground_truth:=true"]
    for name in NIC_MOUNTS + SC_PORTS + FIXTURE_MOUNTS:
        component = scene["components"].get(name)
        arguments.append(f"{name}_present:={'true' if component else 'false'}")
        if component:
            arguments += [
                f"{name}_translation:={component['translation']:.6f}",
                f"{name}_roll:=0.0",
                f"{name}_pitch:=0.0",
                f"{name}_yaw:={component['yaw']:.6f}",
            ]
    return arguments
