"""Random AIC trials: a task board as ``task_board.urdf.xacro`` arguments plus a task.

Each board is one AIC-style trial: an SFP task (SFP plug in hand, target
``nic_card_mount_K/sfp_port_J``) or an SC task (SC plug in hand, target one SC
port), laid out within the engine's limits. Sampling is a pure function of
``(seed, board_id)``, so any worker can regenerate any board and an
interrupted collection resumes deterministically.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

NIC_MOUNTS = tuple(f"nic_card_mount_{index}" for index in range(5))
# aic_engine spawns only sc_port_0 and sc_port_1 (its sc_rail_0 and sc_rail_1),
# and the board places both on the same rail line; sc_port_2..4 are never used.
SC_PORTS = ("sc_port_0", "sc_port_1")
SC_RAILS = (SC_PORTS,)
UNUSED_SC_PORTS = ("sc_port_2", "sc_port_3", "sc_port_4")
VIEWPOINT_STREAM = 2
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
    rng: np.random.Generator,
    names: tuple[str, ...],
    settings: dict[str, Any],
    minimum: int = 0,
) -> dict[str, dict[str, float]]:
    """Choose a random subset of rail slots and their offsets.

    Args:
        rng: Scene random stream.
        names: Candidate slot names.
        settings: ``count``, ``translation``, and ``yaw_deg`` ranges.
        minimum: Fewest slots to fill, on top of the configured lower count.

    Returns:
        Present slots mapped to translation and yaw.
    """

    lower = max(settings["count"][0], minimum)
    count = int(rng.integers(lower, settings["count"][1] + 1))
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
    """Place the SC ports on their rail without overlap, named in rail order.

    SC ports are visually identical, so the model can only tell them apart by
    where they sit: the ``k``-th present port along the rail (by increasing
    translation) is spawned as slot ``sc_port_k``, so its ground-truth frame is
    the schema's ``sc_port_rank_k``. Neighbours keep at least
    ``minimum_spacing``, with uniformly distributed slack.

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


def sample_scene(config: dict[str, Any], board_id: int) -> dict[str, Any]:
    """Sample one trial: a task-board layout and the task performed on it.

    The task mirrors ``aic_engine``'s ``Task``: ``port_type`` is also the type
    of the plug held in the gripper, and ``target_module_name`` plus
    ``port_name`` name the target port as the engine and scoring do. An SFP
    task has at least one NIC card and an SC task at least one SC port.

    Args:
        config: Validated collection configuration.
        board_id: Global board index.

    Returns:
        Board pose in the world frame, the present components, and the task.
    """

    randomization = config["randomization"]
    rng = scene_rng(config["collection"]["seed"], board_id)
    port_type = "sc" if rng.uniform() < randomization["sc_task_probability"] else "sfp"
    board = {
        "x": float(rng.uniform(*randomization["board_x"])),
        "y": float(rng.uniform(*randomization["board_y"])),
        "z": float(randomization["board_z"]),
        "yaw": float(rng.uniform(*randomization["board_yaw"])),
    }
    nic_cards = sample_components(
        rng, NIC_MOUNTS, randomization["nic_cards"], int(port_type == "sfp")
    )
    sc_ports = separate_on_rails(
        rng,
        sample_components(
            rng, SC_PORTS, randomization["sc_ports"], int(port_type == "sc")
        ),
        randomization["sc_ports"],
    )
    if port_type == "sfp":
        modules = sorted(nic_cards)
        task = {
            "port_type": "sfp",
            "target_module_name": modules[int(rng.integers(len(modules)))],
            "port_name": f"sfp_port_{int(rng.integers(2))}",
        }
    else:
        modules = sorted(sc_ports)
        task = {
            "port_type": "sc",
            "target_module_name": modules[int(rng.integers(len(modules)))],
            "port_name": "sc_port_base",
        }
    components = nic_cards | sc_ports
    mounts = randomization["mounts"]
    for name in FIXTURE_MOUNTS:
        if rng.uniform() < mounts["probability"]:
            components[name] = {
                "translation": float(rng.uniform(*mounts["translation"])),
                "yaw": math.radians(float(rng.uniform(*mounts["yaw_deg"]))),
            }
    return {
        "board_id": board_id,
        "board": board,
        "components": components,
        "task": task,
    }


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
    for name in NIC_MOUNTS + SC_PORTS + UNUSED_SC_PORTS + FIXTURE_MOUNTS:
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


def sample_approach(config: dict[str, Any], board_id: int) -> list[dict[str, Any]]:
    """Sample the viewpoints captured for one board, ordered along an approach.

    With ``home_view``, view 0 is HOME, where every trial starts. The
    remaining views descend toward the board's task target port: their
    heights are stratified log-uniformly over ``height_m`` so each board
    covers far and near views, each with a lateral offset that grows with
    height and a small yaw error, as the insertion policy would be aligned.
    The offset radius is ``reach * u ** lateral_power`` for uniform ``u``:
    0.5 spreads views uniformly over the disk, 1.0 concentrates them near the
    port axis. A fraction ``on_axis_fraction`` of the views is placed exactly
    on the port axis (zero planned offset).

    Args:
        config: Validated collection configuration.
        board_id: Global board index.

    Returns:
        ``collection.views_per_board`` viewpoint descriptions.
    """

    settings = config["viewpoints"]
    rng = np.random.default_rng(
        [config["collection"]["seed"], board_id, VIEWPOINT_STREAM]
    )
    low, high = settings["height_m"]
    stops = config["collection"]["views_per_board"] - settings["home_view"]
    views = [{"mode": "home"}] if settings["home_view"] else []
    for stratum in reversed(range(stops)):
        height = float(low * (high / low) ** ((stratum + rng.uniform()) / stops))
        reach = settings["lateral_m"] + settings["lateral_per_height"] * height
        radius = reach * rng.uniform() ** settings["lateral_power"]
        if rng.uniform() < settings["on_axis_fraction"]:
            radius = 0.0
        direction = rng.uniform(0.0, 2.0 * math.pi)
        views.append(
            {
                "mode": "close",
                "height": height,
                "offset": [radius * math.cos(direction), radius * math.sin(direction)],
                "yaw": math.radians(
                    float(rng.uniform(-settings["yaw_deg"], settings["yaw_deg"]))
                ),
            }
        )
    return views
