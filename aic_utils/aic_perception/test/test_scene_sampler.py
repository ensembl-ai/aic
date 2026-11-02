"""Scene randomization."""

from __future__ import annotations

from aic_perception.config import load_config
from aic_perception.scene_sampler import (
    FIXTURE_MOUNTS,
    NIC_MOUNTS,
    SC_PORTS,
    SC_RAILS,
    rail_order_names,
    sample_scene,
    xacro_arguments,
)

CONFIG = load_config("collect")


def test_sampling_is_deterministic_per_scene() -> None:
    """Any worker regenerates the same scene from its ID."""

    assert sample_scene(CONFIG, 17) == sample_scene(CONFIG, 17)
    assert sample_scene(CONFIG, 17) != sample_scene(CONFIG, 18)


def test_samples_respect_configured_ranges() -> None:
    """Board pose, component counts, and offsets stay within limits."""

    randomization = CONFIG["randomization"]
    for scene_id in range(300):
        scene = sample_scene(CONFIG, scene_id)
        board = scene["board"]
        assert randomization["board_x"][0] <= board["x"] <= randomization["board_x"][1]
        assert randomization["board_y"][0] <= board["y"] <= randomization["board_y"][1]
        components = scene["components"]
        nic_count = sum(name in components for name in NIC_MOUNTS)
        sc_count = sum(name in components for name in SC_PORTS)
        assert (
            randomization["nic_cards"]["count"][0]
            <= nic_count
            <= randomization["nic_cards"]["count"][1]
        )
        assert (
            randomization["sc_ports"]["count"][0]
            <= sc_count
            <= randomization["sc_ports"]["count"][1]
        )
        for name in SC_PORTS:
            if name in components:
                low, high = randomization["sc_ports"]["translation"]
                assert low - 1e-9 <= components[name]["translation"] <= high + 1e-9


def test_sc_ports_on_one_rail_never_overlap() -> None:
    """Ports that share a rail keep the configured spacing."""

    spacing = CONFIG["randomization"]["sc_ports"]["minimum_spacing"]
    for scene_id in range(500):
        components = sample_scene(CONFIG, scene_id)["components"]
        for rail in SC_RAILS:
            offsets = sorted(
                components[name]["translation"] for name in rail if name in components
            )
            assert all(b - a >= spacing - 1e-9 for a, b in zip(offsets, offsets[1:]))


def test_sc_port_names_follow_rail_order() -> None:
    """Identical SC ports are named by rank along their rail, from the rail's first name."""

    for scene_id in range(500):
        components = sample_scene(CONFIG, scene_id)["components"]
        for rail in SC_RAILS:
            present = [name for name in rail if name in components]
            assert present == list(rail[: len(present)])
            offsets = [components[name]["translation"] for name in present]
            assert offsets == sorted(offsets)
        assert all(old == new for old, new in rail_order_names(components).items())


def test_rail_order_names_renames_out_of_order_ports() -> None:
    """Legacy scenes with shuffled or gapped SC names map onto rail-order names."""

    components = {
        "sc_port_2": {"translation": -0.05, "yaw": 0.0},
        "sc_port_0": {"translation": 0.03, "yaw": 0.0},
        "sc_port_4": {"translation": 0.01, "yaw": 0.0},
        "nic_card_mount_1": {"translation": 0.0, "yaw": 0.0},
    }
    assert rail_order_names(components) == {
        "sc_port_2": "sc_port_0",
        "sc_port_0": "sc_port_1",
        "sc_port_4": "sc_port_3",
    }


def test_xacro_arguments_list_every_slot() -> None:
    """Every slot is explicit, so xacro defaults never leak in."""

    arguments = xacro_arguments(sample_scene(CONFIG, 3))
    for name in NIC_MOUNTS + SC_PORTS + FIXTURE_MOUNTS:
        assert (
            sum(argument.startswith(f"{name}_present:=") for argument in arguments) == 1
        )
    assert "ground_truth:=true" in arguments
