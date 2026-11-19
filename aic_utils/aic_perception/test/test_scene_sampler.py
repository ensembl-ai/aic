"""Scene randomization."""

from __future__ import annotations

import copy
import math

import pytest

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
    """Engine-style scenes with shuffled or gapped SC slots map onto rail-order names."""

    swapped = {
        "sc_port_0": {"translation": 0.0, "yaw": 0.0},
        "sc_port_1": {"translation": -0.04, "yaw": 0.0},
        "nic_card_mount_1": {"translation": 0.0, "yaw": 0.0},
    }
    assert rail_order_names(swapped) == {
        "sc_port_1": "sc_port_0",
        "sc_port_0": "sc_port_1",
    }
    assert rail_order_names({"sc_port_1": {"translation": -0.055, "yaw": 0.0}}) == {
        "sc_port_1": "sc_port_0"
    }


def test_tasks_match_aic_trials() -> None:
    """Each board is an SFP or SC task whose target port is spawned and in the schema."""

    from aic_perception.schema import BoardSchema

    frames = {
        target.frame: target for target in BoardSchema.from_config(CONFIG).targets
    }
    scenes = [sample_scene(CONFIG, board_id) for board_id in range(2000)]
    sc_tasks = 0
    for scene in scenes:
        task, components = scene["task"], scene["components"]
        assert task["target_module_name"] in components
        frame = (
            f"task_board/{task['target_module_name']}/{task['port_name']}_link_entrance"
        )
        target = frames[frame]
        if task["port_type"] == "sfp":
            assert task["target_module_name"] in NIC_MOUNTS
            assert task["port_name"] in ("sfp_port_0", "sfp_port_1")
            assert target.type == "sfp_port"
        else:
            sc_tasks += 1
            assert task["target_module_name"] in SC_PORTS
            assert task["port_name"] == "sc_port_base"
            assert target.type == "sc_port"
        assert not set(components) & {"sc_port_2", "sc_port_3", "sc_port_4"}
    probability = CONFIG["randomization"]["sc_task_probability"]
    assert abs(sc_tasks / len(scenes) - probability) < 0.03


def test_xacro_arguments_list_every_slot() -> None:
    """Every slot is explicit, so xacro defaults never leak in."""

    arguments = xacro_arguments(sample_scene(CONFIG, 3))
    for name in (
        NIC_MOUNTS + SC_PORTS + ("sc_port_2", "sc_port_3", "sc_port_4") + FIXTURE_MOUNTS
    ):
        assert (
            sum(argument.startswith(f"{name}_present:=") for argument in arguments) == 1
        )
    assert "ground_truth:=true" in arguments


@pytest.mark.parametrize("home_view", [True, False])
def test_approach_descends_toward_one_port(home_view: bool) -> None:
    """Each board's views repeat, optionally start at HOME, and descend through the height range."""

    from aic_perception.scene_sampler import sample_approach

    config = copy.deepcopy(CONFIG)
    config["viewpoints"]["home_view"] = home_view
    settings = config["viewpoints"]
    low, high = settings["height_m"]
    approaches = [sample_approach(config, board_id) for board_id in range(500)]
    assert approaches == [sample_approach(config, board_id) for board_id in range(500)]
    heights = []
    for views in approaches:
        assert len(views) == config["collection"]["views_per_board"]
        assert (views[0] == {"mode": "home"}) == home_view
        close = views[1:] if home_view else views
        assert all(view["mode"] == "close" for view in close)
        assert [view["height"] for view in close] == sorted(
            (view["height"] for view in close), reverse=True
        )
        for view in close:
            reach = (
                settings["lateral_m"] + settings["lateral_per_height"] * view["height"]
            )
            assert math.hypot(*view["offset"]) <= reach + 1e-12
        heights += [view["height"] for view in close]
    assert low <= min(heights) and max(heights) <= high
    assert min(heights) < 1.5 * low and max(heights) > 0.8 * high


def test_lateral_power_concentrates_views_near_the_port_axis() -> None:
    """A higher lateral power puts more views close to the port axis."""

    from aic_perception.scene_sampler import sample_approach

    def fraction_near(power: float) -> float:
        config = copy.deepcopy(CONFIG)
        config["viewpoints"]["lateral_power"] = power
        config["viewpoints"]["on_axis_fraction"] = 0.0
        offsets = [
            math.hypot(*view["offset"])
            / (
                config["viewpoints"]["lateral_m"]
                + config["viewpoints"]["lateral_per_height"] * view["height"]
            )
            for board_id in range(300)
            for view in sample_approach(config, board_id)
            if view["mode"] == "close"
        ]
        return sum(offset < 0.25 for offset in offsets) / len(offsets)

    assert abs(fraction_near(0.5) - 0.0625) < 0.02
    assert abs(fraction_near(1.0) - 0.25) < 0.03


def test_on_axis_views_have_zero_planned_offset() -> None:
    """The configured fraction of views sits exactly on the port axis."""

    from aic_perception.scene_sampler import sample_approach

    views = [
        view
        for board_id in range(500)
        for view in sample_approach(CONFIG, board_id)
        if view["mode"] == "close"
    ]
    on_axis = sum(view["offset"] == [0.0, 0.0] for view in views) / len(views)
    assert abs(on_axis - CONFIG["viewpoints"]["on_axis_fraction"]) < 0.03
