"""Contract tests for the minimal AIC MuJoCo-Warp foundation."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from aic_mujoco.config import deep_merge, load_config
from aic_mujoco.commands import HoldPositionCommand
from aic_mujoco.controllers import JointHoldController
from aic_mujoco.joints import ArmJoints
from aic_mujoco.robot import AICRobot
from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.scene import prepare_scene

BASE = PACKAGE / "configs" / "base.json"
RUN = PACKAGE / "configs" / "run.json"


@pytest.fixture(scope="session")
def configured_scene() -> dict:
    config = load_config(BASE, RUN)
    prepare_scene(config)
    return config


def test_deep_merge_and_strict_validation(tmp_path: Path) -> None:
    base = {"nested": {"left": 1, "right": [1, 2]}, "value": 3}
    overlay = {"nested": {"right": [4]}}
    assert deep_merge(base, overlay) == {
        "nested": {"left": 1, "right": [4]},
        "value": 3,
    }
    assert base["nested"]["right"] == [1, 2]

    def leaf_paths(value: dict, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
        paths: set[tuple[str, ...]] = set()
        for key, item in value.items():
            path = (*prefix, key)
            if isinstance(item, dict):
                paths.update(leaf_paths(item, path))
            else:
                paths.add(path)
        return paths

    base_config = json.loads(BASE.read_text(encoding="utf-8"))
    run_config = json.loads(RUN.read_text(encoding="utf-8"))
    assert leaf_paths(base_config).isdisjoint(leaf_paths(run_config))
    assert run_config["visualization"]["env_ids"] == "all"

    invalid = tmp_path / "invalid.json"
    run_config["physics"]["unrecognized"] = 1
    invalid.write_text(json.dumps(run_config), encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown configuration keys"):
        load_config(BASE, invalid)

    del run_config["physics"]["unrecognized"]
    run_config["visualization"]["env_ids"] = "everything"
    invalid.write_text(json.dumps(run_config), encoding="utf-8")
    with pytest.raises(ValueError, match="must be 'all'"):
        load_config(BASE, invalid)


def test_reduced_scene_contract(configured_scene: dict) -> None:
    path = Path(configured_scene["scene"]["output"])
    text = path.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nv, model.nu) == (6, 6, 6)
    assert (model.ncam, model.nsensor, model.nsensordata) == (3, 2, 6)
    assert model.nmocap == 2
    assert model.nplugin == 0
    assert "cable_end" not in text
    assert "lc_plug_link" not in text
    assert "elasticity.cable" not in text
    assert "gripper/left_finger_joint\"" not in text
    assert "gripper/right_finger_joint\"" not in text
    names = configured_scene["scene"]["names"]
    tool_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names["tool_body"])
    sfp_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, names["sfp_source_body"]
    )
    assert model.body_parentid[sfp_id] == tool_id
    assert not np.any(model.jnt_bodyid == sfp_id)
    fixed_position = configured_scene["scene"]["gripper_fixed_position"]
    for finger_name in (names["left_finger_body"], names["right_finger_body"]):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger_name)
        assert np.linalg.norm(model.body_pos[body_id]) == pytest.approx(fixed_position)
        finger_geoms = model.geom_bodyid == body_id
        assert np.all(model.geom_contype[finger_geoms] == 0)
        assert np.all(model.geom_conaffinity[finger_geoms] == 0)


def test_independent_warp_worlds_and_observations(configured_scene: dict) -> None:
    config = copy.deepcopy(configured_scene)
    config["physics"].update({"device": "cpu", "graph_capture": False})
    config["sensors"].update({"tare_settle_steps": 5, "tare_sample_count": 2})
    config["cameras"].update(
        {"width": 32, "height": 24, "use_textures": False, "use_shadows": False}
    )
    config["visualization"]["enabled"] = False
    config["recording"]["enabled"] = False

    runtime = AICWarpRuntime(config)
    assert isinstance(runtime.robot, AICRobot)
    assert isinstance(runtime.robot.joints, ArmJoints)
    assert isinstance(runtime.hold_command, HoldPositionCommand)
    assert isinstance(runtime.controller, JointHoldController)
    assert runtime.robot.joints.names == tuple(config["scene"]["names"]["joints"])
    assert runtime.robot.joints.actuator_addresses.shape == (6,)
    observations = runtime.observations()
    assert set(observations["rgb"]) == {"center", "left", "right"}
    for image in observations["rgb"].values():
        pixels = image.numpy()
        assert pixels.shape == (2, 24, 32, 3)
        assert pixels.dtype == np.uint8
        assert pixels.max() > pixels.min()
    assert observations["wrench"]["raw"].shape == (2, 6)
    assert observations["wrench"]["tared"].shape == (2, 6)
    assert np.all(observations["wrench"]["tare_ready"].numpy())

    hold = runtime.hold_command.position.numpy()
    assert not np.allclose(hold[0], hold[1])
    lower = np.asarray(config["control"]["home"]) + np.asarray(
        config["control"]["reset_perturbation_lower"]
    )
    upper = np.asarray(config["control"]["home"]) + np.asarray(
        config["control"]["reset_perturbation_upper"]
    )
    assert np.all(hold >= lower) and np.all(hold <= upper)
    board = runtime.board_position.numpy()
    assert np.all(
        board >= np.asarray(config["domain_randomization"]["board_position_lower"]) - 1e-6
    )
    assert np.all(
        board <= np.asarray(config["domain_randomization"]["board_position_upper"]) + 1e-6
    )
    assert set(runtime.nic_rail_index.numpy()).issubset(
        config["domain_randomization"]["nic_rail_indices"]
    )

    for _ in range(25):
        runtime.step()
    np.testing.assert_allclose(runtime.data.qpos.numpy(), hold, atol=1e-5)
    np.testing.assert_array_equal(runtime.episode_steps.numpy(), [25, 25])
    assert np.all(np.isfinite(runtime.raw_wrench.numpy()))
    assert np.all(np.isfinite(runtime.tared_wrench.numpy()))

    first_before = runtime.hold_command.position.numpy()[0].copy()
    second_before = runtime.hold_command.position.numpy()[1].copy()
    runtime.reset([1])
    assert np.array_equal(runtime.tare_ready.numpy(), [True, False])
    np.testing.assert_array_equal(runtime.episode_steps.numpy(), [25, 0])
    current_command = runtime.hold_command.position.numpy()
    np.testing.assert_array_equal(current_command[0], first_before)
    assert not np.array_equal(current_command[1], second_before)
    for _ in range(15):
        runtime.step()
    assert np.all(runtime.tare_ready.numpy())
    np.testing.assert_array_equal(runtime.episode_steps.numpy(), [40, 0])
    runtime.step()
    np.testing.assert_array_equal(runtime.episode_steps.numpy(), [41, 1])
