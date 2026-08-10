"""Tests for the privileged Cartesian teacher and trajectory dataset."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from aic_mujoco.config import load_collection_config
from aic_mujoco.controllers import CartesianMoveController
from aic_mujoco.dataset import EpisodeWriter, SyntheticDataset
from aic_mujoco.joints import required_model_id
from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.scene import prepare_scene
from aic_mujoco.utils.warp_math import rotation_error_world


BASE = PACKAGE / "configs" / "base.json"
COLLECT = PACKAGE / "configs" / "collect.json"


@wp.kernel
def evaluate_rotation_error(
    current: wp.array[wp.mat33],
    target: wp.array[wp.mat33],
    error: wp.array[wp.vec3],
):
    """Evaluate the reusable SO(3) error for test rotations."""

    index = wp.tid()
    error[index] = rotation_error_world(current[index], target[index])


def small_collection_config() -> dict:
    config = load_collection_config(BASE, COLLECT)
    config["runtime"]["num_envs"] = 2
    config["physics"].update({"device": "cpu", "graph_capture": False})
    config["sensors"].update({"tare_settle_steps": 1, "tare_sample_count": 1})
    config["cameras"].update(
        {"width": 32, "height": 24, "use_textures": False, "use_shadows": False}
    )
    config["dataset"].update({"image_width": 16, "image_height": 12})
    return config


def test_collection_config_is_strict(tmp_path: Path) -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    config = json.loads(COLLECT.read_text(encoding="utf-8"))

    def leaf_paths(value: dict, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
        paths: set[tuple[str, ...]] = set()
        for key, item in value.items():
            path = (*prefix, key)
            if isinstance(item, dict):
                paths.update(leaf_paths(item, path))
            else:
                paths.add(path)
        return paths

    assert leaf_paths(base).isdisjoint(leaf_paths(config))
    config["expert"]["goal_offset_rotation_matrix"] = [
        2.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    invalid = tmp_path / "invalid_collection.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"must be a valid SO\(3\) matrix"):
        load_collection_config(BASE, invalid)

    config["expert"]["goal_offset_rotation_matrix"] = [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    config["expert"]["control_hz"] = 10
    invalid.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal cameras.fps"):
        load_collection_config(BASE, invalid)


def test_cartesian_teacher_uses_bounded_device_actions() -> None:
    config = small_collection_config()
    prepare_scene(config)
    runtime = AICWarpRuntime(config)
    controlled_body_id = required_model_id(
        runtime.host_model,
        mujoco.mjtObj.mjOBJ_BODY,
        config["expert"]["controlled_body"],
    )
    target_body_id = required_model_id(
        runtime.host_model,
        mujoco.mjtObj.mjOBJ_BODY,
        config["expert"]["target_body"],
    )
    teacher = CartesianMoveController(
        config,
        runtime.robot.joints,
        runtime.host_model,
        controlled_body_id,
        target_body_id,
        runtime.num_envs,
        runtime.device,
    )
    teacher.set_active([True, False])
    teacher.update_goal_from_target_body(runtime.data)
    teacher.move(runtime.data, runtime.hold_command)

    initial_position_error = float(teacher.position_error.numpy()[0])
    initial_orientation_error = float(teacher.orientation_error.numpy()[0])
    actions = teacher.action.position.numpy()
    maximum_step = np.asarray(config["expert"]["maximum_joint_step"])
    assert np.all(np.abs(actions[0]) <= maximum_step + 1.0e-6)
    np.testing.assert_array_equal(actions[1], np.zeros(6))

    while not runtime.step().camera:
        pass
    teacher.update_goal_from_target_body(runtime.data)
    teacher.move(runtime.data, runtime.hold_command)
    assert float(teacher.position_error.numpy()[0]) < initial_position_error
    assert float(teacher.orientation_error.numpy()[0]) < initial_orientation_error

    first = runtime.sample_reset_batch([0], [123])
    second = runtime.sample_reset_batch([1], [123])
    for key in ("q", "board_pos", "board_quat", "nic_pos", "nic_quat"):
        np.testing.assert_array_equal(first[key][0], second[key][1])
    assert first["rail"][0] == second["rail"][1]
    assert first["translation"][0] == second["translation"][1]


def test_rotation_error_is_an_exact_principal_rotation_vector() -> None:
    """Verify small, quarter-turn, and pi matrix errors have radian norms."""

    identity = np.eye(3, dtype=np.float32)
    quarter_turn_z = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    half_turn_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    current = wp.array(np.stack([identity] * 3), dtype=wp.mat33, device="cpu")
    target = wp.array(
        np.stack([identity, quarter_turn_z, half_turn_x]),
        dtype=wp.mat33,
        device="cpu",
    )
    error = wp.zeros(3, dtype=wp.vec3, device="cpu")
    wp.launch(
        evaluate_rotation_error,
        dim=3,
        inputs=[current, target],
        outputs=[error],
        device="cpu",
    )

    values = error.numpy()
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), [0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(values[1], [0.0, 0.0, np.pi / 2], atol=1.0e-6)
    np.testing.assert_allclose(values[2], [np.pi, 0.0, 0.0], atol=1.0e-6)


def test_episode_dataset_is_atomic_resumable_and_valid(tmp_path: Path) -> None:
    config = copy.deepcopy(load_collection_config(BASE, COLLECT))
    config["dataset"]["output_directory"] = str(tmp_path / "dataset")
    config["dataset"]["image_width"] = 16
    config["dataset"]["image_height"] = 12
    config["dataset"]["splits"] = {"train": 1, "validation": 0, "test": 0}
    dataset = SyntheticDataset(config)
    assignment = dataset.next_assignment()
    assert assignment is not None
    writer = EpisodeWriter(
        dataset,
        assignment,
        {
            "joint_hold": np.zeros(6, dtype=np.float32),
            "board_position": np.zeros(3, dtype=np.float32),
        },
    )
    frames = {
        camera_name: np.zeros((12, 16, 3), dtype=np.uint8)
        for camera_name in config["scene"]["names"]["cameras"]
    }
    sample = {
        "qpos": np.zeros(6, dtype=np.float32),
        "qvel": np.zeros(6, dtype=np.float32),
        "action_delta_q": np.zeros(6, dtype=np.float32),
        "sfp_tip_position": np.zeros(3, dtype=np.float32),
        "sfp_tip_rotation_matrix": np.eye(3),
        "goal_position": np.zeros(3, dtype=np.float32),
        "goal_rotation_matrix": np.eye(3),
        "position_error_m": 0.001,
        "orientation_error_rad": 0.001,
        "tared_wrench": np.zeros(6, dtype=np.float32),
    }
    writer.append(frames, sample)
    writer.append(frames, sample)
    writer.finish(True, "pose_tolerance")
    dataset.validate()

    resumed = SyntheticDataset(config)
    assert resumed.counts() == {"train": 1, "validation": 0, "test": 0}
    assert resumed.is_complete()
