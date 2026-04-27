#!/usr/bin/env python3

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_model.robot import EnsemblRobot


MANIPULATOR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_JOINT_NAMES = [
    "gripper/right_finger_joint",
    "gripper/left_finger_joint",
]


def make_observation(joint_names, joint_positions):
    return SimpleNamespace(
        joint_states=SimpleNamespace(
            name=joint_names,
            position=joint_positions,
        )
    )


def test_observation_sync_uses_only_manipulator_joints():
    observation = make_observation(
        MANIPULATOR_JOINT_NAMES + GRIPPER_JOINT_NAMES,
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.02, 0.02],
    )
    robot = EnsemblRobot(lambda: observation)
    original_joint_values = np.asarray(
        robot.env.getCurrentJointValues(),
        dtype=np.float64,
    ).reshape(-1)

    np.testing.assert_allclose(
        robot.GetActiveDOFValues(),
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
    )
    synced_joint_values = np.asarray(
        robot.env.getCurrentJointValues(),
        dtype=np.float64,
    ).reshape(-1)
    gripper_indices = [
        index
        for index, joint_name in enumerate(robot._active_joint_names)
        if joint_name.startswith("gripper/")
    ]
    np.testing.assert_allclose(
        synced_joint_values[gripper_indices],
        original_joint_values[gripper_indices],
    )


def test_observation_sync_requires_manipulator_joint_names():
    observation = make_observation(
        MANIPULATOR_JOINT_NAMES[:-1] + ["gripper/left_finger_joint"],
        [0.1, -0.2, 0.3, -0.4, 0.5, 0.02],
    )
    robot = EnsemblRobot(lambda: observation)

    try:
        robot.GetActiveDOFValues()
    except RuntimeError as exc:
        assert "wrist_3_joint" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing wrist_3_joint.")


def main():
    test_observation_sync_uses_only_manipulator_joints()
    test_observation_sync_requires_manipulator_joint_names()
    print("Observation sync validation passed.")


if __name__ == "__main__":
    main()
