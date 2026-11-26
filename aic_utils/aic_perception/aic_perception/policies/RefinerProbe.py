"""Evaluation policy: measure the close-range refiner live against simulator ground truth (no insertion).

Approaches exactly like ``PerceptionInsert`` up to the pre-contact pose, then visits a grid of deliberately
perturbed plug poses (heights, sideways offsets, yaw) and at each one records refiner readings next to the true
plug-to-port offset and rotation. Ground truth is used only to score the readings; nothing is controlled from it.
Readings are stored as ``probe`` phases in the trial record. The trial never inserts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.euler import euler2mat

from aic_perception.config import load_config
from aic_perception.policies.insertion import InsertionFailed, PortInsertion

HEIGHTS_M = (0.010, 0.020, 0.040)
OFFSETS_M = ((0.0, 0.0), (0.002, 0.0), (0.0, -0.002), (-0.005, 0.0), (0.0, 0.005))
YAWS_DEG = (0.0, 3.0)
SAMPLES = 3


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    angle = float(np.linalg.norm(vector))
    return np.eye(3) if angle < 1e-12 else axangle2mat(vector / angle, angle)


def angle_deg(rotation: np.ndarray) -> float:
    """Rotation angle of a rotation matrix in degrees."""
    return float(np.degrees(abs(mat2axangle(rotation)[1])))


class RefinerProbe(PortInsertion):
    """Approach as usual, then probe the refiner at perturbed poses and log it against ground truth."""

    def __init__(self, parent_node):
        """Load the refiner even when the insertion policy has it disabled."""
        super().__init__(parent_node)
        if self.refiner is None:
            from aic_perception.refinement import RefinerRuntime

            config = load_config("infer")
            self.refiner = RefinerRuntime(
                Path(config["refiner"]["checkpoint"]), config["refiner"]["device"]
            )

    def truth(self, task) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """True port entrance and plug tip poses in base_link (evaluation only)."""
        port = self.lookup(
            "base_link",
            f"task_board/{task.target_module_name}/{task.port_name}_link_entrance",
        )
        plug = self.lookup("base_link", f"{task.cable_name}/{task.plug_name}_link")
        return port, plug

    def run_phases(self, task, get_observation, move_robot) -> bool:
        """Reach pre-contact like the insertion policy, then probe."""
        grasp = self.grasps[task.plug_type]
        from aic_perception.utils.geometry import from_xyz_quaternion_xyzw

        self.tcp_from_plug = from_xyz_quaternion_xyzw(
            tuple(grasp["translation"]), tuple(grasp["quaternion_xyzw"])
        )
        settings = self.settings
        target_id = self.resolve_target(task, move_robot)
        self.target_id = target_id
        port = self.find_port(move_robot, task, target_id)
        self.port_xy = port[:2, 3].copy()
        coarse = port.copy()
        coarse[2, 3] += settings["coarse_height_m"]
        self.move_plug(
            move_robot, coarse, settings["approach_duration_s"], mode="descent"
        )
        pre_contact = port.copy()
        pre_contact[2, 3] += settings["pre_contact_height_m"]
        port, pre_contact = self.refine(move_robot, task, port, pre_contact)
        for height in HEIGHTS_M:
            for dx, dy in OFFSETS_M:
                for yaw in YAWS_DEG:
                    goal = pre_contact.copy()
                    goal[:3, 3] += (dx, dy, height - settings["pre_contact_height_m"])
                    goal[:3, :3] = (
                        euler2mat(0.0, 0.0, np.radians(yaw)) @ pre_contact[:3, :3]
                    )
                    self.move_plug(move_robot, goal, 1.0, mode="descent")
                    self.wait_until_still(2.0, 0.2)
                    for _ in range(SAMPLES):
                        try:
                            reading = self.measure(get_observation, task.plug_type)
                        except InsertionFailed:
                            continue
                        true_port, true_plug = self.truth(task)
                        if true_port is None or true_plug is None:
                            continue
                        true_offset = true_port[:3, 3] - true_plug[:3, 3]
                        true_rel = true_port[:3, :3] @ true_plug[:3, :3].T
                        est_rel = rotation_from_vector(np.asarray(reading["rotation"]))
                        self.note(
                            "probe",
                            height_mm=1000 * height,
                            dx_mm=1000 * dx,
                            dy_mm=1000 * dy,
                            yaw_deg=yaw,
                            true_offset_mm=1000 * true_offset,
                            est_offset_mm=1000 * np.asarray(reading["offset"]),
                            offset_error_mm=1000
                            * float(
                                np.linalg.norm(
                                    np.asarray(reading["offset"]) - true_offset
                                )
                            ),
                            lateral_error_mm=1000
                            * float(
                                np.linalg.norm(
                                    np.asarray(reading["offset"])[:2] - true_offset[:2]
                                )
                            ),
                            true_rotation_deg=angle_deg(true_rel),
                            est_rotation_deg=angle_deg(est_rel),
                            rotation_error_deg=angle_deg(est_rel.T @ true_rel),
                            plug_error_mm=1000
                            * float(
                                np.linalg.norm(
                                    np.asarray(reading["plug"]) - true_plug[:3, 3]
                                )
                            ),
                        )
        raise InsertionFailed("Refiner probe finished (no insertion by design)")
