"""Offline tool calibration: measure where each plug sits in the gripper.

This is a calibration utility, not a competition policy. It reads simulator
ground truth (``ground_truth:=true``) to measure the plug pose in the
``gripper/tcp`` frame for every task the engine runs, the way a real cell
calibrates a tool once before operation. It does not move the robot.

Results are averaged per plug type and written to
``runs/calibration/plug_grasp.json``; copy them into ``configs/infer.json``
under ``plug_grasp``. It also reports the rotation between each port's
entrance frame and its body frame, which the insertion policy assumes is the
identity.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException

from aic_perception.config import PACKAGE_DIR
from aic_perception.utils.geometry import from_xyz_quaternion_xyzw, quaternion_xyzw

OUTPUT = PACKAGE_DIR / "runs" / "calibration" / "plug_grasp.json"
SAMPLES = 20


class CalibratePlugGrasp(Policy):
    """Measure the plug-in-TCP transform for each task's plug type."""

    def lookup(self, target: str, source: str) -> np.ndarray | None:
        """Look up ``target_from_source`` from TF, or ``None`` if missing."""

        try:
            message = self._parent_node._tf_buffer.lookup_transform(
                target, source, Time()
            )
        except TransformException:
            return None
        t, r = message.transform.translation, message.transform.rotation
        return from_xyz_quaternion_xyzw((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Average the plug pose in the TCP frame and record it.

        Args:
            task: AIC task naming the cable and plug.
            get_observation: Unused.
            move_robot: Unused; the robot stays still.
            send_feedback: Unused.

        Returns:
            Always ``False``: no insertion is attempted.
        """

        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        samples = []
        for _ in range(200):
            transform = self.lookup("gripper/tcp", plug_frame)
            if transform is not None:
                samples.append(transform)
                if len(samples) == SAMPLES:
                    break
            self.sleep_for(0.05)
        if len(samples) < SAMPLES:
            self.get_logger().error(
                f"No ground truth for {plug_frame}; run with ground_truth:=true"
            )
            return False
        translation = np.mean([s[:3, 3] for s in samples], axis=0)
        spread_mm = 1000.0 * float(np.max(np.std([s[:3, 3] for s in samples], axis=0)))
        grasp = {
            "translation": [round(float(v), 6) for v in translation],
            "quaternion_xyzw": [round(v, 6) for v in quaternion_xyzw(samples[-1])],
        }
        self.get_logger().info(
            f"{task.plug_type} grasp: {grasp} (position spread {spread_mm:.3f} mm)"
        )

        body = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance = self.lookup(body, f"{body}_entrance")
        if entrance is not None:
            angle = math.degrees(
                math.acos(np.clip((np.trace(entrance[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
            )
            self.get_logger().info(
                f"{task.port_type} entrance relative to body: offset {np.round(entrance[:3, 3], 4).tolist()} m, "
                f"rotation {angle:.2f} deg"
            )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = (
            json.loads(OUTPUT.read_text()) if OUTPUT.is_file() else {}
        )
        results[task.plug_type] = grasp
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        self.get_logger().info(f"Wrote {OUTPUT}")
        return False
