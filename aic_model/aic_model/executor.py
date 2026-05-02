#
#  Copyright (C) 2026 Ensemble Robotics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from collections.abc import Callable
import logging

import numpy as np

from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from rclpy.duration import Duration
from tesseract_robotics.tesseract_command_language import (
    CompositeInstruction,
    InstructionPoly_as_MoveInstructionPoly,
    WaypointPoly_as_StateWaypointPoly,
)

logger = logging.getLogger(__name__)


class EnsemblExecutor:
    """
    Convert a retimed Tesseract trajectory into controller joint commands.
    """

    def __init__(
        self,
        execute_joint_motion: Callable[[JointMotionUpdate], None] | None = None,
    ):
        """
        Initialize the executor with controller and timing callbacks.
        """

        self._execute_joint_motion = execute_joint_motion

    def ExecuteTrajectory(
        self,
        trajectory: CompositeInstruction,
        stiffness: list[float] | None = None,
        damping: list[float] | None = None,
    ) -> bool:
        """
        Stream a retimed state-waypoint trajectory to the joint controller.
        """

        if self._execute_joint_motion is None:
            logger.warning(
                "ExecuteTrajectory requires an execute_joint_motion callback."
            )
            return False

        flattened_instructions = trajectory.flatten()
        logger.debug(
            f"Executing retimed trajectory with "
            f"{len(flattened_instructions)} flattened instructions."
        )

        joint_motion_update = JointMotionUpdate(
            target_stiffness=(stiffness or [150.0, 150.0, 150.0, 80.0, 80.0, 80.0]),
            target_damping=(damping or [90.0, 90.0, 90.0, 45.0, 45.0, 45.0]),
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )

        waypoint_index = 0
        for instruction in flattened_instructions:
            move = InstructionPoly_as_MoveInstructionPoly(instruction)
            waypoint = move.getWaypoint()
            if not waypoint.isStateWaypoint():
                continue

            state_waypoint = WaypointPoly_as_StateWaypointPoly(waypoint)
            waypoint_time = float(state_waypoint.getTime())
            if waypoint_index == 0:
                waypoint_index += 1
                continue

            positions = np.asarray(
                state_waypoint.getPosition(),
                dtype=np.float64,
            ).reshape(-1)
            velocities = np.asarray(
                state_waypoint.getVelocity(),
                dtype=np.float64,
            ).reshape(-1)
            accelerations = np.asarray(
                state_waypoint.getAcceleration(),
                dtype=np.float64,
            ).reshape(-1)

            joint_motion_update.target_state.positions = positions.tolist()
            joint_motion_update.target_state.velocities = (
                velocities.tolist() if velocities.size == positions.size else []
            )
            joint_motion_update.target_state.accelerations = (
                accelerations.tolist()
                if accelerations.size == positions.size
                else []
            )
            joint_motion_update.target_state.time_from_start = Duration(
                seconds=waypoint_time,
            ).to_msg()

            self._execute_joint_motion(joint_motion_update)

            waypoint_index += 1

        if waypoint_index < 2:
            logger.warning("Trajectory did not contain at least two state waypoints.")
            return False

        logger.debug(
            f"Trajectory execution stream completed with {waypoint_index} waypoints."
        )
        return True
