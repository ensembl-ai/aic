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

from __future__ import annotations

import logging
import os

import numpy as np

os.environ.setdefault("TRAJOPT_LOG_THRESH", "ERROR")

from tesseract_robotics.tesseract_command_language import (
    CompositeInstruction,
    MoveInstruction,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType_FREESPACE,
    ProfileDictionary,
    StateWaypoint,
    StateWaypointPoly_wrap_StateWaypoint,
)
from tesseract_robotics.tesseract_common import ManipulatorInfo
from tesseract_robotics.tesseract_motion_planners import (
    PlannerRequest,
    PlannerResponse,
)
from tesseract_robotics.tesseract_motion_planners_trajopt import (
    TrajOptDefaultCompositeProfile,
    TrajOptDefaultPlanProfile,
    TrajOptMotionPlanner,
)
from tesseract_robotics.tesseract_time_parameterization import (
    InstructionsTrajectory,
    TimeOptimalTrajectoryGeneration,
)

DEFAULT_PROFILE = "DEFAULT"
TRAJOPT_NAMESPACE = "TrajOptMotionPlannerTask"
logger = logging.getLogger(__name__)


class EnsemblPlanner:
    """
    Build and solve native Tesseract motion-planning programs.
    """

    def __init__(
        self,
        robot,
    ):
        """
        Initialize planner state for a specific robot and manipulator.
        """

        self.robot = robot
        self.env = robot.env
        self.manipulator_group_name = robot.manipulator_group_name
        self.base_frame = robot.manipulator_base_frame
        self.tip_frame = robot.manipulator_tip_frame
        self.manipulator_joint_names = list(robot._manipulator_joint_names)
        self.velocity_limits = robot.velocity_limits
        self.acceleration_limits = robot.acceleration_limits
        self.jerk_limits = robot.jerk_limits
        self.last_failure_reason = None

        self._manipulator_info = ManipulatorInfo()
        self._manipulator_info.manipulator = self.manipulator_group_name
        self._manipulator_info.working_frame = self.base_frame
        self._manipulator_info.tcp_frame = self.tip_frame

        self._motion_planner = TrajOptMotionPlanner(TRAJOPT_NAMESPACE)
        self._profiles = ProfileDictionary()
        self._profiles.addProfile(
            TRAJOPT_NAMESPACE,
            DEFAULT_PROFILE,
            TrajOptDefaultPlanProfile(),
        )
        composite_profile = TrajOptDefaultCompositeProfile()
        composite_profile.smooth_accelerations = False
        composite_profile.smooth_jerks = False
        self._profiles.addProfile(
            TRAJOPT_NAMESPACE,
            DEFAULT_PROFILE,
            composite_profile,
        )

    def _solve_program(
        self,
        program: CompositeInstruction,
    ) -> PlannerResponse | None:
        """
        Solve a Tesseract program and return the native planner response.
        """

        self.last_failure_reason = None
        logger.debug(
            f"Submitting planning request with {len(program.flatten())} "
            "flattened instructions."
        )
        request = PlannerRequest()
        request.instructions = program
        request.env = self.env
        request.profiles = self._profiles
        try:
            response = self._motion_planner.solve(request)
        except Exception as exc:
            self.last_failure_reason = (
                f"Planning request failed before planner response: "
                f"{exc.__class__.__name__}: {exc}"
            )
            logger.exception(
                "Planning request failed before a planner response was produced: "
                f"{exc.__class__.__name__}: {exc}"
            )
            return None

        if not response.successful:
            self.last_failure_reason = f"TrajOpt failed: {response.message}"
            logger.warning(f"{self.last_failure_reason}")
            return None
        logger.debug(f"Planning succeeded: {response.message}")
        return response

    def PlanToTarget(
        self,
        target_transform: np.ndarray | list[list[float]],
        max_joint_delta: float = float("inf"),
    ) -> PlannerResponse | None:
        """
        Plan from the current robot state to a target TCP transform.
        """

        current_joint_values = self.robot.GetActiveDOFValues()
        target_joint_values = self.robot.ComputeIK(target_transform)
        if target_joint_values is None:
            self.last_failure_reason = (
                "PlanToTarget failed: no collision-free IK solution."
            )
            logger.warning(f"{self.last_failure_reason}")
            return None

        max_delta = float(np.max(np.abs(target_joint_values - current_joint_values)))
        if max_delta > max_joint_delta:
            self.last_failure_reason = (
                f"PlanToTarget failed: closest IK solution requires max absolute "
                f"joint delta {max_delta:.3f} rad, which exceeds "
                f"max_joint_delta={max_joint_delta:.3f} rad."
            )
            logger.warning(f"{self.last_failure_reason}")
            return None

        return self.PlanToConfiguration(target_joint_values)

    def PlanToConfiguration(
        self,
        target_joint_values: np.ndarray | list[float],
    ) -> PlannerResponse | None:
        """
        Plan from the current robot state to a target joint configuration.
        """

        target_joint_values = np.asarray(target_joint_values, dtype=np.float64)
        current_joint_values = self.robot.GetActiveDOFValues()
        program = CompositeInstruction(DEFAULT_PROFILE)
        program.setManipulatorInfo(self._manipulator_info)

        for joint_values in (current_joint_values, target_joint_values):
            program.appendMoveInstruction(
                MoveInstructionPoly_wrap_MoveInstruction(
                    MoveInstruction(
                        StateWaypointPoly_wrap_StateWaypoint(
                            StateWaypoint(
                                self.manipulator_joint_names,
                                joint_values,
                            )
                        ),
                        MoveInstructionType_FREESPACE,
                        DEFAULT_PROFILE,
                        self._manipulator_info,
                    )
                )
            )

        return self._solve_program(program)

    def Retime(
        self,
        program: CompositeInstruction,
    ) -> CompositeInstruction | None:
        """
        Apply Tesseract time-parameterization to an existing program.
        """

        logger.debug(f"Retiming trajectory with {len(program.flatten())} instructions.")
        trajectory = InstructionsTrajectory(program)
        time_parameterization = TimeOptimalTrajectoryGeneration()
        ok = time_parameterization.compute(
            trajectory,
            self.velocity_limits,
            self.acceleration_limits,
            self.jerk_limits,
        )
        if not ok:
            logger.warning("Time optimal trajectory generation failed.")
            return None

        logger.debug("Retiming succeeded.")
        return program
