from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("TRAJOPT_LOG_THRESH", "ERROR")

from tesseract_robotics.tesseract_command_language import (
    CartesianWaypoint,
    CartesianWaypointPoly_wrap_CartesianWaypoint,
    CompositeInstruction,
    MoveInstruction,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType_FREESPACE,
    ProfileDictionary,
    StateWaypoint,
    StateWaypointPoly_wrap_StateWaypoint,
)
from tesseract_robotics.tesseract_common import Isometry3d, ManipulatorInfo
from tesseract_robotics.tesseract_motion_planners import PlannerRequest, PlannerResponse
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


class EnsemblPlanner:
    def __init__(
        self,
        robot,
        num_waypoints: int = 5,
    ):
        self.robot = robot
        self.env = robot.env
        self.manipulator_group_name = robot.manipulator_group_name
        self.base_frame = robot.manipulator_base_frame
        self.tip_frame = robot.manipulator_tip_frame
        self.manipulator_joint_names = list(robot._manipulator_joint_names)
        kinematic_limits = robot._joint_group.getLimits()
        self.velocity_limits = np.asarray(
            kinematic_limits.velocity_limits,
            dtype=np.float64,
        )
        self.acceleration_limits = np.asarray(
            kinematic_limits.acceleration_limits,
            dtype=np.float64,
        )
        self.jerk_limits = np.asarray(
            kinematic_limits.jerk_limits,
            dtype=np.float64,
        )
        self.num_waypoints = int(num_waypoints)

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
        self._profiles.addProfile(
            TRAJOPT_NAMESPACE,
            DEFAULT_PROFILE,
            TrajOptDefaultCompositeProfile(),
        )

    def _solve_program(self, program: CompositeInstruction) -> PlannerResponse:
        request = PlannerRequest()
        request.instructions = program
        request.env = self.env
        request.profiles = self._profiles
        response = self._motion_planner.solve(request)
        if not response.successful:
            raise RuntimeError(f"TrajOpt failed: {response.message}")
        return response

    def PlanToTarget(
        self,
        target_transform: np.ndarray | list[list[float]],
    ) -> PlannerResponse:
        return self._solve_program(self._make_target_program(target_transform))

    def Retime(
        self,
        program: CompositeInstruction,
    ) -> CompositeInstruction:
        trajectory = InstructionsTrajectory(program)
        time_parameterization = TimeOptimalTrajectoryGeneration()
        ok = time_parameterization.compute(
            trajectory,
            self.velocity_limits,
            self.acceleration_limits,
            self.jerk_limits,
        )
        if not ok:
            raise RuntimeError("Time optimal trajectory generation failed.")

        return program

    def _make_target_program(
        self,
        target_transform: np.ndarray | list[list[float]],
    ) -> CompositeInstruction:
        current_joint_values = np.asarray(
            self.robot.GetActiveDOFValues(),
            dtype=np.float64,
        ).reshape(-1)
        if current_joint_values.size != len(self.manipulator_joint_names):
            raise ValueError(
                "Expected current joint values with shape "
                f"({len(self.manipulator_joint_names)},), got "
                f"{current_joint_values.shape}."
            )
        current_transform = np.asarray(self.robot.ComputeFK(), dtype=np.float64)
        if current_transform.shape != (4, 4):
            raise ValueError(
                "Expected robot.ComputeFK() to return shape "
                f"(4, 4), got {current_transform.shape}."
            )
        target_transform = np.asarray(target_transform, dtype=np.float64)
        if target_transform.shape != (4, 4):
            raise ValueError(
                f"Expected target transform with shape (4, 4), got {target_transform.shape}."
            )

        program = CompositeInstruction(DEFAULT_PROFILE)
        program.setManipulatorInfo(self._manipulator_info)
        program.appendMoveInstruction(
            MoveInstructionPoly_wrap_MoveInstruction(
                MoveInstruction(
                    StateWaypointPoly_wrap_StateWaypoint(
                        StateWaypoint(
                            self.manipulator_joint_names,
                            current_joint_values,
                        )
                    ),
                    MoveInstructionType_FREESPACE,
                    DEFAULT_PROFILE,
                    self._manipulator_info,
                )
            )
        )

        current_position = current_transform[:3, 3]
        alphas = np.linspace(0.0, 1.0, num=self.num_waypoints)[1:]

        for alpha in alphas:
            waypoint_transform = target_transform.copy()
            waypoint_transform[:3, 3] = (
                1.0 - alpha
            ) * current_position + alpha * target_transform[:3, 3]
            isometry = Isometry3d()
            isometry.setMatrix(waypoint_transform)
            program.appendMoveInstruction(
                MoveInstructionPoly_wrap_MoveInstruction(
                    MoveInstruction(
                        CartesianWaypointPoly_wrap_CartesianWaypoint(
                            CartesianWaypoint(isometry)
                        ),
                        MoveInstructionType_FREESPACE,
                        DEFAULT_PROFILE,
                        self._manipulator_info,
                    )
                )
            )
        return program
