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

from contextlib import contextmanager
import os
import signal
import warnings
import numpy as np

os.environ.setdefault("TRAJOPT_LOG_THRESH", "ERROR")

from tesseract_robotics.tesseract_command_language import (
    CartesianWaypoint,
    CartesianWaypointPoly_wrap_CartesianWaypoint,
    CompositeInstruction,
    MoveInstruction,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType_FREESPACE,
    MoveInstructionType_LINEAR,
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
    """Build and solve native Tesseract motion-planning programs."""

    def __init__(
        self,
        robot,
        num_waypoints: int = 5,
        timeout: float = 10.0,
    ):
        """Initialize planner state for a specific robot and manipulator."""

        self.robot = robot
        self.env = robot.env
        self.manipulator_group_name = robot.manipulator_group_name
        self.base_frame = robot.manipulator_base_frame
        self.tip_frame = robot.manipulator_tip_frame
        self.manipulator_joint_names = list(robot._manipulator_joint_names)
        self.velocity_limits = robot.velocity_limits
        self.acceleration_limits = robot.acceleration_limits
        self.jerk_limits = robot.jerk_limits
        self.num_waypoints = int(num_waypoints)
        self.timeout = float(timeout)

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

    @contextmanager
    def _planning_timeout(self):
        """Apply a best-effort wall-clock timeout around a planning call."""

        if self.timeout <= 0.0:
            yield
            return

        try:
            previous_handler = signal.getsignal(signal.SIGALRM)
        except ValueError:
            warnings.warn(
                "Planning timeout is unavailable outside the main thread.",
                RuntimeWarning,
                stacklevel=2,
            )
            yield
            return

        def _raise_timeout(_signum, _frame):
            """Raise a Python timeout once the wall-clock alarm expires."""

            raise TimeoutError(
                f"Planning exceeded timeout of {self.timeout:.3f} seconds."
            )

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _solve_program(self, program: CompositeInstruction) -> PlannerResponse | None:
        """Solve a Tesseract program and return the native planner response."""

        request = PlannerRequest()
        request.instructions = program
        request.env = self.env
        request.profiles = self._profiles
        try:
            with self._planning_timeout():
                response = self._motion_planner.solve(request)
        except TimeoutError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            return None
        except Exception as exc:
            warnings.warn(
                f"TrajOpt planning raised {exc.__class__.__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        if not response.successful:
            warnings.warn(
                f"TrajOpt failed: {response.message}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return response

    def PlanToTarget(
        self,
        target_transform: np.ndarray | list[list[float]],
    ) -> PlannerResponse | None:
        """Plan from the current robot state to a target TCP transform."""

        return self._solve_program(self._make_target_program(target_transform))

    def PlanToConfiguration(
        self,
        target_joint_values: np.ndarray | list[float],
    ) -> PlannerResponse | None:
        """Plan from the current robot state to a target joint configuration."""

        return self._solve_program(
            self._make_configuration_program(target_joint_values)
        )

    def Retime(
        self,
        program: CompositeInstruction,
    ) -> CompositeInstruction | None:
        """Apply Tesseract time-parameterization to an existing program."""

        trajectory = InstructionsTrajectory(program)
        time_parameterization = TimeOptimalTrajectoryGeneration()
        ok = time_parameterization.compute(
            trajectory,
            self.velocity_limits,
            self.acceleration_limits,
            self.jerk_limits,
        )
        if not ok:
            warnings.warn(
                "Time optimal trajectory generation failed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        return program

    def _make_target_program(
        self,
        target_transform: np.ndarray | list[list[float]],
    ) -> CompositeInstruction:
        """Create a Cartesian planning program from the current state to a pose."""

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
        target_position = target_transform[:3, 3]
        interpolated_positions = current_position + np.outer(
            np.linspace(0.0, 1.0, num=self.num_waypoints)[1:],
            target_position - current_position,
        )
        waypoint_transform = np.array(target_transform, copy=True)

        for position in interpolated_positions:
            waypoint_transform[:3, 3] = position
            isometry = Isometry3d()
            isometry.setMatrix(waypoint_transform)
            program.appendMoveInstruction(
                MoveInstructionPoly_wrap_MoveInstruction(
                    MoveInstruction(
                        CartesianWaypointPoly_wrap_CartesianWaypoint(
                            CartesianWaypoint(isometry)
                        ),
                        MoveInstructionType_LINEAR,
                        DEFAULT_PROFILE,
                        self._manipulator_info,
                    )
                )
            )
        return program

    def _make_configuration_program(
        self,
        target_joint_values: np.ndarray | list[float],
    ) -> CompositeInstruction:
        """Create a joint-space planning program from the current state."""

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

        target_joint_values = np.asarray(
            target_joint_values,
            dtype=np.float64,
        ).reshape(-1)
        if target_joint_values.size != len(self.manipulator_joint_names):
            raise ValueError(
                "Expected target joint values with shape "
                f"({len(self.manipulator_joint_names)},), got "
                f"{target_joint_values.shape}."
            )

        program = CompositeInstruction(DEFAULT_PROFILE)
        program.setManipulatorInfo(self._manipulator_info)

        for alpha in np.linspace(0.0, 1.0, num=self.num_waypoints):
            joint_values = (
                (1.0 - alpha) * current_joint_values
                + alpha * target_joint_values
            )
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

        return program
