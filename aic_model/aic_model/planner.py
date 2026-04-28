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
        self._log_info = getattr(robot, "_log_info", None)
        self._log_warn = getattr(robot, "_log_warn", None)

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

    def _info(self, message: str) -> None:
        """Emit an informational planner log when a callback is available."""

        if self._log_info is not None:
            self._log_info(f"[planner] {message}")

    def _warn(self, message: str) -> None:
        """Emit a warning planner log and mirror it to Python warnings."""

        if self._log_warn is not None:
            self._log_warn(f"[planner] {message}")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    @contextmanager
    def _planning_timeout(self):
        """Apply a best-effort wall-clock timeout around a planning call."""

        if self.timeout <= 0.0:
            yield
            return

        try:
            previous_handler = signal.getsignal(signal.SIGALRM)
        except ValueError:
            self._warn("Planning timeout is unavailable outside the main thread.")
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

        self._info(
            "Submitting planning request "
            f"with {len(program.flatten())} flattened instructions and "
            f"timeout={self.timeout:.3f}s."
        )
        request = PlannerRequest()
        request.instructions = program
        request.env = self.env
        request.profiles = self._profiles
        try:
            with self._planning_timeout():
                response = self._motion_planner.solve(request)
        except TimeoutError as exc:
            self._warn(str(exc))
            return None
        except Exception as exc:
            self._warn(
                f"TrajOpt planning raised {exc.__class__.__name__}: {exc}"
            )
            return None

        if not response.successful:
            self._warn(f"TrajOpt failed: {response.message}")
            return None
        self._info(f"Planning succeeded: {response.message}")
        return response

    def PlanToTarget(
        self,
        target_transform: np.ndarray | list[list[float]],
    ) -> PlannerResponse | None:
        """Plan from the current robot state to a target TCP transform."""

        target_transform = np.asarray(target_transform, dtype=np.float64)
        self._info(
            "PlanToTarget request "
            f"target_position={np.array2string(target_transform[:3, 3], precision=3)}"
        )
        return self._solve_program(self._make_target_program(target_transform))

    def PlanToConfiguration(
        self,
        target_joint_values: np.ndarray | list[float],
    ) -> PlannerResponse | None:
        """Plan from the current robot state to a target joint configuration."""

        target_joint_values = np.asarray(target_joint_values, dtype=np.float64).reshape(
            -1
        )
        self._info(
            "PlanToConfiguration request "
            f"target_joint_values={np.array2string(target_joint_values, precision=3)}"
        )
        return self._solve_program(
            self._make_configuration_program(target_joint_values)
        )

    def Retime(
        self,
        program: CompositeInstruction,
    ) -> CompositeInstruction | None:
        """Apply Tesseract time-parameterization to an existing program."""

        self._info(
            "Retiming trajectory "
            f"with {len(program.flatten())} flattened instructions."
        )
        trajectory = InstructionsTrajectory(program)
        time_parameterization = TimeOptimalTrajectoryGeneration()
        ok = time_parameterization.compute(
            trajectory,
            self.velocity_limits,
            self.acceleration_limits,
            self.jerk_limits,
        )
        if not ok:
            self._warn("Time optimal trajectory generation failed.")
            return None

        self._info("Retiming succeeded.")
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
        self._info(
            "Building Cartesian program "
            f"start_position={np.array2string(current_transform[:3, 3], precision=3)} "
            f"target_position={np.array2string(target_transform[:3, 3], precision=3)} "
            f"num_waypoints={self.num_waypoints}"
        )
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
        self._info(
            "Building joint-space program "
            f"start={np.array2string(current_joint_values, precision=3)} "
            f"target={np.array2string(target_joint_values, precision=3)} "
            f"num_waypoints={self.num_waypoints}"
        )

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
