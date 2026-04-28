from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

os.environ.setdefault("TRAJOPT_LOG_THRESH", "ERROR")

from tesseract_robotics.tesseract_command_language import (
    CartesianWaypoint,
    CartesianWaypointPoly_wrap_CartesianWaypoint,
    CompositeInstruction,
    InstructionPoly_as_MoveInstructionPoly,
    MoveInstruction,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType_FREESPACE,
    ProfileDictionary,
    StateWaypoint,
    StateWaypointPoly_wrap_StateWaypoint,
    WaypointPoly_as_StateWaypointPoly,
)
from tesseract_robotics.tesseract_common import Isometry3d, ManipulatorInfo
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_ompl import (
    OMPLMotionPlanner,
    OMPLRealVectorPlanProfile,
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
OMPL_NAMESPACE = "OMPLMotionPlannerTask"


@dataclass
class PlannerResult:
    planner_name: str
    path: np.ndarray
    response: object | None = None


@dataclass
class RetimedTrajectory:
    path: np.ndarray
    time: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    program: CompositeInstruction


class ConfigurationPlanner:
    name = "configuration_planner"

    def plan(self, context: "EnsemblPlanner", program: CompositeInstruction):
        raise NotImplementedError


class TrajOptConfigurationPlanner(ConfigurationPlanner):
    name = "trajopt"
    namespace = TRAJOPT_NAMESPACE

    def __init__(self):
        self._planner = TrajOptMotionPlanner(self.namespace)
        self._profiles = ProfileDictionary()
        self._profiles.addProfile(
            self.namespace,
            DEFAULT_PROFILE,
            TrajOptDefaultPlanProfile(),
        )
        self._profiles.addProfile(
            self.namespace,
            DEFAULT_PROFILE,
            TrajOptDefaultCompositeProfile(),
        )

    def plan(self, context: "EnsemblPlanner", program: CompositeInstruction):
        request = PlannerRequest()
        request.instructions = program
        request.env = context.env
        request.profiles = self._profiles
        return self._planner.solve(request)


class OMPLConfigurationPlanner(ConfigurationPlanner):
    """
    OMPL planner wrapper.

    The current SWIG bindings expose planner configurator classes such as
    RRTstarConfigurator and BiTRRTConfigurator, but do not expose a usable
    OMPLSolverConfig wrapper for assigning them to OMPLRealVectorPlanProfile.
    This wrapper therefore uses Tesseract's default OMPL planner configuration.
    In this build that default includes RRTConnect.
    """

    namespace = OMPL_NAMESPACE

    def __init__(self, name: str = "ompl_default"):
        self.name = name
        self._planner = OMPLMotionPlanner(self.namespace)
        self._profiles = ProfileDictionary()
        self._profiles.addProfile(
            self.namespace,
            DEFAULT_PROFILE,
            OMPLRealVectorPlanProfile(),
        )

    def plan(self, context: "EnsemblPlanner", program: CompositeInstruction):
        request = PlannerRequest()
        request.instructions = program
        request.env = context.env
        request.profiles = self._profiles
        return self._planner.solve(request)


class OMPLThenTrajOptConfigurationPlanner(ConfigurationPlanner):
    name = "ompl_then_trajopt"

    def __init__(self):
        self._ompl = OMPLConfigurationPlanner("ompl_seed")
        self._trajopt = TrajOptConfigurationPlanner()

    def plan(self, context: "EnsemblPlanner", program: CompositeInstruction):
        ompl_response = self._ompl.plan(context, program)
        if not ompl_response.successful:
            return ompl_response
        return self._trajopt.plan(context, ompl_response.results)


class EnsemblPlanner:
    def __init__(
        self,
        env,
        manipulator_group_name: str,
        base_frame: str,
        tip_frame: str,
        manipulator_joint_names: list[str],
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        jerk_limits: np.ndarray,
        num_waypoints: int = 5,
    ):
        self.env = env
        self.manipulator_group_name = manipulator_group_name
        self.base_frame = base_frame
        self.tip_frame = tip_frame
        self.manipulator_joint_names = list(manipulator_joint_names)
        self.velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
        self.acceleration_limits = np.asarray(acceleration_limits, dtype=np.float64)
        self.jerk_limits = np.asarray(jerk_limits, dtype=np.float64)
        self.num_waypoints = int(num_waypoints)

        self._manipulator_info = ManipulatorInfo()
        self._manipulator_info.manipulator = self.manipulator_group_name
        self._manipulator_info.working_frame = self.base_frame
        self._manipulator_info.tcp_frame = self.tip_frame

        # Edit this list to change planner order.
        self._configuration_planners: list[ConfigurationPlanner] = [
            TrajOptConfigurationPlanner(),
            OMPLConfigurationPlanner("ompl_default_rrtconnect"),
            OMPLThenTrajOptConfigurationPlanner(),
        ]

    def PlanToTarget(
        self,
        start_joint_values: np.ndarray | list[float],
        target_transform: np.ndarray | list[list[float]],
        start_transform: np.ndarray | list[list[float]] | None = None,
    ) -> PlannerResult:
        program = self._make_target_program(
            start_joint_values,
            target_transform,
            start_transform,
        )
        messages = []
        for planner in self._configuration_planners:
            response = planner.plan(self, program)
            if response.successful:
                return PlannerResult(
                    planner_name=planner.name,
                    path=self._extract_path(response.results),
                    response=response,
                )
            messages.append(f"{planner.name}: {response.message}")
        raise RuntimeError("All configuration planners failed: " + "; ".join(messages))

    def Retime(
        self,
        path: np.ndarray | list[list[float]],
    ) -> RetimedTrajectory:
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != len(self.manipulator_joint_names):
            raise ValueError(
                "Expected path with shape "
                f"(N, {len(self.manipulator_joint_names)}), got {path.shape}."
            )

        program = self._make_state_program(path)
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

        positions = []
        velocities = []
        accelerations = []
        times = []
        for instruction in program.flatten():
            move = InstructionPoly_as_MoveInstructionPoly(instruction)
            waypoint = WaypointPoly_as_StateWaypointPoly(move.getWaypoint())
            positions.append(
                np.asarray(waypoint.getPosition(), dtype=np.float64).reshape(-1)
            )
            velocities.append(
                np.asarray(waypoint.getVelocity(), dtype=np.float64).reshape(-1)
            )
            accelerations.append(
                np.asarray(waypoint.getAcceleration(), dtype=np.float64).reshape(-1)
            )
            times.append(float(waypoint.getTime()))

        return RetimedTrajectory(
            path=np.vstack(positions),
            time=np.asarray(times, dtype=np.float64),
            velocity=np.vstack(velocities),
            acceleration=np.vstack(accelerations),
            program=program,
        )

    def _make_target_program(
        self,
        start_joint_values: np.ndarray | list[float],
        target_transform: np.ndarray | list[list[float]],
        start_transform: np.ndarray | list[list[float]] | None,
    ) -> CompositeInstruction:
        start_joint_values = np.asarray(start_joint_values, dtype=np.float64).reshape(-1)
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
                            start_joint_values,
                        )
                    ),
                    MoveInstructionType_FREESPACE,
                    DEFAULT_PROFILE,
                    self._manipulator_info,
                )
            )
        )

        if start_transform is None:
            alphas = np.linspace(1.0, 1.0, num=max(self.num_waypoints - 1, 1))
            start_position = target_transform[:3, 3]
        else:
            start_transform = np.asarray(start_transform, dtype=np.float64)
            if start_transform.shape != (4, 4):
                raise ValueError(
                    "Expected start transform with shape "
                    f"(4, 4), got {start_transform.shape}."
                )
            alphas = np.linspace(0.0, 1.0, num=self.num_waypoints)[1:]
            start_position = start_transform[:3, 3]

        for alpha in alphas:
            waypoint_transform = target_transform.copy()
            waypoint_transform[:3, 3] = (
                (1.0 - alpha) * start_position
                + alpha * target_transform[:3, 3]
            )
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

    def _make_state_program(self, path: np.ndarray) -> CompositeInstruction:
        program = CompositeInstruction(DEFAULT_PROFILE)
        program.setManipulatorInfo(self._manipulator_info)
        for joint_values in path:
            program.appendMoveInstruction(
                MoveInstructionPoly_wrap_MoveInstruction(
                    MoveInstruction(
                        StateWaypointPoly_wrap_StateWaypoint(
                            StateWaypoint(
                                self.manipulator_joint_names,
                                np.asarray(joint_values, dtype=np.float64),
                            )
                        ),
                        MoveInstructionType_FREESPACE,
                        DEFAULT_PROFILE,
                        self._manipulator_info,
                    )
                )
            )
        return program

    def _extract_path(self, program: CompositeInstruction) -> np.ndarray:
        positions = []
        for instruction in program.flatten():
            move = InstructionPoly_as_MoveInstructionPoly(instruction)
            waypoint = move.getWaypoint()
            if not waypoint.isStateWaypoint():
                continue
            state_waypoint = WaypointPoly_as_StateWaypointPoly(waypoint)
            positions.append(
                np.asarray(state_waypoint.getPosition(), dtype=np.float64).reshape(-1)
            )
        if not positions:
            raise RuntimeError("Planner response did not contain state waypoints.")
        return np.vstack(positions)
