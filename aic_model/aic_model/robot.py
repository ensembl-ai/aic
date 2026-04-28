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
from functools import wraps
import inspect
import logging
import os
from typing import Any, cast

import numpy as np
import yaml

# External

from ament_index_python.packages import get_package_share_directory
from tesseract_robotics.tesseract_command_language import CompositeInstruction
from tesseract_robotics.tesseract_common import (
    FilesystemPath,
    GeneralResourceLocator,
    Isometry3d,
    VectorIsometry3d,
)
from tesseract_robotics.tesseract_collision import (
    ContactManagerConfig,
    ContactRequest,
    ContactResultMap,
    ContactResultVector,
    ContactTestType_ALL,
    ContactTestType_FIRST,
)
from tesseract_robotics.tesseract_environment import Environment
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs
from tesseract_robotics.tesseract_motion_planners import PlannerResponse
import xacro

# Internal

from aic_model_interfaces.msg import Observation
from aic_model.planner import (
    EnsemblPlanner,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def with_latest_state(method):
    """
    Refresh the Tesseract environment from the latest observation.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._sync_latest_state()
        return method(self, *args, **kwargs)

    return wrapper


def with_resolved_frames(method):
    """
    Resolve and validate base and target frames used by kinematics calls.
    """

    signature = inspect.signature(method)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        base_frame = bound.arguments.get("base_frame") or self.manipulator_base_frame
        target_frame = bound.arguments.get("target_frame") or self.manipulator_tip_frame

        if not all(
            frame in self._manipulator_link_names
            for frame in [target_frame, base_frame]
        ):
            raise ValueError(
                f"Target frame '{target_frame}' or Base frame '{base_frame}' is not "
                "on the configured manipulator chain."
            )

        bound.arguments["base_frame"] = base_frame
        bound.arguments["target_frame"] = target_frame
        return method(*bound.args, **bound.kwargs)

    return wrapper


class EnsemblRobot:

    def __init__(self, get_observation: Callable[[], Observation] | None = None):
        """
        Initialize the robot model.
        If ``get_observation`` is ``None``, the robot runs in simulated mode and
        all active joints are initialized to zero.
        """

        try:
            self._get_observation = get_observation
            self.simulated = get_observation is None

            self.aic_description_share = get_package_share_directory("aic_description")
            self.urdf_xacro_path = f"{self.aic_description_share}/urdf/ur_gz.urdf.xacro"
            self.srdf_path = f"{self.aic_description_share}/srdf/ur_gz.srdf"
            self.kinematics_config_path = (
                f"{self.aic_description_share}/srdf/robot_kinematics_plugins.yaml"
            )
            self.kinematics_config = self._load_kinematics_config()
            (
                self.manipulator_group_name,
                self.manipulator_base_frame,
                self.manipulator_tip_frame,
            ) = self._read_manipulator_config()
            collision_config = self.kinematics_config["collision_checking"]
            self.bool_contact_limit = int(collision_config["bool_contact_limit"])
            self.report_contact_limit = int(collision_config["report_contact_limit"])

            self.locator = GeneralResourceLocator()
            for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
                share_dir = os.path.join(prefix, "share")
                if os.path.isdir(share_dir):
                    self.locator.addPath(FilesystemPath(share_dir))

            self.env = Environment()
            urdf_xml = self._expand_xacro(self.urdf_xacro_path)
            ok = self.env.init(
                urdf_xml,
                self._read_srdf(self.srdf_path),
                self.locator,
            )
            if not ok:
                raise RuntimeError("env.init returned False")

            self._active_joint_names = list(
                self.env.getStateSolver().getActiveJointNames()
            )
            self._joint_group = self.env.getJointGroup(self.manipulator_group_name)
            self._kin_group = self.env.getKinematicGroup(self.manipulator_group_name)
            self._manipulator_link_names = set(self._joint_group.getActiveLinkNames())
            self._manipulator_link_names.update(
                [self.manipulator_base_frame, self.manipulator_tip_frame]
            )
            self._manipulator_joint_names = list(self._joint_group.getJointNames())
            self._active_dof_limits = np.asarray(
                self._joint_group.getLimits().joint_limits,
                dtype=np.float64,
            ).T
            active_joint_index = {
                joint_name: index
                for index, joint_name in enumerate(self._active_joint_names)
            }
            self._manipulator_joint_indices = [
                active_joint_index[joint_name]
                for joint_name in self._manipulator_joint_names
            ]
            self._collision_manager = self.env.getDiscreteContactManager()
            self._collision_object_names = list(
                self._collision_manager.getCollisionObjects()
            )
            self._collision_manager.setActiveCollisionObjects(
                self._collision_object_names
            )
            self._collision_object_poses = VectorIsometry3d()
            self._collision_config = ContactManagerConfig()
            self._collision_config.acm = self.env.getAllowedCollisionMatrix()
            self._collision_config.margin_data = self.env.getCollisionMarginData()
            self._collision_manager.applyContactManagerConfig(self._collision_config)
            self._bool_contact_request = ContactRequest()
            self._bool_contact_request.type = ContactTestType_FIRST
            self._bool_contact_request.calculate_distance = False
            self._bool_contact_request.calculate_penetration = False
            self._bool_contact_request.contact_limit = self.bool_contact_limit
            self._report_contact_request = ContactRequest()
            self._report_contact_request.type = ContactTestType_ALL
            self._report_contact_request.calculate_distance = True
            self._report_contact_request.calculate_penetration = True
            self._report_contact_request.contact_limit = self.report_contact_limit
            self._planner = EnsemblPlanner(self)
            if self.simulated:
                self.env.setState(
                    self._active_joint_names,
                    np.zeros(len(self._active_joint_names), dtype=np.float64),
                )

            logger.info("Tesseract environment initialized successfully")
        except Exception as exc:
            raise RuntimeError("Failed to initialize EnsemblRobot.") from exc

    def _load_kinematics_config(self) -> dict:
        with open(self.kinematics_config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _read_manipulator_config(self) -> tuple[str, str, str]:
        fwd_groups = self.kinematics_config["kinematic_plugins"]["fwd_kin_plugins"]
        group_name = next(iter(fwd_groups))
        group_config = fwd_groups[group_name]
        default_plugin = group_config["default"]
        plugin_config = group_config["plugins"][default_plugin]["config"]
        return group_name, plugin_config["base_link"], plugin_config["tip_link"]

    def _sync_latest_state(self) -> None:
        if self.simulated:
            return

        assert self._get_observation is not None
        obs = self._get_observation()
        if obs is None:
            raise RuntimeError("Observation is unavailable.")

        observed_joint_positions = dict(
            zip(obs.joint_states.name, obs.joint_states.position)
        )
        missing_joint_names = [
            joint_name
            for joint_name in self._manipulator_joint_names
            if joint_name not in observed_joint_positions
        ]
        if missing_joint_names:
            raise RuntimeError(
                "Observation is missing manipulator joints required by Tesseract: "
                f"{missing_joint_names}."
            )

        self.env.setState(
            self._manipulator_joint_names,
            np.asarray(
                [
                    observed_joint_positions[joint_name]
                    for joint_name in self._manipulator_joint_names
                ],
                dtype=np.float64,
            ),
        )

    def _expand_xacro(self, xacro_path: str) -> str:
        """
        Expand the robot xacro into a URDF XML string.
        """

        try:
            doc = cast(Any, xacro.process_file(xacro_path, mappings={"name": "ur"}))
        except Exception as exc:
            raise RuntimeError(
                "Failed to expand robot xacro "
                f"'{xacro_path}'. This usually means a missing ROS package, "
                "an undefined xacro argument, or a bad include path. "
                "The current setup passes the mapping {'name': 'ur'}. "
                f"Original error: {exc}"
            ) from exc
        return doc.toxml()

    def _read_srdf(self, srdf_path: str) -> str:
        """
        Read the SRDF file into a string.
        """

        with open(srdf_path, encoding="utf-8") as f:
            return f.read()

    @with_latest_state
    def GetActiveDOFValues(
        self,
    ) -> np.ndarray:
        """
        Return the current active joint values in Tesseract active-joint order.
        """

        joint_values = np.asarray(
            self.env.getCurrentJointValues(),
            dtype=np.float64,
        ).reshape(-1)
        return joint_values[self._manipulator_joint_indices]

    def GetActiveDOFLimits(self) -> np.ndarray:
        """
        Return active DOF position limits in the same order as ``GetActiveDOFValues``.
        Row 0 contains lower bounds and row 1 contains upper bounds.
        """

        return self._active_dof_limits.copy()

    def PlanToTarget(
        self,
        transform: np.ndarray | list[list[float]],
    ) -> PlannerResponse:
        """
        Plan a joint-space path from the current manipulator state to a target TCP pose.
        """

        return self._planner.PlanToTarget(target_transform=transform)

    def Retime(
        self,
        program: CompositeInstruction,
    ) -> CompositeInstruction:
        """
        Time-parameterize a Tesseract trajectory.
        """

        return self._planner.Retime(program)

    def SetActiveDOFValues(self, joint_values: np.ndarray | list[float]) -> None:
        """
        Set the active joint values when the robot is running in simulated mode.
        """

        if not self.simulated:
            raise RuntimeError(
                "SetActiveDOFValues is only available when get_observation is None."
            )
        joint_values = np.asarray(joint_values, dtype=np.float64).reshape(-1)
        if joint_values.size != len(self._manipulator_joint_names):
            raise ValueError(
                f"Expected {len(self._manipulator_joint_names)} active joint values, "
                f"got {joint_values.size}."
            )
        self.env.setState(self._manipulator_joint_names, joint_values)

    @with_latest_state
    def GetEnv(self) -> Environment:
        """
        Return the refreshed Tesseract environment.
        """

        return self.env

    def _update_collision_manager_state(self) -> None:
        state_solver = self.env.getStateSolver()
        state_solver.setState(
            self._active_joint_names,
            np.asarray(self.env.getCurrentJointValues(), dtype=np.float64).reshape(-1),
        )
        scene_state = state_solver.getState()
        link_transforms = scene_state.link_transforms
        self._collision_object_poses.clear()
        for name in self._collision_object_names:
            self._collision_object_poses.append(link_transforms[name])
        self._collision_manager.setCollisionObjectsTransform(
            self._collision_object_names,
            self._collision_object_poses,
        )

    @with_latest_state
    @with_resolved_frames
    def ComputeFK(
        self,
        target_frame: str | None = None,
        base_frame: str | None = None,
    ) -> np.ndarray:
        """
        Return the current 4x4 transform of ``target_frame`` relative to ``base_frame``.
        """

        joint_positions = self.GetActiveDOFValues()
        link_transforms = self._joint_group.calcFwdKin(joint_positions)
        return np.asarray(
            (
                link_transforms[base_frame].inverse() * link_transforms[target_frame]
            ).matrix(),
            dtype=np.float64,
        )

    @with_latest_state
    @with_resolved_frames
    def ComputeJacobianGeometric(
        self,
        target_frame: str | None = None,
        base_frame: str | None = None,
    ) -> np.ndarray:
        """
        Return the current geometric Jacobian in Tesseract's row ordering.
        The np.zeros(3, dtype=np.float64) in calcJacobian(...) is the point on the target link
        where the Jacobian is evaluated. Zero means “at the target frame origin.”
        For TCP Jacobian, that is the right point: the TCP frame origin.
        """

        joint_positions = self.GetActiveDOFValues()
        return np.asarray(
            self._joint_group.calcJacobian(
                joint_positions,
                base_frame,
                target_frame,
                np.zeros(3, dtype=np.float64),
            ),
            dtype=np.float64,
        )

    @with_latest_state
    @with_resolved_frames
    def ComputeIK(
        self,
        transform: np.ndarray | list[list[float]],
        target_frame: str | None = None,
        base_frame: str | None = None,
        return_all: bool = False,
        check_collision: bool = True,
    ) -> np.ndarray | None:
        """
        Return IK solutions for the configured manipulator tip in ``base_link``.
        When ``return_all`` is ``False``, the solution closest to the current
        manipulator state is returned. Otherwise all solutions are returned.
        When ``check_collision`` is ``True``, colliding IK solutions are filtered out.
        """

        transform_matrix = np.asarray(transform, dtype=np.float64)
        if transform_matrix.shape != (4, 4):
            raise ValueError(
                f"Expected transform with shape (4, 4), got {transform_matrix.shape}."
            )
        transform_isometry = Isometry3d()
        transform_isometry.setMatrix(transform_matrix)

        ik_input = KinGroupIKInput()
        ik_input.pose = transform_isometry
        ik_input.tip_link_name = target_frame
        ik_input.working_frame = base_frame
        ik_inputs = KinGroupIKInputs()
        ik_inputs.append(ik_input)

        current_active_joint_values = self.GetActiveDOFValues()
        raw_solutions = self._kin_group.calcInvKin(
            ik_inputs, current_active_joint_values
        )
        if len(raw_solutions) == 0:
            return
        solutions = [
            np.asarray(raw_solutions[i], dtype=np.float64).reshape(-1)
            for i in range(len(raw_solutions))
        ]

        if check_collision:
            original_active_joint_values = np.asarray(
                self.env.getCurrentJointValues(),
                dtype=np.float64,
            ).reshape(-1)
            collision_free_solutions = []
            try:
                for solution in solutions:
                    self.env.setState(self._manipulator_joint_names, solution)
                    if not EnsemblRobot.CheckCollision.__wrapped__(self):
                        collision_free_solutions.append(solution)
            finally:
                self.env.setState(
                    self._active_joint_names, original_active_joint_values
                )
                solutions = collision_free_solutions

        if not solutions:
            return
        solutions = np.vstack(solutions)

        if return_all:
            return solutions

        delta = solutions - current_active_joint_values
        return solutions[np.argmin(np.linalg.norm(delta, axis=1))]

    @with_latest_state
    def CheckCollision(
        self,
        report: bool = False,
    ) -> bool | tuple[bool, list[dict]]:
        """
        Check the current state for self and environment collisions.
        When ``report`` is ``False``, return ``True`` if any actual collisions are
        present and ``False`` otherwise.
        When ``report`` is ``True``, return ``(in_collision, collision_report)``
        where ``collision_report`` is a list of dictionaries. Each dictionary
        includes:
        - ``pair``: a tuple of colliding link names
        - ``distance``: signed contact distance (penetration is <= 0)
        - ``type_id``: the Tesseract contact type id
        - ``single_contact_point``: whether the result is a single-point contact
        """

        self._update_collision_manager_state()
        if not report:
            collisions = ContactResultMap()
            self._collision_manager.contactTest(
                collisions,
                self._bool_contact_request,
            )
            contacts = ContactResultVector()
            collisions.flattenCopyResults(contacts)
            return contacts.size() > 0
        collisions = ContactResultMap()
        self._collision_manager.contactTest(
            collisions,
            self._report_contact_request,
        )
        contacts = ContactResultVector()
        collisions.flattenCopyResults(contacts)
        collision_report = []
        for i in range(contacts.size()):
            contact = contacts[i]
            collision_report.append(
                {
                    "pair": tuple(str(name) for name in contact.link_names),
                    "distance": float(contact.distance),
                    "type_id": (
                        int(contact.type_id.front()),
                        int(contact.type_id.back()),
                    ),
                    "single_contact_point": bool(contact.single_contact_point),
                }
            )
        return contacts.size() > 0, collision_report
