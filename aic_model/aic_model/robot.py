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
import logging
import os
import numpy as np

# External

from launch import LaunchContext
from launch_ros.substitutions import FindPackageShare
from tesseract_robotics.tesseract_common import (
    FilesystemPath,
    GeneralResourceLocator,
    Isometry3d,
)
from tesseract_robotics.tesseract_collision import (
    ContactManagerConfig,
    ContactRequest,
    ContactResultMap,
    ContactResultVector,
    ContactTestType_ALL,
)
from tesseract_robotics.tesseract_environment import Environment
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs
import xacro

# Internal

from aic_model_interfaces.msg import Observation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def with_latest_state(method):
    """Refresh the Tesseract environment from the latest observation."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.simulated:
            obs = self._get_observation()
            if obs is None:
                raise RuntimeError("Observation is unavailable.")
            self.env.setState(
                dict(zip(obs.joint_states.name, obs.joint_states.position))
            )
        return method(self, *args, **kwargs)

    return wrapper


class EnsemblRobot:
    """Wrap the AIC robot model in a Tesseract environment."""

    MANIPULATOR_GROUP_NAME = "manipulator"
    MANIPULATOR_BASE_FRAME = "base_link"
    MANIPULATOR_TIP_FRAME = "gripper/tcp"

    def __init__(self, get_observation: Callable[[], Observation] | None = None):
        """Initialize the robot model.

        If ``get_observation`` is ``None``, the robot runs in simulated mode and
        all active joints are initialized to zero.
        """
        self._get_observation = get_observation
        self.simulated = get_observation is None

        self.locator = self._create_resource_locator()
        self.env = Environment()
        context = LaunchContext()
        self.aic_description_share = FindPackageShare("aic_description").perform(
            context
        )
        self.urdf_xacro_path = f"{self.aic_description_share}/urdf/ur_gz.urdf.xacro"
        self.srdf_path = f"{self.aic_description_share}/srdf/ur_gz.srdf"
        urdf_xml = self._expand_xacro(self.urdf_xacro_path)
        srdf_xml = self._read_srdf(self.srdf_path)
        ok = self.env.init(
            urdf_xml,
            srdf_xml,
            self.locator,
        )
        if not ok:
            raise RuntimeError("Failed to initialize tesseract environment")
        logger.info("Tesseract environment initialized successfully")

        self._active_joint_names = list(self.env.getStateSolver().getActiveJointNames())
        self._joint_group = self.env.getJointGroup(self.MANIPULATOR_GROUP_NAME)
        self._manipulator_joint_names = list(self._joint_group.getJointNames())
        self._manipulator_target_frames = set(self._joint_group.getActiveLinkNames())
        self._kin_group = self._initialize_kinematic_group()
        self._collision_link_transforms_supported = hasattr(
            self.env.getState(), "link_transforms"
        )
        self._collision_manager_config = ContactManagerConfig()
        self._collision_manager_config.acm = self.env.getAllowedCollisionMatrix()
        self._collision_manager_config.margin_data = self.env.getCollisionMarginData()
        self._collision_request = ContactRequest()
        self._collision_request.type = ContactTestType_ALL
        self._collision_request.calculate_distance = True
        self._collision_request.calculate_penetration = True
        self._collision_request.contact_limit = 100
        self._collision_checker_warning_emitted = False
        self._has_discrete_contact_manager = self._initialize_discrete_contact_manager()

        if self.simulated:
            self.SetActiveDOFValues(
                np.zeros(len(self._active_joint_names), dtype=np.float64)
            )

    def _coerce_vector(
        self,
        values: np.ndarray | list[float],
        expected_size: int,
        label: str,
    ) -> np.ndarray:
        """Return a flat float vector with the expected size."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} {label} values, got {values.size}."
            )
        return values

    def _get_current_manipulator_joint_values(self) -> np.ndarray:
        """Return the current manipulator joint values as a flat array."""
        return np.asarray(
            self.env.getCurrentJointValues(self._manipulator_joint_names),
            dtype=np.float64,
        ).reshape(-1)

    def _compute_geometric_jacobian(
        self,
        joint_positions: np.ndarray,
        target_frame: str,
    ) -> np.ndarray:
        """Return the geometric Jacobian in Tesseract's row ordering."""
        return np.asarray(
            self._joint_group.calcJacobian(
                joint_positions,
                self.MANIPULATOR_BASE_FRAME,
                target_frame,
                np.zeros(3, dtype=np.float64),
            ),
            dtype=np.float64,
        )

    def _require_supported_target_frame(self, target_frame: str) -> str:
        """Return a target frame that belongs to the configured manipulator chain."""
        if target_frame not in self._manipulator_target_frames:
            raise ValueError(
                f"Target frame '{target_frame}' is not on the configured "
                f"manipulator chain."
            )
        return target_frame

    def _initialize_kinematic_group(self):
        """Return the configured kinematics group, if available."""
        try:
            return self.env.getKinematicGroup(self.MANIPULATOR_GROUP_NAME)
        except RuntimeError:
            return None

    def _initialize_discrete_contact_manager(self) -> bool:
        """Return whether collision checking is supported in this environment."""
        try:
            manager = self.env.getDiscreteContactManager()
        except RuntimeError:
            manager = None
        return bool(manager) and self._collision_link_transforms_supported

    def _disable_collision_checker(self, reason: str) -> None:
        """Disable collision checks after a plugin lookup or runtime failure."""
        self._has_discrete_contact_manager = False
        if not self._collision_checker_warning_emitted:
            logger.warning(
                "Collision checking is unavailable in this Tesseract environment; "
                "continuing without collision filtering. Reason: %s",
                reason,
            )
            self._collision_checker_warning_emitted = True

    def _get_discrete_contact_manager(self):
        """Return the discrete contact manager, disabling checks if unavailable."""
        if not self._has_discrete_contact_manager:
            self._disable_collision_checker(
                "no discrete contact manager plugin is configured"
            )
            return None

        try:
            manager = self.env.getDiscreteContactManager()
        except RuntimeError as exc:
            self._disable_collision_checker(str(exc))
            return None

        if not manager:
            self._disable_collision_checker(
                "no discrete contact manager plugin is configured"
            )
            return None

        return manager

    def _require_kinematics_plugin(self) -> None:
        """Raise if no kinematics plugin is configured for the manipulator."""
        if self._kin_group is None:
            raise RuntimeError(
                "This operation requires a kinematics plugin configured for the "
                f"'{self.MANIPULATOR_GROUP_NAME}' group."
            )

    def _get_current_link_transforms(self):
        """Return the environment's current link transform map."""
        return self.env.getState().link_transforms

    def _flatten_contact_results(self, collisions: ContactResultMap) -> list:
        """Return contact results as a Python list."""
        flattened = ContactResultVector()
        collisions.flattenCopyResults(flattened)
        return [flattened[i] for i in range(flattened.size())]

    def _get_actual_collision_contacts(self, collisions: ContactResultMap) -> list:
        """Return only contacts that are touching or penetrating."""
        return [
            contact
            for contact in self._flatten_contact_results(collisions)
            if contact.distance <= 0.0
        ]

    def _build_collision_report(self, contacts: list) -> list[dict]:
        """Convert actual collision contacts into a compact Python report."""
        report = []
        for contact in contacts:
            link_names = tuple(str(name) for name in contact.link_names)
            report.append(
                {
                    "pair": link_names,
                    "distance": float(contact.distance),
                    "type_id": int(contact.type_id),
                    "single_contact_point": bool(contact.single_contact_point),
                }
            )
        report.sort(key=lambda item: (item["distance"], item["pair"]))
        return report

    def _create_resource_locator(self) -> GeneralResourceLocator:
        """Create a Tesseract resource locator from the ROS prefix path."""
        locator = GeneralResourceLocator()
        for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
            if not prefix:
                continue
            share_dir = os.path.join(prefix, "share")
            if os.path.isdir(share_dir):
                locator.addPath(FilesystemPath(share_dir))
        return locator

    def _expand_xacro(self, xacro_path: str) -> str:
        """Expand the robot xacro into a URDF XML string."""
        try:
            doc = xacro.process_file(xacro_path, mappings={"name": "ur"})
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
        """Read the SRDF file into a string."""
        with open(srdf_path, encoding="utf-8") as f:
            return f.read()

    @with_latest_state
    def GetActiveDOFValues(self) -> np.ndarray:
        """Return the current active joint values in Tesseract active-joint order."""
        return np.asarray(self.env.getCurrentJointValues(), dtype=np.float64).reshape(
            -1
        )

    def SetActiveDOFValues(self, joint_values: np.ndarray | list[float]) -> None:
        """Set the active joint values when the robot is running in simulated mode."""
        if not self.simulated:
            raise RuntimeError(
                "SetActiveDOFValues is only available when get_observation is None."
            )

        joint_values = self._coerce_vector(
            joint_values,
            len(self._active_joint_names),
            "active joint",
        )
        self.env.setState(self._active_joint_names, joint_values)

    @with_latest_state
    def GetEnv(self) -> Environment:
        """Return the refreshed Tesseract environment."""
        return self.env

    @with_latest_state
    def ComputeFK(
        self,
        target_frame: str = MANIPULATOR_TIP_FRAME,
    ) -> np.ndarray:
        """Return the current 4x4 transform of ``target_frame`` relative to ``base_link``."""
        joint_positions = self._get_current_manipulator_joint_values()
        target_frame = self._require_supported_target_frame(target_frame)
        link_transforms = self._joint_group.calcFwdKin(joint_positions)
        return np.asarray(
            (
                link_transforms[self.MANIPULATOR_BASE_FRAME].inverse()
                * link_transforms[target_frame]
            ).matrix(),
            dtype=np.float64,
        )

    @with_latest_state
    def ComputeJacobianGeomtric(
        self,
        target_frame: str = MANIPULATOR_TIP_FRAME,
    ) -> np.ndarray:
        """Return the current geometric Jacobian in Tesseract's row ordering."""
        joint_positions = self._get_current_manipulator_joint_values()
        target_frame = self._require_supported_target_frame(target_frame)
        return self._compute_geometric_jacobian(joint_positions, target_frame)

    @with_latest_state
    def ComputeJacobianAnalytical(
        self,
        target_frame: str = MANIPULATOR_TIP_FRAME,
    ) -> np.ndarray:
        """Return the current analytical Jacobian for XYZ plus roll/pitch/yaw."""
        joint_positions = self._get_current_manipulator_joint_values()
        target_frame = self._require_supported_target_frame(target_frame)
        jacobian_geometric = self._compute_geometric_jacobian(
            joint_positions, target_frame
        )
        link_transforms = self._joint_group.calcFwdKin(joint_positions)
        rotation = np.asarray(
            (
                link_transforms[self.MANIPULATOR_BASE_FRAME].inverse()
                * link_transforms[target_frame]
            ).matrix(),
            dtype=np.float64,
        )[:3, :3]

        pitch = np.arctan2(
            -rotation[2, 0],
            np.hypot(rotation[0, 0], rotation[1, 0]),
        )
        cos_pitch = np.cos(pitch)
        if np.isclose(cos_pitch, 0.0):
            raise RuntimeError(
                "ComputeJacobianAnalytical is singular at pitch = +/- pi/2."
            )
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])

        angular_velocity_to_rpy_rates = np.array(
            [
                [1.0, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)],
                [0.0, np.cos(roll), -np.sin(roll)],
                [0.0, np.sin(roll) / cos_pitch, np.cos(roll) / cos_pitch],
            ],
            dtype=np.float64,
        )
        return np.vstack(
            [
                jacobian_geometric[:3, :],
                angular_velocity_to_rpy_rates @ jacobian_geometric[3:, :],
            ]
        )

    @with_latest_state
    def ComputeHessian(
        self,
        target_frame: str = MANIPULATOR_TIP_FRAME,
    ) -> np.ndarray:
        """Return the current numerical geometric Hessian relative to ``base_link``."""
        joint_positions = self._get_current_manipulator_joint_values()
        target_frame = self._require_supported_target_frame(target_frame)
        link_point = np.zeros(3, dtype=np.float64)
        step = 1e-6
        hessian = np.zeros(
            (6, joint_positions.size, joint_positions.size),
            dtype=np.float64,
        )

        for i in range(joint_positions.size):
            q_plus = joint_positions.copy()
            q_plus[i] += step
            q_minus = joint_positions.copy()
            q_minus[i] -= step
            jac_plus = np.asarray(
                self._joint_group.calcJacobian(
                    q_plus,
                    self.MANIPULATOR_BASE_FRAME,
                    target_frame,
                    link_point,
                ),
                dtype=np.float64,
            )
            jac_minus = np.asarray(
                self._joint_group.calcJacobian(
                    q_minus,
                    self.MANIPULATOR_BASE_FRAME,
                    target_frame,
                    link_point,
                ),
                dtype=np.float64,
            )
            hessian[:, :, i] = (jac_plus - jac_minus) / (2.0 * step)

        return hessian

    @with_latest_state
    def ComputeIK(
        self,
        transform: np.ndarray | list[list[float]],
        return_all: bool = False,
        check_collision: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        """Return IK solutions for the configured manipulator tip in ``base_link``.

        When ``return_all`` is ``False``, the solution closest to the current
        manipulator state is returned. Otherwise all solutions are returned.
        When ``check_collision`` is ``True``, colliding IK solutions are filtered out.
        """
        self._require_kinematics_plugin()

        transform_matrix = np.asarray(transform, dtype=np.float64)
        if transform_matrix.shape != (4, 4):
            raise ValueError(
                f"Expected transform with shape (4, 4), got {transform_matrix.shape}."
            )
        transform_isometry = Isometry3d()
        transform_isometry.setMatrix(transform_matrix)

        ik_input = KinGroupIKInput()
        ik_input.pose = transform_isometry
        ik_input.tip_link_name = self.MANIPULATOR_TIP_FRAME
        ik_input.working_frame = self.MANIPULATOR_BASE_FRAME
        ik_inputs = KinGroupIKInputs()
        ik_inputs.append(ik_input)

        seed = self._get_current_manipulator_joint_values()
        raw_solutions = self._kin_group.calcInvKin(ik_inputs, seed)
        solutions = [
            np.asarray(raw_solutions[i], dtype=np.float64).reshape(-1)
            for i in range(len(raw_solutions))
        ]
        if not solutions:
            raise RuntimeError("No IK solutions found.")

        if check_collision:
            manager = self._get_discrete_contact_manager()
            if manager is not None:
                current_active_joint_values = np.asarray(
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
                        self._active_joint_names, current_active_joint_values
                    )
                solutions = collision_free_solutions
                if not solutions:
                    raise RuntimeError("No collision-free IK solutions found.")

        if return_all:
            return solutions

        solution_array = np.stack(solutions, axis=0)
        wrapped_delta = ((solution_array - seed + np.pi) % (2.0 * np.pi)) - np.pi
        return solution_array[np.argmin(np.linalg.norm(wrapped_delta, axis=1))]

    @with_latest_state
    def CheckCollision(self, report: bool = False) -> bool | tuple[bool, list[dict]]:
        """Check the current state for self and environment collisions.

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
        manager = self._get_discrete_contact_manager()
        if manager is None:
            if not report:
                return False
            return False, []
        manager.setCollisionObjectsTransform(self._get_current_link_transforms())
        # This cached config is fast, but if ACM or margins change later
        # (for example after attaching an object), it must be refreshed.
        manager.applyContactManagerConfig(self._collision_manager_config)

        collisions = ContactResultMap()
        manager.contactTest(collisions, self._collision_request)

        actual_collision_contacts = self._get_actual_collision_contacts(collisions)
        if not report:
            return bool(actual_collision_contacts)

        collision_report = self._build_collision_report(actual_collision_contacts)
        return bool(collision_report), collision_report


if __name__ == "__main__":
    robot = EnsemblRobot()

    arm_joint_positions = np.array(
        [0.25, -1.10, 1.35, -1.55, -1.20, 0.45],
        dtype=np.float64,
    )
    active_joint_positions = np.concatenate(
        [arm_joint_positions, np.array([0.01, 0.01], dtype=np.float64)]
    )

    robot.SetActiveDOFValues(active_joint_positions)
    measured_active_joint_positions = robot.GetActiveDOFValues()
    assert np.allclose(measured_active_joint_positions, active_joint_positions)

    tool_transform = robot.ComputeFK()
    hessian = robot.ComputeHessian()
    assert hessian.shape == (6, 6, 6)
    assert np.all(np.isfinite(hessian))

    ik_solutions = robot.ComputeIK(
        tool_transform,
        return_all=True,
        check_collision=False,
    )
    assert ik_solutions

    for ik_solution in ik_solutions:
        robot.env.setState(robot._manipulator_joint_names, ik_solution)
        solution_fk = robot.ComputeFK()
        assert np.allclose(solution_fk, tool_transform, atol=1e-4)

    robot.SetActiveDOFValues(active_joint_positions)

    ik_solution_array = np.stack(ik_solutions, axis=0)
    wrapped_delta = (
        (ik_solution_array - arm_joint_positions + np.pi) % (2.0 * np.pi)
    ) - np.pi
    closest_index = np.argmin(np.linalg.norm(wrapped_delta, axis=1))
    closest_distance = np.linalg.norm(wrapped_delta[closest_index])
    assert closest_distance < 1e-4

    closest_solution = robot.ComputeIK(tool_transform, check_collision=False)
    assert np.allclose(closest_solution, ik_solution_array[closest_index])

    print("GetActiveDOFValues passed.")
    print("ComputeFK passed.")
    print("ComputeHessian passed.")
    print("ComputeIK passed.")
