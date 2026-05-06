from __future__ import annotations

import math

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .aic_task_env_cfg import AICTaskSceneCfg
from .mdp.events import randomize_board_and_parts


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
TCP_BODY = "wrist_3_link"
ROBOT_TCP_CFG = SceneEntityCfg("robot", body_names=TCP_BODY, joint_names=ARM_JOINTS)
INSERTION_AXIS = (0.0, 0.0, 1.0)


@configclass
class AICInsertionSceneCfg(AICTaskSceneCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True


@configclass
class ActionsCfg:
    arm_action: ActionTerm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name=TCP_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="svd",
            ik_params={"k_val": 1.0, "min_singular_value": 1e-5},
        ),
        scale=(0.0015, 0.0015, 0.0015, math.radians(1.0), math.radians(1.0), math.radians(1.0)),
    )


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )

    randomize_board_and_parts = EventTerm(
        func=randomize_board_and_parts,
        mode="reset",
        params={
            "board_scene_name": "task_board",
            "board_default_pos": (0.2837, 0.229, 0.0),
            "board_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005)},
            "parts": [
                {
                    "scene_name": "sc_port",
                    "offset": (0.0067, -0.0362, 0.005),
                    "pose_range": {"x": (-0.005, 0.02)},
                },
                {
                    "scene_name": "sc_port_2",
                    "offset": (0.0076, -0.0783, 0.005),
                    "pose_range": {"x": (-0.005, 0.02)},
                },
                {
                    "scene_name": "nic_card",
                    "offset": (-0.03235, 0.02329, 0.0743),
                    "pose_range": {"y": (0.0, 0.12)},
                    "snap_step": {"y": 0.04},
                },
            ],
            "sync_usd_xforms": False,
        },
    )

    sample_insertion_episode = EventTerm(
        func=mdp.sample_insertion_episode,
        mode="reset",
        params={
            "port_cfg": SceneEntityCfg("sc_port"),
            "entrance_offset_port": (0.0, 0.0, 0.0),
            "entrance_rpy_port": (0.0, 0.0, 0.0),
            "entrance_position_noise": {
                "x": (-0.002, 0.002),
                "y": (-0.002, 0.002),
                "z": (0.0, 0.0),
            },
            "entrance_rpy_noise": {
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.radians(1.0), math.radians(1.0)),
            },
            "gripper_plug_translation": {
                "x": (-0.003, 0.003),
                "y": (-0.003, 0.003),
                "z": (-0.002, 0.002),
            },
            "gripper_plug_rpy": {
                "roll": (-math.radians(2.0), math.radians(2.0)),
                "pitch": (-math.radians(2.0), math.radians(2.0)),
                "yaw": (-math.radians(2.0), math.radians(2.0)),
            },
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        insertion_actor = ObsTerm(
            func=mdp.insertion_actor_observation,
            params={
                "asset_cfg": ROBOT_TCP_CFG,
                "insertion_axis_entrance": INSERTION_AXIS,
                "position_scale_m": 0.05,
                "progress_scale_m": 0.02,
                "force_scale_n": 22.0,
                "torque_scale_nm": 2.0,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        privileged = ObsTerm(
            func=mdp.insertion_privileged_observation,
            params={
                "asset_cfg": ROBOT_TCP_CFG,
                "insertion_axis_entrance": INSERTION_AXIS,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    lateral_alignment = RewTerm(
        func=mdp.insertion_lateral_reward,
        weight=2.0,
        params={"asset_cfg": ROBOT_TCP_CFG, "sigma_m": 0.006},
    )
    angle_alignment = RewTerm(
        func=mdp.insertion_angle_reward,
        weight=1.0,
        params={"asset_cfg": ROBOT_TCP_CFG, "sigma_rad": 0.08},
    )
    insertion_depth = RewTerm(
        func=mdp.insertion_depth_reward,
        weight=2.5,
        params={
            "asset_cfg": ROBOT_TCP_CFG,
            "insertion_axis_entrance": INSERTION_AXIS,
            "depth_scale_m": 0.015,
        },
    )
    insertion_success = RewTerm(
        func=mdp.insertion_success,
        weight=8.0,
        params={
            "asset_cfg": ROBOT_TCP_CFG,
            "insertion_axis_entrance": INSERTION_AXIS,
            "lateral_threshold_m": 0.0025,
            "angle_threshold_rad": 0.04,
            "depth_threshold_m": 0.012,
        },
    )
    force_penalty = RewTerm(
        func=mdp.force_penalty,
        weight=-0.004,
        params={"asset_cfg": ROBOT_TCP_CFG},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    sapu_interpenetration = RewTerm(
        func=mdp.sapu_interpenetration_penalty,
        weight=-4.0,
        params={"asset_cfg": ROBOT_TCP_CFG, "threshold_m": 0.001},
    )
    sdf_query = RewTerm(
        func=mdp.sdf_query_reward,
        weight=0.5,
        params={"asset_cfg": ROBOT_TCP_CFG, "sigma_m": 0.006},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.insertion_success,
        params={
            "asset_cfg": ROBOT_TCP_CFG,
            "insertion_axis_entrance": INSERTION_AXIS,
            "lateral_threshold_m": 0.0025,
            "angle_threshold_rad": 0.04,
            "depth_threshold_m": 0.012,
        },
    )
    force_guard = DoneTerm(
        func=mdp.force_guard,
        params={"asset_cfg": ROBOT_TCP_CFG, "threshold_n": 22.0},
    )
    workspace_guard = DoneTerm(
        func=mdp.insertion_workspace_guard,
        params={"asset_cfg": ROBOT_TCP_CFG, "max_lateral_m": 0.08},
    )


@configclass
class CurriculumCfg:
    sampling = CurrTerm(
        func=mdp.insertion_sampling_curriculum,
        params={
            "asset_cfg": ROBOT_TCP_CFG,
            "insertion_axis_entrance": INSERTION_AXIS,
            "success_advance_threshold": 0.75,
            "success_revert_threshold": 0.50,
            "position_step_m": 0.005,
            "position_max_m": 0.05,
            "orientation_step_rad": math.radians(1.0),
            "orientation_max_rad": math.radians(10.0),
            "lateral_threshold_m": 0.0025,
            "angle_threshold_rad": 0.04,
            "depth_threshold_m": 0.012,
        },
    )


@configclass
class AICInsertionEnvCfg(ManagerBasedRLEnvCfg):
    scene: AICInsertionSceneCfg = AICInsertionSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 240.0
        self.episode_length_s = 3.0
        self.viewer.eye = (1.0, 0.0, 0.8)
