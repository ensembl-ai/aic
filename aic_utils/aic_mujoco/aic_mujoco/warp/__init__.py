"""Direct MuJoCo Warp stack for AIC insertion.

This package is the default R&D path:

  scene_warp.xml -> MuJoCo model -> MuJoCo Warp model/data -> batched stepping

The exported public API is Warp-first and does not expose the old CPU
``mujoco.MjData`` vector prototype as a training interface.
"""

from .physics import (
    DebugJointPolicy,
    SampleObservation,
    TrainingPhysicsConfig,
    apply_debug_joint_policy,
    initialize_data_from_config,
    make_debug_joint_policy,
    run_training_physics,
    sample_observations,
    task_entity_ids,
)

__all__ = [
    "DebugJointPolicy",
    "SampleObservation",
    "TrainingPhysicsConfig",
    "apply_debug_joint_policy",
    "initialize_data_from_config",
    "make_debug_joint_policy",
    "run_training_physics",
    "sample_observations",
    "task_entity_ids",
]
