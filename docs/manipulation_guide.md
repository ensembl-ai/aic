cd ws_aic/src/aic

# Ensure
pixi install
source /opt/ros/kilted/setup.bash

# Install aic_description and other tesseract package dependencies

```bash
source /opt/ros/kilted/setup.bash
colcon build --packages-select aic_assets aic_description --packages-ignore aic_gazebo aic_engine_interfaces --symlink-install --executor sequential
source install/setup.bash
pixi run python -m pip install tesseract-robotics==0.5.1
```

NOTE: The above dependencies are baked into docker and pixi toml

# ONLY for rebuilding when making changes to policy code

pixi reinstall ros-kilted-aic-model ros-kilted-aic-example-policies

# Test policy without gazebo gui
/entrypoint.sh ground_truth:=false start_aic_engine:=true gazebo_gui:=false
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.TestMove

## Robot API quick reference

Use `EnsemblRobot` from `aic_model.robot` as the facade for kinematics, collision
checking, planning, retiming, and execution. In a policy, construct it with the
runtime callback `get_observation`; without it, the robot runs in simulated/offline
mode and starts at zero joint state. Robot state is refreshed from `get_observation`
before FK, IK, collision checks, and planning calls that need the current joints.

```python
import numpy as np
from aic_model.robot import EnsemblRobot

robot = EnsemblRobot(
    get_observation=get_observation,
    execute_joint_motion=execute_joint_motion,  # omit if you will not execute
)
base_T_tip_current = robot.ComputeFK() # get's latest joint angles from observations
base_T_tcp_clearance_box = base_T_tip_current
box_name = robot.AddBoxKinbody(
    dim=[0.1, 0.1, 0.05],
    parent_T_box=base_T_tcp_clearance_box,
    body_name="tcp_clearance_box",  # omit for a UUID-generated unique name
)
# base_T_tip_target: gripper/tcp transform in base_frame, meters, shape (4,4)
base_T_tip_target = np.eye(4, dtype=np.float64)
ik_solutions = robot.ComputeIK(base_T_tip_target, return_all=True) # If return_all is false, it provides closest IK
current_in_collision, contacts = robot.CheckCollision(report=True) # report false is binary check without insights
plan = robot.PlanToTarget(base_T_tip_target)
if plan:
    trajectory = robot.Retime(plan.results)
    if trajectory:
        robot.ExecuteTrajectory(trajectory)  # Requires execute_joint_motion callback.
```

Frame arguments are only needed when you want something other than the configured
manipulator frames. `target_frame=None` or omitted means `robot.manipulator_tip_frame`
(`"gripper/tcp"` today). `base_frame=None` or omitted means
`robot.manipulator_base_frame` (`"base_link"` today). For FK, the returned matrix is
`base_frame_T_target_frame`.

For IK, the input transform is interpreted as the desired
`base_T_target` pose. Both frames must be links on the
configured manipulator chain, otherwise the call raises `ValueError`.

- `robot.ComputeFK(target_frame=None, base_frame=None) -> np.ndarray`: returns a
  `(4, 4)` `float64` homogeneous transform for the current state.
- `robot.ComputeIK(base_T_target, target_frame=None, base_frame=None, return_all=False, check_collision=True)`:
  `base_T_target` must be a `(4, 4)` array-like homogeneous matrix and is converted
  to `np.float64`. Returns the closest collision-free joint vector as shape
  `(6,)`, all solutions as shape `(N, 6)` when `return_all=True`, or `None` if
  no solution is found. Set `check_collision=False` to keep IK solutions that are
  kinematically valid but may collide.
- `robot.CheckCollision(report=False)`: checks the current robot/environment
  state. Returns `True`/`False`, or `(in_collision, contacts)` when
  `report=True`; each contact includes the link pair, signed distance, contact
  type ids, and whether it is a single contact point.
- `robot.AddBoxKinbody(dim, parent_T_box, parent_frame="base_link", collision_enabled=True, body_name=None)`:
  adds a fixed box collision body to the native Tesseract environment. `dim` is
  `[x, y, z]` in meters. `parent_T_box` is a `(4, 4)` homogeneous matrix for the
  box center expressed in `parent_frame`; the box axes follow the `parent_T_box`
  rotation. `body_name` is the Tesseract link name to create; omit it to use a
  UUID-generated unique name like `box_kinbody_<uuid>`. Returns the link name.
  The collision manager is refreshed immediately, so the box is included in
  `CheckCollision`, collision-filtered IK, and subsequent planning calls.
- `robot.PlanToTarget(base_T_tip_target, max_joint_delta=float("inf"))`: plans from the
  current state to the default TCP pose in the default base frame. `base_T_tip_target`
  has the same `(4, 4)` convention as default IK. `max_joint_delta` rejects IK
  goals whose largest per-joint move from the current state is too large. Returns
  a Tesseract `PlannerResponse` or `None`; use `plan.results` as the program.
  For attached frames, compute the target TCP transform first, then pass that
  into `PlanToTarget`.
- `robot.PlanToConfiguration(joint_values)`: plans to a joint vector in
  manipulator joint order. `joint_values` should be array-like length `6` and is
  converted to `float64`.
- `robot.Retime(program)`: time-parameterizes `plan.results` and returns a
  `CompositeInstruction`, or `None` if retiming fails.
- `robot.ExecuteTrajectory(trajectory, stiffness=None, damping=None)`: streams a
  retimed trajectory through the controller callback. Optional stiffness/damping
  are six-element joint lists. Returns `False` if no execution callback was
  provided or the trajectory is invalid.

`GetActiveDOFValues()` returns the current manipulator joints in the same order used
by IK, planning, and execution. `SetActiveDOFValues(joint_values)` is only for
offline/simulated use, i.e. when `EnsemblRobot()` was created without
`get_observation`. `GetEnv()` returns the refreshed native Tesseract environment
for advanced scene/object updates.
