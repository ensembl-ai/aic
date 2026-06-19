# Policy Training Architecture Notes

This document collects the working mental model for moving from the AIC
Gazebo/ROS stack to a MuJoCo/MJLab policy-training stack.

## MuJoCo Mental Model

MuJoCo has two important layers:

- `mjModel`: the compiled, mostly constant description of the world.
- `mjData`: the current dynamic state of one simulation instance.

The XML/MJCF describes the robot, joints, links, actuators, sensors, contacts,
assets, and task objects. After MuJoCo loads it, that structure becomes
`mjModel`.

The changing simulation state lives in `mjData`:

- `qpos`: joint positions and free-body poses.
- `qvel`: joint and free-body velocities.
- `ctrl`: actuator commands.
- `qfrc_applied`: externally applied generalized forces.
- `xpos`, `xquat`, `site_xpos`: derived world-frame poses after forward
  kinematics.
- `sensordata`: MuJoCo sensor outputs.
- contact buffers: active contact information.

Motion happens when code writes an action or control target and advances
physics:

```python
data.ctrl[:] = action
mujoco.mj_step(model, data)
```

The viewer is not the controller. A static viewer usually means the state or
control inputs are not changing in a meaningful way.

## Simulation Time Vs Realtime

Controllers should be written against simulation timestep `dt`, not wall-clock
time.

On a real robot, a controller might run every 1 ms of wall time:

```text
dt = 0.001 s
wall time also advances by about 0.001 s
```

In accelerated simulation, the controller still sees the same simulated `dt`:

```text
dt = 0.001 s
wall time may advance much faster or slower
```

Training should not sleep to match realtime. A training loop should advance
physics as fast as the backend allows:

```python
for _ in range(decimation):
    apply_controller()
    mj_step(model, data)
```

ROS bridges this with `/clock` and `use_sim_time:=true`. A direct MuJoCo/MJLab
training loop bridges it by never depending on wall-clock timers in the first
place.

For AIC specifically, the challenge docs warn that task time limits are based
on simulation time, not wall-clock time. A policy that uses `time.time()` or
wall-clock `sleep()` can behave differently when realtime factor changes. In
MuJoCo/MJLab training, treat `model.opt.timestep` and the number of physics
steps as the clock.

Practical rule:

```text
wall time: only for viewer/debug pacing
sim time: controller, reward, timeout, curriculum, logging metrics
```

In a vectorized RL loop, an "environment step" may contain multiple physics
substeps:

```text
policy action held for decimation steps
  for k in range(decimation):
    apply controller target
    mj_step(model, data)
sim_time += decimation * model.opt.timestep
```

This is the bridge between controller behavior and accelerated training. The
controller still sees the correct physics `dt`; the trainer does not need to
run in realtime.

## What AIC Encourages

The AIC policy API supports both:

- `MotionUpdate`: Cartesian-space commands.
- `JointMotionUpdate`: joint-space commands.

The AIC `Policy.set_pose_target(...)` helper creates a `MotionUpdate` for the
gripper TCP. The controller config also defaults to Cartesian target mode. That
means AIC's policy-facing layer encourages policies to command the end effector
in Cartesian space, while the lower controller handles kinematics and joint
actuation.

For the AIC insertion policy-training path, the preferred initial abstraction is:

```text
policy action = small Cartesian delta
controller/action layer = differential IK + joint target tracking
MuJoCo = physics, contacts, force/torque, cameras
```

Start with 3D translation deltas and fixed orientation. Add orientation deltas
only when the insertion task needs them.

## Why Not Use The ROS AIC Controller Directly?

The AIC controller is a ROS 2 `ros2_control` controller. That is appropriate for
Gazebo, the challenge stack, and single-simulation integration tests.

Heavy RL training in MJLab is different:

- many environments are stepped in parallel,
- actions and observations are batched tensors,
- the backend is MuJoCo Warp,
- there is no ROS middleware in the training inner loop.

The ROS controller is therefore not the right executable object for thousands
of parallel worlds. The correct move is to preserve its policy-facing semantics
and reimplement the relevant control behavior as MJLab action terms.

The current local `aic_mujoco` controller is not identical to AIC's controller.
It is a minimal joint impedance approximation:

```text
tau = Kp * (q_des - q) + Kd * (qd_des - qd) + tau_ff + bias
```

The full AIC controller includes Cartesian and joint target modes, KDL
kinematics, Jacobian mapping, gravity compensation, stiffness/damping smoothing,
feedforward wrench interpolation, wrench feedback gains, limits, nullspace
behavior, and tracking-error handling.

For policy training, we should reproduce only the behavior that matters for the
learning abstraction:

```text
Cartesian delta -> differential IK -> joint target
joint target tracking / impedance-like action application
zeroed force-torque observation
safety/contact termination
task-specific reset/randomization
task-specific rewards
```

## Impedance Vs Admittance

Impedance control maps motion error to force:

```text
desired motion - actual motion -> force/torque response
```

Admittance control maps measured force to motion:

```text
measured force/torque -> adjusted motion command
```

Assembly tasks often use impedance-like behavior because the robot should move
toward a target while remaining compliant under contact. It should resist enough
to insert, but not behave like an infinitely stiff position source.

Force/torque feedback can still be part of the policy:

- as an observation,
- as a safety termination,
- as a reward term,
- or later as an explicit admittance filter.

The first training version should keep the action clean:

```text
policy outputs Cartesian delta
controller tracks it compliantly
force/torque is observation and safety signal
```

An admittance layer can be added later if we want measured wrench to directly
modify the command before IK.

## Force/Torque Bias And Zeroing

Force/torque sensors should be treated as biased sensors. A nonzero wrench at
reset does not necessarily mean the policy caused contact; it may be gravity,
payload preload, cable tension, sensor bias, or controller settling.

The AIC docs expose the same concept through
`/aic_controller/tare_force_torque_sensor`. The controller docs recommend
taring before training episodes, and the interface docs state that evaluation
automatically calls the tare service before the cable is spawned. In direct
MuJoCo training, we reproduce this behavior ourselves:

```text
reset sim state
place robot and payload
hold reset pose for N physics steps
sample raw F/T for K physics steps
wrench_bias = mean(raw samples)
episode observation = raw_wrench - wrench_bias
```

Use the zeroed wrench for observations, safety penalties, and terminations:

```text
force_obs = raw_force - force_bias
torque_obs = raw_torque - torque_bias
```

Step-to-step wrench deltas can be useful in some controllers, but they should
not replace the zeroed absolute wrench. Sustained contact force is important in
insertion, and `current - previous` loses that information.

## MJLab Role

MJLab is not just abstract classes. It provides the manager-based RL framework
and MuJoCo Warp training backend:

- scene composition,
- action manager,
- observation manager,
- reward manager,
- event/reset/randomization manager,
- termination manager,
- metrics/curriculum/command managers,
- parallel GPU simulation through MuJoCo Warp,
- training and play entrypoints.

MJLab does not replace the AIC task logic. The AIC-specific pieces become MJLab
terms:

- AIC scene/assets: MJLab scene/entity configuration.
- Cartesian delta command: action term, preferably based on MJLab's
  `DifferentialIKAction` first.
- AIC-like joint target tracking: built-in joint position targets at first, or
  a custom action term if closer impedance behavior is needed.
- Force/torque zeroing: reset event plus observation term.
- Board/NIC/port/cable randomization: reset event terms.
- Success/failure logic: reward and termination terms.

## Initial MJLab Training Shape

The target training loop should look conceptually like this:

```text
reset envs
  randomize task board / NIC / port / cable state
  reset robot
  zero force/torque baseline

policy action
  Cartesian delta in TCP/task frame

action term
  scale/clamp delta
  differential IK to joint target
  apply joint target / impedance-like torque

physics
  decimation substeps in MuJoCo/MuJoCo Warp

observations
  q, qd
  TCP pose
  relevant task-frame poses
  zeroed force/torque
  previous action
  cameras later if needed

rewards
  SDF alignment to nominal inserted pose
  insertion progress toward port bottom
  zeroed force/torque safety penalty
  action/path regularization after behavior is stable

terminations
  success
  excessive force/torque
  invalid state/out-of-bounds
  timeout
```

Do not define reward math casually. Rewards and observations determine what the
policy learns, so they should be chosen after deciding the exact task and
available state.

## Nominal Insertion Pose And Stopping Condition

The local AIC scoring docs and implementation do not define success as "insert
15 cm." They define:

- Full insertion: correct plug inserted into the correct target port, verified
  by the Gazebo insertion/contact event.
- Partial insertion: plug is inside a bounding box between the port entrance
  frame and the port frame, within 5 mm XY tolerance, with score increasing as
  the plug approaches the port frame.
- Proximity: if not inside the port, score falls with final plug-port distance.

For SFP in the current assets, `sfp_port_0_link_entrance` is offset by
`-0.0458 m` from `sfp_port_0_link` in the port-local Z direction. So the
asset-derived entrance-to-bottom depth is about 45.8 mm. The baseline
`CheatCode` policy descends to `z_offset < -0.015`, which is a simple example
policy stopping depth, not the scoring definition and not 15 cm.

For MuJoCo rewards, the nominal inserted pose should therefore be derived from
scene frames:

```text
world_T_port_entrance = body_transform("sfp_port_0_link_entrance")
world_T_port_bottom   = body_transform("sfp_port_0_link")
desired_world_T_sfp_tip for full insertion is at/near world_T_port_bottom
desired_world_T_sfp_tip for pre-insertion is world_T_port_entrance plus an offset
```

The exact final SFP-tip frame may need a small calibrated offset if the AIC
plug frame is not exactly the physical leading surface, but it should be
calibrated from the generated scene/assets, not guessed as a fixed 15 cm.

## SDF-Based Geometry Reward

For dense insertion shaping, use an IndustReal-style sampled SDF query reward:

```text
offline/reset setup:
  sample ~1000 points on the SFP plug surface in plug-local coordinates
  construct a target SDF for the plug mesh at the nominal inserted pose

per physics/control step:
  transform sampled plug points by current world_T_plug
  query target SDF at those world points
  distance = RMS(abs(sdf_values))
  reward_sdf = -log(distance + eps)
```

This is different from "sum distances between all plug and port points." A
Chamfer-like point distance can be ambiguous near cavities and symmetries. SDF
query distance gives a smoother signal for "make the current plug occupancy
match the desired inserted plug occupancy."

Use two SDF-style signals for different purposes:

```text
target plug SDF:
  alignment reward to nominal inserted pose

port/socket solid SDF:
  collision/interpenetration penalty against socket walls
```

Then combine with simple task terms:

```text
reward =
  w_sdf      * (-log(rms_sdf + eps))
  + w_prog   * insertion_progress_along_port_axis
  - w_force  * max(0, ||zeroed_force|| - force_limit)
  - w_torque * max(0, ||zeroed_torque|| - torque_limit)
  - w_action * ||action||^2
```

The first implementation should keep weights explicit and modest. The SDF term
teaches geometric alignment; progress teaches direction into the port; zeroed
force/torque protects against learning "solve by pushing harder."

We should not reinvent geometry kernels. Use:

- `trimesh` for mesh loading, surface sampling, and proximity/SDF query.
- `scipy` for numerical/proximity support.
- `rtree` for trimesh spatial acceleration.

The current prototype reward utilities live in
`aic_mujoco.mjlab.rewards` and deliberately require the caller to provide the
mesh, sampled points, current transform, and nominal target transform. This
keeps task geometry explicit.

## Recommended Development Sequence

1. Load the generated AIC MuJoCo scene in a single MJLab environment.
2. Use zero and random agents to confirm action/observation plumbing.
3. Start with Cartesian translation deltas via `DifferentialIKAction`.
4. Add force/torque zeroing at reset and expose zeroed wrench observation.
5. Add task-board/NIC/port reset randomization.
6. Define rewards and terminations after the task state is agreed.
7. Scale `num_envs` upward.
8. Add cameras and cable complexity only after the rigid-body version is stable.

## Open Technical Risks

- MuJoCo Warp support for all plugins used by the generated scene, especially
  elastic cable plugins, must be verified.
- The current generated XML uses AIC assets converted from Gazebo/SDF. Asset
  paths and scene composition need to be made robust for MJLab.
- The AIC ROS controller and MJLab action terms will not be bit-identical unless
  we deliberately port the full controller behavior.
- Camera observations are expensive. They should be added after proprioception
  and force/torque training plumbing is stable.
