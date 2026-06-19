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
  to be defined deliberately per task

terminations
  success
  excessive force/torque
  invalid state/out-of-bounds
  timeout
```

Do not define reward math casually. Rewards and observations determine what the
policy learns, so they should be chosen after deciding the exact task and
available state.

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
