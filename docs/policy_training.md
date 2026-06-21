# Policy Training Architecture Notes

This document collects the working mental model for moving from the AIC
Gazebo/ROS stack to a direct MuJoCo/MuJoCo-Warp policy-training stack.

The current direction is deliberately narrow:

```text
generated AIC MuJoCo scene
  -> local aic_mujoco reset/controller/observation/reward utilities
  -> direct MuJoCo Warp vectorized environments
  -> RSL-RL PPO training
```

ROS is not part of the training inner loop. Gazebo/ROS remain useful for
generating/evaluating scenes and understanding official AIC semantics, but
policy training should run directly against MuJoCo state, contacts, sensors,
and batched environment tensors.

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

ROS bridges this with `/clock` and `use_sim_time:=true`. A direct MuJoCo/Warp
training loop bridges it by never depending on wall-clock timers in the first
place.

For AIC specifically, the challenge docs warn that task time limits are based
on simulation time, not wall-clock time. A policy that uses `time.time()` or
wall-clock `sleep()` can behave differently when realtime factor changes. In
MuJoCo/Warp training, treat `model.opt.timestep` and the number of physics
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

## Delta Action To Robot Motion

The direct MuJoCo policy path uses a small Cartesian delta action, not direct
joint teleportation. The current intended chain is:

```text
policy action
  -> action clamp/scale
  -> desired TCP Cartesian delta
  -> differential IK
  -> joint target
  -> joint impedance torque
  -> MuJoCo actuator/generalized force
  -> mj_step physics
  -> updated qpos/qvel/geoms/sensors
  -> observation/reward/done
```

Concrete example:

```text
raw action       = [0, 0, -1]
action_scale     = 0.002 m
delta_world      = [0, 0, -0.002] m
```

That means the action layer asks the TCP to move 2 mm downward for this
environment action. It is a request, not a guaranteed displacement. The
achieved motion can be smaller because:

- differential IK caps the joint increment with `ik_max_dq`,
- joint targets are clipped to joint limits,
- impedance tracking applies finite torque rather than teleporting joints,
- contacts and constraints can block or deflect the motion,
- the controller may need multiple physics steps to track the new target.

The implementation then computes the TCP Jacobian and solves a damped least
squares update:

```text
dq = J.T @ inv(J @ J.T + damping^2 I) @ delta_world
```

The joint target is:

```text
q_des = q_current + capped(dq)
```

The impedance layer converts that target into torque:

```text
tau = kp * (q_des - q) + kd * (qd_des - qd) + optional_bias_compensation
```

Those torques are written into `data.ctrl` for actuator-backed joints, or into
`data.qfrc_applied` for direct generalized-force control. MuJoCo then advances
the actual physics:

```python
mujoco.mj_step(model, data)
```

After stepping, MuJoCo updates joint state, body poses, contacts, sensor data,
and geometry transforms. The viewer/Viser motion is just a rendering of this
updated `mjData`.

### Decimation

`decimation` means "hold one policy action for multiple physics/controller
substeps." With `decimation = 4`, the policy predicts one action, then the
environment applies the controller and advances physics four times before
asking the policy for the next action:

```text
policy action a_t
  substep 1: controller(a_t), mj_step
  substep 2: controller(a_t), mj_step
  substep 3: controller(a_t), mj_step
  substep 4: controller(a_t), mj_step
next policy action a_{t+1}
```

This is not "wait in wall-clock time." It is simulated control hold time:

```text
policy_dt = decimation * model.opt.timestep
```

If `model.opt.timestep = 0.002` and `decimation = 4`, the policy step
represents 8 ms of simulation time.

Decimation does not automatically mean the robot achieves the full requested
2 mm before the next policy action. It only gives the finite-torque controller
four physics steps to move toward the target. If `ik_max_dq`, torque limits, or
contacts restrict motion, the achieved TCP displacement may be much less than
the requested action-space delta.

### Wall-Clock Speed

Headless training should never sleep for visualization. A full episode should
run as fast as MuJoCo/Warp and the controller implementation allow.

If robot motion looks realtime, one of these is happening:

- a viewer/debug script is intentionally pacing frames with `sleep`,
- Viser/browser rendering is the bottleneck,
- the current prototype is looping through envs or geoms in Python,
- the controller/action code is still CPU/NumPy instead of fully batched device
  code,
- the script is rendering every step instead of sampling debug frames.

Realtime visualization is acceptable for inspection. It is a bug for headless
training throughput. Training metrics should separate:

```text
physics_steps_per_second
env_steps_per_second
aggregate_sim_seconds_per_wall_second
render_fps
```

For parallel training, aggregate sim seconds can exceed wall time even if each
single env is not running faster than realtime:

```text
aggregate_sim_seconds_per_wall_second =
  num_envs * policy_dt * env_steps_per_wall_second
```

### Episode Horizon And Happy-Path Length

The maximum episode length must be derived from the commanded motion scale and
the scene geometry, not guessed.

For insertion, compute the nominal travel distance from scene frames:

```text
preinsert point  = port entrance + preinsert offset
success point    = port bottom / calibrated inserted SFP-tip pose
travel_distance  = distance projected along insertion axis
```

Then estimate a happy-path lower bound:

```text
requested_step_distance = action_scale * max_action_along_axis
requested_progress_per_policy_step ~= requested_step_distance
minimum_policy_steps ~= ceil(travel_distance / requested_progress_per_policy_step)
```

But the achieved progress can be smaller than the requested progress, so the
episode horizon needs margin:

```text
horizon >= safety_factor * minimum_policy_steps
```

Use logs to measure:

```text
requested_delta_norm
achieved_tcp_delta_norm
achieved_insertion_progress_per_step
remaining_progress
steps_to_success_estimate
```

If the policy repeatedly exhausts a 250-step horizon before reaching the port
bottom on a straight-line happy path, the setup is wrong. Fix one or more of:

- `action_scale`,
- `ik_max_dq`,
- torque limits / stiffness / damping,
- `decimation`,
- episode horizon,
- success-frame calibration,
- contact geometry that blocks insertion.

The sanity test should include a scripted "straight down" policy and report
whether it can reach the nominal inserted frame within the configured horizon.

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
MuJoCo = physics, contacts, force/torque, task geometry
```

Start with 3D translation deltas and fixed orientation. Add orientation deltas
only when the insertion task needs them. Add cameras only after the
low-dimensional policy-training path is stable.

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

## Direct MuJoCo/Warp Role

The current decision is to host the first training prototype directly in
`aic_mujoco.warp`, not in MJLab. MJLab can be revisited later, but it is not
allowed to sit on the critical path while the basic task pipeline is still
being proven.

The existing local MuJoCo stack already contains the pieces that matter:

- frame utilities and pre-insertion IK target construction,
- joint grouping and passive joint freezing,
- impedance-like joint target tracking,
- reset-time force/torque zeroing,
- force/torque and contact observations,
- progress, force, penetration, and future SDF reward utilities,
- modular JSON config composition,
- direct MuJoCo Warp model/data smoke testing.

Those pieces should be lifted into a batched MuJoCo Warp env, not replaced
with a ROS controller process.

## Initial Training Shape

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
  use current local controller path:
    Cartesian delta target
    IK / differential IK
    joint target
    impedance-like tracking

physics
  decimation substeps in MuJoCo/MuJoCo Warp

observations
  q
  TCP pose
  relevant task-frame poses
  zeroed force/torque
  max contact penetration
  previous action
  no cameras for the first trainer

rewards
  insertion progress toward port bottom
  lateral alignment penalty
  zeroed force/torque safety penalty
  max penetration penalty
  action regularization
  SDF alignment later, once the basic vectorized pipeline works

terminations
  success
  excessive force
  excessive penetration
  invalid state/out-of-bounds
  timeout
```

Do not define reward math casually. Rewards and observations determine what the
policy learns, so they should be chosen after deciding the exact task and
available state.

## Minimal First Trainer

The first trainable system should be intentionally boring:

```text
algorithm: RSL-RL PPO
policy:    feedforward MLP actor-critic
action:    3D Cartesian delta
obs:       low-dimensional privileged state
runtime:   headless direct MuJoCo Warp
viewer:    separate play/debug path, not used for training throughput
```

Do not start with A3C, LSTM, images, or teacher-student distillation. Those
are all valid later, but they add extra axes of failure before the core
environment is proven.

Why PPO over A3C:

- PPO with synchronous vectorized rollouts is the common practical baseline for
  thousands of simulated environments.
- It maps cleanly to RSL-RL and batched simulator tensors.
- It is easier to debug because rollout collection and policy updates are
  explicit phases.
- A3C's asynchronous workers are less attractive when the simulator already
  provides vectorized stepping.

Why no LSTM first:

- The first observation is privileged and low-dimensional.
- The task should be Markov enough with pose, force/torque, penetration,
  progress, and previous action.
- Add recurrence only if partial observability becomes a real measured problem.

The first actor-critic can be:

```text
actor:
  obs_dim -> 256 -> 256 -> action_mean_dim
  learned log_std

critic:
  obs_dim -> 256 -> 256 -> value
```

Start with `num_envs = 32` to prove resets, stepping, reward, and termination.
Then scale upward:

```text
32 -> 128 -> 512 -> as high as MuJoCo Warp supports for this scene
```

The first success criterion is not insertion success. It is pipeline health:

```text
reset works for all envs
step accepts batched actions
observations are finite
rewards are finite
dones reset only the correct envs
RSL-RL can collect rollouts and update a policy
checkpoints can be played back in one viewer env
```

## Model Selection And Upgrade Path

Use one action interface from the start:

```text
policy action = Cartesian delta
```

That action should stay stable across:

- simple PPO teacher,
- later privileged teacher,
- later distilled student,
- later camera/perception student.

This matters because distillation becomes straightforward:

```text
teacher: privileged_obs -> Cartesian delta
student: deployable_obs  -> Cartesian delta
loss:    MSE(student_action, teacher_action)
```

The first PPO policy is effectively a privileged teacher, but we do not need
to call it teacher-student yet. First train the low-dimensional policy. Once it
can solve the simplified insertion task, record rollouts:

```text
privileged_obs
future_student_obs
teacher_action
reward
done
task/randomization metadata
```

Then a student can be introduced without changing the controller, action
space, reset logic, or reward definitions.

## Engineering Architecture

The MuJoCo/Warp stack should split into these layers:

```text
aic_mujoco core
  config.py
  utils.py
  joints.py
  controllers.py
  commands.py

aic_mujoco.mjlab utilities
  reset.py
  step.py
  observations.py
  rewards.py
  logging.py

aic_mujoco.warp direct prototype layer
  env.py
  warp_smoke.py
  rsl_rl_wrapper.py
  rsl_rl_cfg.py

script layer
  prepare_warp_scene.py
  train_warp_smoke.py
  train_rsl_rl_direct.py
  viz_warp_envs.py
```

The production training target keeps the batch in MuJoCo Warp device state,
not in Python loops over `mjData` objects. The inner loop is:

```text
action tensor
  -> controller/action term
  -> MuJoCo Warp physics
  -> tensor observations/rewards/dones
```

No ROS messages, no controller manager, no Zenoh, no `/clock`, no
`ros2_control` in the training process. Defaults are:

```text
backend = direct MuJoCo Warp
device  = CUDA
trainer = RSL-RL
```

Missing MuJoCo Warp, RSL-RL, or CUDA is a configuration error. The code
should fail directly rather than switching to another backend.

The RSL-RL wrapper should expose the expected vectorized interface:

```text
num_envs
num_obs
num_actions
episode_length_buf
reset()
step(actions)
get_observations()
```

The training script should be headless:

```text
scripts/train_warp_smoke.py
  loads env config
  runs direct MuJoCo Warp preflight
  runs prototype env smoke loop
  logs throughput and task metrics

scripts/train_rsl_rl_direct.py
  creates AicInsertionVecEnv
  wraps it as an RSL-RL VecEnv
  constructs MLP actor, MLP critic, PPO, rollout storage, and optimizer
  saves TensorBoard logs and checkpoints
```

The play/debug script should be separate:

```text
scripts/viz_warp_envs.py
  runs selected env copy with MuJoCo viewer
```

A Viser dashboard can be added later as a debug tool, but it should sample a
small subset of envs. It is not the training renderer.

## Visualization And Throughput

Viewer speed is not training speed. The current viewer demos deliberately sleep
or sync with a GUI. That is useful for seeing behavior, but it is the slow path.

Training should log:

```text
env_steps_per_sec
physics_steps_per_sec
sim_seconds_per_wall_second
rollout_time_ms
update_time_ms
reward_mean
episode_length_mean
success_rate
force_norm_max
max_penetration_mean
max_penetration_max
progress_mean
```

Visualization should log separately:

```text
viz_render_time_ms
viz_env_count
viewer_fps
```

Expected relationship:

```text
viewer/debug:
  one or a few envs
  slow, visual, human-paced

headless training:
  many envs
  no rendering
  no sleeps
  much higher aggregate step throughput
```

If a Viser view is added, it should show only high-signal state:

```text
sfp_tip frame
port_entrance frame
port_bottom frame
tcp frame
insertion axis
current tip-to-port vector
reward/progress/force/max-penetration statistics
```

Do not render all envs during serious training.

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

## Current Architecture Decision

Skip MJLab manager/entity composition for the first working prototype. The
direct path is:

```text
scene.xml / scene_warp.xml
  -> mujoco.MjModel
  -> reset/action/observation/reward modules
  -> MuJoCo Warp smoke preflight
  -> direct RSL-RL wrapper later
```

The reason is practical, not philosophical: the plain MuJoCo scene, reset, IK,
and controller path works, while the MJLab entity/spec attach layer introduced
native failures before any useful policy-training signal was available.

The active prototype code lives in:

```text
aic_mujoco/warp/env.py          direct vector env with the AIC task contract
aic_mujoco/warp/warp_smoke.py   direct MuJoCo Warp model/data smoke test
scripts/train_warp_smoke.py     headless prototype smoke and throughput stats
scripts/viz_warp_envs.py        MuJoCo viewer for one selected env copy
```

Two XMLs have different roles:

```text
scene.xml:
  semantic/debug scene with cable, plug, sensors, and viewer/controller behavior

scene_warp.xml:
  stripped Warp preflight scene with plug rigidly attached to the gripper
  because MuJoCo Warp does not support the cable body plugin
```

The prototype env currently uses one shared `MjModel` and multiple `MjData`
copies. That is not the final high-throughput implementation, but it preserves
the correct task semantics in a readable place. The next implementation step is
to replace the per-env `MjData` loop with batched MuJoCo Warp state while
keeping the same API:

```text
reset_idx(env_ids)
obs = get_observations()
obs, reward, done, extras = step(action)
```

## Recommended Development Sequence

1. Keep `hold_fixed_target.py` and `demo_joint_target_control.py` as the visual
   reference debuggers.
2. Keep the direct `aic_mujoco.warp` prototype env as the source of truth for
   reset/action/observation/reward semantics.
3. Use 3D Cartesian delta actions only.
4. Expose low-dimensional observations first: joint state, TCP/plug/port
   positions, zeroed force/torque, max penetration, previous action.
5. Add simple progress, lateral error, force, penetration, and action rewards.
6. Run `prepare_warp_scene.py` and `train_warp_smoke.py` before any PPO work.
7. Replace Python `MjData` loops with batched MuJoCo Warp data/state.
8. Wrap the direct env API for RSL-RL PPO.
9. Train headless with small `num_envs`.
10. Scale `num_envs` upward and measure throughput.
11. Add SDF reward after the basic pipeline works.
12. Add randomization after the policy can solve a fixed or lightly randomized
    scene.
13. Add teacher-student distillation only after a privileged PPO policy is
    competent.
14. Add cameras only after low-dimensional training is stable.

## Open Technical Risks

- MuJoCo Warp does not support the cable body plugin, so the Warp scene uses a
  rigidly held plug and no cable plugin.
- The current generated XML uses AIC assets converted from Gazebo/SDF. Asset
  paths and scene composition need to stay robust across scene regeneration.
- The local `aic_mujoco` controller is a training abstraction, not a
  bit-identical port of the AIC ROS controller. That is acceptable if the
  policy-facing action semantics remain compatible.
- Python loops over `N` separate `mjData` objects are fine for proving the
  pipeline, but true massive parallelism should move the same env API onto
  batched MuJoCo Warp data/state.
- Camera observations are expensive. They should be added after proprioception
  and force/torque training plumbing is stable.
- Contact-rich insertion rewards can be gamed. Force and max-penetration
  penalties must stay visible in logs from the beginning.
