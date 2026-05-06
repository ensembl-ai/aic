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
current_base_tcp = robot.ComputeFK() # get's latest joint angles from observations
box_name = robot.AddBoxKinbody(
    dim=[0.1, 0.1, 0.05],
    transform=current_base_tcp,
    body_name="tcp_clearance_box",  # omit for a UUID-generated unique name
)
# target_base_tcp: gripper/tcp transform in base_frame, meters, shape (4,4)
target_base_tcp = np.eye(4, dtype=np.float64)
ik_solutions = robot.ComputeIK(target_base_tcp, return_all=True) # If return_all is false, it provides closest IK
current_in_collision, contacts = robot.CheckCollision(report=True) # report false is binary check without insights
plan = robot.PlanToTarget(target_base_tcp)
if plan:
    trajectory = robot.Retime(plan.results)
    if trajectory:
        robot.ExecuteTrajectory(trajectory)  # Requires execute_joint_motion callback.
```

Frame arguments are only needed when you want something other than the configured
manipulator frames. `target_frame=None` or omitted means `robot.manipulator_tip_frame`
(`"gripper/tcp"` today). `base_frame=None` or omitted means
`robot.manipulator_base_frame` (`"base_link"` today). For FK, the returned matrix is
`T_base_frame_target_frame`.

For IK, the input transform is interpreted as the desired
`target_frame` pose expressed in `base_frame`. Both frames must be links on the
configured manipulator chain, otherwise the call raises `ValueError`.

- `robot.ComputeFK(joint_values=None, target_frame=None, base_frame=None) -> np.ndarray`:
  returns a `(4, 4)` `float64` homogeneous transform. With `joint_values=None`,
  it uses the current observed manipulator joints. With `joint_values` supplied,
  it computes FK for that explicit manipulator joint vector.
- `robot.ComputeIK(transform, target_frame=None, base_frame=None, return_all=False, check_collision=True)`:
  `transform` must be a `(4, 4)` array-like homogeneous matrix and is converted
  to `np.float64`. Returns the closest collision-free joint vector as shape
  `(6,)`, all solutions as shape `(N, 6)` when `return_all=True`, or `None` if
  no solution is found. Set `check_collision=False` to keep IK solutions that are
  kinematically valid but may collide.
- `robot.CheckCollision(report=False)`: checks the current robot/environment
  state. Returns `True`/`False`, or `(in_collision, contacts)` when
  `report=True`; each contact includes the link pair, signed distance, contact
  type ids, and whether it is a single contact point.
- `robot.AddBoxKinbody(dim, transform, parent_frame="base_link", collision_enabled=True, body_name=None)`:
  adds a fixed box collision body to the native Tesseract environment. `dim` is
  `[x, y, z]` in meters. `transform` is a `(4, 4)` homogeneous matrix for the
  box center expressed in `parent_frame`; the box axes follow the transform
  rotation. `body_name` is the Tesseract link name to create; omit it to use a
  UUID-generated unique name like `box_kinbody_<uuid>`. Returns the link name.
  The collision manager is refreshed immediately, so the box is included in
  `CheckCollision`, collision-filtered IK, and subsequent planning calls.
- `robot.PlanToTarget(transform, max_joint_delta=float("inf"))`: plans from the
  current state to the default TCP pose in the default base frame. The transform
  has the same `(4, 4)` convention as default IK. `max_joint_delta` rejects IK
  goals whose largest per-joint move from the current state is too large. Returns
  a Tesseract `PlannerResponse` or `None`; use `plan.results` as the program.
  For non-default frames, call `ComputeIK(..., target_frame=..., base_frame=...)`
  first, then pass the returned joint vector to `PlanToConfiguration`.
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


## Isaac Sim LeRobot Collection

These commands are for a fresh `aic_eval`/Runpod container after the AIC repo is
available at `/app/ws_aic/src/aic`. The flow is intentionally split: Pixi
installs the normal dependencies, then pip installs only the Isaac Sim wheel
layer with `--no-deps` so the AIC Pixi-locked Gazebo and LeRobot stack stays
intact.

Check that the container can see the GPU and that Pixi is on `PATH`:

```bash
nvidia-smi
command -v pixi
```

Install the repo environment, then install the Isaac Sim packages from the
single Isaac requirements file. Do not omit `--no-deps`; allowing dependency
resolution here can replace Torch, CUDA, Numpy, or LeRobot versions from
`pixi.lock`. The expected Isaac-side stack is Isaac Sim `6.0.0`, IsaacLab
`v3.0.0-beta`, Python `3.12`, repo-locked Torch `2.7.1`, Torchvision `0.22.1`,
and LeRobot `0.5.1`.

```bash
cd /app/ws_aic/src/aic
pixi install --locked
pixi run pip install --no-deps --upgrade --no-cache-dir --extra-index-url https://pypi.nvidia.com \
  -r aic_utils/aic_isaac/aic_isaaclab/requirements-isaac.txt
pixi run pip install -e aic_utils/aic_isaac/aic_isaaclab/source/aic_task

pixi run python -c "import isaacsim; print('isaacsim ok')"
pixi run python -c "import aic_task; print('aic_task ok')"
```

Because the Isaac wheel layer is installed with `--no-deps`, the requirements
file also explicitly lists small runtime wheels that Isaac Sim/IsaacLab imports
before an environment can be constructed, currently `scipy`, `h5py`, and
`trimesh`. It also lists the Isaac GUI/editor wheels used by noVNC/X11
`--viz kit` runs. Pixi tracks ordinary Python helpers that Isaac/Kit may import
from optional startup paths, including `coverage` and `botocore` for
video/camera playback, `moviepy` for Gymnasium video recording, and `osqp` for
the optional Isaac wheeled-robot GUI extension.
If a shell was already set up before the Isaac-side wheels were added, install
the updated file once:

```bash
cd /app/ws_aic/src/aic
pixi run python -m pip install --no-deps --upgrade --no-cache-dir --extra-index-url https://pypi.nvidia.com \
  -r aic_utils/aic_isaac/aic_isaaclab/requirements-isaac.txt
```

Check out IsaacLab next to the AIC repo:

```bash
cd /app/ws_aic/src
git clone --branch v3.0.0-beta --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git IsaacLab
```

If `IsaacLab` already exists, put it on the expected beta tag instead:

```bash
cd /app/ws_aic/src/IsaacLab
git fetch origin v3.0.0-beta
git checkout v3.0.0-beta
```

The IsaacLab checkout is source code, not automatically importable just because
it exists under `/app/ws_aic/src`. The repo activation script
`pixi_env_setup.sh` adds these paths when entering a fresh `pixi shell`. If the
shell was already open, either exit and re-enter `pixi shell`, or source the
activation script once:

```bash
cd /app/ws_aic/src/aic
source pixi_env_setup.sh
```

Manual equivalent, useful when debugging outside `pixi shell`:

```bash
export ISAACLAB_ROOT=/app/ws_aic/src/IsaacLab
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_tasks:${ISAACLAB_ROOT}/source/isaaclab_rl:${ISAACLAB_ROOT}/source/isaaclab_visualizers:/app/ws_aic/src/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task:${PYTHONPATH}"
```

The task USDs are expected in the gitignored directory below.

```text
aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets/
```

Download the assets:

```bash
cd /app/ws_aic/src/aic
ASSET_PARENT="aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task"
AIC_ISAAC_CACHE_ROOT="/app/ws_aic/src/.cache/aic_isaac"
mkdir -p "$AIC_ISAAC_CACHE_ROOT" "$ASSET_PARENT"
curl -L --fail \
  -o "$AIC_ISAAC_CACHE_ROOT/Intrinsic_assets.zip" \
  "https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip"
rm -rf "$ASSET_PARENT/Intrinsic_assets"
unzip -q "$AIC_ISAAC_CACHE_ROOT/Intrinsic_assets.zip" -d "$ASSET_PARENT"
```

## Isaac insertion smoke tests

Run these before full-scale PPO. They are ordered from cheapest to most useful.
Use a fresh `pixi shell` so the IsaacLab `PYTHONPATH` additions from
`pixi_env_setup.sh` are active.

```bash
cd /app/ws_aic/src/aic
pixi shell
```

Inside `pixi shell`, run the commands below as written. From a fresh host shell,
prefix each `python` command with `pixi run`.

Basic imports:

```bash
python -c "import isaacsim; print('isaacsim ok')"
python -c "import isaaclab; import isaaclab_tasks; import isaaclab_rl; import aic_task; print('isaaclab/aic_task ok')"
```

Direct Gym registration check without launching Kit:

```bash
python -c "import gymnasium as gym; import aic_task.tasks; print([spec.id for spec in gym.registry.values() if 'AIC-' in spec.id])"
```

Expected output includes:

```text
['AIC-Task-v0', 'AIC-Insertion-v0']
```

IsaacLab/Kit environment listing:

```bash
python aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py --keyword AIC-Insertion
```

Zero-action reset/step smoke. Exit code `124` from `timeout` is expected here:
it means the environment kept stepping until the timeout without a traceback.
`AIC-Insertion-v0` disables inherited wrist cameras, so no `--enable_cameras`
flag is required.

```bash
timeout 30s python aic_utils/aic_isaac/aic_isaaclab/scripts/zero_agent.py \
  --task AIC-Insertion-v0 \
  --num_envs 1 \
  --device cuda:0
```

Random-action action/IK/contact smoke. Exit code `124` from `timeout` has the
same meaning here.

```bash
timeout 30s python aic_utils/aic_isaac/aic_isaaclab/scripts/random_agent.py \
  --task AIC-Insertion-v0 \
  --num_envs 4 \
  --device cuda:0
```

One-iteration PPO plumbing check:

```bash
python aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Insertion-v0 \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

GUI/noVNC visual smoke. Use this only when the container has an X server/noVNC
display. On this Runpod setup the display is `:1`; use the active display if it
differs. `--viz kit` selects the Kit GUI/visualizer path. This works because
`pixi_env_setup.sh` adds the IsaacLab source checkout, including
`source/isaaclab_visualizers`, to `PYTHONPATH`. Do not add `--headless`.

```bash
DISPLAY=:1 HEADLESS=0 LIVESTREAM=0 pixi run python aic_utils/aic_isaac/aic_isaaclab/scripts/zero_agent.py \
  --task AIC-Insertion-v0 \
  --num_envs 1 \
  --device cuda:0 \
  --viz kit \
  --real-time \
  --disable_fabric
```

If the viewport opens but looks empty or black, wait until the log prints
`[INFO]: Completed setting up the environment...`, then select
`/World/envs/env_0/Robot` or `/World/envs/env_0/task_board` in the Stage tree and
press `F` to frame it. Keep `--disable_fabric` for one-env visual debugging; it
is slower, but makes the USD stage easier to inspect. The visual smoke script
also accepts explicit viewport coordinates:

```bash
DISPLAY=:1 HEADLESS=0 LIVESTREAM=0 pixi run python aic_utils/aic_isaac/aic_isaaclab/scripts/zero_agent.py \
  --task AIC-Insertion-v0 \
  --num_envs 1 \
  --device cuda:0 \
  --viz kit \
  --real-time \
  --disable_fabric \
  --viewer-eye 0.85 -0.55 0.55 \
  --viewer-target 0.28 0.20 0.04
```

To visualize a saved policy checkpoint in the GUI:

```bash
DISPLAY=:1 HEADLESS=0 LIVESTREAM=0 pixi run python aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
  --task AIC-Insertion-v0 \
  --num_envs 1 \
  --device cuda:0 \
  --checkpoint logs/rsl_rl/aic_insertion/<run-name>/model_<iter>.pt \
  --viz kit \
  --real-time
```

To watch PPO collection/training step the environment in the GUI, run a tiny
visual-only training job. `train.py` does not accept `--disable_fabric` or
`--real-time`; use those only with the smoke/play scripts that define them.

```bash
DISPLAY=:1 HEADLESS=0 LIVESTREAM=0 pixi run python aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Insertion-v0 \
  --num_envs 1 \
  --max_iterations 10 \
  --device cuda:0 \
  --viz kit
```

The `--video` flag is separate from the live GUI. It records frames returned by
the sim environment render path and writes MP4 files under the run's `videos`
directory; it is not a noVNC desktop/screen capture.

The smoke-test pass criteria are:

- Imports succeed without `ModuleNotFoundError`.
- `AIC-Insertion-v0` appears in Gym registration.
- Zero-action run prints observation/action spaces and steps without reset,
  asset, observation, reward, or termination-manager exceptions.
- Random-action run steps without IK/action-shape/contact-manager exceptions.
- One PPO iteration reaches an update and writes a run under
  `logs/rsl_rl/aic_insertion/`.

## Insertion policy

The insertion-only policy lives under `aic_model/aic_model/insertion_policy`.
It deliberately does not use the task-board ground-truth transform path at
runtime. The expected runtime inputs are:

- `geometry_msgs/PoseStamped` on `/insertion_entrance_pose`, with
  `header.frame_id=base_link`. This is the noisy insertion-entrance estimate
  produced by the upstream perception/localization algorithm.
- `get_observation()`, specifically joint state, controller TCP pose/velocity,
  and wrist wrench.

The policy computes all target poses in `base_link`, validates the target TCP
through `EnsemblRobot.ComputeIK`, verifies the returned joint vector with
`EnsemblRobot.ComputeFK(joint_values=...)`, then streams `MotionUpdate` commands to
the Cartesian controller. Transforms use `transforms3d`; the policy code does
not implement custom quaternion math.

Run the controller policy:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_model.insertion_policy.InsertionPolicy \
  -p insertion_policy.config_path:=/app/ws_aic/src/aic/aic_model/config/noisy_entrance_insertion_policy.yaml
```

The config is strict: missing keys, empty frame/topic names, non-unit insertion
axes, invalid force guards, and invalid curriculum bounds fail at startup rather
than falling back to hidden defaults.

## Insertion RL training

The RL side is split into small pieces:

- `aic_model.insertion_policy.training.observation`: deployable actor
  observation encoder. Actor inputs are joint position/velocity, TCP pose
  relative to noisy entrance, TCP velocity, wrist wrench, and previous action.
  True port/board state is not encoded.
- `aic_model.insertion_policy.training.actor_critic`: asymmetric actor-critic
  MLP. The actor consumes deployable observations; the critic can consume
  simulator-only privileged state during PPO.
- `aic_utils/aic_isaac/.../aic_task/insertion_env_cfg.py`: Isaac Lab
  `AIC-Insertion-v0` task registration with insertion-specific observations,
  rewards, terminations, start-state curriculum, and domain randomization.
- `aic_utils/aic_isaac/.../agents/rsl_rl_insertion_ppo_cfg.py`: RSL-RL PPO
  config for the insertion task.

Training launch:

```bash
pixi run ros2 run aic_model aic_insertion_train -- \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 7 \
  --run_name aic_insertion_4096env \
  --headless
```

Isaac Sim must be usable non-interactively before that command can run. In this
container, `import isaacsim` currently stops at NVIDIA's Omniverse EULA prompt.
Accept it once in an interactive shell, then re-run the training command.

```bash
pixi run python -c "import isaacsim"
```

The training task resets the TCP near the true entrance using `EnsemblRobot` IK.
The sampled start offset radius begins at `0.005 m` and expands up to `0.05 m`.
The start orientation curriculum begins at `0 deg` and expands to `+-10 deg`
about each XYZ axis. The curriculum update mirrors the IndustReal pattern:

```text
if success_rate > 0.75:
  position_max += 0.005 m
  orientation_max += 1 deg
elif success_rate < 0.50:
  position_max -= 0.005 m
  orientation_max -= 1 deg
```

The actor is trained against noisy entrance estimates. The critic additionally
sees privileged true entrance error and estimator error, so PPO can learn a
cleaner value function without leaking privileged state into deployment.

PPO math used by RSL-RL:

```text
r_t(theta) = pi_theta(a_t | o_t) / pi_old(a_t | o_t)
L_policy = -mean(min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t))
L_value = mean((V(s_t) - R_t)^2), optionally clipped
L_total = L_policy + c_v L_value - c_e entropy
```

The insertion reward terms are:

```text
R = 2.0 * exp(-||xy_error||^2 / 0.006^2)
  + 1.0 * exp(-angle_error^2 / 0.08^2)
  + 2.5 * clamp(insertion_depth / 0.015, -1, 1)
  + 8.0 * success
  - 0.004 * ||tcp_force||
  - 0.005 * ||action_t - action_{t-1}||^2
  - 0.0002 * ||joint_velocity||^2
```

Success terminates an episode when lateral error is at most `0.0025 m`,
orientation error is at most `0.04 rad`, and insertion depth is at least
`0.012 m`. Force guard termination trips above `22 N`.

### Training time accounting

The `timeout 30s` smoke tests above use wall-clock time only. They are bounded
health checks for scripts that otherwise run forever; they are not training
horizons.

For training, count aggregate simulated time across all vectorized environments.
The insertion sim config uses `dt=1/240 s` and `decimation=2`, so each RL control
step advances:

```text
2 * (1 / 240) = 1 / 120 simulated seconds per environment
```

With `num_envs=4096`, one RL step represents:

```text
4096 * (1 / 120) = 34.13 aggregate simulated seconds
```

With `num_envs=4096`, `num_steps_per_env=32`, each PPO update collects:

```text
4096 * 32 = 131072 transitions/update
131072 / 8 minibatches = 16384 samples/minibatch
131072 / 120 = 1092.27 aggregate simulated seconds/update
= 18.20 aggregate simulated minutes/update
```

At `3000` PPO updates, that is:

```text
3000 * 1092.27 s = 3276800 aggregate simulated seconds
= 910.22 aggregate simulated hours
```

For a larger intuition check:

```text
1000000 RL steps at 4096 envs
= 1000000 * 4096 / 120 / 3600
= 9481.48 aggregate simulated hours
```

Wall-clock speed depends on PhysX/contact workload, rendering, GPU occupancy,
and reset cost. The nominal vectorized environment step capacity before
physics/contact overhead is:

```text
4096 envs * 120 control steps/s = 491520 env-steps/s
```

Measured on this machine's RTX 5090 with TF32 enabled, the standalone actor MLP
forward pass for batch `4096` ran:

```text
elapsed = 0.177490 s for 1000 forwards
mean forward = 0.1775 ms per 4096-observation batch
actor throughput = 23077356 observations/s
```

Kernel/performance choices made here:

- Actor and critic are plain fused-friendly MLPs with ELU activations and
  contiguous concatenated observations.
- Cameras are excluded from the insertion actor observation to keep the policy
  proprioceptive and fast.
- PPO uses large GPU batches: `4096` envs and `131072` transitions per update.
- TF32 is enabled for CUDA matmul and cuDNN in the training launch path.
- The simulator-side actor observation is one concatenated tensor; privileged
  critic terms are a separate observation group for asymmetric training.

No Isaac PPO loss curves or trained success-rate numbers were produced in this
pass because Isaac Sim stopped at the interactive EULA prompt before the
environment could start. The PyTorch actor path, package install path, and
non-Isaac tests were verified.
