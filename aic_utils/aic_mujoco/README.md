# AIC MuJoCo Integration

This package provides documentation, scripts, and utilities for loading the AI for Industry Challenge (AIC) environment in MuJoCo.

## Overview

[MuJoCo](https://mujoco.org/) is a physics engine designed for research and development in robotics, biomechanics, graphics and animation. In collaboration with **Google DeepMind**, this integration enables participants to:

- Convert Gazebo SDF worlds to MuJoCo MJCF format using `sdformat_mjcf`
- Load the AIC task board and robot from exported Gazebo worlds (`/tmp/aic.sdf`)
- Access camera images, joint states, FT sensor data, and command the simulated robot over the same ROS topics
- Collect data and run policies unchanged between Gazebo and MuJoCo

This guide is split into two independent parts:

| | What | ROS 2 Control needed? |
|---|---|---|
| [**Part 1**](#part-1-building-the-mujoco-scene) | Generate the MJCF scene from Gazebo and view it in MuJoCo | No |
| [**Part 2**](#part-2-mujoco-with-ros-2-control) | Run the scene with `ros2_control` (same controller interface as Gazebo) | Yes |

## Import MuJoCo Dependencies

From your ROS 2 workspace, import all required MuJoCo repositories:

```bash
cd ~/ws_aic/src
vcs import < aic/aic_utils/aic_mujoco/mujoco.repos
```

This adds:
- `gz-mujoco` (with `sdformat_mjcf` tool) — Converts Gazebo SDF files to MuJoCo MJCF format
- `mujoco_vendor` (v0.0.6) — ROS 2 wrapper for MuJoCo 3.x with plugins (elasticity, actuator, sensor, SDF) and the `simulate` binary
- `mujoco_ros2_control` — Integration between MuJoCo and ros2_control

---

## Part 1: Building the MuJoCo Scene

This section covers generating and viewing the AIC scene in MuJoCo **without** requiring `ros2_control`. You only need the `sdformat_mjcf` converter and a MuJoCo viewer.

### Prerequisites

#### 1. Install `sdformat_mjcf` Python Bindings

The `sdf2mjcf` CLI tool requires Python bindings for SDFormat and Gazebo Math that are **not** resolved by `rosdep`. Install them from the OSRF Gazebo apt repository:

```bash
# Add the OSRF Gazebo stable apt repository (if not already added)
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update

# Install required Python bindings
sudo apt install -y python3-sdformat16 python3-gz-math9
```

Verify the bindings are importable:

```bash
python3 -c "import sdformat; print('sdformat OK')"
python3 -c "from gz.math import Vector3d; print('gz.math OK')"
```

#### 2. Build the Converter

Build the `sdformat_mjcf` package:

```bash
cd ~/ws_aic
source /opt/ros/kilted/setup.bash
colcon build --packages-select sdformat_mjcf
source install/setup.bash
```

### Scene Generation Workflow

#### 1. Export from Gazebo

- Launch `aic_gz_bringup` with your desired domain randomization parameters. For example:
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py spawn_task_board:=true spawn_cable:=true   cable_type:=sfp_sc_cable   attach_cable_to_gripper:=true   ground_truth:=true
```
- Gazebo will export the world to `/tmp/aic.sdf`.

See [Scene Description](../../docs/scene_description.md) for more details.

#### 2. Fix Exported SDF

The exported `/tmp/aic.sdf` contains two known URI corruption issues that must be fixed before conversion.

##### Issue 1: `<urdf-string>` in mesh URIs

When models are spawned from URDF strings (via `ros_gz_sim create -string`), the SDFormat parser uses the placeholder path `<urdf-string>` as the file source. On world export, this leaks into mesh URIs as `file://<urdf-string>/model://...`, which breaks XML parsing because `<urdf-string>` is interpreted as an XML tag.

```bash
# Fix corrupted model:// URIs
sed -i 's|file://<urdf-string>/model://|model://|g' /tmp/aic.sdf
```

##### Issue 2: Broken relative mesh URIs

Some included models (SC Plug, LC Plug, SFP Module) use relative mesh URIs (e.g., `<uri>sc_plug_visual.glb</uri>`). When the world is exported, these lose their model-relative context and become root-path URIs like `file:///sc_plug_visual.glb`, which point to nonexistent files.

```bash
# Fix broken mesh URIs by pointing to the actual files in aic_assets
sed -i 's|file:///lc_plug_visual.glb|model://LC Plug/lc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sc_plug_visual.glb|model://SC Plug/sc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sfp_module_visual.glb|model://SFP Module/sfp_module_visual.glb|g' /tmp/aic.sdf
```

> **Note:** These issues originate in the SDFormat library's handling of string-parsed URDFs and relative URIs during world save. They will occur every time you re-export the world from Gazebo.

#### 3. Convert SDF to MJCF

- Use the `sdf2mjcf` CLI tool to convert the fixed `/tmp/aic.sdf` to MJCF format:
  ```bash
  source ~/ws_aic/install/setup.bash
  mkdir -p ~/aic_mujoco_world
  sdf2mjcf /tmp/aic.sdf ~/aic_mujoco_world/aic_world.xml
  ```
- This generates MJCF XML file and mesh assets in `~/aic_mujoco_world`.

#### 4. Organize MJCF Files


- You **must always**  copy or symlink the generated mesh assets (`.obj` and `.png` files) from `~/aic_mujoco_world` into the `mjcf` folder so MuJoCo can find them.
  ```bash
  cp ~/aic_mujoco_world/* ~/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf
  ```

#### 5. Generate Final MJCF Files

The `sdformat_mjcf` converter produces a single monolithic MJCF file. The `add_cable_plugin.py` script splits and refines it into separate robot/world/scene files and applies corrections that the converter cannot handle automatically:

- **Splits into three files:** Separates the monolithic `aic_world.xml` into `aic_robot.xml` (robot bodies, actuators, sensors), `aic_world.xml` (environment, task board, cable), and `scene.xml` (top-level file that includes both).
- **Adds motor actuators:** Inserts position-controlled actuators for all 6 UR5e joints and the Robotiq gripper finger joints.
- **Adds gripper mimic joint:** Couples the right finger to the left finger via an equality constraint (removing the redundant right finger motor).
- **Adds FT sensor:** Attaches force and torque sensors to the `AtiForceTorqueSensor` site.
- **Adds `gripper_tcp` site:** Inserts a tool-center-point site at the gripper tip for policy use.
- **Fixes robot quaternions:** Normalizes near-identity and noisy quaternions on robot links (e.g., `shoulder_link`, `upper_arm_link`, `wrist_*_link`) to clean values.
- **Configures cameras:** Adds orientation (`quat`), field of view (`fovy`), and resolution to the center/left/right cameras.
- **Configures the cable plugin:** Activates `mujoco.elasticity.cable`, sets twist/bend stiffness, adds joint damping, and attaches the plugin to all cable bodies.
- **Reparents cable link_1:** Moves `link_1` from `cable_end_0` to `cable_connection_0` with a computed relative pose (required for correct cable attachment).
- **Tunes cable physics:** Reduces cable body inertias from `0.01` to `1e-6`, sets `cable_connection_1` (SC plug end) inertia to `4e-4`, adds damping to `joint_connection_end_0`, and lifts `cable_end_0` by 5cm.
- **Adds weld constraint:** Welds the LC plug to the gripper tool link (`ati/tool_link`) with a tuned relative pose.
- **Adds contact exclusions:** Prevents self-collision between `tabletop`↔`shoulder_link`, gripper fingers, `sc_port`↔`sc_plug`, and `cable_end_0`↔`link_1`.
- **Partitions assets:** Assigns meshes, materials, and textures to the correct file (robot vs. world) based on keyword matching.

Make sure you run this without sourcing the ROS 2 workspace in a new terminal (use a virtual env as necessary):

  ```bash
  cd ~/ws_aic/src/aic/aic_utils/aic_mujoco/
  python3 scripts/add_cable_plugin.py --input mjcf/aic_world.xml --output mjcf/aic_world.xml --robot_output mjcf/aic_robot.xml --scene_output mjcf/scene.xml
  cd ~/ws_aic && colcon build --packages-select aic_mujoco
  ```
  - `--input`: Path to the initial MJCF world file (usually `aic_world.xml`).
  - `--output`: Path for the final world file (`aic_world.xml`).
  - `--robot_output`: Path for the robot-only file (`aic_robot.xml`).
  - `--scene_output`: Path for the scene file (`scene.xml`).



#### 6. View in MuJoCo

At this point you can view the generated scene in MuJoCo **without** any ROS 2 control setup. Comment out the following lines from the `aic_world.xml` before running the viewer:

```
 <equality>
    <weld body1="ati/tool_link" body2="lc_plug_link" relpose="-0.000711 0.001759 0.168213 0.577301 0.816105 -0.021418 -0.015395" solref="0.002 1" solimp="0.99 0.999 0.001"/>
 </equality>
 ```


The weld constraint forces the robot end effector ati/tool_link to mate with the lc_plug_link. When we start the mujoco viewer as a standalone application, the ROS2 controller that enforces the initial condition of the robot is not present. The robot is spawned much farther away and you will observe flickering caused by the weld constraint. You should see both the wire and the robot collapse under gravity. ** Make sure you uncomment this line after testing this step. **


##### Using pixi environment

The Python viewer starts in **paused mode by default**. Press Space to start/pause simulation.

```bash
# Enter pixi shell
pixi shell

# Option 1: Launch empty viewer (then drag and drop scene.xml into the window)
python -m mujoco.viewer

# Option 2: Use the provided convenience script (starts paused)
cd ~/ws_aic
python src/aic/aic_utils/aic_mujoco/scripts/view_scene.py ~/aic_mujoco_world/scene.xml

# Option 3: Use a one-liner Python command (paused mode)
python -c "import mujoco, mujoco.viewer; m = mujoco.MjModel.from_xml_path('~/aic_mujoco_world/scene.xml'); d = mujoco.MjData(m); v = mujoco.viewer.launch_passive(m, d); v.sync(); exec('while v.is_running(): v.sync()')"
```

> **Tip:** Press Space in the viewer to start/pause simulation, Backspace to reset.

##### Using the `simulate` binary

> **Note:** The `simulate` binary is provided by `mujoco_vendor`, which is built in [Part 2](#part-2-mujoco-with-ros-2-control). If you have already completed Part 2, you can also use:

```bash
simulate ~/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf/scene.xml
```

---

## Part 2: MuJoCo with ROS 2 Control

![](../../../media/wave_arm_policy_mujoco.gif)

MuJoCo's integration with `ros2_control` allows you to control the UR5e robot using the same `aic_controller` interface as in Gazebo, ensuring your policy code remains simulator-agnostic.

### Installation Steps

> **Note:** If you already imported dependencies via `mujoco.repos` in the [Import MuJoCo Dependencies](#import-mujoco-dependencies) step above, the repositories are already cloned. Continue with the steps below to install and build.

#### 1. Install Dependencies

Install dependencies for the MuJoCo packages:

```bash
cd ~/ws_aic
rosdep install --from-paths src --ignore-src --rosdistro kilted -yr --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev"
```

#### 2. Build the Workspace

```bash
cd ~/ws_aic
source /opt/ros/kilted/setup.bash

# Build all packages (including aic_mujoco)
GZ_BUILD_FROM_SOURCE=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --merge-install --symlink-install --packages-ignore lerobot_robot_aic
```

#### 3. Verify Installation

```bash
# Source the workspace (if not already done)
source ~/ws_aic/install/setup.bash

# Check MUJOCO_DIR is automatically set by the environment hook
echo $MUJOCO_DIR
# Should output something like:
# /home/user/ws_aic/install/opt/mujoco_vendor

# Check MUJOCO_PLUGIN_PATH is set (this is how MuJoCo finds plugins)
echo $MUJOCO_PLUGIN_PATH
# Should output something like:
# /home/user/ws_aic/install/opt/mujoco_vendor/lib

# Check MuJoCo installation directory
ls $MUJOCO_DIR
# Should show: bin, include, lib, share, simulate directories

# Check that plugin libraries are installed
ls $MUJOCO_DIR/lib/*.so
# Should show: libelasticity.so, libactuator.so, libsensor.so, libsdf_plugin.so, libmujoco.so*

# Verify MuJoCo simulate binary works
which simulate
# Should output:
# /home/user/ws_aic/install/opt/mujoco_vendor/bin/simulate
```

> **⚠️ Important:** If you have a previous MuJoCo installation, it may conflict with `mujoco_vendor`. Check for and remove any existing `MUJOCO_PATH`, `MUJOCO_PLUGIN_PATH`, or `MUJOCO_DIR` environment variables from your shell configuration (`~/.bashrc`, `~/.zshrc`, etc.) before building. After cleaning the environment, restart your shell and rebuild the workspace:
> ```bash
> # Check for conflicting environment variables
> env | grep MUJOCO
>
> # If you see MUJOCO_PATH or MUJOCO_PLUGIN_PATH pointing to a different location,
> # remove those exports from ~/.bashrc (or ~/.zshrc) and restart shell
>
> # Then rebuild mujoco_vendor
> cd ~/ws_aic
> colcon build --packages-select mujoco_vendor --cmake-clean-cache
> source install/setup.bash
>
> # Verify the correct MUJOCO_PLUGIN_PATH is set
> echo $MUJOCO_PLUGIN_PATH
> # Should point to: /home/user/ws_aic/install/opt/mujoco_vendor/lib
> ```

### Launching MuJoCo with ros2_control

The `aic_mujoco_bringup.launch.py` launch file starts MuJoCo simulation with ros2_control, loading the same controllers as the Gazebo simulation.

#### Basic Launch Example

```bash
# terminal 1: Start the Zenoh router if not already running
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 run rmw_zenoh_cpp rmw_zenohd
```

```bash
# terminal 2: Launch MuJoCo simulation with ros2_control
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'
ros2 launch aic_mujoco aic_mujoco_bringup.launch.py
```

The robot can now be teleoperated using the `aic_teleoperation` package. See the [teleoperation](../../docs/teleoperation.md) section for details. For cartesian teleop use:

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp 
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
source ~/ws_aic/install/setup.bash
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

Any of the policies in `aic_example_policies` can be used to control the robot in MuJoCo. See the [example policies](../../docs/example_policies.md) section for details.

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [mujoco_ros2_control GitHub](https://github.com/ros-controls/mujoco_ros2_control)
- [AIC Getting Started Guide](../../docs/getting_started.md)
- [AIC Scene Description](../../docs/scene_description.md)

---

## AIC MuJoCo-Warp foundation

This package provides three deliberately separate capabilities:

1. the existing one-time conversion utilities that turn an exported AIC Gazebo
   world into MJCF and its mesh/texture assets; and
2. a small, ROS-free MuJoCo-Warp runtime for independently randomized SFP/NIC
   HOLD environments; and
3. a privileged Cartesian teacher that generates synthetic demonstration
   trajectories for later visuomotor-policy training.

The foundation runtime uses MuJoCo-Warp for every physics step, camera frame,
and F/T observation. Regular MuJoCo is used only on the host to compile and
validate MJCF before the model is uploaded. It does not call `mj_step`, create a
CPU `Renderer`, or download worlds for visualization.

The committed run is configured in [`configs/run.json`](configs/run.json). Run it
from the repository root without command-line arguments:

```bash
pixi run python aic_utils/aic_mujoco/run.py
```

The committed run requires a CUDA device. A missing CUDA device is an error;
there is no CPU fallback. It runs until `Ctrl+C`. Viser displays the reduced
scene's actual meshes at `http://127.0.0.1:8080`.

Synthetic demonstrations use the separate strict
[`configs/collect.json`](configs/collect.json) overlay. Run the complete
collector from the repository root with one command and no arguments:

```bash
pixi run python aic_utils/aic_mujoco/scripts/collect_data.py
```

The collector stops after the configured train, validation, and test counts are
complete. `Ctrl+C` safely closes active videos; rerunning the same command
removes only incomplete episode directories and resumes from completed
episodes. The configured environments are also published through the same
Viser scene used by the HOLD runtime, allowing trajectories to be inspected
while they are collected.

### Algorithmic design

#### Reduced SFP/NIC scene

[`scene.py`](aic_mujoco/scene.py) deterministically derives
`mjcf/scene_warp.xml` from the converted `aic_robot.xml` and `aic_world.xml`.
The source conversion outputs are not edited.

The reduced scene contains only:

- the six-joint UR5e;
- ATI wrist F/T site and force/torque sensors;
- the fixed Hand-E geometry;
- one SFP fixed directly to the tool;
- the task-board base and one target NIC fixture;
- the center, left, and right Basler cameras; and
- one AIC ceiling light.

It intentionally excludes the cable, LC plug, cable plugin, SC task, distractor
cards, enclosure, walls, and floor. The cable is irrelevant to this HOLD
foundation and is one of the model features that makes a Warp upload and large
batched simulation unnecessarily complicated.

The two finger joints and gripper actuator are removed from the reduced MJCF.
Before removal, the generator bakes the configured AIC SFP grasp position of
0.0073 m into each finger body transform. This is the `sfp_sc_cable` override
in `aic_bringup/launch/aic_gz_bringup.launch.py`, not the generic 0.00655 m
gripper value. The resulting closed finger geometry is fixed, and its collision
masks are disabled. The SFP subtree is rigidly attached to `ati/tool_link`
using the composed
tool-to-LC and LC-to-SFP transforms from the converted AIC scene. Consequently:

- the grasp needs no gripper controller;
- no finger/SFP internal contact can contaminate the wrist F/T reading; and
- SFP contact with the target NIC remains active.

Scene preparation then compiles the output and requires this exact contract:

```text
nq=6  nv=6  nu=6
ncam=3  nsensor=2  nsensordata=6
nmocap=2  nplugin=0
```

Any missing MJCF name, asset, sensor, camera, actuator, or unexpected model
dimension raises an error before MJWarp is created.

#### Independent environment reset

One compiled host model is uploaded once, then `N` device worlds are allocated.
Every world owns its own dynamic and reset state:

```text
shared MJWarp model
├── env 0: qpos, qvel, HOLD target, board/NIC pose, wrench tare, contacts
├── env 1: qpos, qvel, HOLD target, board/NIC pose, wrench tare, contacts
└── env N-1: independent copies of the same state
```

Randomness is derived from `(configured seed, environment ID, reset count)`.
Resetting environment 7 therefore does not alter environment 3 and does not
depend on the order in which other environments were reset.

The robot reset target is:

```text
q_hold[env] = AIC_HOME + uniform(joint_lower, joint_upper)
```

The AIC HOME values come from `aic_engine/config/eval_config.yaml`. AIC does not
publish an arm reset-noise standard, so `base.json` exposes the initial
prototype range explicitly as ±0.02 rad per joint. It is not hidden in code.

The board/NIC ranges are also explicit:

- board `x`: 0.16 m;
- board `y`: uniform from -0.21 to 0.05 m;
- board `z`: 1.14 m;
- board yaw: π plus a deviation in ±0.04159265 rad, wrapped at ±π;
- target NIC rail: rail 2 or rail 4;
- NIC rail translation: -0.0215 to 0.0234 m.

The board envelope and target rails span AIC evaluation SFP trials 1 and 2.
The NIC translation limits are the canonical `task_board_limits.nic_rail`
values in the AIC evaluation config. Sampling yaw uniformly from -3.1 to +3.1
would be wrong: it would include almost every orientation instead of the small
neighborhood around π represented by the two trials.

Both fixtures are MJCF mocap bodies. Their positions and orientations are
stored per world in `data.mocap_pos` and `data.mocap_quat`; the latter is an
unavoidable MuJoCo/MJWarp API-boundary representation. Controller commands,
controller errors, and stored dataset poses use rotation matrices. The shared
model's static body arrays are never mutated.

#### HOLD control

Each 2 ms step executes:

```text
mjw.step1
    computes current kinematics, velocity terms, sensors, and qfrc_bias
        ↓
Warp HOLD kernel
        ↓
mjw.step2
    actuation, acceleration, constraint solve, acceleration sensors, integration
```

For each world and each of the six arm joints, the Warp kernel computes:

```text
tau = kp * (q_hold - qpos) - kd * qvel + qfrc_bias
```

- `kp` is stiffness: restoring torque per unit position error.
- `kd` is damping: torque opposing joint velocity and oscillation.
- `qfrc_bias` is MuJoCo-Warp's current gravity, Coriolis, and centrifugal bias
  force for that DOF. It is a library-computed device tensor, not a constant or
  a separate controller.

The AIC engine HOME command supplies the configured stiffness
`[100,100,100,50,50,50]` and damping `[40,40,40,15,15,15]`. Torque is clipped
again in the kernel even though MJCF actuator control ranges are also bounded.
This prevents a large reset error or transient from writing an unbounded
control value. The current 120/60 N·m limits are an explicit prototype safety
envelope inherited from the branch control configuration; unlike HOME and the
gains, they are not claimed to be an AIC evaluation constant and remain
editable in `base.json`.

The important ordering is `step1 → controller → step2`. Computing control after
a complete `mjw.step` would use the previous step's `qfrc_bias`.

The control path is deliberately explicit:

```text
AICRobot
  └── ArmJoints: names → qpos/dof/actuator addresses
          ↓
HoldPositionCommand.position  (N, 6)
          ↓
JointHoldController Warp kernel
          ↓
MJWarp data.ctrl              (N, 6)
```

`AICRobot` is the simulation-side robot interface, not the Tesseract
`EnsemblRobot` planning facade. It validates the compiled MJCF and owns the
resolved arm, camera, wrench-sensor, and fixture interfaces. `ArmJoints` owns
only ordered joint/actuator bookkeeping. `HoldPositionCommand` is the actual
per-world command object. `JointHoldController` owns the gains, limits, device
address arrays, and the impedance kernel. `AICWarpRuntime` orchestrates those
objects and owns the changing MJWarp state.

#### Privileged Cartesian teacher

The data generator adds one generic Cartesian motion layer above the existing
HOLD controller. It does not teleport the robot and does not replace the
low-level torque law:

```text
exact SFP-tip pose + configured goal pose
                    ↓
       CartesianMoveController at 20 Hz
                    ↓
        bounded action_delta_q (N, 6)
                    ↓
       HoldPositionCommand.position
                    ↓
       JointHoldController at 500 Hz
                    ↓
             MJWarp physics
```

For every active environment, the teacher performs one closed-loop update:

1. read current and target world-frame `xmat` rotation matrices and calculate
   the three-dimensional SO(3) logarithm (axis multiplied by angle in radians);
2. clip each error along its original direction to the configured Cartesian
   step;
3. build the six-joint geometric Jacobian from MJWarp's current `xanchor` and
   `xaxis` tensors; and
4. solve the damped least-squares differential IK equation
   `dq = Jᵀ (J Jᵀ + λ² I)⁻¹ dx` in a Warp kernel.

Euler angles are not used because they have singularities and multiple
representations for the same orientation. Quaternions are not used by the
teacher or dataset because MuJoCo and Warp expose different component-order
conventions at relevant API boundaries. A 3×3 SO(3) matrix is already available
as MJWarp `xmat`, composes directly, and has no sign ambiguity. The controller
reduces matrix error to the standard three-dimensional tangent-space vector
needed by differential IK; it does not regress a nine-dimensional action.

Each joint increment is clipped by `expert.maximum_joint_step`, and the
resulting joint target is kept inside the compiled MJCF limit minus the explicit
margin. The six-joint impedance controller tracks that target through normal
physics. Recomputing this action from every new observation produces a full
closed-loop correction trajectory; recording only one final joint target would
not.

The controlled and goal bodies are names, not hard-coded IDs. The committed
task moves `sfp_tip_link` to a pre-insertion pose relative to
`sfp_port_0_link_entrance`. In the converted AIC model the entrance frame's
local +Z axis points inward toward the NIC body. Therefore the configured
`[0, 0, -0.005]` m offset is 5 mm outside the entrance. Changing its sign moves
the target inside the connector and is an insertion target, not a pre-insertion
target.

Success requires both configured position and orientation tolerances for five
consecutive 20 Hz samples. A 200-sample limit bounds every attempt. Failed
attempts are kept for diagnosis and resampled until the requested successful
trajectory counts are reached, subject to the explicit failure limit.

#### RGB observations

The AIC cameras are RGB-only at their native 1152 × 1024 resolution and 20 Hz.
At the 500 Hz physics rate, one camera update occurs every 25 physics steps.
MJWarp internally stores packed pixels in flat per-camera regions. The runtime
unpacks them in a Warp kernel and exposes named device tensors:

```text
rgb.center  (N, 1024, 1152, 3) uint8
rgb.left    (N, 1024, 1152, 3) uint8
rgb.right   (N, 1024, 1152, 3) uint8
```

There is no depth allocation and the runtime performs no implicit resize. The
dataset writer explicitly downsamples the native device tensors to the
`dataset.image_width` and `dataset.image_height` values before host transfer.
The committed 288 × 256 size preserves the native 9:8 aspect ratio while
reducing video encoding and storage cost. It is a dataset choice, not a change
to the simulated AIC cameras.

Native RGB is expensive: three cameras produce 3,538,944 rays per environment
per frame. Choose `runtime.num_envs` against actual GPU memory and measured
throughput rather than assuming the low-dimensional physics batch size will
also be a sensible image batch size.

#### F/T observations and taring

The reduced MJCF preserves the standard MuJoCo force and torque sensors:

```xml
<force name="AtiForceTorqueSensor_force" site="AtiForceTorqueSensor"/>
<torque name="AtiForceTorqueSensor_torque" site="AtiForceTorqueSensor"/>
```

MJWarp exposes one `sensordata` row per environment. The generator/runtime
validate the addresses and dimensions instead of assuming them. In this scene:

```text
sensordata       (N, 6)
force address    0, dimensions 0:3
torque address   3, dimensions 3:6
```

The public observations are:

```text
wrench.raw        (N, 6) float32
wrench.tared      (N, 6) float32
wrench.tare_ready (N,)   bool
```

For each reset environment independently, taring:

1. holds the robot while `tare_settle_steps` elapse;
2. samples its raw wrench at 100 Hz;
3. averages `tare_sample_count` samples on the device; and
4. subtracts that baseline from subsequent readings.

This is asynchronous. A reset environment can settle and retare while all
other environments continue stepping. Raw wrench is sampled every 5 physics
steps (100 Hz); the 50 Hz publication clock fires every 10 steps for future
evaluation-interface integration. Each environment's simulation time is reset
when its tare completes, and its independent `episode_steps` counter advances
only while `tare_ready` is true.

### Software design

#### Engineering principles

The complete simulation foundation follows four project-level principles:

- simulation behavior is explicit, configuration-driven, and validated before
  execution;
- scene construction, physics/control, observations, and data workflows have
  separate responsibilities and small interfaces;
- MuJoCo and MJWarp provide the canonical physics, kinematics, rendering, and
  sensor primitives instead of parallel application-level implementations; and
- shared infrastructure remains reusable so controllers, data generators, and
  future learned policies can evolve without rewriting the runtime.

These principles apply to the full active simulation code developed across the
foundation and policy branches.

#### Minimal file layout

```text
aic_mujoco/
├── aic_mujoco/
│   ├── collection.py   asynchronous teacher/data orchestration
│   ├── commands.py     HOLD, Cartesian-pose, and joint-delta objects
│   ├── config.py       strict JSON merge and validation
│   ├── controllers.py  Warp impedance and Cartesian DLS controllers
│   ├── dataset.py      atomic episode/video storage and resume logic
│   ├── joints.py       named joint/actuator address mapping
│   ├── outputs.py      selected-world Viser geometry bridge
│   ├── robot.py        validated AIC simulation robot interface
│   ├── runtime.py      reset, MJWarp physics, RGB, and F/T tensors
│   ├── scene.py        deterministic reduced-MJCF generator
│   └── utils/          reusable array, image, and numerical operations
├── run.py              configured continuous HOLD entry point
├── configs/
│   ├── base.json    stable scene/task/control configuration
│   ├── run.json     HOLD execution/device/output overlay
│   └── collect.json collection/expert/dataset overlay
├── mjcf/
│   ├── aic_robot.xml
│   ├── aic_world.xml
│   └── scene_warp.xml
├── scripts/             executable data and conversion workflows
└── test/                simulation and data-pipeline verification
```

There are exactly two control layers because both now have consumers:
`JointHoldController` keeps the robot alive and tracks joint targets, while
`CartesianMoveController` is the privileged demonstration teacher. There are no
unused velocity commands, trajectory planner, Tesseract `EnsemblRobot`, MJLab,
reward, or learned-policy layers. A future policy can replace the teacher by
writing the same `JointDeltaAction` without changing physics or impedance
control.

#### Base and execution overlays

Every executable still loads exactly two complementary files. `base.json`
contains stable scene, physics, control, randomization, sensor, and camera
values. The continuous HOLD executable deep-merges `base.json + run.json`; the
collector deep-merges `base.json + collect.json`. `run.json` supplies HOLD
device/output choices. `collect.json` supplies collection device choices, the
Cartesian expert, and dataset behavior. No executable combines both execution
overlays.

The loader performs a recursive deep merge. For example:

```jsonc
// base.json
{"physics": {"timestep": 0.002, "iterations": 200}}

// run.json
{"physics": {"device": "cuda:0", "graph_capture": true}}

// result
{"physics": {
  "timestep": 0.002,
  "iterations": 200,
  "device": "cuda:0",
  "graph_capture": true
}}
```

A shallow top-level merge would replace the entire `physics` object and lose
`timestep` and `iterations`. Deep merge preserves them while adding `device`
and `graph_capture`. Objects merge recursively; scalar values and arrays
replace a matching base value in full. Arrays are never combined element by
element because that would make six-joint configuration ambiguous.

There are no include graphs, environment-variable overrides, CLI overrides,
defaults, or fallback values. After the merge, validation rejects:

- missing or unknown keys;
- wrong scalar types or vector lengths;
- duplicate/missing MJCF names;
- invalid environment IDs;
- inverted/non-finite ranges;
- cadences that do not divide the physics rate; and
- CPU graph capture or an unavailable configured device.

This fail-fast contract keeps downstream code direct: it indexes known config
keys instead of carrying `.get(...)` defaults throughout the simulation.

Collection validation additionally rejects invalid body names at controller
construction, goal matrices outside SO(3), nonpositive motion bounds,
mismatched camera/action rates, odd video dimensions, invalid split counts, and
any existing dataset whose saved contract differs from `collect.json`.

#### Runtime ownership

`AICWarpRuntime` owns the shared device model, batched device data, observation
tensors, reset metadata, and captured execution graph. It composes the robot,
command, and controller objects rather than duplicating their responsibilities.
Its public surface is small:

```python
runtime = AICWarpRuntime(config)
events = runtime.step()
observations = runtime.observations()
reset_parameters = runtime.reset_state()
runtime.reset([3, 9])
runtime.reset([3, 9], randomization_ids=[1042, 1043])
```

`observations()` returns Warp arrays. A training framework can consume these on
device; it does not need to flatten cameras or split `sensordata` itself.
`reset_state()` exposes the sampled joint target, board/NIC transforms, rail,
and NIC translation for debugging and future episode logging.

Normal interactive resets derive randomness from `(seed, environment ID, reset
count)`. Dataset resets instead use `(seed, randomization ID)`. This makes an
episode's randomized scene reproducible even if asynchronous scheduling assigns
it to a different environment after a restart.

On CUDA, the physics/HOLD/tare sequence is captured as a Warp graph when
`physics.graph_capture` is true. Camera rendering remains on its independent
20 Hz clock. Contact and constraint capacities are explicit config values and
are checked after initial settling; they are not silently guessed.

#### MJWarp boundary and known constraints

The implementation is intentionally designed around the supported subset in
the pinned MuJoCo-Warp 3.5.0 stack:

- MuJoCo still parses/compiles MJCF on the host; MJWarp accepts the resulting
  model and owns all application physics stepping.
- Cable plugins are not uploaded. The reduced model has `nplugin=0` and no
  flexible cable topology.
- Per-world fixture randomization uses `mocap_pos`/`mocap_quat`; changing the
  shared model's `body_pos` would move the fixture identically in every world.
- `nconmax` and `njmax` are explicit allocations because MJWarp cannot grow
  contact/constraint buffers transparently during a captured training loop.
- Device physics is float32 and GPU contact reductions may not be bitwise
  deterministic. Seeded reset samples are reproducible; a full trajectory is
  not promised to be bit-identical across GPU architectures.
- MJWarp's native ray renderer supplies the policy RGB tensors. Its appearance
  should not be assumed pixel-identical to Gazebo or MuJoCo's OpenGL viewer.

These are deliberate boundaries of this foundation, not silent fallbacks.

#### Dataset contract

The collector records observations and the teacher action before that action is
tracked over the next 50 ms. Every successful episode is published atomically:

```text
data/far_approach/
├── dataset.json              exact generation contract
├── collection_state.json    next randomization ID and failure count
├── manifest.jsonl            successful episode index
├── train/
│   └── episode_000000/
│       ├── center.mp4
│       ├── left.mp4
│       ├── right.mp4
│       ├── trajectory.npz
│       └── episode.json
├── validation/
├── test/
└── failures/
```

`trajectory.npz` contains aligned arrays for ordered `qpos`, `qvel`, bounded
`action_delta_q`, SFP-tip pose, ground-truth goal pose, position/orientation
error, and tared wrench. Pose orientations are explicit `(T, 3, 3)` SO(3)
rotation matrices named `sfp_tip_rotation_matrix` and
`goal_rotation_matrix`; no quaternion convention leaks into the dataset.
Language and reset/randomization metadata live in `episode.json`. Videos are
H.264 rather than loose PNG files.

The committed split requests 1,000 training, 100 validation, and 200 test
trajectories across 16 asynchronous worlds. Finishing one environment resets
and re-tares only that environment; all others continue. These are distinct
held-out samples from the configured randomization distribution, not a claim of
a held-out real-world or out-of-distribution test set.

The local `data`, `runs`, and `checkpoints` directories are ignored by Git.
Source, configs, dataset contracts, metrics, and intentionally selected report
figures can still be tracked separately.

#### Visualization

The Viser path sends the compiled visual meshes once, then downloads only the
selected MJWarp `geom_xpos` and `geom_xmat` poses at the camera cadence. It does
not download or encode camera images. It never calls regular MuJoCo physics or
`mujoco.Renderer`.

`visualization.env_ids` explicitly selects the displayed worlds. The committed
value `"all"` expands to every environment ID from `0` through `N-1`; an
explicit integer list remains supported for selected-world debugging. Multiple
displayed worlds are arranged row-by-row using the configured `grid_columns`
and `[x, y]` `grid_spacing`. This affects only the human view, not physics.
Initial viewer position/look-at and real-time pacing are explicit in the
configuration.

The full AIC cell, enclosure, walls, and floor are intentionally absent because
they are absent from the reduced simulation scene. The 3D viewer shows the
actual foundation contents: tabletop/robot, fixed gripper and SFP, randomized
board/NIC, and camera hardware geometry. It does not invent display-only cell
geometry that the policy's physics world does not contain or project images
into the 3D scene.
