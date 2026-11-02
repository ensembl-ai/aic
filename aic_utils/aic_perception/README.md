# AIC Perception: Multiview Scene Estimation

Estimates the AIC task board and every named port (`nic_card_mount_K/sfp_port_J`, `sc_port_K`) as poses in the robot `base_link` frame from the three wrist cameras. A task planner can then select a port by name and receive its pose, presence, and confidence.

## Algorithm

```text
left / center / right RGB ──► shared ResNet-18 ──► feature tokens per camera
camera intrinsics + base_from_camera (TF) ──► per-token ray embedding (origin, direction)
                     all tokens ──► transformer encoder (cross-view fusion)
     one learned query per schema target ──► transformer decoder
       per target: presence, translation (base frame), 6D rotation, confidence
```

- **Schema-driven output.** `configs/base.json` defines a versioned board schema (`aic_v1`): an ordered list of target IDs and their ground-truth frames. Query *i* always means the *i*-th target ID. Planners see IDs, never indices. A changed board means a new schema version and a fine-tune.
- **Calibration is an input.** Every image token carries its base-frame viewing ray, computed from the camera matrix and the camera pose from TF. Calibration changes appear as input changes rather than being baked into the weights.
- **Supervision.** Presence is trained on every target; it means spawned and inside at least one camera's view. Pose (L1 translation, chordal rotation) and confidence are trained on present targets. The confidence target is whether the current pose error is within the configured tolerance.
- **Data.** Privileged synthetic labels from Gazebo. Each scene is a randomized task board observed from HOME: board pose, which NIC cards, SC ports, and fixtures are present, rail offsets, and yaw offsets. Ground truth comes from the board's pose publisher, the same source AIC scoring uses.
- **Scope of version 1.** One observation pose (HOME), no cable in the gripper, exact calibration, no active perception.

## Software

| Stage | Script | Runs in | Produces |
|---|---|---|---|
| Collect | `scripts/collect_data.py` | host; one `aic_eval` container per worker | `train/` and `val/` images and labels |
| Curate | `scripts/curate_dataset.py` | pixi | `manifest.json`, `previews/` |
| Train | `scripts/train.py` | pixi, GPU | `runs/<name>/checkpoints/{best,last}.pt`, W&B run |
| Evaluate | `scripts/evaluate.py` | pixi, GPU | `runs/<name>/evaluation.json` |
| Infer | `scripts/run_inference.py` | pixi, next to a running AIC sim | `/aic_perception/scene_estimate`, TF `aic_perception/<id>` |

- **Configuration only.** Every script reads `configs/base.json` plus one overlay: `collect.json`, `train.json`, or `infer.json`. Validation is strict, so unknown or missing keys are errors. Relative paths resolve against `configs/`.
- **Parallel collection.** Each worker is an isolated `aic_eval` container (its own network namespace) running the unmodified AIC bringup headless with `ground_truth:=true`. It spawns boards through `/gz_server/spawn_entity` exactly like `aic_engine`, waits for a post-spawn `/observations` message, and labels it from `/scoring/tf`.
- **Interfaces.** `aic_perception_interfaces` defines `SceneEstimate` (header in `base_link`, schema name, targets) and `TargetEstimate` (id, type, present, presence probability, confidence, pose).

### Dataset layout

```text
data/aic_v1/
  provenance.json                     scene-defining settings; resuming with different ones is refused
  manifest.json                       usable scene IDs per split, normalization, statistics (curation)
  previews/                           val scenes with projected labels (curation)
  train/images/scene_0000123_left.jpg one flat folder: scene_<id>_<camera>.jpg
  train/labels/scene_0000123.json     one label per scene, written last
  val/images/  val/labels/            same layout
```

The split of each scene is fixed at collection time from `(collection.seed, scene_id)` with `dataset.validation_fraction`. A scene counts as complete once its label exists.

## Instructions

All commands run from the repository root (`~/Software/ws_aic/src/aic`).

### 0. One-time setup

```bash
pixi install --locked
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
pixi run --locked wandb login
```

Set `docker.rendering` in `configs/collect.json` for your machine:

| Value | Use on | Notes |
|---|---|---|
| `wsl_gpu` | WSL2 | GPU rendering via `/dev/dxg` and Mesa d3d12. |
| `nvidia_gpu` | native Linux | Requires the NVIDIA Container Toolkit. |
| `cpu` | anywhere | Each instance needs about 18 cores; set `cpus_per_worker` accordingly. |

### 1. Collect

```bash
pixi run --locked python aic_utils/aic_perception/scripts/collect_data.py
```

- `collection.scenes` sets the dataset size, and `collection.workers` takes `auto` or a number. `auto` fits workers to CPU cores and memory, capped by `docker.max_gpu_workers`.
- Progress, rate, and ETA print every `progress_interval_s`.
- Ctrl+C stops all workers. Run the same command again to resume; finished scenes are skipped.
- More than about 4 to 6 concurrent GPU instances occasionally segfault Gazebo at startup. Workers launch `launch_interval_s` apart, and a crashed worker is relaunched up to `max_restarts` times. Inspect a failed worker with `docker logs aic_perception_worker_<i>`.
- To grow the dataset, raise `collection.scenes` and run again. To change seed, split, schema, image size, or randomization, use a new `dataset.directory`.

### 2. Curate

```bash
pixi run --locked python aic_utils/aic_perception/scripts/curate_dataset.py
```

- Curation verifies every image and label and drops scenes where the board is not visible. It writes `manifest.json` and draws `previews/`.
- **Open a few previews before training:** each marker should sit on its port.
- Rerun curation after any further collection.

### 3. Train

```bash
pixi run --locked python aic_utils/aic_perception/scripts/train.py
```

- `configs/train.json` sets the run name, model size, epochs, batch size, learning rates, loss weights, augmentation, and W&B project/mode. Use `offline` or `disabled` mode to skip W&B.
- After every epoch, the run saves `runs/<name>/checkpoints/last.pt` and keeps `best.pt` by mean port translation error on `val`.

### 4. Evaluate

```bash
pixi run --locked python aic_utils/aic_perception/scripts/evaluate.py
```

Writes per-target and per-type translation and rotation errors and presence accuracy to `runs/<name>/evaluation.json`.

### 5. Run live

Start the AIC simulation as in [Getting Started](../../docs/getting_started.md). Then run:

```bash
pixi run --locked python aic_utils/aic_perception/scripts/run_inference.py
pixi run --locked ros2 topic echo /aic_perception/scene_estimate
```

`configs/infer.json` selects the checkpoint, topics, rate, and presence threshold.

### 6. Far approach on AIC trials

`aic_perception.policies.PerceptionFarApproach` is an `aic_model` policy. It reads the target SFP port from `SceneEstimate`, using the median of `far_approach.estimate_count` fresh estimates. It then moves the plug tip to `hover_height_m` above the perceived port entrance and stops, without inserting. Placing the plug uses its ground-truth pose in the gripper, so the simulation must run with `ground_truth:=true`. The policy logs the perception error and the final hover error against ground truth. SC tasks are rejected: the engine's SC port names cannot be mapped onto rail-order names from the task alone.

Run each command in its own terminal from the repository root, in this order, and start the third within the engine's model discovery timeout:

```bash
# 1. AIC simulation and engine (sample_config: two SFP trials, one SC trial)
docker run --rm --name aic_eval --network host \
  -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
  --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e GALLIUM_DRIVER=d3d12 -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  ground_truth:=true start_aic_engine:=true gazebo_gui:=true launch_rviz:=true model_discovery_timeout_seconds:=300

# 2. Scene estimator
pixi run --locked python aic_utils/aic_perception/scripts/run_inference.py

# 3. Policy
PYTHONPATH=$PWD/aic_utils/aic_perception pixi run --locked ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_perception.policies.PerceptionFarApproach
```

The `DISPLAY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, and `/mnt/wslg` options give the container the WSLg display for RViz and the Gazebo GUI. Drop them and set `gazebo_gui:=false launch_rviz:=false` to run headless. In RViz, a TF display shows each perceived port as `aic_perception/<target id>` next to its ground truth frame.

To run the evaluation trials instead, add `-v $PWD/aic_engine/config/eval_config.yaml:/eval_config.yaml:ro` to the `docker run` options and `aic_engine_config_file:=/eval_config.yaml` to its arguments; the image itself only ships `sample_config.yaml`.

### Tests

```bash
pixi run --locked python -m pytest aic_utils/aic_perception/test
```
