# AIC Perception: Multiview Scene Estimation and Close-Range Refinement

Two models feed the insertion policy. The **scene estimator** estimates the AIC task board and every port (`nic_card_mount_K/sfp_port_J`, and `sc_port_rank_0/1` for the SC ports by order along their rail) as poses in the robot `base_link` frame from the three wrist cameras. A task planner can then select a port by name and receive its pose, presence, and confidence. The **close-range refiner** then measures, from the wrist images at pre-contact height, where the port is relative to the plug actually held, which the policy servos on before contact.

## Algorithm

```text
left / center / right RGB ──► shared backbone (DINOv2 ViT-S/14 or ResNet-18) ──► feature tokens per camera
camera intrinsics + base_from_camera (TF) ──► per-token ray embedding (origin, direction)
                     all tokens ──► transformer encoder (cross-view fusion)
     one learned query per schema target ──► transformer decoder
       per target: presence, translation (base frame), 6D rotation, confidence
```

- **Backbone.** `model.backbone` selects DINOv2 ViT-S/14 with registers (14-pixel patches, input resized to `backbone_input`) or ResNet-18 (32-pixel cells). Checkpoints carry their configuration, so older ResNet checkpoints keep loading.
- **Schema-driven output.** `configs/base.json` defines a versioned board schema (`aic_v2`, 13 targets): an ordered list of target IDs and their ground-truth frames. Query *i* always means the *i*-th target ID. Planners see IDs, never indices. A changed board means a new schema version and a fine-tune.
- **Calibration is an input.** Every image token carries its base-frame viewing ray, computed from the camera matrix and the camera pose from TF. Calibration changes appear as input changes rather than being baked into the weights.
- **Supervision.** Presence is trained on every target; it means spawned and inside at least one camera's view. Pose (L1 translation, chordal rotation) and confidence are trained on present targets. The confidence target is whether the current pose error is within the configured tolerance.
- **Data.** Privileged synthetic labels from Gazebo. Each board is one AIC-style trial within the engine's limits: an SFP task (25% are SC tasks instead) with the matching cable attached as `aic_engine` attaches it (`sfp_sc_cable` with the SFP plug in hand, or `sfp_sc_cable_reversed` with the SC plug in hand) and a target port that is spawned. Pose, NIC cards, SC ports (at most two, on the one rail the engine uses), fixtures, rail offsets, and yaw offsets are randomized. Labels record the task, the held cable, and the true plug pose. Each board is captured from several views along an approach: HOME, then the calibrated plug tip above one port at heights stratified log-uniformly from 25 cm down to 1 cm, with a lateral offset growing with height, so the model sees the port from every distance the policy estimates from. Ground truth comes from the board's pose publisher, the same source AIC scoring uses.
- **SC identity.** The engine names SC ports by config slot (`sc_port_0`, `sc_port_1`), but both slots share one rail and look identical, so the slot cannot be seen in images. The model reports SC ports by rank along the rail; with two SC ports present, the policy must choose which rank the task's slot is.

## Close-range refiner

```text
three fixed crops around the held plug (one per camera) ──► shared DINOv2 ViT-S/14
   + per-camera position embedding + plug type embedding ──► transformer (2 layers) ──► readout
   ──► port minus plug (3), plug position (3), plug-to-port rotation (3), in the centre camera frame
```

- **Why relative.** Absolute regression plateaus at about 2 mm even up close. The plug is rigid in the gripper, so it projects to the same pixels in every image; crops centred there always contain the plug tip and the nearby port, and the offset between them is what the final approach needs. Moving the TCP by the offset aligns the plug whatever the grasp calibration says, and repeating the measurement cancels the arm settling short under the cable load.
- **Data.** Close views of aic_v3 (plug within 6 cm above and 3 cm beside its target port) plus aic_v4_close, a targeted collection without HOME views whose plug hovers 6 to 40 mm above the port with offsets concentrated near the axis (46% within 2 mm). Labels hold the true plug and port poses, so no annotation is needed.

## Software


| Stage | Script | Runs in | Produces |
|---|---|---|---|
| Collect | `scripts/collect_data.py` | host; one `aic_eval` container per worker | `train/` and `val/` images and labels |
| Curate | `scripts/curate_dataset.py` | pixi | `manifest.json`, `previews/` |
| Train | `scripts/train.py` | pixi, GPU | `runs/<name>/checkpoints/{best,last}.pt`, W&B run |
| Evaluate | `scripts/evaluate.py` | pixi, GPU | errors per type and distance, presence, confidence calibration; `runs/<name>/evaluation.json` |
| Train refiner | `scripts/train_refiner.py` | pixi, GPU | `runs/<refiner>/checkpoints/{best,last}.pt`, W&B run |
| Evaluate refiner | `scripts/evaluate_refiner.py` | pixi, GPU | offset error per plug type, starting misalignment, and height |
| Infer | `scripts/run_inference.py` | pixi, next to a running AIC sim | `/aic_perception/scene_estimate`, TF `aic_perception/<id>`, `runs/estimates/*.jsonl` |
| Summarize trials | `scripts/summarize_trials.py` | pixi | per-trial table and totals from `runs/trials/*.jsonl` |

- **Configuration only.** Every script reads `configs/base.json` plus one overlay: `collect.json`, `train.json`, `refine.json`, or `infer.json`. `collect.json` may override `dataset` to collect into its own directory; `refine.json` lists the datasets the refiner trains on. Validation is strict, so unknown or missing keys are errors. Relative paths resolve against `configs/`.
- **Parallel collection.** Each worker is an isolated `aic_eval` container (its own network namespace) running the unmodified AIC bringup headless with `ground_truth:=true`. It spawns boards through `/gz_server/spawn_entity` exactly like `aic_engine`, waits for a post-spawn `/observations` message, and labels it from `/scoring/tf`.
- **Interfaces.** `aic_perception_interfaces` defines `SceneEstimate` (header in `base_link`, schema name, targets) and `TargetEstimate` (id, type, present, presence probability, confidence, pose).

### Dataset layout

```text
data/aic_v3/
  provenance.json                     scene-defining settings; resuming with different ones is refused
  manifest.json                       usable scene IDs per split, normalization, statistics (curation)
  previews/                           val scenes with projected labels (curation)
  train/images/scene_0000123_left.jpg one flat folder: scene_<id>_<camera>.jpg
  train/labels/scene_0000123.json     one label per scene, written last
  val/images/  val/labels/            same layout
```

Scene `board_id * views_per_board + view` is one view of one board. The split is fixed per board from `(collection.seed, board_id)` with `dataset.validation_fraction`, so validation never shares a board with training. A scene counts as complete once its label exists.

## Instructions

All commands run from the repository root (`~/Software/ws_aic/src/aic`).

### 0. One-time setup

```bash
pixi install --locked
docker buildx build -f docker/aic_eval/Dockerfile -t aic_eval:phase1 --load .   # simulator built from this repo (phase 1 board)
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

- `collection.boards` and `collection.views_per_board` set the dataset size (boards × views scenes), and `collection.workers` takes `auto` or a number. `auto` fits workers to CPU cores and memory, capped by `docker.max_gpu_workers`.
- Progress, rate, and ETA print every `progress_interval_s`.
- Ctrl+C stops all workers. Run the same command again to resume; finished scenes are skipped.
- More than about 4 to 6 concurrent GPU instances occasionally segfault Gazebo at startup. Workers launch `launch_interval_s` apart, and a crashed worker is relaunched up to `max_restarts` times. Inspect a failed worker with `docker logs aic_perception_worker_<i>`.
- To grow the dataset, raise `collection.boards` and run again. To change seed, split, schema, image size, or randomization, use a new `dataset.directory`.
- Boards and cables are built from the board description inside the `aic_eval` image (`collection.task_board_xacro`, `cable.xacro`), not the repo copy, so training data always matches the simulator the trials run in. The repo's `aic_description` can differ from a published image: the April 2026 image keeps `sc_port_1` on SC rail 1, the repo (June, phase 1) moved it to rail 0.
- `collect.json` currently describes aic_v4_close (no HOME view, `viewpoints.lateral_power` 1.0 to concentrate views near the port axis, its own `dataset` and seed). aic_v3's settings are recorded in `data/aic_v3/provenance.json`.

### 2. Curate

```bash
pixi run --locked python aic_utils/aic_perception/scripts/curate_dataset.py
```

- Curation covers the scene estimator's dataset and every dataset in `refine.json`. It verifies every image and label and drops scenes where the board is not visible, then writes `manifest.json` and draws `previews/` in each.
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

Prints errors per target type and per viewpoint distance (HOME, far, mid, close), presence precision and recall, and how well the confidence is calibrated; the full report, including every target, goes to `runs/<name>/evaluation.json`. Training logs the same per-distance errors to W&B every epoch.

### 4b. Train and evaluate the refiner

```bash
pixi run --locked python aic_utils/aic_perception/scripts/train_refiner.py
pixi run --locked python aic_utils/aic_perception/scripts/evaluate_refiner.py
```

`configs/refine.json` lists the datasets and sets the selection, crop size, model, and optimization. The evaluation reports the offset error next to the misalignment it corrects, per plug type, starting misalignment, and plug height; the `0-2mm` rows are the ones the last servo step relies on.

### 5. Run live

Start the AIC simulation as in [Getting Started](../../docs/getting_started.md). Then run:

```bash
pixi run --locked python aic_utils/aic_perception/scripts/run_inference.py
pixi run --locked ros2 topic echo /aic_perception/scene_estimate
```

`configs/infer.json` selects the checkpoint, topics, rate, and presence threshold. With `logging.log_estimates`, every estimate is appended to `runs/estimates/estimates_<time>.jsonl` (stamp, latency, and each target's presence, confidence, and position).

### 6. Policies on AIC trials

Two `aic_model` policies live in `aic_perception/policies/`:

| Policy | What it does |
|---|---|
| `PerceptionInsert` | Straight from the start pose to `coarse_height_m` above the perceived port, re-estimate there, down to `pre_contact_height_m`, closed-loop servo on the refiner (measuring only once the arm and plug are still), force baseline, guarded descent, spiral search under light force, force-limited push-in. Without a refiner checkpoint it re-estimates with the scene estimator instead. |
| `CalibratePlugGrasp` | Offline tool calibration (not a competition policy): measures the plug pose in the gripper from ground truth and writes `runs/calibration/plug_grasp.json`; copy it into `configs/infer.json` under `plug_grasp`. |

**Competition honesty.** `PerceptionInsert` controls the robot only from what a submission receives: the task message, robot TF, the wrist wrench in the observation, `SceneEstimate`, CAD port depths, and the offline grasp calibration. They work with `ground_truth:=false`. With `ground_truth:=true` and `insertion.log_ground_truth`, they additionally log errors against ground truth for evaluation; nothing logged feeds back into control. The force baseline is averaged at the pre-contact pose because the sensor cannot be tared during evaluation, and `max_force_n` is validated to stay below the 20 N scoring limit.

**SC targets.** The task names the engine's SC slot, which cannot be seen. With one SC port present it is the target; with two, `sc_rank_for_slot` maps the slot onto a rank along the rail. The shipped rule (`sc_port_1` is rank 0) follows the only two-port example, `eval_config.yaml` trial 3, and is an assumption.

All thresholds, speeds, impedances, and spiral parameters are in the `insertion` section of `configs/infer.json`; the servo (iterations, samples per iteration, tolerance, step and yaw limits, spiral radius after convergence) is in `refiner`.

**Trial log.** Every task appends one JSON line to `runs/trials/trials_<time>.jsonl`: the task, each phase's measurements (target resolution, servo offsets, descent, spiral, depth), ground-truth errors when published, the outcome, and why it failed. Summarize the newest log with:

```bash
pixi run --locked python aic_utils/aic_perception/scripts/summarize_trials.py
```

Run each command in its own terminal from the repository root, in this order, and start the third within the engine's model discovery timeout:

```bash
# 1. AIC simulation and engine (sample_config: two SFP trials, one SC trial)
docker run --rm --name aic_eval --network host \
  -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
  --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e GALLIUM_DRIVER=d3d12 -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  aic_eval:phase1 \
  ground_truth:=true start_aic_engine:=true gazebo_gui:=true launch_rviz:=true model_discovery_timeout_seconds:=300

# 2. Scene estimator (the policy loads the refiner itself)
pixi run --locked python aic_utils/aic_perception/scripts/run_inference.py

# 3. Policy (PerceptionInsert, or CalibratePlugGrasp for the offline grasp calibration)
PYTHONPATH=$PWD/aic_utils/aic_perception pixi run --locked ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_perception.policies.PerceptionInsert
```

The `DISPLAY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, and `/mnt/wslg` options give the container the WSLg display for RViz and the Gazebo GUI. Drop them and set `gazebo_gui:=false launch_rviz:=false` to run headless. In RViz, a TF display shows each perceived port as `aic_perception/<target id>` next to its ground truth frame.

To run the evaluation trials instead, add `-v $PWD/aic_engine/config/eval_config.yaml:/eval_config.yaml:ro` to the `docker run` options and `aic_engine_config_file:=/eval_config.yaml` to its arguments; the image itself only ships `sample_config.yaml`.

### Tests

```bash
pixi run --locked python -m pytest aic_utils/aic_perception/test
```
